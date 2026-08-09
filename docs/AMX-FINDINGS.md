# Does Intel AMX speed up DeepVariant? Measured, not assumed.

This project set out to build a booth demo whose headline was an **AMX ON / AMX
OFF race** on DeepVariant. That headline does not survive contact with the
hardware. This document records what was actually measured, how, and what the
demo does instead.

Everything below was produced on the demo machine itself. No number here is
estimated, scaled, or borrowed from a vendor slide.

**Hardware:** Intel Xeon 6740P, dual socket, 192 logical CPUs, 1 TB RAM.
**Software:** `google/deepvariant:1.10.0` **and** `google/deepvariant:1.5.0`.
**Data:** HG002 chr20, GRCh38, 192 shards, `run_deepvariant` BAM-in path.

> ### Read this first: the version is the whole story
>
> **AMX works on DeepVariant 1.5 and does not work on 1.10.** This is not a
> tuning difference, it is an architectural one. 1.5 runs a TF1 static graph;
> 1.6 migrated to TF2/Keras SavedModel. The bf16 graph rewrite that engages
> AMX is safe on the former and breaks correctness on the latter.
>
> And the twist that matters commercially: **stock 1.10 without AMX is still
> 2.5x faster than 1.5 with AMX.** Findings 1-3 below are the 1.10 story;
> Finding 4 is the 1.5 story; Finding 5 is the comparison that decides which
> one you should actually deploy.

---

## Finding 1 — Raising the oneDNN ISA ceiling to AMX changes nothing

The obvious implementation of an AMX toggle is `ONEDNN_MAX_CPU_ISA`:

| Leg | `ONEDNN_MAX_CPU_ISA` |
| --- | --- |
| AMX ON | `AVX512_CORE_AMX` |
| AMX OFF | `AVX512_CORE` |

Full chr20, everything else identical:

| Leg | Wall clock |
| --- | --- |
| AMX OFF | 128.4 s |
| AMX ON | 126.0 s |
| **Speedup** | **0.98x** |

That is noise. `ONEDNN_VERBOSE=1` explains why. Counting the implementation
string on every primitive execution (`app/parsing.py`, `IsaUsage`):

```
0 of 95 compute primitives used AMX
(brgconv:avx512_core x54, brgconv_1x1:avx512_core x40, x64:gemm:jit x1)
```

Every convolution ran on AVX-512. Every tensor was `f32`. **AMX has no fp32
path**, and oneDNN will not silently downcast a model. Setting the ceiling to
`AVX512_CORE_AMX` only tells oneDNN that AMX *may* be used; with an fp32 graph
there is nothing for the tiles to do, so they sit idle.

`ONEDNN_DEFAULT_FP32_MATH_MODE=BF16` does not change this either (measured
0.96x). `call_variants --helpfull` exposes no precision flag, and there is no
Intel-optimized DeepVariant image published on Docker Hub.

### The honesty bug this exposed in our own code

An earlier version of `app/runner.py` parsed oneDNN's *startup banner*
(`Intel AMX with bfloat16 and 8-bit integer support`) and reported "AMX
confirmed". That banner describes the **CPU's capability ceiling**, not what
ran. The code was truthfully reporting a fact that meant nothing, in a context
that implied it meant everything.

`IsaUsage` was added to close that gap. It counts AMX kernels among actual
`convolution` / `inner_product` / `matmul` / `deconvolution` executions and
reports `N of M compute primitives used AMX`. Reorders are deliberately
excluded — they are data-layout plumbing, and counting them would dilute the
signal until idle tiles looked busy.

**Ceiling is not use.** Any demo that verifies the banner and stops is, without
meaning to, lying.

## Finding 2 — Forcing bf16 does engage AMX, and makes things worse

TensorFlow can rewrite an fp32 graph to bf16 through a Grappler pass. This is
what Intel Labs' Open-Omics-DeepVariant fork does (via TF1's `RewriterConfig`);
DeepVariant 1.10 is TF2, where the equivalent is:

```python
tf.config.optimizer.set_experimental_options(
    {"auto_mixed_precision_onednn_bfloat16": True}
)
```

DeepVariant's binaries are Bazel zipapps launched by shell wrappers, so there
is no source file to patch and no flag to set. `app/container_inject/sitecustomize.py`
is mounted read-only into the container with `PYTHONPATH` pointing at it —
Python imports `sitecustomize` automatically, and the hook fires the moment the
application imports TensorFlow. The image is not modified.

It works. Dispatch flips:

```
94 of 95 compute primitives used AMX
(brgconv:avx512_core_amx x54, brgconv_1x1:avx512_core_amx x40)
```

TensorFlow confirms the rewrite: `Converted 313/1102 nodes to bfloat16
precision using 1 cast(s)`.

And it is **slower**. `call_variants` inference rate on identical 192-shard
chr20 examples, same container harness, machine otherwise idle:

| Configuration | sec / 100 examples |
| --- | --- |
| AMX off, fp32 (AVX-512) | 0.041 |
| AMX permitted, fp32 | 0.042 |
| **AMX engaged, bf16** | **0.064 – 0.066** (two runs) |

Roughly **1.6x slower**. Only 313 of 1102 nodes convert, so the graph gains
cast operations and layout reorders on every boundary between the bf16 island
and the surrounding fp32 nodes; on a model this small that overhead exceeds
what the tiles win back.

### It is also incorrect

The bf16 run does not merely underperform — it produces output DeepVariant
itself rejects:

```
File ".../deepvariant/call_variants.py", line 263, in round_gls
ValueError: Invalid genotype likelihoods do not sum to one:
sum([0.00793457 0.98828125 0.00344849]) = 0.999664306640625
```

bf16 has ~8 bits of mantissa against fp32's 24. The softmax output no longer
sums to 1.0 within DeepVariant's tolerance. The exception is raised inside a
`multiprocessing` post-processing worker, which dies; the parent then waits on
a queue that will never be fed. The symptom is a process pinned at **0% CPU
with 389 threads parked in `futex_wait_queue_me`, forever** — a hang that looks
like a scheduler or thread-oversubscription bug and is in fact a precision bug
three layers down.

This cost real debugging time and is worth stating plainly: **an accuracy
failure can present as a hang.**

`amx.use_bf16_injection` is therefore `false` by default. The mechanism stays
in the repo, documented and one config flag away, so the finding is
reproducible rather than merely asserted.

## Finding 3 — Instrumentation is itself a confound

`ONEDNN_VERBOSE=1` prints one line per primitive execution: **270 MB** of log
on a chr20 run. bf16 emits *more* primitives than fp32 (the casts and reorders
above), so the instrumentation penalises the AMX leg specifically — the exact
leg under test. Timing with verbose on would have understated AMX and quietly
corrupted the headline.

`ONEDNN_VERBOSE=dispatch`, the low-overhead alternative, is **not supported in
this oneDNN build** — it silently disables verbose output entirely rather than
erroring, which is its own small trap.

`app/race.py` therefore runs two phases:

1. **Verify** — a small region with verbose on, to prove which kernels ran.
   Not timed.
2. **Measure** — the real region with verbose off. Timed.

## Finding 4 — On DeepVariant 1.5, AMX works properly, via a supported flag

Intel's Open-Omics-DeepVariant fork is based on **v1.5**, not 1.10. That single
fact explains everything above.

v1.5 uses TensorFlow 2.11 in **TF1 compatibility mode**: `call_variants.py`
calls `tf.compat.v1.disable_eager_execution()`, loads a TF1 checkpoint
(`model.ckpt.meta` / `.index` / `.data`), and runs inference through an
estimator with a `tf.compat.v1.ConfigProto`. DeepVariant 1.6 replaced all of
that with TF2/Keras and a `saved_model.pb`.

Intel's fork changes exactly one thing in `call_variants.py`: it builds that
ConfigProto with a bf16 rewrite enabled.

```python
# IntelLabs/open-omics-deepvariant, r1.5, deepvariant/call_variants.py
info = cpuinfo.get_cpu_info()
if 'amx_bf16' in info['flags']:
    graph_options = tf.compat.v1.GraphOptions(
        rewrite_options=rewriter_config_pb2.RewriterConfig(
            auto_mixed_precision_onednn_bfloat16=rewriter_config_pb2.RewriterConfig.ON))
    config = tf.compat.v1.ConfigProto(graph_options=graph_options)
```

**You do not need their fork, and you do not need to build anything.** Stock
`google/deepvariant:1.5.0` already exposes a `--config_string` flag that is
`text_format.Parse`d straight into that same ConfigProto:

```bash
--call_variants_extra_args="config_string='graph_options: {rewrite_options: {auto_mixed_precision_onednn_bfloat16: ON}}'"
```

Measured on the stock 1.5.0 image with that flag:

```
108 convolution    brgconv:avx512_core_amx_bf16
 82 convolution brgconv_1x1:avx512_core_amx_bf16
```

**Every convolution on AMX. Zero `round_gls` failures.** The TF1 Grappler pass
operates on the full static graph and keeps the final softmax in fp32, so the
genotype likelihoods still sum to 1.0. That is precisely what the TF2 path
fails to guarantee.

Full chr20, 192 shards, verbose logging off:

| Stage | fp32 | bf16 + AMX | |
| --- | --- | --- | --- |
| make_examples | 47.5 s | 48.9 s | unchanged, as expected |
| **call_variants** | **284.2 s** | **240.1 s** | **1.18x** |
| postprocess_variants | 24.4 s | 23.1 s | unchanged |
| **End to end** | **358 s** | **314 s** | **1.14x** |

Variants called: **215,899 in both legs — identical.** No accuracy trade at
the call-set level.

So AMX is real, it is correct, and on this workload it is worth about **18% on
the inference stage and 14% end to end**. That is a genuine result. It is also
a long way from the order-of-magnitude figures AMX is usually marketed with,
because `call_variants` is only part of the pipeline and the model is small.

### A note on Intel's published numbers

The Open-Omics-DeepVariant README claims "up to **YYx** speedup" — a literal
unfilled placeholder, still present on the `r1.5` branch. There is no published
benchmark figure in the repository, and no prebuilt Intel-optimized DeepVariant
image exists on any public registry. Intel's pipeline benchmarks that do
circulate measure **fq2vcf end to end**, including bwa-mem2 alignment, on
whole-genome data — so they are not comparable to a `call_variants` number, and
a large part of any such figure is core count rather than AMX.

## Finding 5 — the version costs more than AMX gains

The decisive measurement. Same box, same chr20 BAM, same 192 shards:

| Configuration | End to end | call_variants | Variants |
| --- | --- | --- | --- |
| 1.5.0, fp32 | 358 s | 284.2 s | 215,899 |
| 1.5.0, bf16 + AMX | 314 s | 240.1 s | 215,899 |
| **1.10.0, fp32, AMX idle** | **123 s** | **~40 s** | 210,390 |

**Stock 1.10 with the AMX tiles doing nothing is 2.5x faster end to end than
1.5 with AMX fully engaged — and roughly 6x faster on the inference stage
itself.** DeepVariant 1.10 routes easy candidates through a small model that
never touches the CNN, which saves far more work than accelerating the CNN
does.

This is the uncomfortable, useful conclusion: **on this workload, staying
current beats turning on AMX, by a wide margin.** Choosing 1.5 to obtain a
1.14x AMX win costs you a 2.9x version win. If a comparison is framed as
"AMX on versus AMX off" on a pinned old version, it is answering a question
nobody deploying this pipeline should be asking.

(The variant counts differ between 1.5 and 1.10 — 215,899 vs 210,390 — because
they are different models and different callers. That is a version difference,
not an AMX effect: within each version the two legs agree exactly.)

## What the demo does instead

The booth headline is a **core-scaling race**: identical binary, identical
data, identical shard count, identical ISA settings, and the only difference is
how many logical CPUs Docker lets the container use (`--cpuset-cpus`).

No precision trade, no asterisk, and the output is provably identical — both
legs call exactly **210,390 variants** on chr20.

`RunSpec.fingerprint()` excludes exactly one field per race mode (the AMX
state, or the cpuset). Everything else must match or `time_ratio()` returns
`None` and the UI refuses to display a speedup at all.

The AMX comparison is kept as an **evidence panel** rather than deleted. Booth
visitors ask about AMX, and "here is the measurement, and here is why the tiles
are idle on this workload" is a better answer than either a shrug or a
manufactured number.

## Reproducing this

```bash
# Finding 4: AMX genuinely working, on stock DeepVariant 1.5.0, no fork needed
docker run --rm -e ONEDNN_MAX_CPU_ISA=AVX512_CORE_AMX -e ONEDNN_VERBOSE=1 \
  ... google/deepvariant:1.5.0 /opt/deepvariant/bin/run_deepvariant \
  --model_type=WGS ... \
  --call_variants_extra_args="config_string='graph_options: {rewrite_options: {auto_mixed_precision_onednn_bfloat16: ON}}'"
# expect: brgconv:avx512_core_amx_bf16, and no round_gls ValueError

# The honest headline
python -m app.race --mode scaling --sample chr20

# Finding 1: AMX permitted vs disabled, both fp32
python -m app.race --mode amx --sample chr20

# Finding 2: set amx.use_bf16_injection: true in config.yaml first.
# Expect AMX kernels in phase 1, then a hang after call_variants when the
# post-processing worker dies. That hang is the finding.
python -m app.race --mode amx --sample chr20
```

## What would actually make AMX pay off here

Not attempted in this repo, listed so the question has an answer:

- **On 1.10 specifically:** a TF2 path that honours the Keras classification
  head's `dtype=tf.float32` annotation end to end. The 1.5 result proves the
  bf16 rewrite is sound in principle — it is the TF2 SavedModel plumbing that
  loses the fp32 output contract.
- A model **trained or fine-tuned in bf16**, so no cast boundaries are needed
  and the softmax stays inside tolerance.
- **int8 post-training quantization** with calibration, which AMX also
  accelerates and which has far more headroom than bf16.
- Intel's Open-Omics-DeepVariant fork built from source (~30 min Bazel build,
  no prebuilt image). Note their published figures measure the whole fq2vcf
  pipeline — bwa-mem2 alignment included — not `call_variants` in isolation, so
  they are not directly comparable to anything above.
- A larger batch or a larger model, where tile utilisation can amortise the
  reorder cost.

---

# Appendix — assessing third-party pipeline numbers

A figure of **22.57 minutes** for "full workload end-to-end (fq2bam and
DeepVariant v1.5)" was put to this project. It cannot be reconciled with
anything measured here, and the reasons are worth writing down, because the
same traps apply to any vendor number a booth visitor quotes at you.

## Reference points, all verifiable

Google's own DeepVariant metrics, whole genome, all chromosomes:

| Version | Machine | make_examples | call_variants | postprocess | Total |
| --- | --- | --- | --- | --- | --- |
| 1.5 | 64 vCPU GCP | ~103 min | ~185 min | ~48 min | **~336 min** |
| 1.10 | n2-standard-96 | 46m 15s | 15m 58s | 6m 45s | **~69 min** |

Source: `github.com/google/deepvariant/blob/r1.5/docs/metrics.md` and `.../r1.10/docs/metrics.md`.
Note the same version effect measured here on chr20: 1.10's `call_variants` is
~11.6x faster than 1.5's, because of the small-model fast path.

Intel Labs' own published scaling for the Open-Omics `fq2vcf` pipeline
(bwa-mem2 + their DeepVariant 1.5 fork), quoted verbatim from
`community.intel.com/t5/Blogs/Tech-Innovation/Artificial-Intelligence-AI/Intel-Xeon-is-all-you-need-for-AI-inference-Performance/post/1506083`:

> "Open Omics consumes just 109 mins on a single socket of the Intel CPU.
> Moreover, it consumes just 8.5 mins on 8 dual-socket Intel CPUs (16 sockets),
> which is nearly 1.9 times faster than a DGX A100 GPU system with 8 A100 GPUs.
> On further scaling to 32 dual-socket Intel CPUs (64 sockets), Open Omics
> consumes just 3 mins."

Hardware footnote, verbatim: *"1-node (1,2 socket), 2/4/8/16/32-nodes
(4/8/16/32/64 sockets), Each socket is 1x Intel Xeon Platinum 8480+, 56 cores"*.
Dataset: **HG001, 30x** — not HG002 at 35x.

NVIDIA Clara Parabricks, from `developer.nvidia.com/blog/long-read-sequencing-workflows-and-higher-throughputs-in-nvidia-parabricks-4-1/`:

> "a 30x whole genome in 16 mins on a DGX A100 GPU [8xA100 GPUs] compared to
> 21 mins in v4.0 and ~24 hours on CPU-only."

## Why 22.57 min is not a single-server number

**1. Intel's own single-socket figure is 109 minutes.** Their pipeline reaches
single-digit minutes only by scaling out over MPI to 8-32 dual-socket *nodes*.
22.57 min sits between their 1-socket and 16-socket points — implying roughly
2-3 dual-socket nodes. This booth has one. A cluster number is not a server
number.

**2. `fq2bam` is NVIDIA's tool, not Intel's.** Parabricks' `fq2bam` is
GPU-only ("A 30x whole genome can be run through FQ2BAM in as little as 6
minutes on an NVIDIA DGX system"). Intel's container is `fq2bams` — plural —
and is bwa-mem2 + samtools sort, no CUDA anywhere in its Dockerfile. If the
figure genuinely refers to `fq2bam`, part of it ran on GPUs. Note also that
Parabricks v4.0 was **21 min** on a DGX A100, which is uncomfortably close to
22.57.

**3. The scope differs.** Parabricks `fq2bam` includes duplicate marking and
BQSR. Intel's `fq2bams` includes neither. Two pipelines with different work in
them cannot be compared on wall clock alone.

**4. Arithmetic.** Scaling this project's measured chr20 v1.5 numbers to a
whole genome puts DeepVariant alone — before any alignment — at roughly
**3.8-4.5 hours** on this 192-thread server, consistent with Google's own
336-minute figure on 64 vCPUs. Reaching 22.57 min including alignment on one
node would require `call_variants` to scale linearly across 192 threads (it
does not; it is memory-bandwidth bound), an int8 model (none is published),
and sub-3-minute bwa-mem2 alignment (realistically 15-25 min).

## The point

None of this makes the Intel number false. Their 8.5 min on 16 sockets is
almost certainly real, and it is a genuinely impressive result *for a cluster*.
The failure mode is comparing it to one server, or to a BAM-in run, or to a
pipeline that also marks duplicates.

Before repeating any third-party figure at the booth, get four things:
**node count, tool identity, dataset coverage, and stage scope.** Without all
four the number is not comparable to anything this demo shows.

---

# Postscript — the whole-genome number, and a correction

The full HG002 35x genome has now actually been run on this box, rather than
extrapolated. DeepVariant 1.10.0, all 192 threads, 192 shards:

| Stage | Time |
| --- | --- |
| make_examples | 13m 50s |
| call_variants | 11m 05s |
| postprocess_variants | 49s |
| **Total** | **25m 47s** |

7,709,239 variants.

**This corrects two things written earlier in this session.**

First, the extrapolation. Scaling the measured chr20 time by genome size
predicted ~46 minutes. The real figure is 25m 47s. Whole-genome runs have far
more independent work available than a single chromosome, so make_examples
parallelises better across 192 threads than chr20 allows. **chr20 understates
whole-genome scaling**, and the error is not small. Extrapolating from one
contig to a genome is not sound; it is recorded here as a caution, not a
method.

Second, and more importantly: the appendix above argued that a 22.57-minute
single-node figure was "15-20x too fast" and implied a cluster. **That
reasoning was wrong**, and it was wrong because it was anchored on v1.5, which
genuinely does take hours here. On v1.10 this single dual-socket server does
the whole genome in 25m 47s. A 22.57-minute end-to-end result on one node is
therefore entirely plausible, and the burden-of-proof framing in the appendix
overstated the case.

What survives from that appendix is narrower and still worth asking about:
`fq2bam` is NVIDIA Parabricks' GPU tool and not an Intel one; Intel's own
published *single-socket* figure for their v1.5-based pipeline is 109 minutes;
and stage scope still differs between pipelines (duplicate marking and BQSR).
Those are questions to ask about a number, not grounds to reject it.

The general lesson is the one this whole document keeps arriving at: an
estimate derived from a smaller run is a hypothesis, and it stays a hypothesis
until the real workload is measured. That applies to estimates that flatter
this machine as much as to ones that flatter someone else's.
