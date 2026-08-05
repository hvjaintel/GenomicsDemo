# Does Intel AMX speed up DeepVariant? Measured, not assumed.

This project set out to build a booth demo whose headline was an **AMX ON / AMX
OFF race** on DeepVariant. That headline does not survive contact with the
hardware. This document records what was actually measured, how, and what the
demo does instead.

Everything below was produced on the demo machine itself. No number here is
estimated, scaled, or borrowed from a vendor slide.

**Hardware:** Intel Xeon 6740P, dual socket, 192 logical CPUs, 1 TB RAM.
**Software:** `google/deepvariant:1.10.0`, oneDNN as shipped in that image.
**Data:** HG002 chr20, GRCh38, 192 shards, `run_deepvariant` BAM-in path.

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
