# Audience handout — prompt to recreate this demo

Hand this to a visitor who wants to rebuild the demo on their own hardware. It is
**self-contained**: it references only public data and a public container image, and does
not depend on access to this (private) repository.

They paste the block below into an AI coding agent (Copilot CLI, Claude Code, Cursor, or
similar) on the target machine.

> **Why a prompt and not a repo link.** This repository is private, so `git clone` fails
> for anyone outside the org with a confusing auth error. The prompt is also more useful:
> it forces the values that must not travel between machines — cpusets, measured wall
> clocks — to be re-derived on the target rather than copied and quietly believed.

---

## The prompt

````text
Build me a benchmark that demonstrates how a multi-core server scales on a real
genomics workload: germline variant calling with DeepVariant on human sequencing
data. I want a defensible measurement, not a marketing number.

## The workload

Use the published DeepVariant container and run it on a pre-aligned BAM. Do NOT
build an aligner into this; alignment is a separate workload and mixing them makes
the timing impossible to interpret.

  Container: google/deepvariant:1.10.0   (docker pull, no build needed)
  Entrypoint: /opt/deepvariant/bin/run_deepvariant
  Stages it runs internally: make_examples -> call_variants -> postprocess_variants

## The data (all public, no login, checksums are authoritative)

Reference genome - GRCh38 no-ALT analysis set, 886344618 bytes:
  https://ftp-trace.ncbi.nlm.nih.gov/ReferenceSamples/giab/release/references/GRCh38/GCA_000001405.15_GRCh38_no_alt_analysis_set.fasta.gz
  sha256 3c8def6d325c5d1e934b2dd530c4d1709f027677a15859467071fddf3fff2026
  Also fetch the .fai and .gzi sidecars from the same directory.

Reads - HG002 (GIAB Ashkenazim son), NovaSeq PCR-free 35x, chr20 only,
1093938924 bytes:
  https://storage.googleapis.com/deepvariant/case-study-testdata/HG002.novaseq.pcr-free.35x.dedup.grch38_no_alt.chr20.bam
  sha256 34ac157739e1feeb590f6eb7e11046ccc2aa3277fd55a3ce0942e774d931ed81
  Also fetch the .bam.bai index alongside it.

Start with chr20 (~1 GB, minutes per run). The whole genome is the same URL
without ".chr20" (~46 GB, far longer) - only move to it once chr20 works.

Verify the sha256 of every download before using it and fail loudly on mismatch.
Do not proceed with a file that does not match.

## The measurement: core-to-core scaling

Run the SAME workload twice, changing ONLY the number of physical cores, and
compare wall-clock time. Use --cpuset-cpus on docker run to pin each leg.

CRITICAL - do not let SMT/hyperthreading contaminate this. Derive the cpusets
from the actual topology of THIS machine:

    lscpu -p=CPU,CORE

Group CPUs by their CORE id and take exactly ONE CPU per physical core. A cpuset
like "0-15" is only 16 physical cores on machines that enumerate siblings in a
second block; on machines that enumerate siblings adjacently it is 8 cores plus 8
hyperthreads. Both legs must be SMT-free, or the comparison silently becomes
"cores plus SMT vs cores" and the ratio stops meaning what you will say it means.
Nothing errors when you get this wrong - that is exactly why it needs a check.

Write an automated test that fails if either leg's cpuset contains two threads of
the same physical core, and another that fails if the fast leg does not cover
every physical core on the box. Run them on this machine, and confirm they
actually ran rather than skipped.

Set --num_shards to the number of physical cores in that leg.

## Correctness matters as much as speed

Both legs must produce the same variant calls. Count the variants in each output
VCF and compare. A speedup that changes the answer is not a speedup. Report the
counts next to the times.

Optionally, score against the GIAB truth set for real accuracy numbers:
  https://ftp-trace.ncbi.nlm.nih.gov/ReferenceSamples/giab/release/AshkenazimTrio/HG002_NA24385_son/NISTv4.2.1/GRCh38/HG002_GRCh38_1_22_v4.2.1_benchmark.vcf.gz

## Honesty rules - please follow these strictly

1. Do NOT add an "AMX on/off" toggle or claim any AMX/VNNI/AVX-512 speedup unless
   you have MEASURED it on this exact build. DeepVariant 1.10's call_variants runs
   an fp32 TensorFlow model. AMX is a bf16/int8 engine, so it may do literally zero
   work here. Verify before claiming: set ONEDNN_VERBOSE=1 and count how many
   primitives actually dispatch to an AMX implementation. If it is zero, say so.
   An A/B difference of a few percent with no dispatched primitives is run-to-run
   noise, not acceleration.

2. Never print a wall-clock number without the hardware it was measured on
   (CPU model, physical core count, RAM). A time without its machine is not a
   claim anyone can check.

3. Label anything you did not measure on THIS box as illustrative, clearly, in
   the output. Do not carry numbers over from documentation.

4. Run each configuration at least twice and report both. If they disagree
   meaningfully, investigate before reporting either.

## Practical setup notes that will otherwise bite

- Put the data and all run output on a large, fast data disk, not the OS disk.
  Outputs and scratch are much larger than the inputs.
- DeepVariant shards through GNU parallel, which buffers per-shard stdout under
  the container's /tmp. Bind-mount that to the data disk too, or a long run can
  fill the OS disk and fail with a confusing "Cannot append to buffer file"
  error that looks nothing like a disk-space problem.
- Run the container as the invoking user (-u $(id -u):$(id -g)) so outputs are
  not root-owned.
- Do a short smoke run first: slice a small region (e.g. chr20:10000000-10100000)
  out of the BAM and call on that. It exercises the whole path in about a minute
  and catches mismatched-reference errors before a long run does.
- The reference build must match the BAM. Both files above are GRCh38 no-ALT.
  Pairing a GRCh38 reference with an hg19-aligned BAM produces "0 bases found in
  common among our input files", which reads like a corrupt file but is not.

## Deliverable

A script or small app I can re-run that: verifies the data, derives SMT-free
cpusets for this machine, runs both legs, and prints a table of wall-clock time
and variant count per leg, with the hardware stated alongside. Show me the exact
docker command for each leg so the run is auditable.
````

---

## If they ask what results to expect

Give them **our** numbers with our hardware attached, and be explicit that theirs will
differ:

> On 2× Intel Xeon 6740P (96 cores / 192 threads, 1 TB RAM), chr20 core-to-core:
> 96 cores **127.4 s**, 16 cores **339.4 s** — a **2.66×** ratio, identical variant calls
> on both legs. Your machine will produce different numbers; the method is what transfers,
> not the times.

Do not promise them a 2.66×. That ratio is specific to this core count, this dataset, and
this DeepVariant version.
