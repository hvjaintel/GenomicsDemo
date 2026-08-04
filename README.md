# Genomics on Built-In Xeon Acceleration

A booth demo that runs **real germline variant calling (DeepVariant)** on an Intel Xeon
server and proves three things to a live audience:

1. Deep-learning variant calling runs fast on standard Xeon CPUs using the processor's
   **built-in accelerators (Intel AMX + AVX-512)** — no add-in cards.
2. **AMX (BFloat16) measurably accelerates the workload** versus an AMX-off AVX-512 baseline.
3. It does so on a **quiet, air-cooled, bench-deployable** server co-engineered by Intel and
   Kontron — deploy where the science happens, not in a loud data hall.

---

## The honesty rule

This demo shows real numbers or it shows nothing.

- Timings come from real `docker run` wall clocks. Variant counts are parsed from the VCF
  the pipeline actually produced.
- The app asks oneDNN to report the instruction set it **actually** selected, parses it back
  out of the logs, and displays it. If the reported ISA contradicts the requested ceiling,
  the run is marked **failed** rather than reporting a number we can't stand behind.
- A speedup is only ever shown when both legs succeeded **and** every parameter other than
  the AMX state was identical (enforced by a run fingerprint comparison).
- Anything estimated — expected runtimes, TCO figures — is rendered in a visually distinct
  "Illustrative" style and labelled as an assumption.
- Replay mode shows a permanent, unmistakable banner. Replays are recordings of real runs,
  never synthesised.

---

## How the AMX toggle works (staff talking point)

> Intel AMX is a matrix-multiply engine built into every core of this Xeon. DeepVariant's
> neural network runs through oneDNN, which picks the best instruction set available. We
> flip a single environment variable — `ONEDNN_MAX_CPU_ISA` — to cap what oneDNN is allowed
> to use. Set it to `AVX512_CORE_AMX` and the BF16 matrix tiles light up; set it to
> `AVX512_CORE` and oneDNN falls back to plain AVX-512 vector maths. Same server, same data,
> same core count, same container image — the only difference is whether the silicon that's
> already in the CPU is switched on. That's the whole story: **the acceleration is already
> inside the processor, you just have to turn it on.**

Everything else is held constant by construction: both legs of a race are built from a single
`RunSpec`, and the app compares a fingerprint of every shared parameter before it will report
a speedup.

---

## Quick start

```bash
git clone https://github.com/hvjaintel/GenomicsDemo.git
cd GenomicsDemo

./scripts/fetch_data.sh          # ~2 GB: reference + smoke + chr20
docker pull google/deepvariant:1.10.0
./scripts/preflight.sh --smoke   # verify the stack end to end
./run_demo.sh                    # launch the booth UI on :7860
```

Open `http://localhost:7860` full screen on the booth monitor.

---

## Prerequisites

| Requirement | Notes |
|---|---|
| Intel Xeon with AMX | Needs `amx_tile` + `amx_bf16` in `lscpu`. Sapphire Rapids or newer. |
| Docker Engine | The user running the demo must be in the `docker` group. |
| Python 3.10+ | `run_demo.sh` builds its own isolated virtualenv. |
| Free disk | ~10 GB for the core tier; ~56 GB more for the full WGS BAM. |
| `numactl` | Optional but recommended: `sudo apt-get install -y numactl` |

### Docker group access

If pre-flight reports **"cannot reach the daemon socket"**, this is a one-time fix:

```bash
sudo usermod -aG docker $USER
newgrp docker          # or log out and back in
```

### Staging the U.2 NVMe

`config.yaml` expects the data drive at `paths.data_root` (`/data/genomics`). If it isn't
mounted the app falls back to `./data` and pre-flight warns — the OS drive will **not** fit
the 46 GB WGS BAM. To claim a spare U.2 device (destructive — check the device name first
with `lsblk`):

```bash
sudo mkfs.ext4 -L genomics /dev/nvme2n1
sudo mkdir -p /data
echo 'LABEL=genomics /data ext4 defaults,noatime 0 2' | sudo tee -a /etc/fstab
sudo mount /data
sudo mkdir -p /data/genomics && sudo chown "$USER:$USER" /data/genomics
```

---

## Data staging

`scripts/fetch_data.sh` downloads from NIST GIAB and the public DeepVariant GCS bucket,
verifies everything, and lays files out to match `config.yaml`.

```bash
./scripts/fetch_data.sh                # core tier: reference + smoke + chr20 (~2 GB)
./scripts/fetch_data.sh --with-truth   # + GIAB HG002 v4.2.1 truth set (~170 MB)
./scripts/fetch_data.sh --with-wgs     # + full 35x WGS BAM (~46 GB)
./scripts/fetch_data.sh --all          # everything except FASTQ
./scripts/fetch_data.sh --verify-only  # re-verify staged files, download nothing
```

**Integrity.** GCS publishes an authoritative MD5 in the `x-goog-hash` response header; the
script captures it at download time and verifies against it. NCBI sibling checksum files are
used where published. A local sha256 is computed for every file and written to
`data/CHECKSUMS.sha256`. Once a sha256 is recorded in `config.yaml` it is **authoritative** —
a mismatch is a hard failure, and the script will never silently regenerate it.

The script prints the computed hashes on first download so you can paste them into
`config.yaml`. The core-tier values are already pinned there.

| Dataset | Size | Tier |
|---|---|---|
| GRCh38 no-ALT reference | 886 MB gz → 3.1 GB | core |
| NA12878 chr20 100 kb smoke BAM | 3.9 MB | core |
| HG002 chr20 35x BAM | 1.09 GB | core |
| HG002 full WGS 35x BAM | 46.0 GB | `--with-wgs` |
| HG002 GIAB truth v4.2.1 + BED | 168 MB | `--with-truth` |

> The reference `.fai` index is built by `scripts/build_fai.py`, a dependency-free generator
> whose output is byte-identical to the index GIAB publishes. No `samtools` needed.

> **Note on GIAB URLs:** older documentation points at
> `AshkenazimTrio/HG002_NA24385_son/latest/GRCh38/` — that path now 404s, because `latest`
> was repointed at a release with no `GRCh38` subdirectory. This demo pins the explicit
> `NISTv4.2.1` release instead, which is both reachable and reproducible.

---

## Booth-day runbook

### The morning before doors open

```bash
./scripts/fetch_data.sh --with-truth   # confirm data is staged and verified
./scripts/preflight.sh --pull --smoke  # pull the image, run a real 60s smoke test
```

Every check must be green, or a known-and-accepted amber. Then:

```bash
./run_demo.sh
```

Put the browser full screen (F11). Leave the **Status** tab up between visitors.

### The six-click visitor demo

1. **Status** — "96 cores, a terabyte of RAM, and the accelerators are already in the CPU."
2. **Dataset** — pick *HG002 chr20*. "A real human chromosome at 35x depth."
3. **AMX race** — press **RACE: AMX ON vs AMX OFF**.
4. Watch AMX-OFF run first, then AMX-ON overtake it.
5. **Speedup card** lands: "Built-in AMX = X.X× faster."
6. **Results** — same variant counts either way. Speed without an accuracy trade.

### Resetting between visitors

Nothing to reset. Press the button again — each run writes to its own directory under
`<data_root>/runs/`. To clear the screen, switch to **Status** and back.

### When something goes wrong

| Symptom | Action |
|---|---|
| A run fails mid-demo | The error is shown on screen. Press START again — one click. |
| Docker died | `sudo systemctl restart docker`, then re-run. |
| UI unresponsive | Ctrl-C, then `./run_demo.sh --skip-checks` (~2 s restart). |
| Hardware unavailable entirely | Set `demo_mode.policy: always` in `config.yaml` to replay a recorded run. The replay banner makes this obvious to the audience. |
| Disk filling up | `rm -rf <data_root>/runs/*` — run outputs only, never the staged inputs. |

### End of day

`Ctrl-C` in the terminal running `run_demo.sh`.

---

## Demo mode (replay)

If the hardware is offline the booth should still show something, clearly marked as a replay.

- `demo_mode.policy: auto` — replay only when a live run is impossible (the default)
- `demo_mode.policy: always` — force replay, for a dry rehearsal
- `demo_mode.policy: never` — always attempt a real run and surface any failure

Traces live in `traces/`. **A trace is a recording of a real run** — record one after a
successful live race. The repo deliberately ships without a fabricated trace; pre-flight
warns until you record a genuine one.

---

## Configuration

Everything tunable lives in `config.yaml`: image tags, dataset URLs and checksums, shard
count, NUMA policy, the AMX ISA values, sample menu, acoustics, and the TCO assumptions.
Nothing is hardcoded in the app.

For a machine-specific tweak that shouldn't be committed, create `config.local.yaml` — it is
deep-merged over `config.yaml` and is gitignored.

Useful knobs:

```yaml
compute:
  num_shards: null              # null = one per hardware thread (192 here)
  numa_policy: interleave_all   # or: none | bind_node
demo:
  port: 7860
demo_mode:
  policy: auto
```

---

## Pipeline paths

**BAM-in (default).** Feeds a pre-aligned BAM straight to `run_deepvariant`. Fast enough for
a live booth run. Uses `google/deepvariant:1.10.0`, which is pullable from Docker Hub and
runs on TensorFlow/oneDNN — so it honours the AMX toggle.

**fq2vcf (optional).** The full Open-Omics pipeline: FASTQ → bwa-mem2 → sort → DeepVariant.
Longer, best for the headline WGS story. The Intel-optimised images have no prebuilt Docker
Hub tag and must be built from the
[Open-Omics Acceleration Framework](https://github.com/IntelLabs/Open-Omics-Acceleration-Framework):

```bash
git clone https://github.com/IntelLabs/Open-Omics-Acceleration-Framework.git
cd Open-Omics-Acceleration-Framework/pipelines/deepvariant-based-germline-variant-calling-fq2vcf
docker build -f Dockerfile_fq2bams  -t open-omics/fq2bams:r1.5  .
docker build -f Dockerfile_bams2vcf -t open-omics/bams2vcf:r1.5 .
```

Then stage the FASTQs and build the index (slow, ~70 GB — do this days ahead):

```bash
./scripts/fetch_data.sh --with-fastq
./scripts/build_index.sh
python -m app.fq2vcf          # report readiness
```

These surface the BF16-quantised call-variant module (~2.7× faster call-variant) and the
parallel multi-process post-process module. Switch the active engine via `engines.*.default`
in `config.yaml`.

---

## Layout

```
config.yaml            single source of truth for every tunable
run_demo.sh            one-command launcher (builds its own venv)
docker-compose.yml     alternative launcher; DeepVariant still runs on the host
app/
  config.py            config.yaml loader + validation
  sysinfo.py           real CPU / NUMA / RAM / Docker / acoustics probing
  preflight.py         green-red readiness checks + smoke test
  runner.py            Docker orchestration and THE AMX TOGGLE
  parsing.py           stage progress, ISA verification, VCF counting
  replay.py            demo mode
  tco.py               efficiency panel maths (assumptions only)
  fq2vcf.py            optional FASTQ-in path
  main.py              Gradio app, six panels
  theme.py             booth theme, readable at 3 m
scripts/
  fetch_data.sh        tiered download + checksum verification
  preflight.sh         CLI pre-flight wrapper
  build_fai.py         dependency-free FASTA index builder
  build_index.sh       bwa-mem2 index (fq2vcf path only)
tests/                 unit tests for the toggle, parsing and TCO logic
data/                  staged datasets (gitignored)
traces/                recorded runs for demo mode
```

---

## Offline operation

The show floor may have no usable network. Before you travel:

```bash
./run_demo.sh --setup-only
.venv/bin/pip download -r requirements.txt -d wheelhouse
./scripts/fetch_data.sh --all
docker pull google/deepvariant:1.10.0
docker save google/deepvariant:1.10.0 | gzip > deepvariant-image.tar.gz
```

On site, with no network:

```bash
docker load < deepvariant-image.tar.gz
./run_demo.sh --offline
```

---

## Tests

```bash
.venv/bin/python -m pytest tests/ -q
```

The suite covers the guarantees that matter: that the AMX toggle changes nothing but the ISA,
that a speedup is refused when a comparison is invalid, that VCF counting is correct, and that
ISA verification catches a mismatch.

---

## Credits

Pipeline science: [DeepVariant](https://github.com/google/deepvariant) and Intel's
[Open-Omics](https://github.com/IntelLabs/Open-Omics-Acceleration-Framework) optimisations.
Data: [NIST GIAB](https://www.nist.gov/programs-projects/genome-bottle) and the public
DeepVariant test-data bucket. Hardware co-engineering: Intel + Kontron.
