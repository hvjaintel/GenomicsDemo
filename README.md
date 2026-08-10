# Genomics on Built-In Xeon Acceleration

A booth demo that runs **real germline variant calling (DeepVariant)** on an Intel Xeon
server and proves three things to a live audience:

1. Deep-learning variant calling runs fast on standard Xeon CPUs using the processor's
   **built-in vector units (AVX-512)** — no add-in cards, no GPU.
2. **It scales across Xeon cores**: HG002 chr20 takes ~343 s on 16 cores and ~123 s on all
   192 threads — a measured **2.8×** (2.78× and 2.86× on two separate races), with both
   legs calling exactly the same 210,390 variants. Same arithmetic, more cores, no
   trade-off.
3. It does so on a **quiet, air-cooled, bench-deployable** server — deploy where the science
   happens, not in a loud data hall.

> **About AMX.** This demo was originally designed around an AMX ON/OFF race. Measurement
> killed that idea and we kept the measurement instead of the idea: stock DeepVariant 1.10
> ships an **fp32** model, AMX has no fp32 path, and the tiles never execute a single
> kernel (0 of 95 compute primitives). Forcing bf16 does engage AMX — and is ~1.6× *slower*
> and produces genotype likelihoods DeepVariant itself rejects. The full evidence,
> including how to reproduce it, is in **[docs/AMX-FINDINGS.md](docs/AMX-FINDINGS.md)**.
> The AMX comparison is still in the app, as an evidence panel rather than a headline.

---

## The honesty rule

This demo shows real numbers or it shows nothing.

- Timings come from real `docker run` wall clocks. Variant counts are parsed from the VCF
  the pipeline actually produced.
- The app asks oneDNN to report the instruction set it **actually** selected, parses it back
  out of the logs, and displays it. If the reported ISA contradicts the requested ceiling,
  the run is marked **failed** rather than reporting a number we can't stand behind.
- A speedup is only ever shown when both legs succeeded **and** every parameter other than
  the single thing under test — the core budget, or the AMX state — was identical
  (enforced by a run fingerprint comparison).
- The app reports what oneDNN was **permitted** to use and what it **actually** used as two
  separate facts. Counting AMX kernel dispatches is the only way to tell the difference
  between "AMX was available" and "AMX did the work"; conflating them is exactly how this
  project nearly shipped a 1.00× speedup as a headline.
- Anything estimated — expected runtimes, TCO figures — is rendered in a visually distinct
  "Illustrative" style and labelled as an assumption.
- Replay mode shows a permanent, unmistakable banner. Replays are recordings of real runs,
  never synthesised.
- The DNA helix on the Run console is an **activity** indicator, not a progress bar. It turns
  only while the container is producing output and freezes when output stops, so a wedged run
  looks wedged. Progress is the stage bars, which are parsed from DeepVariant's own output.

---

## How the scaling race works (staff talking point)

> We run the same genome, the same container, the same binary and the same 192 shards
> twice. The only thing we change is how many of this server's cores the job is allowed to
> touch — 16 cores, then all 192 threads. Nothing else moves: same data, same instruction
> set, same everything. Both runs call exactly the same 210,390 variants, so this isn't a
> quality trade — it's the same work, finished sooner. That's the point of a server like
> this: **the throughput is already in the box, and it scales.**

### If a visitor asks about AMX (staff talking point)

> Good question, and the honest answer is more interesting than a slide. AMX is a
> matrix-multiply engine in every core of this Xeon, and it's genuinely fast — but it only
> works on bf16 and int8. The public DeepVariant model is fp32, so oneDNN runs every
> convolution on AVX-512 and the AMX tiles sit idle. We measured it: zero of ninety-five
> compute operations touched AMX. We then forced the model into bf16, which *does* light up
> the tiles — and it came out slower, and the accuracy dropped far enough that DeepVariant's
> own validation rejected the output. So on this workload, today, the Xeon story is core
> scaling and AVX-512. Give us a bf16-trained or int8-quantised model and AMX becomes the
> story instead.

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

## The bench server

Everything measured in this repo was run on this machine. Quote numbers from here
only alongside this configuration — a wall-clock time without its hardware is not a
claim anyone can check.

| | |
|---|---|
| CPU | 2x Intel Xeon 6740P — 48 cores/socket, **96 cores / 192 threads** total |
| Memory | 1 TB |
| OS disk | 1x M.2 NVMe 960 GB |
| Data disks | 2x U.2 NVMe 4 TB |

Datasets live on a U.2 data disk (`/mnt/nvme2n1`), never the OS disk — the WGS BAM
alone is 46 GB. See "Persisting the data mount" below.

### Measured whole-genome run

HG002, 35x, GRCh38, DeepVariant 1.10.0, all 192 threads, 192 shards:

| Stage | Time |
|---|---|
| `make_examples` | 13m 50s |
| `call_variants` | 11m 05s |
| `postprocess_variants` | 49s |
| **Total** | **25m 47s** |

7,709,239 variants called. For scale, Google's own published figure for the same
version is 69 minutes on a 96-vCPU cloud instance.

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

`config.yaml` expects the data drive at `paths.data_root`, which is
`/mnt/nvme2n1/genomics`. To claim a spare U.2 device (destructive — check the device
name against `lsblk` first, and be certain it is not the OS disk):

```bash
sudo mkfs.ext4 -L genomics /dev/nvme2n1
sudo mkdir -p /mnt/nvme2n1
sudo mount /dev/nvme2n1 /mnt/nvme2n1
sudo mkdir -p /mnt/nvme2n1/genomics && sudo chown "$USER:$USER" /mnt/nvme2n1/genomics
```

### Persisting the data mount

**Do this, or a reboot will appear to delete all your data.** A mount created with
`mount` alone does not survive a restart. When it vanishes, the configured data root
stops existing, the app falls back to `./data` on the OS drive, and every dataset
reports as missing — which looks exactly like data loss but is not. The files are
still on the unmounted disk.

Add it to `/etc/fstab` so it comes back automatically:

```bash
# Substitutes the UUID itself -- do not retype it, and do not paste a
# placeholder. A literal "<uuid>" in fstab is accepted silently by `mount -a`
# when `nofail` is set, so the mount just never happens and the next reboot
# looks like data loss again.
echo "UUID=$(sudo blkid -s UUID -o value /dev/nvme2n1) /mnt/nvme2n1 ext4 defaults,noatime,nofail 0 2" \
  | sudo tee -a /etc/fstab

sudo systemctl daemon-reload
findmnt --verify                            # must report no [E] lines
sudo mount -a                               # verify it mounts cleanly NOW
findmnt /mnt/nvme2n1                        # should print the device
```

`findmnt --verify` is the check that matters. If it prints
`[E] unreachable on boot required source: UUID=<uuid>`, the placeholder was
pasted literally; fix that line before rebooting:

```bash
sudo sed -i "s|^UUID=<uuid> /mnt/nvme2n1 .*$|UUID=$(sudo blkid -s UUID -o value /dev/nvme2n1) /mnt/nvme2n1 ext4 defaults,noatime,nofail 0 2|" /etc/fstab
```

`nofail` matters: without it, a missing data disk drops the machine to an emergency
shell at boot rather than starting normally.

If a reboot has already dropped the mount, nothing is lost — just remount:

```bash
sudo mount /mnt/nvme2n1
```

`fetch_data.sh` and `python -m app.run` both detect this state (an empty mountpoint)
and refuse to run rather than re-downloading tens of GB onto the OS disk. Override
with `--allow-fallback-root` only if you genuinely intend to stage data on `/`.

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

### Where the scratch space goes

`run_deepvariant` starts one `make_examples` process per shard, and each is a self-extracting
Bazel binary that unpacks its runfiles tree into `$TMPDIR`. Measured on this box at
`num_shards: 192`:

| | |
|---|---|
| Peak scratch during `make_examples` | **6.5 GB** |
| Files at peak | ~55,000 |
| What it is | ~34 MB of Bazel runfiles × 192 shards, plus GNU parallel's per-job output buffers |

That figure is from the **smoke** sample — 100 kb of chr20. It scales with `num_shards`, not
with genome size, so a whole-genome run needs the same headroom for the same reason.

The container's `/tmp` is therefore bind-mounted to `<data_root>/runs/<run>-tmp` on the U.2,
and `TMPDIR`, `HOME` and `MPLCONFIGDIR` all point into it. Without that mount it lands on the
container's writable layer — `/var/lib/docker` on the OS disk — and the run dies with:

```
parallel: Error: Cannot append to buffer file in /tmp.
parallel: Error: Is the disk full?
DeepVariant exited with code 255
```

The directory is deleted when the run ends. Pre-flight's **OS disk** row watches the
filesystem behind Docker independently of the data volume, because a healthy U.2 says nothing
about the disk holding the images.

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
3. **Scaling race** — press **RACE: 16 cores vs all 192 threads**.
4. Watch the 16-core leg run first (~6 min), then the full machine overtake it (~2 min).
5. **Speedup card** lands: "Xeon scales: all 192 threads vs 16 cores — ~2.8×", alongside the
   variant counts proving both legs produced the same answer.
6. **Results** — same variant counts either way. Speed without an accuracy trade.

### Resetting between visitors

Nothing to reset. Press the button again — each run writes to its own directory under
`<data_root>/runs/`. To clear the screen, switch to **Status** and back.

### When something goes wrong

| Symptom | Action |
|---|---|
| A run fails mid-demo | The error is shown on screen. Press START again — one click. |
| The helix stops turning mid-run | Expected during long `call_variants` batches; the caption says how long output has been quiet. If it stays frozen for minutes, check `docker ps` — the container may be wedged. |
| Docker died | `sudo systemctl restart docker`, then re-run. |
| UI unresponsive | Ctrl-C, then `./run_demo.sh --skip-checks` (~2 s restart). |
| Hardware unavailable entirely | Set `demo_mode.policy: always` in `config.yaml` to replay a recorded run. The replay banner makes this obvious to the audience. |
| Disk filling up | `rm -rf <data_root>/runs/*` — run outputs and scratch only, never the staged inputs. |
| `parallel: Cannot append to buffer file in /tmp. Is the disk full?` | The container's scratch is not landing on the data volume. Check the `docker run` line on screen for a `-v .../runs/<run>-tmp:/tmp` mount, and check pre-flight's **OS disk** row. See "Where the scratch space goes" below. |

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
runs on TensorFlow/oneDNN — so it honours the AMX toggle (which, as documented in
docs/AMX-FINDINGS.md, changes nothing measurable on the shipped fp32 model).

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
  runner.py            Docker orchestration, core budget and ISA toggle
  race.py              headless two-phase race (verify dispatch, then measure)
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

The suite covers the guarantees that matter: that the scaling race changes nothing but
`--cpuset-cpus`, that the AMX toggle changes nothing but the ISA,
that a speedup is refused when a comparison is invalid, that VCF counting is correct, and that
ISA verification catches a mismatch.

---

## Credits

Pipeline science: [DeepVariant](https://github.com/google/deepvariant) and Intel's
[Open-Omics](https://github.com/IntelLabs/Open-Omics-Acceleration-Framework) optimisations.
Data: [NIST GIAB](https://www.nist.gov/programs-projects/genome-bottle) and the public
DeepVariant test-data bucket.
