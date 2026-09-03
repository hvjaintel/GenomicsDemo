"""Optional FASTQ-in path: bwa-mem2 alignment before variant calling.

This is the Open-Omics `fq2vcf` pipeline. It is deliberately NOT the default:
it needs a bwa-mem2 index of the reference (a slow, ~70 GB build) and takes far
longer than a booth visitor will wait. It exists for the headline WGS story and
for anyone who asks "but what about alignment?".

The instruction-set pinning applies here exactly as it does to the BAM-in path — same
ONEDNN_MAX_CPU_ISA ceiling, same everything else.
"""

from __future__ import annotations

import os
import shlex
from dataclasses import dataclass
from pathlib import Path

from .config import Config

CONTAINER_REF_DIR = "/ref"
CONTAINER_IN_DIR = "/input"
CONTAINER_OUT_DIR = "/output"

# bwa-mem2 emits these alongside the reference; all must exist for an index to
# count as complete.
BWA_MEM2_INDEX_SUFFIXES = (".0123", ".amb", ".ann", ".bwt.2bit.64", ".pac")


@dataclass
class Fq2VcfInputs:
    reference: Path
    r1: Path
    r2: Path
    index_prefix: Path


def index_is_complete(reference: Path) -> bool:
    return all((Path(str(reference) + suffix)).exists() for suffix in BWA_MEM2_INDEX_SUFFIXES)


def missing_index_files(reference: Path) -> list[str]:
    return [
        suffix
        for suffix in BWA_MEM2_INDEX_SUFFIXES
        if not Path(str(reference) + suffix).exists()
    ]


def resolve_inputs(cfg: Config) -> Fq2VcfInputs:
    """Locate the FASTQ pair and reference for the fq2vcf path."""
    fastq = cfg.fastq
    if not fastq.get("r1") or not fastq.get("r2"):
        raise RuntimeError(
            "fastq.r1 / fastq.r2 are not set in config.yaml. GIAB rotates these "
            "filenames, so they are a config value — check the current GIAB "
            "listing and update config.yaml."
        )

    data_root = cfg.data_root
    fq_dir = data_root / str(fastq.get("local_dir", "fastq"))
    r1 = fq_dir / str(fastq["r1"])
    r2 = fq_dir / str(fastq["r2"])

    gz = cfg.dataset_path("reference")
    reference = Path(str(gz)[:-3]) if str(gz).endswith(".gz") else gz

    for path, label in ((r1, "FASTQ R1"), (r2, "FASTQ R2"), (reference, "reference")):
        if not path.exists():
            raise RuntimeError(
                f"{label} not staged: {path}\n"
                "Run ./scripts/fetch_data.sh --with-fastq"
            )

    if not index_is_complete(reference):
        missing = ", ".join(missing_index_files(reference))
        raise RuntimeError(
            f"bwa-mem2 index is incomplete for {reference.name} (missing: {missing}).\n"
            "Build it with ./scripts/build_index.sh — this takes a long time and "
            "needs roughly 70 GB of free space, so do it well before the show."
        )

    return Fq2VcfInputs(reference=reference, r1=r1, r2=r2, index_prefix=reference)


def build_fq2bams_command(
    cfg: Config,
    inputs: Fq2VcfInputs,
    amx_on: bool,
    out_dir: Path,
    engine_key: str = "open_omics_fq2bams",
) -> list[str]:
    """`docker run` argv for the alignment stage.

    The ISA env is set the same way as the BAM-in path, so an fq2vcf run is
    just as valid a demonstration of the core-scaling race.
    """
    engine = cfg.engine(engine_key)
    isa = cfg.isa_for(amx_on)

    cmd: list[str] = [
        "docker", "run", "--rm",
        "-u", f"{os.getuid()}:{os.getgid()}",
        "-v", f"{inputs.reference.parent}:{CONTAINER_REF_DIR}:ro",
        "-v", f"{inputs.r1.parent}:{CONTAINER_IN_DIR}:ro",
        "-v", f"{out_dir}:{CONTAINER_OUT_DIR}",
        "-e", f"ONEDNN_MAX_CPU_ISA={isa}",
        "-e", f"DNNL_MAX_CPU_ISA={isa}",
    ]
    shm = cfg.compute.get("docker_shm_size")
    if shm:
        cmd += ["--shm-size", str(shm)]

    cmd += [
        engine.image,
        "python3", "/fq2bams.py",
        "--ref", f"{CONTAINER_REF_DIR}/{inputs.reference.name}",
        "--reads", f"{CONTAINER_IN_DIR}/{inputs.r1.name}", f"{CONTAINER_IN_DIR}/{inputs.r2.name}",
        "--output", CONTAINER_OUT_DIR,
        "--cpus", str(cfg.num_shards),
    ]
    return cmd


def describe(cfg: Config) -> str:
    """Human-readable readiness summary for the docs / UI."""
    try:
        inputs = resolve_inputs(cfg)
    except RuntimeError as exc:
        return f"fq2vcf path NOT ready:\n{exc}"
    return (
        "fq2vcf path ready:\n"
        f"  reference : {inputs.reference}\n"
        f"  R1        : {inputs.r1}\n"
        f"  R2        : {inputs.r2}\n"
        f"  bwa-mem2 index: complete\n"
        f"  command   : {' '.join(shlex.quote(c) for c in build_fq2bams_command(cfg, inputs, True, Path('/tmp/fq2vcf')))}"
    )


def main() -> int:
    from .config import get_config

    print(describe(get_config()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
