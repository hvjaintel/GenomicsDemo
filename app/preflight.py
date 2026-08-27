"""Pre-flight readiness checks — the green/red panel booth staff see first.

Each check returns a real, actionable verdict. A check never reports OK on the
basis of an assumption; if something cannot be determined it reports WARN and
says why.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from .config import Config
from .sysinfo import SystemSnapshot, snapshot


class Status(str, Enum):
    OK = "ok"
    WARN = "warn"
    FAIL = "fail"

    @property
    def icon(self) -> str:
        return {"ok": "✅", "warn": "⚠️", "fail": "❌"}[self.value]


@dataclass
class Check:
    name: str
    status: Status
    detail: str
    remedy: str = ""
    blocking: bool = False

    def as_row(self) -> list[str]:
        return [self.status.icon, self.name, self.detail, self.remedy]


@dataclass
class PreflightReport:
    checks: list[Check]
    snap: SystemSnapshot

    @property
    def ready(self) -> bool:
        """True when nothing blocking has failed — i.e. a live run is possible."""
        return not any(c.status is Status.FAIL and c.blocking for c in self.checks)

    @property
    def all_green(self) -> bool:
        return all(c.status is Status.OK for c in self.checks)

    @property
    def summary(self) -> str:
        if self.all_green:
            return "READY — all checks green"
        if self.ready:
            warns = sum(1 for c in self.checks if c.status is not Status.OK)
            return f"READY WITH WARNINGS — {warns} item(s) need attention"
        return "NOT READY — live runs are blocked"

    def as_rows(self) -> list[list[str]]:
        return [c.as_row() for c in self.checks]


def _check_avx512(snap: SystemSnapshot) -> Check:
    cpu = snap.cpu
    if cpu.has_avx512:
        return Check(
            "Built-in acceleration",
            Status.OK,
            f"{cpu.model} — {', '.join(cpu.present_accel_flags)}",
        )
    return Check(
        "Built-in acceleration",
        Status.FAIL,
        f"{cpu.model} does not expose AVX-512",
        "This demo runs DeepVariant on the CPU's AVX-512 vector units. "
        "Use an Intel Xeon with avx512f.",
        blocking=True,
    )


def _check_topology(snap: SystemSnapshot) -> Check:
    cpu = snap.cpu
    detail = (
        f"{cpu.sockets} sockets · {cpu.physical_cores} cores · "
        f"{cpu.logical_cpus} threads · {cpu.numa_nodes} NUMA nodes · "
        f"{snap.memory.total_gb} GB RAM"
    )
    return Check("CPU topology & memory", Status.OK, detail)


def _check_docker(snap: SystemSnapshot) -> Check:
    docker = snap.docker
    if not docker.installed:
        return Check(
            "Docker",
            Status.FAIL,
            docker.error,
            "Install Docker Engine: https://docs.docker.com/engine/install/ubuntu/",
            blocking=True,
        )
    if docker.permission_problem:
        return Check(
            "Docker",
            Status.FAIL,
            f"{docker.version} installed, but this user cannot reach the daemon socket",
            "One-time fix, then log out and back in:\n"
            "    sudo usermod -aG docker $USER && newgrp docker",
            blocking=True,
        )
    if not docker.reachable:
        return Check(
            "Docker",
            Status.FAIL,
            docker.error or "daemon unreachable",
            "Start the daemon:  sudo systemctl start docker",
            blocking=True,
        )
    return Check("Docker", Status.OK, f"{docker.version} — daemon reachable")


def _check_image(cfg: Config, snap: SystemSnapshot) -> Check:
    engine = cfg.default_engine
    if not snap.docker.reachable:
        return Check(
            "Pipeline image",
            Status.WARN,
            f"cannot verify {engine.image} — Docker is unreachable",
            "Resolve the Docker check first.",
        )
    if engine.image in snap.docker.images:
        return Check("Pipeline image", Status.OK, f"{engine.image} present locally")
    return Check(
        "Pipeline image",
        Status.FAIL,
        f"{engine.image} not pulled",
        f"Pull it before the show floor opens (it must be local to run offline):\n"
        f"    docker pull {engine.image}",
        blocking=True,
    )


def _check_datasets(cfg: Config) -> list[Check]:
    checks: list[Check] = []
    # When the data volume is merely unmounted, every dataset reports missing.
    # Telling someone to re-download 46 GB in that state is actively harmful --
    # it targets the OS disk and fills it. Point at the mount instead.
    unmounted = cfg.data_disk_looks_unmounted
    mountpoint = Path(cfg.raw["paths"]["data_root"]).parent
    for sample in cfg.booth_samples:
        dataset = cfg.dataset(sample.dataset)
        path = cfg.data_root / dataset.local
        missing = [] if path.exists() else [dataset.local]
        for sidecar in dataset.sidecars:
            if not (cfg.data_root / sidecar.local).exists():
                missing.append(sidecar.local)

        flag = " --with-wgs" if dataset.tier == "wgs" else ""
        if missing:
            if unmounted:
                detail = "unreachable — the data volume is not mounted"
                fix = f"sudo mount {mountpoint}   (do NOT re-download; the files are on that disk)"
            else:
                detail = f"not staged ({len(missing)} file(s) missing)"
                fix = f"./scripts/fetch_data.sh{flag}"
            checks.append(
                Check(f"Dataset · {sample.label}", Status.WARN, detail, fix)
            )
            continue

        size = path.stat().st_size
        if dataset.size_bytes and size != dataset.size_bytes:
            checks.append(
                Check(
                    f"Dataset · {sample.label}",
                    Status.FAIL,
                    f"size mismatch — expected {dataset.size_bytes:,} bytes, found {size:,}",
                    f"File is truncated or changed upstream. Delete and re-fetch:\n"
                    f"    rm {path} && ./scripts/fetch_data.sh{flag}",
                    blocking=True,
                )
            )
            continue

        note = "verified size" if dataset.size_bytes else "present"
        if dataset.is_derived:
            # A derived file has no upstream checksum by definition. Saying
            # "sha256 not pinned" here would read as a gap when it is simply
            # not applicable -- state the real provenance instead.
            note = dataset.provenance
        elif dataset.has_recorded_checksum:
            note += " · sha256 pinned in config"
        else:
            note += " · sha256 not yet pinned in config.yaml"
        checks.append(Check(f"Dataset · {sample.label}", Status.OK, f"{size / 1024**3:.2f} GB — {note}"))
    return checks


def _check_reference(cfg: Config) -> Check:
    gz = cfg.dataset_path("reference")
    plain = Path(str(gz)[:-3]) if str(gz).endswith(".gz") else gz
    if plain.exists() and (plain.with_suffix(plain.suffix + ".fai")).exists():
        return Check("Reference genome", Status.OK, f"{plain.name} + .fai ready")
    if plain.exists():
        return Check(
            "Reference genome",
            Status.WARN,
            f"{plain.name} present but .fai index is missing",
            "Build it:  samtools faidx " + str(plain) + "\n"
            "(or install samtools:  sudo apt-get install -y samtools)",
        )
    if gz.exists():
        return Check(
            "Reference genome",
            Status.WARN,
            "only the bgzipped reference is staged; it must be decompressed",
            "./scripts/fetch_data.sh",
        )
    if cfg.data_disk_looks_unmounted:
        mountpoint = Path(cfg.raw["paths"]["data_root"]).parent
        return Check(
            "Reference genome",
            Status.FAIL,
            "unreachable — the data volume is not mounted",
            f"sudo mount {mountpoint}   (do NOT re-download; the 3 GB reference is on that disk)",
            blocking=True,
        )
    return Check(
        "Reference genome",
        Status.FAIL,
        "not staged",
        "./scripts/fetch_data.sh",
        blocking=True,
    )


def _check_storage(cfg: Config, snap: SystemSnapshot) -> Check:
    storage = snap.storage
    free_gb = storage.free_gb
    if storage.is_fallback:
        configured = cfg.raw["paths"]["data_root"]
        if cfg.data_disk_looks_unmounted:
            mountpoint = Path(configured).parent
            return Check(
                "Data storage",
                Status.FAIL,
                f"the data volume is NOT MOUNTED — {mountpoint} exists but is empty. "
                f"Falling back to {storage.path} on the OS drive ({free_gb:.0f} GB free).",
                f"Your data is still on that disk; nothing was lost. Remount it:\n"
                f"    sudo mount {mountpoint}\n"
                f"Then add it to /etc/fstab so a reboot cannot drop it again "
                f"(see README -> Persisting the data mount).",
                blocking=True,
            )
        return Check(
            "Data storage",
            Status.WARN,
            f"using repo-local fallback {storage.path} — {free_gb:.0f} GB free. "
            f"Configured U.2 path {configured} is not mounted or not writable.",
            "Mount and claim the U.2 NVMe (see README -> Staging the U.2 NVMe). "
            "The full 46 GB WGS sample will not fit on the OS drive.",
        )
    if free_gb < 20:
        return Check(
            "Data storage",
            Status.FAIL,
            f"{storage.path} has only {free_gb:.0f} GB free",
            "Free space or move paths.data_root to the U.2 NVMe.",
            blocking=True,
        )
    return Check("Data storage", Status.OK, f"{storage.path} — {free_gb:.0f} GB free")


def _lv_extend_hint() -> str:
    """Suggest extending the root LV, naming the device actually in use.

    The volume group name is read from the live mount rather than written into
    the source. This box has two similarly named groups -- ubuntu-vg-1 backs the
    running system, ubuntu-vg belongs to an idle install on the other NVMe --
    and a hardcoded hint is one transcription slip away from pointing booth
    staff at the wrong one.
    """
    try:
        source = subprocess.run(
            ["findmnt", "-no", "SOURCE", "/"],
            capture_output=True, text=True, check=False, timeout=5,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""
    if not source.startswith("/dev/mapper/"):
        return ""

    # Device-mapper escapes a literal hyphen by doubling it, so unescaping has
    # to happen before the vg-lv split or "ubuntu--vg--1-ubuntu--lv" parses as
    # the group "ubuntu".
    name = source.split("/dev/mapper/", 1)[1]
    parts = [p.replace("\0", "-") for p in name.replace("--", "\0").split("-")]
    if len(parts) != 2:
        return ""
    vg, lv = parts
    return (
        f"\nIf the root LV is smaller than its volume group, extend it "
        f"(online, no reboot):\n"
        f"    sudo vgs {vg}                       # confirm VFree first\n"
        f"    sudo lvextend -r -l +100%FREE /dev/{vg}/{lv}"
    )


def _check_os_disk(cfg: Config) -> Check:
    """Free space on the filesystem backing Docker.

    Distinct from the data check: even with scratch and outputs on the U.2, the
    OS disk still carries Docker's images and every container's writable layer.
    This is the filesystem that took the demo down -- DeepVariant's per-shard
    Bazel runfiles filled it and GNU parallel died with "Cannot append to buffer
    file in /tmp. Is the disk full?" -- so it is worth its own line rather than
    being inferred from a healthy-looking data volume.
    """
    docker_root = Path("/var/lib/docker")
    target = docker_root if docker_root.exists() else Path("/")
    try:
        usage = shutil.disk_usage(target)
    except OSError as exc:
        return Check("OS disk", Status.WARN, f"could not read {target}: {exc}", "")

    free_gb = usage.free / 1024**3
    used_pct = 100.0 * (usage.total - usage.free) / usage.total if usage.total else 0.0
    detail = f"{target} — {free_gb:.0f} GB free ({used_pct:.0f}% used)"
    remedy = (
        "Docker images and container layers live here. Reclaim with:\n"
        "    docker system df             # what is taking the space\n"
        "    docker image prune -a        # unused images\n"
        f"{_lv_extend_hint()}"
    )
    if free_gb < 10:
        return Check("OS disk", Status.FAIL, detail, remedy, blocking=True)
    if free_gb < 25:
        return Check("OS disk", Status.WARN, detail, remedy)
    return Check("OS disk", Status.OK, detail)


def _check_numactl(snap: SystemSnapshot) -> Check:
    if snap.numactl_available:
        return Check("NUMA pinning", Status.OK, "numactl available on the host")
    return Check(
        "NUMA pinning",
        Status.WARN,
        "numactl not installed — runs proceed without explicit NUMA placement",
        "Optional but recommended for repeatable timings on this 4-node box:\n"
        "    sudo apt-get install -y numactl",
    )


def _check_trace(cfg: Config) -> Check:
    trace = Path(cfg.demo_mode.get("trace_file", ""))
    if not trace.is_absolute():
        trace = Path(cfg.raw["paths"].get("traces_dir", "./traces")).parent / trace
    trace = (Path(__file__).resolve().parent.parent / cfg.demo_mode.get("trace_file", "")).resolve()
    if trace.exists():
        return Check("Demo-mode fallback", Status.OK, f"replay trace present ({trace.name})")
    return Check(
        "Demo-mode fallback",
        Status.WARN,
        "no recorded trace — the booth has no fallback if hardware runs fail",
        "Record one after a successful live run (Run Console -> Save as replay trace).",
    )


def run_preflight(cfg: Config) -> PreflightReport:
    snap = snapshot(cfg)
    checks: list[Check] = [
        _check_avx512(snap),
        _check_topology(snap),
        _check_docker(snap),
        _check_image(cfg, snap),
        _check_storage(cfg, snap),
        _check_os_disk(cfg),
        _check_reference(cfg),
        *_check_datasets(cfg),
        _check_numactl(snap),
        _check_trace(cfg),
    ]
    return PreflightReport(checks=checks, snap=snap)


def smoke_test(cfg: Config, timeout_s: int = 900) -> tuple[bool, str]:
    """Run the tiny BAM end to end. The only honest proof the stack works."""
    from .runner import DeepVariantRunner, RunnerError

    runner = DeepVariantRunner(cfg)
    try:
        spec = runner.build_spec("smoke")
    except RunnerError as exc:
        return False, str(exc)

    lines: list[str] = []
    deadline = timeout_s
    result = None
    import time

    started = time.time()
    for event in runner.run(spec, amx_on=True, run_id="smoke"):
        if event.kind == "log":
            lines.append(event.line)
        elif event.kind == "done":
            result = event.result
        if time.time() - started > deadline:
            runner.cancel()
            return False, f"smoke test exceeded {timeout_s}s"

    if result is None:
        return False, "smoke test produced no result"
    if not result.succeeded:
        return False, f"{result.error}\n" + "\n".join(lines[-15:])

    counts = result.variant_counts
    found = counts.total if counts else 0
    if found == 0:
        return False, "smoke test completed but produced zero variants — check inputs"
    return True, (
        f"smoke test passed in {result.wall_clock_s:.1f}s — "
        f"{found:,} variants, ISA reported: {result.reported_isa or 'not reported'}"
    )


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m app.preflight",
        description="Check the booth box is ready to run the demo.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="after the checks, run DeepVariant for real on the tiny smoke BAM",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=900,
        help="seconds to allow the smoke run (default: 900)",
    )
    args = parser.parse_args(argv)

    cfg = Config.load()
    report = run_preflight(cfg)
    width = max(len(c.name) for c in report.checks) + 2
    print()
    print("=" * 78)
    print("  PRE-FLIGHT — Intel Xeon Genomics Demo")
    print("=" * 78)
    for check in report.checks:
        print(f"{check.status.icon}  {check.name:<{width}} {check.detail}")
        if check.remedy:
            for line in check.remedy.splitlines():
                print(f"    {'':<{width}} → {line}")
    print("-" * 78)
    print(f"  {report.summary}")
    print("=" * 78)
    print()

    if not args.smoke:
        return 0 if report.ready else 1

    if not report.ready:
        print("Skipping smoke run — blocking issues above must be resolved first.")
        return 1

    print("Running DeepVariant for real on the smoke BAM. This is not a simulation.")
    print("Expect a few minutes on first run.\n")
    passed, detail = smoke_test(cfg, timeout_s=args.timeout)
    print(f"{'✅' if passed else '❌'}  {detail}\n")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
