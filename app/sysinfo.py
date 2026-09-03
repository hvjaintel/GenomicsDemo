"""Real system probing for the HOME / STATUS panel.

Everything here reads the actual machine. Where a value cannot be measured
(most often the acoustic reading, which needs a physical dB meter), the value
is returned flagged as a specification figure so the UI can label it honestly
rather than passing an assumption off as a measurement.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import Config

# Instruction-set extensions that matter to the story. This workload runs on
# the AVX-512 vector units, so those are what the demo reports.
ACCEL_FLAGS = {
    "avx512f": "AVX-512 Foundation",
    "avx512_bf16": "AVX-512 BFloat16",
    "avx512_vnni": "AVX-512 VNNI",
    "avx512_vbmi": "AVX-512 VBMI",
    "avx512dq": "AVX-512 Doubleword & Quadword",
}


def _run(cmd: list[str], timeout: int = 15) -> str:
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, check=False
        )
        return proc.stdout
    except (OSError, subprocess.SubprocessError):
        return ""


@dataclass
class CpuInfo:
    model: str = "unknown"
    sockets: int = 0
    cores_per_socket: int = 0
    threads_per_core: int = 0
    logical_cpus: int = 0
    numa_nodes: int = 0
    numa_map: dict[str, str] = field(default_factory=dict)
    l3_cache: str = ""
    flags: set[str] = field(default_factory=set)

    @property
    def physical_cores(self) -> int:
        return self.sockets * self.cores_per_socket

    @property
    def has_avx512(self) -> bool:
        return "avx512f" in self.flags

    @property
    def accel_summary(self) -> str:
        return "AVX-512" if self.has_avx512 else "none detected"

    @property
    def present_accel_flags(self) -> list[str]:
        return [f for f in ACCEL_FLAGS if f in self.flags]


def probe_cpu() -> CpuInfo:
    """Parse `lscpu`, falling back to /proc/cpuinfo for the flag list."""
    info = CpuInfo(logical_cpus=os.cpu_count() or 0)
    text = _run(["lscpu"])

    for line in text.splitlines():
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        key, value = key.strip(), value.strip()
        if key == "Model name":
            info.model = value
        elif key == "Socket(s)":
            info.sockets = int(value or 0)
        elif key == "Core(s) per socket":
            info.cores_per_socket = int(value or 0)
        elif key == "Thread(s) per core":
            info.threads_per_core = int(value or 0)
        elif key == "CPU(s)" and value.isdigit():
            info.logical_cpus = int(value)
        elif key == "NUMA node(s)":
            info.numa_nodes = int(value or 0)
        elif key.startswith("NUMA node") and key.endswith("CPU(s)"):
            info.numa_map[key.replace(" CPU(s)", "")] = value
        elif key == "L3 cache":
            info.l3_cache = value
        elif key == "Flags":
            info.flags = set(value.split())

    if not info.flags:
        try:
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("flags"):
                    info.flags = set(line.partition(":")[2].split())
                    break
        except OSError:
            pass

    return info


@dataclass
class MemoryInfo:
    total_bytes: int = 0
    available_bytes: int = 0

    @property
    def total_gb(self) -> int:
        return round(self.total_bytes / 1024**3)

    @property
    def available_gb(self) -> int:
        return round(self.available_bytes / 1024**3)


def probe_memory() -> MemoryInfo:
    info = MemoryInfo()
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, _, value = line.partition(":")
            kb = int(value.strip().split()[0])
            if key == "MemTotal":
                info.total_bytes = kb * 1024
            elif key == "MemAvailable":
                info.available_bytes = kb * 1024
    except (OSError, ValueError, IndexError):
        pass
    return info


@dataclass
class StorageInfo:
    path: str
    total_bytes: int
    free_bytes: int
    is_fallback: bool

    @property
    def free_gb(self) -> float:
        return self.free_bytes / 1024**3


def probe_storage(cfg: Config) -> StorageInfo:
    root = cfg.data_root
    root.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(root)
    return StorageInfo(
        path=str(root),
        total_bytes=usage.total,
        free_bytes=usage.free,
        is_fallback=cfg.using_fallback_data_root,
    )


@dataclass
class DockerInfo:
    installed: bool = False
    version: str = ""
    reachable: bool = False
    error: str = ""
    images: list[str] = field(default_factory=list)

    @property
    def permission_problem(self) -> bool:
        return self.installed and not self.reachable and "permission denied" in self.error.lower()


def probe_docker() -> DockerInfo:
    info = DockerInfo()
    if not shutil.which("docker"):
        info.error = "docker binary not found on PATH"
        return info
    info.installed = True
    info.version = _run(["docker", "--version"]).strip()

    proc = subprocess.run(
        ["docker", "image", "ls", "--format", "{{.Repository}}:{{.Tag}}"],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode == 0:
        info.reachable = True
        info.images = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]
    else:
        info.error = (proc.stderr or "docker daemon unreachable").strip()
    return info


@dataclass
class AcousticReading:
    dba: float
    live: bool
    note: str

    @property
    def label(self) -> str:
        return "live measurement" if self.live else "specification figure"


def probe_acoustics(cfg: Config) -> AcousticReading:
    """Return a live dB(A) reading if a meter is configured and present.

    There is deliberately no synthetic fallback reading: if we cannot measure,
    we return the configured specification number and say so.
    """
    acoustics = cfg.acoustics
    device = acoustics.get("live_meter_device")
    if acoustics.get("live_meter_enabled") and device and Path(str(device)).exists():
        try:
            raw = Path(str(device)).read_text().strip()
            match = re.search(r"\d+(\.\d+)?", raw)
            if match:
                return AcousticReading(
                    dba=float(match.group()),
                    live=True,
                    note=f"live from {device}",
                )
        except OSError:
            pass
    return AcousticReading(
        dba=float(acoustics.get("static_dba", 0)),
        live=False,
        note=str(acoustics.get("note", "")),
    )


@dataclass
class SystemSnapshot:
    cpu: CpuInfo
    memory: MemoryInfo
    storage: StorageInfo
    docker: DockerInfo
    acoustics: AcousticReading
    numactl_available: bool
    os_pretty: str
    kernel: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "cpu_model": self.cpu.model,
            "sockets": self.cpu.sockets,
            "physical_cores": self.cpu.physical_cores,
            "logical_cpus": self.cpu.logical_cpus,
            "numa_nodes": self.cpu.numa_nodes,
            "ram_gb": self.memory.total_gb,
            "accel": self.cpu.accel_summary,
            "avx512": self.cpu.has_avx512,
            "os": self.os_pretty,
            "kernel": self.kernel,
        }


def _os_pretty() -> str:
    try:
        for line in Path("/etc/os-release").read_text().splitlines():
            if line.startswith("PRETTY_NAME="):
                return line.partition("=")[2].strip().strip('"')
    except OSError:
        pass
    return "unknown"


def snapshot(cfg: Config) -> SystemSnapshot:
    return SystemSnapshot(
        cpu=probe_cpu(),
        memory=probe_memory(),
        storage=probe_storage(cfg),
        docker=probe_docker(),
        acoustics=probe_acoustics(cfg),
        numactl_available=shutil.which("numactl") is not None,
        os_pretty=_os_pretty(),
        kernel=_run(["uname", "-r"]).strip(),
    )
