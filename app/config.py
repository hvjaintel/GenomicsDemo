"""Loads and validates config.yaml — the single source of truth for the demo.

Nothing else in the app is allowed to hardcode a path, image name, URL or
assumption. If it is tunable, it lives in config.yaml and arrives here.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = Path(os.environ.get("GENOMICS_DEMO_CONFIG", REPO_ROOT / "config.yaml"))
LOCAL_OVERRIDE_PATH = REPO_ROOT / "config.local.yaml"


class ConfigError(RuntimeError):
    """Raised when config.yaml is missing required structure."""


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


@dataclass(frozen=True)
class Sidecar:
    """An index or companion file that must sit next to a primary dataset."""

    url: str
    local: str
    md5_b64: str | None = None


@dataclass(frozen=True)
class Dataset:
    key: str
    name: str
    url: str
    local: str
    size_bytes: int | None
    sha256: str
    tier: str
    md5_b64: str | None = None
    sidecars: list[Sidecar] = field(default_factory=list)

    @property
    def has_recorded_checksum(self) -> bool:
        return bool(self.sha256.strip())


@dataclass(frozen=True)
class Sample:
    """A booth-facing menu entry on the dataset picker."""

    id: str
    label: str
    dataset: str
    regions: str | None
    blurb: str
    runtime_estimate_amx_on_s: int | None
    runtime_estimate_amx_off_s: int | None
    illustrative: bool
    show_in_booth: bool


@dataclass(frozen=True)
class Engine:
    key: str
    label: str
    image: str
    path: str
    buildable: bool
    default: bool
    build_repo: str | None = None
    build_context: str | None = None
    build_dockerfile: str | None = None


class Config:
    """Parsed view over config.yaml with light validation and path resolution."""

    def __init__(self, raw: dict[str, Any], source: Path):
        self.raw = raw
        self.source = source
        self._validate()

    # -- construction ----------------------------------------------------

    @classmethod
    def load(cls, path: Path | str | None = None) -> "Config":
        cfg_path = Path(path) if path else CONFIG_PATH
        if not cfg_path.exists():
            raise ConfigError(f"config.yaml not found at {cfg_path}")
        with cfg_path.open() as handle:
            raw = yaml.safe_load(handle) or {}
        # An optional, gitignored local override lets booth staff tweak a single
        # value on the day without editing the tracked config.
        if LOCAL_OVERRIDE_PATH.exists():
            with LOCAL_OVERRIDE_PATH.open() as handle:
                raw = _deep_merge(raw, yaml.safe_load(handle) or {})
        return cls(raw, cfg_path)

    def _validate(self) -> None:
        for section in ("demo", "paths", "engines", "amx", "compute", "datasets", "samples"):
            if section not in self.raw:
                raise ConfigError(f"config.yaml is missing required section '{section}'")
        for state in ("enabled", "disabled"):
            isa = self.raw["amx"].get(state, {}).get("onednn_max_cpu_isa")
            if not isa:
                raise ConfigError(f"amx.{state}.onednn_max_cpu_isa must be set")
        if self.raw["amx"]["enabled"]["onednn_max_cpu_isa"] == self.raw["amx"]["disabled"]["onednn_max_cpu_isa"]:
            raise ConfigError(
                "amx.on and amx.off resolve to the same ISA ceiling — the toggle "
                "would be meaningless and any reported speedup dishonest"
            )
        known = set(self.raw["datasets"])
        for sample in self.raw["samples"]:
            if sample["dataset"] not in known:
                raise ConfigError(
                    f"sample '{sample['id']}' references unknown dataset '{sample['dataset']}'"
                )

    # -- storage ---------------------------------------------------------

    @property
    def data_root(self) -> Path:
        """Preferred data root, falling back when the U.2 NVMe is not mounted."""
        primary = Path(self.raw["paths"]["data_root"])
        if primary.is_dir() and os.access(primary, os.W_OK):
            return primary
        fallback = self.raw["paths"].get("data_root_fallback", "./data")
        return (REPO_ROOT / fallback).resolve()

    @property
    def using_fallback_data_root(self) -> bool:
        primary = Path(self.raw["paths"]["data_root"])
        return not (primary.is_dir() and os.access(primary, os.W_OK))

    @property
    def runs_dir(self) -> Path:
        return self.data_root / self.raw["paths"].get("runs_dir", "runs")

    @property
    def traces_dir(self) -> Path:
        return (REPO_ROOT / self.raw["paths"].get("traces_dir", "./traces")).resolve()

    def dataset_path(self, key: str) -> Path:
        return self.data_root / self.dataset(key).local

    # -- datasets --------------------------------------------------------

    @property
    def datasets(self) -> dict[str, Dataset]:
        out: dict[str, Dataset] = {}
        for key, entry in self.raw["datasets"].items():
            out[key] = Dataset(
                key=key,
                name=entry["name"],
                url=entry["url"],
                local=entry["local"],
                size_bytes=entry.get("size_bytes"),
                sha256=entry.get("sha256") or "",
                tier=entry.get("tier", "core"),
                md5_b64=entry.get("md5_b64"),
                sidecars=[
                    Sidecar(url=s["url"], local=s["local"], md5_b64=s.get("md5_b64"))
                    for s in entry.get("sidecars", []) or []
                ],
            )
        return out

    def dataset(self, key: str) -> Dataset:
        try:
            return self.datasets[key]
        except KeyError as exc:
            raise ConfigError(f"unknown dataset '{key}'") from exc

    # -- samples ---------------------------------------------------------

    @property
    def samples(self) -> list[Sample]:
        return [
            Sample(
                id=s["id"],
                label=s["label"],
                dataset=s["dataset"],
                regions=s.get("regions"),
                blurb=s.get("blurb", ""),
                runtime_estimate_amx_on_s=s.get("runtime_estimate_amx_on_s"),
                runtime_estimate_amx_off_s=s.get("runtime_estimate_amx_off_s"),
                illustrative=bool(s.get("illustrative", True)),
                show_in_booth=bool(s.get("show_in_booth", True)),
            )
            for s in self.raw["samples"]
        ]

    def sample(self, sample_id: str) -> Sample:
        for sample in self.samples:
            if sample.id == sample_id:
                return sample
        raise ConfigError(f"unknown sample '{sample_id}'")

    @property
    def booth_samples(self) -> list[Sample]:
        return [s for s in self.samples if s.show_in_booth]

    # -- engines ---------------------------------------------------------

    @property
    def engines(self) -> dict[str, Engine]:
        return {
            key: Engine(
                key=key,
                label=entry["label"],
                image=entry["image"],
                path=entry.get("path", "bam_in"),
                buildable=bool(entry.get("buildable", False)),
                default=bool(entry.get("default", False)),
                build_repo=entry.get("build_repo"),
                build_context=entry.get("build_context"),
                build_dockerfile=entry.get("build_dockerfile"),
            )
            for key, entry in self.raw["engines"].items()
        }

    @property
    def default_engine(self) -> Engine:
        for engine in self.engines.values():
            if engine.default:
                return engine
        return next(iter(self.engines.values()))

    def engine(self, key: str) -> Engine:
        try:
            return self.engines[key]
        except KeyError as exc:
            raise ConfigError(f"unknown engine '{key}'") from exc

    # -- AMX -------------------------------------------------------------

    def isa_for(self, amx_on: bool) -> str:
        return self.raw["amx"]["enabled" if amx_on else "disabled"]["onednn_max_cpu_isa"]

    def amx_label(self, amx_on: bool) -> str:
        return self.raw["amx"]["enabled" if amx_on else "disabled"]["label"]

    @property
    def amx(self) -> dict[str, Any]:
        return self.raw["amx"]

    # -- misc sections ---------------------------------------------------

    @property
    def compute(self) -> dict[str, Any]:
        return self.raw["compute"]

    @property
    def demo(self) -> dict[str, Any]:
        return self.raw["demo"]

    @property
    def demo_mode(self) -> dict[str, Any]:
        return self.raw.get("demo_mode", {})

    @property
    def acoustics(self) -> dict[str, Any]:
        return self.raw.get("acoustics", {})

    @property
    def tco(self) -> dict[str, Any]:
        return self.raw.get("tco", {})

    @property
    def fastq(self) -> dict[str, Any]:
        return self.raw.get("fastq", {})

    @property
    def num_shards(self) -> int:
        configured = self.compute.get("num_shards")
        if configured:
            return int(configured)
        return os.cpu_count() or 1


_cached: Config | None = None


def get_config(reload: bool = False) -> Config:
    """Process-wide config singleton."""
    global _cached
    if _cached is None or reload:
        _cached = Config.load()
    return _cached
