"""DEMO MODE — replay a recorded run so the booth never shows a blank screen.

A replay is a recording of a REAL run. It is never synthesised, and the UI is
required to display an unmistakable banner throughout so nobody can mistake a
replay for live hardware.
"""

from __future__ import annotations

import json
import platform
import socket
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from .config import Config
from .parsing import VariantCounts
from .runner import RunEvent, RunResult

#: Fields that distinguish a recording of a real run from a hand-authored file.
#: A trace missing any of these is rejected rather than replayed, because the
#: booth would otherwise present invented numbers as measurements.
PROVENANCE_FIELDS = ("recorded_at", "host", "legs")


class TraceError(Exception):
    """Raised when a trace cannot be trusted as a recording of a real run."""


@dataclass
class Trace:
    """A recorded run: the events, plus the result that ended it."""

    path: Path
    amx_on: bool
    recorded_at: str
    sample_id: str
    events: list[dict]
    result: dict

    @property
    def wall_clock_s(self) -> float | None:
        return self.result.get("wall_clock_s")


def record_trace(
    cfg: Config,
    name: str,
    sample_id: str,
    legs: dict[str, tuple[list[str], RunResult]],
) -> Path:
    """Persist a completed real run so it can be replayed later.

    `legs` maps "amx_on"/"amx_off" to (log lines, result).
    """
    cfg.traces_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "host": f"{socket.gethostname()} ({platform.platform()})",
        "sample_id": sample_id,
        "source": "real run recorded on this machine",
        "legs": {
            key: {
                "log": lines,
                "result": result.to_dict(),
            }
            for key, (lines, result) in legs.items()
        },
    }
    out = cfg.traces_dir / (name if name.endswith(".json") else f"{name}.json")
    out.write_text(json.dumps(payload, indent=2, default=str))
    return out


def validate_trace(data: dict) -> list[str]:
    """Return the reasons `data` cannot be trusted. Empty list means it is usable.

    This is deliberately strict. The cost of rejecting a good trace is a blank
    replay panel; the cost of accepting a fabricated one is showing a made-up
    speedup to a customer.
    """
    problems: list[str] = []
    if not isinstance(data, dict):
        return ["trace is not a JSON object"]

    for field_name in PROVENANCE_FIELDS:
        if not data.get(field_name):
            problems.append(f"missing '{field_name}' — cannot confirm this recorded a real run")

    legs = data.get("legs")
    if isinstance(legs, dict):
        if not legs:
            problems.append("'legs' is empty — nothing was recorded")
        for key, leg in legs.items():
            result = (leg or {}).get("result") if isinstance(leg, dict) else None
            if not isinstance(result, dict):
                problems.append(f"leg '{key}' has no result block")
                continue
            if "reported_isa" not in result:
                problems.append(
                    f"leg '{key}' has no 'reported_isa' — the ISA actually used was never "
                    "captured, so the AMX claim is unverifiable"
                )
            if result.get("started_at") in (None, 0) or result.get("finished_at") in (None, 0):
                problems.append(f"leg '{key}' has no real start/finish timestamps")
    elif legs is not None:
        problems.append("'legs' is not an object")

    return problems


def load_trace(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if validate_trace(data):
        return None
    return data


def list_traces(cfg: Config) -> list[tuple[Path, dict | None, list[str]]]:
    """Every trace on disk with its parsed payload and any provenance problems."""
    out = []
    if not cfg.traces_dir.exists():
        return out
    for path in sorted(cfg.traces_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            out.append((path, None, [f"unreadable: {exc}"]))
            continue
        out.append((path, data, validate_trace(data)))
    return out


def default_trace(cfg: Config) -> dict | None:
    rel = cfg.demo_mode.get("trace_file")
    if not rel:
        return None
    path = Path(rel)
    if not path.is_absolute():
        path = (Path(__file__).resolve().parent.parent / rel).resolve()
    return load_trace(path)


def _result_from_dict(data: dict) -> RunResult:
    counts_data = data.get("variant_counts")
    result = RunResult(
        run_id=data.get("run_id", "replay"),
        amx_on=bool(data.get("amx_on")),
        requested_isa=data.get("requested_isa", ""),
        reported_isa=data.get("reported_isa"),
        isa_verified=data.get("isa_verified"),
        started_at=data.get("started_at") or None,
        finished_at=data.get("finished_at"),
        exit_code=data.get("exit_code"),
        stage_durations=data.get("stage_durations", {}) or {},
        vcf_path=data.get("vcf_path"),
        output_dir=data.get("output_dir"),
        error=data.get("error"),
        fingerprint=data.get("fingerprint", ""),
        command=data.get("command", ""),
    )
    if counts_data:
        result.variant_counts = VariantCounts(
            **{k: v for k, v in counts_data.items() if k in VariantCounts.__annotations__}
        )
    return result


def replay(
    trace: dict,
    leg: str,
    speed: float = 1.0,
    max_line_delay_s: float = 0.03,
) -> Iterator[RunEvent]:
    """Yield the recorded events of one leg, paced to feel like a live run."""
    leg_data = (trace.get("legs") or {}).get(leg)
    if not leg_data:
        return

    result = _result_from_dict(leg_data.get("result", {}))
    total = result.wall_clock_s or 1.0
    lines = leg_data.get("log", [])
    n = max(1, len(lines))
    # Compress the original wall clock into something a booth visitor will
    # actually stand through, while keeping the recorded timings on screen.
    per_line = min(max_line_delay_s, (total / n) / max(speed, 0.001))

    for index, line in enumerate(lines, start=1):
        elapsed = total * index / n
        yield RunEvent(
            kind="log",
            run_id=result.run_id,
            amx_on=result.amx_on,
            line=line,
            elapsed_s=elapsed,
            overall_percent=100.0 * index / n,
            reported_isa=result.reported_isa,
        )
        time.sleep(per_line)

    yield RunEvent(
        kind="done",
        run_id=result.run_id,
        amx_on=result.amx_on,
        elapsed_s=total,
        overall_percent=100.0,
        reported_isa=result.reported_isa,
        result=result,
    )


def should_replay(cfg: Config, live_possible: bool) -> bool:
    policy = str(cfg.demo_mode.get("policy", "auto")).lower()
    if policy == "always":
        return default_trace(cfg) is not None
    if policy == "never":
        return False
    return not live_possible and default_trace(cfg) is not None


def banner_text(cfg: Config) -> str:
    return str(cfg.demo_mode.get("banner_text", "REPLAY MODE — recorded run, not live hardware"))


def _main(argv: list[str] | None = None) -> int:
    import argparse

    from .config import get_config

    parser = argparse.ArgumentParser(
        prog="python -m app.replay",
        description="Inspect DEMO MODE traces. Traces must be recordings of real runs.",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--list", action="store_true", help="list traces and their provenance")
    group.add_argument("--verify", metavar="FILE", help="verify one trace file")
    args = parser.parse_args(argv)

    cfg = get_config()

    if args.list:
        traces = list_traces(cfg)
        if not traces:
            print(f"No traces in {cfg.traces_dir}.")
            print("Record one from a real run — never write one by hand.")
            return 0
        for path, data, problems in traces:
            mark = "OK   " if not problems else "REJECT"
            recorded = (data or {}).get("recorded_at", "?")
            host = (data or {}).get("host", "?")
            print(f"[{mark}] {path.name}  recorded_at={recorded}  host={host}")
            for problem in problems:
                print(f"          - {problem}")
        return 0

    path = Path(args.verify)
    if not path.exists():
        print(f"No such trace: {path}")
        return 2
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        print(f"Unreadable trace: {exc}")
        return 2
    problems = validate_trace(data)
    if problems:
        print(f"REJECTED — {path} cannot be trusted as a recording of a real run:")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print(f"OK — {path} carries full provenance.")
    print(f"  recorded_at : {data.get('recorded_at')}")
    print(f"  host        : {data.get('host')}")
    print(f"  sample      : {data.get('sample_id')}")
    for key, leg in (data.get("legs") or {}).items():
        result = leg.get("result", {})
        print(f"  leg {key:<8}: isa={result.get('reported_isa')!r} "
              f"wall={result.get('wall_clock_s')}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
