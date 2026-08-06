"""Run ONE workload, headless, and report what actually happened.

`app.race` runs two legs to compare them. That is the wrong tool when you just
want the workload itself -- on the full genome a race means running 46 GB of
reads twice, which is hours of machine time to answer a question you did not
ask. This module runs a single leg.

Use it for the headline whole-genome run, for filling in a measured runtime in
config.yaml, or any time you want a number rather than a ratio.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time
from pathlib import Path

from .config import Config
from .runner import DeepVariantRunner, RunResult, time_ratio


def _fmt_hms(seconds: float | None) -> str:
    if seconds is None:
        return "unknown"
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m:02d}m {s:02d}s"
    if m:
        return f"{m}m {s:02d}s"
    return f"{s}s"


def report(result: RunResult, digest: str) -> None:
    """Print the evidence, not just the verdict."""
    print("\n" + "=" * 70)
    print("RESULT")
    print("=" * 70)

    if not result.succeeded:
        print(f"FAILED (exit {result.exit_code})")
        if result.error:
            print(f"  {result.error}")
        return

    print(f"Wall clock : {_fmt_hms(result.wall_clock_s)}")
    if result.stage_durations:
        width = max(len(name) for name in result.stage_durations)
        for name, secs in result.stage_durations.items():
            print(f"  {name.ljust(width)}  {_fmt_hms(secs)}")

    counts = result.variant_counts
    if counts is not None:
        print(f"Variants   : {counts.total:,}")

    # Ceiling vs use. The banner says what the CPU allows; the primitive counts
    # say what the run actually did. Only the second one is evidence.
    print(f"ISA ceiling: {result.reported_isa or 'not reported'} (permitted)")
    if result.compute_primitives:
        pct = 100.0 * result.amx_primitives / result.compute_primitives
        print(
            f"AMX kernels: {result.amx_primitives:,} of "
            f"{result.compute_primitives:,} compute primitives ({pct:.1f}%)"
        )
    else:
        print("AMX kernels: not measured (run with --verbose-isa to count)")

    if result.vcf_path:
        print(f"VCF        : {result.vcf_path}")
    print(f"Fingerprint: {digest}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m app.run",
        description="Run one DeepVariant workload and report measured results.",
    )
    parser.add_argument("--sample", default="chr20", help="sample id from config.yaml")
    parser.add_argument(
        "--cores",
        help="cpuset to pin to, e.g. '0-15'. Default: every core on the box.",
    )
    parser.add_argument(
        "--amx",
        action="store_true",
        help="raise the oneDNN ISA ceiling to permit AMX (does not force it)",
    )
    parser.add_argument(
        "--verbose-isa",
        action="store_true",
        help=(
            "count which kernels oneDNN dispatches. Costs wall clock and a very "
            "large log, so do NOT combine with a timing you intend to quote."
        ),
    )
    parser.add_argument("--quiet", action="store_true", help="progress only, not every log line")
    parser.add_argument("--json", metavar="PATH", help="also write the result as JSON")
    args = parser.parse_args(argv)

    cfg = Config.load()
    runner = DeepVariantRunner(cfg)
    spec = runner.build_spec(args.sample)
    if args.cores:
        spec = dataclasses.replace(spec, cpuset=args.cores)

    sample = cfg.sample(args.sample)
    print(f"Sample     : {args.sample}  ({sample.label})")
    print(f"BAM        : {spec.bam}")
    print(f"Reference  : {spec.reference}")
    print(f"Regions    : {spec.regions or 'whole genome'}")
    print(f"Shards     : {spec.num_shards}")
    cores = spec.core_count or os.cpu_count()
    print(f"Cores      : {spec.cpuset or 'all'} ({cores} logical)")
    print(f"Image      : {spec.engine_image}")
    print(f"AMX        : {'permitted' if args.amx else 'disabled'} (ISA ceiling)")
    print(f"Fingerprint: {spec.fingerprint_digest()}")

    if args.verbose_isa:
        print(
            "\nNOTE: verbose ISA logging is ON. It proves which kernels ran, but it\n"
            "      also slows the run and inflates the log. Do not quote this timing."
        )

    # A whole-genome run is hours long. Say so before it starts, using the
    # estimate in config.yaml, and be explicit that it is only an estimate.
    est = sample.runtime_fast_s
    if est:
        qualifier = "estimate" if sample.illustrative else "measured previously"
        print(f"\nExpected   : ~{_fmt_hms(est)} ({qualifier})")

    print("\n" + "=" * 70)
    started = time.time()
    result: RunResult | None = None
    last_pct = -1.0
    for event in runner.run(
        spec,
        amx_on=args.amx,
        run_id=f"run-{args.sample}",
        verbose_isa=args.verbose_isa,
        leg_id="single",
    ):
        if event.kind == "log" and not args.quiet:
            print(event.line, flush=True)
        elif event.kind == "progress":
            pct = event.overall_percent
            if pct - last_pct >= 1.0:
                last_pct = pct
                elapsed = _fmt_hms(time.time() - started)
                print(f"  [{pct:5.1f}%]  {elapsed} elapsed", flush=True)
        if event.result is not None:
            result = event.result

    if result is None:
        print("No result produced.", file=sys.stderr)
        return 1

    report(result, spec.fingerprint_digest())

    if args.json:
        payload = dataclasses.asdict(result)
        Path(args.json).write_text(json.dumps(payload, indent=2, default=str))
        print(f"\nWrote {args.json}")

    return 0 if result.succeeded else 1


if __name__ == "__main__":
    raise SystemExit(main())
