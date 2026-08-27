"""Run a two-leg DeepVariant race from the command line and record a trace.

The booth UI can do this too, but a headless path matters: it is how you
validate a fresh machine over SSH, and how you produce the replay trace that
keeps the stand running when the hardware is busy or the network is down.

    python -m app.race --sample smoke
    python -m app.race --mode scaling --sample chr20 --record chr20-booth.json

Two modes, because measurement said so:

  scaling (default)
      Same binary, same data, same ISA settings -- only the core budget
      differs (Docker --cpuset-cpus). A real, large, honest Xeon result with
      no precision trade. This is the booth headline.

  amx
      AMX permitted versus AMX disabled, on the one DeepVariant build that can
      actually dispatch it. Stock 1.10 runs an fp32 graph and AMX has no fp32
      path, so its tiles never run a kernel; 1.5.0 accepts a bf16 graph-rewrite
      flag and puts every convolution on AMX. The race therefore runs 1.5.0 and
      says so on screen, along with the awkward part: 1.10 with the tiles idle
      still finishes faster than 1.5.0 with them fully engaged. See
      docs/AMX-FINDINGS.md.

Both legs run sequentially, never concurrently. Two simultaneous 192-shard
runs would fight over the same cores and neither number would mean anything.
"""

from __future__ import annotations

import argparse
import dataclasses
import time
import sys

from .config import Config
from .replay import record_trace
from .run import PROGRESS_EVERY_S, _active_stage
from .runner import DeepVariantRunner, RunResult, RunSpec, time_ratio


# A small slice of the same BAM, used only to observe which oneDNN kernels get
# dispatched. Same reference, same model, same code path -- just less of it.
VERIFY_REGION = "chr20:10,000,000-10,200,000"
VERIFY_SHARDS = 16


def _verify_dispatch(runner: DeepVariantRunner, spec: RunSpec, amx_on: bool) -> RunResult | None:
    """Prove which kernels oneDNN actually chose, with verbose logging ON.

    Kept separate from the timed run on purpose: ONEDNN_VERBOSE=1 prints a line
    per primitive execution (270 MB on a chr20 run) and bf16 issues more
    primitives than fp32, so leaving it on during timing would penalise the AMX
    leg and understate the very thing we are measuring.
    """
    small = dataclasses.replace(spec, regions=VERIFY_REGION, num_shards=VERIFY_SHARDS)
    result = None
    for event in runner.run(small, amx_on=amx_on, run_id="verify", verbose_isa=True):
        if event.result is not None:
            result = event.result
    return result




@dataclasses.dataclass
class Leg:
    """One side of a race: a label, a spec, and the AMX state to run it with.

    Everything that differs between two legs lives here, so it is obvious at a
    glance what the race is actually varying.
    """

    key: str          # directory/container suffix, e.g. "full" or "amx-on"
    label: str        # what a human sees
    spec: RunSpec
    amx_on: bool


def _run_leg(
    runner: DeepVariantRunner,
    leg: Leg,
    run_id: str,
    quiet: bool,
) -> tuple[list[str], RunResult | None]:
    print(f"\n=== {leg.label} — ISA {runner.cfg.isa_for(leg.amx_on)}"
          f"{', cores ' + leg.spec.cpuset if leg.spec.cpuset else ''} ===", flush=True)

    log: list[str] = []
    result: RunResult | None = None
    last_pct = -1.0
    last_print = 0.0

    for event in runner.run(
        leg.spec, amx_on=leg.amx_on, run_id=run_id, verbose_isa=False, leg_id=leg.key
    ):
        if event.kind == "log" and event.line:
            log.append(event.line)
            if not quiet:
                print(f"  {event.line}", flush=True)
        elif event.kind == "heartbeat" and quiet:
            # make_examples emits no percentage at all and is by far the longest
            # stage, so a percentage-only heartbeat goes silent for over an hour
            # on a whole genome and looks like a hang. Report the stage instead
            # of inventing a number for it.
            pct = event.overall_percent or 0.0
            now = time.time()
            if pct - last_pct >= 10:
                last_pct = pct
                last_print = now
                print(f"  ... {pct:.0f}%  ({event.elapsed_s:.0f}s)", flush=True)
            elif now - last_print >= PROGRESS_EVERY_S:
                last_print = now
                stage = _active_stage(event.stages) or "working"
                print(f"  ... {stage}  ({event.elapsed_s:.0f}s)", flush=True)
        if event.result is not None:
            result = event.result

    if result is None:
        print(f"  {leg.label}: produced no result", file=sys.stderr)
        return log, None

    if result.error:
        print(f"  {leg.label}: FAILED — {result.error}", file=sys.stderr)
    else:
        print(f"  {leg.label}: {result.wall_clock_s:.1f}s", flush=True)
    print(f"  ISA reported: {result.reported_isa or 'not reported'}", flush=True)
    return log, result


def build_legs(cfg: Config, spec: RunSpec, mode: str) -> tuple[Leg, Leg]:
    """Return (candidate, baseline) for the requested race mode.

    Exactly one attribute differs between the two legs. That is the whole
    point: anything else varying would invalidate the comparison, and the
    fingerprint check in `time_ratio` would then refuse to print a number.
    """
    if mode == "amx":
        engine = cfg.amx_engine
        if engine is None:
            raise SystemExit(
                "No engine in config.yaml declares an AMX mechanism, so an AMX "
                "race would compare two identical fp32 runs and report noise as "
                "a result. See docs/AMX-FINDINGS.md."
            )
        # Swap the whole race onto the AMX-capable build. Both legs move
        # together, so the only thing that differs remains the AMX state.
        amx_spec = dataclasses.replace(
            spec,
            engine_image=engine.image,
            amx_extra_args=engine.amx_extra_args,
        )
        return (
            Leg("amx-on", cfg.amx_label(True), amx_spec, True),
            Leg("amx-off", cfg.amx_label(False), amx_spec, False),
        )

    scaling = cfg.scaling
    cpuset = scaling.get("baseline_cpuset")
    if not cpuset:
        raise SystemExit(
            "scaling.baseline_cpuset is not set in config.yaml — cannot run a "
            "core-scaling race without knowing what the reduced core budget is."
        )
    # Both legs keep the same AMX setting. Whatever it is, it is held constant,
    # so it cannot contaminate the core-scaling result either way.
    amx_on = True
    baseline = dataclasses.replace(spec, cpuset=cpuset)
    return (
        Leg("full", scaling.get("full_label", "all cores"), spec, amx_on),
        Leg("limited", scaling.get("baseline_label", f"cores {cpuset}"), baseline, amx_on),
    )


def dispatch_objection(mode: str, legs, dispatch: dict) -> str | None:
    """Why this race's speedup must not be shown, or None if it may be.

    An AMX race is only a measurement of AMX if the tiles actually ran
    something. When verification says they did not, the two legs are the same
    fp32 work twice over and the ratio is measuring run-to-run variance. The
    honest output there is no number at all -- not a small number.

    Verification is skippable (--skip-verify), and an unverified race is not a
    disproven one, so absence of evidence returns None and the caller prints
    the "not verified this run" label it already has.
    """
    if mode != "amx":
        return None
    for leg in legs:
        if not leg.amx_on:
            continue
        res = dispatch.get(leg.key)
        if res is None or not res.compute_primitives:
            continue
        if not res.amx_primitives:
            return (
                f"the AMX-on leg ran {res.compute_primitives} compute kernels and "
                "not one of them used AMX, so there is no AMX effect to measure"
            )
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m app.race",
        description="Race DeepVariant two ways, honestly.",
    )
    parser.add_argument("--sample", default="chr20", help="sample id from config.yaml")
    parser.add_argument(
        "--mode",
        default="scaling",
        choices=("scaling", "amx"),
        help="scaling: core budget (booth headline). amx: AMX on vs off (evidence).",
    )
    parser.add_argument("--record", metavar="NAME", help="save a replay trace under traces/")
    parser.add_argument("--quiet", action="store_true", help="progress only, not every log line")
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="skip the kernel-dispatch verification pass (faster, less evidence)",
    )
    args = parser.parse_args(argv)

    cfg = Config.load()
    runner = DeepVariantRunner(cfg)
    spec = runner.build_spec(args.sample)
    candidate, baseline = build_legs(cfg, spec, args.mode)

    print(f"Mode       : {args.mode}")
    print(f"Sample     : {args.sample}")
    print(f"BAM        : {spec.bam}")
    print(f"Reference  : {spec.reference}")
    print(f"Regions    : {spec.regions or 'whole genome'}")
    print(f"Shards     : {spec.num_shards}")
    print(f"Image      : {candidate.spec.engine_image}")
    print(f"Fingerprint: {candidate.spec.fingerprint_digest()}")
    print(f"Comparing  : {candidate.label}  vs  {baseline.label}")

    run_id = f"race-{args.mode}-{args.sample}"

    # ---- Phase 1: prove what the hardware really did -----------------------
    # Separate from the timed run on purpose: ONEDNN_VERBOSE prints a line per
    # primitive execution (270 MB on chr20), and bf16 emits more primitives
    # than fp32, so leaving it on during timing would skew one leg specifically.
    dispatch: dict[str, RunResult | None] = {}
    if not args.skip_verify:
        print("\n" + "=" * 70)
        print("PHASE 1 — verifying which kernels oneDNN really dispatches")
        print(f"(small slice: {VERIFY_REGION}, verbose logging on, NOT timed)")
        print("=" * 70)
        for leg in (candidate, baseline):
            res = _verify_dispatch(runner, leg.spec, leg.amx_on)
            dispatch[leg.key] = res
            print(f"{leg.label}: {res.isa_impl_summary if res else 'no result'}")

        for leg in (candidate, baseline):
            res = dispatch.get(leg.key)
            if res and leg.amx_on and res.compute_primitives and not res.amx_primitives:
                print(
                    f"\n  NOTE: {leg.label} permitted AMX but NO kernel used it.\n"
                    "  Permitting AMX is not the same as using it: the ISA setting\n"
                    "  only lifts a ceiling. On an fp32 graph there is no AMX path,\n"
                    "  so the tiles stay idle. Any time difference measured from here\n"
                    "  is noise, and the verdict below will refuse to report it.\n"
                    "  See docs/AMX-FINDINGS.md."
                )
            if res and not leg.amx_on and res.amx_primitives:
                print(f"\n  WARNING: {leg.label} used AMX kernels. It is not a baseline.")

    # ---- Phase 2: measure, with instrumentation off ------------------------
    print("\n" + "=" * 70)
    print("PHASE 2 — timed run (oneDNN verbose OFF so logging cannot skew it)")
    print("=" * 70)

    # Baseline (the slower leg) first: the reveal lands better, and a cold page
    # cache then penalises the baseline rather than the candidate.
    base_log, base = _run_leg(runner, baseline, run_id, args.quiet)
    cand_log, cand = _run_leg(runner, candidate, run_id, args.quiet)

    print("\n" + "=" * 70)
    if cand is None or base is None:
        print("No verdict: a leg produced no result.")
        return 1

    for leg, result in ((candidate, cand), (baseline, base)):
        verified = dispatch.get(leg.key)
        evidence = verified.isa_impl_summary if verified else "not verified this run"
        print(f"{leg.label:<20} requested {result.requested_isa:<16} | {evidence}")

    ratio = time_ratio(cand, base)
    objection = dispatch_objection(args.mode, (candidate, baseline), dispatch)
    if objection:
        print(f"\nNo speedup shown: {objection}.")
        print(f"  For the record — {baseline.label}: {base.wall_clock_s:.1f}s, "
              f"{candidate.label}: {cand.wall_clock_s:.1f}s.")
        print("  Two timings of identical work, shown as timings and not as a ratio.")
        return 1
    if ratio is None:
        print("\nNo speedup shown. The comparison was not valid:")
        if cand.error or base.error:
            print("  - a leg failed")
        if cand.fingerprint != base.fingerprint:
            print("  - the two legs did not run identical parameters")
        return 1

    print(f"\n{baseline.label:<20}: {base.wall_clock_s:>8.1f}s")
    print(f"{candidate.label:<20}: {cand.wall_clock_s:>8.1f}s")
    print(f"{'Speedup':<20}: {ratio:>8.2f}x  (same binary, same data, same shards)")

    caveat = getattr(cfg.amx_engine, "amx_caveat", None) if args.mode == "amx" else None
    if caveat:
        print(f"\nCAVEAT: {caveat}")

    if cand.variant_counts and base.variant_counts:
        print(f"\nVariants: {candidate.label} {cand.variant_counts.total:,} · "
              f"{baseline.label} {base.variant_counts.total:,}")
        if cand.variant_counts.total != base.variant_counts.total:
            print("  NOTE: the two legs called different numbers of variants.")

    if args.record:
        path = record_trace(
            cfg, args.record, args.sample,
            {candidate.key: (cand_log, cand), baseline.key: (base_log, base)},
        )
        print(f"\nTrace recorded: {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
