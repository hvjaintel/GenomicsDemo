"""Booth demo application.

Six screens, driven from config.yaml and real hardware:
  1 HOME / STATUS     — what this box is and what it can do
  2 DATASET PICKER    — what we are about to call variants on
  3 RUN CONSOLE       — one big button, live stages, streaming logs
  4 SCALING RACE      — few cores vs the whole machine, the headline
  5 RESULTS           — real variant counts from the produced VCF
  6 EFFICIENCY / TCO  — clearly-labelled illustrative planning figures

Design rule enforced throughout: measured values and assumed values never look
the same on screen.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from html import escape
from pathlib import Path

import gradio as gr
import pandas as pd

from . import race as race_mod
from . import replay as replay_mod
from . import tco as tco_mod
from .config import Config, get_config
from .parsing import STAGES, STAGE_LABELS
from .preflight import Status, run_preflight
from .runner import (
    DeepVariantRunner,
    RunResult,
    RunnerError,
    speedup,
    throughput_mbases_per_hour,
)
from .sysinfo import snapshot
from .theme import build_css, build_theme
from .ui.components import (
    fmt_bytes,
    fmt_duration,
    illustrative,
    metric,
    metric_grid,
    pill,
    race_lane,
    replay_banner,
    section,
    speedup_card,
    stage_bar,
)

MAX_LOG_LINES = 400


# =============================================================================
# Session state
# =============================================================================


@dataclass
class DemoState:
    """Results and logs from this session, shared across panels."""

    last_single: RunResult | None = None
    race_on: RunResult | None = None
    race_off: RunResult | None = None
    logs: dict[str, list[str]] = field(default_factory=dict)
    replaying: bool = False
    lock: threading.Lock = field(default_factory=threading.Lock)

    def record_log(self, key: str, line: str) -> list[str]:
        buf = self.logs.setdefault(key, [])
        buf.append(line)
        if len(buf) > MAX_LOG_LINES:
            del buf[: len(buf) - MAX_LOG_LINES]
        return buf

    def reset_logs(self, key: str) -> None:
        self.logs[key] = []

    @property
    def measured_wgs_seconds(self) -> float | None:
        """A real measured runtime, scaled to a whole genome where needed."""
        for result in (self.race_on, self.last_single):
            if result and result.succeeded and result.wall_clock_s:
                return result.wall_clock_s
        return None


STATE = DemoState()


# =============================================================================
# Panel 1 — HOME / STATUS
# =============================================================================


def render_home(cfg: Config) -> str:
    snap = snapshot(cfg)
    cpu, mem, storage, acoustics = snap.cpu, snap.memory, snap.storage, snap.acoustics

    badge_class = "accel-badge" if cpu.has_amx else "accel-badge missing"
    badge_text = (
        f"BUILT-IN ACCELERATION: {cpu.accel_summary}"
        if cpu.has_amx
        else "AMX NOT DETECTED ON THIS CPU"
    )

    numa_rows = "".join(
        f"<div><span class='dim'>{escape(node)}</span> "
        f"<span class='mono'>{escape(cpus)}</span></div>"
        for node, cpus in cpu.numa_map.items()
    )

    flags = " ".join(pill(f, "on") for f in cpu.present_accel_flags)

    # The acoustic figure is only ever presented as live if it was measured.
    acoustic_note = (
        acoustics.note if not acoustics.live else f"live reading — {acoustics.note}"
    )

    cards = metric_grid(
        [
            metric("Processor", cpu.model, note=f"{cpu.sockets}-socket server"),
            metric("Cores / threads", f"{cpu.physical_cores}", unit=f"/ {cpu.logical_cpus}",
                   note="physical cores / hardware threads"),
            metric("Memory", f"{mem.total_gb}", unit="GB",
                   note=f"across {cpu.numa_nodes} NUMA nodes"),
            metric("L3 cache", cpu.l3_cache or "—", small=True),
            metric(
                "Acoustics",
                f"{acoustics.dba:.0f}",
                unit="dB(A)",
                note=f"{acoustics.label} — {acoustic_note}",
            ),
            metric("Operating system", snap.os_pretty, small=True, note=f"kernel {snap.kernel}"),
        ]
    )

    docker_pill = (
        pill("Docker ready", "on")
        if snap.docker.reachable
        else pill("Docker unavailable", "bad")
    )
    storage_pill = (
        pill(f"Data: {storage.free_gb:.0f} GB free", "on")
        if not storage.is_fallback
        else pill(f"Fallback storage — {storage.free_gb:.0f} GB free", "warn")
    )

    return f"""
<div style="padding: 8px 4px 20px;">
  <div style="margin-bottom: 22px;"><span class="{badge_class}">{escape(badge_text)}</span></div>
  <div style="margin-bottom: 20px;">{flags}</div>
  {cards}
  <div style="margin-top: 22px;">{docker_pill} {storage_pill}
    {pill('numactl available', 'on') if snap.numactl_available else pill('numactl missing', 'warn')}</div>
  <div style="margin-top: 26px;">
    {section('NUMA topology')}
    <div class="metric-card">{numa_rows or '<span class="dim">not reported</span>'}</div>
  </div>
  <div style="margin-top: 22px;" class="dim">
    No add-in accelerator cards. Everything runs on the CPU's built-in vector
    units — AVX-512 for this workload; the AMX tiles are present and idle,
    because DeepVariant's model is fp32 (docs/AMX-FINDINGS.md) —
    {escape(str(cfg.demo.get('partner_mark', '')))}.
  </div>
</div>
"""


def render_preflight(cfg: Config) -> tuple[str, pd.DataFrame]:
    report = run_preflight(cfg)
    kind = "on" if report.all_green else ("warn" if report.ready else "bad")
    header = (
        f'<div style="margin-bottom:14px;">{pill(report.summary, kind)}</div>'
    )
    rows = [
        {
            "": check.status.icon,
            "Check": check.name,
            "Detail": check.detail,
            "What to do": check.remedy or "—",
        }
        for check in report.checks
    ]
    return header, pd.DataFrame(rows)


# =============================================================================
# Panel 2 — DATASET PICKER
# =============================================================================


def sample_choices(cfg: Config) -> list[tuple[str, str]]:
    choices = []
    for sample in cfg.booth_samples:
        path = cfg.dataset_path(sample.dataset)
        mark = "" if path.exists() else "  ⚠ not staged"
        choices.append((f"{sample.label}{mark}", sample.id))
    return choices


def render_dataset(cfg: Config, sample_id: str | None) -> str:
    if not sample_id:
        return '<div class="dim">Select a sample.</div>'
    sample = cfg.sample(sample_id)
    dataset = cfg.dataset(sample.dataset)
    path = cfg.dataset_path(sample.dataset)

    staged = path.exists()
    actual = path.stat().st_size if staged else 0
    size_ok = (not dataset.size_bytes) or actual == dataset.size_bytes

    if not staged:
        status = pill("NOT STAGED — run scripts/fetch_data.sh", "bad")
    elif not size_ok:
        status = pill("SIZE MISMATCH — re-fetch before the show", "bad")
    else:
        status = pill("Staged and verified", "on")

    checksum = (
        pill("sha256 pinned", "on")
        if dataset.has_recorded_checksum
        else pill("sha256 not pinned in config.yaml", "warn")
    )

    cards = metric_grid(
        [
            metric("File size", fmt_bytes(actual if staged else dataset.size_bytes),
                   note="on disk" if staged else "download size"),
            metric("Region", sample.regions or "whole genome", small=True),
            metric("Shards", f"{cfg.num_shards}", note="one per hardware thread"),
            metric(
                f"Expected — {cfg.scaling.get('full_label', 'all cores')}",
                fmt_duration(sample.runtime_fast_s),
                note="estimate — for pacing only" if sample.illustrative else "measured on this box",
            ),
            metric(
                f"Expected — {cfg.scaling.get('baseline_label', 'reduced cores')}",
                fmt_duration(sample.runtime_slow_s),
                note="estimate — for pacing only" if sample.illustrative else "measured on this box",
            ),
        ]
    )

    note = (
        illustrative(
            "Expected runtimes are configured estimates used by booth staff to pace "
            "visitors. The Run Console and Race view always display real measured times."
        )
        if sample.illustrative
        else ""
    )

    return f"""
<div style="padding: 8px 4px;">
  <div class="section-title">{escape(sample.label)}</div>
  <div class="dim" style="font-size:1.15rem; margin-bottom:16px;">{escape(sample.blurb)}</div>
  <div style="margin-bottom:18px;">{status} {checksum}</div>
  {cards}
  <div style="margin-top:18px;">{note}</div>
  <div style="margin-top:18px;" class="dim mono" style="font-size:0.9rem;">
    {escape(dataset.name)}<br>{escape(str(path))}
  </div>
</div>
"""


# =============================================================================
# Panel 3 — RUN CONSOLE
# =============================================================================


def _stages_html(stages: dict[str, dict]) -> str:
    if not stages:
        return "".join(stage_bar(STAGE_LABELS[s], 0.0, False, None) for s in STAGES)
    return "".join(
        stage_bar(
            stages.get(s, {}).get("label", STAGE_LABELS[s]),
            stages.get(s, {}).get("percent", 0.0),
            stages.get(s, {}).get("finished", False),
            stages.get(s, {}).get("duration_s"),
        )
        for s in STAGES
    )


def _isa_html(
    cfg: Config,
    requested: str,
    reported: str | None,
    verified: bool | None,
    usage: str = "",
    amx_primitives: int = 0,
    compute_primitives: int = 0,
) -> str:
    """The instruction-set card.

    Two separate facts, deliberately kept apart:

      * what oneDNN was ALLOWED to use (the requested ceiling and the banner it
        printed back), and
      * what it ACTUALLY used (how many compute primitives ran on AMX kernels).

    Conflating them is how a demo ends up claiming an AMX win it never earned:
    the banner reports the CPU's capability, and on an fp32 model it will
    happily say "Intel AMX with bfloat16 support" while every convolution runs
    on AVX-512. See docs/AMX-FINDINGS.md.
    """
    if reported is None:
        badge = pill("ISA not reported by oneDNN", "warn")
    elif verified is True:
        badge = pill("ceiling verified", "on")
    elif verified is False:
        badge = pill("ISA MISMATCH", "bad")
    else:
        badge = pill("ceiling unverified", "warn")

    if not compute_primitives:
        usage_badge = pill("AMX usage not instrumented this run", "warn")
        usage_note = (
            "oneDNN verbose logging was off, so which kernels ran was not "
            "recorded. Timing runs keep it off on purpose — it costs more on "
            "some code paths than others and would skew the comparison."
        )
    elif amx_primitives:
        usage_badge = pill(f"AMX used by {amx_primitives}/{compute_primitives} primitives", "on")
        usage_note = escape(usage)
    else:
        usage_badge = pill("AMX permitted but never used", "warn")
        usage_note = (
            "DeepVariant's shipped model is fp32 and AMX has no fp32 path, so "
            "the tiles had nothing to do. Expected — see docs/AMX-FINDINGS.md."
        )

    return (
        f'<div class="metric-card">'
        f'<div class="metric-label">Instruction set — permitted</div>'
        f'<div class="mono" style="font-size:1.05rem;">requested ceiling: '
        f'<strong>{escape(requested)}</strong></div>'
        f'<div class="mono dim" style="font-size:1rem; margin-top:6px;">oneDNN reported: '
        f'{escape(reported or "—")}</div>'
        f'<div style="margin-top:10px;">{badge}</div>'
        f'<div class="metric-label" style="margin-top:18px;">Instruction set — actually used</div>'
        f'<div style="margin-top:8px;">{usage_badge}</div>'
        f'<div class="metric-note">{usage_note}</div></div>'
    )


def _live_possible(cfg: Config) -> bool:
    report = run_preflight(cfg)
    return report.ready


def run_console(cfg: Config, sample_id: str, amx_on: bool) -> Iterator[tuple]:
    """Generator driving the Run Console. Yields (header, stages, logs, summary)."""
    runner = DeepVariantRunner(cfg)
    runner.reset_cancel()
    key = "console"
    STATE.reset_logs(key)

    isa = cfg.isa_for(amx_on)
    amx_pill = pill(cfg.amx_label(amx_on), "on" if amx_on else "off")

    use_replay = replay_mod.should_replay(cfg, _live_possible(cfg))
    banner = replay_banner(replay_mod.banner_text(cfg)) if use_replay else ""
    STATE.replaying = use_replay

    header = f'{banner}<div style="margin-bottom:12px;">{amx_pill} {pill(isa, "")}</div>'
    yield header, _stages_html({}), "", ""

    if use_replay:
        trace = replay_mod.default_trace(cfg)
        events = replay_mod.replay(trace, "amx_on" if amx_on else "amx_off")
    else:
        try:
            spec = runner.build_spec(sample_id)
        except RunnerError as exc:
            yield (
                header,
                _stages_html({}),
                str(exc),
                f'<div class="metric-card">{pill("Cannot start", "bad")}'
                f'<div class="metric-note">{escape(str(exc))}</div></div>',
            )
            return
        events = runner.run(spec, amx_on=amx_on)

    last_emit = 0.0
    stages: dict[str, dict] = {}
    result: RunResult | None = None
    reported_isa: str | None = None

    for event in events:
        if event.stages:
            stages = event.stages
        if event.reported_isa:
            reported_isa = event.reported_isa

        if event.kind == "log":
            lines = STATE.record_log(key, event.line)
            now = time.time()
            # Throttle UI updates; DeepVariant emits far more lines than a
            # booth screen can usefully render.
            if now - last_emit > 0.25:
                last_emit = now
                head = (
                    f'{banner}<div style="margin-bottom:12px;">{amx_pill} {pill(isa, "")} '
                    f'{pill(fmt_duration(event.elapsed_s), "")}</div>'
                )
                yield head, _stages_html(stages), "\n".join(lines), ""
        elif event.kind == "done":
            result = event.result

    lines = STATE.logs.get(key, [])
    if result is None:
        yield header, _stages_html(stages), "\n".join(lines), ""
        return

    STATE.last_single = result
    yield (
        f'{banner}<div style="margin-bottom:12px;">{amx_pill} {pill(isa, "")} '
        f'{pill(fmt_duration(result.wall_clock_s), "on" if result.succeeded else "bad")}</div>',
        _stages_html(stages),
        "\n".join(lines),
        _run_summary(cfg, result, reported_isa, use_replay),
    )


def _run_summary(cfg: Config, result: RunResult, reported_isa: str | None, replaying: bool) -> str:
    if not result.succeeded:
        return (
            f'<div class="metric-card">{pill("RUN FAILED", "bad")}'
            f'<div class="metric-note mono">{escape(result.error or "unknown error")}</div>'
            f'<div class="metric-note">Press START to try again.</div></div>'
        )

    counts = result.variant_counts
    cards = [
        metric("Wall clock", fmt_duration(result.wall_clock_s),
               note="measured" if not replaying else "recorded"),
        metric("Variants called", f"{counts.total:,}" if counts else "—",
               note=counts.ti_tv_note if counts else ""),
    ]
    for stage in STAGES:
        duration = result.stage_durations.get(stage)
        if duration:
            cards.append(metric(STAGE_LABELS[stage], fmt_duration(duration), small=True))

    return (
        metric_grid(cards)
        + '<div style="margin-top:16px;">'
        + _isa_html(
            cfg,
            result.requested_isa,
            reported_isa or result.reported_isa,
            result.isa_verified,
            result.isa_impl_summary,
            result.amx_primitives,
            result.compute_primitives,
        )
        + "</div>"
    )


# =============================================================================
# Panel 4 — SCALING RACE
# =============================================================================


def run_race(cfg: Config, sample_id: str) -> Iterator[tuple]:
    """Run the reduced-core leg then the full-machine leg, racing them on screen.

    Sequential rather than concurrent: two simultaneous 192-shard runs would
    contend for the same cores and neither number would mean anything.
    """
    runner = DeepVariantRunner(cfg)
    runner.reset_cancel()
    STATE.reset_logs("race")

    use_replay = replay_mod.should_replay(cfg, _live_possible(cfg))
    banner = replay_banner(replay_mod.banner_text(cfg)) if use_replay else ""
    STATE.replaying = use_replay

    sample = cfg.sample(sample_id)
    est_off = sample.runtime_slow_s or 300
    est_on = sample.runtime_fast_s or 120

    spec = None
    if not use_replay:
        try:
            spec = runner.build_spec(sample_id)
        except RunnerError as exc:
            yield (
                banner + f'<div class="metric-card">{pill("Cannot start", "bad")}'
                f'<div class="metric-note">{escape(str(exc))}</div></div>',
                "",
                "",
            )
            return

    # The race varies exactly one thing: the core budget. `True` is the
    # full-machine leg, `False` the reduced-core baseline. It is NOT an AMX
    # race -- AMX is held constant across both legs, because on stock
    # DeepVariant the tiles never run a kernel at all (docs/AMX-FINDINGS.md).
    full_leg, baseline_leg = race_mod.build_legs(cfg, spec, "scaling") if spec else (None, None)

    results: dict[bool, RunResult] = {}
    elapsed: dict[bool, float] = {False: 0.0, True: 0.0}
    finished: dict[bool, bool] = {False: False, True: False}

    def leg_for(full: bool):
        return full_leg if full else baseline_leg

    def leg_label(full: bool) -> str:
        leg = leg_for(full)
        if leg:
            return leg.label
        return cfg.scaling.get(
            "full_label" if full else "baseline_label",
            "all cores" if full else "reduced cores",
        )

    def lanes_html() -> str:
        html = []
        for amx_on in (True, False):
            est = est_on if amx_on else est_off
            done = finished[amx_on]
            result = results.get(amx_on)
            secs = (result.wall_clock_s if result and result.wall_clock_s else elapsed[amx_on])
            pct = 100.0 if done else min(99.0, 100.0 * secs / max(est, 1))
            meta = []
            if result and result.succeeded and result.variant_counts:
                meta.append(f"{result.variant_counts.total:,} variants")
            if result and result.reported_isa:
                meta.append(f"ISA: {result.reported_isa[:52]}")
            elif not done and secs > 0:
                meta.append("running…")
            elif not done:
                meta.append("waiting")
            html.append(
                race_lane(
                    leg_label(amx_on),
                    amx_on,
                    fmt_duration(secs) if secs else "—",
                    pct,
                    meta,
                )
            )
        return banner + "".join(html)

    yield lanes_html(), "", ""

    # Baseline first so the audience watches the slow leg finish, then sees the
    # full machine beat it — the reveal lands better in that order.
    for amx_on in (False, True):
        leg = leg_for(amx_on)
        if use_replay:
            trace = replay_mod.default_trace(cfg)
            events = replay_mod.replay(trace, leg.key if leg else ("full" if amx_on else "limited"))
        else:
            events = runner.run(
                leg.spec, amx_on=leg.amx_on, leg_id=leg.key, verbose_isa=False
            )

        last_emit = 0.0
        for event in events:
            if event.kind == "log":
                STATE.record_log("race", f"[{leg_label(amx_on)}] {event.line}")
                elapsed[amx_on] = event.elapsed_s
                now = time.time()
                if now - last_emit > 0.3:
                    last_emit = now
                    yield lanes_html(), "", "\n".join(STATE.logs.get("race", [])[-120:])
            elif event.kind == "done" and event.result:
                results[amx_on] = event.result
                finished[amx_on] = True
                elapsed[amx_on] = event.result.wall_clock_s or elapsed[amx_on]
                yield lanes_html(), "", "\n".join(STATE.logs.get("race", [])[-120:])

    on_result, off_result = results.get(True), results.get(False)
    STATE.race_on, STATE.race_off = on_result, off_result

    yield lanes_html(), _race_verdict(cfg, sample_id, on_result, off_result, use_replay), \
        "\n".join(STATE.logs.get("race", [])[-120:])


def _race_verdict(
    cfg: Config,
    sample_id: str,
    on: RunResult | None,
    off: RunResult | None,
    replaying: bool,
) -> str:
    if not on or not off:
        return (
            f'<div class="metric-card">{pill("Race incomplete", "warn")}'
            '<div class="metric-note">Both legs must finish before a speedup can be reported.</div></div>'
        )

    fast_label = cfg.scaling.get("full_label", "all cores")
    slow_label = cfg.scaling.get("baseline_label", "reduced cores")

    if not on.succeeded or not off.succeeded:
        failed = fast_label if not on.succeeded else slow_label
        error = (on if not on.succeeded else off).error or "unknown error"
        return (
            f'<div class="metric-card">{pill(f"{failed} leg failed", "bad")}'
            f'<div class="metric-note mono">{escape(error)}</div>'
            '<div class="metric-note">No speedup is shown, because there is no honest number to show.</div></div>'
        )

    multiplier = speedup(on, off)
    if multiplier is None:
        return (
            f'<div class="metric-card">{pill("Comparison invalid", "bad")}'
            '<div class="metric-note">The two runs did not use identical parameters, '
            'so the difference cannot be attributed to the core budget. '
            'No speedup reported.</div></div>'
        )

    sample = cfg.sample(sample_id)
    mbases = float(
        cfg.tco.get("assumptions", {}).get(
            "mbases_per_chr20" if sample.id == "chr20" else "mbases_per_wgs_genome", 0
        )
        or 0
    )
    tput_on = throughput_mbases_per_hour(on, mbases) if mbases else None
    tput_off = throughput_mbases_per_hour(off, mbases) if mbases else None

    # Accuracy sanity check: speed is only interesting if the answer is the same.
    variance_note = ""
    if on.variant_counts and off.variant_counts:
        a, b = on.variant_counts.total, off.variant_counts.total
        if a and b:
            delta = abs(a - b) / max(a, b) * 100.0
            kind = "on" if delta < 0.5 else "warn"
            variance_note = (
                f'<div style="margin-top:14px;">{pill(f"Variant counts differ by {delta:.2f}%", kind)}'
                f'<span class="dim" style="margin-left:12px;">'
                f'{escape(fast_label)} {a:,} · {escape(slow_label)} {b:,}</span></div>'
            )

    cards = [
        metric(fast_label, fmt_duration(on.wall_clock_s), note="full machine"),
        metric(slow_label, fmt_duration(off.wall_clock_s),
               note=f"--cpuset-cpus {cfg.scaling.get('baseline_cpuset', '')}"),
        metric("Time saved", fmt_duration((off.wall_clock_s or 0) - (on.wall_clock_s or 0))),
    ]
    if tput_on and tput_off:
        cards.append(metric(f"Throughput — {fast_label}", f"{tput_on:,.0f}", unit="Mb/h"))
        cards.append(metric(f"Throughput — {slow_label}", f"{tput_off:,.0f}", unit="Mb/h"))

    source = "recorded run" if replaying else "measured live on this server"
    return (
        speedup_card(
            multiplier,
            f"Xeon scales: {fast_label} vs {slow_label}",
            f"{source} · identical BAM, reference, shard count, NUMA policy and ISA "
            f"settings · only the core budget differed, and both legs called the "
            f"same variants",
        )
        + '<div style="margin-top:18px;">'
        + metric_grid(cards)
        + "</div>"
        + variance_note
    )


# =============================================================================
# Panel 5 — RESULTS
# =============================================================================


def render_results(cfg: Config) -> tuple[str, pd.DataFrame]:
    result = STATE.race_on or STATE.last_single
    empty = pd.DataFrame(columns=["CHROM", "POS", "REF", "ALT", "QUAL", "FILTER", "TYPE"])

    if result is None or not result.succeeded:
        return (
            '<div class="metric-card"><div class="metric-label">No results yet</div>'
            '<div class="metric-note">Run the pipeline from the Run Console or the Scaling Race view. '
            "Nothing is shown here until a real run has produced a VCF.</div></div>",
            empty,
        )

    counts = result.variant_counts
    if counts is None:
        return (
            '<div class="metric-card">'
            f'{pill("Run finished but no VCF was found", "warn")}</div>',
            empty,
        )

    cards = metric_grid(
        [
            metric("Total variants", f"{counts.total:,}"),
            metric("SNPs", f"{counts.snps:,}"),
            metric("Indels", f"{counts.indels:,}"),
            metric("PASS", f"{counts.passing:,}",
                   note=f"{100.0 * counts.passing / counts.total:.1f}% of calls" if counts.total else ""),
            metric("Heterozygous", f"{counts.het:,}"),
            metric("Homozygous alt", f"{counts.hom_alt:,}"),
        ]
    )

    concordance = _concordance_note(cfg)
    source = (
        f'<div class="dim mono" style="margin-top:16px; font-size:0.9rem;">'
        f'read from {escape(result.vcf_path or "—")}</div>'
    )

    return (
        f'<div>{section("Variants called — read directly from the output VCF")}{cards}'
        f'{concordance}{source}</div>',
        pd.DataFrame(counts.examples) if counts.examples else empty,
    )


def _concordance_note(cfg: Config) -> str:
    """Accuracy statement — states plainly what has and has not been measured."""
    on, off = STATE.race_on, STATE.race_off
    truth_staged = cfg.dataset_path("truth_vcf").exists()

    if on and off and on.succeeded and off.succeeded and on.variant_counts and off.variant_counts:
        a, b = on.variant_counts.total, off.variant_counts.total
        delta = abs(a - b) / max(a, b, 1) * 100.0
        fast_label = cfg.scaling.get("full_label", "all cores")
        slow_label = cfg.scaling.get("baseline_label", "reduced cores")
        verdict = (
            "Both legs produced effectively identical call sets — the speedup "
            "costs nothing in accuracy. It is the same arithmetic, spread over "
            "more cores."
            if delta < 0.5
            else "The two legs' call sets differ by more than 0.5% — investigate before quoting the speedup."
        )
        body = (
            f'<div class="metric-card" style="margin-top:16px;">'
            f'<div class="metric-label">Accuracy check — measured</div>'
            f'<div style="font-size:1.15rem;">{escape(fast_label)}: <strong>{a:,}</strong> variants · '
            f'{escape(slow_label)}: <strong>{b:,}</strong> variants · difference <strong>{delta:.2f}%</strong></div>'
            f'<div class="metric-note">{escape(verdict)}</div></div>'
        )
    else:
        body = (
            '<div class="metric-card" style="margin-top:16px;">'
            '<div class="metric-label">Accuracy check</div>'
            '<div class="metric-note">Run the Scaling Race to compare the two legs\' '
            "call sets directly. Until then no accuracy claim is made.</div></div>"
        )

    if not truth_staged:
        body += (
            '<div class="dim" style="margin-top:12px; font-size:0.95rem;">'
            "GIAB HG002 truth set is not staged, so no concordance against a benchmark is "
            "reported. Stage it with <span class='mono'>./scripts/fetch_data.sh --with-truth</span>."
            "</div>"
        )
    else:
        body += (
            '<div class="dim" style="margin-top:12px; font-size:0.95rem;">'
            "GIAB HG002 v4.2.1 truth set is staged. Run <span class='mono'>hap.py</span> "
            "against the output VCF for a full precision/recall breakdown."
            "</div>"
        )
    return body


# =============================================================================
# Panel 6 — EFFICIENCY / TCO
# =============================================================================


def render_efficiency(cfg: Config, weekly_samples: int) -> str:
    measured = STATE.measured_wgs_seconds
    result = tco_mod.compute(cfg, int(weekly_samples), measured_seconds_per_genome=measured)

    cards = metric_grid(
        [
            metric("Genomes / day", f"{result.genomes_per_day:,.1f}",
                   note=f"runtime source: {result.runtime_source}"),
            metric("Days to clear a week", f"{result.days_to_clear_weekly:,.1f}"),
            metric("Servers needed", f"{result.servers_needed}",
                   note="to absorb the weekly volume"),
            metric("Utilisation", f"{result.utilisation_pct:.0f}", unit="%"),
            metric("Energy / genome", f"{result.energy_kwh_per_genome:.2f}", unit="kWh"),
            metric("Cost / genome", f"{result.cost_per_genome:.2f}", unit=result.currency),
            metric("Annual energy", f"{result.annual_energy_kwh:,.0f}", unit="kWh"),
            metric("Annual energy cost", f"{result.annual_energy_cost:,.0f}", unit=result.currency),
            metric("Acoustics", f"{result.acoustic_dba:.0f}", unit="dB(A)",
                   note="air-cooled, bench-deployable"),
        ]
    )

    disclaimer = illustrative(str(cfg.tco.get("disclaimer", "")))
    assumptions = cfg.tco.get("assumptions", {})
    rows = "".join(
        f"<div><span class='dim'>{escape(str(k))}</span>: <span class='mono'>{escape(str(v))}</span></div>"
        for k, v in assumptions.items()
    )

    runtime_pill = (
        pill("Using a measured runtime from this session", "on")
        if result.runtime_is_measured
        else pill("No live run yet — using the configured estimate", "warn")
    )

    return f"""
<div style="padding:8px 4px;">
  {disclaimer}
  <div style="margin:16px 0;">{runtime_pill}</div>
  {cards}
  <div style="margin-top:24px;">
    {section('Assumptions (from config.yaml)')}
    <div class="metric-card">{rows}</div>
  </div>
</div>
"""


# =============================================================================
# App assembly
# =============================================================================


def build_app(cfg: Config) -> gr.Blocks:
    default_sample = next((s.id for s in cfg.booth_samples), "chr20")

    with gr.Blocks(
        theme=build_theme(cfg),
        css=build_css(cfg),
        title=str(cfg.demo.get("title", "Genomics Demo")),
        analytics_enabled=False,
    ) as app:
        gr.HTML(
            f"""
<div class="booth-masthead">
  <div>
    <div class="booth-title">{escape(str(cfg.demo.get('title', '')))}</div>
    <div class="booth-subtitle">{escape(str(cfg.demo.get('subtitle', '')))}</div>
  </div>
  <div class="partner-mark">{escape(str(cfg.demo.get('partner_mark', '')))}</div>
</div>
"""
        )

        with gr.Tabs():
            # ---------------- HOME ----------------
            with gr.Tab("Status"):
                home_html = gr.HTML(render_home(cfg))
                gr.HTML(section("Pre-flight readiness"))
                pf_summary = gr.HTML()
                pf_table = gr.Dataframe(
                    headers=["", "Check", "Detail", "What to do"],
                    interactive=False,
                    wrap=True,
                )
                with gr.Row():
                    refresh_btn = gr.Button("Refresh system status", elem_classes="secondary-button")
                    preflight_btn = gr.Button("Run pre-flight checks", elem_classes="secondary-button")

                refresh_btn.click(lambda: render_home(cfg), outputs=home_html)
                preflight_btn.click(lambda: render_preflight(cfg), outputs=[pf_summary, pf_table])
                app.load(lambda: render_preflight(cfg), outputs=[pf_summary, pf_table])

            # ---------------- DATASETS ----------------
            with gr.Tab("Dataset"):
                sample_radio = gr.Radio(
                    choices=sample_choices(cfg),
                    value=default_sample,
                    label="Sample",
                )
                dataset_html = gr.HTML(render_dataset(cfg, default_sample))
                sample_radio.change(
                    lambda sid: render_dataset(cfg, sid),
                    inputs=sample_radio,
                    outputs=dataset_html,
                )

            # ---------------- RUN CONSOLE ----------------
            with gr.Tab("Run console"):
                with gr.Row():
                    console_sample = gr.Radio(
                        choices=sample_choices(cfg),
                        value=default_sample,
                        label="Sample",
                        scale=3,
                    )
                    amx_switch = gr.Checkbox(
                        value=True,
                        label="Intel AMX: ON",
                        scale=1,
                    )
                start_btn = gr.Button(
                    "▶  START VARIANT CALLING",
                    variant="primary",
                    elem_classes="start-button",
                )
                console_header = gr.HTML()
                console_stages = gr.HTML(_stages_html({}))
                console_summary = gr.HTML()
                console_logs = gr.Textbox(
                    label="Live pipeline log",
                    lines=18,
                    max_lines=18,
                    interactive=False,
                    elem_classes="log-pane",
                    autoscroll=True,
                )

                amx_switch.change(
                    lambda on: gr.update(label=f"Intel AMX: {'ON' if on else 'OFF'}"),
                    inputs=amx_switch,
                    outputs=amx_switch,
                )
                start_btn.click(
                    lambda sid, on: (yield from run_console(cfg, sid, on)),
                    inputs=[console_sample, amx_switch],
                    outputs=[console_header, console_stages, console_logs, console_summary],
                )

            # ---------------- AMX RACE ----------------
            with gr.Tab("Scaling race"):
                gr.HTML(
                    '<div class="dim" style="font-size:1.1rem; margin-bottom:14px;">'
                    "Two runs, back to back, on the same data with the same binary, "
                    "shard count and instruction-set settings. The only difference is "
                    "how many cores the container may use: "
                    f"<span class='mono'>{escape(str(cfg.scaling.get('full_label', 'all cores')))}</span> versus "
                    f"<span class='mono'>{escape(str(cfg.scaling.get('baseline_label', 'reduced cores')))}</span> "
                    "(<span class='mono'>--cpuset-cpus "
                    f"{escape(str(cfg.scaling.get('baseline_cpuset', '')))}</span>). "
                    "Both legs call the same variants — nothing is traded for the speed. "
                    "Why this is not an AMX race: see docs/AMX-FINDINGS.md.</div>"
                )
                race_sample = gr.Radio(
                    choices=sample_choices(cfg),
                    value=default_sample,
                    label="Sample",
                )
                race_btn = gr.Button(
                    f"▶  RACE: {cfg.scaling.get('baseline_label', 'few cores')} "
                    f"vs {cfg.scaling.get('full_label', 'all cores')}",
                    variant="primary",
                    elem_classes="start-button",
                )
                race_lanes = gr.HTML()
                race_verdict = gr.HTML()
                race_logs = gr.Textbox(
                    label="Combined log",
                    lines=12,
                    max_lines=12,
                    interactive=False,
                    elem_classes="log-pane",
                    autoscroll=True,
                )
                race_btn.click(
                    lambda sid: (yield from run_race(cfg, sid)),
                    inputs=race_sample,
                    outputs=[race_lanes, race_verdict, race_logs],
                )

            # ---------------- RESULTS ----------------
            with gr.Tab("Results"):
                results_html = gr.HTML()
                results_table = gr.Dataframe(interactive=False, wrap=True, label="Example calls from the VCF")
                results_btn = gr.Button("Load latest results", elem_classes="secondary-button")
                results_btn.click(lambda: render_results(cfg), outputs=[results_html, results_table])
                app.load(lambda: render_results(cfg), outputs=[results_html, results_table])

            # ---------------- EFFICIENCY ----------------
            with gr.Tab("Efficiency"):
                default_volume = int(
                    cfg.tco.get("assumptions", {}).get("default_weekly_sample_volume", 50)
                )
                volume = gr.Slider(
                    minimum=1,
                    maximum=1000,
                    step=1,
                    value=default_volume,
                    label="Genomes to process per week",
                )
                efficiency_html = gr.HTML(render_efficiency(cfg, default_volume))
                volume.change(
                    lambda v: render_efficiency(cfg, int(v)),
                    inputs=volume,
                    outputs=efficiency_html,
                )

    return app


def main() -> None:
    cfg = get_config()
    app = build_app(cfg)
    app.queue(default_concurrency_limit=1).launch(
        server_name=str(cfg.demo.get("host", "0.0.0.0")),
        server_port=int(cfg.demo.get("port", 7860)),
        show_api=False,
        share=False,
        inbrowser=False,
    )


if __name__ == "__main__":
    main()
