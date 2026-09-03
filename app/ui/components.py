"""Shared HTML fragment builders for the booth panels.

Kept in one place so the visual language stays consistent, and so the rules
about labelling estimates versus measurements are enforced by construction
rather than by remembering to do it on each screen.
"""

from __future__ import annotations

from html import escape


def metric(label: str, value: str, unit: str = "", note: str = "", small: bool = False) -> str:
    size_class = " small" if small else ""
    unit_html = f'<span class="metric-unit">{escape(unit)}</span>' if unit else ""
    note_html = f'<div class="metric-note">{escape(note)}</div>' if note else ""
    return (
        f'<div class="metric-card">'
        f'<div class="metric-label">{escape(label)}</div>'
        f'<div class="metric-value{size_class}">{escape(value)}{unit_html}</div>'
        f"{note_html}</div>"
    )


def metric_grid(cards: list[str]) -> str:
    return f'<div class="metric-grid">{"".join(cards)}</div>'


def pill(text: str, kind: str = "") -> str:
    return f'<span class="pill {kind}">{escape(text)}</span>'


def section(title: str) -> str:
    return f'<div class="section-title">{escape(title)}</div>'


def illustrative(text: str) -> str:
    """Wrap any figure that is an assumption rather than a measurement."""
    return f'<div class="illustrative"><strong>Illustrative — </strong>{escape(text)}</div>'


def replay_banner(text: str) -> str:
    return f'<div class="replay-banner">{escape(text)}</div>'


def stage_bar(label: str, percent: float, done: bool, duration_s: float | None) -> str:
    fill_class = "stage-fill done" if done else "stage-fill"
    timing = f"{duration_s:.0f}s" if duration_s is not None else (f"{percent:.0f}%" if percent else "—")
    return (
        f'<div class="stage-row">'
        f'<div class="stage-name">{escape(label)}</div>'
        f'<div class="stage-track"><div class="{fill_class}" style="width:{max(0.0, min(100.0, percent)):.1f}%"></div></div>'
        f'<div class="stage-time">{escape(timing)}</div>'
        f"</div>"
    )


DNA_STATES = ("idle", "running", "quiet", "done", "failed")


def dna_helix(state: str, caption: str, pairs: int = 24) -> str:
    """A turning double helix that reports whether the pipeline is alive.

    This is deliberately *not* a progress bar. Progress is already shown by the
    stage bars, which are driven by parsed DeepVariant output; a spinner that
    invented its own notion of "how far along" would be exactly the kind of
    decoration this demo exists to argue against. What the helix shows is one
    real fact: whether the container is still emitting output. It turns while
    log lines arrive and stops when they stop, so a wedged run looks wedged on
    the booth screen instead of looking busy.

    The animation is pure CSS on static markup. Nothing here is driven by a
    timer on the Python side, because the caller re-renders the console several
    times a second and any JavaScript-driven state would be thrown away with
    each replacement of the DOM node.
    """
    if state not in DNA_STATES:
        raise ValueError(f"unknown helix state {state!r}; expected one of {DNA_STATES}")

    rungs = "".join(
        f'<span class="dna-pair" style="--i:{i}">'
        f'<i class="dna-rung"></i><i class="dna-node a"></i><i class="dna-node b"></i>'
        f"</span>"
        for i in range(pairs)
    )
    return (
        f'<div class="dna-panel {state}">'
        f'<div class="dna-helix" aria-hidden="true">{rungs}</div>'
        f'<div class="dna-caption" role="status">{escape(caption)}</div>'
        f"</div>"
    )


def race_lane(
    label: str,
    on: bool,
    elapsed: str,
    percent: float,
    meta: list[str],
) -> str:
    side = "on" if on else "off"
    meta_html = "".join(f"<div>{escape(m)}</div>" for m in meta)
    return (
        f'<div class="race-lane {side}">'
        f'<div class="race-head">'
        f'<div class="race-label {side}">{escape(label)}</div>'
        f'<div class="race-clock">{escape(elapsed)}</div>'
        f"</div>"
        f'<div class="race-track"><div class="race-fill {side}" '
        f'style="width:{max(0.0, min(100.0, percent)):.1f}%"></div></div>'
        f'<div class="race-meta">{meta_html}</div>'
        f"</div>"
    )


def speedup_card(multiplier: float | None, caption: str, sub: str) -> str:
    number = f"{multiplier:.1f}×" if multiplier is not None else "—"
    return (
        f'<div class="speedup-card">'
        f'<div class="speedup-number">{escape(number)}</div>'
        f'<div class="speedup-caption">{escape(caption)}</div>'
        f'<div class="speedup-sub">{escape(sub)}</div>'
        f"</div>"
    )


def fmt_duration(seconds: float | None) -> str:
    if seconds is None:
        return "—"
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes, secs = divmod(int(seconds), 60)
    if minutes < 60:
        return f"{minutes}m {secs:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m"


def fmt_bytes(num: float | None) -> str:
    if not num:
        return "—"
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(num) < 1024:
            return f"{num:.1f} {unit}" if unit != "B" else f"{num:.0f} B"
        num /= 1024
    return f"{num:.1f} PB"
