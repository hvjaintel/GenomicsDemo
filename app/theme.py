"""Booth visual theme.

Design target: a 55"+ monitor read from three metres away by someone walking
past. That means very large type, very high contrast, few words, and numbers
that dominate. Gradio's defaults are built for desktop dashboards, so most of
the work here is overriding them.
"""

from __future__ import annotations

import base64
import functools
from pathlib import Path

import gradio as gr

from .config import Config

FONT_DIR = Path(__file__).parent / "static" / "fonts"

# (family, filename, css weight range). Both files are variable fonts, so one
# binary covers every weight we use -- Google serves the same file for 400 and
# 600, which is why bundling per-weight copies would just be the same bytes
# twice.
BUNDLED_FONTS = (
    ("Inter", "Inter-var.woff2", "100 900"),
    ("JetBrains Mono", "JetBrainsMono-var.woff2", "100 800"),
)


@functools.lru_cache(maxsize=1)
def font_face_css() -> str:
    """@font-face rules with the fonts inlined as data URIs.

    Inlining rather than serving from a static route is deliberate. A booth
    machine may be on a hostile network or none at all, and a data URI cannot
    fail to resolve, cannot be blocked, and does not depend on Gradio's
    file-serving paths staying put across versions. The cost is ~79 KB added to
    a page that is only ever loaded on the local machine.

    A missing font file degrades to the system stack rather than raising -- a
    booth screen with the wrong typeface still runs the demo, whereas one that
    will not start does not.
    """
    rules = []
    for family, filename, weight_range in BUNDLED_FONTS:
        path = FONT_DIR / filename
        try:
            encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        except OSError:
            continue
        rules.append(
            f"@font-face{{font-family:'{family}';font-style:normal;"
            f"font-weight:{weight_range};font-display:swap;"
            f"src:url(data:font/woff2;base64,{encoded}) format('woff2');}}"
        )
    return "\n".join(rules)


def build_theme(cfg: Config) -> gr.Theme:
    accent = cfg.demo.get("theme_accent", "#0068B5")
    return gr.themes.Base(
        primary_hue=gr.themes.colors.blue,
        secondary_hue=gr.themes.colors.sky,
        neutral_hue=gr.themes.colors.slate,
        # Plain Font, not GoogleFont: GoogleFont injects a <link> to
        # fonts.googleapis.com, which is render-blocking. With no network that
        # fails fast, but on conference wi-fi that accepts the connection and
        # then stalls, it can hold the booth screen blank for seconds. The
        # families are supplied by font_face_css() instead.
        font=[gr.themes.Font("Inter"), "system-ui", "sans-serif"],
        font_mono=[gr.themes.Font("JetBrains Mono"), "monospace"],
    ).set(
        body_background_fill="#0A0E14",
        body_background_fill_dark="#0A0E14",
        body_text_color="#E8EEF5",
        body_text_color_dark="#E8EEF5",
        block_background_fill="#111823",
        block_background_fill_dark="#111823",
        block_border_color="#1F2C3D",
        block_border_color_dark="#1F2C3D",
        block_label_text_color="#8FA3BA",
        block_title_text_color="#E8EEF5",
        border_color_primary="#1F2C3D",
        button_primary_background_fill=accent,
        button_primary_background_fill_hover="#0080D6",
        button_primary_text_color="#FFFFFF",
        input_background_fill="#0D141D",
        panel_background_fill="#111823",
        # The general surface fills. These were the real gap: the theme set
        # block_background_fill and panel_background_fill but left
        # background_fill_primary/secondary at their light defaults, and Gradio
        # uses those for tables and the tab bar. The pre-flight table was
        # rendering near-white text on white at a contrast ratio of 1.12:1 --
        # invisible, not merely hard to read.
        background_fill_primary="#111823",
        background_fill_primary_dark="#111823",
        background_fill_secondary="#0D141D",
        background_fill_secondary_dark="#0D141D",
        table_even_background_fill="#111823",
        table_even_background_fill_dark="#111823",
        table_odd_background_fill="#0D141D",
        table_odd_background_fill_dark="#0D141D",
        table_text_color="#E8EEF5",
        table_text_color_dark="#E8EEF5",
        table_border_color="#1F2C3D",
        table_border_color_dark="#1F2C3D",
        # Radio and checkbox option labels.
        #
        # These need setting explicitly. Gradio defaults checkbox_label_text_color
        # to *body_text_color -- which is near-white here -- while defaulting
        # checkbox_label_background_fill to *button_secondary_background_fill,
        # which stays a LIGHT gradient. The result on the booth screen was
        # near-white text in a white box: the sample selector was effectively
        # unreadable. Both halves have to be pinned, not just one.
        checkbox_label_background_fill="#0D141D",
        checkbox_label_background_fill_dark="#0D141D",
        checkbox_label_background_fill_hover="#18222F",
        checkbox_label_background_fill_hover_dark="#18222F",
        checkbox_label_background_fill_selected=accent,
        checkbox_label_background_fill_selected_dark=accent,
        checkbox_label_text_color="#E8EEF5",
        checkbox_label_text_color_dark="#E8EEF5",
        checkbox_label_text_color_selected="#FFFFFF",
        checkbox_label_text_color_selected_dark="#FFFFFF",
        checkbox_label_border_color="#1F2C3D",
        checkbox_label_border_color_dark="#1F2C3D",
        checkbox_label_border_color_hover="#2C3E55",
        checkbox_label_border_color_hover_dark="#2C3E55",
        checkbox_background_color="#0D141D",
        checkbox_background_color_dark="#0D141D",
        checkbox_background_color_selected=accent,
        checkbox_background_color_selected_dark=accent,
        checkbox_border_color="#2C3E55",
        checkbox_border_color_dark="#2C3E55",
        # Secondary buttons inherit the same light default; pin them too so the
        # CSS override in build_css() is a refinement rather than the only thing
        # standing between the booth and unreadable controls.
        button_secondary_background_fill="#18222F",
        button_secondary_background_fill_dark="#18222F",
        button_secondary_background_fill_hover="#22303F",
        button_secondary_background_fill_hover_dark="#22303F",
        button_secondary_text_color="#E8EEF5",
        button_secondary_text_color_dark="#E8EEF5",
    )


def build_css(cfg: Config) -> str:
    accent = cfg.demo.get("theme_accent", "#0068B5")
    accent_bright = cfg.demo.get("theme_accent_bright", "#00C7FD")
    badge_start = cfg.demo.get("theme_badge_start", "#00AEEF")
    return f"""
{font_face_css()}
:root {{
  --intel-blue: {accent};
  --intel-bright: {accent_bright};
  --accent-ok: #00E08F;
  --accent-warn: #FF9E3D;
  --ink: #E8EEF5;
  --ink-dim: #8FA3BA;
  --panel: #111823;
  --panel-2: #0D141D;
  --edge: #1F2C3D;
}}

.gradio-container {{
  max-width: 100% !important;
  padding: 0 !important;
  background: radial-gradient(1200px 700px at 20% -10%, #12243A 0%, #0A0E14 60%) !important;
}}

/* ---------- masthead ---------- */
.booth-masthead {{
  display: flex; align-items: center; justify-content: space-between;
  gap: 24px; padding: 20px 34px;
  border-bottom: 2px solid var(--edge);
  background: linear-gradient(90deg, rgba(0,104,181,0.20) 0%, rgba(10,14,20,0) 70%);
}}
.booth-title {{
  font-size: 2.5rem; font-weight: 800; letter-spacing: -0.02em;
  color: var(--ink); line-height: 1.1; margin: 0;
}}
.booth-subtitle {{ font-size: 1.15rem; color: var(--ink-dim); margin: 6px 0 0; }}
.partner-mark {{
  font-size: 1rem; font-weight: 700; color: var(--intel-bright);
  border: 2px solid var(--intel-blue); border-radius: 999px;
  padding: 10px 22px; white-space: nowrap;
}}

/* ---------- big numbers ---------- */
.metric-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 18px; }}
.metric-card {{
  background: var(--panel); border: 1px solid var(--edge); border-radius: 16px;
  padding: 22px 26px;
}}
.metric-label {{
  font-size: 0.95rem; text-transform: uppercase; letter-spacing: 0.12em;
  color: var(--ink-dim); margin-bottom: 10px; font-weight: 600;
}}
.metric-value {{ font-size: 3.1rem; font-weight: 800; line-height: 1; color: var(--ink); }}
.metric-value.small {{ font-size: 1.9rem; }}
.metric-unit {{ font-size: 1.1rem; color: var(--ink-dim); margin-left: 8px; font-weight: 500; }}
.metric-note {{ font-size: 0.9rem; color: var(--ink-dim); margin-top: 10px; }}

/* ---------- badges ---------- */
.accel-badge {{
  display: inline-block; padding: 16px 30px; border-radius: 14px;
  /* Both gradient stops must stay light enough for the dark text below.
     This used to start at --intel-blue (#0068B5), which gave only 3.28:1
     against #04121F -- the left third of the badge failed WCAG AA while the
     right end passed, so it read fine on a laptop and washed out on the booth
     screen. #00AEEF is Intel's brighter brand cyan and scores 7.47:1. */
  background: linear-gradient(135deg, {badge_start}, var(--intel-bright));
  color: #04121F; font-weight: 800; font-size: 1.5rem; letter-spacing: 0.04em;
}}
.accel-badge.missing {{ background: #6B2020; color: #FFD9D9; }}
.pill {{
  display: inline-block; padding: 6px 16px; border-radius: 999px;
  font-weight: 700; font-size: 0.95rem; border: 1px solid var(--edge);
}}
.pill.on {{ background: rgba(0,224,143,0.14); color: var(--accent-ok); border-color: var(--accent-ok); }}
.pill.off {{ background: rgba(255,158,61,0.14); color: var(--accent-warn); border-color: var(--accent-warn); }}
.pill.warn {{ background: rgba(255,158,61,0.14); color: var(--accent-warn); border-color: var(--accent-warn); }}
.pill.bad {{ background: rgba(255,90,90,0.14); color: #FF6B6B; border-color: #FF6B6B; }}

/* Anything estimated rather than measured must look different from real data. */
.illustrative {{
  border-left: 4px solid var(--accent-warn);
  background: rgba(255,158,61,0.07);
  padding: 12px 18px; border-radius: 0 10px 10px 0;
  color: #FFD9A8; font-size: 0.95rem;
}}

/* Replay mode must be impossible to miss or mistake for a live run. */
.replay-banner {{
  background: repeating-linear-gradient(45deg, #7A3E00, #7A3E00 18px, #8F4A00 18px, #8F4A00 36px);
  color: #FFF3E0; font-weight: 800; font-size: 1.5rem; letter-spacing: 0.06em;
  text-align: center; padding: 16px; border-radius: 12px; margin-bottom: 14px;
  border: 2px solid #FFB259;
}}

/* ---------- giant action button ---------- */
.start-button button {{
  font-size: 2.2rem !important; font-weight: 800 !important;
  padding: 34px 20px !important; border-radius: 18px !important;
  letter-spacing: 0.03em; min-height: 120px;
  background: linear-gradient(135deg, var(--intel-blue), #00A3E0) !important;
  border: none !important; box-shadow: 0 10px 34px rgba(0,104,181,0.42) !important;
}}
.start-button button:hover {{ transform: translateY(-2px); transition: transform .15s ease; }}
.start-button button:disabled {{ opacity: 0.45 !important; box-shadow: none !important; }}
.secondary-button button {{
  font-size: 1.1rem !important; font-weight: 700 !important;
  padding: 18px 20px !important; border-radius: 12px !important; min-height: 62px;
}}

/* ---------- stage progress ---------- */
.stage-row {{ display: flex; align-items: center; gap: 16px; margin-bottom: 14px; }}
.stage-name {{
  width: 250px; font-size: 1.15rem; font-weight: 700; color: var(--ink);
  flex-shrink: 0;
}}
.stage-track {{
  flex: 1; height: 30px; background: var(--panel-2);
  border: 1px solid var(--edge); border-radius: 999px; overflow: hidden;
}}
.stage-fill {{
  height: 100%; border-radius: 999px;
  background: linear-gradient(90deg, var(--intel-blue), var(--intel-bright));
  transition: width .35s ease;
}}
.stage-fill.done {{ background: linear-gradient(90deg, #00A86B, var(--accent-ok)); }}
.stage-time {{ width: 110px; text-align: right; font-variant-numeric: tabular-nums;
  font-size: 1.1rem; color: var(--ink-dim); font-weight: 600; }}

/* ---------- DNA activity helix ---------- */
/* A liveness indicator, not a progress bar -- see components.dna_helix.
   Each pair is one column; the two nodes swap top and bottom over one period
   and the rung stretches between them. Giving column i a negative delay
   proportional to i offsets the phase along the row, which is what turns a set
   of independent bobbing dots into a helix that appears to rotate. */
.dna-panel {{
  background: var(--panel); border: 1px solid var(--edge);
  border-radius: 14px; padding: 18px 24px 14px; margin-bottom: 18px;
}}
.dna-helix {{
  display: flex; align-items: stretch; justify-content: center;
  gap: 10px; height: 86px;
}}
.dna-pair {{
  position: relative; width: 10px; flex: 0 0 auto;
  --period: 2.4s;
  --phase: calc(var(--i) * var(--period) / -12);
}}
.dna-node, .dna-rung {{
  position: absolute; left: 50%; transform: translateX(-50%);
  animation-duration: var(--period);
  animation-timing-function: ease-in-out;
  animation-iteration-count: infinite;
  animation-direction: alternate;
}}
.dna-node {{
  width: 10px; height: 10px; border-radius: 50%;
  animation-name: dna-bob; animation-delay: var(--phase);
}}
.dna-node.a {{ background: var(--intel-bright); box-shadow: 0 0 10px rgba(0,199,253,0.55); }}
.dna-node.b {{
  background: var(--intel-blue); box-shadow: 0 0 10px rgba(0,104,181,0.55);
  /* One full period, not half. With animation-direction: alternate the cycle
     is two periods long, so a period/2 offset puts B a quarter-cycle behind A
     rather than opposite it -- the strands then cross out of step with the
     rung, which closes at its own rhythm and the helix reads as noise. */
  animation-delay: calc(var(--phase) - var(--period));
}}
.dna-rung {{
  width: 3px; border-radius: 2px;
  background: linear-gradient(180deg, var(--intel-bright), var(--intel-blue));
  opacity: 0.55;
  /* The rung spans node centres, so it closes to nothing twice per period --
     once at each crossing -- hence the halved duration. */
  animation-name: dna-rung-top, dna-rung-height;
  animation-duration: calc(var(--period) / 2), calc(var(--period) / 2);
  animation-delay: var(--phase), var(--phase);
  /* ease-in, not the ease-in-out the nodes use. The rung tracks the nearer of
     the two nodes, which over half a period traces only the first half of the
     nodes' curve -- the accelerating half. Matching ease-in-out here leaves the
     rung visibly short of the nodes at mid-phase. */
  animation-timing-function: ease-in;
}}
@keyframes dna-bob {{
  from {{ top: 4px; }}
  to   {{ top: calc(100% - 14px); }}
}}
@keyframes dna-rung-top {{
  from {{ top: 9px; }}
  to   {{ top: 50%; }}
}}
@keyframes dna-rung-height {{
  from {{ height: calc(100% - 18px); }}
  to   {{ height: 0px; }}
}}

/* Only the running state animates. Every other state freezes the helix, which
   is the whole point: if output stops, the screen stops. */
.dna-panel:not(.running) .dna-node,
.dna-panel:not(.running) .dna-rung {{ animation-play-state: paused; }}
.dna-panel.idle .dna-helix {{ opacity: 0.35; }}
.dna-panel.quiet .dna-helix {{ opacity: 0.55; }}
.dna-panel.done .dna-node.a, .dna-panel.done .dna-node.b {{
  background: var(--accent-ok); box-shadow: 0 0 10px rgba(0,224,143,0.5);
}}
.dna-panel.done .dna-rung {{ background: var(--accent-ok); }}
.dna-panel.failed .dna-node.a, .dna-panel.failed .dna-node.b {{
  background: #FF6B6B; box-shadow: none;
}}
.dna-panel.failed .dna-rung {{ background: #FF6B6B; }}
.dna-caption {{
  margin-top: 12px; text-align: center;
  font-size: 1rem; color: var(--ink-dim); line-height: 1.45;
}}
.dna-panel.running .dna-caption {{ color: var(--ink); }}
.dna-panel.quiet .dna-caption {{ color: var(--accent-warn); }}
.dna-panel.failed .dna-caption {{ color: #FF8F8F; }}

@media (prefers-reduced-motion: reduce) {{
  .dna-node, .dna-rung {{ animation: none !important; }}
  .dna-node.a {{ top: 4px; }}
  .dna-node.b {{ top: calc(100% - 14px); }}
  .dna-rung {{ top: 9px; height: calc(100% - 18px); }}
}}

/* ---------- race ---------- */
.race-lane {{
  background: var(--panel); border: 1px solid var(--edge);
  border-radius: 16px; padding: 20px 24px; margin-bottom: 16px;
}}
.race-lane.on {{ border-color: var(--accent-ok); box-shadow: 0 0 26px rgba(0,224,143,0.14); }}
.race-lane.off {{ border-color: var(--accent-warn); }}
.race-head {{ display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 12px; }}
.race-label {{ font-size: 1.6rem; font-weight: 800; }}
.race-label.on {{ color: var(--accent-ok); }}
.race-label.off {{ color: var(--accent-warn); }}
.race-clock {{ font-size: 2.6rem; font-weight: 800; font-variant-numeric: tabular-nums; }}
.race-track {{ height: 34px; background: var(--panel-2); border-radius: 999px;
  border: 1px solid var(--edge); overflow: hidden; }}
.race-fill {{ height: 100%; border-radius: 999px; transition: width .4s ease; }}
.race-fill.on {{ background: linear-gradient(90deg, #00A86B, var(--accent-ok)); }}
.race-fill.off {{ background: linear-gradient(90deg, #C4701F, var(--accent-warn)); }}
.race-meta {{ display: flex; gap: 26px; margin-top: 12px; color: var(--ink-dim); font-size: 1rem; }}

.speedup-card {{
  text-align: center; padding: 40px 24px; border-radius: 20px;
  background: linear-gradient(135deg, rgba(0,104,181,0.28), rgba(0,199,253,0.12));
  border: 2px solid var(--intel-bright); margin-top: 12px;
}}
.speedup-number {{
  font-size: 7rem; font-weight: 900; line-height: 0.95; color: #FFFFFF;
  text-shadow: 0 0 44px rgba(0,199,253,0.55); font-variant-numeric: tabular-nums;
}}
.speedup-caption {{ font-size: 1.7rem; font-weight: 700; color: var(--intel-bright); margin-top: 12px; }}
.speedup-sub {{ font-size: 1.05rem; color: var(--ink-dim); margin-top: 12px; }}

/* ---------- log pane ---------- */
.log-pane textarea {{
  font-family: 'JetBrains Mono', monospace !important;
  font-size: 0.95rem !important; line-height: 1.5 !important;
  background: #060A10 !important; color: #9FE8C0 !important;
  border: 1px solid var(--edge) !important;
}}

/* ---------- tabs ---------- */
.tab-nav button {{
  font-size: 1.25rem !important; font-weight: 700 !important;
  padding: 18px 30px !important;
}}
.tab-nav button.selected {{
  color: var(--intel-bright) !important;
  border-bottom: 4px solid var(--intel-bright) !important;
}}

/* ---------- tables ---------- */
table {{ font-size: 1.05rem !important; }}
table th {{ color: var(--ink-dim) !important; text-transform: uppercase;
  font-size: 0.85rem !important; letter-spacing: 0.08em; }}

/* ---------- misc ---------- */
.section-title {{
  font-size: 1.45rem; font-weight: 800; color: var(--ink);
  margin: 6px 0 14px; letter-spacing: -0.01em;
}}
.dim {{ color: var(--ink-dim); }}
.mono {{ font-family: 'JetBrains Mono', monospace; }}
footer {{ display: none !important; }}
"""
