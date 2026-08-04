"""Booth visual theme.

Design target: a 55"+ monitor read from three metres away by someone walking
past. That means very large type, very high contrast, few words, and numbers
that dominate. Gradio's defaults are built for desktop dashboards, so most of
the work here is overriding them.
"""

from __future__ import annotations

import gradio as gr

from .config import Config


def build_theme(cfg: Config) -> gr.Theme:
    accent = cfg.demo.get("theme_accent", "#0068B5")
    return gr.themes.Base(
        primary_hue=gr.themes.colors.blue,
        secondary_hue=gr.themes.colors.sky,
        neutral_hue=gr.themes.colors.slate,
        font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
        font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "monospace"],
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
    )


def build_css(cfg: Config) -> str:
    accent = cfg.demo.get("theme_accent", "#0068B5")
    accent_bright = cfg.demo.get("theme_accent_bright", "#00C7FD")
    return f"""
:root {{
  --intel-blue: {accent};
  --intel-bright: {accent_bright};
  --amx-on: #00E08F;
  --amx-off: #FF9E3D;
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
  background: linear-gradient(135deg, var(--intel-blue), var(--intel-bright));
  color: #04121F; font-weight: 800; font-size: 1.5rem; letter-spacing: 0.04em;
}}
.accel-badge.missing {{ background: #6B2020; color: #FFD9D9; }}
.pill {{
  display: inline-block; padding: 6px 16px; border-radius: 999px;
  font-weight: 700; font-size: 0.95rem; border: 1px solid var(--edge);
}}
.pill.on {{ background: rgba(0,224,143,0.14); color: var(--amx-on); border-color: var(--amx-on); }}
.pill.off {{ background: rgba(255,158,61,0.14); color: var(--amx-off); border-color: var(--amx-off); }}
.pill.warn {{ background: rgba(255,158,61,0.14); color: var(--amx-off); border-color: var(--amx-off); }}
.pill.bad {{ background: rgba(255,90,90,0.14); color: #FF6B6B; border-color: #FF6B6B; }}

/* Anything estimated rather than measured must look different from real data. */
.illustrative {{
  border-left: 4px solid var(--amx-off);
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
.stage-fill.done {{ background: linear-gradient(90deg, #00A86B, var(--amx-on)); }}
.stage-time {{ width: 110px; text-align: right; font-variant-numeric: tabular-nums;
  font-size: 1.1rem; color: var(--ink-dim); font-weight: 600; }}

/* ---------- race ---------- */
.race-lane {{
  background: var(--panel); border: 1px solid var(--edge);
  border-radius: 16px; padding: 20px 24px; margin-bottom: 16px;
}}
.race-lane.on {{ border-color: var(--amx-on); box-shadow: 0 0 26px rgba(0,224,143,0.14); }}
.race-lane.off {{ border-color: var(--amx-off); }}
.race-head {{ display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 12px; }}
.race-label {{ font-size: 1.6rem; font-weight: 800; }}
.race-label.on {{ color: var(--amx-on); }}
.race-label.off {{ color: var(--amx-off); }}
.race-clock {{ font-size: 2.6rem; font-weight: 800; font-variant-numeric: tabular-nums; }}
.race-track {{ height: 34px; background: var(--panel-2); border-radius: 999px;
  border: 1px solid var(--edge); overflow: hidden; }}
.race-fill {{ height: 100%; border-radius: 999px; transition: width .4s ease; }}
.race-fill.on {{ background: linear-gradient(90deg, #00A86B, var(--amx-on)); }}
.race-fill.off {{ background: linear-gradient(90deg, #C4701F, var(--amx-off)); }}
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
