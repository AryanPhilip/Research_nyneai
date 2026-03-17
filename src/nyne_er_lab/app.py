"""Entity Resolution Lab — interactive Streamlit dashboard."""

from __future__ import annotations

import base64
from html import escape
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from sklearn.calibration import calibration_curve

from nyne_er_lab.app_data import DashboardData, load_dashboard_data, _pair_key
from nyne_er_lab.eval.metrics import summarize_predictions, threshold_sweep


def _logo_b64() -> str:
    """Return base64-encoded logo PNG for use in img src."""
    logo_path = Path(__file__).parent / "demo" / "logo.png"
    if logo_path.exists():
        return base64.b64encode(logo_path.read_bytes()).decode("utf-8")
    return ""


# ---------------------------------------------------------------------------
# Helpers (must be defined before use in Streamlit's top-to-bottom execution)
# ---------------------------------------------------------------------------

def _hex_to_rgb(hex_color: str) -> str:
    """Convert '#06b6d4' to '6,182,212'."""
    h = hex_color.lstrip("#")
    return ",".join(str(int(h[i:i+2], 16)) for i in (0, 2, 4))

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Entity Resolution Lab",
    page_icon="🍌",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Theme CSS
# ---------------------------------------------------------------------------

THEME_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --bg: #191A1A;
    --surface: #202222;
    --surface-2: #2F3336;
    --border: #2F3336;
    --cyan: #3B82F6;
    --emerald: #10B981;
    --rose: #EF4444;
    --amber: #F59E0B;
    --violet: #8B5CF6;
    --text: #E8E8E6;
    --muted: #8E8E8E;
    --dim: #5F6368;
}

.stApp {
    background: var(--bg);
    font-family: 'Inter', -apple-system, sans-serif;
}
.stApp header { background: transparent !important; }
.stApp, .stApp p, .stApp li, .stApp span, .stApp label { color: var(--text); }
h1, h2, h3, h4 { color: var(--text) !important; font-weight: 700 !important; }

/* Hero header */
.hero-header {
    display: flex;
    align-items: center;
    gap: 18px;
    margin-bottom: 24px;
}
.hero-logo {
    flex-shrink: 0;
    width: 52px;
    height: 52px;
    border-radius: 12px;
    overflow: hidden;
    background: var(--surface);
    border: 1px solid var(--border);
}
.hero-logo img { width: 100%; height: 100%; object-fit: cover; display: block; }
.hero-text { flex: 1; }
.hero-title {
    font-size: 1.7rem;
    font-weight: 700;
    color: var(--text);
    margin: 0 0 2px;
    line-height: 1.2;
}
.hero-sub {
    color: var(--muted);
    font-size: 0.9rem;
    margin: 0;
}

/* Metric cards */
[data-testid="stMetric"] {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 18px 22px;
    transition: border-color 0.2s;
}
[data-testid="stMetric"]:hover { border-color: var(--cyan); }
[data-testid="stMetricLabel"] { color: var(--muted) !important; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.06em; }
[data-testid="stMetricValue"] { color: var(--text) !important; font-weight: 700; font-family: 'JetBrains Mono', monospace; }
[data-testid="stMetricDelta"] > div { color: var(--emerald) !important; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 2px;
    background: var(--surface);
    border-radius: 12px;
    padding: 4px;
    border: 1px solid var(--border);
}
.stTabs [data-baseweb="tab"] {
    color: var(--muted);
    border-radius: 8px;
    padding: 10px 18px;
    font-weight: 600;
    font-size: 0.85rem;
    letter-spacing: 0.02em;
}
.stTabs [aria-selected="true"] {
    background: var(--surface-2) !important;
    color: var(--text) !important;
    border: 1px solid var(--border);
}

/* Expanders */
details[data-testid="stExpander"] {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
}

/* Inputs */
[data-baseweb="select"] { background: var(--surface); }
[data-baseweb="select"] * { color: var(--text) !important; }
[data-baseweb="input"] { background: var(--surface) !important; }
[data-baseweb="textarea"] { background: var(--surface) !important; }
.stTextInput input, .stTextArea textarea {
    background: var(--surface) !important;
    color: var(--text) !important;
    border-color: var(--border) !important;
    border-radius: 10px !important;
}
.stTextInput input:focus, .stTextArea textarea:focus {
    border-color: var(--cyan) !important;
    box-shadow: 0 0 0 1px var(--cyan) !important;
}

/* Card */
.card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 12px;
    transition: border-color 0.2s, transform 0.2s;
}
.card:hover { border-color: var(--cyan); transform: translateY(-1px); }

/* Glow card for live results -> changed to solid border accent */
.card-glow {
    background: var(--surface);
    border: 1px solid var(--cyan);
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 12px;
}

/* Pills */
.pill {
    display: inline-block;
    background: var(--surface-2);
    color: var(--text);
    padding: 3px 12px;
    border-radius: 999px;
    font-size: 11px;
    margin: 2px 4px 2px 0;
    font-weight: 600;
    letter-spacing: 0.03em;
    text-transform: uppercase;
}
.pill-match { background: rgba(16,185,129,0.1); color: var(--emerald); }
.pill-non_match { background: rgba(239,68,68,0.1); color: var(--rose); }
.pill-abstain { background: rgba(245,158,11,0.1); color: var(--amber); }
.pill-violet { background: rgba(139,92,246,0.1); color: var(--violet); }

/* Badges */
.badge {
    display: inline-block;
    padding: 5px 16px;
    border-radius: 999px;
    font-size: 12px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}
.badge-match { background: rgba(16,185,129,0.1); color: var(--emerald); border: 1px solid rgba(16,185,129,0.2); }
.badge-non_match { background: rgba(239,68,68,0.1); color: var(--rose); border: 1px solid rgba(239,68,68,0.2); }
.badge-abstain { background: rgba(245,158,11,0.1); color: var(--amber); border: 1px solid rgba(245,158,11,0.2); }
.badge-high { background: rgba(16,185,129,0.1); color: var(--emerald); border: 1px solid rgba(16,185,129,0.2); }
.badge-medium { background: rgba(59,130,246,0.1); color: var(--cyan); border: 1px solid rgba(59,130,246,0.2); }
.badge-low { background: rgba(245,158,11,0.1); color: var(--amber); border: 1px solid rgba(245,158,11,0.2); }
.badge-uncertain { background: rgba(239,68,68,0.1); color: var(--rose); border: 1px solid rgba(239,68,68,0.2); }

/* Confidence bar */
.conf-bar-bg {
    width: 100%; height: 6px; background: var(--surface-2);
    border-radius: 999px; overflow: hidden; margin: 8px 0;
}
.conf-bar-fill { height: 100%; border-radius: 999px; }

/* Signal colors */
.signal-supporting { color: var(--emerald); font-weight: 600; }
.signal-contradicting { color: var(--rose); font-weight: 600; }

/* Step indicator for pipeline walkthrough */
.step-indicator {
    display: flex; gap: 8px; margin: 16px 0;
}
.step {
    flex: 1; text-align: center; padding: 10px 8px;
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 10px; font-size: 11px; color: var(--muted);
    font-weight: 600; text-transform: uppercase; letter-spacing: 0.04em;
}
.step-active {
    border-color: var(--cyan);
    color: var(--cyan);
    background: rgba(59,130,246,0.08);
}

/* Live status pulse */
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
}
.live-dot {
    display: inline-block;
    width: 8px; height: 8px;
    background: var(--emerald);
    border-radius: 50%;
    margin-right: 6px;
    animation: pulse 2s ease-in-out infinite;
}

/* Divider */
.subtle-divider {
    border: none;
    height: 1px;
    background: var(--border);
    margin: 24px 0;
}
</style>
"""

st.markdown(THEME_CSS, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Plotly defaults
# ---------------------------------------------------------------------------

PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(25,26,26,0)",
    plot_bgcolor="rgba(25,26,26,0)",
    font=dict(family="Inter, -apple-system, sans-serif", color="#E8E8E6", size=12),
    margin=dict(l=40, r=20, t=50, b=40),
    title=dict(font=dict(size=14, color="#E8E8E6", family="Inter, -apple-system, sans-serif"), x=0.01, xanchor="left"),
    legend_font=dict(size=11, color="#8E8E8E"),
    hoverlabel=dict(font_family="Inter, -apple-system, sans-serif", bgcolor="#202222", bordercolor="#2F3336"),
    xaxis=dict(title_font=dict(size=12, color="#8E8E8E"), tickfont=dict(size=11, color="#8E8E8E"), gridcolor="#2F3336", zerolinecolor="#2F3336"),
    yaxis=dict(title_font=dict(size=12, color="#8E8E8E"), tickfont=dict(size=11, color="#8E8E8E"), gridcolor="#2F3336", zerolinecolor="#2F3336"),
)

C = {
    "cyan": "#3B82F6",
    "emerald": "#10B981",
    "rose": "#EF4444",
    "amber": "#F59E0B",
    "violet": "#8B5CF6",
    "slate": "#8E8E8E",
    "muted": "#8E8E8E",
    "dim": "#5F6368",
    "text": "#E8E8E6",
}

SOURCE_COLORS = {
    "github": "#3B82F6",
    "personal_site": "#8B5CF6",
    "conference_bio": "#F59E0B",
    "company_profile": "#10B981",
    "podcast_guest": "#EF4444",
    "huggingface": "#FBBF24",
}

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

data: DashboardData = load_dashboard_data()


# ---------------------------------------------------------------------------
# Shared rendering helpers
# ---------------------------------------------------------------------------

def _render_profile_card(profile, compact: bool = False) -> str:
    """Return HTML for a profile card."""
    src_color = SOURCE_COLORS.get(profile.source_type, C["slate"])
    orgs_str = ", ".join(o.name for o in profile.organizations[:3]) if profile.organizations else "—"
    hl = escape(profile.headline or "")
    if compact:
        return (
            f'<div class="card">'
            f'<strong>{escape(profile.display_name)}</strong> '
            f'<span class="pill" style="background:rgba({_hex_to_rgb(src_color)},0.15);color:{src_color}">'
            f'{profile.source_type}</span><br>'
            f'<small style="color:{C["muted"]}">{hl}</small><br>'
            f'<small style="color:{C["dim"]}">{escape(orgs_str)}</small>'
            f'</div>'
        )
    topics_html = "".join(f'<span class="pill">{escape(t)}</span>' for t in profile.topics[:6])
    return (
        f'<div class="card">'
        f'<strong style="font-size:1.1rem">{escape(profile.display_name)}</strong><br>'
        f'<span class="pill" style="background:rgba({_hex_to_rgb(src_color)},0.15);color:{src_color}">'
        f'{profile.source_type}</span><br>'
        f'<em style="color:{C["muted"]}">{hl}</em><br>'
        f'<small style="color:{C["dim"]}"><strong>Orgs:</strong> {escape(orgs_str)}</small><br>'
        f'{topics_html}'
        f'</div>'
    )


def _render_live_results(result, profile_lookup) -> None:
    """Render the results of a live resolution into the current Streamlit context."""
    from nyne_er_lab.live import LiveResult
    if not isinstance(result, LiveResult):
        return

    st.markdown('<hr class="subtle-divider">', unsafe_allow_html=True)

    if result.best_match_profile:
        st.markdown(
            f'<div class="card-glow">'
            f'<strong style="color:{C["emerald"]};font-size:1.1rem">Match found!</strong><br>'
            f'Best match: <strong>{escape(result.best_match_profile.display_name)}</strong> '
            f'<span class="pill">{result.best_match_profile.source_type}</span>',
            unsafe_allow_html=True,
        )
        if result.best_match_identity_name:
            st.markdown(
                f'Part of identity cluster: <strong>{escape(result.best_match_identity_name)}</strong>',
                unsafe_allow_html=True,
            )
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown(
            f'<div class="card">'
            f'<strong style="color:{C["amber"]}">No confident matches found.</strong><br>'
            f'<span style="color:{C["muted"]}">This profile may represent a new identity not in the corpus.</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown("#### All Comparisons (ranked by score)")
    for rp in result.matches[:12]:
        matched_profile = profile_lookup.get(rp.right_profile_id)
        if not matched_profile:
            continue
        dec_color = C["emerald"] if rp.decision == "match" else C["rose"] if rp.decision == "non_match" else C["amber"]
        st.markdown(
            f'<div class="card" style="border-top: 2px solid {dec_color}">'
            f'<strong>{escape(matched_profile.display_name)}</strong> '
            f'<span class="pill">{matched_profile.source_type}</span> '
            f'<span class="badge badge-{rp.decision}">{rp.decision}</span> '
            f'<span style="font-family:JetBrains Mono;color:{C["cyan"]};margin-left:8px">{rp.score:.4f}</span><br>'
            f'<small style="color:{C["muted"]}">{escape(rp.evidence_card.final_explanation)}</small>'
            f'</div>',
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

_logo = _logo_b64()
_logo_img = f'<img src="data:image/png;base64,{_logo}" alt="Entity Resolution Lab">' if _logo else ""
st.markdown(
    f'<div class="hero-header">'
    f'<div class="hero-logo">{_logo_img}</div>'
    f'<div class="hero-text">'
    f'<p class="hero-title">Entity Resolution Lab</p>'
    f'<p class="hero-sub">Public-web identity resolution — honest benchmarking, evidence-led decisions, conservative abstention.</p>'
    f'</div>'
    f'</div>',
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_overview, tab_network, tab_pairs, tab_explainer, tab_whatif, tab_search, tab_live, tab_inspector = st.tabs([
    "Benchmark",
    "Identity Graph",
    "Failure Lab",
    "Evidence Ledger",
    "Alias Stress",
    "Search Trace",
    "Resolve",
    "Pipeline Inspector",
])


# ========================== TAB 1: OVERVIEW ================================

with tab_overview:
    hybrid_metrics = next(m for m in data.metrics if m["name"] == "hybrid")
    best_baseline_f1 = max(m["f1"] for m in data.metrics if m["name"] != "hybrid")
    leakage_failures = [item for item in data.leakage_checks if not item["passed"]]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Hybrid F1", f"{hybrid_metrics['f1']:.3f}",
              delta=f"+{hybrid_metrics['f1'] - best_baseline_f1:.3f} vs best baseline")
    c2.metric("Precision", f"{hybrid_metrics['precision']:.3f}")
    c3.metric("Recall", f"{hybrid_metrics['recall']:.3f}")
    c4.metric("Leakage Checks", f"{len(data.leakage_checks) - len(leakage_failures)}/{len(data.leakage_checks)}")

    st.markdown('<hr class="subtle-divider">', unsafe_allow_html=True)

    # Model comparison + score distribution side by side
    col_models, col_scores = st.columns([3, 2])

    with col_models:
        model_names = [m["name"] for m in data.metrics]
        metric_keys = ["precision", "recall", "f1", "average_precision"]
        metric_labels = ["Precision", "Recall", "F1", "Avg Precision"]
        bar_colors = [C["slate"], C["rose"], C["cyan"], C["emerald"]]

        fig_models = go.Figure()
        for i, (key, label) in enumerate(zip(metric_keys, metric_labels)):
            fig_models.add_trace(go.Bar(
                name=label, x=model_names, y=[m[key] for m in data.metrics],
                marker_color=bar_colors[i],
            ))
        fig_models.update_layout(
            **PLOTLY_LAYOUT, barmode="group", title_text="Model Comparison",
            yaxis_range=[0, 1.05],
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=380,
        )
        st.plotly_chart(fig_models, use_container_width=True)

    with col_scores:
        # Score distribution by decision type
        fig_dist = go.Figure()
        decision_colors = {"match": C["emerald"], "non_match": C["rose"], "abstain": C["amber"]}
        for decision, scores in data.score_by_decision.items():
            if scores:
                fig_dist.add_trace(go.Violin(
                    y=scores, name=decision,
                    marker_color=decision_colors.get(decision, C["slate"]),
                    box_visible=True, meanline_visible=True,
                    fillcolor=decision_colors.get(decision, C["slate"]),
                    opacity=0.6, line_color=decision_colors.get(decision, C["slate"]),
                ))
        fig_dist.update_layout(
            **PLOTLY_LAYOUT, title_text="Score Distribution by Decision",
            yaxis_title="Confidence Score",
            showlegend=False, height=380,
        )
        st.plotly_chart(fig_dist, use_container_width=True)

    # Ablation + Calibration row
    col_abl, col_cal = st.columns(2)

    with col_abl:
        abl_names = [a["name"] for a in data.ablations]
        abl_f1 = [a["f1"] for a in data.ablations]
        fig_abl = go.Figure(go.Bar(
            y=abl_names, x=abl_f1, orientation="h",
            marker_color=[C["emerald"] if n == "full" else C["amber"] for n in abl_names],
            text=[f"{v:.3f}" for v in abl_f1], textposition="outside",
            textfont=dict(color=C["slate"], family="Inter, -apple-system, sans-serif"),
        ))
        fig_abl.update_layout(
            **PLOTLY_LAYOUT, title_text="Feature Ablation (F1)",
            xaxis_range=[0, 1.15],
            height=280,
        )
        st.plotly_chart(fig_abl, use_container_width=True)

    with col_cal:
        cs = data.confusion_slices
        st.markdown("#### Calibration & Abstention")
        mc1, mc2 = st.columns(2)
        mc1.metric("Accepted Precision", f"{cs['accepted_precision']:.3f}")
        mc2.metric("Accepted Recall", f"{cs['accepted_recall']:.3f}")
        mc3, mc4 = st.columns(2)
        mc3.metric("Abstain Rate", f"{cs['abstain_rate']:.3f}")
        mc4.metric("Brier (calibrated)", f"{data.hybrid_run.calibrated_brier:.4f}")

    st.markdown("#### Stress Suites")
    stress_col1, stress_col2 = st.columns(2)
    with stress_col1:
        stress = data.stress_metrics["hard_negative_bank"]
        st.markdown(
            f'<div class="card"><strong>Hard Negative Bank</strong><br>'
            f'F1: <span style="color:{C["cyan"]}">{stress["f1"]:.3f}</span><br>'
            f'Precision: {stress["precision"]:.3f} · Recall: {stress["recall"]:.3f}<br>'
            f'Abstain rate: {stress["abstain_rate"]:.3f}</div>',
            unsafe_allow_html=True,
        )
    with stress_col2:
        stress = data.stress_metrics["synthetic_stress"]
        st.markdown(
            f'<div class="card"><strong>Synthetic Stress</strong><br>'
            f'F1: <span style="color:{C["cyan"]}">{stress["f1"]:.3f}</span><br>'
            f'Precision: {stress["precision"]:.3f} · Recall: {stress["recall"]:.3f}<br>'
            f'Abstain rate: {stress["abstain_rate"]:.3f}</div>',
            unsafe_allow_html=True,
        )

    st.markdown("#### Failure Gallery")
    fg_cols = st.columns(3)
    for idx, item in enumerate(data.failure_gallery[:6]):
        with fg_cols[idx % 3]:
            color = C["emerald"] if item["decision"] == "match" else C["rose"] if item["decision"] == "non_match" else C["amber"]
            counterfactual = " | ".join(item.get("counterfactuals", [])) or "No extra counterfactuals."
            st.markdown(
                f'<div class="card" style="border-top: 2px solid {color}">'
                f'<span class="pill pill-{item["decision"]}">{item["bucket"]}</span><br>'
                f'<strong>{escape(item["left_name"])} vs {escape(item["right_name"])}</strong><br>'
                f'<small style="color:{C["muted"]}">{escape(item["explanation"])}</small><br>'
                f'<small style="color:{C["dim"]}">{escape(counterfactual)}</small>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.caption(
        f"{len(data.profiles)} profiles  ·  "
        f"{len(set(p.source_type for p in data.profiles))} source types  ·  "
        f"{len(data.metrics)} averaged models  ·  "
        f"{data.dataset_summary['headline_dataset']}"
    )


# ========================== TAB 2: IDENTITY NETWORK ========================

with tab_network:
    st.markdown("#### Identity Network Graph")
    st.caption("Nodes are profiles. Edges show pairwise decisions. Clusters emerge from match edges.")

    net = data.network
    if net.node_ids:
        # Edge filter
        edge_filter = st.radio(
            "Show edges",
            ["Matches only", "Matches + Abstains", "All decisions"],
            horizontal=True,
            key="net_edge_filter",
        )

        fig_net = go.Figure()

        # Draw edges
        decision_edge_colors = {"match": C["emerald"], "abstain": C["amber"], "non_match": C["rose"]}
        show_decisions = {"match"}
        if edge_filter == "Matches + Abstains":
            show_decisions = {"match", "abstain"}
        elif edge_filter == "All decisions":
            show_decisions = {"match", "abstain", "non_match"}

        pos_lookup = {nid: (x, y) for nid, x, y in zip(net.node_ids, net.node_x, net.node_y)}

        for decision_type in show_decisions:
            edge_x, edge_y = [], []
            for left, right, dec, score in zip(net.edge_left, net.edge_right, net.edge_decision, net.edge_score):
                if dec != decision_type:
                    continue
                if left in pos_lookup and right in pos_lookup:
                    x0, y0 = pos_lookup[left]
                    x1, y1 = pos_lookup[right]
                    edge_x.extend([x0, x1, None])
                    edge_y.extend([y0, y1, None])

            if edge_x:
                fig_net.add_trace(go.Scatter(
                    x=edge_x, y=edge_y, mode="lines",
                    line=dict(
                        color=decision_edge_colors.get(decision_type, C["dim"]),
                        width=2 if decision_type == "match" else 1,
                    ),
                    opacity=0.6 if decision_type == "match" else 0.25,
                    hoverinfo="skip", showlegend=True, name=decision_type,
                ))

        # Draw nodes colored by source type
        for source_type in set(net.node_types):
            mask = [i for i, t in enumerate(net.node_types) if t == source_type]
            fig_net.add_trace(go.Scatter(
                x=[net.node_x[i] for i in mask],
                y=[net.node_y[i] for i in mask],
                mode="markers+text",
                marker=dict(
                    size=14,
                    color=SOURCE_COLORS.get(source_type, C["slate"]),
                    line=dict(width=2, color="rgba(25,26,26,0.9)"),
                ),
                text=[net.node_names[i].split()[0] for i in mask],
                textposition="top center",
                textfont=dict(size=10, color=C["muted"], family="Inter, -apple-system, sans-serif"),
                hovertext=[
                    f"<b>{net.node_names[i]}</b><br>"
                    f"Source: {net.node_types[i]}<br>"
                    f"Cluster: {net.node_cluster[i]}"
                    for i in mask
                ],
                hoverinfo="text",
                name=source_type,
            ))

        fig_net.update_layout(
            **PLOTLY_LAYOUT,
            height=650,
            xaxis_showgrid=False, xaxis_zeroline=False, xaxis_showticklabels=False,
            yaxis_showgrid=False, yaxis_zeroline=False, yaxis_showticklabels=False,
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5,
                font=dict(size=11, color=C["muted"]),
            ),
        )
        st.plotly_chart(fig_net, use_container_width=True)

        # Cluster summary below graph
        st.markdown('<hr class="subtle-divider">', unsafe_allow_html=True)
        identities = data.identities
        cols = st.columns(min(len(identities), 4))
        for idx, identity in enumerate(identities[:8]):
            with cols[idx % len(cols)]:
                member_count = len(identity.member_profile_ids)
                st.markdown(
                    f'<div class="card">'
                    f'<strong>{escape(identity.canonical_name)}</strong><br>'
                    f'<span class="pill pill-{identity.confidence_band}">{identity.confidence_band}</span> '
                    f'<span style="color:{C["muted"]}">{member_count} profile{"s" if member_count > 1 else ""}</span><br>'
                    f'<small style="color:{C["dim"]}">{", ".join(identity.key_orgs[:2]) if identity.key_orgs else "—"}</small>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
    else:
        st.info("No network data available.")


# ========================== TAB 3: PAIR EXPLORER ===========================

with tab_pairs:
    col_filter, col_detail = st.columns([1, 3])

    with col_filter:
        st.markdown("#### Filters")
        decision_filter = st.selectbox(
            "Decision", ["All", "match", "non_match", "abstain"],
            key="pair_decision_filter",
        )
        conf_range = st.slider(
            "Confidence", 0.0, 1.0, (0.0, 1.0), step=0.01,
            key="pair_conf_range",
        )

        filtered_pairs = [
            p for p in data.resolved_pairs
            if (decision_filter == "All" or p.decision == decision_filter)
            and conf_range[0] <= p.score <= conf_range[1]
        ]
        st.markdown(f"**{len(filtered_pairs)}** pairs")

        # Quick stats
        if filtered_pairs:
            avg_score = sum(p.score for p in filtered_pairs) / len(filtered_pairs)
            st.caption(f"Avg score: {avg_score:.3f}")

    with col_detail:
        if not filtered_pairs:
            st.info("No pairs match filters.")
        else:
            nice_labels = []
            for p in filtered_pairs:
                lp = data.profile_lookup.get(p.left_profile_id)
                rp = data.profile_lookup.get(p.right_profile_id)
                ln = lp.display_name if lp else p.left_profile_id[:12]
                rn = rp.display_name if rp else p.right_profile_id[:12]
                nice_labels.append(f"{ln} vs {rn} ({p.decision} @ {p.score:.2f})")

            selected_idx = st.selectbox(
                "Select pair", range(len(filtered_pairs)),
                format_func=lambda i: nice_labels[i],
                key="pair_selector",
            )
            pair = filtered_pairs[selected_idx]
            left_profile = data.profile_lookup.get(pair.left_profile_id)
            right_profile = data.profile_lookup.get(pair.right_profile_id)

            # Pipeline step indicator
            st.markdown(
                '<div class="step-indicator">'
                '<div class="step step-active">Blocking</div>'
                '<div class="step step-active">Features</div>'
                '<div class="step step-active">Scoring</div>'
                '<div class="step step-active">Decision</div>'
                '</div>',
                unsafe_allow_html=True,
            )

            # Side-by-side profiles
            pcol1, pcol2 = st.columns(2)
            for col, profile in [(pcol1, left_profile), (pcol2, right_profile)]:
                if profile is None:
                    col.warning("Profile not found")
                    continue
                with col:
                    src_color = SOURCE_COLORS.get(profile.source_type, C["slate"])
                    st.markdown(
                        f'<div class="card">'
                        f'<strong style="font-size:1.1rem">{escape(profile.display_name)}</strong><br>'
                        f'<span class="pill" style="background:rgba({_hex_to_rgb(src_color)},0.15);color:{src_color}">'
                        f'{profile.source_type}</span>',
                        unsafe_allow_html=True,
                    )
                    if profile.headline:
                        st.markdown(f"*{profile.headline}*")
                    with st.expander("Bio"):
                        st.write(profile.bio_text[:500] + ("..." if len(profile.bio_text) > 500 else ""))
                    if profile.organizations:
                        st.markdown("**Orgs:** " + ", ".join(o.name for o in profile.organizations))
                    if profile.locations:
                        st.markdown("**Locations:** " + ", ".join(profile.locations))
                    if profile.topics:
                        pills = "".join(f'<span class="pill">{escape(t)}</span>' for t in profile.topics[:8])
                        st.markdown(pills, unsafe_allow_html=True)
                    st.markdown(f"[Source]({profile.url})")
                    st.markdown('</div>', unsafe_allow_html=True)

            # Feature vector
            ex_key = _pair_key(pair.left_profile_id, pair.right_profile_id)
            example = data.example_lookup.get(ex_key)
            if example:
                feat_names = list(example.features.keys())
                feat_vals = list(example.features.values())
                fig_feat = go.Figure(go.Bar(
                    y=feat_names, x=feat_vals, orientation="h",
                    marker_color=[
                        C["cyan"] if v >= 0.5 else C["amber"] if v >= 0.2 else C["dim"]
                        for v in feat_vals
                    ],
                    text=[f"{v:.3f}" for v in feat_vals], textposition="outside",
                    textfont=dict(color=C["slate"], size=10, family="Inter, -apple-system, sans-serif"),
                ))
                fig_feat.update_layout(
                    **PLOTLY_LAYOUT, title_text="Feature Vector", height=400,
                    xaxis_range=[0, max(feat_vals) * 1.3 + 0.1] if feat_vals else [0, 1],
                )
                st.plotly_chart(fig_feat, use_container_width=True)

            # Evidence card
            ec = pair.evidence_card
            st.markdown(
                f'<span class="badge badge-{pair.decision}">{pair.decision}</span> '
                f'&nbsp; <span style="font-family:JetBrains Mono;color:{C["cyan"]}">{pair.score:.4f}</span>',
                unsafe_allow_html=True,
            )
            bar_color = C["emerald"] if pair.decision == "match" else C["rose"] if pair.decision == "non_match" else C["amber"]
            st.markdown(
                f'<div class="conf-bar-bg"><div class="conf-bar-fill" '
                f'style="width:{pair.score*100:.0f}%;background:{bar_color}"></div></div>',
                unsafe_allow_html=True,
            )

            sig_col1, sig_col2 = st.columns(2)
            with sig_col1:
                if ec.supporting_signals:
                    st.markdown('<span class="signal-supporting">Supporting</span>', unsafe_allow_html=True)
                    for sig in ec.supporting_signals:
                        st.markdown(f"- {sig.description}")
            with sig_col2:
                if ec.contradicting_signals:
                    st.markdown('<span class="signal-contradicting">Contradicting</span>', unsafe_allow_html=True)
                    for sig in ec.contradicting_signals:
                        st.markdown(f"- {sig.description}")

            if ec.reason_codes:
                pills = "".join(f'<span class="pill pill-{pair.decision}">{escape(rc)}</span>' for rc in ec.reason_codes)
                st.markdown(pills, unsafe_allow_html=True)

            features = example.features if example else {}
            counterfactuals = []
            if features:
                if features.get("shared_domain_count", 0) == 0:
                    counterfactuals.append("No shared domain evidence")
                if features.get("org_overlap_count", 0) == 0:
                    counterfactuals.append("No overlapping organizations")
                if features.get("location_conflict", 0) >= 1.0:
                    counterfactuals.append("Conflicting locations")
                if features.get("embedding_cosine", 0) < 0.2:
                    counterfactuals.append("Weak semantic similarity")
            if counterfactuals:
                st.markdown("**Counterfactual Panel**")
                for item in counterfactuals:
                    st.markdown(f"- {item}")


# ========================== TAB 4: EXPLAINER ==============================

with tab_explainer:
    st.markdown("#### How Entity Resolution Works")
    st.caption("A step-by-step walkthrough of the full pipeline on a real example pair.")

    # Pick an interesting pair to explain — find a match, abstain, and non-match
    example_pairs = {}
    for rp in data.resolved_pairs:
        if rp.decision not in example_pairs:
            example_pairs[rp.decision] = rp
        if len(example_pairs) >= 3:
            break

    explain_decision = st.radio(
        "Show example of:",
        [d for d in ["match", "abstain", "non_match"] if d in example_pairs],
        horizontal=True,
        key="explain_decision",
    )
    if explain_decision and explain_decision in example_pairs:
        ep = example_pairs[explain_decision]
        left_p = data.profile_lookup.get(ep.left_profile_id)
        right_p = data.profile_lookup.get(ep.right_profile_id)

        if left_p and right_p:
            # STEP 1: Raw profiles
            st.markdown(
                '<div class="step-indicator">'
                '<div class="step step-active">1. Raw Profiles</div>'
                '<div class="step">2. Blocking</div>'
                '<div class="step">3. Features</div>'
                '<div class="step">4. Scoring</div>'
                '<div class="step">5. Decision</div>'
                '</div>',
                unsafe_allow_html=True,
            )

            st.markdown("##### Step 1: Raw Profile Ingestion")
            st.markdown(
                "The system ingests public profiles from different sources. "
                "Each profile is parsed into a structured `ProfileRecord` with name, bio, orgs, topics, etc."
            )
            pcol1, pcol2 = st.columns(2)
            with pcol1:
                st.markdown(_render_profile_card(left_p), unsafe_allow_html=True)
            with pcol2:
                st.markdown(_render_profile_card(right_p), unsafe_allow_html=True)

            st.markdown('<hr class="subtle-divider">', unsafe_allow_html=True)

            # STEP 2: Blocking
            st.markdown(
                '<div class="step-indicator">'
                '<div class="step step-active">1. Raw Profiles</div>'
                '<div class="step step-active">2. Blocking</div>'
                '<div class="step">3. Features</div>'
                '<div class="step">4. Scoring</div>'
                '<div class="step">5. Decision</div>'
                '</div>',
                unsafe_allow_html=True,
            )
            st.markdown("##### Step 2: Candidate Blocking")
            ex_key = _pair_key(ep.left_profile_id, ep.right_profile_id)
            example = data.example_lookup.get(ex_key)
            if example and example.blocking_reasons:
                reasons_pills = "".join(
                    f'<span class="pill pill-violet">{escape(r)}</span>'
                    for r in example.blocking_reasons
                )
                muted = C["muted"]
                st.markdown(
                    f"This pair was generated by blocking rules: {reasons_pills}<br>"
                    f"<small style='color:{muted}'>Blocking reduces O(n^2) comparisons to a manageable set. "
                    f"Rules check name similarity, shared domains, org overlap, and TF-IDF embedding neighbors.</small>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown("This pair was identified as a candidate by the blocking stage.")

            st.markdown(
                f'<div class="card">'
                f'<strong>Corpus stats:</strong> {len(data.profiles)} profiles generate '
                f'{len(data.resolved_pairs)} candidate pairs '
                f'(vs {len(data.profiles) * (len(data.profiles) - 1) // 2} possible). '
                f'That is a <strong>{len(data.resolved_pairs) / max(len(data.profiles) * (len(data.profiles) - 1) // 2, 1) * 100:.1f}%</strong> '
                f'volume ratio.'
                f'</div>',
                unsafe_allow_html=True,
            )

            st.markdown('<hr class="subtle-divider">', unsafe_allow_html=True)

            # STEP 3: Features
            st.markdown(
                '<div class="step-indicator">'
                '<div class="step step-active">1. Raw Profiles</div>'
                '<div class="step step-active">2. Blocking</div>'
                '<div class="step step-active">3. Features</div>'
                '<div class="step">4. Scoring</div>'
                '<div class="step">5. Decision</div>'
                '</div>',
                unsafe_allow_html=True,
            )
            st.markdown("##### Step 3: Feature Extraction")
            st.markdown(
                "The `PairFeatureExtractor` computes **15 features** for each candidate pair across 4 categories:"
            )

            cat_col1, cat_col2, cat_col3, cat_col4 = st.columns(4)
            cat_col1.markdown(
                f'<div class="card"><strong style="color:{C["cyan"]}">Lexical</strong><br>'
                f'<small>name_similarity<br>alias_similarity<br>headline_similarity<br>bio_similarity<br>same_source_type</small></div>',
                unsafe_allow_html=True,
            )
            cat_col2.markdown(
                f'<div class="card"><strong style="color:{C["emerald"]}">Structured</strong><br>'
                f'<small>shared_domain_count<br>org_overlap_count<br>org_jaccard<br>topic_overlap_count<br>topic_jaccard</small></div>',
                unsafe_allow_html=True,
            )
            cat_col3.markdown(
                f'<div class="card"><strong style="color:{C["amber"]}">Contradiction</strong><br>'
                f'<small>location_overlap<br>location_conflict<br>temporal_overlap_count<br>temporal_distance</small></div>',
                unsafe_allow_html=True,
            )
            cat_col4.markdown(
                f'<div class="card"><strong style="color:{C["violet"]}">Semantic</strong><br>'
                f'<small>embedding_cosine<br>(TF-IDF cosine similarity)</small></div>',
                unsafe_allow_html=True,
            )

            if example:
                feat_names = list(example.features.keys())
                feat_vals = list(example.features.values())
                fig_feat = go.Figure(go.Bar(
                    y=feat_names, x=feat_vals, orientation="h",
                    marker_color=[
                        C["cyan"] if v >= 0.5 else C["amber"] if v >= 0.2 else C["dim"]
                        for v in feat_vals
                    ],
                    text=[f"{v:.3f}" for v in feat_vals], textposition="outside",
                    textfont=dict(color=C["slate"], size=10, family="Inter, -apple-system, sans-serif"),
                ))
                fig_feat.update_layout(
                    **PLOTLY_LAYOUT, title_text="Feature Vector for This Pair", height=400,
                    xaxis_range=[0, max(feat_vals) * 1.3 + 0.1] if feat_vals else [0, 1],
                )
                st.plotly_chart(fig_feat, use_container_width=True)

            st.markdown('<hr class="subtle-divider">', unsafe_allow_html=True)

            # STEP 4: Scoring
            st.markdown(
                '<div class="step-indicator">'
                '<div class="step step-active">1. Raw Profiles</div>'
                '<div class="step step-active">2. Blocking</div>'
                '<div class="step step-active">3. Features</div>'
                '<div class="step step-active">4. Scoring</div>'
                '<div class="step">5. Decision</div>'
                '</div>',
                unsafe_allow_html=True,
            )
            st.markdown("##### Step 4: Model Scoring & Calibration")
            st.markdown(
                "A calibrated logistic regression scores the 15-feature vector. "
                "Isotonic calibration maps raw probabilities to well-calibrated confidence scores."
            )
            sc1, sc2, sc3 = st.columns(3)
            sc1.metric("Raw Score", f"{ep.score:.4f}")
            sc2.metric("Match Threshold", f"{data.matcher.match_threshold:.4f}")
            sc3.metric("Non-match Threshold", f"{data.matcher.non_match_threshold:.4f}")

            score_position = "above match threshold" if ep.score >= data.matcher.match_threshold else \
                "below non-match threshold" if ep.score <= data.matcher.non_match_threshold else \
                "in the abstention band"
            st.markdown(
                f'Score **{ep.score:.4f}** is **{score_position}** '
                f'(match >= {data.matcher.match_threshold:.3f}, non-match <= {data.matcher.non_match_threshold:.3f})'
            )

            st.markdown('<hr class="subtle-divider">', unsafe_allow_html=True)

            # STEP 5: Decision
            st.markdown(
                '<div class="step-indicator">'
                '<div class="step step-active">1. Raw Profiles</div>'
                '<div class="step step-active">2. Blocking</div>'
                '<div class="step step-active">3. Features</div>'
                '<div class="step step-active">4. Scoring</div>'
                '<div class="step step-active">5. Decision</div>'
                '</div>',
                unsafe_allow_html=True,
            )
            st.markdown("##### Step 5: Decision & Evidence")
            st.markdown(
                f'<span class="badge badge-{ep.decision}">{ep.decision}</span> '
                f'<span style="font-family:JetBrains Mono;color:{C["cyan"]};margin-left:8px">{ep.score:.4f}</span>',
                unsafe_allow_html=True,
            )

            ec = ep.evidence_card
            if ep.decision == "abstain":
                st.markdown(
                    f"The model **abstains** when the score falls in the uncertainty band "
                    f"({data.matcher.non_match_threshold:.3f}–{data.matcher.match_threshold:.3f}) "
                    f"OR when a contradiction veto fires (high name similarity + conflicting structured evidence)."
                )
            elif ep.decision == "match":
                st.markdown("The model is **confident** this is the same person based on corroborating evidence.")
            else:
                st.markdown("The model is **confident** these are different people.")

            sig_col1, sig_col2 = st.columns(2)
            with sig_col1:
                if ec.supporting_signals:
                    st.markdown(f'<span class="signal-supporting">Supporting signals</span>', unsafe_allow_html=True)
                    for sig in ec.supporting_signals:
                        st.markdown(f"- {sig.description}")
            with sig_col2:
                if ec.contradicting_signals:
                    st.markdown(f'<span class="signal-contradicting">Contradicting signals</span>', unsafe_allow_html=True)
                    for sig in ec.contradicting_signals:
                        st.markdown(f"- {sig.description}")

            st.markdown(f"**Final explanation:** {ec.final_explanation}")


# ========================== TAB 5: WHAT-IF LAB =============================

with tab_whatif:
    st.markdown("#### What-If Comparator")
    st.caption("Pick any two profiles and see exactly how the pipeline scores them.")

    profile_names = {p.profile_id: p.display_name for p in data.profiles}
    profile_ids = list(profile_names.keys())

    wi_col1, wi_col2 = st.columns(2)
    with wi_col1:
        left_id = st.selectbox(
            "Profile A",
            profile_ids,
            format_func=lambda pid: f"{profile_names[pid]} ({data.profile_lookup[pid].source_type})",
            key="whatif_left",
        )
    with wi_col2:
        right_options = [pid for pid in profile_ids if pid != left_id]
        right_id = st.selectbox(
            "Profile B",
            right_options,
            format_func=lambda pid: f"{profile_names[pid]} ({data.profile_lookup[pid].source_type})",
            key="whatif_right",
        )

    if left_id and right_id:
        left_p = data.profile_lookup[left_id]
        right_p = data.profile_lookup[right_id]

        st.markdown("#### Alias Stress Controls")
        stress_col1, stress_col2, stress_col3, stress_col4 = st.columns(4)
        force_initials = stress_col1.checkbox("Use initials only", key="stress_initials")
        drop_orgs = stress_col2.checkbox("Drop org evidence", key="stress_orgs")
        drop_locations = stress_col3.checkbox("Drop location", key="stress_locs")
        keep_only_alias = stress_col4.checkbox("Alias-only profile", key="stress_alias")

        stressed_left = left_p
        if force_initials and len(stressed_left.display_name.split()) >= 2:
            parts = stressed_left.display_name.split()
            stressed_left = stressed_left.model_copy(update={"display_name": f"{parts[0][0]}. {parts[-1]}"})
        if keep_only_alias:
            alias = stressed_left.aliases[0] if stressed_left.aliases else stressed_left.display_name.split()[0]
            stressed_left = stressed_left.model_copy(update={"aliases": [alias], "headline": None})
        if drop_orgs:
            stressed_left = stressed_left.model_copy(update={"organizations": []})
        if drop_locations:
            stressed_left = stressed_left.model_copy(update={"locations": []})

        # Compute features live
        features = data.extractor.featurize_pair(stressed_left, right_p)

        # Score through the matcher
        from nyne_er_lab.features.dataset import PairExample
        live_example = PairExample(
            left_profile_id=left_id,
            right_profile_id=right_id,
            left_canonical_id=stressed_left.canonical_person_id or "unknown",
            right_canonical_id=right_p.canonical_person_id or "unknown",
            split="test",
            label=int(stressed_left.canonical_person_id == right_p.canonical_person_id) if stressed_left.canonical_person_id else 0,
            features=features,
            blocking_reasons=("whatif",),
        )
        scores, decisions = data.matcher.score_examples([live_example], data.extractor)
        score = scores[0]
        decision = decisions[0]

        # Ground truth
        same_person = (
            stressed_left.canonical_person_id
            and stressed_left.canonical_person_id == right_p.canonical_person_id
        )

        # Step-through visualization
        st.markdown(
            '<div class="step-indicator">'
            '<div class="step step-active">1. Profiles Selected</div>'
            '<div class="step step-active">2. Features Extracted</div>'
            '<div class="step step-active">3. Model Scored</div>'
            '<div class="step step-active">4. Decision Made</div>'
            '</div>',
            unsafe_allow_html=True,
        )

        # Result header
        result_col1, result_col2, result_col3 = st.columns(3)
        result_col1.metric("Decision", decision.upper())
        result_col2.metric("Score", f"{score:.4f}")
        if same_person is not None:
            result_col3.metric("Ground Truth", "Same Person" if same_person else "Different People")

        # Profiles side by side (compact)
        pcol1, pcol2 = st.columns(2)
        for col, profile in [(pcol1, stressed_left), (pcol2, right_p)]:
            with col:
                src_color = SOURCE_COLORS.get(profile.source_type, C["slate"])
                st.markdown(
                    f'<div class="card">'
                    f'<strong>{escape(profile.display_name)}</strong> '
                    f'<span class="pill" style="background:rgba({_hex_to_rgb(src_color)},0.15);color:{src_color}">'
                    f'{profile.source_type}</span><br>'
                    f'<small style="color:{C["muted"]}">{escape(profile.headline or "")}</small><br>'
                    f'<small style="color:{C["dim"]}">'
                    f'{", ".join(o.name for o in profile.organizations[:3]) if profile.organizations else "—"}'
                    f'</small>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        # Feature comparison — radar chart
        feat_names = list(features.keys())
        feat_vals = [min(v, 1.0) for v in features.values()]  # Clip for radar

        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=feat_vals + [feat_vals[0]],
            theta=feat_names + [feat_names[0]],
            fill="toself",
            fillcolor=f"rgba({_hex_to_rgb(C['cyan'])},0.15)",
            line=dict(color=C["cyan"], width=2),
            name="Feature Values",
        ))
        fig_radar.update_layout(
            **PLOTLY_LAYOUT,
            polar=dict(
                bgcolor="rgba(25,26,26,0)",
                radialaxis=dict(visible=True, range=[0, 1], gridcolor="#2F3336", tickfont=dict(size=10, color=C["muted"])),
                angularaxis=dict(gridcolor="#2F3336", tickfont=dict(size=11, color=C["text"])),
            ),
            title_text="Feature Fingerprint",
            height=450,
            showlegend=False,
        )
        st.plotly_chart(fig_radar, use_container_width=True)

        # Feature bar (detailed values)
        with st.expander("Detailed feature values"):
            fig_detail = go.Figure(go.Bar(
                y=feat_names, x=list(features.values()), orientation="h",
                marker_color=[C["cyan"] if v >= 0.5 else C["amber"] if v >= 0.2 else C["dim"] for v in features.values()],
                text=[f"{v:.4f}" for v in features.values()], textposition="outside",
                textfont=dict(size=10, color=C["slate"], family="Inter, -apple-system, sans-serif"),
            ))
            fig_detail.update_layout(
                **PLOTLY_LAYOUT, height=400,
            )
            st.plotly_chart(fig_detail, use_container_width=True)

        st.markdown("#### Counterfactual Panel")
        if features.get("shared_domain_count", 0) == 0:
            st.markdown("- No shared domain or outbound-link support")
        if features.get("org_overlap_count", 0) == 0:
            st.markdown("- No overlapping organization evidence")
        if features.get("location_conflict", 0) >= 1.0:
            st.markdown("- Structured locations actively conflict")
        if features.get("embedding_cosine", 0) < 0.2:
            st.markdown("- Semantic similarity is weak enough that the match is brittle")


# ========================== TAB 6: WEB SEARCH =============================

with tab_search:
    st.markdown(
        '<span class="live-dot"></span> <strong style="font-size:1.1rem">Search Trace</strong>',
        unsafe_allow_html=True,
    )
    st.caption(
        "Open-world retrieval is reported separately from pairwise benchmark metrics. "
        "Below is the benchmark retrieval trace, followed by optional live web search."
    )

    trace = data.search_trace
    t1, t2, t3 = st.columns(3)
    t1.metric("Recall@1", f"{trace.get('recall_at_1', 0.0):.3f}")
    t2.metric("Recall@3", f"{trace.get('recall_at_3', 0.0):.3f}")
    t3.metric("Recall@5", f"{trace.get('recall_at_5', 0.0):.3f}")

    for item in trace.get("traces", [])[:4]:
        with st.expander(f"Retrieval trace: {item['query_name']}", expanded=False):
            for candidate in item["top_candidates"]:
                color = C["emerald"] if candidate["same_person"] else C["muted"]
                st.markdown(
                    f'<div class="card" style="border-top: 2px solid {color}">'
                    f'<strong>{escape(candidate["display_name"])}</strong> '
                    f'<span class="pill">{candidate["source_type"]}</span> '
                    f'<span style="font-family:JetBrains Mono;color:{C["cyan"]};margin-left:8px">{candidate["score"]:.3f}</span><br>'
                    f'<small style="color:{C["muted"]}">same_person={candidate["same_person"]}</small>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

    st.markdown('<hr class="subtle-divider">', unsafe_allow_html=True)
    st.markdown("#### Optional Live Web Search")

    search_name = st.text_input(
        "Person name",
        placeholder="e.g. Andrej Karpathy, Chip Huyen, Yann LeCun",
        key="search_name",
    )
    search_col1, search_col2 = st.columns([1, 3])
    with search_col1:
        max_results = st.slider("Max results", 3, 12, 6, key="search_max")
    with search_col2:
        search_btn = st.button("Search & Resolve", key="search_btn", type="primary")

    if search_btn and search_name.strip():
        from nyne_er_lab.search import search_person
        from nyne_er_lab.live import profile_from_text, resolve_live_profile

        with st.spinner(f"Searching the web for '{search_name.strip()}'..."):
            hits = search_person(search_name.strip(), max_results=max_results)

        if not hits:
            st.warning(
                "No search results found. DuckDuckGo may be rate-limiting. "
                "Try again in a moment or use the Live Resolver tab with manual input."
            )
        else:
            st.markdown(f"#### Found {len(hits)} results")

            for i, hit in enumerate(hits):
                src_color = SOURCE_COLORS.get(hit.source_type_guess, C["slate"])
                with st.expander(
                    f"{hit.title[:80]} — {hit.source_type_guess}",
                    expanded=(i == 0),
                ):
                    st.markdown(
                        f'<span class="pill" style="background:rgba({_hex_to_rgb(src_color)},0.15);color:{src_color}">'
                        f'{hit.source_type_guess}</span> '
                        f'<a href="{escape(hit.url)}" style="color:{C["cyan"]}">{escape(hit.url[:80])}</a>',
                        unsafe_allow_html=True,
                    )
                    muted_color = C["muted"]
                    st.markdown(f"<small style='color:{muted_color}'>{escape(hit.snippet[:300])}</small>", unsafe_allow_html=True)

                    if st.button(f"Resolve this result", key=f"resolve_hit_{i}"):
                        with st.spinner("Building profile and resolving..."):
                            query_profile = profile_from_text(
                                display_name=search_name.strip(),
                                bio=hit.snippet or f"Profile of {search_name}.",
                                source_type=hit.source_type_guess,
                                headline=hit.title[:120] if hit.title else "",
                                url=hit.url,
                            )
                            result = resolve_live_profile(
                                query_profile, data.profiles, data.extractor,
                                data.matcher, data.identities,
                            )
                        _render_live_results(result, data.profile_lookup)


# ========================== TAB 7: LIVE RESOLVER ===========================

with tab_live:
    st.markdown(
        '<span class="live-dot"></span> <strong style="font-size:1.1rem">Live Resolver</strong>',
        unsafe_allow_html=True,
    )
    st.caption("Enter a profile and resolve it against all known identities in real-time.")

    input_mode = st.radio(
        "Input method", ["Text fields", "Paste URL"],
        horizontal=True, key="live_input_mode",
    )

    if input_mode == "Text fields":
        lc1, lc2 = st.columns(2)
        with lc1:
            live_name = st.text_input("Display name", placeholder="e.g. Jane Smith", key="live_name")
            live_headline = st.text_input("Headline", placeholder="e.g. ML Engineer at Acme", key="live_headline")
            live_orgs = st.text_input("Organizations (comma-separated)", placeholder="e.g. Google, Stanford", key="live_orgs")
        with lc2:
            live_bio = st.text_area("Bio / About", placeholder="Paste bio text here...", height=100, key="live_bio")
            live_locations = st.text_input("Locations (comma-separated)", placeholder="e.g. San Francisco, CA", key="live_locs")
            live_topics = st.text_input("Topics (comma-separated)", placeholder="e.g. NLP, transformers, MLOps", key="live_topics")

        if st.button("Resolve", key="live_resolve_btn", type="primary"):
            if not live_name.strip():
                st.warning("Please enter a display name.")
            else:
                from nyne_er_lab.live import profile_from_text, resolve_live_profile

                with st.spinner("Resolving against known identities..."):
                    query_profile = profile_from_text(
                        display_name=live_name.strip(),
                        bio=live_bio.strip() or f"{live_name} professional profile.",
                        headline=live_headline.strip(),
                        organizations=[o.strip() for o in live_orgs.split(",") if o.strip()] if live_orgs else [],
                        locations=[l.strip() for l in live_locations.split(",") if l.strip()] if live_locations else [],
                        topics=[t.strip() for t in live_topics.split(",") if t.strip()] if live_topics else [],
                    )
                    result = resolve_live_profile(
                        query_profile, data.profiles, data.extractor, data.matcher, data.identities,
                    )

                _render_live_results(result, data.profile_lookup)

    else:  # Paste URL
        url_col1, url_col2 = st.columns([3, 1])
        with url_col1:
            live_url = st.text_input("URL", placeholder="https://github.com/username", key="live_url")
        with url_col2:
            live_source_type = st.selectbox(
                "Source type",
                ["github", "personal_site", "conference_bio", "company_profile", "podcast_guest", "huggingface"],
                key="live_source_type",
            )
        live_name_hint = st.text_input("Name hint (optional, helps if parsing fails)", key="live_name_hint")

        if st.button("Fetch & Resolve", key="live_url_resolve_btn", type="primary"):
            if not live_url.strip():
                st.warning("Please enter a URL.")
            else:
                from nyne_er_lab.live import profile_from_url, profile_from_text, resolve_live_profile

                with st.spinner("Fetching URL and resolving..."):
                    query_profile = profile_from_url(
                        live_url.strip(), live_source_type,
                        display_name_hint=live_name_hint.strip() or None,
                    )
                    if query_profile is None:
                        st.error(
                            "Could not parse the URL. The page may not match the expected source type format. "
                            "Try using 'Text fields' mode instead."
                        )
                    else:
                        result = resolve_live_profile(
                            query_profile, data.profiles, data.extractor, data.matcher, data.identities,
                        )
                        _render_live_results(result, data.profile_lookup)


# ========================== TAB 6: PIPELINE INSPECTOR =====================

with tab_inspector:
    st.markdown("#### Leakage Checks")
    for item in data.leakage_checks:
        color = C["emerald"] if item["passed"] else C["rose"]
        status = "PASS" if item["passed"] else "FAIL"
        st.markdown(
            f'<div class="card" style="border-top: 2px solid {color}">'
            f'<strong>{status}</strong> · {escape(item["name"])}<br>'
            f'<small style="color:{C["muted"]}">{escape(item["detail"])}</small>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown('<hr class="subtle-divider">', unsafe_allow_html=True)

    col_fi, col_cc = st.columns(2)

    with col_fi:
        fi_names = [name for name, _ in data.feature_importance]
        fi_coefs = [coef for _, coef in data.feature_importance]
        fig_fi = go.Figure(go.Bar(
            y=fi_names, x=fi_coefs, orientation="h",
            marker_color=[C["cyan"] if c >= 0 else C["rose"] for c in fi_coefs],
            text=[f"{v:+.3f}" for v in fi_coefs], textposition="outside",
            textfont=dict(color=C["slate"], size=10, family="Inter, -apple-system, sans-serif"),
        ))
        fig_fi.update_layout(
            **PLOTLY_LAYOUT, title_text="Feature Importance (LR Coefficients)", height=450,
        )
        st.plotly_chart(fig_fi, use_container_width=True)

    with col_cc:
        test_labels_arr = np.array(data.test_labels)
        test_scores_arr = np.array(data.test_scores)
        raw_scores_arr = np.array(data.hybrid_run.raw_scores)

        if len(set(data.test_labels)) > 1:
            n_bins = min(10, max(3, len(data.test_labels) // 3))
            frac_pos_cal, mean_pred_cal = calibration_curve(test_labels_arr, test_scores_arr, n_bins=n_bins)
            frac_pos_raw, mean_pred_raw = calibration_curve(test_labels_arr, raw_scores_arr, n_bins=n_bins)

            fig_cal = go.Figure()
            fig_cal.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1], mode="lines",
                line=dict(dash="dash", color=C["dim"]), name="Perfect",
            ))
            fig_cal.add_trace(go.Scatter(
                x=mean_pred_raw.tolist(), y=frac_pos_raw.tolist(),
                mode="lines+markers",
                name=f"Raw (Brier={data.hybrid_run.raw_brier:.4f})",
                line=dict(color=C["amber"]),
            ))
            fig_cal.add_trace(go.Scatter(
                x=mean_pred_cal.tolist(), y=frac_pos_cal.tolist(),
                mode="lines+markers",
                name=f"Calibrated (Brier={data.hybrid_run.calibrated_brier:.4f})",
                line=dict(color=C["cyan"]),
            ))
            fig_cal.update_layout(
                **PLOTLY_LAYOUT, title_text="Calibration Curve",
                xaxis_title="Mean predicted probability",
                yaxis_title="Fraction of positives",
                height=450,
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
            )
            st.plotly_chart(fig_cal, use_container_width=True)
        else:
            st.info("Cannot plot calibration curve — test set has only one class.")

    st.markdown('<hr class="subtle-divider">', unsafe_allow_html=True)

    # Blocking rule effectiveness
    if data.blocking_stats:
        rule_names = list(data.blocking_stats.keys())
        rule_counts = [data.blocking_stats[r]["count"] for r in rule_names]
        rule_tp = [data.blocking_stats[r]["true_positives"] for r in rule_names]
        rule_prec = [data.blocking_stats[r]["precision"] for r in rule_names]

        fig_block = go.Figure()
        fig_block.add_trace(go.Bar(
            name="Candidates", x=rule_names, y=rule_counts,
            marker_color=C["dim"],
        ))
        fig_block.add_trace(go.Bar(
            name="True Positives", x=rule_names, y=rule_tp,
            marker_color=C["emerald"],
        ))
        fig_block.update_layout(
            **PLOTLY_LAYOUT, barmode="group", title_text="Blocking Rule Effectiveness",
        )
        st.plotly_chart(fig_block, use_container_width=True)

        prec_cols = st.columns(len(rule_names))
        for i, rule in enumerate(rule_names):
            prec_cols[i].metric(f"{rule} prec.", f"{rule_prec[i]:.2f}")

    st.markdown('<hr class="subtle-divider">', unsafe_allow_html=True)

    # Threshold explorer
    st.markdown("#### Interactive Threshold Explorer")
    threshold_val = st.slider(
        "Decision threshold", 0.0, 1.0,
        value=float(data.matcher.threshold), step=0.01,
        key="threshold_slider",
    )

    sweep = threshold_sweep(data.test_labels, data.test_scores, n_points=80)
    if sweep:
        sweep_t = [s["threshold"] for s in sweep]
        sweep_p = [s["precision"] for s in sweep]
        sweep_r = [s["recall"] for s in sweep]
        sweep_f = [s["f1"] for s in sweep]

        fig_pr = go.Figure()
        fig_pr.add_trace(go.Scatter(x=sweep_t, y=sweep_p, name="Precision", line=dict(color=C["cyan"], width=2)))
        fig_pr.add_trace(go.Scatter(x=sweep_t, y=sweep_r, name="Recall", line=dict(color=C["amber"], width=2)))
        fig_pr.add_trace(go.Scatter(x=sweep_t, y=sweep_f, name="F1", line=dict(color=C["emerald"], width=2)))
        fig_pr.add_vline(x=threshold_val, line_dash="dash", line_color=C["rose"],
                         annotation_text="threshold", annotation_font_color=C["rose"],
                         annotation_font=dict(family="Inter, -apple-system, sans-serif", size=10))
        fig_pr.update_layout(
            **PLOTLY_LAYOUT, title_text="Precision / Recall / F1 vs Threshold",
            xaxis_title="Threshold",
            yaxis_range=[0, 1.05],
            height=400,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_pr, use_container_width=True)

    at_threshold = summarize_predictions(data.test_labels, data.test_scores, threshold_val)
    tc1, tc2, tc3 = st.columns(3)
    tc1.metric("Precision @ threshold", f"{at_threshold.precision:.3f}")
    tc2.metric("Recall @ threshold", f"{at_threshold.recall:.3f}")
    tc3.metric("F1 @ threshold", f"{at_threshold.f1:.3f}")

    # Score histograms by true label
    st.markdown('<hr class="subtle-divider">', unsafe_allow_html=True)
    st.markdown("#### Score Distribution by True Label")

    fig_hist = go.Figure()
    for label, color, name in [(1, C["emerald"], "Positive (same person)"), (0, C["rose"], "Negative (different)")]:
        if data.score_by_label.get(label):
            fig_hist.add_trace(go.Histogram(
                x=data.score_by_label[label],
                name=name, marker_color=color, opacity=0.6,
                nbinsx=20,
            ))
    fig_hist.add_vline(x=threshold_val, line_dash="dash", line_color=C["amber"],
                       annotation_text="threshold", annotation_font_color=C["amber"],
                       annotation_font=dict(family="Inter, -apple-system, sans-serif", size=10))
    fig_hist.update_layout(
        **PLOTLY_LAYOUT, barmode="overlay",
        title_text="How well does the model separate classes?",
        xaxis_title="Calibrated Score", yaxis_title="Count",
        height=350,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_hist, use_container_width=True)
