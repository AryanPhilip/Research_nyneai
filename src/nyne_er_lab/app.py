"""Entity Resolution Dashboard — interactive Streamlit frontend."""

from __future__ import annotations

import streamlit as st
import plotly.graph_objects as go
from sklearn.calibration import calibration_curve

from nyne_er_lab.app_data import DashboardData, load_dashboard_data, _pair_key
from nyne_er_lab.eval.metrics import summarize_predictions, threshold_sweep

# ---------------------------------------------------------------------------
# Page config & theme CSS
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Entity Resolution Lab",
    page_icon="🔗",
    layout="wide",
    initial_sidebar_state="collapsed",
)

THEME_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

.stApp {
    background-color: #0a0f1a;
    font-family: 'Inter', sans-serif;
}

/* Header */
.stApp header { background-color: #0a0f1a !important; }

/* Main container text */
.stApp, .stApp p, .stApp li, .stApp span, .stApp label {
    color: #e2e8f0;
}

/* Muted text */
.stApp .stCaption, .eyebrow-text {
    color: #94a3b8 !important;
}

/* Metric cards */
[data-testid="stMetric"] {
    background: #111827;
    border: 1px solid #1e293b;
    border-radius: 12px;
    padding: 16px 20px;
}
[data-testid="stMetricLabel"] {
    color: #94a3b8 !important;
}
[data-testid="stMetricValue"] {
    color: #06b6d4 !important;
    font-weight: 700;
}
[data-testid="stMetricDelta"] > div {
    color: #10b981 !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 0;
    background: #111827;
    border-radius: 12px;
    padding: 4px;
    border: 1px solid #1e293b;
}
.stTabs [data-baseweb="tab"] {
    color: #94a3b8;
    border-radius: 8px;
    padding: 8px 20px;
    font-weight: 500;
}
.stTabs [aria-selected="true"] {
    background: #1e293b !important;
    color: #06b6d4 !important;
}

/* Expanders */
.streamlit-expanderHeader {
    background: #111827 !important;
    border: 1px solid #1e293b !important;
    border-radius: 12px !important;
    color: #e2e8f0 !important;
}
details[data-testid="stExpander"] {
    background: #111827;
    border: 1px solid #1e293b;
    border-radius: 12px;
}

/* Selectbox & slider */
[data-baseweb="select"] { background: #111827; }
[data-baseweb="select"] * { color: #e2e8f0 !important; }

/* Card helper */
.card {
    background: #111827;
    border: 1px solid #1e293b;
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 12px;
}

/* Pills */
.pill {
    display: inline-block;
    background: #1e293b;
    color: #06b6d4;
    padding: 2px 10px;
    border-radius: 999px;
    font-size: 12px;
    margin: 2px 4px 2px 0;
    font-weight: 500;
}
.pill-match { background: #064e3b; color: #10b981; }
.pill-non_match { background: #4c0519; color: #f43f5e; }
.pill-abstain { background: #451a03; color: #f59e0b; }

/* Decision badges */
.badge {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 999px;
    font-size: 13px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
.badge-match { background: #064e3b; color: #10b981; }
.badge-non_match { background: #4c0519; color: #f43f5e; }
.badge-abstain { background: #451a03; color: #f59e0b; }
.badge-high { background: #064e3b; color: #10b981; }
.badge-medium { background: #1e293b; color: #06b6d4; }
.badge-low { background: #451a03; color: #f59e0b; }
.badge-uncertain { background: #4c0519; color: #f43f5e; }

/* Confidence bar */
.conf-bar-bg {
    width: 100%;
    height: 8px;
    background: #1e293b;
    border-radius: 999px;
    overflow: hidden;
    margin: 6px 0;
}
.conf-bar-fill {
    height: 100%;
    border-radius: 999px;
}

/* Signal lists */
.signal-supporting { color: #10b981; }
.signal-contradicting { color: #f43f5e; }

h1, h2, h3 { color: #e2e8f0 !important; }
</style>
"""

st.markdown(THEME_CSS, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Plotly theme defaults
# ---------------------------------------------------------------------------

PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="#111827",
    plot_bgcolor="#111827",
    font=dict(family="Inter, sans-serif", color="#e2e8f0"),
    margin=dict(l=40, r=20, t=40, b=40),
)

COLORS = {
    "cyan": "#06b6d4",
    "emerald": "#10b981",
    "rose": "#f43f5e",
    "amber": "#f59e0b",
    "slate": "#94a3b8",
}

MODEL_COLORS = ["#94a3b8", "#64748b", "#06b6d4", "#10b981"]

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

data: DashboardData = load_dashboard_data()

# ---------------------------------------------------------------------------
# Title
# ---------------------------------------------------------------------------

st.markdown("## Entity Resolution Lab")
st.caption("Benchmark-first entity resolution for fragmented public professional identities")

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_overview, tab_pairs, tab_clusters, tab_inspector = st.tabs([
    "Overview", "Pairwise Explorer", "Identity Clusters", "Pipeline Inspector",
])

# ========================== TAB 1: OVERVIEW ================================

with tab_overview:
    # KPI row
    hybrid_metrics = next(m for m in data.metrics if m["name"] == "hybrid")
    best_baseline_f1 = max(m["f1"] for m in data.metrics if m["name"] != "hybrid")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Hybrid F1", f"{hybrid_metrics['f1']:.3f}", delta=f"+{hybrid_metrics['f1'] - best_baseline_f1:.3f} vs best baseline")
    c2.metric("Precision", f"{hybrid_metrics['precision']:.3f}")
    c3.metric("Recall", f"{hybrid_metrics['recall']:.3f}")
    c4.metric("Cluster B³ F1", f"{data.cluster_f1:.3f}")

    st.markdown("---")

    # Model comparison bar chart
    model_names = [m["name"] for m in data.metrics]
    metric_keys = ["precision", "recall", "f1", "average_precision"]
    metric_labels = ["Precision", "Recall", "F1", "Avg Precision"]

    fig_models = go.Figure()
    bar_colors = [COLORS["slate"], COLORS["rose"], COLORS["cyan"], COLORS["emerald"]]
    for i, (key, label) in enumerate(zip(metric_keys, metric_labels)):
        fig_models.add_trace(go.Bar(
            name=label,
            x=model_names,
            y=[m[key] for m in data.metrics],
            marker_color=bar_colors[i],
        ))
    fig_models.update_layout(
        **PLOTLY_LAYOUT,
        barmode="group",
        title="Model Comparison",
        yaxis=dict(range=[0, 1.05]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_models, use_container_width=True)

    # Ablation + calibration row
    col_abl, col_cal = st.columns(2)

    with col_abl:
        abl_names = [a["name"] for a in data.ablations]
        abl_f1 = [a["f1"] for a in data.ablations]
        fig_abl = go.Figure(go.Bar(
            y=abl_names,
            x=abl_f1,
            orientation="h",
            marker_color=[COLORS["emerald"] if n == "full" else COLORS["amber"] for n in abl_names],
            text=[f"{v:.3f}" for v in abl_f1],
            textposition="outside",
        ))
        fig_abl.update_layout(
            **PLOTLY_LAYOUT,
            title="Feature Ablation (F1)",
            xaxis=dict(range=[0, 1.1]),
        )
        st.plotly_chart(fig_abl, use_container_width=True)

    with col_cal:
        cs = data.confusion_slices
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### Calibration & Abstention")
        mc1, mc2 = st.columns(2)
        mc1.metric("Accepted Precision", f"{cs['accepted_precision']:.3f}")
        mc2.metric("Accepted Recall", f"{cs['accepted_recall']:.3f}")
        mc3, mc4 = st.columns(2)
        mc3.metric("Abstain Rate", f"{cs['abstain_rate']:.3f}")
        mc4.metric("Brier (calibrated)", f"{data.hybrid_run.calibrated_brier:.4f}")
        st.markdown(
            f"Raw Brier score: **{data.hybrid_run.raw_brier:.4f}** → "
            f"Calibrated: **{data.hybrid_run.calibrated_brier:.4f}**"
        )
        st.markdown('</div>', unsafe_allow_html=True)

    st.caption(
        f"{len(data.profiles)} profiles  |  "
        f"{len(set(p.source_type for p in data.profiles))} source types  |  "
        f"15 features  |  {len(data.metrics)} models benchmarked"
    )


# ========================== TAB 2: PAIRWISE EXPLORER =======================

with tab_pairs:
    col_filter, col_detail = st.columns([1, 3])

    with col_filter:
        decision_filter = st.selectbox(
            "Decision type",
            ["All", "match", "non_match", "abstain"],
            key="pair_decision_filter",
        )
        conf_range = st.slider(
            "Confidence range",
            0.0, 1.0, (0.0, 1.0),
            step=0.01,
            key="pair_conf_range",
        )

        filtered_pairs = [
            p for p in data.resolved_pairs
            if (decision_filter == "All" or p.decision == decision_filter)
            and conf_range[0] <= p.score <= conf_range[1]
        ]
        st.markdown(f"**{len(filtered_pairs)}** pairs")

    with col_detail:
        if not filtered_pairs:
            st.info("No pairs match the current filters.")
        else:
            pair_labels = [
                f"{p.left_profile_id.split('-')[0]}… vs {p.right_profile_id.split('-')[0]}… ({p.decision} @ {p.score:.2f})"
                for p in filtered_pairs
            ]
            # Build nicer labels using profile names
            nice_labels = []
            for p in filtered_pairs:
                left_name = data.profile_lookup.get(p.left_profile_id)
                right_name = data.profile_lookup.get(p.right_profile_id)
                ln = left_name.display_name if left_name else p.left_profile_id[:12]
                rn = right_name.display_name if right_name else p.right_profile_id[:12]
                nice_labels.append(f"{ln} vs {rn} ({p.decision} @ {p.score:.2f})")

            selected_idx = st.selectbox(
                "Select pair",
                range(len(filtered_pairs)),
                format_func=lambda i: nice_labels[i],
                key="pair_selector",
            )
            pair = filtered_pairs[selected_idx]
            left_profile = data.profile_lookup.get(pair.left_profile_id)
            right_profile = data.profile_lookup.get(pair.right_profile_id)

            # Side-by-side profiles
            pcol1, pcol2 = st.columns(2)
            for col, profile in [(pcol1, left_profile), (pcol2, right_profile)]:
                if profile is None:
                    col.warning("Profile not found")
                    continue
                with col:
                    st.markdown(f"**{profile.display_name}**")
                    st.markdown(f'<span class="pill">{profile.source_type}</span>', unsafe_allow_html=True)
                    if profile.headline:
                        st.markdown(f"*{profile.headline}*")
                    with st.expander("Bio"):
                        st.write(profile.bio_text[:500] + ("..." if len(profile.bio_text) > 500 else ""))
                    if profile.organizations:
                        st.markdown("**Orgs:** " + ", ".join(o.name for o in profile.organizations))
                    if profile.locations:
                        st.markdown("**Locations:** " + ", ".join(profile.locations))
                    if profile.topics:
                        pills = "".join(f'<span class="pill">{t}</span>' for t in profile.topics[:8])
                        st.markdown(pills, unsafe_allow_html=True)
                    st.markdown(f"[Source]({profile.url})")

            st.markdown("---")

            # Feature vector chart
            ex_key = _pair_key(pair.left_profile_id, pair.right_profile_id)
            example = data.example_lookup.get(ex_key)
            if example:
                feat_names = list(example.features.keys())
                feat_vals = list(example.features.values())
                fig_feat = go.Figure(go.Bar(
                    y=feat_names,
                    x=feat_vals,
                    orientation="h",
                    marker_color=[
                        COLORS["cyan"] if v >= 0.5 else COLORS["amber"] if v >= 0.2 else COLORS["slate"]
                        for v in feat_vals
                    ],
                    text=[f"{v:.3f}" for v in feat_vals],
                    textposition="outside",
                ))
                fig_feat.update_layout(
                    **PLOTLY_LAYOUT,
                    title="Feature Vector",
                    height=400,
                    xaxis=dict(range=[0, max(feat_vals) * 1.2 + 0.1] if feat_vals else [0, 1]),
                )
                st.plotly_chart(fig_feat, use_container_width=True)

            st.markdown("---")

            # Evidence card
            ec = pair.evidence_card
            decision_class = pair.decision
            st.markdown(
                f'<span class="badge badge-{decision_class}">{pair.decision}</span> '
                f'&nbsp; confidence: **{pair.score:.3f}**',
                unsafe_allow_html=True,
            )
            bar_color = (
                COLORS["emerald"] if pair.decision == "match"
                else COLORS["rose"] if pair.decision == "non_match"
                else COLORS["amber"]
            )
            st.markdown(
                f'<div class="conf-bar-bg"><div class="conf-bar-fill" '
                f'style="width:{pair.score*100:.0f}%;background:{bar_color}"></div></div>',
                unsafe_allow_html=True,
            )

            if ec.supporting_signals:
                st.markdown('<span class="signal-supporting">Supporting signals:</span>', unsafe_allow_html=True)
                for sig in ec.supporting_signals:
                    st.markdown(f"- {sig.description} (weight: {sig.weight:.2f})")
            if ec.contradicting_signals:
                st.markdown('<span class="signal-contradicting">Contradicting signals:</span>', unsafe_allow_html=True)
                for sig in ec.contradicting_signals:
                    st.markdown(f"- {sig.description} (weight: {sig.weight:.2f})")
            if ec.reason_codes:
                pills = "".join(f'<span class="pill pill-{decision_class}">{rc}</span>' for rc in ec.reason_codes)
                st.markdown(pills, unsafe_allow_html=True)
            st.markdown(f"**Explanation:** {ec.final_explanation}")


# ========================== TAB 3: IDENTITY CLUSTERS =======================

with tab_clusters:
    identities = data.identities
    multi_member = [i for i in identities if len(i.member_profile_ids) > 1]
    singletons = [i for i in identities if len(i.member_profile_ids) == 1]
    avg_size = sum(len(i.member_profile_ids) for i in identities) / max(len(identities), 1)

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Identities", len(identities))
    c2.metric("Multi-member", len(multi_member))
    c3.metric("Avg Cluster Size", f"{avg_size:.1f}")

    # Confidence band donut
    from collections import Counter
    band_counts = Counter(i.confidence_band for i in identities)
    band_colors = {
        "high": COLORS["emerald"],
        "medium": COLORS["cyan"],
        "low": COLORS["amber"],
        "uncertain": COLORS["rose"],
    }
    fig_donut = go.Figure(go.Pie(
        labels=list(band_counts.keys()),
        values=list(band_counts.values()),
        hole=0.55,
        marker=dict(colors=[band_colors.get(b, COLORS["slate"]) for b in band_counts.keys()]),
        textinfo="label+value",
    ))
    fig_donut.update_layout(
        **PLOTLY_LAYOUT,
        title="Confidence Band Distribution",
        height=350,
        showlegend=False,
    )
    st.plotly_chart(fig_donut, use_container_width=True)

    st.markdown("---")

    # Identity expanders
    for identity in identities:
        badge_class = identity.confidence_band
        header = (
            f"{identity.canonical_name} — "
            f"{len(identity.member_profile_ids)} profile(s) — "
            f"{identity.confidence_band}"
        )
        with st.expander(header):
            st.markdown(
                f'<span class="badge badge-{badge_class}">{identity.confidence_band}</span>',
                unsafe_allow_html=True,
            )
            st.write(identity.summary)
            if identity.key_orgs:
                pills = "".join(f'<span class="pill">{o}</span>' for o in identity.key_orgs)
                st.markdown(f"**Key orgs:** {pills}", unsafe_allow_html=True)
            if identity.key_links:
                for link in identity.key_links:
                    st.markdown(f"- [{link}]({link})")

            # Member mini-cards
            cols = st.columns(min(len(identity.member_profile_ids), 3))
            for idx, pid in enumerate(identity.member_profile_ids):
                profile = data.profile_lookup.get(pid)
                if profile is None:
                    continue
                with cols[idx % len(cols)]:
                    st.markdown(
                        f'<div class="card">'
                        f'<strong>{profile.display_name}</strong><br>'
                        f'<span class="pill">{profile.source_type}</span><br>'
                        f'<em>{(profile.headline or "")[:80]}</em>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )


# ========================== TAB 4: PIPELINE INSPECTOR =====================

with tab_inspector:
    col_fi, col_cc = st.columns(2)

    # Feature importance
    with col_fi:
        fi_names = [name for name, _ in data.feature_importance]
        fi_coefs = [coef for _, coef in data.feature_importance]
        fig_fi = go.Figure(go.Bar(
            y=fi_names,
            x=fi_coefs,
            orientation="h",
            marker_color=[COLORS["cyan"] if c >= 0 else COLORS["rose"] for c in fi_coefs],
            text=[f"{v:+.3f}" for v in fi_coefs],
            textposition="outside",
        ))
        fig_fi.update_layout(
            **PLOTLY_LAYOUT,
            title="Feature Importance (LR Coefficients)",
            height=450,
        )
        st.plotly_chart(fig_fi, use_container_width=True)

    # Calibration curve
    with col_cc:
        import numpy as np

        test_labels_arr = np.array(data.test_labels)
        test_scores_arr = np.array(data.test_scores)
        raw_scores_arr = np.array(data.hybrid_run.raw_scores)

        # Only plot if we have both classes
        if len(set(data.test_labels)) > 1:
            n_bins = min(10, max(3, len(data.test_labels) // 3))
            frac_pos_cal, mean_pred_cal = calibration_curve(test_labels_arr, test_scores_arr, n_bins=n_bins)
            frac_pos_raw, mean_pred_raw = calibration_curve(test_labels_arr, raw_scores_arr, n_bins=n_bins)

            fig_cal = go.Figure()
            fig_cal.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1],
                mode="lines",
                line=dict(dash="dash", color=COLORS["slate"]),
                name="Perfect",
            ))
            fig_cal.add_trace(go.Scatter(
                x=mean_pred_raw.tolist(), y=frac_pos_raw.tolist(),
                mode="lines+markers",
                name=f"Raw (Brier={data.hybrid_run.raw_brier:.4f})",
                line=dict(color=COLORS["amber"]),
            ))
            fig_cal.add_trace(go.Scatter(
                x=mean_pred_cal.tolist(), y=frac_pos_cal.tolist(),
                mode="lines+markers",
                name=f"Calibrated (Brier={data.hybrid_run.calibrated_brier:.4f})",
                line=dict(color=COLORS["cyan"]),
            ))
            fig_cal.update_layout(
                **PLOTLY_LAYOUT,
                title="Calibration Curve",
                xaxis_title="Mean predicted probability",
                yaxis_title="Fraction of positives",
                height=450,
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
            )
            st.plotly_chart(fig_cal, use_container_width=True)
        else:
            st.info("Cannot plot calibration curve — test set has only one class.")

    st.markdown("---")

    # Blocking rule effectiveness
    if data.blocking_stats:
        rule_names = list(data.blocking_stats.keys())
        rule_counts = [data.blocking_stats[r]["count"] for r in rule_names]
        rule_tp = [data.blocking_stats[r]["true_positives"] for r in rule_names]
        rule_prec = [data.blocking_stats[r]["precision"] for r in rule_names]

        fig_block = go.Figure()
        fig_block.add_trace(go.Bar(
            name="Candidates",
            x=rule_names,
            y=rule_counts,
            marker_color=COLORS["slate"],
        ))
        fig_block.add_trace(go.Bar(
            name="True Positives",
            x=rule_names,
            y=rule_tp,
            marker_color=COLORS["emerald"],
        ))
        fig_block.update_layout(
            **PLOTLY_LAYOUT,
            barmode="group",
            title="Blocking Rule Effectiveness",
        )
        st.plotly_chart(fig_block, use_container_width=True)

        # Rule precision display
        prec_cols = st.columns(len(rule_names))
        for i, rule in enumerate(rule_names):
            prec_cols[i].metric(f"{rule} precision", f"{rule_prec[i]:.2f}")

    st.markdown("---")

    # Interactive threshold slider
    st.markdown("#### Interactive Threshold Explorer")
    threshold_val = st.slider(
        "Decision threshold",
        0.0, 1.0,
        value=float(data.matcher.threshold),
        step=0.01,
        key="threshold_slider",
    )

    sweep = threshold_sweep(data.test_labels, data.test_scores, n_points=80)
    if sweep:
        sweep_t = [s["threshold"] for s in sweep]
        sweep_p = [s["precision"] for s in sweep]
        sweep_r = [s["recall"] for s in sweep]
        sweep_f = [s["f1"] for s in sweep]

        fig_pr = go.Figure()
        fig_pr.add_trace(go.Scatter(x=sweep_t, y=sweep_p, name="Precision", line=dict(color=COLORS["cyan"])))
        fig_pr.add_trace(go.Scatter(x=sweep_t, y=sweep_r, name="Recall", line=dict(color=COLORS["amber"])))
        fig_pr.add_trace(go.Scatter(x=sweep_t, y=sweep_f, name="F1", line=dict(color=COLORS["emerald"])))
        fig_pr.add_vline(x=threshold_val, line_dash="dash", line_color=COLORS["rose"], annotation_text="threshold")
        fig_pr.update_layout(
            **PLOTLY_LAYOUT,
            title="Precision / Recall / F1 vs Threshold",
            xaxis_title="Threshold",
            yaxis=dict(range=[0, 1.05]),
            height=400,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_pr, use_container_width=True)

    # Metrics at selected threshold
    at_threshold = summarize_predictions(data.test_labels, data.test_scores, threshold_val)
    tc1, tc2, tc3 = st.columns(3)
    tc1.metric("Precision @ threshold", f"{at_threshold.precision:.3f}")
    tc2.metric("Recall @ threshold", f"{at_threshold.recall:.3f}")
    tc3.metric("F1 @ threshold", f"{at_threshold.f1:.3f}")
