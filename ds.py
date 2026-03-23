import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from sentinel_backend import SentinelEngine
from io import BytesIO
from collections import Counter


# ══════════════════════════════════════════════════════════════
# PAGE CONFIG  (must be first Streamlit call)
# ══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Sentinel HUD",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ══════════════════════════════════════════════════════════════
# CSS  —  dark HUD aesthetic, JetBrains Mono
# ══════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&display=swap');

/* ── Base ── */
html, body, .stApp { background-color: #060606 !important; }
* { font-family: 'JetBrains Mono', monospace !important; color: #e8e8e8 !important; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #0d0d0d !important;
    border-right: 1px solid #1a0000 !important;
}

/* ── Buttons ── */
div.stButton > button {
    background: #8B0000 !important;
    color: #fff !important;
    border: none !important;
    border-radius: 3px !important;
    padding: 0.5rem 1.5rem !important;
    font-weight: 700 !important;
    letter-spacing: 1px !important;
    transition: background 0.2s !important;
}
div.stButton > button:hover { background: #cc0000 !important; }

/* ── Dropdowns / Selects ── */
div[data-baseweb="select"] > div {
    background-color: #111 !important;
    border: 1px solid #8B0000 !important;
}
div[data-baseweb="popover"] ul { background-color: #111 !important; }
div[data-baseweb="popover"] li:hover { background-color: #1a0000 !important; }

/* ── Text area / inputs ── */
textarea, .stTextArea textarea {
    background: #0d0d0d !important;
    border: 1px solid #330000 !important;
    color: #e8e8e8 !important;
    font-size: 0.85rem !important;
}

/* ── Slider ── */
div[data-testid="stSlider"] * { color: #e8e8e8 !important; }
div[data-testid="stSlider"] [role="slider"] { background: #8B0000 !important; }

/* ── HUD Cards ── */
.hud-card {
    background: rgba(18, 8, 8, 0.95);
    border: 1px solid #220000;
    border-left: 4px solid #cc0000;
    padding: 16px 20px;
    margin-bottom: 12px;
    border-radius: 3px;
}
.hud-label {
    font-size: 0.65rem;
    letter-spacing: 2px;
    color: #666 !important;
    text-transform: uppercase;
    margin-bottom: 4px;
}
.hud-value {
    font-size: 2rem;
    font-weight: 700;
    color: #FF2222 !important;
    text-shadow: 0 0 12px rgba(255,34,34,0.4);
    line-height: 1.1;
}
.hud-value-green {
    font-size: 2rem;
    font-weight: 700;
    color: #00EB93 !important;
    text-shadow: 0 0 12px rgba(0,235,147,0.35);
    line-height: 1.1;
}
.hud-value-amber {
    font-size: 2rem;
    font-weight: 700;
    color: #FFAA00 !important;
    text-shadow: 0 0 12px rgba(255,170,0,0.35);
    line-height: 1.1;
}

/* ── Section headers ── */
.section-header {
    border-bottom: 1px solid #8B0000;
    padding-bottom: 8px;
    margin-bottom: 20px;
    font-size: 1.1rem;
    font-weight: 700;
    letter-spacing: 2px;
    color: #e8e8e8 !important;
}

/* ── Forensic row ── */
.forensic-row {
    border-bottom: 1px solid #141414;
    padding: 10px 0;
}
.forensic-verdict { font-weight: 700; font-size: 0.85rem; }
.forensic-type    { font-size: 0.72rem; color: #555 !important; }
.forensic-text    { font-size: 0.8rem;  color: #aaa !important; margin-top: 4px; }

/* ── Metric pills ── */
.pill {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 12px;
    font-size: 0.7rem;
    font-weight: 700;
    margin-right: 5px;
}
.pill-fake     { background: #3a0000; color: #FF4444 !important; border: 1px solid #660000; }
.pill-real     { background: #003a1a; color: #00EB93 !important; border: 1px solid #006633; }
.pill-sus      { background: #2e2000; color: #FFAA00 !important; border: 1px solid #554400; }
.pill-auth     { background: #001a33; color: #4499FF !important; border: 1px solid #003366; }

/* ── Radio ── */
div[data-testid="stRadio"] label { font-size: 0.8rem !important; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# CHART HELPERS
# ══════════════════════════════════════════════════════════════

_DARK_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#e8e8e8", family="JetBrains Mono"),
    margin=dict(t=30, b=10, l=10, r=10),
)


def draw_risk_gauge(score_str: str) -> go.Figure:
    """Gauge showing integrity score 0-100."""
    try:
        score = float(score_str.split("/")[0])
    except (ValueError, IndexError):
        score = 0.0

    color = "#00EB93" if score >= 80 else ("#FFAA00" if score >= 60 else "#FF2222")

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={
            "suffix": "%",
            "font":   {"size": 36, "color": color},
        },
        gauge={
            "axis": {
                "range":     [0, 100],
                "tickcolor": "#444",
                "tickfont":  {"size": 9},
            },
            "bar":    {"color": color, "thickness": 0.25},
            "bgcolor": "#0a0a0a",
            "steps": [
                {"range": [0,  60], "color": "#1a0000"},
                {"range": [60, 80], "color": "#1a1200"},
                {"range": [80, 100],"color": "#001a0d"},
            ],
            "threshold": {
                "line":      {"color": color, "width": 3},
                "thickness": 0.75,
                "value":     score,
            },
        },
    ))
    fig.update_layout(height=240, **_DARK_LAYOUT)
    return fig


def draw_threat_radar(analysis_res: dict) -> go.Figure:
    """Radar showing four detection axes for a single review."""
    try:
        fake_conf = float(analysis_res.get("FakeConfidence", "50%").replace("%", "")) / 100
    except ValueError:
        fake_conf = 0.5

    fraud_type = analysis_res.get("Type", "")
    trust_raw  = float(analysis_res.get("Trust", "50%").replace("%", "")) / 100

    mismatch    = 1.0  if "MANIPULATION" in fraud_type  else 0.15
    slang_score = 0.85 if "SLANG"        in fraud_type  else 0.1
    bot_score   = 1.0  if "BOT"          in fraud_type  else fake_conf * 0.6
    authenticity = trust_raw

    categories = [
        "Bot Probability",
        "Rating Mismatch",
        "Slang / Hostility",
        "Authenticity",
    ]
    values = [bot_score, mismatch, slang_score, authenticity]
    # Close the polygon
    categories_closed = categories + [categories[0]]
    values_closed     = values     + [values[0]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values_closed,
        theta=categories_closed,
        fill="toself",
        fillcolor="rgba(200, 0, 0, 0.18)",
        line=dict(color="#FF2222", width=2),
        name="Signal",
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                gridcolor="#222",
                tickfont={"size": 8, "color": "#555"},
                tickvals=[0.25, 0.5, 0.75, 1.0],
            ),
            angularaxis=dict(gridcolor="#222"),
            bgcolor="rgba(0,0,0,0)",
        ),
        showlegend=False,
        height=290,
        **_DARK_LAYOUT,
    )
    return fig


def draw_breakdown_bar(stats: dict) -> go.Figure:
    """Horizontal bar chart of the fake-type breakdown for a business."""
    labels = list(stats.keys())
    values = list(stats.values())
    colors = {
        "AUTHENTIC":           "#00EB93",
        "AI BOT SIGNATURE":    "#FF2222",
        "RATING MANIPULATION": "#FF6600",
        "LOW-QUALITY / SLANG": "#FFAA00",
    }
    bar_colors = [colors.get(l, "#888") for l in labels]

    fig = go.Figure(go.Bar(
        x=values,
        y=labels,
        orientation="h",
        marker_color=bar_colors,
        marker_line_width=0,
        text=values,
        textposition="inside",
        textfont=dict(size=11, color="#fff"),
    ))
    fig.update_layout(
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(tickfont=dict(size=10)),
        height=200,
        **_DARK_LAYOUT,
    )
    return fig


# ══════════════════════════════════════════════════════════════
# ENGINE  —  cached so it only trains once per session
# ══════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Initialising Sentinel Engine…")
def init_engine() -> SentinelEngine:
    engine = SentinelEngine()
    files = [
        "fitness_updated.csv",
        "clinics_updated.csv",
        "restaurants_updated.csv",
        "coaching_updated.csv",
    ]
    engine.prepare_and_train(files)
    return engine


try:
    engine = init_engine()
except Exception as exc:
    st.error(f"⛔ Engine failed to initialise: {exc}")
    st.stop()


# ══════════════════════════════════════════════════════════════
# SIDEBAR  —  navigation
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(
        "<h1 style='color:#cc0000 !important; letter-spacing:3px;'>🛡️ SENTINEL</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='color:#444 !important; font-size:0.7rem; letter-spacing:1px;'>FAKE REVIEW DETECTION SYSTEM</p>",
        unsafe_allow_html=True,
    )
    st.markdown("---")
    menu = st.radio(
        "OPERATIONAL MODULES",
        ["🏢 TARGET AUDIT", "🔍 SIGNAL SCAN"],
        label_visibility="visible",
    )
    st.markdown("---")
    st.markdown(
        "<p style='color:#333 !important; font-size:0.65rem;'>Model: Calibrated Random Forest<br>"
        "Features: char n-gram TF-IDF + 8 numeric signals</p>",
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════
# MODULE A — BUSINESS AUDIT
# ══════════════════════════════════════════════════════════════

if menu == "🏢 TARGET AUDIT":
    st.markdown("<div class='section-header'>◈ INFRASTRUCTURE SCANNER</div>", unsafe_allow_html=True)

    if engine.global_df is None or engine.global_df.empty:
        st.warning("No data loaded. Check CSV files.")
        st.stop()

    biz_list = sorted(engine.global_df["business_name"].dropna().unique())
    if not biz_list:
        st.warning("No businesses found in dataset.")
        st.stop()

    selected_biz = st.selectbox("Select Target Registry", biz_list)

    if st.button("⚡ INITIATE DEEP SCAN"):
        with st.spinner("Scanning…"):
            report = engine.get_business_verdict(selected_biz)

        # FIX 2b: report is now always a dict, so .get("Error") works correctly
        if report.get("Error"):
            st.error(report["Error"])
            st.stop()

        # ── Row 1: key metrics ──────────────────────────────
        c1, c2, c3 = st.columns(3)
        integrity_num = float(report["Integrity Score"].split("/")[0])
        val_class = (
            "hud-value-green" if integrity_num >= 80
            else "hud-value-amber" if integrity_num >= 60
            else "hud-value"
        )

        with c1:
            st.markdown(
                f"<div class='hud-card'>"
                f"<div class='hud-label'>INTEGRITY SCORE</div>"
                f"<div class='{val_class}'>{report['Integrity Score']}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                f"<div class='hud-card'>"
                f"<div class='hud-label'>VERDICT</div>"
                f"<div class='{val_class}' style='font-size:1rem !important;'>{report['Recommendation']}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
        with c3:
            total = len(report["Full_Details"])
            authentic = report["Breakdown"].get("AUTHENTIC", 0)
            st.markdown(
                f"<div class='hud-card'>"
                f"<div class='hud-label'>REVIEWS SCANNED</div>"
                f"<div class='hud-value-green'>{total}</div>"
                f"<div style='font-size:0.7rem; color:#555 !important;'>{authentic} authentic</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

        # ── Row 2: gauge + radar + bar ──────────────────────
        col_g, col_r, col_b = st.columns(3)
        with col_g:
            st.markdown("<div class='hud-label' style='padding-left:5px;'>INTEGRITY GAUGE</div>",
                        unsafe_allow_html=True)
            st.plotly_chart(draw_risk_gauge(report["Integrity Score"]), use_container_width=True)

        with col_r:
            st.markdown("<div class='hud-label' style='padding-left:5px;'>THREAT RADAR (SAMPLE)</div>",
                        unsafe_allow_html=True)
            if report["Full_Details"]:
                sample_res = report["Full_Details"][0][1]
                st.plotly_chart(draw_threat_radar(sample_res), use_container_width=True)

        with col_b:
            st.markdown("<div class='hud-label' style='padding-left:5px;'>FRAUD BREAKDOWN</div>",
                        unsafe_allow_html=True)
            st.plotly_chart(draw_breakdown_bar(report["Breakdown"]), use_container_width=True)

        # ── Row 3: forensic log ─────────────────────────────
        st.markdown("<div class='section-header' style='margin-top:20px;'>🧬 FORENSIC LOG</div>",
                    unsafe_allow_html=True)

        # Filter controls
        filter_cols = st.columns([1, 1, 1, 2])
        with filter_cols[0]:
            show_fake    = st.checkbox("🚨 Fake",        value=True)
        with filter_cols[1]:
            show_sus     = st.checkbox("⚠️ Suspicious",  value=True)
        with filter_cols[2]:
            show_real    = st.checkbox("✅ Real",         value=True)

        for text, res in report["Full_Details"]:
            verdict = res["Verdict"]
            if "FAKE"       in verdict and not show_fake: continue
            if "SUSPICIOUS" in verdict and not show_sus:  continue
            if "REAL"       in verdict and not show_real: continue

            if "FAKE"       in verdict: pill_cls, color = "pill-fake",  "#FF4444"
            elif "SUSPICIOUS" in verdict: pill_cls, color = "pill-sus", "#FFAA00"
            else:                         pill_cls, color = "pill-real","#00EB93"

            st.markdown(
                f"""<div class='forensic-row'>
                    <span class='pill {pill_cls}'>{res['Verdict']}</span>
                    <span class='forensic-type'>  TYPE: {res['Type']}  ·  
                    TRUST: {res['Trust']}  ·  FAKE CONF: {res['FakeConfidence']}</span>
                    <div class='forensic-text'>{text[:220]}{"…" if len(text)>220 else ""}</div>
                    <div style='font-size:0.68rem; color:#444 !important; margin-top:3px;'>
                        📋 {res['Explanation']}</div>
                </div>""",
                unsafe_allow_html=True,
            )


# ══════════════════════════════════════════════════════════════
# MODULE B — SINGLE REVIEW SCAN
# ══════════════════════════════════════════════════════════════

elif menu == "🔍 SIGNAL SCAN":
    st.markdown("<div class='section-header'>◈ HEURISTIC SIGNAL ANALYSIS</div>",
                unsafe_allow_html=True)

    txt    = st.text_area("PASTE REVIEW TEXT", height=180,
                          placeholder="Paste any review here to analyse…")
    rating = st.slider("DECLARED STAR RATING", min_value=1, max_value=5, value=5)

    if st.button("⚡ DECODE SIGNAL"):
        if not txt.strip():
            st.warning("Please enter review text before scanning.")
        else:
            with st.spinner("Analysing signal…"):
                res = engine.analyze_review(txt, rating)

            # Determine overall colour
            if "FAKE"        in res["Verdict"]: main_color, pill_cls = "#FF2222", "pill-fake"
            elif "SUSPICIOUS" in res["Verdict"]:main_color, pill_cls = "#FFAA00", "pill-sus"
            else:                               main_color, pill_cls = "#00EB93", "pill-real"

            c1, c2 = st.columns([1, 1])

            with c1:
                st.plotly_chart(draw_threat_radar(res), use_container_width=True)

            with c2:
                st.markdown(
                    f"<div class='hud-card'>"
                    f"<div class='hud-label'>VERDICT</div>"
                    f"<div style='font-size:1.6rem; font-weight:700; color:{main_color} !important;'>"
                    f"{res['Verdict']}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"<div class='hud-card'>"
                    f"<div class='hud-label'>FRAUD TYPE</div>"
                    f"<div style='font-size:1rem; font-weight:700; color:{main_color} !important;'>"
                    f"{res['Type']}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
                col_t, col_f = st.columns(2)
                col_t.metric("Trust Score",     res["Trust"])
                col_f.metric("Fake Confidence", res["FakeConfidence"])

                # FIX 3: Match the exact fallback string returned by the backend
                if res["Explanation"] != "Organic human feedback.":
                    st.markdown(
                        f"<div style='background:#1a0000; border:1px solid #330000; padding:12px; "
                        f"border-radius:3px; font-size:0.78rem; color:#ccc !important; margin-top:12px;'>"
                        f"⚠️ {res['Explanation']}</div>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.success("✅ No red flags detected. Review appears organic.")