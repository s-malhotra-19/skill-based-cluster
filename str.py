"""
╔══════════════════════════════════════════════════════════════════╗
║  Skill-Based Resume Clustering System                            ║
║  Frontend: Gradio + Plotly                                       ║
║  Consumes: clusters.json  (TF-IDF + DBSCAN + PCA backend)       ║
║  Target:   Non-technical users / portfolio showcase              ║
╚══════════════════════════════════════════════════════════════════╝
"""

import json
import os
import random
import gradio as gr
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from collections import Counter

# ══════════════════════════════════════════════════════════════════
# 1.  CUSTOM CSS  — warm editorial dark theme
#     Fonts: Fraunces (display) + Outfit (body)
# ══════════════════════════════════════════════════════════════════
CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,wght@0,400;0,700;1,400&family=Outfit:wght@300;400;500;600&display=swap');

/* ── Variables ──────────────────────────────── */
:root {
    --bg:        #13110f;
    --surface:   #1c1916;
    --surface2:  #252220;
    --border:    #33302d;
    --accent:    #e8a87c;   /* warm amber */
    --accent2:   #7eb8a4;   /* sage green  */
    --accent3:   #c98fa0;   /* dusty rose  */
    --accent4:   #8faec9;   /* slate blue  */
    --text:      #f0ebe4;
    --muted:     #7a7470;
    --radius:    16px;
}

/* ── Base ───────────────────────────────────── */
*, *::before, *::after { box-sizing: border-box; }
body, .gradio-container {
    font-family: 'Outfit', sans-serif !important;
    background: var(--bg) !important;
    color: var(--text) !important;
    min-height: 100vh;
}
.gradio-container { max-width: 1340px !important; margin: 0 auto; padding: 0 !important; }
footer { display: none !important; }

/* ── Hero ───────────────────────────────────── */
.hero-wrap {
    background: linear-gradient(135deg, #1c1916 0%, #13110f 50%);
    border-bottom: 1px solid var(--border);
    padding: 3.2rem 3rem 2.6rem;
    position: relative;
    overflow: hidden;
}
.hero-wrap::after {
    content: '';
    position: absolute;
    top: -80px; right: -80px;
    width: 340px; height: 340px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(232,168,124,0.07) 0%, transparent 65%);
    pointer-events: none;
}
.hero-eyebrow {
    font-family: 'Outfit', sans-serif;
    font-size: 0.72rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--accent);
    margin-bottom: 0.7rem;
}
.hero-title {
    font-family: 'Fraunces', serif;
    font-size: clamp(2rem, 4vw, 3rem);
    font-weight: 700;
    line-height: 1.15;
    margin-bottom: 0.6rem;
    color: var(--text);
}
.hero-title em { font-style: italic; color: var(--accent); }
.hero-sub {
    font-size: 1rem;
    color: var(--muted);
    max-width: 580px;
    line-height: 1.6;
}

/* ── Main layout ────────────────────────────── */
.main-body { display: flex; gap: 0; }
.sidebar {
    width: 300px;
    min-width: 270px;
    background: var(--surface);
    border-right: 1px solid var(--border);
    padding: 2rem 1.5rem;
    flex-shrink: 0;
}
.content-area {
    flex: 1;
    padding: 2.2rem 2.6rem;
    overflow: auto;
}

/* ── Section heading ────────────────────────── */
.s-label {
    font-family: 'Outfit', sans-serif;
    font-size: 0.65rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--muted);
    border-bottom: 1px solid var(--border);
    padding-bottom: 0.5rem;
    margin: 0 0 1.2rem;
}

/* ── Metric cards ───────────────────────────── */
.metrics-row { display: grid; grid-template-columns: repeat(4,1fr); gap: 1rem; margin-bottom: 2rem; }
.mcard {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.3rem 1.4rem 1.1rem;
    text-align: center;
    transition: border-color .2s, transform .2s;
    cursor: default;
}
.mcard:hover { border-color: var(--accent); transform: translateY(-2px); }
.mcard-val {
    font-family: 'Fraunces', serif;
    font-size: 2.2rem;
    font-weight: 700;
    line-height: 1;
    margin-bottom: 0.3rem;
}
.mcard-lbl { font-size: 0.74rem; color: var(--muted); letter-spacing: 0.06em; text-transform: uppercase; }

/* ── Alerts ─────────────────────────────────── */
.alert {
    border-radius: 10px;
    padding: 0.7rem 1rem;
    font-size: 0.84rem;
    margin-bottom: 0.8rem;
    display: flex;
    align-items: flex-start;
    gap: 0.5rem;
    line-height: 1.5;
}
.alert-success { background: rgba(126,184,164,.12); border: 1px solid rgba(126,184,164,.3); color: var(--accent2); }
.alert-warning  { background: rgba(232,168,124,.10); border: 1px solid rgba(232,168,124,.25); color: var(--accent); }
.alert-info     { background: rgba(143,174,201,.10); border: 1px solid rgba(143,174,201,.25); color: var(--accent4); }
.alert-error    { background: rgba(201,143,160,.12); border: 1px solid rgba(201,143,160,.3);  color: var(--accent3); }

/* ── Cluster cards ──────────────────────────── */
.cluster-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-left: 3px solid var(--accent);
    border-radius: var(--radius);
    padding: 1.4rem 1.6rem 1.2rem;
    margin-bottom: 1.1rem;
    transition: border-left-color .2s, box-shadow .2s;
}
.cluster-card:hover {
    border-left-color: var(--accent2);
    box-shadow: 0 4px 24px rgba(0,0,0,.35);
}
.cc-title {
    font-family: 'Fraunces', serif;
    font-size: 1.15rem;
    font-weight: 700;
    color: var(--accent);
    margin-bottom: 0.2rem;
}
.cc-meta { font-size: 0.78rem; color: var(--muted); margin-bottom: 0.9rem; }
.cc-section { font-size: 0.68rem; letter-spacing: 0.14em; text-transform: uppercase; color: var(--muted); margin-bottom: 0.4rem; }

/* ── Keyword chips ───────────────────────────── */
.chips { display: flex; flex-wrap: wrap; gap: 5px; margin-bottom: 1rem; }
.chip {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 3px 11px;
    font-size: 0.73rem;
    color: var(--accent);
    font-family: 'Outfit', monospace;
    letter-spacing: 0.02em;
}
.chip.green  { color: var(--accent2); border-color: rgba(126,184,164,.3); }
.chip.blue   { color: var(--accent4); border-color: rgba(143,174,201,.3); }
.chip.rose   { color: var(--accent3); border-color: rgba(201,143,160,.3); }

/* ── Sample resume snippet ───────────────────── */
.snippet {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 0.9rem 1.1rem;
    font-size: 0.83rem;
    line-height: 1.6;
    color: var(--text);
    margin-bottom: 0.6rem;
}
.snippet-cat {
    font-size: 0.68rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--accent2);
    margin-bottom: 0.4rem;
}

/* ── How it works ────────────────────────────── */
.how-card {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.2rem 1.4rem;
    margin-bottom: 0.7rem;
    display: flex;
    gap: 1rem;
    align-items: flex-start;
}
.how-num {
    font-family: 'Fraunces', serif;
    font-size: 1.5rem;
    font-weight: 700;
    color: var(--accent);
    line-height: 1;
    flex-shrink: 0;
    width: 36px;
}
.how-body { font-size: 0.87rem; line-height: 1.6; }
.how-body strong { color: var(--text); }
.how-body span   { color: var(--muted); }

/* ── Sidebar controls ────────────────────────── */
.sidebar .gr-dropdown, .sidebar .gr-textbox {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--text) !important;
    font-family: 'Outfit', sans-serif !important;
    font-size: 0.88rem !important;
}
.sidebar label { font-size: 0.7rem !important; letter-spacing: 0.12em; text-transform: uppercase; color: var(--muted) !important; }

/* ── App footer ──────────────────────────────── */
.app-footer {
    text-align: center;
    padding: 1.8rem 0 2.5rem;
    font-size: 0.78rem;
    color: var(--muted);
    border-top: 1px solid var(--border);
    margin-top: 2.5rem;
}
.app-footer b { color: var(--accent); font-weight: 500; }

/* ── Plotly wrapper ──────────────────────────── */
.plotly-wrap { border-radius: var(--radius); overflow: hidden; }

/* ── Divider ─────────────────────────────────── */
.divider { border: none; border-top: 1px solid var(--border); margin: 1.8rem 0; }

/* ── Gradio overrides ────────────────────────── */
.gr-button-primary {
    background: var(--accent) !important;
    color: #13110f !important;
    border: none !important;
    font-family: 'Outfit', sans-serif !important;
    font-weight: 600 !important;
    border-radius: 10px !important;
}
.gr-button-secondary {
    background: var(--surface2) !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
}
.gr-form, .gr-box { background: transparent !important; border: none !important; }
.gr-panel { background: transparent !important; }
"""


# ══════════════════════════════════════════════════════════════════
# 2.  DATA LOADING
# ══════════════════════════════════════════════════════════════════

def make_demo_data():
    """Return synthetic clusters.json-shaped data for demo / missing file."""
    random.seed(7)

    def _cluster(cid, label, kws, cats, n):
        cx, cy = random.uniform(-4, 4), random.uniform(-4, 4)
        return {
            "id": cid, "label": label, "count": n,
            "keywords": [{"word": w, "score": round(random.uniform(.3, 1.), 3)} for w in kws],
            "top_categories": [{"cat": c, "n": random.randint(2, 12)} for c in cats],
            "samples": [
                {"category": cats[0], "snippet": f"Experienced {label} professional with {random.randint(2,8)} years of expertise in {kws[0]} and {kws[1]}, delivering high-impact results."},
                {"category": cats[-1], "snippet": f"Strong background in {kws[2]}, skilled at cross-functional collaboration and agile environments."},
            ],
            "x": [cx + random.gauss(0, .6) for _ in range(n)],
            "y": [cy + random.gauss(0, .6) for _ in range(n)],
        }

    clusters = [
        _cluster(0,  "Data Science & ML",    ["python","sklearn","tensorflow","pandas","numpy"],    ["Data Science","Analytics"],     30),
        _cluster(1,  "Web Development",       ["javascript","react","node.js","typescript","css"],   ["Frontend","Fullstack"],         37),
        _cluster(2,  "DevOps & Cloud",        ["docker","kubernetes","aws","terraform","ci/cd"],     ["DevOps","Cloud Engineering"],   21),
        _cluster(3,  "NLP & AI Research",     ["nlp","transformers","bert","pytorch","llm"],         ["AI Research","NLP"],            15),
        _cluster(4,  "Database Engineering",  ["sql","postgresql","mongodb","redis","spark"],        ["Database","Data Engineering"],  24),
        _cluster(5,  "Cybersecurity",         ["pentesting","siem","cryptography","nmap","soc"],     ["Security","Compliance"],        18),
        _cluster(-1, "Noise / Unassigned",    ["varied","misc","general"],                           ["Uncategorised"],                9),
    ]
    real = [c for c in clusters if c["id"] != -1]
    return {
        "total_resumes": sum(c["count"] for c in clusters),
        "n_clusters": len(real),
        "noise_count": 9,
        "clusters": clusters,
    }


def load_clusters(path: str = "clusters.json"):
    """Load clusters.json or fall back to demo data."""
    if not os.path.exists(path):
        return make_demo_data(), "demo"
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f), "ok"
    except Exception as e:
        return make_demo_data(), f"error:{e}"


# ══════════════════════════════════════════════════════════════════
# 3.  HTML BUILDERS  (pure helper functions)
# ══════════════════════════════════════════════════════════════════

def _alert(kind: str, icon: str, msg: str) -> str:
    return f'<div class="alert alert-{kind}">{icon}&nbsp; {msg}</div>'


def _chip(word: str, color: str = "") -> str:
    return f'<span class="chip {color}">{word}</span>'


def _chips(words, color="") -> str:
    return '<div class="chips">' + "".join(_chip(w, color) for w in words) + "</div>"


def build_hero() -> str:
    return """
    <div class="hero-wrap">
        <div class="hero-eyebrow">● AI-Powered Resume Intelligence</div>
        <div class="hero-title">Skill-Based Resume<br><em>Clustering</em> System</div>
        <div class="hero-sub">
            Unsupervised machine learning automatically groups resumes by skill patterns —
            making it easy to discover talent clusters, explore domains, and understand
            what skills define each group.
        </div>
    </div>
    """


def build_metrics(data: dict) -> str:
    total    = data.get("total_resumes", 0)
    n_clust  = data.get("n_clusters", 0)
    noise    = data.get("noise_count", 0)
    avg_size = round(total / n_clust, 1) if n_clust else 0

    def card(val, lbl, color):
        return f"""
        <div class="mcard">
            <div class="mcard-val" style="color:{color}">{val}</div>
            <div class="mcard-lbl">{lbl}</div>
        </div>"""

    return f"""
    <div class="s-label">Overview Metrics</div>
    <div class="metrics-row">
        {card(total,    "Total Resumes",     "#e8a87c")}
        {card(n_clust,  "Skill Clusters",    "#7eb8a4")}
        {card(noise,    "Noise Points",      "#c98fa0")}
        {card(avg_size, "Avg Cluster Size",  "#8faec9")}
    </div>
    """


def build_alerts(data: dict, status: str) -> str:
    html = ""
    total = data.get("total_resumes", 0)
    noise = data.get("noise_count", 0)

    if status == "demo":
        html += _alert("info", "🔬", "<b>Demo Mode:</b> No <code>clusters.json</code> found — showing synthetic sample data. Place your real file in the same folder to load live results.")
    elif status == "ok":
        html += _alert("success", "✅", f"<b>Clusters loaded successfully</b> — {total} resumes across {data.get('n_clusters',0)} skill groups.")
    elif status.startswith("error"):
        html += _alert("error", "❌", f"Could not read clusters.json: <code>{status[6:]}</code>")

    if noise / total > 0.2 if total else False:
        html += _alert("warning", "⚠️", f"<b>High noise detected:</b> {noise} resumes ({noise/total:.0%}) couldn't be assigned to any cluster. Consider tuning DBSCAN parameters.")

    return html


def build_how_it_works() -> str:
    steps = [
        ("01", "Reading Resumes", "Every resume is read as plain text — just like reading a document normally."),
        ("02", "Converting to Numbers", "Each resume is turned into a list of numbers that represent how important each skill-word is. This is called <b>TF-IDF</b>."),
        ("03", "Finding Similar Resumes", "An algorithm called <b>DBSCAN</b> automatically groups resumes that use similar words — no human labelling needed."),
        ("04", "Visualising in 2D", "<b>PCA</b> shrinks the complex data into an X-Y map you can actually see and explore in the scatter plot above."),
        ("05", "Labelling Clusters", "The most common skill words in each group become the cluster's name and keywords."),
    ]
    cards = "".join(f"""
    <div class="how-card">
        <div class="how-num">{n}</div>
        <div class="how-body"><strong>{t}</strong><br><span>{d}</span></div>
    </div>""" for n, t, d in steps)

    return f"""
    <div class="s-label" style="margin-top:2rem">How does this system work?</div>
    <div style="font-size:0.85rem;color:var(--muted);margin-bottom:1.1rem;">
        No technical knowledge needed — here's the simple story behind the AI.
    </div>
    {cards}
    """


def build_cluster_cards(clusters: list, selected: str, keyword: str) -> str:
    real = [c for c in clusters if c.get("id") != -1]

    # Filter by selected cluster
    if selected and selected != "All Clusters":
        real = [c for c in real if c.get("label") == selected]

    # Filter by keyword search
    if keyword.strip():
        terms = [t.strip().lower() for t in keyword.split(",") if t.strip()]
        real  = [c for c in real if any(
            any(t in kw["word"].lower() for t in terms)
            for kw in c.get("keywords", [])
        )]

    if not real:
        return _alert("warning", "🔍", "No matching clusters found. Try different keywords or select <b>All Clusters</b> from the dropdown.")

    palette = ["", "green", "blue", "rose", "", "green", "blue"]
    html = f'<div class="s-label">Cluster Detail Cards <span style="color:var(--muted);font-size:0.8em">({len(real)} shown)</span></div>'

    for i, c in enumerate(real):
        cid    = c.get("id", "?")
        label  = c.get("label", f"Cluster {cid}")
        count  = c.get("count", 0)
        kws    = c.get("keywords", [])[:10]
        cats   = c.get("top_categories", [])
        samples = c.get("samples", [])
        color  = palette[i % len(palette)]

        cat_str = " · ".join(f'{x["cat"]} ({x["n"]})' for x in cats[:3]) or "—"
        chip_html = _chips([kw["word"] for kw in kws], color)

        sample_html = ""
        for s in samples:
            sample_html += f"""
            <div class="snippet">
                <div class="snippet-cat">{s.get('category','—')}</div>
                {s.get('snippet','No snippet.')}
            </div>"""

        html += f"""
        <div class="cluster-card">
            <div class="cc-title">◆ {label}</div>
            <div class="cc-meta">Cluster #{cid} &nbsp;·&nbsp; {count} resumes &nbsp;·&nbsp; {cat_str}</div>
            <div class="cc-section">Top Skills</div>
            {chip_html}
            <div class="cc-section">Sample Resumes</div>
            {sample_html}
        </div>"""

    return html


# ══════════════════════════════════════════════════════════════════
# 4.  PLOTLY FIGURES
# ══════════════════════════════════════════════════════════════════

PALETTE = ["#e8a87c","#7eb8a4","#c98fa0","#8faec9","#c4b5a0","#a8c4b5","#b5a8c4"]
BG      = "#13110f"
SURFACE = "#1c1916"
GRID    = "#252220"
MUTED   = "#7a7470"


def fig_scatter(clusters: list) -> go.Figure:
    """Interactive PCA 2-D scatter plot."""
    rows = []
    for c in clusters:
        cid   = c.get("id")
        label = c.get("label", f"Cluster {cid}")
        count = c.get("count", 0)
        for x, y in zip(c.get("x", []), c.get("y", [])):
            rows.append({"x": x, "y": y, "Cluster": "Noise" if cid == -1 else label,
                         "Resumes": count})

    df = pd.DataFrame(rows)
    unique = [cl for cl in df["Cluster"].unique() if cl != "Noise"] + (["Noise"] if "Noise" in df["Cluster"].values else [])
    color_map = {cl: (PALETTE[i % len(PALETTE)] if cl != "Noise" else "#3d3a37") for i, cl in enumerate(unique)}

    fig = px.scatter(
        df, x="x", y="y", color="Cluster",
        color_discrete_map=color_map,
        custom_data=["Cluster","Resumes"],
        height=480,
    )
    fig.update_traces(
        marker=dict(size=9, opacity=0.85, line=dict(width=0.5, color=BG)),
        hovertemplate="<b>%{customdata[0]}</b><br>%{customdata[1]} resumes<extra></extra>",
    )
    fig.update_layout(
        paper_bgcolor=SURFACE, plot_bgcolor=BG,
        font=dict(family="Outfit", color=MUTED, size=12),
        legend=dict(bgcolor="rgba(28,25,22,.85)", bordercolor="#33302d",
                    borderwidth=1, title_text=""),
        xaxis=dict(showgrid=True, gridcolor=GRID, zeroline=False, title="PCA Component 1"),
        yaxis=dict(showgrid=True, gridcolor=GRID, zeroline=False, title="PCA Component 2"),
        margin=dict(l=10,r=10,t=10,b=10),
    )
    return fig


def fig_keywords(clusters: list) -> go.Figure:
    """Aggregated keyword importance horizontal bar chart."""
    scores: Counter = Counter()
    for c in clusters:
        if c.get("id") == -1:
            continue
        for kw in c.get("keywords", []):
            scores[kw["word"]] += kw["score"]

    top = scores.most_common(18)
    if not top:
        return go.Figure()

    df = pd.DataFrame(top[::-1], columns=["Keyword","Score"])
    fig = go.Figure(go.Bar(
        x=df["Score"], y=df["Keyword"], orientation="h",
        marker=dict(color=df["Score"], colorscale=[[0,GRID],[1,"#e8a87c"]], showscale=False),
        text=df["Score"].round(2), textposition="outside",
        textfont=dict(color=MUTED, size=11),
        hovertemplate="<b>%{y}</b>: %{x:.3f}<extra></extra>",
    ))
    fig.update_layout(
        paper_bgcolor=SURFACE, plot_bgcolor=BG,
        font=dict(family="Outfit", color=MUTED, size=12),
        xaxis=dict(showgrid=True, gridcolor=GRID, title="Cumulative TF-IDF Score"),
        yaxis=dict(showgrid=False),
        height=500, margin=dict(l=10,r=60,t=10,b=10),
    )
    return fig


def fig_cluster_sizes(clusters: list) -> go.Figure:
    """Cluster resume count bar chart."""
    real = [c for c in clusters if c.get("id") != -1]
    df = pd.DataFrame([{"Cluster": c.get("label","?"), "Resumes": c.get("count",0)}
                       for c in real]).sort_values("Resumes", ascending=False)

    fig = go.Figure(go.Bar(
        x=df["Cluster"], y=df["Resumes"],
        marker=dict(color=df["Resumes"],
                    colorscale=[[0,GRID],[1,"#7eb8a4"]],
                    showscale=False),
        text=df["Resumes"], textposition="outside",
        textfont=dict(color=MUTED, size=11),
        hovertemplate="<b>%{x}</b>: %{y} resumes<extra></extra>",
    ))
    fig.update_layout(
        paper_bgcolor=SURFACE, plot_bgcolor=BG,
        font=dict(family="Outfit", color=MUTED, size=12),
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor=GRID, title="Resumes"),
        height=360, margin=dict(l=10,r=10,t=10,b=10),
    )
    return fig


# ══════════════════════════════════════════════════════════════════
# 5.  GRADIO APP DEFINITION
# ══════════════════════════════════════════════════════════════════

# Load data once at startup
_DATA, _STATUS = load_clusters()
_CLUSTERS      = _DATA.get("clusters", [])
_REAL_LABELS   = ["All Clusters"] + [c["label"] for c in _CLUSTERS if c.get("id") != -1]


def refresh_dashboard(cluster_filter: str, keyword_search: str):
    """Main callback — returns all UI components."""
    alerts_html  = build_alerts(_DATA, _STATUS)
    metrics_html = build_metrics(_DATA)
    scatter      = fig_scatter(_CLUSTERS)
    bar_kw       = fig_keywords(_CLUSTERS)
    bar_sz       = fig_cluster_sizes(_CLUSTERS)
    cards_html   = build_cluster_cards(_CLUSTERS, cluster_filter, keyword_search)
    how_html     = build_how_it_works()

    footer = """
    <div class="app-footer">
        Built with <b>TF-IDF</b> · <b>DBSCAN</b> · <b>PCA</b> · <b>Plotly</b> · <b>Gradio</b>
        &nbsp;—&nbsp; Skill-Based Resume Clustering System &nbsp;·&nbsp; College Major Project
    </div>"""

    return alerts_html, metrics_html, scatter, bar_kw, bar_sz, cards_html, how_html, footer


# ══════════════════════════════════════════════════════════════════
# 6.  BUILD UI
# ══════════════════════════════════════════════════════════════════

with gr.Blocks(css=CUSTOM_CSS, title="Resume Clustering Dashboard") as demo:

    # ── Hero ──────────────────────────────────────────────────────
    gr.HTML(build_hero())

    # ── Alerts ────────────────────────────────────────────────────
    alerts_out = gr.HTML()

    # ── Sidebar + Content two-column layout ───────────────────────
    with gr.Row():

        # LEFT: sidebar
        with gr.Column(scale=1, min_width=260, elem_classes=["sidebar"]):
            gr.HTML('<div class="s-label" style="margin-top:0">Filter & Explore</div>')

            cluster_dd = gr.Dropdown(
                choices=_REAL_LABELS,
                value="All Clusters",
                label="Select a Cluster",
                info="Choose a skill group to focus on",
            )
            keyword_tb = gr.Textbox(
                label="Search Skills / Keywords",
                placeholder="e.g. python, docker, sql …",
                info="Comma-separated keywords to filter clusters",
            )
            apply_btn = gr.Button("🔍  Apply Filters", variant="primary", size="sm")
            reset_btn = gr.Button("↺  Reset",          variant="secondary", size="sm")

            gr.HTML("""
            <hr class="divider">
            <div class="alert alert-info" style="font-size:.8rem;margin-top:.5rem;">
                ℹ️ Select a cluster to explore similar resume profiles in detail.
            </div>
            <div class="alert alert-warning" style="font-size:.8rem;">
                ⚠️ <b>Noise points</b> are resumes that didn't fit strongly into any group —
                they may be unique or mixed-skill profiles.
            </div>
            """)

        # RIGHT: main content
        with gr.Column(scale=4, elem_classes=["content-area"]):

            # Metrics
            metrics_out = gr.HTML()

            gr.HTML('<hr class="divider">')
            gr.HTML('<div class="s-label">Resume Cluster Map (PCA Visualisation)</div>')
            gr.HTML('<p style="font-size:.83rem;color:var(--muted);margin-bottom:.8rem;">Each dot is a resume. Colours show cluster groups. Hover to explore · Scroll to zoom · Drag to pan.</p>')

            scatter_out = gr.Plot(label="")

            with gr.Row():
                with gr.Column():
                    gr.HTML('<div class="s-label" style="margin-top:1.5rem">Top Skills by TF-IDF Score</div>')
                    kw_bar_out = gr.Plot(label="")
                with gr.Column():
                    gr.HTML('<div class="s-label" style="margin-top:1.5rem">Resumes per Cluster</div>')
                    sz_bar_out = gr.Plot(label="")

            gr.HTML('<hr class="divider">')
            cards_out = gr.HTML()

            gr.HTML('<hr class="divider">')
            how_out   = gr.HTML()

            footer_out = gr.HTML()

    # ── Wire up callbacks ─────────────────────────────────────────
    outputs = [alerts_out, metrics_out, scatter_out, kw_bar_out, sz_bar_out, cards_out, how_out, footer_out]

    apply_btn.click(fn=refresh_dashboard, inputs=[cluster_dd, keyword_tb], outputs=outputs)
    cluster_dd.change(fn=refresh_dashboard, inputs=[cluster_dd, keyword_tb], outputs=outputs)
    keyword_tb.submit(fn=refresh_dashboard, inputs=[cluster_dd, keyword_tb], outputs=outputs)

    def reset():
        return refresh_dashboard("All Clusters", "")

    reset_btn.click(fn=reset, inputs=[], outputs=outputs)

    # ── Load on startup ───────────────────────────────────────────
    demo.load(fn=lambda: refresh_dashboard("All Clusters", ""), inputs=[], outputs=outputs)


# ══════════════════════════════════════════════════════════════════
# 7.  ENTRY POINT
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",  # Use localhost — works correctly in Windows browsers
        server_port=7860,
        share=False,
        show_error=True,
    )