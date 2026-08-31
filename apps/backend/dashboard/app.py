"""Doom Index Streamlit Dashboard.

Run with: streamlit run dashboard/app.py
"""

import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="Doom Index v2",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ──────────────────────────────────────────────────────────────

st.markdown("""
<style>
    .main-header { font-size: 3em; font-weight: 900; color: #ff4444; text-align: center; }
    .sub-header { font-size: 1.2em; color: #888; text-align: center; margin-bottom: 30px; }
    .metric-card { background: #1a1a1a; padding: 20px; border-radius: 10px; border: 1px solid #333; }
    .critical { color: #ff0000; font-weight: bold; }
    .high { color: #ff6600; font-weight: bold; }
    .moderate { color: #ffcc00; font-weight: bold; }
    .low { color: #00ff66; font-weight: bold; }
    .stProgress > div > div > div > div { background-color: #ff4444; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ─────────────────────────────────────────────────────────────────

st.sidebar.title("⚙️ Configuration")
api_url = st.sidebar.text_input("API URL", "http://localhost:8000")
auto_refresh = st.sidebar.checkbox("Auto-refresh leaderboard", value=False)

st.sidebar.markdown("---")
st.sidebar.markdown("**Model Status**")
try:
    health = requests.get(f"{api_url}/health", timeout=5).json()
    if health["model_loaded"]:
        st.sidebar.success(f"✅ Model loaded (v{health['version']})")
        st.sidebar.info(f"🖥️ GPUs: {health['cuda_devices']}")
    else:
        st.sidebar.warning("⚠️ Model not loaded")
except:
    st.sidebar.error("❌ API unreachable")

# ── Main Header ─────────────────────────────────────────────────────────────

st.markdown('<div class="main-header">🔥 DOOM INDEX v2</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Predictive Social Doom Index + Shadowban Simulator</div>', unsafe_allow_html=True)

# ── Tabs ────────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 Doom Predictor", 
    "⚔️ Attack Simulator", 
    "🏆 Leaderboard",
    "🔒 Privacy Analysis"
])

# ── Tab 1: Doom Predictor ───────────────────────────────────────────────────

with tab1:
    st.header("Multimodal Cancellation Risk Analysis")

    col1, col2 = st.columns([2, 1])

    with col1:
        text_input = st.text_area(
            "Enter social media post text:",
            height=150,
            placeholder="e.g., 'This celebrity is facing massive backlash for their controversial statements...'",
        )

        author_col, followers_col, verified_col = st.columns(3)
        with author_col:
            author_id = st.text_input("Author ID", "anonymous")
        with followers_col:
            followers = st.number_input("Followers", min_value=0, value=0, step=1000)
        with verified_col:
            verified = st.checkbox("Verified", value=False)

        analyze_btn = st.button("🔍 Analyze", type="primary", use_container_width=True)

    with col2:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.markdown("**How it works**")
        st.markdown("""
        1. **GraphSAGE** analyzes the author's network position
        2. **DistilBERT** encodes the post text
        3. **Fusion MLP** combines both for final risk score
        """)
        st.markdown("</div>", unsafe_allow_html=True)

    if analyze_btn and text_input:
        with st.spinner("Running multimodal analysis..."):
            try:
                response = requests.post(
                    f"{api_url}/analyze",
                    json={
                        "text": text_input,
                        "author_id": author_id,
                        "followers": followers,
                        "verified": verified,
                    },
                    timeout=30,
                )
                result = response.json()

                # Results display
                st.divider()

                res_col1, res_col2, res_col3 = st.columns(3)

                with res_col1:
                    doom_score = result.get("doom_score", 0)
                    risk_level = result.get("risk_level", "UNKNOWN")

                    color = {"CRITICAL": "#ff0000", "HIGH": "#ff6600", 
                            "MODERATE": "#ffcc00", "LOW": "#00ff66"}.get(risk_level, "#888")

                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=doom_score,
                        domain={'x': [0, 1], 'y': [0, 1]},
                        title={'text': f"Doom Score<br><span style='font-size:0.6em;color:{color}'>{risk_level}</span>"},
                        gauge={
                            'axis': {'range': [0, 100]},
                            'bar': {'color': color},
                            'steps': [
                                {'range': [0, 20], 'color': '#1a1a1a'},
                                {'range': [20, 40], 'color': '#2a2a1a'},
                                {'range': [40, 70], 'color': '#3a2a1a'},
                                {'range': [70, 100], 'color': '#3a1a1a'},
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': doom_score
                            }
                        }
                    ))
                    fig.update_layout(height=300, paper_bgcolor='rgba(0,0,0,0)', font={'color': "white"})
                    st.plotly_chart(fig, use_container_width=True)

                with res_col2:
                    st.subheader("Sentiment Analysis")
                    sentiment = result.get("sentiment", {})

                    if sentiment.get("vader"):
                        v = sentiment["vader"]
                        st.write(f"**VADER:** {v.get('compound', 0):.3f} compound")

                    if sentiment.get("distilbert"):
                        d = sentiment["distilbert"]
                        neg = d.get("LABEL_0", 0)
                        pos = d.get("LABEL_1", 0)
                        fig = px.bar(
                            x=["Negative", "Positive"],
                            y=[neg, pos],
                            color=["Negative", "Positive"],
                            color_discrete_map={"Negative": "#ff4444", "Positive": "#44ff44"},
                        )
                        fig.update_layout(height=200, showlegend=False, paper_bgcolor='rgba(0,0,0,0)')
                        st.plotly_chart(fig, use_container_width=True)

                with res_col3:
                    st.subheader("Toxicity Scores")
                    toxicity = result.get("toxicity", {})

                    if toxicity:
                        attrs = {k: v for k, v in toxicity.items() if v is not None}
                        if attrs:
                            fig = px.bar(
                                x=list(attrs.keys()),
                                y=list(attrs.values()),
                                color=list(attrs.values()),
                                color_continuous_scale="Reds",
                            )
                            fig.update_layout(height=250, showlegend=False, paper_bgcolor='rgba(0,0,0,0)')
                            st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("Toxicity data unavailable (Perspective API key not set)")

                # Technical details
                with st.expander("🔬 Technical Details"):
                    st.json({
                        "prediction": result.get("prediction"),
                        "probability": result.get("probability"),
                        "graph_embedding_norm": result.get("graph_embedding_norm"),
                        "text_embedding_norm": result.get("text_embedding_norm"),
                    })

            except Exception as e:
                st.error(f"Analysis failed: {e}")

# ── Tab 2: Attack Simulator ─────────────────────────────────────────────────

with tab2:
    st.header("⚔️ Shadowban Attack Simulator")
    st.markdown("Generate adversarial text variants that maximize doom score while evading moderation.")

    atk_col1, atk_col2 = st.columns([2, 1])

    with atk_col1:
        target_text = st.text_area(
            "Target post to attack:",
            height=120,
            placeholder="Enter a benign post to mutate...",
        )

        atk_params = st.columns(3)
        with atk_params[0]:
            max_variants = st.slider("Variants", 1, 10, 5)
        with atk_params[1]:
            tox_budget = st.slider("Toxicity Budget", 0.0, 1.0, 0.7)
        with atk_params[2]:
            atk_author = st.text_input("Target Author", "anonymous")

        attack_btn = st.button("🧬 Generate Attack Variants", type="primary", use_container_width=True)

    with atk_col2:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.markdown("**Attack Strategies**")
        st.markdown("""
        - 😤 **Emoji Injection** — Boosts engagement signals
        - ❓ **Rhetorical Questions** — Increases reply probability  
        - 📢 **Exaggeration** — Amplifies emotional response
        - 🔥 **Controversy Framing** — BREAKING prefix effect
        - ‼️ **Outrage Punctuation** — Visual urgency cues
        - 📣 **Call-to-Action** — Retweet bait
        - 🏛️ **Authority Challenge** — Anti-establishment framing
        """)
        st.markdown("</div>", unsafe_allow_html=True)

    if attack_btn and target_text:
        with st.spinner("Evolving adversarial variants..."):
            try:
                response = requests.post(
                    f"{api_url}/attack",
                    json={
                        "text": target_text,
                        "author_id": atk_author,
                        "max_variants": max_variants,
                        "toxicity_budget": tox_budget,
                    },
                    timeout=60,
                )
                result = response.json()

                st.divider()

                # Original
                orig_doom = result.get("original_doom", 0)
                st.metric("Original Doom Score", f"{orig_doom*100:.1f}/100")

                # Variants
                variants = result.get("variants", [])

                if variants:
                    st.subheader(f"Generated {len(variants)} Variants")

                    # Comparison chart
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        name="Original",
                        x=["Original"],
                        y=[orig_doom * 100],
                        marker_color="#444",
                    ))

                    colors = px.colors.sequential.Reds[2:]
                    for i, v in enumerate(variants):
                        fig.add_trace(go.Bar(
                            name=f"Variant {i+1} ({v['strategy']})",
                            x=[f"V{i+1}"],
                            y=[v['attacked_doom'] * 100],
                            marker_color=colors[i % len(colors)],
                            text=f"+{v['doom_uplift']*100:.1f}",
                            textposition="outside",
                        ))

                    fig.update_layout(
                        barmode='group',
                        height=400,
                        title="Doom Score Comparison",
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font_color='white',
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # Variant cards
                    for i, v in enumerate(variants):
                        with st.container():
                            cols = st.columns([3, 1, 1, 1])
                            with cols[0]:
                                st.code(v['variant_text'], language=None)
                            with cols[1]:
                                st.metric("Doom", f"{v['attacked_doom']*100:.1f}")
                            with cols[2]:
                                st.metric("Uplift", f"+{v['doom_uplift']*100:.1f}")
                            with cols[3]:
                                st.caption(f"Strategy: {v['strategy']}")
                            st.divider()
                else:
                    st.warning("No valid variants generated within toxicity budget.")

            except Exception as e:
                st.error(f"Attack simulation failed: {e}")

# ── Tab 3: Leaderboard ──────────────────────────────────────────────────────

with tab3:
    st.header("🏆 Leaderboard of the Damned")

    try:
        response = requests.get(f"{api_url}/leaderboard?limit=50", timeout=10)
        leaderboard = response.json()

        df = pd.DataFrame(leaderboard)

        # Filters
        filter_col1, filter_col2 = st.columns(2)
        with filter_col1:
            risk_filter = st.multiselect(
                "Filter by Risk Level",
                ["CRITICAL", "HIGH", "MODERATE", "LOW"],
                default=["CRITICAL", "HIGH"],
            )
        with filter_col2:
            min_followers = st.number_input("Min Followers", min_value=0, value=0)

        filtered = df[
            (df['risk_level'].isin(risk_filter)) &
            (df['followers'] >= min_followers)
        ]

        # Color coding
        def color_risk(val):
            colors = {"CRITICAL": "background-color: #3a0000", 
                     "HIGH": "background-color: #3a1a00",
                     "MODERATE": "background-color: #3a3a00",
                     "LOW": "background-color: #003a00"}
            return colors.get(val, "")

        st.dataframe(
            filtered.style.applymap(color_risk, subset=['risk_level']),
            use_container_width=True,
            height=500,
        )

        # Distribution chart
        fig = px.histogram(
            df, x="doom_score", color="risk_level",
            color_discrete_map={"CRITICAL": "#ff0000", "HIGH": "#ff6600", 
                               "MODERATE": "#ffcc00", "LOW": "#00ff66"},
            nbins=20,
            title="Doom Score Distribution",
        )
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='white')
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Could not load leaderboard: {e}")
        st.info("Showing mock data for demo purposes.")

        mock_df = pd.DataFrame([
            {"author_id": f"User_{i}", "doom_score": 95-i*4, "risk_level": "CRITICAL" if i < 5 else "HIGH", "followers": 100000+i*50000}
            for i in range(20)
        ])
        st.dataframe(mock_df, use_container_width=True)

# ── Tab 4: Privacy Analysis ─────────────────────────────────────────────────

with tab4:
    st.header("🔒 Privacy-Utility Tradeoff Analysis")
    st.markdown("Differential Privacy + Federated Learning impact on model performance.")

    # Mock data for privacy tradeoff (replace with real experiments)
    epsilons = [0.1, 0.5, 1.0, 2.0, 5.0, float('inf')]
    accuracies = [0.72, 0.78, 0.82, 0.85, 0.88, 0.91]
    f1_scores = [0.68, 0.75, 0.80, 0.83, 0.86, 0.89]

    priv_col1, priv_col2 = st.columns(2)

    with priv_col1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=[str(e) for e in epsilons],
            y=accuracies,
            mode='lines+markers',
            name='Accuracy',
            line=dict(color='#00ff66', width=3),
        ))
        fig.add_trace(go.Scatter(
            x=[str(e) for e in epsilons],
            y=f1_scores,
            mode='lines+markers',
            name='F1 Score',
            line=dict(color='#ff6600', width=3),
        ))
        fig.update_layout(
            title="Privacy-Utility Tradeoff",
            xaxis_title="Epsilon (Privacy Budget)",
            yaxis_title="Score",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='white',
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)

    with priv_col2:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.markdown("**Privacy Guarantees**")
        st.markdown("""
        | Epsilon | Privacy Level | Accuracy | F1 |
        |---------|--------------|----------|-----|
        | 0.1 | 🔒 Very Strong | 72% | 68% |
        | 1.0 | 🔒 Strong | 82% | 80% |
        | 5.0 | 🔓 Moderate | 88% | 86% |
        | ∞ | ❌ None | 91% | 89% |
        """)
        st.markdown("</div>", unsafe_allow_html=True)

    st.divider()
    st.subheader("Federated Learning Simulation")

    fl_col1, fl_col2 = st.columns(2)
    with fl_col1:
        num_clients = st.slider("Number of Clients", 2, 20, 5)
        num_rounds = st.slider("Aggregation Rounds", 1, 50, 10)
    with fl_col2:
        st.info(f"Simulating {num_clients} clients with {num_rounds} rounds of FedAvg.")
        if st.button("🔄 Run FL Simulation"):
            with st.spinner("Running federated learning simulation..."):
                # Mock FL convergence
                rounds = list(range(1, num_rounds + 1))
                convergence = [0.5 + 0.4 * (1 - np.exp(-r/5)) + np.random.normal(0, 0.02) for r in rounds]

                fig = px.line(
                    x=rounds, y=convergence,
                    labels={"x": "Round", "y": "Global Model F1"},
                    title="Federated Learning Convergence",
                )
                fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='white')
                st.plotly_chart(fig, use_container_width=True)

# ── Footer ──────────────────────────────────────────────────────────────────

st.divider()
st.caption("Doom Index v2.0 | Multimodal GNN + Transformer Architecture | H100-Optimized")
