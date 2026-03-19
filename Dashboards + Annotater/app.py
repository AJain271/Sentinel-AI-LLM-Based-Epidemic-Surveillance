import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
import os

# --- PAGE CONFIG ---
st.set_page_config(layout="wide", page_title="Symptom Cluster Dashboard", page_icon="🩺")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, "..", "initial results", "clustering_results.csv")
METRICS_PATH = os.path.join(SCRIPT_DIR, "..", "Clustering", "cluster_metrics.json")
TRANSCRIPT_DIR = os.path.join(SCRIPT_DIR, "..", "Clean Transcripts")

# --- LOAD DATA ---
@st.cache_data
def load_data():
    if not os.path.exists(DATA_PATH):
        return None, None
    df = pd.read_csv(DATA_PATH)
    metrics = {}
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, 'r') as f:
            metrics = json.load(f)
    return df, metrics

df, metrics = load_data()

if df is None:
    st.error("No data found. Run `python cluster_analysis.py` first.")
    st.stop()

# Identify symptom columns
SYMPTOM_COLS = [c for c in df.columns if c not in [
    'Filename', 'Predicted_Category', 'Cluster', 'TSNE_1', 'TSNE_2', 'Silhouette'
]]

# ==========================================
# SIDEBAR
# ==========================================
st.sidebar.title("🩺 Navigation")
page = st.sidebar.radio("Go to", [
    "Overview & Metrics",
    "Cluster Explorer",
    "Patient Inspector",
    "Feature Analysis"
])

# ==========================================
# PAGE 1: OVERVIEW
# ==========================================
if page == "Overview & Metrics":
    st.title("Symptom Cluster Analysis Dashboard")
    st.markdown("Zero-Shot LLM extraction → DBSCAN clustering on **272 clinical transcripts**.")

    # --- KPI Row ---
    c1, c2, c3, c4 = st.columns(4)
    n_clusters = metrics.get("n_clusters", "?")
    c1.metric("Total Patients", metrics.get("n_samples", len(df)))
    c2.metric("Clusters", n_clusters)
    c3.metric("Silhouette Coeff.", f"{metrics.get('silhouette_coefficient', 'N/A')}")
    c4.metric("Davies-Bouldin Idx.", f"{metrics.get('davies_bouldin_index', 'N/A')}")

    st.markdown("---")

    # --- Metric Explanations ---
    col_ex1, col_ex2 = st.columns(2)
    with col_ex1:
        st.info(
            "**Silhouette Coefficient** ranges from -1 to 1. "
            "Higher values indicate well-separated, compact clusters. "
            "Values above 0.5 are generally considered good."
        )
    with col_ex2:
        st.info(
            "**Davies-Bouldin Index** measures average similarity between clusters. "
            "Lower values indicate better separation. A score of 0 means perfect clustering."
        )

    st.markdown("---")

    # --- Cluster Map ---
    st.subheader("Patient Cluster Map (t-SNE Projection)")

    # Make cluster labels strings for discrete coloring
    df_plot = df.copy()
    df_plot['Cluster_Label'] = df_plot['Cluster'].apply(lambda x: f"Noise" if x == -1 else f"Cluster {x}")

    fig = px.scatter(
        df_plot, x="TSNE_1", y="TSNE_2",
        color="Cluster_Label",
        symbol="Predicted_Category",
        hover_data=["Filename", "Predicted_Category"],
        title=f"DBSCAN Clustering — {n_clusters} Clusters",
        height=650,
        color_discrete_sequence=px.colors.qualitative.Bold
    )
    fig.update_traces(marker=dict(size=8, opacity=0.8, line=dict(width=0.5, color='DarkSlateGrey')))
    fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=-0.25))
    st.plotly_chart(fig, use_container_width=True)

    # --- Cluster Size Breakdown ---
    st.subheader("Cluster Size Distribution")
    sizes = metrics.get("cluster_sizes", {})
    if sizes:
        size_df = pd.DataFrame(list(sizes.items()), columns=["Cluster", "Count"])
        size_df['Label'] = size_df['Cluster'].apply(lambda x: "Noise" if x == "-1" else f"Cluster {x}")
        fig_bar = px.bar(size_df, x="Label", y="Count", color="Label",
                         color_discrete_sequence=px.colors.qualitative.Bold)
        fig_bar.update_layout(showlegend=False, height=350)
        st.plotly_chart(fig_bar, use_container_width=True)

# ==========================================
# PAGE 2: CLUSTER EXPLORER
# ==========================================
elif page == "Cluster Explorer":
    st.title("Cluster Explorer")

    clusters_available = sorted(df['Cluster'].unique())
    cluster_labels = [f"Noise (-1)" if c == -1 else f"Cluster {c}" for c in clusters_available]
    selected_label = st.selectbox("Select Cluster", cluster_labels)

    # Map back to int
    if "Noise" in selected_label:
        selected_cluster = -1
    else:
        selected_cluster = int(selected_label.split(" ")[1])

    cluster_df = df[df['Cluster'] == selected_cluster]
    st.markdown(f"**{len(cluster_df)} patients** in {selected_label}")

    # --- Category Breakdown ---
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Category Breakdown")
        cat_counts = cluster_df['Predicted_Category'].value_counts().reset_index()
        cat_counts.columns = ['Category', 'Count']
        fig_pie = px.pie(cat_counts, values='Count', names='Category',
                         color_discrete_sequence=px.colors.qualitative.Set2)
        fig_pie.update_layout(height=350)
        st.plotly_chart(fig_pie, use_container_width=True)

    with col2:
        st.subheader("Symptom Prevalence in this Cluster")
        # Mean symptom score across the cluster
        mean_scores = cluster_df[SYMPTOM_COLS].mean().sort_values(ascending=True)
        fig_symp = px.bar(
            x=mean_scores.values, y=mean_scores.index,
            orientation='h', labels={'x': 'Mean Score (0-2)', 'y': 'Symptom'},
            color=mean_scores.values,
            color_continuous_scale='YlOrRd'
        )
        fig_symp.update_layout(height=500, coloraxis_showscale=False)
        st.plotly_chart(fig_symp, use_container_width=True)

    # --- Comparison with Global ---
    st.subheader("Cluster vs. Global Average")
    global_mean = df[SYMPTOM_COLS].mean()
    cluster_mean = cluster_df[SYMPTOM_COLS].mean()
    diff = (cluster_mean - global_mean).sort_values(ascending=False)

    fig_diff = go.Figure()
    fig_diff.add_trace(go.Bar(
        x=diff.index, y=diff.values,
        marker_color=['#2ecc71' if v > 0 else '#e74c3c' for v in diff.values]
    ))
    fig_diff.update_layout(
        title="Symptom Deviation from Global Mean (positive = more prevalent in cluster)",
        yaxis_title="Difference from Global Mean",
        height=400
    )
    st.plotly_chart(fig_diff, use_container_width=True)

    # --- Per-sample silhouette ---
    if 'Silhouette' in cluster_df.columns and cluster_df['Silhouette'].notna().any():
        st.subheader("Per-Patient Silhouette Score")
        sil_df = cluster_df[['Filename', 'Silhouette']].dropna().sort_values('Silhouette')
        fig_sil = px.bar(sil_df, x='Filename', y='Silhouette',
                         color='Silhouette', color_continuous_scale='RdYlGn',
                         title="How well each patient fits this cluster (higher = better fit)")
        fig_sil.update_layout(height=350, xaxis_tickangle=-45)
        st.plotly_chart(fig_sil, use_container_width=True)

# ==========================================
# PAGE 3: PATIENT INSPECTOR
# ==========================================
elif page == "Patient Inspector":
    st.title("Patient Inspector")

    # Select patient
    all_patients = df['Filename'].tolist()
    selected_patient = st.selectbox("Select a Patient", all_patients)

    if selected_patient:
        row = df[df['Filename'] == selected_patient].iloc[0]

        # Header info
        c1, c2, c3 = st.columns(3)
        c1.metric("Cluster", int(row['Cluster']))
        c2.metric("Category", row['Predicted_Category'])
        sil_val = row.get('Silhouette', None)
        c3.metric("Silhouette Score", f"{sil_val:.3f}" if pd.notna(sil_val) else "N/A (noise)")

        st.markdown("---")

        col_left, col_right = st.columns([1, 1])

        with col_left:
            st.subheader("Extracted Symptom Profile")

            # Build a nice table
            symptom_data = []
            for s in SYMPTOM_COLS:
                val = int(row[s])
                label = {0: "Not Mentioned", 1: "Negated", 2: "Present"}.get(val, str(val))
                emoji = {0: "⚪", 1: "🔴", 2: "🟢"}.get(val, "")
                symptom_data.append({"Symptom": s, "Status": f"{emoji} {label}", "Score": val})

            symptom_df = pd.DataFrame(symptom_data)
            # Show present first, then negated, then not mentioned
            symptom_df = symptom_df.sort_values('Score', ascending=False)
            st.dataframe(symptom_df[['Symptom', 'Status']], use_container_width=True, height=500)

        with col_right:
            st.subheader("Original Transcript")
            transcript_path = os.path.join(TRANSCRIPT_DIR, selected_patient)
            try:
                with open(transcript_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                st.text_area("Transcript", text, height=500)
            except FileNotFoundError:
                st.warning(f"Transcript file not found: {transcript_path}")

# ==========================================
# PAGE 4: FEATURE ANALYSIS
# ==========================================
elif page == "Feature Analysis":
    st.title("Feature Analysis")

    # --- Symptom Frequency Heatmap by Cluster ---
    st.subheader("Symptom Presence Heatmap (Mean Score by Cluster)")

    # Exclude noise for heatmap
    df_no_noise = df[df['Cluster'] != -1]

    if len(df_no_noise) > 0:
        cluster_symptom_means = df_no_noise.groupby('Cluster')[SYMPTOM_COLS].mean()

        fig_heat = px.imshow(
            cluster_symptom_means.T,
            labels=dict(x="Cluster", y="Symptom", color="Mean Score"),
            x=[f"Cluster {c}" for c in cluster_symptom_means.index],
            y=cluster_symptom_means.columns.tolist(),
            color_continuous_scale="YlOrRd",
            aspect="auto",
            title="Which symptoms define each cluster?"
        )
        fig_heat.update_layout(height=700)
        st.plotly_chart(fig_heat, use_container_width=True)
    else:
        st.warning("All points are noise — no clusters to analyze.")

    st.markdown("---")

    # --- Symptom Correlation ---
    st.subheader("Symptom Co-occurrence Matrix")
    # Binary presence (score >= 2)
    binary = (df[SYMPTOM_COLS] >= 2).astype(int)
    corr = binary.corr()

    fig_corr = px.imshow(
        corr,
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1,
        title="Which symptoms tend to appear together?",
        aspect="auto"
    )
    fig_corr.update_layout(height=700)
    st.plotly_chart(fig_corr, use_container_width=True)

    st.markdown("---")

    # --- Overall Symptom Distribution ---
    st.subheader("Overall Symptom Distribution")
    total_present = (df[SYMPTOM_COLS] == 2).sum().sort_values(ascending=True)
    total_negated = (df[SYMPTOM_COLS] == 1).sum().reindex(total_present.index)

    fig_dist = go.Figure()
    fig_dist.add_trace(go.Bar(name='Present (2)', y=total_present.index, x=total_present.values,
                              orientation='h', marker_color='#2ecc71'))
    fig_dist.add_trace(go.Bar(name='Negated (1)', y=total_negated.index, x=total_negated.values,
                              orientation='h', marker_color='#e74c3c'))
    fig_dist.update_layout(barmode='stack', height=600, title="Symptom Mentions Across All 272 Patients",
                           xaxis_title="Count", yaxis_title="Symptom")
    st.plotly_chart(fig_dist, use_container_width=True)
