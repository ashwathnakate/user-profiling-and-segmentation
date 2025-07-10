import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import numpy as np

# ─── App Config ───
st.set_page_config(page_title="User Profiling & Segmentation", layout="wide")

@st.cache_data
def load_data(path):
    return pd.read_csv(path)

@st.cache_resource
def build_pipeline(n_clusters):
    num = ['Time Spent Online (hrs/weekday)', 'Time Spent Online (hrs/weekend)',
           'Likes and Reactions', 'Click-Through Rates (CTR)']
    cat = ['Age', 'Gender', 'Income Level']
    pre = ColumnTransformer([
        ('num', StandardScaler(), num),
        ('cat', OneHotEncoder(), cat),
    ])
    return Pipeline([
        ('pre', pre),
        ('kmeans', KMeans(n_clusters=n_clusters, random_state=42, n_init=10))
    ])

# ─── Sidebar ───
st.sidebar.header("Settings")
csv_path = st.sidebar.text_input("CSV path or URL", "user_profiles_for_ads.csv")
clusters = st.sidebar.slider("Number of clusters", 2, 10, 5)

# ─── Main ───
st.title("👥 User Profiling & Segmentation")

try:
    df = load_data(csv_path)
    st.write(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")

    pipeline = build_pipeline(clusters)
    df['Cluster'] = pipeline.fit_predict(df)

    # ── Elbow Method ──
    st.subheader("Optimal Number of Clusters: Elbow Method & Silhouette Score")
    wcss = []
    silhouette_scores = []
    K_range = range(2, 11)
    for k in K_range:
        pipe = build_pipeline(k)
        preds = pipe.fit_predict(df)
        wcss.append(pipe.named_steps['kmeans'].inertia_)
        silhouette_scores.append(silhouette_score(pipe.named_steps['pre'].transform(df), preds))

    fig_elbow = go.Figure()
    fig_elbow.add_trace(go.Scatter(x=list(K_range), y=wcss, mode='lines+markers', name='WCSS'))
    fig_elbow.add_trace(go.Scatter(x=list(K_range), y=silhouette_scores, mode='lines+markers', name='Silhouette'))
    fig_elbow.update_layout(title="Elbow & Silhouette Analysis", xaxis_title="Number of Clusters", yaxis_title="Score")
    st.plotly_chart(fig_elbow)

    # ── Cluster Counts ──
    st.subheader("Cluster Counts")
    cluster_counts = df['Cluster'].value_counts().reset_index()
    cluster_counts.columns = ['Cluster', 'Count']
    fig_bar = px.bar(cluster_counts, x='Cluster', y='Count', color='Cluster', title="Cluster Counts")
    st.plotly_chart(fig_bar)

    # ── Cluster Centers ──
    st.subheader("Cluster Centers (numeric features)")
    centers = pipeline.named_steps['kmeans'].cluster_centers_
    num_features = pipeline.named_steps['pre'].transformers_[0][2]
    center_df = pd.DataFrame(centers[:, :len(num_features)], columns=num_features)
    st.dataframe(center_df.style.background_gradient(axis=1))

    # ── Radar Chart ──
    st.subheader("Cluster Profiles: Radar Chart")
    radar_fig = go.Figure()
    for i in range(center_df.shape[0]):
        radar_fig.add_trace(go.Scatterpolar(r=center_df.iloc[i], theta=num_features, fill='toself', name=f'Cluster {i}'))
    radar_fig.update_layout(polar=dict(radialaxis=dict(visible=True)), showlegend=True)
    st.plotly_chart(radar_fig)

    # ── Scatter Plot ──
    st.subheader("Scatter plot: Clusters (PCA 2D)")
    X = pipeline.named_steps['pre'].transform(df)
    pca = PCA(2, random_state=42)
    coords = pca.fit_transform(X)
    pca_df = pd.DataFrame(coords, columns=['PC1', 'PC2'])
    pca_df['Cluster'] = df['Cluster']
    fig_scatter = px.scatter(pca_df, x='PC1', y='PC2', color=pca_df['Cluster'].astype(str), title="PCA Cluster Scatter")
    st.plotly_chart(fig_scatter)

    # ── Boxplots ──
    st.subheader("Boxplots by Cluster")
    selected_num_feature = st.selectbox("Select Numeric Feature", num_features)
    fig_box = px.box(df, x='Cluster', y=selected_num_feature, color=df['Cluster'].astype(str), title=f"Boxplot of {selected_num_feature} by Cluster")
    st.plotly_chart(fig_box)

    # ── Cluster Mean Heatmap ──
    st.subheader("Cluster-wise Mean Values")
    cluster_means = df.groupby('Cluster')[num_features].mean().reset_index()
    fig_means = px.imshow(cluster_means.set_index('Cluster'), text_auto=True, aspect='auto', title="Cluster-wise Mean Heatmap")
    st.plotly_chart(fig_means)

    # ── Cluster Summary Cards ──
    st.subheader("Cluster Summaries")
    for c in sorted(df['Cluster'].unique()):
        cluster_data = df[df['Cluster'] == c]
        st.markdown(f"### Cluster {c} Summary")
        st.write(f"Total Users: {len(cluster_data)}")
        st.write(cluster_data[num_features].mean().round(2))
        for cat_col in ['Age', 'Gender', 'Income Level']:
            st.write(f"Most Common {cat_col}: {cluster_data[cat_col].mode()[0]}")

    # ── Download Button ──
    st.subheader("Download Clustered Data")
    csv_download = df.to_csv(index=False).encode('utf-8')
    st.download_button("Download CSV", data=csv_download, file_name='clustered_users.csv', mime='text/csv')

    with st.expander("📋 View full data with cluster labels"):
        st.dataframe(df)

except Exception as e:
    st.error(f"Error loading data:\n{e}")
