import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

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

    st.subheader("Cluster Counts")
    cluster_counts = df['Cluster'].value_counts().reset_index()
    cluster_counts.columns = ['Cluster', 'Count']
    fig_bar = px.bar(cluster_counts, x='Cluster', y='Count', color='Cluster', title="Cluster Counts")
    st.plotly_chart(fig_bar)

    st.subheader("Cluster Centers (numeric features)")
    centers = pipeline.named_steps['kmeans'].cluster_centers_
    num_features = pipeline.named_steps['pre'].transformers_[0][2]
    center_df = pd.DataFrame(centers[:, :len(num_features)], columns=num_features)
    st.dataframe(center_df.style.background_gradient(axis=1))

    st.subheader("Scatter plot: Clusters (PCA 2D)")
    X = pipeline.named_steps['pre'].transform(df)
    pca = PCA(2, random_state=42)
    coords = pca.fit_transform(X)
    pca_df = pd.DataFrame(coords, columns=['PC1', 'PC2'])
    pca_df['Cluster'] = df['Cluster']
    fig_scatter = px.scatter(pca_df, x='PC1', y='PC2', color=pca_df['Cluster'].astype(str), title="PCA Cluster Scatter")
    st.plotly_chart(fig_scatter)

    st.subheader("Boxplots by Cluster")
    selected_num_feature = st.selectbox("Select Numeric Feature", num_features)
    fig_box = px.box(df, x='Cluster', y=selected_num_feature, color=df['Cluster'].astype(str), title=f"Boxplot of {selected_num_feature} by Cluster")
    st.plotly_chart(fig_box)

    st.subheader("Cluster-wise Mean Values")
    cluster_means = df.groupby('Cluster')[num_features].mean().reset_index()
    fig_means = px.imshow(cluster_means.set_index('Cluster'), text_auto=True, aspect='auto', title="Cluster-wise Mean Heatmap")
    st.plotly_chart(fig_means)

    with st.expander("📋 View full data with cluster labels"):
        st.dataframe(df)

except Exception as e:
    st.error(f"Error loading data:\n{e}")
