# iris_app.py — DuoEngineer Edition 🚀

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# === PAGE CONFIGURATION ===
st.set_page_config(
    page_title="Iris Dataset Analyzer",
    page_icon="🌸",
    layout="wide"
)

# === HEADER SECTION ===
st.markdown("""
<style>
.big-title {
    font-size:36px;
    font-weight:700;
    color:#5C5CFF;
}
.footer {
    position: fixed;
    bottom: 0;
    width: 100%;
    color: gray;
    background-color: #f9f9f9;
    text-align: center;
    font-size: 14px;
    padding: 10px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='big-title'>🌸 Iris Analyzer by DuoEngineer</div>", unsafe_allow_html=True)
st.subheader("🚀 Built with Python, ML logic & Streamlit magic")
st.info("👨‍💻 Developed by: **Abdul Rehman & Talha Abdul Rauf** — The DuoEngineer Team™")

# === LOAD DATA ===
@st.cache_data
def load_data():
    iris = load_iris(as_frame=True)
    df = iris.frame
    df['target'] = df['target'].replace(dict(enumerate(iris.target_names)))
    return df, iris.target_names.tolist(), iris.feature_names

df, species_list, feature_names = load_data()

# === SIDEBAR CONTROLS ===
st.sidebar.title("⚙️ Controls")
selected_species = st.sidebar.multiselect("Select Species", species_list, default=species_list)
x_feature = st.sidebar.selectbox("X-Axis Feature", feature_names, index=0)
y_feature = st.sidebar.selectbox("Y-Axis Feature", feature_names, index=2)

filtered_df = df[df['target'].isin(selected_species)]

# === RAW DATA ===
st.markdown("---")
if st.checkbox("🔍 Show Raw Dataset"):
    st.dataframe(df)

# === STATS ===
if st.checkbox("📊 Show Descriptive Statistics"):
    stats = filtered_df.groupby('target').agg(['mean', 'std', 'var'])
    st.dataframe(stats)

# === CORRELATION ===
if st.checkbox("🔗 Show Correlation Matrix"):
    corr = filtered_df.drop(columns=["target"]).corr()
    st.dataframe(corr.round(2))

# === VECTOR PLOT ===
st.markdown("---")
st.subheader("🧭 Feature Vector Plot")
size = 3 + 0.5 * len(selected_species)
fig, ax = plt.subplots(figsize=(size, size), dpi=100, layout='constrained')

plot_data = filtered_df[[x_feature, y_feature, 'target']]
fig, ax = plt.subplots(figsize=(9, 9), dpi=100, layout='constrained')
plt.tight_layout(pad=2)
scale = 2 + (len(selected_species) * 0.5)
fig, ax = plt.subplots(figsize=(scale, scale), dpi=100)

for species in selected_species:
    subset = plot_data[plot_data['target'] == species].head(5)
    origin = np.zeros((subset.shape[0], 2))
    vectors = subset[[x_feature, y_feature]].values
    colors = ['red', 'green', 'blue']  # Define colors for species
for i, species in enumerate(selected_species):
    subset = plot_data[plot_data['target'] == species].head(5)
    origin = np.zeros((subset.shape[0], 2))
    vectors = subset[[x_feature, y_feature]].values
    
    ax.quiver(*origin.T, vectors[:, 0], vectors[:, 1],
              angles='xy', scale_units='xy', scale=1,
              color=colors[i % len(colors)], label=species)
# Set X and Y limits with some margin
x_max = plot_data[x_feature].max() * 1.2
y_max = plot_data[y_feature].max() * 1.2
ax.set_xlim(0, x_max)
ax.set_ylim(0, y_max)


ax.set_xlabel(x_feature)
ax.set_ylabel(y_feature)
ax.set_title(f"{x_feature} vs {y_feature} Vectors")
ax.grid(True)
ax.legend()
with st.container():
    st.markdown(
        "<div style='text-align:center;'>",
        unsafe_allow_html=True
    )
    st.pyplot(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# === FOOTER ===
st.markdown("<div class='footer'>🚀 Made with 💻 by <strong>Abdul Rehman & Talha Abdul Rauf</strong> — DuoEngineer | All Rights Reserved © 2025</div>", unsafe_allow_html=True)
st.markdown("""
<style>
.big-title {
    font-size:36px;
    font-weight:700;
    color:#5C5CFF;
}
.footer {
    position: fixed;
    bottom: 0;
    width: 100%;
    color: #f5f5f5;
    background-color: #212121;
    text-align: center;
    font-size: 14px;
    padding: 10px;
    border-top: 2px solid #5C5CFF;
}
</style>
""", unsafe_allow_html=True)
