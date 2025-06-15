import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import base64
from io import BytesIO

# --- SETTINGS ---
st.set_page_config(page_title="Pathwise Career AI", layout="wide", page_icon="💼")

# --- LOGO + HEADER ---
col_logo, col_title = st.columns([1, 8])
with col_logo:
    st.image("PathWise_Logo.png", width=80)
with col_title:
    st.title("Pathwise: AI-Powered Career Recommendation")
    st.caption("Empowering students, graduates, and job seekers with data-driven insights")

# --- Load saved model ---
@st.cache_resource
def load_model():
    with open('cosine_similarity_model.sav', 'rb') as f:
        model = pickle.load(f)
    return model

model = load_model()
scaler = model['scaler']
X_train = model['X_train']
X_train_scaled = model['X_train_scaled']
df_model = model['df']

# --- Load datasets ---
df_data = pd.read_csv("Data_final.csv")
df_info = pd.read_csv("career_info_enhanced.csv")

# --- Recommendation Function ---
def recommend_careers(user_features, num_recommendations=5):
    user_df = pd.DataFrame([user_features], columns=X_train.columns)
    user_scaled = scaler.transform(user_df)
    similarity = cosine_similarity(user_scaled, X_train_scaled)
    similarity_scores = similarity[0]
    indices = similarity_scores.argsort()[::-1][:num_recommendations]
    recommended = [(df_model['Career'].iloc[i], similarity_scores[i]) for i in indices]
    return recommended, similarity_scores

# --- Sidebar Navigation ---
st.sidebar.title("🧭 Navigasi")
page = st.sidebar.radio("Pilih Halaman:", ["Wizard Rekomendasi", "Visualisasi Data", "PCA Karakteristik"])

# Optional: Career Info Manual Search
with st.sidebar.expander("🔍 Cari Info Karier Manual"):
    selected_career = st.selectbox("Pilih Karier", df_info["Career"].unique())
    info_row = df_info[df_info["Career"] == selected_career].squeeze()
    st.markdown(f"📘 **Deskripsi**: {info_row.get('Deskripsi', '-')}")
    st.markdown(f"🧠 **Tips**: {info_row.get('Tips', '-')}")
    st.markdown(f"💸 **Rata-rata Gaji**: {info_row.get('Rata_rata_gaji', '-')}")
    st.markdown(f"🎓 **Kualifikasi Formal**: {info_row.get('Kualifikasi_Formal', '-')}")

# --- Wizard Page ---
if page == "Wizard Rekomendasi":
    st.subheader("📊 Step 1: Masukkan Nilai Karakteristik dan Aptitude Anda")
    step_progress = st.progress(0)

    user_input = []
    col1, col2 = st.columns(2)
    for i, feature in enumerate(X_train.columns):
        with (col1 if i % 2 == 0 else col2):
            value = st.number_input(f"{feature}", min_value=0.0, max_value=10.0, step=0.1, value=5.0)
            user_input.append(value)
    step_progress.progress(50)

    top_n = st.slider("🔢 Step 2: Pilih Jumlah Rekomendasi", min_value=1, max_value=20, value=5)
    step_progress.progress(80)

    if st.button("🚀 Lihat Rekomendasi Karier"):
        step_progress.progress(100)
        recommendations, sim_scores = recommend_careers(user_input, top_n)

        st.subheader("🎯 Rekomendasi Karier Teratas:")
        for i, (career, score) in enumerate(recommendations, 1):
            info_row = df_info[df_info["Career"] == career].squeeze()
            deskripsi = info_row.get("Deskripsi", "Tidak ada deskripsi.")
            tips = info_row.get("Tips", "Tidak ada tips.")
            gaji = info_row.get("Rata_rata_gaji", "N/A")
            kualifikasi = info_row.get("Kualifikasi_Formal", "Tidak tersedia.")

            with st.expander(f"**{i}. {career}** — Skor Kemiripan: `{score:.4f}`"):
                st.markdown(f"📘 **Deskripsi**: {deskripsi}")
                st.markdown(f"🧠 **Tips**: {tips}")
                st.markdown(f"💸 **Rata-rata Gaji**: {gaji}")
                st.markdown(f"🎓 **Kualifikasi Formal**: {kualifikasi}")

        st.subheader("🔥 Kontribusi Fitur terhadap Kemiripan")
        df_feat = pd.DataFrame({
            "Fitur": X_train.columns,
            "User": user_input,
            "Mean Dataset": X_train.mean().values
        })
        fig_contrib = px.bar(df_feat, x="Fitur", y=["User", "Mean Dataset"], barmode="group")
        st.plotly_chart(fig_contrib, use_container_width=True)

        st.subheader("📥 Simpan Hasil ke Excel")
        out_df = pd.DataFrame(recommendations, columns=["Karier", "Skor Kemiripan"])
        towrite = BytesIO()
        out_df.to_excel(towrite, index=False, sheet_name="Rekomendasi")
        towrite.seek(0)
        b64 = base64.b64encode(towrite.read()).decode()
        st.markdown(f'<a href="data:application/octet-stream;base64,{b64}" download="rekomendasi_karier.xlsx">📎 Unduh Rekomendasi</a>', unsafe_allow_html=True)

# --- Visualization Page ---
elif page == "Visualisasi Data":
    st.subheader("📊 Eksplorasi dan Visualisasi Fitur")
    feature_to_plot = st.selectbox("Pilih fitur untuk eksplorasi:", df_data.columns[:-1])
    fig = px.histogram(df_data, x=feature_to_plot, color='Career', barmode='overlay', marginal='box', nbins=20)
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(df_data.describe(include='all'))

# --- PCA Clustering ---
elif page == "PCA Karakteristik":
    st.subheader("🔬 PCA & Clustering Karakteristik")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_train_scaled)
    df_pca = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
    df_pca["Career"] = df_model["Career"]
    fig = px.scatter(df_pca, x="PC1", y="PC2", color="Career", title="Visualisasi 2D PCA dari Data Karakteristik")
    st.plotly_chart(fig, use_container_width=True)

# --- FOOTER ---
st.markdown("""
---
🎓 *Dikembangkan oleh Abdul Majid — Versi Beta untuk demonstrasi akademik dan eksplorasi AI rekomendasi karier.*
""")
