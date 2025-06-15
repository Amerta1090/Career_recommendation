import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
from sklearn.metrics.pairwise import cosine_similarity

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

# --- Load dataset for visualization ---
df_data = pd.read_csv("Data_final.csv")

# --- Recommendation Function ---
def recommend_careers(user_features, num_recommendations=5):
    user_df = pd.DataFrame([user_features], columns=X_train.columns)
    user_scaled = scaler.transform(user_df)
    similarity = cosine_similarity(user_scaled, X_train_scaled)
    similarity_scores = similarity[0]
    indices = similarity_scores.argsort()[::-1][:num_recommendations]
    recommended = [(df_model['Career'].iloc[i], similarity_scores[i]) for i in indices]
    return recommended

# --- Sidebar Navigation ---
st.sidebar.title("📊 Navigasi")
page = st.sidebar.radio("Pilih Halaman:", ["Rekomendasi Karier", "Visualisasi Data"])

# --- Career Recommendation Page ---
if page == "Rekomendasi Karier":
    st.title("💼 Sistem Rekomendasi Karier (Float-Input Edition)")
    st.markdown("Masukkan skor kamu (dalam pecahan diperbolehkan, antara 0-10):")

    user_input = []
    col1, col2 = st.columns(2)
    for i, feature in enumerate(X_train.columns):
        with (col1 if i % 2 == 0 else col2):
            value = st.number_input(f"{feature}", min_value=0.0, max_value=10.0, step=0.1, value=5.0)
            user_input.append(value)

    top_n = st.slider("🔢 Jumlah rekomendasi:", min_value=1, max_value=20, value=5)

    if st.button("🚀 Temukan Karier Cocok"):
        recommendations = recommend_careers(user_input, top_n)
        st.subheader("🎯 Rekomendasi Karier Teratas:")
        for i, (career, score) in enumerate(recommendations, 1):
            st.success(f"{i}. {career} — 🔢 Skor Kemiripan: {score:.4f}")

# --- Data Visualization Page ---
elif page == "Visualisasi Data":
    st.title("📈 Eksplorasi dan Visualisasi Data")
    st.markdown("Analisis data berdasarkan dimensi OCEAN dan Aptitude Test.")

    feature_to_plot = st.selectbox("Pilih fitur untuk eksplorasi:", df_data.columns[:-1])

    fig = px.histogram(df_data, x=feature_to_plot, color='Career', barmode='overlay',
                       marginal='box', nbins=20,
                       title=f"Distribusi {feature_to_plot} berdasarkan Karier")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("📌 Statistik Deskriptif")
    st.dataframe(df_data.describe(include='all'))
