import streamlit as st
import pandas as pd
import numpy as np
import pickle
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
df = model['df']

# --- Rekomendasi karier ---
def recommend_careers(user_features, num_recommendations=5):
    user_df = pd.DataFrame([user_features], columns=X_train.columns)
    user_scaled = scaler.transform(user_df)
    similarity = cosine_similarity(user_scaled, X_train_scaled)
    similarity_scores = similarity[0]
    indices = similarity_scores.argsort()[::-1][:num_recommendations]
    recommended = [(df['Career'].iloc[i], similarity_scores[i]) for i in indices]
    return recommended


# --- Streamlit UI ---
st.title("💼 Career Recommendation System")
st.markdown("Masukkan skor kamu untuk setiap fitur (0-10):")

user_input = []
for feature in X_train.columns:
    value = st.slider(f"{feature}", 0, 10, 5)
    user_input.append(value)

top_n = st.number_input("Jumlah rekomendasi:", min_value=1, max_value=20, value=5)

if st.button("🔍 Cari Karier Cocok"):
    recommendations = recommend_careers(user_input, top_n)
    st.subheader("🎯 Rekomendasi Karier:")
    for i, (career, score) in enumerate(recommendations, 1):
        st.write(f"{i}. {career} — 🔢 Skor Kemiripan: `{score:.4f}`")

