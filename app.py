import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import base64
from io import BytesIO

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Pathwise Career AI",
    layout="wide",
    page_icon="PathWise_Logo.png",
    initial_sidebar_state="collapsed"
)

st.markdown("""
    <style>
    /* === GLOBAL FONT & BASE === */
    html, body, [class*="css"]  {
        font-family: 'Segoe UI', sans-serif;
        font-size: 16px;
    }

    /* === HEADER STYLING === */
    .stApp > header {
        background: linear-gradient(to right, #0077B5, #00BFA6);
        color: white;
        padding: 1rem 2rem;
        border-radius: 0 0 12px 12px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    }

    /* === BUTTON STYLING === */
    div.stButton > button {
        background: linear-gradient(to right, #0077B5, #00BFA6);
        color: white;
        border: none;
        padding: 0.6rem 1.2rem;
        border-radius: 10px;
        font-weight: bold;
        transition: 0.3s ease-in-out;
    }
    div.stButton > button:hover {
        background: linear-gradient(to right, #00BFA6, #0077B5);
        transform: scale(1.03);
    }

    /* === EXPANDER STYLING === */
    .streamlit-expanderHeader {
        font-size: 1.1rem;
        font-weight: 600;
        color: #0077B5;
    }

    /* === CUSTOM CARD FOR RECOMMENDATIONS === */
    .career-card {
        background: #ffffff;
        border-left: 6px solid #0077B5;
        border-radius: 12px;
        padding: 1rem 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 3px 10px rgba(0,0,0,0.08);
    }

    /* === CUSTOM DOWNLOAD BUTTON === */
    a.stButton {
        background: linear-gradient(to right, #28A745, #218838);
        color: white !important;
        text-align: center;
        display: inline-block;
        padding: 10px 24px;
        border-radius: 8px;
        text-decoration: none;
        font-weight: 600;
    }

    a.stButton:hover {
        background: linear-gradient(to right, #218838, #28A745);
    }

    /* === SLIDER STYLING === */
    .stSlider > div[data-baseweb="slider"] {
        padding: 0.8rem;
        border-radius: 12px;
        margin-top: 0.8rem;
    }

    /* === PROGRESS BAR === */
    .stProgress > div > div > div > div {
        background: linear-gradient(to right, #0077B5, #00BFA6);
    }

    /* === NUMBER INPUT FIELD === */
    input[type="number"] {
        border-radius: 8px;
        padding: 8px;
        border: 1px solid #ccc;
    }

    </style>
""", unsafe_allow_html=True)


# --- LOGO + HEADER ---
col_logo, col_title = st.columns([1, 8])
with col_logo:
    st.image("PathWise_Logo.png", width=100)
with col_title:
    st.title("Pathwise: AI-Powered Career Recommendation")
    st.caption("Empowering students, graduates, and job seekers with data-driven insights")

# --- Load saved model ---
@st.cache_resource
def load_model():
    """Loads the pre-trained model components."""
    try:
        with open('cosine_similarity_model.sav', 'rb') as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        st.error("Error: Model file 'cosine_similarity_model.sav' not found. Please ensure it's in the same directory.")
        st.stop() # Stop the app if model isn't found
    except Exception as e:
        st.error(f"Error loading model: {e}. Please check the model file.")
        st.stop()

model = load_model()
scaler = model['scaler']
X_train = model['X_train']
X_train_scaled = model['X_train_scaled']
df_model = model['df']

# --- Load datasets ---
@st.cache_data
def load_data():
    """Loads and caches the dataframes."""
    try:
        df_data = pd.read_csv("Data_final.csv")
        df_info = pd.read_csv("career_info_enhanced.csv")
        return df_data, df_info
    except FileNotFoundError:
        st.error("Error: Data files 'Data_final.csv' or 'career_info_enhanced.csv' not found. Please ensure they are in the same directory.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading data: {e}. Please check your CSV files.")
        st.stop()

df_data, df_info = load_data()


# --- Recommendation Function ---
def recommend_careers(user_features, num_recommendations=5):
    """
    Recommends careers based on user features using cosine similarity.
    Scales user input and returns top N recommendations with similarity scores.
    """
    user_df = pd.DataFrame([user_features], columns=X_train.columns)
    user_scaled = scaler.transform(user_df)
    similarity = cosine_similarity(user_scaled, X_train_scaled)
    similarity_scores = similarity[0]
    indices = similarity_scores.argsort()[::-1][:num_recommendations]
    recommended = [(df_model['Career'].iloc[i], similarity_scores[i]) for i in indices]
    return recommended, similarity_scores

# --- Sidebar Navigation ---
st.sidebar.title("🧭 Navigasi")
if "page" not in st.session_state:
    st.session_state.page = "Tentang Aplikasi Ini"

page = st.sidebar.radio(
    "Pilih Halaman:",
    ["Wizard Rekomendasi", "Visualisasi Data", "PCA Karakteristik", "Tentang Aplikasi Ini"],
    index=["Wizard Rekomendasi", "Visualisasi Data", "PCA Karakteristik", "Tentang Aplikasi Ini"].index(st.session_state.page),
    key="page_radio"
)


# Optional: Career Info Manual Search
with st.sidebar.expander("🔍 Cari Info Karier Manual"):
    selected_career = st.selectbox("Pilih Karier", df_info["Career"].unique())
    info_row = df_info[df_info["Career"] == selected_career].squeeze()
    st.markdown(f"**📘 Deskripsi**: {info_row.get('Deskripsi', 'Tidak ada deskripsi.')}")
    st.markdown(f"**🧠 Tips**: {info_row.get('Tips', 'Tidak ada tips.')}")
    st.markdown(f"**💸 Rata-rata Gaji**: {info_row.get('Rata_rata_gaji', 'N/A')}")
    st.markdown(f"**🎓 Kualifikasi Formal**: {info_row.get('Kualifikasi_Formal', 'Tidak tersedia.')}")

# --- Initialize session state for wizard steps ---
if "wizard_step" not in st.session_state:
    st.session_state.wizard_step = 0
if "user_input_features" not in st.session_state:
    # Initialize with default values (e.g., mean or 5.0)
    st.session_state.user_input_features = {feature: 5.0 for feature in X_train.columns}
if "num_recommendations" not in st.session_state:
    st.session_state.num_recommendations = 1

# --- Wizard Page ---
if page == "Wizard Rekomendasi":
    st.subheader("💡 Wizard Rekomendasi Karier")

    # Calculate progress based on steps
    total_steps = 3 # Step 1: Input, Step 2: Top N, Step 3: View Results
    current_progress = (st.session_state.wizard_step / total_steps) * 100
    step_progress = st.progress(int(current_progress), text=f"Progress: {int(current_progress)}%")

    # Step 0: Welcome and Start
    if st.session_state.wizard_step == 0:
        st.markdown("""
        Selamat datang di **Wizard Rekomendasi Karier Pathwise**! 🚀
        Aplikasi ini akan membantu Anda menemukan jalur karier yang cocok berdasarkan karakteristik dan aptitude Anda.
        Ikuti langkah-langkah mudah di bawah ini untuk mendapatkan rekomendasi terbaik Anda.
        """)
        st.info("Klik 'Mulai Wizard' untuk memulai perjalanan Anda.")
        if st.button("Mulai Wizard", key="start_wizard_0"):
            st.session_state.wizard_step = 1
            st.rerun()

    # Step 1: Input Characteristics and Aptitude
    elif st.session_state.wizard_step == 1:
        st.markdown("### 📊 Langkah 1: Masukkan Nilai Karakteristik dan Aptitude Anda")
        st.write("Silakan berikan nilai dari **0 hingga 10** untuk setiap karakteristik dan aptitude yang paling menggambarkan diri Anda.")

        user_input_temp = {}
        col1, col2 = st.columns(2)

        for i, feature in enumerate(X_train.columns):
            default_value = st.session_state.user_input_features.get(feature, 5.0) # Ensure default is from session state
            with (col1 if i % 2 == 0 else col2):
                value = st.number_input(f"**{feature}**", min_value=0.0, max_value=10.0, step=0.1, value=float(default_value), key=f"input_{feature}")
                user_input_temp[feature] = value

        st.markdown("---")
        if st.button("Lanjutkan ke Langkah 2", key="next_step_1"):
            st.session_state.user_input_features = user_input_temp
            st.session_state.wizard_step = 2
            st.rerun()

    # Step 2: Choose Number of Recommendations
    elif st.session_state.wizard_step == 2:
        st.markdown("### 🔢 Langkah 2: Pilih Jumlah Rekomendasi")
        st.write("Berapa banyak rekomendasi karier teratas yang ingin Anda lihat?")

        top_n_value = st.slider(
            "Jumlah Rekomendasi",
            min_value=1,
            max_value=5,
            value=st.session_state.num_recommendations,
            key="num_reco_slider",
            help="Geser untuk memilih jumlah karier yang direkomendasikan."
        )
        st.session_state.num_recommendations = top_n_value

        st.markdown("---")
        col_buttons = st.columns(2)
        with col_buttons[0]:
            if st.button("⬅️ Kembali ke Langkah 1", key="back_step_2"):
                st.session_state.wizard_step = 1
                st.rerun()
        with col_buttons[1]:
            if st.button("Lihat Rekomendasi Karier 🚀", key="view_reco"):
                st.session_state.wizard_step = 3
                st.rerun()

    # Step 3: Display Recommendations and Results
    elif st.session_state.wizard_step == 3:
        st.markdown("### 🚀 Hasil Rekomendasi Karier Anda")
        st.success("🎉 Rekomendasi telah berhasil dibuat!")

        user_input_list = [st.session_state.user_input_features[feature] for feature in X_train.columns]
        recommendations, sim_scores = recommend_careers(user_input_list, st.session_state.num_recommendations)

        if recommendations:
            st.subheader("🎯 Rekomendasi Karier Teratas:")
            for i, (career, score) in enumerate(recommendations, 1):
                info_row = df_info[df_info["Career"] == career].squeeze()
                deskripsi = info_row.get("Deskripsi", "Tidak ada deskripsi tersedia.")
                tips = info_row.get("Tips", "Tidak ada tips tersedia.")
                gaji = info_row.get("Rata_rata_gaji", "N/A")
                kualifikasi = info_row.get("Kualifikasi_Formal", "Tidak tersedia.")

                with st.expander(f"**{i}. {career}** — Skor Kemiripan: `{score:.4f}`"):
                    st.markdown(f"**📘 Deskripsi**: {deskripsi}")
                    st.markdown(f"**🧠 Tips**: {tips}")
                    st.markdown(f"**💸 Rata-rata Gaji**: {gaji}")
                    st.markdown(f"**🎓 Kualifikasi Formal**: {kualifikasi}")
        else:
            st.warning("Tidak ada rekomendasi yang ditemukan. Coba sesuaikan input Anda.")

        st.markdown("---")
        st.subheader("🔥 Kontribusi Fitur Anda vs. Rata-rata Dataset")
        st.info("Grafik ini menunjukkan bagaimana input Anda (nilai biru) dibandingkan dengan rata-rata nilai fitur di seluruh dataset (nilai oranye).")
        df_feat = pd.DataFrame({
            "Fitur": X_train.columns,
            "Input Anda": user_input_list,
            "Rata-rata Dataset": X_train.mean().values
        })
        fig_contrib = px.bar(
            df_feat,
            x="Fitur",
            y=["Input Anda", "Rata-rata Dataset"],
            barmode="group",
            title="Perbandingan Nilai Fitur: Input Anda vs. Rata-rata Dataset",
            color_discrete_map={"Input Anda": "#0077B5", "Rata-rata Dataset": "#FF9900"} # Custom colors
        )
        st.plotly_chart(fig_contrib, use_container_width=True)

        st.markdown("---")
        st.subheader("📥 Simpan Hasil ke Excel")
        st.write("Anda dapat mengunduh daftar rekomendasi karier ini ke file Excel.")
        out_df = pd.DataFrame(recommendations, columns=["Karier", "Skor Kemiripan"])
        towrite = BytesIO()
        out_df.to_excel(towrite, index=False, sheet_name="Rekomendasi")
        towrite.seek(0)
        b64 = base64.b64encode(towrite.read()).decode()
        href = f'<a href="data:application/octet-stream;base64,{b64}" download="rekomendasi_karier.xlsx" class="stButton" style="background-color: #28A745; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px;">📎 Unduh Rekomendasi (.xlsx)</a>'
        st.markdown(href, unsafe_allow_html=True)

        st.markdown("---")
        if st.button("Mulai Rekomendasi Baru ✨", key="start_new_reco"):
            st.session_state.wizard_step = 0
            st.session_state.user_input_features = {feature: 5.0 for feature in X_train.columns} # Reset to default
            st.session_state.num_recommendations = 5 # Reset to default
            st.rerun()

# --- Visualization Page ---
elif page == "Visualisasi Data":
    st.subheader("📊 Eksplorasi dan Visualisasi Fitur Data")
    st.info("Jelajahi distribusi karakteristik dan aptitude di seluruh dataset. Pilih fitur dari dropdown di bawah.")
    feature_to_plot = st.selectbox("Pilih fitur untuk eksplorasi:", df_data.columns[:-1])
    fig = px.histogram(
        df_data,
        x=feature_to_plot,
        color='Career',
        barmode='overlay',
        marginal='box',
        nbins=20,
        title=f"Distribusi Fitur: {feature_to_plot} per Karier"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Ringkasan Statistik Dataset")
    st.dataframe(df_data.describe(include='all').T.style.set_properties(**{'background-color': '#f8f8f8', 'color': '#333'}))


# --- PCA Clustering ---
elif page == "PCA Karakteristik":
    st.subheader("🔬 Visualisasi PCA & Clustering Karakteristik")
    st.info("Gunakan Analisis Komponen Utama (PCA) untuk mengurangi dimensi data dan melihat bagaimana karier mengelompok berdasarkan karakteristik mereka.")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_train_scaled)
    df_pca = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
    df_pca["Career"] = df_model["Career"]
    fig = px.scatter(
        df_pca,
        x="PC1",
        y="PC2",
        color="Career",
        title="Visualisasi 2D PCA dari Data Karakteristik",
        hover_data={"Career": True} # Show career name on hover
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(f"**Varian yang Dijelaskan oleh PC1:** `{pca.explained_variance_ratio_[0]:.2f}`")
    st.markdown(f"**Varian yang Dijelaskan oleh PC2:** `{pca.explained_variance_ratio_[1]:.2f}`")

# --- About Page ---
elif page == "Tentang Aplikasi Ini":
    st.subheader("ℹ️ Tentang Pathwise: AI-Powered Career Recommendation")
    st.markdown("""
    Pathwise adalah sebuah aplikasi berbasis kecerdasan buatan yang dirancang untuk membantu individu dalam menemukan jalur karier yang paling sesuai dengan karakteristik dan aptitude mereka. Dengan menganalisis input Anda dan membandingkannya dengan profil karier yang ada dalam dataset kami, Pathwise memberikan rekomendasi yang dipersonalisasi.

    ---

    #### **Bagaimana Cara Kerjanya?**
    Aplikasi ini menggunakan algoritma **Cosine Similarity** untuk mengukur kemiripan antara profil karakteristik Anda dan karakteristik yang dibutuhkan oleh berbagai karier. Data karier telah diproses dan dinormalisasi untuk memastikan perbandingan yang akurat.

    ---

    #### **Fitur Utama:**
    * **Wizard Rekomendasi:** Panduan langkah demi langkah untuk mendapatkan rekomendasi karier.
    * **Visualisasi Data:** Jelajahi distribusi fitur-fitur penting dalam dataset.
    * **PCA Karakteristik:** Lihat bagaimana berbagai karier dikelompokkan secara visual berdasarkan karakteristik utama mereka.
    * **Informasi Karier Detail:** Dapatkan deskripsi, tips, rata-rata gaji, dan kualifikasi formal untuk setiap karier.

    ---

    #### **Pengembang:**
    Aplikasi ini dikembangkan oleh **Abdul Majid** sebagai demonstrasi akademik dan eksplorasi dalam bidang AI untuk rekomendasi karier.

    ---

    Kami harap Pathwise dapat menjadi alat yang bermanfaat dalam perjalanan karier Anda!
    """)
    
    st.markdown("---")
    st.info("Siap memulai eksplorasi karier Anda?")
    if st.button("🚀 Mulai Rekomendasi Karier Sekarang"):
        st.session_state.page = "Wizard Rekomendasi"
        page = "Wizard Rekomendasi"
        st.rerun()


# --- FOOTER ---
st.markdown("""
---
🎓 *Dikembangkan oleh Abdul Majid — Versi Beta untuk demonstrasi akademik dan eksplorasi AI rekomendasi karier.*
""")
