import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

st.set_page_config(page_title="SocialFlow - Kelompok 4", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
    html, body, [class*="css"] {
        font-family: 'Inter', 'Segoe UI', Helvetica, Arial, sans-serif;
    }
    [data-testid="stSidebar"] {
        background-color: #0f172a;
    }
    
    .nav-btn {
        display: flex;
        align-items: center;
        background-color: transparent;
        color: #94a3b8 !important;
        text-decoration: none !important;
        padding: 12px 15px;
        margin-bottom: 10px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        font-size: 14px;
        font-weight: 500;
        height: 50px;
        box-sizing: border-box;
        transition: all 0.3s ease;
    }
    .nav-btn:hover {
        background-color: #1e293b !important;
        color: #ffffff !important;
        padding-left: 22px;
    }
    .nav-btn-active {
        background-color: #3b82f6 !important;
        border-color: #3b82f6 !important;
        color: #ffffff !important;
        font-weight: 600;
    }
   
    div[data-testid="stMetricValue"] {
        font-size: 20px;
        font-weight: 800;
        color: #3b82f6;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 11px;
        color: #64748b;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    div[data-testid="stForm"] {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 30px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        transition: all 0.3s ease;
    }
    div[data-testid="stForm"]:hover {
        border-color: #3b82f6;
        box-shadow: 0 10px 15px -3px rgba(59, 130, 246, 0.1);
    }
    div.main .stButton>button {
        background-color: #3b82f6;
        color: white;
        border-radius: 8px;
        font-weight: 600;
        height: 50px;
        border: none;
        transition: all 0.3s ease;
    }
    div.main .stButton>button:hover {
        background-color: #2563eb;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.4);
    }
    </style>
    """, unsafe_allow_html=True)

# =====================================================================
# LOAD MODEL & DATASET OPERASIONAL
# =====================================================================
@st.cache_resource
def load_artifacts():
    nb_model = joblib.load('model_naive_bayes.pkl')
    lr_model = joblib.load('model_logistic_regression.pkl')
    expected_columns = joblib.load('X_columns.pkl')
    return nb_model, lr_model, expected_columns

try:
    nb_model, lr_model, expected_columns = load_artifacts()
    df_visual = pd.read_csv('social_media_with_clusters.csv') 
    df_visual['cluster_performa'] = df_visual['cluster_performa'].astype(str)
    df_elbow = pd.read_csv('elbow_data.csv')
    df_cm = pd.read_csv('confusion_matrix_data.csv')
except Exception as e:
    st.error(f"System Error: Gagal memuat file operasional. Log: {e}")
    st.stop()

# =====================================================================
# SIDEBAR NAVIGATION
# =====================================================================
with st.sidebar:
    st.markdown("<h2 style='color: white; margin-bottom: 0px;'>SocialFlow</h2>", unsafe_allow_html=True)
    st.caption(" <p style='font-size: 14px; color: #64748b; margin: 0;'>Marketing Analytics Engine</p>", unsafe_allow_html=True)
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    
    if 'page' not in st.session_state:
        st.session_state.page = "Visualisasi Clustering"
        
    query_params = st.query_params
    if "p" in query_params:
        st.session_state.page = query_params["p"]

    pages = [
        {"label": "Visualisasi Clustering", "name": "Visualisasi Clustering"},
        {"label": "Evaluasi Akurasi Model", "name": "Evaluasi Akurasi Model"},
        {"label": "Prediction Engine", "name": "Prediction Engine"},
    ]

    for p in pages:
        is_active = "nav-btn-active" if st.session_state.page == p["name"] else ""
        st.markdown(f'<a href="?p={p["name"]}" target="_self" class="nav-btn {is_active}">{p["label"]}</a>', unsafe_allow_html=True)
    
    menu = st.session_state.page
    
    st.markdown("""
        <div style='margin-top: 150px; padding-top: 20px; border-top: 1px solid rgba(255, 255, 255, 0.05); text-align: center;'>
            <p style='font-size: 12px; color: #64748b; margin: 0;'>© 2026 - Kelompok 4</p>
            <p style='font-size: 12px; color: #64748b; margin: 0;'>Sistem Informasi</p>
            <p style='font-size: 11px; color: #64748b; margin: 0;'>Telkom University Surabaya</p>
        </div>
    """, unsafe_allow_html=True)
    
# =====================================================================
# VISUALISASI CLUSTERING
# =====================================================================
if menu == "Visualisasi Clustering":
    st.title("Insight Segmentasi K-Means")
    st.markdown("Analisis objektif terhadap perilaku audiens berdasarkan interaksi historis murni tanpa label.")
    
    col_m1, col_m2, col_m3 = st.columns(3)
    with col_m1:
        st.metric("Total Konten Dianalisis", f"{len(df_visual):,}")
    with col_m2:
        st.metric("Jumlah Klaster Optimal", "2 Segmen")
    with col_m3:
        st.metric("Fokus Evaluasi Clustering", "Views, Likes, Comments, Shares")
        
    st.markdown("---")
    
    # 1. Bagian Elbow Method
    st.markdown("### Penentuan K-Optimal (Elbow Method)")
    st.caption("Elbow Method digunakan untuk menemukan jumlah klaster (k) terbaik dengan mengukur grafik penurunan SSE (Sum of Squared Errors). Titik tekukan tajam atau 'siku' pada k=2 menandakan bahwa penambahan jumlah klaster setelahnya tidak lagi memberikan pembagian kelompok data yang signifikan secara statistik.")
    fig_elbow = px.line(df_elbow, x='Jumlah Cluster (k)', y='Nilai SSE (Inertia)', markers=True, color_discrete_sequence=['#3b82f6'])
    fig_elbow.add_vline(x=2, line_dash="dash", line_color="#f82c2c")
    fig_elbow.update_layout(margin=dict(l=0, r=0, t=10, b=0), plot_bgcolor='rgba(0,0,0,0)', height=350)
    st.plotly_chart(fig_elbow, use_container_width=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.divider()
    
    # 2. Bagian Tabel Centroid Karakteristik
    st.markdown("### Centroid Tiap Klaster")
    st.caption("Nilai Centroid mewakili titik pusat atau nilai rata-rata (mean) dari fitur numerik murni (Views, Likes, Comments, Shares) pada masing-masing kelompok data. Perbandingan nilai antar-klaster ini digunakan secara objektif sebagai indikator utama untuk mendefinisikan karakteristik unik serta melabeli kelompok mana yang masuk ke dalam kategori Performa Tinggi (High Engagement) dan Performa Rendah (Low Engagement).")
    features_num = ['views', 'likes', 'comments', 'shares']
    cluster_profile = df_visual.groupby('cluster_performa')[features_num].mean().round(2)
    st.dataframe(cluster_profile, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.divider()

    # 3. Bagian Scatter Plot Sebaran Data
    st.markdown("### Visualisasi Sebaran Objek Konten (Scatter Plot Views vs Likes)")
    st.caption("Scatter Plot memvisualisasikan distribusi dan korelasi antar-dua variabel interaksi utama secara individual. Pembagian warna titik data merepresentasikan batasan ruang keputusan (decision boundary) hasil pengelompokan algoritma K-Means, yang memisahkan pola sebaran konten performa tinggi (Cluster 1) dengan performa standar (Cluster 2) berdasarkan kepadatan jarak terdekatnya.")
    fig_scatter = px.scatter(
        df_visual, x='views', y='likes', color='cluster_performa', opacity=0.6, 
        color_discrete_map={'1': '#3b82f6', '2': '#f82c2c'},
        labels={'views': 'Total Tayangan (Views)', 'likes': 'Total Suka (Likes)', 'cluster_performa': 'Klaster'},
        category_orders={"cluster_performa": ["1", "2"]}
    )
    fig_scatter.update_layout(margin=dict(l=0, r=0, t=20, b=0), plot_bgcolor='rgba(0,0,0,0.02)', height=500)
    st.plotly_chart(fig_scatter, use_container_width=True)


# =====================================================================
# EVALUASI AKURASI MODEL
# =====================================================================
elif menu == "Evaluasi Akurasi Model":
    st.title("Perbandingan Hasil Akurasi Model Klasifikasi")
    st.markdown("Menguji performa algoritma *supervised learning* dalam memetakan target akurasi berdasarkan data uji (*testing data*).")
    st.divider()
    
    df_lr_calc = df_cm[df_cm['Model'] == 'Logistic Regression']
    total_lr = df_lr_calc['Value'].sum()
    benar_lr = df_lr_calc[df_lr_calc['Actual'] == df_lr_calc['Predicted']]['Value'].sum()
    acc_lr_calc = (benar_lr / total_lr) * 100
    
    tp_lr_high = df_lr_calc[(df_lr_calc['Actual'] == 'High') & (df_lr_calc['Predicted'] == 'High')]['Value'].sum()
    fp_lr_high = df_lr_calc[(df_lr_calc['Actual'] == 'Low') & (df_lr_calc['Predicted'] == 'High')]['Value'].sum()
    fn_lr_high = df_lr_calc[(df_lr_calc['Actual'] == 'High') & (df_lr_calc['Predicted'] == 'Low')]['Value'].sum()
    tn_lr_high = df_lr_calc[(df_lr_calc['Actual'] == 'Low') & (df_lr_calc['Predicted'] == 'Low')]['Value'].sum()
    
    prec_lr_l = tn_lr_high / (tn_lr_high + fn_lr_high) if (tn_lr_high + fn_lr_high) > 0 else 0
    rec_lr_l = tn_lr_high / (tn_lr_high + fp_lr_high) if (tn_lr_high + fp_lr_high) > 0 else 0
    f1_lr_l = 2 * (prec_lr_l * rec_lr_l) / (prec_lr_l + rec_lr_l) if (prec_lr_l + rec_lr_l) > 0 else 0
    
    prec_lr_h = tp_lr_high / (tp_lr_high + fp_lr_high) if (tp_lr_high + fp_lr_high) > 0 else 0
    rec_lr_h = tp_lr_high / (tp_lr_high + fn_lr_high) if (tp_lr_high + fn_lr_high) > 0 else 0
    f1_lr_h = 2 * (prec_lr_h * rec_lr_h) / (prec_lr_h + rec_lr_h) if (prec_lr_h + rec_lr_h) > 0 else 0

    df_nb_calc = df_cm[df_cm['Model'] == 'Bernoulli Naive Bayes']
    total_nb = df_nb_calc['Value'].sum()
    benar_nb = df_nb_calc[df_nb_calc['Actual'] == df_nb_calc['Predicted']]['Value'].sum()
    acc_nb_calc = (benar_nb / total_nb) * 100
    
    tp_nb_high = df_nb_calc[(df_nb_calc['Actual'] == 'High') & (df_nb_calc['Predicted'] == 'High')]['Value'].sum()
    fp_nb_high = df_nb_calc[(df_nb_calc['Actual'] == 'Low') & (df_nb_calc['Predicted'] == 'High')]['Value'].sum()
    fn_nb_high = df_nb_calc[(df_nb_calc['Actual'] == 'High') & (df_nb_calc['Predicted'] == 'Low')]['Value'].sum()
    tn_nb_high = df_nb_calc[(df_nb_calc['Actual'] == 'Low') & (df_nb_calc['Predicted'] == 'Low')]['Value'].sum()
    
    prec_nb_l = tn_nb_high / (tn_nb_high + fn_nb_high) if (tn_nb_high + fn_nb_high) > 0 else 0
    rec_nb_l = tn_nb_high / (tn_nb_high + fp_nb_high) if (tn_nb_high + fp_nb_high) > 0 else 0
    f1_nb_l = 2 * (prec_nb_l * rec_nb_l) / (prec_nb_l + rec_nb_l) if (prec_nb_l + rec_nb_l) > 0 else 0
    
    prec_nb_h = tp_nb_high / (tp_nb_high + fp_nb_high) if (tp_nb_high + fp_nb_high) > 0 else 0
    rec_nb_h = tp_nb_high / (tp_nb_high + fn_nb_high) if (tp_nb_high + fn_nb_high) > 0 else 0
    f1_nb_h = 2 * (prec_nb_h * rec_nb_h) / (prec_nb_h + rec_nb_h) if (prec_nb_h + rec_nb_h) > 0 else 0
    
    col_acc1, col_acc2 = st.columns(2)
    with col_acc1:
        st.metric(label="Akurasi Logistic Regression", value=f"{acc_lr_calc:.2f}%", delta="Model Terbaik")
    with col_acc2:
        st.metric(label="Akurasi Bernoulli Naïve Bayes", value=f"{acc_nb_calc:.2f}%")
        
    st.divider()
   
    st.subheader("Laporan Kinerja Klasifikasi")
    tab_high, tab_low = st.tabs([" Cluster 1 (High)", " Cluster 2 (Low)"])
    
    with tab_high:
        st.markdown(f"""
        **Metrik Sukses Kelas Performa Tinggi:**
        * **Logistic Regression:**
            * *Precision:* `{prec_lr_l:.2f}`
            * *Recall:* `{rec_lr_l:.2f}`
            * *F1-Score:* `{f1_lr_l:.2f}`
        * **Bernoulli Naïve Bayes:**
            * *Precision:* `{prec_nb_l:.2f}`
            * *Recall:* `{rec_nb_l:.2f}`
            * *F1-Score:* `{f1_nb_l:.2f}`
        """)
        
    with tab_low:
        st.markdown(f"""
        **Metrik Sukses Kelas Performa Rendah:**
        * **Logistic Regression:**
            * *Precision:* `{prec_lr_h:.2f}`
            * *Recall:* `{rec_lr_h:.2f}`
            * *F1-Score:* `{f1_lr_h:.2f}`
        * **Bernoulli Naïve Bayes:**
            * *Precision:* `{prec_nb_h:.2f}`
            * *Recall:* `{rec_nb_h:.2f}`
            * *F1-Score:* `{f1_nb_h:.2f}`
        """)
        
    st.info(f" **Kesimpulan Analisis:** Logistic Regression dipilih karena unggul dalam akurasi global ({acc_lr_calc:.2f}%) serta memiliki jangkauan penemuan kasus positif tertinggi.")
    
    st.divider()
    st.markdown("<br>", unsafe_allow_html=True)
    
    st.subheader("Interactive Confusion Matrix Heatmap")
    df_cm_lr = df_cm[df_cm['Model'] == 'Logistic Regression']
    df_cm_nb = df_cm[df_cm['Model'] == 'Bernoulli Naive Bayes']
    
    st.markdown("**1. Model: Logistic Regression**")
    fig_cm_lr = px.density_heatmap(df_cm_lr, x="Predicted", y="Actual", z="Value", text_auto=True, color_continuous_scale="Oranges")
    fig_cm_lr.update_layout(coloraxis_showscale=False, height=350, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig_cm_lr, use_container_width=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    st.markdown("**2. Model: Bernoulli Naïve Bayes (Blues)**")
    fig_cm_nb = px.density_heatmap(df_cm_nb, x="Predicted", y="Actual", z="Value", text_auto=True, color_continuous_scale="Blues")
    fig_cm_nb.update_layout(coloraxis_showscale=False, height=350, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig_cm_nb, use_container_width=True)


# =====================================================================
# PREDICTION ENGINE
# =====================================================================
elif menu == "Prediction Engine":
    st.title("Modul Klasifikasi Prediktif")
    st.markdown("Sistem rekomendasi penentuan potensi performa draf konten pemasaran digital.")
    
    with st.form("engine_form"):
        st.markdown("Atribut Rencana Konten")
        col_form1, col_form2 = st.columns(2)
        
        with col_form1:
            platform = st.selectbox("Platform Distribusi", ["Instagram", "TikTok", "YouTube Shorts", "X"])
            content_type = st.selectbox("Format Media", ["video", "carousel", "image", "text"])
            
        with col_form2:
            topic = st.selectbox("Kategori Topik", ["Technology", "Sports", "Entertainment", "Education", "Politics"])
            model_choice = st.radio("Inti Algoritma Klasifikasi Penguji", ["Logistic Regression", "Bernoulli Naive Bayes"])
            
        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button("EKSEKUSI ANALISIS PREDIKTIF", use_container_width=True)

    if submitted:
        input_data = pd.DataFrame(0, index=[0], columns=expected_columns)
        col_platform = f"platform_{platform}"
        col_type = f"content_type_{content_type}"
        col_topic = f"topic_{topic}"
        
        for col in [col_platform, col_type, col_topic]:
            if col in input_data.columns:
                input_data[col] = 1

        if "Logistic Regression" in model_choice:
            prediction = lr_model.predict(input_data)[0]
        else:
            prediction = nb_model.predict(input_data)[0]
            
        st.markdown("### Laporan Hasil Analisis Sistem", unsafe_allow_html=True)
        
        rekomendasi_tambahan = ""
        if platform == "TikTok" or platform == "YouTube Shorts":
            rekomendasi_tambahan = "Prioritaskan struktur 'Hook' pada 3 detik pertama dan optimasi audio tren terkini." if content_type == "video" else f"Tantangan jangkauan organik terdeteksi. Algoritma {platform} memprioritaskan format video dibandingkan {content_type}."
        elif platform == "Instagram":
            rekomendasi_tambahan = "Format edukasi slide-by-slide efektif memicu metrik penyimpanan." if content_type == "carousel" else "Sertakan CTA spesifik pada caption untuk menginisiasi percakapan dua arah."
        elif platform == "X":
            rekomendasi_tambahan = "Aktivitas responsif di 60 menit pertama pasca-rilis sangat menentukan distribusi algoritma rekomendasi."

        if prediction == 1:
            st.success("STATUS: OPTIMAL")
            st.markdown(f"""
            **KLASTER PERFORMA TINGGI (High Engagement)**
            
            Berdasarkan kalkulasi matriks probabilitas model, kombinasi atribut konten yang Anda masukkan diproyeksikan memiliki peluang besar untuk mencapai **penyerapan audiens yang masif (interaksi dan jangkauan di atas rata-rata)** pada topik **{topic}**.
            
            * **Strategi Eksekusi ({platform}):** {rekomendasi_tambahan} Komposisi draf ini sudah memenuhi standar kelayakan algoritmik untuk dijadikan target materi utama dalam kalender editorial konten kelompok Anda.
            """)
        else:
            st.warning("STATUS: PERLU PENINJAUAN")
            st.markdown(f"""
            **KLASTER PERFORMA STANDAR (Low/Standard Engagement)**
            
            Model mendeteksi adanya risiko **kejenuhan interaksi (retensi audiens cenderung pasif)** jika kombinasi topik **{topic}** dipaksakan rilis menggunakan format **{content_type}** pada platform **{platform}**.
            
            * **Saran Perbaikan Teknis:** {rekomendasi_tambahan} Sangat disarankan untuk melakukan *A/B Testing* pada visual atau memformulasikan ulang pesan utama (*copywriting*) agar karakteristik konten lebih interaktif sebelum dipublikasikan secara resmi.
            """)