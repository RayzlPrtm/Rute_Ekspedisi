import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import folium
from streamlit_folium import st_folium
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import os

# Konfigurasi Streamlit
st.set_page_config(page_title="Prediksi Keterlambatan Ekspedisi", page_icon="🚚", layout="wide")
st.title("🚚 Prediksi Keterlambatan Pengiriman Ekspedisi")

# Load data ekspedisi
file_path = "dataset_rute_ekspedisi.xlsx"
if not os.path.exists(file_path):
    st.error(f"❌ File '{file_path}' tidak ditemukan.")
    st.stop()

data = pd.read_excel(file_path)

st.subheader("📁 Contoh Data Rute Ekspedisi")
st.dataframe(data.tail())

# ====================== PETA RUTE ======================
st.subheader("🗺️ Visualisasi Peta Rute Ekspedisi")

# Koordinat sederhana untuk 5 kota
kota_koordinat = {
    "Jakarta": [-6.2, 106.8],
    "Bandung": [-6.9, 107.6],
    "Yogyakarta": [-7.8, 110.4],
    "Surabaya": [-7.3, 112.7],
    "Semarang": [-7.0, 110.4]
}

# Ambil 1 data terakhir untuk divisualisasikan
if not data.empty:
    asal = data.iloc[-1]["Asal"]
    tujuan = data.iloc[-1]["Tujuan"]
    lokasi_asal = kota_koordinat.get(asal)
    lokasi_tujuan = kota_koordinat.get(tujuan)

    if lokasi_asal and lokasi_tujuan:
        peta = folium.Map(location=lokasi_asal, zoom_start=6)
        folium.Marker(lokasi_asal, tooltip=f"Asal: {asal}", icon=folium.Icon(color='green')).add_to(peta)
        folium.Marker(lokasi_tujuan, tooltip=f"Tujuan: {tujuan}", icon=folium.Icon(color='red')).add_to(peta)
        folium.PolyLine(locations=[lokasi_asal, lokasi_tujuan], color='blue').add_to(peta)
        st_folium(peta, width=700, height=400)
    else:
        st.info("🛈 Koordinat kota tidak tersedia untuk peta.")

# ====================== FORM INPUT TAMBAHAN DATA EKSPEDISI ======================
st.subheader("➕ Tambah Data Ekspedisi Baru")
with st.form("form_tambah"):
    col1, col2 = st.columns(2)
    with col1:
        asal = st.selectbox("Kota Asal", list(kota_koordinat.keys()))
        tujuan = st.selectbox("Kota Tujuan", list(kota_koordinat.keys()))
        jarak = st.number_input("Jarak (km)", min_value=10.0, max_value=1000.0, value=200.0)
        cuaca = st.selectbox("Cuaca", ["Cerah", "Hujan", "Berawan"])
    with col2:
        kendaraan = st.selectbox("Kendaraan", ["Truk", "Pickup", "Van", "Motor Box"])
        waktu_berangkat = st.slider("Waktu Berangkat (jam)", 4.0, 10.0, 6.5, step=0.1)
        kecepatan = st.number_input("Kecepatan (km/jam)", min_value=20.0, max_value=120.0, value=60.0)

    simpan = st.form_submit_button("💾 Simpan Data")

    if simpan:
        durasi = jarak / kecepatan
        waktu_tiba = waktu_berangkat + durasi
        keterlambatan = max((waktu_tiba - 8.0) * 60, 0)
        status = "Ya" if keterlambatan > 0 else "Tidak"

        data_baru = {
            "Asal": asal,
            "Tujuan": tujuan,
            "Jarak (km)": jarak,
            "Waktu Berangkat (jam)": waktu_berangkat,
            "Cuaca": cuaca,
            "Kendaraan": kendaraan,
            "Kecepatan (km/jam)": kecepatan,
            "Durasi (jam)": round(durasi, 2),
            "Waktu Tiba (jam)": round(waktu_tiba, 2),
            "Keterlambatan (menit)": round(keterlambatan, 0),
            "Status Keterlambatan": status
        }

        data = pd.concat([data, pd.DataFrame([data_baru])], ignore_index=True)
        data.to_excel(file_path, index=False)
        st.success("✅ Data ekspedisi baru berhasil ditambahkan.")

# ===================== PREDIKSI =====================

# Encoding dan preprocessing
data_encoded = data.copy()
le_cuaca = LabelEncoder()
le_kendaraan = LabelEncoder()
le_status = LabelEncoder()

data_encoded["Cuaca"] = le_cuaca.fit_transform(data_encoded["Cuaca"])
data_encoded["Kendaraan"] = le_kendaraan.fit_transform(data_encoded["Kendaraan"])
data_encoded["Status Keterlambatan"] = le_status.fit_transform(data_encoded["Status Keterlambatan"])

fitur = ["Jarak (km)", "Waktu Berangkat (jam)", "Cuaca", "Kendaraan"]
X = data_encoded[fitur]
y = data_encoded["Status Keterlambatan"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
akurasi = accuracy_score(y_test, model.predict(X_test))

# Evaluasi
st.subheader("📊 Evaluasi Model Prediksi")
col1, col2 = st.columns(2)
with col1:
    st.metric("Akurasi Model", f"{akurasi*100:.2f}%")
with col2:
    fig, ax = plt.subplots()
    data["Status Keterlambatan"].value_counts().plot.pie(
        autopct='%1.1f%%',
        startangle=90,
        colors=["#ff9999", "#66b3ff"],
        ax=ax
    )
    ax.axis("equal")
    ax.set_title("Distribusi Keterlambatan")
    st.pyplot(fig)

# Form Prediksi
st.subheader("📝 Prediksi Rute Ekspedisi Baru")

with st.form("form_prediksi"):
    col1, col2 = st.columns(2)
    with col1:
        jarak = st.slider("Jarak (km)", 10.0, 1000.0, 150.0, step=1.0)
        berangkat = st.slider("Waktu Berangkat (jam)", 4.0, 10.0, 6.5, step=0.1)
    with col2:
        cuaca = st.selectbox("Cuaca", le_cuaca.classes_)
        kendaraan = st.selectbox("Jenis Kendaraan", le_kendaraan.classes_)

    submit = st.form_submit_button("🔍 Prediksi")

    if submit:
        input_data = np.array([[
            jarak,
            berangkat,
            le_cuaca.transform([cuaca])[0],
            le_kendaraan.transform([kendaraan])[0]
        ]])
        input_scaled = scaler.transform(input_data)
        hasil = model.predict(input_scaled)
        label = le_status.inverse_transform(hasil)[0]

        if label == "Tidak":
            st.success("✅ Diprediksi TEPAT WAKTU")
        else:
            st.error("❗ Diprediksi TERLAMBAT dalam pengiriman")

st.markdown("---")
st.markdown("📦 Dibuat untuk *Pembacaan Rute Ekspedisi Berbasis Data Geospasial*")
