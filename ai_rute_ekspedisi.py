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

# Konfigurasi Aplikasi
st.set_page_config(
    page_title="Sistem Prediksi Keterlambatan Ekspedisi Seluruh Indonesia",
    page_icon="🚚", 
    layout="wide"
)
st.title("🌏 Sistem Prediksi Keterlambatan Pengiriman - Seluruh Indonesia")

# Data Geospasial untuk Seluruh Indonesia
KOTA_KOORDINAT = {
    # Sumatera
    "Banda Aceh": [5.55, 95.3167],
    "Medan": [3.5894, 98.6739],
    "Pekanbaru": [0.5097, 101.4479],
    "Padang": [-0.95, 100.3531],
    "Palembang": [-2.9911, 104.7567],
    # Jawa
    "Jakarta": [-6.2088, 106.8456],
    "Bandung": [-6.9175, 107.6191],
    "Semarang": [-6.9667, 110.4167],
    "Yogyakarta": [-7.7956, 110.3696],
    "Surabaya": [-7.2575, 112.7521],
    # Kalimantan
    "Pontianak": [-0.0275, 109.3425],
    "Banjarmasin": [-3.3199, 114.5908],
    "Balikpapan": [-1.2379, 116.8521],
    "Samarinda": [-0.5027, 117.1536],
    # Sulawesi
    "Makassar": [-5.1477, 119.4327],
    "Manado": [1.4870, 124.8355],
    "Palu": [-0.9000, 119.8500],
    # Bali & Nusa Tenggara
    "Denpasar": [-8.6562, 115.2160],
    "Mataram": [-8.5833, 116.1167],
    "Kupang": [-10.1639, 123.6028],
    # Maluku & Papua
    "Ambon": [-3.7000, 128.1667],
    "Jayapura": [-2.5333, 140.7000]
}

# Fungsi Pembantu
def haversine_distance(lat1, lon1, lat2, lon2):
    """Menghitung jarak antara dua koordinat dalam km"""
    R = 6371  # Radius bumi dalam km
    dLat = np.radians(lat2 - lat1)
    dLon = np.radians(lon2 - lon1)
    a = (np.sin(dLat/2) * np.sin(dLat/2) + 
         np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * 
         np.sin(dLon/2) * np.sin(dLon/2))
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return R * c

# Memuat Data
@st.cache_data
def load_data():
    file_path = "dataset_rute_ekspedisi.xlsx"
    if not os.path.exists(file_path):
        # Membuat dataset baru jika tidak ada
        kolom = [
            "Asal", "Tujuan", "Jarak (km)", "Waktu Berangkat (jam)", 
            "Cuaca", "Kendaraan", "Kecepatan (km/jam)", 
            "Durasi (jam)", "Waktu Tiba (jam)", 
            "Keterlambatan (menit)", "Status Keterlambatan"
        ]
        return pd.DataFrame(columns=kolom)
    return pd.read_excel(file_path)

data = load_data()

# Tampilan UI
tab1, tab2, tab3 = st.tabs(["📊 Data Ekspedisi", "🗺️ Peta Rute", "🔮 Prediksi"])

with tab1:
    st.subheader("Data Historis Pengiriman")
    st.dataframe(data, use_container_width=True)

    # Form Tambah Data
    with st.expander("➕ Tambah Data Pengiriman Baru"):
        with st.form("form_tambah_data"):
            col1, col2 = st.columns(2)
            with col1:
                asal = st.selectbox("Kota Asal", sorted(KOTA_KOORDINAT.keys()))
                tujuan = st.selectbox("Kota Tujuan", sorted(KOTA_KOORDINAT.keys()))
                waktu_berangkat = st.slider("Waktu Berangkat (jam)", 4.0, 10.0, 6.5, step=0.1)
            with col2:
                cuaca = st.selectbox("Kondisi Cuaca", ["Cerah", "Hujan", "Berawan"])
                kendaraan = st.selectbox("Jenis Kendaraan", ["Truk", "Pickup", "Van", "Motor Box"])
                kecepatan = st.slider("Kecepatan Rata-rata (km/jam)", 20, 120, 60)

            if st.form_submit_button("Simpan Data"):
                # Hitung jarak otomatis
                koordinat_asal = KOTA_KOORDINAT[asal]
                koordinat_tujuan = KOTA_KOORDINAT[tujuan]
                jarak = haversine_distance(
                    koordinat_asal[0], koordinat_asal[1],
                    koordinat_tujuan[0], koordinat_tujuan[1]
                )
                
                durasi = jarak / kecepatan
                waktu_tiba = waktu_berangkat + durasi
                keterlambatan = max((waktu_tiba - 8.0) * 60, 0)
                status = "Ya" if keterlambatan > 0 else "Tidak"

                data_baru = pd.DataFrame([{
                    "Asal": asal,
                    "Tujuan": tujuan,
                    "Jarak (km)": round(jarak, 2),
                    "Waktu Berangkat (jam)": waktu_berangkat,
                    "Cuaca": cuaca,
                    "Kendaraan": kendaraan,
                    "Kecepatan (km/jam)": kecepatan,
                    "Durasi (jam)": round(durasi, 2),
                    "Waktu Tiba (jam)": round(waktu_tiba, 2),
                    "Keterlambatan (menit)": round(keterlambatan, 0),
                    "Status Keterlambatan": status
                }])

                data = pd.concat([data, data_baru], ignore_index=True)
                data.to_excel("dataset_rute_ekspedisi.xlsx", index=False)
                st.success("Data berhasil disimpan!")
                st.rerun()

with tab2:
    st.subheader("Visualisasi Rute Pengiriman")
    
    if len(data) > 0:
        pilihan_rute = st.selectbox(
            "Pilih Rute untuk Divisualisasikan",
            options=data.apply(lambda x: f"{x['Asal']} → {x['Tujuan']}", axis=1).unique()
        )
        
        asal, tujuan = pilihan_rute.split(" → ")
        lokasi_asal = KOTA_KOORDINAT[asal]
        lokasi_tujuan = KOTA_KOORDINAT[tujuan]
        
        peta = folium.Map(
            location=[(lokasi_asal[0] + lokasi_tujuan[0])/2, 
                     (lokasi_asal[1] + lokasi_tujuan[1])/2],
            zoom_start=5
        )
        
        # Tambahkan marker untuk semua kota
        for kota, koordinat in KOTA_KOORDINAT.items():
            folium.Marker(
                koordinat,
                tooltip=kota,
                icon=folium.Icon(color='blue', icon='map-marker-alt')
            ).add_to(peta)
        
        # Highlight rute yang dipilih
        folium.Marker(
            lokasi_asal,
            tooltip=f"Asal: {asal}",
            icon=folium.Icon(color='green', icon='truck')
        ).add_to(peta)
        
        folium.Marker(
            lokasi_tujuan,
            tooltip=f"Tujuan: {tujuan}",
            icon=folium.Icon(color='red', icon='warehouse')
        ).add_to(peta)
        
        folium.PolyLine(
            locations=[lokasi_asal, lokasi_tujuan],
            color='blue',
            weight=3,
            tooltip=f"{asal} ke {tujuan}"
        ).add_to(peta)
        
        st_folium(peta, width=700, height=500)
    else:
        st.warning("Belum ada data rute untuk ditampilkan")

with tab3:
    st.subheader("Prediksi Keterlambatan Pengiriman")
    
    # Persiapkan model
    if len(data) > 10:  # Minimal ada 10 data untuk training
        # Encoding
        data_encoded = data.copy()
        le_cuaca = LabelEncoder()
        le_kendaraan = LabelEncoder()
        le_status = LabelEncoder()
        
        data_encoded["Cuaca"] = le_cuaca.fit_transform(data["Cuaca"])
        data_encoded["Kendaraan"] = le_kendaraan.fit_transform(data["Kendaraan"])
        data_encoded["Status Keterlambatan"] = le_status.fit_transform(data["Status Keterlambatan"])
        
        # Training
        X = data_encoded[["Jarak (km)", "Waktu Berangkat (jam)", "Cuaca", "Kendaraan"]]
        y = data_encoded["Status Keterlambatan"]
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        akurasi = accuracy_score(y_test, model.predict(X_test))
        
        # UI Prediksi
        with st.form("form_prediksi"):
            st.markdown("### Masukkan Parameter Pengiriman")
            
            col1, col2 = st.columns(2)
            with col1:
                asal = st.selectbox("Kota Asal", sorted(KOTA_KOORDINAT.keys()))
                tujuan = st.selectbox("Kota Tujuan", sorted(KOTA_KOORDINAT.keys()))
                waktu_berangkat = st.slider("Waktu Berangkat", 4.0, 10.0, 6.5, step=0.1)
            with col2:
                cuaca = st.selectbox("Prakiraan Cuaca", le_cuaca.classes_)
                kendaraan = st.selectbox("Jenis Kendaraan", le_kendaraan.classes_)
                kecepatan = st.slider("Estimasi Kecepatan (km/jam)", 20, 120, 60)
            
            if st.form_submit_button("Prediksi"):
                # Hitung jarak
                koordinat_asal = KOTA_KOORDINAT[asal]
                koordinat_tujuan = KOTA_KOORDINAT[tujuan]
                jarak = haversine_distance(
                    koordinat_asal[0], koordinat_asal[1],
                    koordinat_tujuan[0], koordinat_tujuan[1]
                )
                
                # Persiapkan input model
                input_data = np.array([[
                    jarak,
                    waktu_berangkat,
                    le_cuaca.transform([cuaca])[0],
                    le_kendaraan.transform([kendaraan])[0]
                ]])
                input_scaled = scaler.transform(input_data)
                
                # Prediksi
                hasil = model.predict(input_scaled)
                label = le_status.inverse_transform(hasil)[0]
                
                # Hitung estimasi waktu
                durasi = jarak / kecepatan
                waktu_tiba = waktu_berangkat + durasi
                keterlambatan = max((waktu_tiba - 8.0) * 60, 0)
                
                # Tampilkan hasil
                st.markdown("---")
                col_res1, col_res2 = st.columns(2)
                with col_res1:
                    st.metric("Estimasi Jarak", f"{jarak:.1f} km")
                    st.metric("Estimasi Durasi", f"{durasi:.1f} jam")
                with col_res2:
                    st.metric("Estimasi Waktu Tiba", f"{waktu_tiba:.1f} jam")
                    st.metric("Akurasi Model", f"{akurasi*100:.1f}%")
                
                if label == "Tidak":
                    st.success("✅ Prediksi: PENGIRIMAN TEPAT WAKTU")
                else:
                    st.error(f"❗ Prediksi: TERLAMBAT {keterlambatan:.0f} menit")
                
                # Visualisasi sederhana
                fig, ax = plt.subplots(figsize=(8, 2))
                ax.barh(["Waktu Pengiriman"], [durasi], color='skyblue')
                ax.axvline(x=8-waktu_berangkat, color='red', linestyle='--', label='Batas Waktu')
                ax.set_xlim(0, max(durasi+1, 8-waktu_berangkat+1))
                ax.set_title("Perbandingan Durasi vs Batas Waktu")
                ax.legend()
                st.pyplot(fig)
    else:
        st.warning("⚠️ Masukkan minimal 10 data pengiriman untuk mengaktifkan fitur prediksi")

st.markdown("---")
st.caption("Sistem Prediksi Keterlambatan Pengiriman Ekspedisi - © 2023")
