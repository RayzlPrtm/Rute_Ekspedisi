import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Konfigurasi Streamlit
st.set_page_config(page_title="Prediksi Keterlambatan Ekspedisi", page_icon="🚚", layout="wide")
st.title("🚚 Prediksi Keterlambatan Pengiriman Ekspedisi")

# Load data ekspedisi
try:
    data = pd.read_excel("dataset_rute_ekspedisi.xlsx")
except FileNotFoundError:
    st.error("❌ File 'dataset_rute_ekspedisi.xlsx' tidak ditemukan.")
    st.stop()

st.subheader("📁 Contoh Data Rute Ekspedisi")
st.dataframe(data.head())

# Encoding dan preprocessing
data_encoded = data.copy()
le_cuaca = LabelEncoder()
le_kendaraan = LabelEncoder()
le_status = LabelEncoder()

data_encoded["Cuaca"] = le_cuaca.fit_transform(data_encoded["Cuaca"])
data_encoded["Kendaraan"] = le_kendaraan.fit_transform(data_encoded["Kendaraan"])
data_encoded["Status Keterlambatan"] = le_status.fit_transform(data_encoded["Status Keterlambatan"])

fitur = [
    "Jarak (km)", "Waktu Berangkat (jam)", "Cuaca", "Kendaraan"
]
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
