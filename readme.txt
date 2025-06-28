# 🚚 Prediksi Keterlambatan Ekspedisi Berbasis Data Geospasial

Proyek ini adalah aplikasi Python interaktif menggunakan Streamlit untuk memvisualisasikan dan memprediksi keterlambatan pengiriman ekspedisi berdasarkan data geospasial, cuaca, dan jenis kendaraan.

---

## ✅ Apa yang Dilakukan Aplikasi Ini?

* Menghasilkan data simulasi pengiriman ekspedisi secara acak di seluruh Indonesia
* Menentukan rute menggunakan klik dua titik di peta interaktif
* Menghitung jarak otomatis berdasarkan titik geografis
* Menghitung durasi perjalanan dan keterlambatan
* Memprediksi status keterlambatan menggunakan model machine learning
* Menampilkan peta, tabel, dan grafik analisis

---

## 📂 Output File

📁 `dataset_rute_ekspedisi.xlsx` — Dataset ekspedisi berisi asal, tujuan, cuaca, kecepatan, durasi, dan status keterlambatan

---

## 🧰 Persyaratan

* Python ≥ 3.7
* Library Python:

  * `pandas`
  * `numpy`
  * `openpyxl`
  * `streamlit`
  * `matplotlib`
  * `folium`
  * `streamlit-folium`
  * `scikit-learn`
  * `seaborn`
  * `geopy`

---

## ⚙️ Instalasi & Menjalankan

1. **Install dependencies**:

```bash
pip install -r requirements.txt
```

2. **Jalankan aplikasi Streamlit**:

```bash
streamlit run ai_rute_ekspedisi.py
```

---

## 🖼️ Tampilan Aplikasi

Aplikasi menampilkan:

* Peta interaktif untuk menentukan rute ekspedisi
* Form input data pengiriman
* Prediksi keterlambatan ekspedisi
* Visualisasi akurasi model & confusion matrix
* Simpan dan lihat dataset di Excel

---

## 📝 Catatan Tambahan

* Dataset dapat diperbarui langsung melalui aplikasi
* Jarak bisa dihitung otomatis dari peta atau diisi manual
* Model machine learning akan dilatih ulang setiap ada data baru

---

## 👨‍💻 Dibuat Oleh

**Nama Anda**
📧 [41122047@mahasiswa.undira.ac.id](mailto:411222047@mahasiswa.undira.ac.id
)

[![GitHub](https://img.shields.io/badge/GitHub-rute--ekspedisi-blue?logo=github)](https://github.com/username/rute-ekspedisi)
[![Streamlit Cloud](https://img.shields.io/badge/Streamlit-App-red?logo=streamlit)](https://rute-ekspedisi.streamlit.app)

---

## 📄 Lisensi

Aplikasi ini bebas digunakan untuk keperluan pendidikan dan pembelajaran.
