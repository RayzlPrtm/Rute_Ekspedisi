import pandas as pd
import numpy as np

np.random.seed(42)
n = 1000
waktu_tiba_target = 8.00

# Data kota-kota besar di seluruh Indonesia
asal_kota = [
    "Jakarta", "Bandung", "Yogyakarta", "Surabaya", "Semarang", "Medan", "Makassar", "Palembang",
    "Balikpapan", "Pekanbaru", "Manado", "Jayapura", "Kupang", "Padang", "Pontianak"
]

tujuan_kota = [
    "Surabaya", "Malang", "Semarang", "Jakarta", "Yogyakarta", "Denpasar", "Banjarmasin", "Ambon",
    "Ternate", "Batam", "Kendari", "Mataram", "Bandar Lampung", "Jambi", "Samarinda"
]

cuaca_opsi = ["Cerah", "Hujan", "Berawan"]
kendaraan_opsi = ["Truk", "Pickup", "Van", "Motor Box"]

data = {
    "Asal": np.random.choice(asal_kota, n),
    "Tujuan": np.random.choice(tujuan_kota, n),
    "Jarak (km)": np.round(np.random.uniform(50, 2500, n), 1),
    "Waktu Berangkat (jam)": np.round(np.random.uniform(4.0, 10.0, n), 2),
    "Cuaca": np.random.choice(cuaca_opsi, n),
    "Kendaraan": np.random.choice(kendaraan_opsi, n),
}

df = pd.DataFrame(data)

# Simulasikan kecepatan berdasarkan cuaca dan jenis kendaraan
kecepatan = []
for i in range(n):
    base = 60
    if df.loc[i, "Cuaca"] == "Hujan":
        base -= 10
    if df.loc[i, "Kendaraan"] == "Motor Box":
        base -= 10
    elif df.loc[i, "Kendaraan"] == "Truk":
        base -= 5
    kecepatan.append(np.clip(np.random.normal(base, 5), 30, 100))

df["Kecepatan (km/jam)"] = np.round(kecepatan, 2)
df["Durasi (jam)"] = (df["Jarak (km)"] / df["Kecepatan (km/jam)"]).round(2)
df["Waktu Tiba (jam)"] = df["Waktu Berangkat (jam)"] + df["Durasi (jam)"]
df["Keterlambatan (menit)"] = np.maximum((df["Waktu Tiba (jam)"] - waktu_tiba_target) * 60, 0).round(0)
df["Status Keterlambatan"] = df["Keterlambatan (menit)"].apply(lambda x: "Ya" if x > 0 else "Tidak")

# Simpan ke Excel
df.to_excel("dataset_rute_ekspedisi.xlsx", index=False)
print("✅ File 'dataset_rute_ekspedisi.xlsx' berhasil dibuat dengan cakupan seluruh Indonesia.")
