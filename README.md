![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-green)
![Accuracy](https://img.shields.io/badge/Accuracy-95.45%25-brightgreen)
![MAE](https://img.shields.io/badge/MAE-0.002-blue)
![Deployment](https://img.shields.io/badge/Deployment-HuggingFace-yellow)

#  Cek-Gula

## Deteksi Kandungan Gizi dan Risiko Gula Darah pada Jajanan Pasar Indonesia Menggunakan Deep Learning dan Generative AI

---

## Deskripsi Proyek

Cek-Gula adalah aplikasi berbasis Artificial Intelligence yang membantu pengguna mengenali jajanan pasar Indonesia melalui gambar dan menampilkan informasi nutrisi secara otomatis.

Aplikasi ini dikembangkan sebagai upaya edukasi kesehatan untuk meningkatkan kesadaran masyarakat terhadap konsumsi gula dan risiko diabetes melalui analisis makanan tradisional Indonesia.

Pengguna cukup mengunggah gambar jajanan pasar, kemudian sistem akan:

1. Mengidentifikasi jenis makanan menggunakan Deep Learning.
2. Menampilkan informasi nutrisi makanan.
3. Menampilkan estimasi Glycemic Index (GI).
4. Menampilkan estimasi Glycemic Load (GL).
5. Memberikan rekomendasi konsumsi menggunakan Large Language Model (LLM).

---

# Latar Belakang

Peningkatan kasus diabetes dan penyakit metabolik di Indonesia menjadi perhatian serius.

Di sisi lain, jajanan pasar tradisional umumnya tidak memiliki informasi nutrisi yang jelas sehingga masyarakat sulit mengetahui:

* Kandungan gula
* Kandungan kalori
* Kandungan karbohidrat
* Risiko lonjakan gula darah

Cek-Gula hadir sebagai solusi berbasis AI yang mampu mengenali makanan tradisional Indonesia hanya dari gambar dan memberikan informasi nutrisi secara otomatis.

---

# Solusi yang Ditawarkan

Aplikasi menggabungkan:

* Computer Vision
* Deep Learning
* Database Nutrisi
* Generative AI

untuk membantu pengguna memahami risiko konsumsi makanan secara lebih mudah dan cepat.

---

# Arsitektur Sistem

```text
┌─────────────────────┐
│       User          │
└──────────┬──────────┘
           │ Upload Gambar
           ▼
┌─────────────────────┐
│      Frontend       │
└──────────┬──────────┘
           │ HTTP Request
           ▼
┌─────────────────────┐
│      FastAPI        │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────────────┐
│ TensorFlow Deep Learning    │
│ MobileNetV3Large            │
│ + AdaptivePreprocessing     │
│ + SqueezeExcitation         │
│ + FocalLoss                 │
└──────────┬──────────────────┘
           │
           ▼
┌─────────────────────────────┐
│ Prediksi Jenis Jajanan      │
└───────┬───────────┬─────────┘
        │           │
        ▼           ▼

┌─────────────┐  ┌─────────────┐
│ Dataset     │  │ Groq API    │
│ Nutrisi     │  │ Llama 3.3   │
└──────┬──────┘  └──────┬──────┘
       │                │
       ▼                ▼

Informasi Nutrisi   Rekomendasi AI

       ▼
┌─────────────────────┐
│   JSON Response     │
└─────────────────────┘
```

---

# Model Deep Learning

## Backbone

MobileNetV3Large

### Transfer Learning

Model menggunakan pretrained weights ImageNet kemudian dilakukan:

* Frozen Training
* Fine-Tuning
* Full Fine-Tuning

---

## Custom Layer

### AdaptivePreprocessing

Melakukan:

* Normalisasi gambar
* Standardisasi berdasarkan mean dan standar deviasi citra

### SqueezeExcitation

Memberikan attention terhadap fitur penting sebelum proses klasifikasi.

---

## Custom Loss Function

### Focal Loss

Digunakan untuk:

* Mengurangi dampak class imbalance
* Memfokuskan training pada sampel yang sulit diprediksi

Formula:

```
FL(pt) = -(1-pt)^γ log(pt)
```

---

## Custom Callback

### TargetAccuracyCallback

Training otomatis dihentikan apabila:

```
Validation Accuracy ≥ 95%
```

---

# Hasil Training

## Metrics

| Metric              | Hasil  |
| ------------------- | ------ |
| Validation Accuracy | 95.45% |
| Target Accuracy     | 85%    |
| Validation MAE      | 0.002  |
| Target MAE          | < 0.02 |

Status:

✅ Target Akurasi Tercapai

✅ Target MAE Tercapai

---

# Strategi Training

## Phase 1

Frozen Backbone

* MobileNetV3Large tidak dilatih
* Hanya classifier head yang dilatih

## Phase 2

Partial Fine-Tuning

* 80 layer terakhir di-unfreeze

## Phase 3

Full Fine-Tuning

* Seluruh layer model dilatih kembali

---

# Dataset

Dataset terdiri dari citra berbagai jajanan pasar Indonesia.

Contoh kelas:

* Klepon
* Dadar Gulung
* Lapis Legit
* Onde-onde
* Serabi
* Wingko Babat
* Bolu Gulung
* dan lainnya

Jumlah kelas:

**100 kelas makanan tradisional Indonesia**

---

# Informasi Nutrisi

Setelah model mengenali makanan, sistem mengambil data nutrisi dari dataset:

```
dataset_nutrisi_lengkap2.csv
```

Informasi yang ditampilkan:

* Kalori
* Karbohidrat
* Gula
* Protein
* Lemak
* Glycemic Index (GI)
* Glycemic Load (GL)
* Risiko gula darah
* Saran konsumsi

---

# Fitur Generative AI

Menggunakan:

* Groq API
* Llama 3.3 70B Versatile

Fungsi:

* Menjelaskan makanan
* Memberikan edukasi kesehatan
* Menjelaskan risiko gula darah
* Memberikan saran pola makan sehat

Contoh output:

```
Dadar gulung adalah kue tradisional Indonesia yang terbuat dari tepung dan gula merah.

Risiko gula darah: tinggi.

Konsumsi secukupnya dan imbangi dengan aktivitas fisik.
```

---

# 🔌 REST API

## Base URL

```
https://aulian12-model-cek-gula.hf.space
```

---

## Dokumentasi API (Swagger)

```
https://aulian12-model-cek-gula.hf.space/docs
```

---

## GET /

Cek status API.

Response:

```json
{
  "message": "API Cek-Gula aktif"
}
```

---

## POST /predict

Upload gambar makanan.

### Request

```bash
curl -X POST \
"https://aulian12-model-cek-gula.hf.space/predict" \
-H "accept: application/json" \
-H "Content-Type: multipart/form-data" \
-F "file=@gambar.png"
```

### Response

```json
{
  "prediction": "dadar gulung",
  "confidence": 99.91,
  "nutrisi": {
    "porsi": "100 g",
    "kalori": "139.0 kkal",
    "karbohidrat": "17.15 g",
    "gula": "3.48 g",
    "protein": "2.82 g",
    "lemak": "6.79 g"
  },
  "rekomendasi_ai": "Dadar gulung adalah kue tradisional Indonesia..."
}
```

---

# 📁 Struktur Proyek

```text
cek-gula/
│
├── app.py
├── groq_helper.py
├── requirements.txt
├── README.md
│
├── model_cek_gula_v3.keras
│
├── dataset_nutrisi_lengkap2.csv
│
├── model-cek-gula.ipynb
│
└── logs/
```

---

# Teknologi yang Digunakan

## Machine Learning

* TensorFlow
* Keras
* NumPy

## Backend

* FastAPI
* Uvicorn

## Data Processing

* Pandas
* Pillow

## Generative AI

* Groq API
* Llama 3.3 70B Versatile

## Deployment

* Hugging Face Spaces

---

# Deployment

Model dan API dideploy menggunakan:

* Hugging Face Spaces
* FastAPI
* TensorFlow

Deployment menyediakan:

* REST API
* Swagger Documentation
* Real-time Inference

---

# Pengembangan Selanjutnya

* Perhitungan nutrisi berdasarkan jumlah porsi
* Input jumlah potong makanan
* Estimasi berat makanan dari gambar
* Riwayat konsumsi pengguna
* Monitoring konsumsi gula harian
* Rekomendasi makanan rendah gula

---

# Tim Pengembang

### AI Engineer

* Pengembangan model Deep Learning
* Training dan evaluasi model

### Data Scientist

* Dataset
* Nutritional Database
* Analisis data

### Full Stack Developer

* Backend API
* Frontend
* Deployment

---

# Lisensi

© 2026 Tim CC26-PSU007

Proyek ini dikembangkan oleh Tim CC26-PSU007 sebagai bagian dari Capstone Project DBS Foundation Coding Camp 2026.
