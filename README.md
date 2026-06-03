# 🍬 Cek-Gula - AI Food Nutrition & Blood Sugar Risk Detection
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-green)
![Accuracy](https://img.shields.io/badge/Accuracy-95.45%25-brightgreen)
![MAE](https://img.shields.io/badge/MAE-0.002-blue)
![Deployment](https://img.shields.io/badge/Deployment-HuggingFace-yellow)

## Deteksi Kandungan Gizi dan Risiko Gula Darah pada Jajanan Pasar Indonesia Menggunakan Deep Learning dan Generative AI

---

# 📖 Deskripsi Proyek

Cek-Gula adalah aplikasi berbasis Artificial Intelligence yang membantu pengguna mengenali jajanan pasar Indonesia melalui gambar dan menampilkan informasi nutrisi secara otomatis.

Aplikasi ini dikembangkan sebagai upaya edukasi kesehatan untuk meningkatkan kesadaran masyarakat terhadap konsumsi gula dan risiko diabetes melalui analisis makanan tradisional Indonesia.

Pengguna cukup mengunggah gambar jajanan pasar, kemudian sistem akan:

1. Mengidentifikasi jenis makanan menggunakan Deep Learning.
2. Menampilkan informasi nutrisi makanan.
3. Menampilkan estimasi Glycemic Index (GI).
4. Menampilkan estimasi Glycemic Load (GL).
5. Memberikan rekomendasi konsumsi menggunakan Large Language Model (LLM).

---

# 🩺 Latar Belakang

Peningkatan kasus diabetes dan penyakit metabolik di Indonesia menjadi perhatian serius.

Di sisi lain, jajanan pasar tradisional umumnya tidak memiliki informasi nutrisi yang jelas sehingga masyarakat sulit mengetahui:

- Kandungan gula
- Kandungan kalori
- Kandungan karbohidrat
- Risiko lonjakan gula darah

Cek-Gula hadir sebagai solusi berbasis AI yang mampu mengenali makanan tradisional Indonesia hanya dari gambar dan memberikan informasi nutrisi secara otomatis.

---

# 💡 Solusi yang Ditawarkan

Aplikasi menggabungkan:

- Computer Vision
- Deep Learning
- Database Nutrisi
- Generative AI

untuk membantu pengguna memahami risiko konsumsi makanan secara lebih mudah dan cepat.

---

# 🏗️ Arsitektur Sistem

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

# 🧠 Model Deep Learning

## Backbone

**MobileNetV3Large**

### Transfer Learning

Model menggunakan pretrained weights ImageNet kemudian dilakukan:

- Frozen Training
- Fine-Tuning
- Full Fine-Tuning

### Custom Layer

#### AdaptivePreprocessing

- Normalisasi gambar
- Standardisasi berdasarkan mean dan standar deviasi citra

#### SqueezeExcitation

- Memberikan attention terhadap fitur penting sebelum proses klasifikasi

### Custom Loss Function

#### Focal Loss

Digunakan untuk:

- Mengurangi dampak class imbalance
- Memfokuskan training pada sampel yang sulit diprediksi

```text
FL(pt) = -(1-pt)^γ log(pt)
```

### Custom Callback

#### TargetAccuracyCallback

```text
Validation Accuracy ≥ 95%
```

---

# 📊 Hasil Training

| Metric | Hasil |
|----------|----------|
| Validation Accuracy | 95.45% |
| Target Accuracy | 85% |
| Validation MAE | 0.002 |
| Target MAE | < 0.02 |

### Status

✅ Target Akurasi Tercapai

✅ Target MAE Tercapai

---

# 🗂️ Dataset

Dataset terdiri dari citra berbagai jajanan pasar Indonesia.

Contoh kelas:

- Klepon
- Dadar Gulung
- Lapis Legit
- Onde-onde
- Serabi
- Wingko Babat
- Bolu Gulung
- dan lainnya

Jumlah kelas:

**100 Kelas Makanan Tradisional Indonesia**

---

# 🥗 Informasi Nutrisi

Setelah model mengenali makanan, sistem mengambil data nutrisi dari:

```text
dataset_nutrisi_lengkap2.csv
```

Informasi yang ditampilkan:

- Kalori
- Karbohidrat
- Gula
- Protein
- Lemak
- Glycemic Index (GI)
- Glycemic Load (GL)
- Risiko Gula Darah
- Saran Konsumsi

---

# 🤖 Fitur Generative AI

Menggunakan:

- Groq API
- Llama 3.3 70B Versatile

Fungsi:

- Menjelaskan makanan
- Memberikan edukasi kesehatan
- Menjelaskan risiko gula darah
- Memberikan saran pola makan sehat

Contoh output:

```text
Dadar gulung adalah kue tradisional Indonesia yang terbuat dari tepung dan gula merah.

Risiko gula darah: tinggi.

Konsumsi secukupnya dan imbangi dengan aktivitas fisik.
```

---

# ⚙️ Setup Environment

## 1. Clone Repository

```bash
git clone https://github.com/radhiaauliaa/model-cek-gula.git
cd model-cek-gula
```

## 2. Buat Virtual Environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

## 3. Install Dependensi

```bash
pip install -r requirements.txt
```

## 4. Konfigurasi Environment Variable

Buat file `.env`:

```env
GROQ_API_KEY=your_groq_api_key
```

---

# 🤖 Model Machine Learning

Model TensorFlow (.keras) dapat diunduh melalui:

👉 **[Download Model Cek-Gula](https://drive.google.com/drive/folders/1Ok6y3BmgaMr9ifeOtdfZ4fX7u8AkEUg-?usp=sharing)**

Setelah diunduh, letakkan file model pada root project:

```text
model_cek_gula_v3.keras
```

### Cara Memuat Model

```python
import tensorflow as tf

model = tf.keras.models.load_model(
    "model_cek_gula_v3.keras",
    custom_objects={
        "AdaptivePreprocessing": AdaptivePreprocessing,
        "SqueezeExcitation": SqueezeExcitation,
        "FocalLoss": FocalLoss
    }
)
```

---

# 🚀 Cara Menjalankan Aplikasi

Jalankan FastAPI:

```bash
uvicorn app:app --reload
```

Akses aplikasi:

```text
http://127.0.0.1:8000
```

Swagger Documentation:

```text
http://127.0.0.1:8000/docs
```

---

# 🔌 REST API

## Base URL

```text
https://aulian12-model-cek-gula.hf.space
```

## Swagger Documentation

```text
https://aulian12-model-cek-gula.hf.space/docs
```

### GET /

Response:

```json
{
  "message": "API Cek-Gula aktif"
}
```

### POST /predict

Request:

```bash
curl -X POST \
"https://aulian12-model-cek-gula.hf.space/predict" \
-H "accept: application/json" \
-H "Content-Type: multipart/form-data" \
-F "file=@gambar.png"
```

Response:

```json
{
  "prediction": "dadar gulung",
  "confidence": 99.91
}
```

---

# 📁 Struktur Proyek

```text
PROYEK CEK GULA
│
├── __pycache__/
├── logs/
│
├── .env
├── .gitattributes
├── .gitignore
│
├── app.py
├── groq_helper.py
├── inference.py
│
├── best_model.keras
├── best_model_phase1.keras
├── model_cek_gula_v3.keras
│
├── cek_gula.ipynb
│
├── dataset_nutrisi_lengkap.csv
├── dataset_nutrisi_lengkap2.csv
│
├── Dockerfile
├── requirements.txt
├── README.md
│
├── train_history_v3.png
└── images.png
```

---

# 🛠️ Teknologi & Dependensi Utama

| Teknologi | Deskripsi |
|------------|------------|
| TensorFlow | Framework Deep Learning untuk training dan inferensi model klasifikasi gambar. |
| Keras | High-level API TensorFlow untuk membangun dan melatih model neural network. |
| MobileNetV3Large | Backbone CNN ringan dan efisien yang digunakan untuk klasifikasi jajanan pasar. |
| FastAPI | Framework backend modern untuk membangun REST API berperforma tinggi. |
| Uvicorn | ASGI server untuk menjalankan aplikasi FastAPI. |
| Pandas | Pengolahan dataset nutrisi dan analisis data tabular. |
| NumPy | Operasi numerik dan manipulasi array multidimensi. |
| Pillow (PIL) | Pemrosesan dan preprocessing gambar sebelum inferensi model. |
| Groq API | Integrasi Large Language Model untuk menghasilkan rekomendasi konsumsi makanan. |
| Llama 3.3 70B Versatile | Model Generative AI yang digunakan untuk edukasi kesehatan dan rekomendasi makanan. |
| Hugging Face Spaces | Platform deployment aplikasi AI dan REST API. |
| Docker | Containerization untuk deployment yang konsisten dan mudah direproduksi. |

---

# ☁️ Deployment

Deployment menggunakan:

- Hugging Face Spaces
- FastAPI
- TensorFlow
- Docker

Fitur deployment:

- REST API
- Swagger Documentation
- Real-time Inference

---

# 🔮 Pengembangan Selanjutnya

- Perhitungan nutrisi berdasarkan jumlah porsi
- Input jumlah potong makanan
- Estimasi berat makanan dari gambar
- Riwayat konsumsi pengguna
- Monitoring konsumsi gula harian
- Rekomendasi makanan rendah gula

---

# 👥 Tim Pengembang

### AI Engineer

- Pengembangan model Deep Learning
- Training dan evaluasi model

### Data Scientist

- Dataset
- Nutritional Database
- Analisis Data

### Full Stack Developer

- Backend API
- Frontend
- Deployment

---

# 📄 Lisensi

© 2026 Tim CC26-PSU007

Proyek ini dikembangkan oleh Tim CC26-PSU007 sebagai bagian dari Capstone Project DBS Foundation Coding Camp 2026.
