# Cek Gula Darah dari Mata Menggunakan Deep Learning

## Deskripsi Proyek
Proyek ini membangun model Deep Learning berbasis TensorFlow untuk mendeteksi kondisi gula darah melalui citra mata.

Model dikembangkan menggunakan TensorFlow Functional API dengan beberapa komponen custom seperti:
- Custom Layer
- Custom Loss Function
- Custom Callback

## Fitur
- Training model Deep Learning
- TensorBoard monitoring
- Inference gambar
- REST API menggunakan FastAPI
- Penyimpanan model format .keras dan SavedModel

## Teknologi
- TensorFlow
- FastAPI
- NumPy
- Pillow

## Fitur Generative AI

Aplikasi menggunakan Google Gemini API sebagai fitur sekunder untuk:
- memberikan rekomendasi konsumsi
- alternatif jajanan lebih sehat
- edukasi gula darah
- ringkasan konsumsi harian

## Cara Menjalankan API

```bash
uvicorn app:app --reload

