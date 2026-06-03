---
title: Model Cek Gula
sdk: docker
---

# Aplikasi Cek Gula

## Deskripsi Proyek
Deteksi Kandungan Gizi dan Prediksi Indeks Glikemik Jajanan Pasar Indonesia. Proyek ini berfokus pada pencegahan diabetes dengan membantu pengguna dalam mengenali kandungan gula pada makanan tradisional.

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

Aplikasi menggunakan Groq API sebagai fitur sekunder untuk:
- memberikan rekomendasi konsumsi
- alternatif jajanan lebih sehat
- edukasi gula
- ringkasan konsumsi harian

## Cara Menjalankan API

```bash
uvicorn app:app --reload
