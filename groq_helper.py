import os
from groq import Groq

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

def get_rekomendasi(
    prediksi_label,
    confidence,
    kondisi_pengguna=""
):

    kondisi = (
        kondisi_pengguna
        if kondisi_pengguna
        else "orang dewasa umum"
    )

    prompt = f"""
Kamu adalah asisten kesehatan.

Model Deep Learning mendeteksi:

Nama makanan:
{prediksi_label}

Confidence:
{confidence:.2f}%

Kondisi pengguna:
{kondisi}

Berikan:

1. Penjelasan singkat makanan
2. Risiko gula darah
3. Tips pola hidup sehat
4. Makanan/minuman yg perlu dikurangi

Jawaban maksimal 120 kata.
"""

    try:

        response = client.chat.completions.create(

            model="llama-3.3-70b-versatile",

            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]

        )

        return response.choices[0].message.content

    except Exception as e:

        return f"AI Error: {str(e)}"