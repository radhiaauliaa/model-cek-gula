from fastapi import FastAPI, UploadFile, File
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
from PIL import Image
import io

from groq_helper import get_rekomendasi

app = FastAPI()

CLASS_NAMES = [
    'ampyang',
    'arem-arem',
    'arum manis',
    'bakpao',
    'bakwan',
    'biji salak',
    'bika ambon',
    'bolen',
    'bolu gulung',
    'bolu karamel',
    'bolu kukus',
    'brem',
    'bubur mutiara',
    'bubur sumsum',
    'cenil',
    'cilok',
    'cimol',
    'cireng',
    'combro',
    'dadar gulung',
    'dodol',
    'donat',
    'gathot',
    'geplak',
    'getas',
    'gethuk pisang',
    'getuk goreng',
    'getuk lindri',
    'kembang goyang',
    'keripik pisang',
    'keripik singkong',
    'keripik tempe',
    'keukarah',
    'klepon',
    'kroket',
    'kue akar kelapa',
    'kue ape',
    'kue apem',
    'kue balok',
    'kue bangkit',
    'kue bawang',
    'kue bingka',
    'kue carabikang',
    'kue cubit',
    'kue cucur',
    'kue gemblong',
    'kue jipang',
    'kue kacang',
    'kue kamir',
    'kue klemben',
    'kue lekker',
    'kue lidah kucing',
    'kue lumpang',
    'kue lumpur',
    'kue lupis',
    'kue manco',
    'kue mendut',
    'kue mochi',
    'kue padamaran',
    'kue pancong',
    'kue pastel',
    'kue pasung',
    'kue perut ayam',
    'kue pudak',
    'kue pukis',
    'kue putri mandi',
    'kue putu',
    'kue putu ayu',
    'kue rangi',
    'kue sagon',
    'kue semprong',
    'kue sus',
    'kue talam',
    'kue thok',
    'kue wajik',
    'kue yangko',
    'lapis beras',
    'lapis legit',
    'lemper',
    'lumpia',
    'madu mongso',
    'martabak manis',
    'nagasari',
    'onde-onde',
    'otak-otak',
    'panada',
    'pisang goreng',
    'rempeyek',
    'rengginang',
    'risol',
    'sale pisang',
    'sawut',
    'serabi',
    'sosis solo',
    'tahu isi',
    'tape ketan',
    'tape singkong',
    'tempe mendoan',
    'tiwul',
    'wingko babat'
]

class AdaptivePreprocessing(tf.keras.layers.Layer):
    def __init__(self, epsilon=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon

    def call(self, inputs, training=None):
        x = tf.cast(inputs, tf.float32)

        x = (x / 127.5) - 1.0

        mean = tf.reduce_mean(
            x,
            axis=[1, 2],
            keepdims=True
        )

        std = tf.math.reduce_std(
            x,
            axis=[1, 2],
            keepdims=True
        )

        x = (x - mean) / (std + self.epsilon)

        return x

    def get_config(self):
        config = super().get_config()

        config.update({
            "epsilon": self.epsilon
        })

        return config

class SqueezeExcitation(tf.keras.layers.Layer):
    def __init__(self, ratio=4, **kwargs):
        super().__init__(**kwargs)
        self.ratio = ratio

    def build(self, input_shape):
        channels = input_shape[-1]

        reduced = max(1, channels // self.ratio)

        self.fc1 = layers.Dense(
            reduced,
            activation='relu'
        )

        self.fc2 = layers.Dense(
            channels,
            activation='sigmoid'
        )

        super().build(input_shape)

    def call(self, inputs):
        se = self.fc1(inputs)

        se = self.fc2(se)

        return inputs * se

    def get_config(self):
        config = super().get_config()

        config.update({
            "ratio": self.ratio
        })

        return config

class FocalLoss(tf.keras.losses.Loss):
    def __init__(
        self,
        gamma=2.0,
        label_smoothing=0.1,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def call(self, y_true, y_pred):

        num_classes = tf.cast(
            tf.shape(y_true)[-1],
            tf.float32
        )

        y_true = (
            y_true * (1.0 - self.label_smoothing)
            + (self.label_smoothing / num_classes)
        )

        y_pred = tf.clip_by_value(
            y_pred,
            1e-7,
            1.0
        )

        ce = -tf.reduce_sum(
            y_true * tf.math.log(y_pred),
            axis=-1
        )

        pt = tf.reduce_sum(
            y_true * y_pred,
            axis=-1
        )

        focal_weight = tf.pow(
            1.0 - pt,
            self.gamma
        )

        return tf.reduce_mean(
            focal_weight * ce
        )

    def get_config(self):
        config = super().get_config()

        config.update({
            "gamma": self.gamma,
            "label_smoothing": self.label_smoothing
        })

        return config

model = tf.keras.models.load_model(
    "model_cek_gula_v3.keras",
    custom_objects={
        "AdaptivePreprocessing": AdaptivePreprocessing,
        "SqueezeExcitation": SqueezeExcitation,
        "FocalLoss": FocalLoss,
    }
)

def preprocess_image(image):

    image = image.convert("RGB")

    image = image.resize((224, 224))

    image = np.array(image).astype(np.float32)

    image = np.expand_dims(image, axis=0)

    return image

@app.get("/")
def home():
    return {
        "message": "API Cek-Gula aktif"
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    contents = await file.read()

    image = Image.open(io.BytesIO(contents))

    processed_image = preprocess_image(image)

    prediction = model.predict(processed_image)

    predicted_idx = int(np.argmax(prediction))

    predicted_class = CLASS_NAMES[predicted_idx]

    confidence = float(np.max(prediction))

    rekomendasi_ai = get_rekomendasi(
        prediksi_label=predicted_class,
        confidence=confidence
    )

    return {
        "prediction": predicted_class,
        "confidence": round(confidence * 100, 2),
        "rekomendasi_ai": rekomendasi_ai
    }