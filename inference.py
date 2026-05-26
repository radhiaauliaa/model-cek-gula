import tensorflow as tf
import numpy as np
from PIL import Image

MODEL_PATH = "model_cek_gula_v3.keras"

model = tf.keras.models.load_model(MODEL_PATH)

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

def preprocess_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img = img.resize((224, 224))

    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    return img

image = preprocess_image("sample_images/contoh.jpg")

prediction = model.predict(image)

predicted_class = CLASS_NAMES[np.argmax(prediction)]

print("Hasil Prediksi:", predicted_class)