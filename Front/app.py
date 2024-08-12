from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
import os
import numpy as np
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
import pickle
from PIL import Image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join('Front','static', 'uploads')

# Asegúrate de que la carpeta de cargas exista
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Cargar el modelo preentrenado (asegúrate de que esté en la misma ruta)
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model = Model(inputs=base_model.input, outputs=base_model.layers[-1].output)

# Cargar variables
with open('C:/Users/ADA_LAB04/Documents/ImagesSearchCosineSimilarity/Saves/train_features.pkl', 'rb') as f:
    train_features_flat = pickle.load(f)

with open('C:/Users/ADA_LAB04/Documents/ImagesSearchCosineSimilarity/Saves/train_labels.pkl', 'rb') as f:
    train_labels_flat = pickle.load(f)

with open('C:/Users/ADA_LAB04/Documents/ImagesSearchCosineSimilarity/Saves/train_images.pkl', 'rb') as f:
    train_images_flat = pickle.load(f)

# Definir funciones
def preprocess_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, (224, 224))
    image = tf.cast(image, tf.float32) / 255.0
    return image

def retrieve_similar_images(query_feature, train_features, train_labels, train_images, top_n=5):
    query_feature = normalize([query_feature], axis=1)
    similarities = cosine_similarity(query_feature, train_features)[0]
    nearest_indices = np.argsort(similarities)[::-1][:top_n]
    return train_labels[nearest_indices], similarities[nearest_indices], train_images[nearest_indices]

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        if file and file.filename != '':
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            query_image_preprocessed = preprocess_image(file_path)
            query_feature = model.predict(np.expand_dims(query_image_preprocessed, axis=0)).flatten()
            query_feature = normalize([query_feature], axis=1)[0]

            similar_labels, similar_similarities, similar_images = retrieve_similar_images(
                query_feature, train_features_flat, train_labels_flat, train_images_flat
            )

            # Guardar las imágenes similares para mostrarlas en la web
            similar_image_paths = []
            for i, img_array in enumerate(similar_images):
                img = Image.fromarray((img_array * 255).astype(np.uint8))
                similar_img_filename = f"similar_{i}.png"
                similar_img_path = os.path.join(app.config['UPLOAD_FOLDER'], similar_img_filename)

                img.save(similar_img_path)
                similar_image_paths.append(similar_img_filename)
                print(f"Guardado {similar_img_path}")  # Imprimir la ruta de la imagen guardada

            return render_template('index.html', filename=filename, similar_labels=zip(similar_labels, similar_image_paths))

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
