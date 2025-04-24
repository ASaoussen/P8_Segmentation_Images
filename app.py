import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify, send_file
from PIL import Image
from matplotlib import colors
from azure.storage.blob import BlobServiceClient
from tensorflow.keras.models import load_model
# Import custom metrics if used
from segmentation_models.metrics import precision, recall
from tensorflow.keras.metrics import BinaryAccuracy, BinaryIoU
# --- Configuration Azure ---
account_url = secrets.BLOB_URL
account_key = secrets.BLOB_ACCOUNT_KEY  # 🔐 Remplace ceci par ta vraie clé
container_name = "container1"
blob_prefix = "Model"
local_model_path = "model"

# Créer le répertoire local pour sauvegarder le modèle si nécessaire
os.makedirs(local_model_path, exist_ok=True)

# Connexion à Azure Blob Storage avec clé
blob_service_client = BlobServiceClient(account_url=account_url, credential=account_key)
container_client = blob_service_client.get_container_client(container_name)

# Télécharger tous les blobs dans le dossier "Model"
model_files = []  # Liste pour stocker les fichiers du modèle
for blob in container_client.list_blobs(name_starts_with=blob_prefix):
    blob_name = blob.name
    local_model_file_path = os.path.join(local_model_path, os.path.relpath(blob_name, blob_prefix))
    
    # Créer les répertoires nécessaires
    os.makedirs(os.path.dirname(local_model_file_path), exist_ok=True)

    # Télécharger le blob
    blob_client = container_client.get_blob_client(blob_name)
    with open(local_model_file_path, "wb") as f:
        blob_data = blob_client.download_blob()
        blob_data.readinto(f)

    print(f"Modèle téléchargé depuis Azure Blob Storage à {local_model_file_path}")
    model_files.append(local_model_file_path)

# Chargement du modèle Keras avec les objets personnalisés
custom_objects = {
    "binary_iou": BinaryIoU(name="binary_iou"),
    "binary_accuracy": BinaryAccuracy(),
    "precision": precision,
    "recall": recall
}

# Vérification et chargement du modèle Keras
if model_files:
    model_path = model_files[-1]  # On suppose que le dernier fichier est le modèle principal
    try:
        MODEL = load_model(local_model_path, custom_objects=custom_objects)
        print("Modèle chargé avec succès!")
    except Exception as e:
        print(f"Erreur lors du chargement du modèle : {e}")
else:
    print("Aucun modèle trouvé dans le container Azure Blob Storage.")
    MODEL = None  # Si aucun modèle n'est trouvé

# Dimensions d'entrée attendues par le modèle
MODEL_INPUT_WIDTH = 256
MODEL_INPUT_HEIGHT = 128

def generate_img_from_mask(mask, colors_palette=['black', 'green', 'red', 'cyan', 'magenta', 'yellow', 'blue', 'white']):
    id2category = {0: 'void', 1: 'flat', 2: 'construction', 3: 'object',
                   4: 'nature', 5: 'sky', 6: 'human', 7: 'vehicle'}
    
    img_seg = np.zeros((mask.shape[0], mask.shape[1], 3), dtype='float')
    for cat in id2category.keys():
        img_seg[:, :, 0] += mask[:, :, cat] * colors.to_rgb(colors_palette[cat])[0]
        img_seg[:, :, 1] += mask[:, :, cat] * colors.to_rgb(colors_palette[cat])[1]
        img_seg[:, :, 2] += mask[:, :, cat] * colors.to_rgb(colors_palette[cat])[2]
    return img_seg

def predict_segmentation(image_array, image_width, image_height):
    # Redimensionner l'image
    image_array = Image.fromarray(image_array).resize((image_width, image_height))
    image_array = np.expand_dims(np.array(image_array), axis=0)
    
    if MODEL is None:
        print("Erreur : le modèle n'est pas disponible.")
        return np.zeros((image_height, image_width, 3))  # Retourner une image vide en cas d'erreur
    
    mask_predict = MODEL.predict(image_array)
    mask_predict = np.squeeze(mask_predict, axis=0)
    mask_class = np.argmax(mask_predict, axis=-1)
    mask_one_hot = np.eye(mask_predict.shape[-1])[mask_class]
    mask_color = generate_img_from_mask(mask_one_hot) * 255
    return mask_color

# Flask app
app = Flask(__name__)

@app.route("/")
def hello():
    return "Hello, welcome to the segmentation API"

@app.route("/predict_mask", methods=["POST"])
def segment_image():
    # Vérifier si un fichier est bien envoyé dans la requête
    if 'image' not in request.files:
        return jsonify({"error": "Aucune image fournie"}), 400

    file = request.files['image']
    
    try:
        # Ouvrir l'image
        img = Image.open(file.stream)
    except Exception as e:
        return jsonify({"error": f"Erreur lors de l'ouverture de l'image : {str(e)}"}), 400

    # Prétraiter l'image et obtenir la prédiction de la segmentation
    mask_color = predict_segmentation(np.array(img), MODEL_INPUT_WIDTH, MODEL_INPUT_HEIGHT)
    
    # Sauvegarde temporaire de l'image générée
    tmp_path = "tmp.png"
    Image.fromarray(mask_color.astype(np.uint8)).save(tmp_path)
    
    # Retourner l'image générée
    return send_file(tmp_path, mimetype="image/png")

if __name__ == "__main__":
    app.run(debug=True)
