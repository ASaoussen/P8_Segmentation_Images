import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify, send_file
from PIL import Image
from matplotlib import colors
from azure.storage.blob import BlobServiceClient
from tensorflow.keras.models import load_model
from segmentation_models.metrics import precision, recall
from tensorflow.keras.metrics import BinaryAccuracy, BinaryIoU
from flasgger import Swagger, swag_from  # <<< AJOUT

# --- Configuration Azure ---
account_url = os.getenv("BLOB_URL")
account_key = os.getenv("BLOB_ACCOUNT_KEY")
container_name = "container1"
blob_prefix = "Model_B"
local_model_path = "model_B"

os.makedirs(local_model_path, exist_ok=True)

blob_service_client = BlobServiceClient(account_url=account_url, credential=account_key)
container_client = blob_service_client.get_container_client(container_name)

model_files = []
for blob in container_client.list_blobs(name_starts_with=blob_prefix):
    blob_name = blob.name
    local_model_file_path = os.path.join(local_model_path, os.path.relpath(blob_name, blob_prefix))
    os.makedirs(os.path.dirname(local_model_file_path), exist_ok=True)
    blob_client = container_client.get_blob_client(blob_name)
    with open(local_model_file_path, "wb") as f:
        blob_data = blob_client.download_blob()
        blob_data.readinto(f)
    print(f"Modèle téléchargé depuis Azure Blob Storage à {local_model_file_path}")
    model_files.append(local_model_file_path)

custom_objects = {
    "binary_iou": BinaryIoU(name="binary_iou"),
    "binary_accuracy": BinaryAccuracy(),
    "precision": precision,
    "recall": recall
}

if model_files:
    model_path = model_files[-1]
    try:
        MODEL = load_model(local_model_path, custom_objects=custom_objects)
        print("Modèle chargé avec succès!")
    except Exception as e:
        print(f"Erreur lors du chargement du modèle : {e}")
else:
    print("Aucun modèle trouvé dans le container Azure Blob Storage.")
    MODEL = None

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
    image_array = Image.fromarray(image_array).resize((image_width, image_height))
    image_array = np.expand_dims(np.array(image_array), axis=0)

    if MODEL is None:
        print("Erreur : le modèle n'est pas disponible.")
        return np.zeros((image_height, image_width, 3))
    
    mask_predict = MODEL.predict(image_array)
    mask_predict = np.squeeze(mask_predict, axis=0)
    mask_class = np.argmax(mask_predict, axis=-1)
    mask_one_hot = np.eye(mask_predict.shape[-1])[mask_class]
    mask_color = generate_img_from_mask(mask_one_hot) * 255
    return mask_color

# Flask app
app = Flask(__name__)
swagger = Swagger(app)  # <<< AJOUT

@app.route("/")
def hello():
    """Accueil
    ---
    get:
      description: Bienvenue sur l'API de segmentation
      responses:
        200:
          description: Message de bienvenue
    """
    return "Hello, welcome to the segmentation API"

@app.route("/predict_mask", methods=["POST"])
@swag_from({
    'tags': ['Segmentation'],
    'parameters': [
        {
            'name': 'image',
            'in': 'formData',
            'type': 'file',
            'required': True,
            'description': 'Image à segmenter'
        }
    ],
    'responses': {
        200: {
            'description': 'Image segmentée en sortie',
            'content': {
                'image/png': {}
            }
        },
        400: {
            'description': 'Erreur dans l\'upload de l\'image'
        }
    }
})
def segment_image():
    if 'image' not in request.files:
        return jsonify({"error": "Aucune image fournie"}), 400

    file = request.files['image']
    
    try:
        img = Image.open(file.stream)
    except Exception as e:
        return jsonify({"error": f"Erreur lors de l'ouverture de l'image : {str(e)}"}), 400

    mask_color = predict_segmentation(np.array(img), MODEL_INPUT_WIDTH, MODEL_INPUT_HEIGHT)

    tmp_path = "tmp.png"
    Image.fromarray(mask_color.astype(np.uint8)).save(tmp_path)

    return send_file(tmp_path, mimetype="image/png")

#if __name__ == "__main__":
    #app.run(debug=True)
