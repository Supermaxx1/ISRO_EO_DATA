from flask import Flask, request, jsonify
from fusion_layer import ProjectionAlignment
from vision_encoder import extract_image_features
from text_model import get_text_embeddings
from flask_cors import CORS
import torch

app = Flask(__name__)
CORS(app)  # Add this directly after app initialization

# Load trained projection layer
projection = ProjectionAlignment()
projection.load_state_dict(torch.load("models/projection.pt", map_location="cpu"))
projection.eval()

@app.route('/analyze', methods=['POST'])
def analyze():
    img_file = request.files['image']
    query = request.form['query']
    # Save uploaded image temporarily
    img_path = "data/temp_uploaded_image.tif"
    img_file.save(img_path)
    # Extract image features
    image_embeds = extract_image_features(img_path)
    projected_embeds = projection(image_embeds).detach()
    # Text query
    text_embeds = get_text_embeddings(query).detach()
    # Simple similarity score (cosine)
    sim_score = torch.nn.functional.cosine_similarity(projected_embeds, text_embeds).item()
    # Response
    return jsonify({
        "similarity_score": sim_score,
        "query": query,
        "status": "success"
    })

if __name__ == '__main__':
    app.run(debug=True, port=8000)
