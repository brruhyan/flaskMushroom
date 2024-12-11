from flask import Flask, render_template, request, redirect, url_for, jsonify
from werkzeug.utils import secure_filename
from PIL import Image, ImageDraw, ImageFont
import requests
import os

# Flask app configuration
app = Flask(__name__)

UPLOAD_FOLDER = './static/uploads'
RESULT_FOLDER = './static/results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

# Roboflow API configuration
ROBOFLOW_API_URL = "https://outline.roboflow.com/final-data-set-zs3lw/2"
ROBOFLOW_API_KEY = "5yNSbYmPfQArT0CrClDY"

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "Empty filename"}), 400

    if file:
        # Save the uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Perform inference using Roboflow
        try:
            predictions = get_roboflow_predictions(filepath)

            # Overlay predictions on the image
            result_path = overlay_predictions(filepath, predictions)

            # Count classes in the predictions
            counts = count_prediction_classes(predictions)
            total_mushrooms = sum(counts.values())  # Total mushrooms

            # Return paths to original and result images, along with prediction counts
            return jsonify({
                "original_image_url": f"/static/uploads/{filename}",
                "processed_image_url": f"/static/results/{os.path.basename(result_path)}",
                "ready": counts.get("Ready", 0),
                "notReady": counts.get("Not-Ready", 0),
                "overdue": counts.get("Overdue", 0),
                "totalMushrooms": total_mushrooms  # Send the total captured mushrooms count
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    return jsonify({"error": "File processing failed"}), 500

def get_roboflow_predictions(image_path):
    """Send the uploaded image to Roboflow for predictions."""
    with open(image_path, "rb") as image_file:
        response = requests.post(
            f"{ROBOFLOW_API_URL}?api_key={ROBOFLOW_API_KEY}",
            files={"file": image_file}
        )
    response.raise_for_status()
    return response.json()

def overlay_predictions(image_path, predictions):
    """Overlay predictions on the uploaded image."""
    image = Image.open(image_path).convert("RGBA")
    overlay = Image.new("RGBA", image.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)

    font = ImageFont.load_default()

    for prediction in predictions.get("predictions", []):
        points = [(p["x"], p["y"]) for p in prediction["points"]]
        label = prediction["class"]

        # Choose color based on class
        color = {
            "Ready": (72, 211, 138, 128),      # Green
            "Not-Ready": (255, 215, 0, 128),  # Yellow
            "Overdue": (255, 0, 0, 128),      # Red
        }.get(label, (0, 0, 0, 128))

        # Draw polygon
        draw.polygon(points, fill=color, outline=color[:3])

        # Draw label
        label_position = (points[0][0], points[0][1] - 20)
        draw.text(label_position, label, fill=(255, 255, 255), font=font)

    # Composite overlay onto the image
    result = Image.alpha_composite(image, overlay)
    result_path = os.path.join(app.config['RESULT_FOLDER'], os.path.basename(image_path))
    result.save(result_path, format="PNG")
    return result_path

def count_prediction_classes(predictions):
    """Count the number of predictions for each class."""
    class_counts = {}
    for prediction in predictions.get("predictions", []):
        label = prediction["class"]
        class_counts[label] = class_counts.get(label, 0) + 1
    return class_counts

if __name__ == "__main__":
    app.run(debug=True)
