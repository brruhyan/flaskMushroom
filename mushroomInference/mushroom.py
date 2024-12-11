from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from PIL import Image, ImageDraw, ImageFont
import requests
import os

app = Flask(__name__)

UPLOAD_FOLDER = './uploads'
RESULT_FOLDER = './results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

ROBOFLOW_API_URL = "https://outline.roboflow.com/final-data-set-zs3lw/2"
ROBOFLOW_API_KEY = "5yNSbYmPfQArT0CrClDY"

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)

    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Perform inference using Roboflow
        predictions = get_roboflow_predictions(filepath)

        # Overlay predictions on the image
        result_path = overlay_predictions(filepath, predictions)
        result_image = os.path.basename(result_path)

        # Pass the result image name to the template
        return render_template('index.html', result_image=result_image)

    return redirect(url_for('home'))

def get_roboflow_predictions(image_path):
    with open(image_path, "rb") as image_file:
        response = requests.post(
            f"{ROBOFLOW_API_URL}?api_key={ROBOFLOW_API_KEY}",
            files={"file": image_file}
        )
    response.raise_for_status()
    return response.json()

def overlay_predictions(image_path, predictions):
    image = Image.open(image_path).convert("RGBA")
    overlay = Image.new("RGBA", image.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)

    font = ImageFont.load_default()

    for prediction in predictions.get("predictions", []):
        points = [(p["x"], p["y"]) for p in prediction["points"]]
        label = prediction["class"]

        # Choose color based on class
        color = {
            "Ready": (72, 211, 138, 128),
            "Not-Ready": (255, 215, 0, 128),
            "Overdue": (255, 0, 0, 128),
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

if __name__ == "__main__":
    app.run(debug=True)
