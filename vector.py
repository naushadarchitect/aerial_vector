from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import cv2
import numpy as np
import json
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)
UPLOAD_FOLDER = "uploads"
RESULT_FOLDER = "results"
VECTOR_FOLDER = "vectors"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER
app.config['VECTOR_FOLDER'] = VECTOR_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)
os.makedirs(VECTOR_FOLDER, exist_ok=True)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB limit

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "tif", "tiff"}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_image(image_path, output_image_path):
    image = cv2.imread(image_path)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    feature_data = {
        "buildings": [], "roads": [], "tree_canopy": [], "swimming_pools": [],
        "pavements": [], "driveways": [], "forests": [], "maintained_grass": [], "open_water": []
    }
    
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 5, True)
        poly = [[int(point[0][0]), int(point[0][1])] for point in approx]
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)
        
        if area > 5000:
            feature_data["buildings"].append(poly)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
        elif area > 2000:
            feature_data["roads"].append(poly)
            cv2.line(image, (x, y), (x + w, y + h), (255, 255, 255), 2)
        elif area > 1000:
            feature_data["tree_canopy"].append(poly)
            cv2.circle(image, (x + w//2, y + h//2), 10, (0, 255, 0), -1)
        elif area > 500:
            feature_data["pavements"].append(poly)
            cv2.rectangle(image, (x, y), (x + w, y + h), (200, 200, 200), 1)
        else:
            feature_data["open_water"].append(poly)
            cv2.ellipse(image, (x + w//2, y + h//2), (w//3, h//3), 0, 0, 360, (255, 0, 0), 2)
    
    cv2.imwrite(output_image_path, image)
    return feature_data

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type or no file selected"}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    output_image_path = os.path.join(app.config['RESULT_FOLDER'], filename)

    feature_data = process_image(filepath, output_image_path)
    vector_path = os.path.join(app.config['VECTOR_FOLDER'], filename.rsplit('.', 1)[0] + '.json')
    
    if os.path.exists(vector_path):
        os.remove(vector_path)
    
    with open(vector_path, 'w') as f:
        json.dump(feature_data, f, indent=4)
    
    return jsonify({"image_id": filename, "vector_id": vector_path, "processed_image": output_image_path})

@app.route('/result/<image_id>', methods=['GET'])
def get_result(image_id):
    result_path = os.path.join(app.config['VECTOR_FOLDER'], image_id.rsplit('.', 1)[0] + '.json')
    if not os.path.exists(result_path):
        return jsonify({"error": "Result not found"}), 404
    return send_file(result_path, mimetype='application/json')

@app.route('/processed_image/<image_id>', methods=['GET'])
def get_processed_image(image_id):
    processed_image_path = os.path.join(app.config['RESULT_FOLDER'], image_id)
    if not os.path.exists(processed_image_path):
        return jsonify({"error": "Processed image not found"}), 404
    return send_file(processed_image_path, mimetype='image/jpeg')

if __name__ == '__main__':
    print("🚀 Backend running at: http://127.0.0.1:5000/")
    app.run(debug=True, host="0.0.0.0", port=5000)
