from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import cv2
import numpy as np
import os

# Define the image path (Update this to match your file location)
image_path = r"C:\Users\Manasa Kavuri\OneDrive\Documents\Images Texas\vector\texas__2-6-2025__x7485-y13490-z15 (1).jpg"

# Check if the file exists
if not os.path.exists(image_path):
    print(f"❌ Error: File not found at {image_path}")
    exit()

# Load the image
image = cv2.imread(image_path)

# Handle OpenCV read failure
if image is None:
    print("❌ Error: Unable to load image. Check file path or format.")
    exit()

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 1️⃣ **Contrast Enhancement using CLAHE**
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
enhanced_gray = clahe.apply(gray)

# 2️⃣ **Adaptive Thresholding for better segmentation**
thresh = cv2.adaptiveThreshold(enhanced_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 11, 2)

# 3️⃣ **Edge Detection using Optimized Canny**
edges = cv2.Canny(thresh, 50, 150)

# 4️⃣ **Morphological Operations to remove noise**
kernel = np.ones((3, 3), np.uint8)
edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

# 5️⃣ **Find and classify contours**
contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

for cnt in contours:
    area = cv2.contourArea(cnt)
    if area > 500:  # Ignore small objects
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = float(w) / h

        # Classify objects based on area and shape
        if 0.8 < aspect_ratio < 1.2 and area > 2000:  # Square-like = Building
            color = (0, 0, 255)  # Red
        elif aspect_ratio > 2 or aspect_ratio < 0.5:  # Long = Road
            color = (255, 255, 255)  # White
        else:  # Irregular shape = Vegetation
            color = (0, 255, 0)  # Green
        
        cv2.drawContours(image, [cnt], -1, color, 2)

# Save and show the result
output_path = r"C:\Users\Manasa Kavuri\OneDrive\Documents\Images Texas\vector\processed_image.jpg"
cv2.imwrite(output_path, image)
cv2.imshow("Processed Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

print(f"✅ Processed image saved at {output_path}")
