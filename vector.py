import cv2
import numpy as np
import os

# Define image path
image_path = r"C:\Users\Manasa Kavuri\OneDrive\Documents\Images Texas\vector\texas__2-6-2025__x7486-y13491-z15.jpg"
output_path = r"C:\Users\Manasa Kavuri\OneDrive\Documents\Images Texas\vector\processed_image.jpg"

if not os.path.exists(image_path):
    print(f"❌ Error: File not found at {image_path}")
    exit()

# Load image
image = cv2.imread(image_path)
if image is None:
    print("❌ Error: Unable to load image. Check file path or format.")
    exit()

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
enhanced_gray = clahe.apply(gray)
thresh = cv2.adaptiveThreshold(enhanced_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 11, 2)
edges = cv2.Canny(thresh, 50, 150)
kernel = np.ones((3, 3), np.uint8)
edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

for cnt in contours:
    area = cv2.contourArea(cnt)
    if area < 500:
        continue
    
    x, y, w, h = cv2.boundingRect(cnt)
    aspect_ratio = float(w) / h
    
    if 0.8 < aspect_ratio < 1.2 and area > 2000:
        color = (0, 0, 255)  # Red - Buildings
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
    elif aspect_ratio > 2 or aspect_ratio < 0.5:
        color = (255, 255, 255)  # White - Roads
        cv2.line(image, (x, y), (x + w, y + h), color, 2)
    elif area > 1000 and area < 5000:
        color = (0, 255, 0)  # Green - Tree Canopy
        cv2.circle(image, (x + w // 2, y + h // 2), max(w, h) // 3, color, 2)
    elif 5000 < area < 15000 and 1.2 < aspect_ratio < 2:
        color = (255, 0, 0)  # Blue - Swimming Pools
        cv2.ellipse(image, (x + w // 2, y + h // 2), (w // 2, h // 2), 0, 0, 360, color, 2)
    elif 2000 < area < 7000:
        color = (0, 255, 255)  # Yellow - Pavements
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 1, lineType=cv2.LINE_4)
    elif 7000 < area < 15000:
        color = (0, 165, 255)  # Orange - Driveways
        cv2.arrowedLine(image, (x, y), (x + w, y + h), color, 2)
    elif area > 15000:
        color = (34, 139, 34)  # Dark Green - Forests
        for i in range(5):
            cv2.circle(image, (x + np.random.randint(w), y + np.random.randint(h)), 10, color, 2)
    elif 3000 < area < 8000:
        color = (144, 238, 144)  # Light Green - Maintained Grass
        cv2.rectangle(image, (x, y), (x + w, y + h), color, -1)
    elif area > 10000 and 1.5 < aspect_ratio < 3:
        color = (255, 0, 0)  # Blue - Open Water
        cv2.ellipse(image, (x + w // 2, y + h // 2), (w // 2, h // 2), 0, 0, 360, color, -1)

cv2.imwrite(output_path, image)
cv2.imshow("Processed Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

print(f"✅ Processed image saved at {output_path}")
