import cv2
import easyocr
import matplotlib.pyplot as plt
import os

# === Load Haar Cascade for Russian-style license plates ===
cascade_path = cv2.data.haarcascades + 'haarcascade_russian_plate_number.xml'
plate_cascade = cv2.CascadeClassifier(cascade_path)

# === Load the image ===
image_path = r'C:\Users\HP\Downloads\car.jpeg'

if not os.path.exists(image_path):
    raise FileNotFoundError(f"Image not found: {image_path}")

image = cv2.imread(image_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# === Detect license plates ===
plates = plate_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

# === Initialize EasyOCR reader ===
reader = easyocr.Reader(['en'], gpu=False)  # set gpu=True if you have CUDA setup

# === Process each detected plate ===
if len(plates) == 0:
    print("No license plates detected.")
else:
    for i, (x, y, w, h) in enumerate(plates):
        roi = image[y:y + h, x:x + w]

        # OCR on ROI
        results = reader.readtext(roi)

        # Draw rectangle and overlay text
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        if results:
            for detection in results:
                text = detection[1]
                print(f"Detected Plate {i+1}: {text}")
                cv2.putText(image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.9, (0, 255, 255), 2)

# === Show the final image ===
plt.figure(figsize=(10, 6))
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.title('License Plate Detection')
plt.show()

# === Optional: Save the output ===
# cv2.imwrite("C:\\Users\\HP\\Downloads\\output_detected_plate.jpg", image)
