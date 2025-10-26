# Ex.No 3 & 4 — Segmentation & Object Labeling
import cv2
import numpy as np

# --- SEGMENTATION ---
img = cv2.imread(r"D:\dl\pe.jpg")
if img is None:
    print("Image not found!")
    exit()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 1. Simple Threshold
_, thresh_simple = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
cv2.imshow("Simple Threshold", thresh_simple)

# 2. Adaptive Threshold
thresh_adapt = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 11, 2)
cv2.imshow("Adaptive Threshold", thresh_adapt)

# 3. Otsu's Threshold
_, thresh_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
cv2.imshow("Otsu's Threshold", thresh_otsu)

# 4. K-means Clustering
Z = img.reshape((-1,3)).astype(np.float32)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 4
_, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
img_kmeans = center[label.flatten()].reshape(img.shape).astype(np.uint8)
cv2.imshow("K-means Segmentation", img_kmeans)

# --- OBJECT LABELING ---
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
color_ranges = [(0,100,100),(10,255,255),   # Red
                (25,100,100),(35,255,255),  # Yellow
                (100,100,100),(120,255,255)] # Blue
labels = ["Red", "Yellow", "Blue"]

labeled_img = img.copy()
for i in range(0, len(color_ranges), 2):
    lower = np.array(color_ranges[i])
    upper = np.array(color_ranges[i+1])
    mask = cv2.inRange(hsv, lower, upper)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.contourArea(cnt) < 100:
            continue
        x,y,w,h = cv2.boundingRect(cnt)
        cv2.rectangle(labeled_img, (x,y), (x+w,y+h), (0,255,0), 2)
        cv2.putText(labeled_img, labels[i//2], (x,y-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
cv2.imshow("Labeled Image", labeled_img)

cv2.waitKey(0)
cv2.destroyAllWindows()
