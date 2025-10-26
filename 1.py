# Ex.No: 1 & 2 — Implementation of Various Filter Techniques and Histogram
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Load image
img = cv2.imread(r"C:\Users\admin\Downloads\peacock.jpg")
if img is None:
    print("Image not found!")
    exit()

# Convert to RGB and Grayscale
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Step 2: Apply Filters
avg = cv2.blur(img_rgb, (5,5))
gauss = cv2.GaussianBlur(img_rgb, (5,5), 0)
median = cv2.medianBlur(img_rgb, 5)
bilateral = cv2.bilateralFilter(img_rgb, 9, 75, 75)
canny = cv2.Canny(gray, 100, 200)

# Step 3: Display Filter Outputs
titles = ['Original', 'Average', 'Gaussian', 'Median', 'Bilateral', 'Canny Edge']
images = [img_rgb, avg, gauss, median, bilateral, canny]

plt.figure(figsize=(10,7))
for i in range(6):
    plt.subplot(2,3,i+1)
    plt.imshow(images[i], cmap='gray' if len(images[i].shape)==2 else None)
    plt.title(titles[i])
    plt.axis('off')
plt.tight_layout()
plt.show()

# Step 4: Histogram Analysis
# Grayscale Histogram
hist_gray = cv2.calcHist([gray], [0], None, [256], [0,256])
plt.figure()
plt.title('Grayscale Histogram')
plt.plot(hist_gray, color='black')
plt.show()


# Color Histograms
colors = ('b','g','r')
plt.figure()
for i,col in enumerate(colors):
    hist = cv2.calcHist([img], [i], None, [256], [0,256])
    plt.plot(hist, color=col)
plt.title('Color Histograms')
plt.show()

# Step 5: Histogram Equalization
eq = cv2.equalizeHist(gray)
cv2.imshow('Original Gray', gray)
cv2.imshow('Equalized Gray', eq)
cv2.waitKey(0)
cv2.destroyAllWindows()

