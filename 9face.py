# Ex.No: 9 & 10 — Combined Face Recognition System
import cv2
import numpy as np
import matplotlib.pyplot as plt

# ---------- COLOR MODEL BASED FACE RECOGNITION ----------
def recognize_faces_color_histograms(test_image, known_images, threshold=0.5):
    def calc_hist(img):
        hist = cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8],
                            [0, 256, 0, 256, 0, 256])
        return cv2.normalize(hist, hist).flatten()

    def compare(h1, h2):
        return cv2.compareHist(h1, h2, cv2.HISTCMP_BHATTACHARYYA)

    test_hist = calc_hist(test_image)
    distances = [compare(test_hist, calc_hist(k)) for k in known_images]
    min_index = np.argmin(distances)
    if distances[min_index] <= threshold:
        print(f"Face recognized as Person {min_index + 1}")
    else:
        print("Face not recognized")

    plt.figure(figsize=(10, 5))
    plt.subplot(1, len(known_images) + 1, 1)
    plt.imshow(cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB))
    plt.title('Test Image')
    for i, img in enumerate(known_images):
        plt.subplot(1, len(known_images) + 1, i + 2)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title(f'Known {i + 1}')
    plt.tight_layout()
    plt.show()


# ---------- AUTHORIZED FACE RECOGNITION USING FEATURE MATCHING ----------
def authorized_face_check(auth_img_path, test_img_path):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    orb = cv2.ORB_create()

    auth_img = cv2.imread(auth_img_path, cv2.IMREAD_GRAYSCALE)
    test_img = cv2.imread(test_img_path)
    if auth_img is None or test_img is None:
        print("Error: One of the images not found!")
        return

    kp1, des1 = orb.detectAndCompute(auth_img, None)
    gray_test = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_test, 1.1, 4)

    for (x, y, w, h) in faces:
        face_roi = gray_test[y:y+h, x:x+w]
        kp2, des2 = orb.detectAndCompute(face_roi, None)
        if des2 is not None:
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = sorted(bf.match(des1, des2), key=lambda x: x.distance)
            good = [m for m in matches if m.distance < 50]

            match_img = cv2.drawMatches(auth_img, kp1, face_roi, kp2, matches[:10], None, flags=2)
            plt.figure(figsize=(10, 6))
            plt.title("Feature Matching - Authorized vs Test Face")
            plt.imshow(cv2.cvtColor(match_img, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            plt.show()

            if len(good) > 10:
                print("✅ Authorized Face Detected")
                return
    print("❌ Unauthorized Face")


# ---------- RUN ALL ----------
if __name__ == "__main__":
    test_img = cv2.imread(r"D:\DL\female1.jpg")
    known_imgs = [cv2.imread(r"D:\DL\female.jpg") for _ in range(3)]
    recognize_faces_color_histograms(test_img, known_imgs)
    authorized_face_check(r"D:\DL\obma1.jpg", r"D:\DL\Obma.jpg")
