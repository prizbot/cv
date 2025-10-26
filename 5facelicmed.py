# Ex.No 5,6,7 — Face Recognition, License Plate, Medical Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pytesseract
import os

# ------------------ FACE DETECTION ------------------
def dnn_face_detection(image_path):
    modelFile = r"D:\DL\res10_300x300_ssd_iter_140000.caffemodel"
    configFile = r"D:\DL\deploy.prototxt"
    net = cv2.dnn.readNetFromCaffe(configFile, modelFile)

    img = cv2.imread(image_path)
    if img is None:
        print("Face image not found!")
        return

    h, w = img.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    count = 0
    for i in range(detections.shape[2]):
        confidence = detections[0,0,i,2]
        if confidence > 0.5:
            box = detections[0,0,i,3:7]*[w,h,w,h]
            startX,startY,endX,endY = box.astype(int)
            cv2.rectangle(img, (startX,startY), (endX,endY), (0,255,0), 2)
            cv2.putText(img, "Face", (startX,startY-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0),2)
            count += 1
    print(f"Faces detected: {count}")
    cv2.putText(img, f"Faces: {count}", (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),2)
    cv2.imshow("Face Detection", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ------------------ LICENSE PLATE IDENTIFICATION ------------------
def license_plate_detection(image_path):
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    cascade_path = r"D:\dl\haarcascade_russian_plate_number.xml"
    if not os.path.exists(cascade_path):
        print("Cascade XML not found!")
        return
    cascade = cv2.CascadeClassifier(cascade_path)
    states = {"AN":"Andaman","AP":"Andhra Pradesh","DL":"Delhi","MH":"Maharashtra"} # simplified for brevity

    img = cv2.imread(image_path)
    if img is None:
        print("License plate image not found!")
        return
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    plates = cascade.detectMultiScale(gray, 1.1, 4)
    for (x,y,w,h) in plates:
        plate = img[y:y+h, x:x+w]
        _, plate_bin = cv2.threshold(cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY), 127, 255, cv2.THRESH_BINARY)
        text = pytesseract.image_to_string(plate_bin, config='--psm 8')
        text = ''.join(filter(str.isalnum,text)).upper()
        state = states.get(text[:2],"Unknown")
        print(f"Detected: {text}, State: {state}")
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,0,255),2)
        cv2.putText(img, text, (x,y-10), cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
    cv2.imshow("License Plate", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ------------------ MEDICAL IMAGE PROCESSING ------------------
def medical_image_processing(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Medical image not found!")
        return
    blurred = cv2.GaussianBlur(img,(5,5),0)
    _, thresh = cv2.threshold(blurred,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    output = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(output, contours, -1, (0,255,0),2)
    plt.figure(figsize=(15,5))
    for i,img_set in enumerate([img,blurred,thresh,output]):
        plt.subplot(1,4,i+1)
        plt.imshow(img_set, cmap='gray' if i!=3 else None)
        plt.axis('off')
        plt.title(["Original","Blurred","Thresholded","Contours"][i])
    plt.tight_layout()
    plt.show()


# ------------------ RUN ALL ------------------
if __name__=="__main__":
    dnn_face_detection(r"D:\DL\grppeople.jpg")
    license_plate_detection(r"D:\dl\car.jpg")
    medical_image_processing(r'D:\dl\brain.png')
