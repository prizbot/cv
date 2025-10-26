import cv2
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# ----------------- HUMAN POSE ESTIMATION -----------------
pose_net = cv.dnn.readNetFromTensorflow(r"D:\dl\graph_opt.pb")
inWidth, inHeight, thr = 368, 368, 0.2
BODY_PARTS = {"Nose":0,"Neck":1,"RShoulder":2,"RElbow":3,"RWrist":4,
              "LShoulder":5,"LElbow":6,"LWrist":7,"RHip":8,"RKnee":9,"RAnkle":10,
              "LHip":11,"LKnee":12,"LAnkle":13,"REye":14,"LEye":15,"REar":16,"LEar":17}
POSE_PAIRS = [("Neck","RShoulder"),("Neck","LShoulder"),("RShoulder","RElbow"),("RElbow","RWrist"),
              ("LShoulder","LElbow"),("LElbow","LWrist"),("Neck","RHip"),("RHip","RKnee"),("RKnee","RAnkle"),
              ("Neck","LHip"),("LHip","LKnee"),("LKnee","LAnkle"),("Neck","Nose"),("Nose","REye"),
              ("REye","REar"),("Nose","LEye"),("LEye","LEar")]

def pose_estimation(frame):
    h, w = frame.shape[:2]
    blob = cv.dnn.blobFromImage(frame, 1.0, (inWidth,inHeight),(127.5,127.5,127.5), swapRB=True, crop=False)
    pose_net.setInput(blob)
    out = pose_net.forward()[:, :len(BODY_PARTS), :, :]
    points = []
    for i in range(len(BODY_PARTS)):
        _, conf, _, point = cv.minMaxLoc(out[0,i,:,:])
        x, y = int(point[0]*w/out.shape[3]), int(point[1]*h/out.shape[2])
        points.append((x,y) if conf>thr else None)
    for pair in POSE_PAIRS:
        if points[BODY_PARTS[pair[0]]] and points[BODY_PARTS[pair[1]]]:
            cv.line(frame, points[BODY_PARTS[pair[0]]], points[BODY_PARTS[pair[1]]], (0,255,0), 3)
            cv.circle(frame, points[BODY_PARTS[pair[0]]], 3, (0,0,255), -1)
            cv.circle(frame, points[BODY_PARTS[pair[1]]], 3, (0,0,255), -1)
    return frame

# ----------------- VEHICLE DETECTION YOLOv4 -----------------
yolo_net = cv.dnn.readNet(r"D:\dl\yolov4.weights", r"D:\dl\yolov4.cfg")
with open(r"D:\dl\coco.names") as f: classes = [line.strip() for line in f]
video_path = r"traffic.mp4"

def vehicle_detection(frame):
    h, w = frame.shape[:2]
    blob = cv.dnn.blobFromImage(frame,1/255.0,(608,608),(0,0,0),swapRB=True,crop=False)
    yolo_net.setInput(blob)
    outs = yolo_net.forward([yolo_net.getLayerNames()[i-1] for i in yolo_net.getUnconnectedOutLayers().flatten()])
    boxes, confidences, class_ids = [], [], []
    for out in outs:
        for det in out:
            scores = det[5:]
            class_id = np.argmax(scores)
            conf = scores[class_id]
            if conf>0.3 and classes[class_id] in ["car","truck","bus","motorbike"]:
                cx, cy, bw, bh = [int(det[i]*w) if i<4 else 0 for i in range(4)]
                x, y = cx - bw//2, cy - bh//2
                boxes.append([x,y,bw,bh]); confidences.append(float(conf)); class_ids.append(class_id)
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.3, 0.4)
    for i in indexes.flatten() if len(indexes)>0 else []:
        x, y, bw, bh = boxes[i]
        cv.rectangle(frame,(x,y),(x+bw,y+bh),(0,255,0),2)
        cv.putText(frame, classes[class_ids[i]],(x,y-10),cv.FONT_HERSHEY_PLAIN,1,(0,255,0),2)
    return frame

# ----------------- HUMAN ACTION RECOGNITION -----------------
action_model = load_model(r"/content/model.h5")
action_classes = ['walking','running','jumping','standing','sitting','falling','other']

def predict_action(img_path):
    img = image.load_img(img_path,target_size=(90,3),color_mode='grayscale')
    arr = np.expand_dims(image.img_to_array(img), axis=0)/255.0
    return action_classes[np.argmax(action_model.predict(arr))]

# ----------------- MAIN EXECUTION -----------------
if __name__=="__main__":
    # Pose estimation example
    img = cv.imread(r"D:\dl\humafomal.jpg")
    pose_img = pose_estimation(img)
    plt.imshow(cv.cvtColor(pose_img,cv.COLOR_BGR2RGB)); plt.axis("off"); plt.show()

    # Vehicle detection example
    cap = cv.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        out_frame = vehicle_detection(frame)
        cv.imshow("Vehicle Detection", out_frame)
        if cv.waitKey(10)&0xFF==ord('q'): break
    cap.release(); cv.destroyAllWindows()

    # Human action recognition example
    action = predict_action(r'/content/jump 4.jpg')
    print(f"Predicted Action: {action}")
