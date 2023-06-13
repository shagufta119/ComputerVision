import os
import socket
import subprocess
import cv2
import imutils
import numpy as np
from flask import Flask, app, Response, request,render_template

socket.getaddrinfo('localhost', 8080)

from werkzeug.utils import secure_filename

app = Flask(__name__)
video_path = "test_video.mp4"
cap = cv2.VideoCapture(video_path)
def gen_video():
    prototxt = "MobileNetSSD_deploy.prototxt"
    model_path = "MobileNetSSD_deploy.caffemodel"
    net = cv2.dnn.readNetFromCaffe(prototxt, model_path)

    person_count = 0
    person_ids = []
    tracked_people = {}
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = imutils.resize(frame, width=600)
        blob = cv2.dnn.blobFromImage(frame, 0.007843, (frame.shape[1], frame.shape[0]), 127.5)
        net.setInput(blob)
        detections = net.forward()

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                class_id = int(detections[0, 0, i, 1])
                if class_id == 15:
                    box = detections[0, 0, i, 3:7] * np.array(
                        [frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
                    (startX, startY, endX, endY) = box.astype("int")

                    centroid = ((startX + endX) // 2, (startY + endY) // 2)

                    person_id = -1
                    for pid, track in tracked_people.items():
                        (prev_centroid, _) = track
                        distance = np.linalg.norm(np.array(prev_centroid) - np.array(centroid))

                        if distance < 50:
                            person_id = pid
                            break
                    if person_id == -1:
                        person_id = person_count
                        person_count += 1
                    tracked_people[person_id] = (centroid, box)

                    cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                    cv2.putText(frame, f"Person {person_id}", (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                                (0, 255, 0), 2)
        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

    @app.route('/home')
    def home():
        return "HUMAN DETECTION WEBSITE"

    app.run(debug=True)
    prototxt = "MobileNetSSD_deploy.prototxt"
    model_path = "MobileNetSSD_deploy.caffemodel"
    net = cv2.dnn.readNetFromCaffe(prototxt, model_path)

    person_count = 0
    person_ids = []
    tracked_people = {}
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = imutils.resize(frame, width=600)
        blob = cv2.dnn.blobFromImage(frame, 0.007843, (frame.shape[1], frame.shape[0]), 127.5)
        net.setInput(blob)
        detections = net.forward()

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                class_id = int(detections[0, 0, i, 1])
                if class_id == 15:
                    box = detections[0, 0, i, 3:7] * np.array(
                        [frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
                    (startX, startY, endX, endY) = box.astype("int")

                    centroid = ((startX + endX) // 2, (startY + endY) // 2)

                    person_id = -1
                    for pid, track in tracked_people.items():
                        (prev_centroid, _) = track
                        distance = np.linalg.norm(np.array(prev_centroid) - np.array(centroid))

                        if distance < 50:
                            person_id = pid
                            break
                    if person_id == -1:
                        person_id = person_count
                        person_count += 1
                    tracked_people[person_id] = (centroid, box)

                    cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                    cv2.putText(frame, f"Person {person_id}", (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                                (0, 255, 0), 2)
        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

    @app.route("/detect", methods=['POST'])
    def detect(uploads_dir=None):
        if not request.method == "POST":
            return
        video = request.files['video']
        video.save(os.path.join(uploads_dir, secure_filename(video.filename)))
        print(video)
        subprocess.run("ls")
        subprocess.run(['python3', 'detect.py', '--source', os.path.join(uploads_dir, secure_filename(video.filename))])

        # return os.path.join(uploads_dir, secure_filename(video.filename))
        obj = secure_filename(video.filename)
        return obj
@app.route('/')
def welcome():
    return render_template("index.html")



@app.route('/video',methods =["POST"])
def video_feed():
    return Response(gen_video())
if __name__=='__main__':
        app.run(debug=True)



