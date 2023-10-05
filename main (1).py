'''

                            Online Python Compiler.
                Code, Compile, Run and Debug python program online.
Write your code in this editor and press "Run" button to execute it.

'''

# Load the face detection model
face_detection_net = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'res10_300x300_ssd_iter_140000.caffemodel')

# Load the face recognition model (if applicable)
# face_recognition_model = load_face_recognition_model()
# Capture frames from a video
cap = cv2.VideoCapture('video.mp4')

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Perform face detection on the frame
    faces = detect_faces(frame, face_detection_net)

    # If you're doing face recognition, identify the recognized faces
    # recognized_faces = recognize_faces(frame, faces, face_recognition_model)

    # Draw bounding boxes around detected faces
    draw_faces(frame, faces)

    # Display the processed frame
    cv2.imshow('Face Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
def detect_faces(frame, face_detection_net):
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104, 177, 123))
    face_detection_net.setInput(blob)
    detections = face_detection_net.forward()

    faces = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:  # You can adjust the confidence threshold
            box = detections[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
            (startX, startY, endX, endY) = box.astype(int)
            faces.append((startX, startY, endX, endY))

    return faces


