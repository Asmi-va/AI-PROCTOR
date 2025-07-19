import cv2
import dlib
from scipy.spatial import distance as dist
import imutils
import numpy as np
import os

# Function to calculate Eye Aspect Ratio (EAR)
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Helper function to estimate gaze direction
# Returns True if looking forward, False if looking away
# Uses the ratio of white space on left/right of the eye

def is_looking_forward(eye):
    # Compute the min/max x-coordinates
    min_x = np.min(eye[:, 0])
    max_x = np.max(eye[:, 0])
    # Approximate the center of the eye
    center_x = np.mean(eye[:, 0])
    # Ratio: how centered is the eye
    ratio = (center_x - min_x) / (max_x - min_x + 1e-6)
    # If ratio is near 0.5, looking forward; else, looking left/right
    return 0.35 < ratio < 0.65

# Head pose estimation helper
# Returns True if looking forward, False if looking away

def is_facing_forward(shape):
    # 3D model points of facial landmarks (nose, eyes, mouth corners, chin)
    model_points = np.array([
        (0.0, 0.0, 0.0),             # Nose tip
        (0.0, -330.0, -65.0),        # Chin
        (-225.0, 170.0, -135.0),     # Left eye left corner
        (225.0, 170.0, -135.0),      # Right eye right corner
        (-150.0, -150.0, -125.0),    # Left mouth corner
        (150.0, -150.0, -125.0)      # Right mouth corner
    ])
    # 2D image points from detected landmarks
    image_points = np.array([
        shape[30],     # Nose tip
        shape[8],      # Chin
        shape[36],     # Left eye left corner
        shape[45],     # Right eye right corner
        shape[48],     # Left mouth corner
        shape[54]      # Right mouth corner
    ], dtype='double')
    # Camera internals
    size = (frame.shape[1], frame.shape[0])
    focal_length = size[1]
    center = (size[1] / 2, size[0] / 2)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]],
         [0, focal_length, center[1]],
         [0, 0, 1]], dtype='double')
    dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
    # SolvePnP to get rotation vector
    success, rotation_vector, translation_vector = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
    if not success:
        return True  # If can't estimate, assume facing forward
    # Convert rotation vector to rotation matrix
    rotation_mat, _ = cv2.Rodrigues(rotation_vector)
    pose_mat = cv2.hconcat((rotation_mat, translation_vector))
    _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_mat)
    yaw = euler_angles[1, 0]
    # Debug print for calibration
    print(f"Yaw: {yaw:.2f}")
    # Only use yaw for left/right detection
    if abs(yaw) < 35:
        return True   # Looking forward
    return False      # Looking away

# Constants for blink detection
EYE_AR_THRESH = 0.23
EYE_AR_CONSEC_FRAMES = 3

COUNTER = 0
BLINKED = False

# Load the pre-trained Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load dlib's face detector and facial landmark predictor
predictor_path = os.path.join(os.path.dirname(__file__), 'shape_predictor_68_face_landmarks.dat')
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

# Indexes for left and right eye landmarks
(lStart, lEnd) = (42, 48)
(rStart, rEnd) = (36, 42)

# Start video capture from the default webcam (0)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Press 'q' to quit.")
print("Liveness detection: Blink your eyes to prove you are live!")

# Add a global variable to store the last yaw value
last_yaw = 0.0

# Add YOLOv5 for object detection
import torch

yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
prohibited_items = {'cell phone', 'laptop', 'book', 'tv', 'remote'}

def detect_prohibited_items(frame):
    # YOLO expects RGB
    results = yolo_model(frame[..., ::-1])
    detected = []
    for *box, conf, cls in results.xyxy[0].tolist():
        label = results.names[int(cls)]
        if label in prohibited_items:
            detected.append((label, box))
    return detected

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    frame = imutils.resize(frame, width=600)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces using Haar Cascade
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Detect faces using dlib for landmarks
    rects = detector(gray, 0)
    liveness_text = ""
    face_count = len(rects)
    warning_text = ""
    if face_count > 1:
        warning_text = f"Warning: {face_count} faces detected! Only one allowed."

    gaze_warning = ""
    for rect in rects:
        shape = predictor(gray, rect)
        shape = np.array([[p.x, p.y] for p in shape.parts()])
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0

        # Head pose estimation for gaze
        # Save the yaw value for display
        model_points = np.array([
            (0.0, 0.0, 0.0),             # Nose tip
            (0.0, -330.0, -65.0),        # Chin
            (-225.0, 170.0, -135.0),     # Left eye left corner
            (225.0, 170.0, -135.0),      # Right eye right corner
            (-150.0, -150.0, -125.0),    # Left mouth corner
            (150.0, -150.0, -125.0)      # Right mouth corner
        ])
        image_points = np.array([
            shape[30],     # Nose tip
            shape[8],      # Chin
            shape[36],     # Left eye left corner
            shape[45],     # Right eye right corner
            shape[48],     # Left mouth corner
            shape[54]      # Right mouth corner
        ], dtype='double')
        size = (frame.shape[1], frame.shape[0])
        focal_length = size[1]
        center = (size[1] / 2, size[0] / 2)
        camera_matrix = np.array(
            [[focal_length, 0, center[0]],
             [0, focal_length, center[1]],
             [0, 0, 1]], dtype='double')
        dist_coeffs = np.zeros((4, 1))
        success, rotation_vector, translation_vector = cv2.solvePnP(
            model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
        if success:
            rotation_mat, _ = cv2.Rodrigues(rotation_vector)
            pose_mat = cv2.hconcat((rotation_mat, translation_vector))
            _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_mat)
            yaw = euler_angles[1, 0]
            last_yaw = yaw
            print(f"Yaw: {yaw:.2f}")
            if abs(yaw) >= 35:
                gaze_warning = "Warning: Please look at the camera!"
        # Blink detection
        if ear < EYE_AR_THRESH:
            COUNTER += 1
        else:
            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                BLINKED = True
            COUNTER = 0
        if BLINKED:
            liveness_text = "Live (Blinked)"
            BLINKED = False
        # Draw a smaller rectangle around face (tighter box)
        (x, y, w, h) = (rect.left(), rect.top(), rect.width(), rect.height())
        shrink_factor = 0.2
        x_new = int(x + w * shrink_factor / 2)
        y_new = int(y + h * shrink_factor / 2)
        w_new = int(w * (1 - shrink_factor))
        h_new = int(h * (1 - shrink_factor))
        cv2.rectangle(frame, (x_new, y_new), (x_new + w_new, y_new + h_new), (0, 255, 0), 2)

    # Draw rectangles for Haar Cascade faces (optional, for comparison)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 1)

    # In the main loop, after reading the frame:
    prohibited_warning = ""
    detected_items = detect_prohibited_items(frame)
    for label, box in detected_items:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
    if detected_items:
        prohibited_warning = "Warning: Prohibited item detected!"

    # Show liveness status only if a blink is detected
    if liveness_text:
        cv2.putText(frame, f"Liveness: {liveness_text}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    if warning_text:
        cv2.putText(frame, warning_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
    if gaze_warning:
        cv2.putText(frame, gaze_warning, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
    if prohibited_warning:
        cv2.putText(frame, prohibited_warning, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
    # Show yaw value for calibration
    cv2.putText(frame, f"Yaw: {last_yaw:.2f}", (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    cv2.imshow('Real-Time Face & Liveness Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Instructions:
# 1. Download shape_predictor_68_face_landmarks.dat from:
#    http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
#    (Unzip and place in the same directory as this script)
# 2. Install dependencies: pip install -r requirements.txt
# 3. Run the script: python face_detection_realtime.py
# 4. Blink your eyes to trigger liveness detection.
# 5. Press 'q' to quit the application. 