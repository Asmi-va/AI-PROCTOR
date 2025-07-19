import cv2
from ultralytics import YOLO
import numpy as np

# Load YOLOv8-pose model (pretrained on COCO keypoints)
model = YOLO('yolov8n-pose.pt')  # Use 'yolov8n-pose.pt' for speed, or 'yolov8s-pose.pt' for more accuracy

# Suspicious activity detection thresholds
LEAN_EDGE_THRESH = 0.12  # % of frame width/height
REACH_DIST_THRESH = 0.35  # % of torso length
TYPING_Y_THRESH = 0.15    # % of frame height from bottom

# Keypoint indices (COCO format)
NOSE = 0
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_ELBOW = 7
RIGHT_ELBOW = 8
LEFT_WRIST = 9
RIGHT_WRIST = 10
LEFT_HIP = 11
RIGHT_HIP = 12

# Helper to check if a keypoint is valid
is_valid = lambda kp: kp[2] > 0.3

def suspicious_activity(keypoints, frame_shape):
    h, w = frame_shape[:2]
    warnings = []
    kp = keypoints
    # Leaning out of frame
    for idx in [NOSE, LEFT_SHOULDER, RIGHT_SHOULDER]:
        if is_valid(kp[idx]):
            x, y = kp[idx][0] * w, kp[idx][1] * h
            if x < w * LEAN_EDGE_THRESH or x > w * (1 - LEAN_EDGE_THRESH):
                warnings.append('Leaning out of frame')
            if y < h * LEAN_EDGE_THRESH or y > h * (1 - LEAN_EDGE_THRESH):
                warnings.append('Leaning out of frame')
    # Reaching/hand gesture (remove this warning)
    for wrist, shoulder, hip in [
        (LEFT_WRIST, LEFT_SHOULDER, LEFT_HIP),
        (RIGHT_WRIST, RIGHT_SHOULDER, RIGHT_HIP)
    ]:
        if is_valid(kp[wrist]) and is_valid(kp[shoulder]) and is_valid(kp[hip]):
            wx, wy = kp[wrist][0] * w, kp[wrist][1] * h
            sx, sy = kp[shoulder][0] * w, kp[shoulder][1] * h
            hx, hy = kp[hip][0] * w, kp[hip][1] * h
            torso_len = np.linalg.norm([sx - hx, sy - hy])
            # reach_dist = np.linalg.norm([wx - sx, wy - sy])
            # (Removed: if reach_dist > REACH_DIST_THRESH * torso_len: warnings.append('Reaching/hand gesture detected'))
            # Hand out of frame
            if wx < 0 or wx > w or wy < 0 or wy > h:
                warnings.append('Hand out of frame')
    # Typing (both wrists low and close) (remove this warning)
    # if all(is_valid(kp[idx]) for idx in [LEFT_WRIST, RIGHT_WRIST, LEFT_HIP, RIGHT_HIP]):
    #     lwx, lwy = kp[LEFT_WRIST][0] * w, kp[LEFT_WRIST][1] * h
    #     rwx, rwy = kp[RIGHT_WRIST][0] * w, kp[RIGHT_WRIST][1] * h
    #     lhy = kp[LEFT_HIP][1] * h
    #     rhy = kp[RIGHT_HIP][1] * h
    #     if lwy > h * (1 - TYPING_Y_THRESH) and rwy > h * (1 - TYPING_Y_THRESH):
    #         if abs(lwy - rwy) < 0.1 * h:
    #             warnings.append('Possible typing or device interaction')
    # Talking to someone off-camera (head turned)
    if is_valid(kp[NOSE]) and is_valid(kp[LEFT_SHOULDER]) and is_valid(kp[RIGHT_SHOULDER]):
        nx = kp[NOSE][0] * w
        lsx = kp[LEFT_SHOULDER][0] * w
        rsx = kp[RIGHT_SHOULDER][0] * w
        center = (lsx + rsx) / 2
        if abs(nx - center) > 0.18 * w:
            warnings.append('Possible talking to someone off-camera')
    return list(set(warnings))

def draw_pose(frame, keypoints):
    h, w = frame.shape[:2]
    # Draw keypoints
    for x, y, c in keypoints:
        if c > 0.3:
            cv2.circle(frame, (int(x * w), int(y * h)), 4, (0,255,0), -1)
    # Draw skeleton (shoulders, arms, hips)
    skeleton = [
        (LEFT_SHOULDER, RIGHT_SHOULDER),
        (LEFT_SHOULDER, LEFT_ELBOW), (LEFT_ELBOW, LEFT_WRIST),
        (RIGHT_SHOULDER, RIGHT_ELBOW), (RIGHT_ELBOW, RIGHT_WRIST),
        (LEFT_SHOULDER, LEFT_HIP), (RIGHT_SHOULDER, RIGHT_HIP),
        (LEFT_HIP, RIGHT_HIP)
    ]
    for i, j in skeleton:
        if is_valid(keypoints[i]) and is_valid(keypoints[j]):
            pt1 = (int(keypoints[i][0] * w), int(keypoints[i][1] * h))
            pt2 = (int(keypoints[j][0] * w), int(keypoints[j][1] * h))
            cv2.line(frame, pt1, pt2, (255,0,0), 2)

# Webcam loop
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print('Error: Could not open webcam.')
    exit()
print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    results = model(frame)
    for pose in results:
        for kp in pose.keypoints.xy:
            keypoints = kp.cpu().numpy()
            # Each keypoint: [x, y, (confidence)]
            # Normalize to [0,1] for easier processing
            keypoints_norm = np.array([
                [kp[0]/frame.shape[1], kp[1]/frame.shape[0], kp[2] if len(kp) > 2 else 1.0]
                for kp in keypoints
            ])
            warnings = suspicious_activity(keypoints_norm, frame.shape)
            draw_pose(frame, keypoints_norm)
            y0 = 30
            for w in warnings:
                cv2.putText(frame, w, (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                y0 += 30
    cv2.imshow('YOLOv8 Pose Activity & Gesture Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

# Instructions:
# 1. Install requirements: pip install -r requirements.txt
# 2. Download YOLOv8-pose model if not auto-downloaded: https://github.com/ultralytics/ultralytics
# 3. Run: python activity_gesture_detection.py
# 4. Watch for warnings on the video feed. 