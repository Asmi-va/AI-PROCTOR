from fastapi import FastAPI, UploadFile, File, Form, Request
from pydantic import BaseModel
import requests
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
from fastapi.responses import JSONResponse
from ai_audio_detection import detect_ai_audio
import tempfile
from fastapi.responses import StreamingResponse
from io import BytesIO

# In-memory log store
EVENT_LOGS = []

# In-memory storage for latest candidate frame
LATEST_FRAME = None

# In-memory storage for interviewer frame
INTERVIEWER_FRAME = None

# Candidate session management
CANDIDATE_SESSIONS = {}

# Interviewer session management
INTERVIEWER_SESSION = {
    "is_online": False,
    "last_seen": None,
    "current_frame": None
}

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- AI Text Detection ---
API_URL = "https://api-inference.huggingface.co/models/roberta-base-openai-detector"
HF_TOKEN = "YOUR_HF_TOKEN"  # Do not commit your real token!
headers = {"Authorization": f"Bearer {HF_TOKEN}"}

class TextRequest(BaseModel):
    text: str

@app.post("/detect-ai-text")
def detect_ai_text(req: TextRequest):
    response = requests.post(API_URL, json={"inputs": req.text}, headers=headers)
    try:
        result = response.json()
    except Exception as e:
        result = {"error": str(e), "raw_response": response.text}
    EVENT_LOGS.append({"type": "ai-text-detection", "result": result, "timestamp": datetime.utcnow().isoformat()})
    return result

# --- Face/Liveness Detection ---
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_face_liveness(image_np):
    gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return {"faces_detected": len(faces)}

@app.post("/detect-face")
def detect_face(file: UploadFile = File(...)):
    contents = file.file.read()
    npimg = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    result = detect_face_liveness(img)
    EVENT_LOGS.append({"type": "face-detection", "result": result, "timestamp": datetime.utcnow().isoformat()})
    return result

# --- Object Detection (YOLOv5) ---
yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
prohibited_items = {'cell phone', 'laptop', 'book', 'tv', 'remote'}

def detect_objects(image_np):
    results = yolo_model(image_np[..., ::-1])
    detected = []
    for *box, conf, cls in results.xyxy[0].tolist():
        label = results.names[int(cls)]
        if label in prohibited_items:
            detected.append(label)
    return {"prohibited_items": detected}

@app.post("/detect-object")
def detect_object(file: UploadFile = File(...)):
    contents = file.file.read()
    npimg = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    result = detect_objects(img)
    EVENT_LOGS.append({"type": "object-detection", "result": result, "timestamp": datetime.utcnow().isoformat()})
    return result

# --- Activity/Gesture Detection (YOLOv8-pose) ---
yolo_pose_model = YOLO('yolov8n-pose.pt')
NOSE = 0
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6

is_valid = lambda kp: kp[2] > 0.3

def detect_activity(image_np):
    results = yolo_pose_model(image_np)
    suspicious = []
    for pose in results:
        for kp in pose.keypoints.xy:
            keypoints = kp.cpu().numpy()
            keypoints_norm = np.array([
                [kp[0]/image_np.shape[1], kp[1]/image_np.shape[0], kp[2] if len(kp) > 2 else 1.0]
                for kp in keypoints
            ])
            h, w = image_np.shape[:2]
            # Leaning out of frame
            for idx in [NOSE, LEFT_SHOULDER, RIGHT_SHOULDER]:
                if is_valid(keypoints_norm[idx]):
                    x, y = keypoints_norm[idx][0] * w, keypoints_norm[idx][1] * h
                    if x < w * 0.12 or x > w * (1 - 0.12):
                        suspicious.append('Leaning out of frame')
                    if y < h * 0.12 or y > h * (1 - 0.12):
                        suspicious.append('Leaning out of frame')
            # Hand out of frame
            for wrist in [9, 10]:
                if is_valid(keypoints_norm[wrist]):
                    wx, wy = keypoints_norm[wrist][0] * w, keypoints_norm[wrist][1] * h
                    if wx < 0 or wx > w or wy < 0 or wy > h:
                        suspicious.append('Hand out of frame')
            # Talking to someone off-camera
            if is_valid(keypoints_norm[NOSE]) and is_valid(keypoints_norm[LEFT_SHOULDER]) and is_valid(keypoints_norm[RIGHT_SHOULDER]):
                nx = keypoints_norm[NOSE][0] * w
                lsx = keypoints_norm[LEFT_SHOULDER][0] * w
                rsx = keypoints_norm[RIGHT_SHOULDER][0] * w
                center = (lsx + rsx) / 2
                if abs(nx - center) > 0.18 * w:
                    suspicious.append('Possible talking to someone off-camera')
    return {"suspicious_activity": list(set(suspicious))}

@app.post("/detect-activity")
def detect_activity_endpoint(file: UploadFile = File(...)):
    contents = file.file.read()
    npimg = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    result = detect_activity(img)
    EVENT_LOGS.append({"type": "activity-detection", "result": result, "timestamp": datetime.utcnow().isoformat()})
    return result

@app.post("/detect-ai-audio")
def detect_ai_audio_endpoint(file: UploadFile = File(...)):
    # Save uploaded file to temp location
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        tmp.write(file.file.read())
        tmp_path = tmp.name
    result = detect_ai_audio(tmp_path)
    EVENT_LOGS.append({"type": "ai-audio-detection", "result": result, "timestamp": datetime.utcnow().isoformat()})
    return {"result": result}

@app.post("/upload-frame")
def upload_frame(file: UploadFile = File(...)):
    global LATEST_FRAME
    LATEST_FRAME = file.file.read()
    return {"status": "received"}

@app.get("/latest-frame")
def latest_frame():
    if LATEST_FRAME is None:
        return JSONResponse(content={"error": "No frame available"}, status_code=404)
    return StreamingResponse(BytesIO(LATEST_FRAME), media_type="image/jpeg")

@app.post("/tab-switch")
async def tab_switch(request: Request):
    data = await request.json()
    user_id = data.get("userId", "unknown")
    event = data.get("event", "unknown")
    timestamp = datetime.utcnow().isoformat()
    log_entry = {"type": "tab-switch", "userId": user_id, "event": event, "timestamp": timestamp}
    EVENT_LOGS.append(log_entry)
    print(f"Tab event: {event} by {user_id} at {timestamp}")
    return {"status": "received", "event": event, "userId": user_id, "timestamp": timestamp}

@app.get("/logs")
def get_logs():
    return JSONResponse(content=EVENT_LOGS)

# --- Candidate Browser Endpoints ---

class CandidateSession(BaseModel):
    userId: str
    status: str = "active"
    camera_enabled: bool = True
    microphone_enabled: bool = True

@app.post("/candidate/connect")
async def candidate_connect(session: CandidateSession):
    """Register a candidate session"""
    global CANDIDATE_SESSIONS
    CANDIDATE_SESSIONS[session.userId] = {
        "userId": session.userId,
        "status": session.status,
        "camera_enabled": session.camera_enabled,
        "microphone_enabled": session.microphone_enabled,
        "connected_at": datetime.utcnow().isoformat(),
        "last_seen": datetime.utcnow().isoformat()
    }
    EVENT_LOGS.append({
        "type": "candidate-connect", 
        "userId": session.userId, 
        "timestamp": datetime.utcnow().isoformat()
    })
    return {"status": "connected", "userId": session.userId}

@app.post("/candidate/disconnect")
async def candidate_disconnect(userId: str):
    """Disconnect a candidate session"""
    global CANDIDATE_SESSIONS
    if userId in CANDIDATE_SESSIONS:
        CANDIDATE_SESSIONS[userId]["status"] = "disconnected"
        CANDIDATE_SESSIONS[userId]["disconnected_at"] = datetime.utcnow().isoformat()
        EVENT_LOGS.append({
            "type": "candidate-disconnect", 
            "userId": userId, 
            "timestamp": datetime.utcnow().isoformat()
        })
    return {"status": "disconnected", "userId": userId}

@app.post("/candidate/update-status")
async def candidate_update_status(userId: str, camera_enabled: bool = True, microphone_enabled: bool = True):
    """Update candidate camera/microphone status"""
    global CANDIDATE_SESSIONS
    if userId in CANDIDATE_SESSIONS:
        if camera_enabled is not None:
            CANDIDATE_SESSIONS[userId]["camera_enabled"] = camera_enabled
        if microphone_enabled is not None:
            CANDIDATE_SESSIONS[userId]["microphone_enabled"] = microphone_enabled
        CANDIDATE_SESSIONS[userId]["last_seen"] = datetime.utcnow().isoformat()
        EVENT_LOGS.append({
            "type": "candidate-status-update", 
            "userId": userId,
            "camera_enabled": camera_enabled,
            "microphone_enabled": microphone_enabled,
            "timestamp": datetime.utcnow().isoformat()
        })
    return {"status": "updated", "userId": userId}

@app.get("/candidate/sessions")
def get_candidate_sessions():
    """Get all active candidate sessions"""
    return JSONResponse(content=CANDIDATE_SESSIONS)

@app.post("/interviewer/connect")
async def interviewer_connect():
    """Register interviewer as online"""
    global INTERVIEWER_SESSION
    INTERVIEWER_SESSION["is_online"] = True
    INTERVIEWER_SESSION["last_seen"] = datetime.utcnow().isoformat()
    EVENT_LOGS.append({
        "type": "interviewer-connect", 
        "timestamp": datetime.utcnow().isoformat()
    })
    return {"status": "connected"}

@app.post("/interviewer/disconnect")
async def interviewer_disconnect():
    """Register interviewer as offline"""
    global INTERVIEWER_SESSION
    INTERVIEWER_SESSION["is_online"] = False
    INTERVIEWER_SESSION["last_seen"] = datetime.utcnow().isoformat()
    EVENT_LOGS.append({
        "type": "interviewer-disconnect", 
        "timestamp": datetime.utcnow().isoformat()
    })
    return {"status": "disconnected"}

@app.post("/interviewer/upload-frame")
async def interviewer_upload_frame(file: UploadFile = File(...)):
    """Upload interviewer video frame for candidates to see"""
    global INTERVIEWER_FRAME, INTERVIEWER_SESSION
    INTERVIEWER_FRAME = file.file.read()
    INTERVIEWER_SESSION["current_frame"] = datetime.utcnow().isoformat()
    INTERVIEWER_SESSION["last_seen"] = datetime.utcnow().isoformat()
    return {"status": "received"}

@app.get("/interviewer/latest-frame")
def interviewer_latest_frame():
    """Get latest interviewer frame for candidates"""
    if INTERVIEWER_FRAME is None:
        return JSONResponse(content={"error": "No interviewer frame available"}, status_code=404)
    return StreamingResponse(BytesIO(INTERVIEWER_FRAME), media_type="image/jpeg")

@app.get("/interviewer/status")
def interviewer_status():
    """Get interviewer online status"""
    return JSONResponse(content=INTERVIEWER_SESSION)

@app.post("/candidate/heartbeat")
async def candidate_heartbeat(userId: str):
    """Update candidate last seen timestamp"""
    global CANDIDATE_SESSIONS
    if userId in CANDIDATE_SESSIONS:
        CANDIDATE_SESSIONS[userId]["last_seen"] = datetime.utcnow().isoformat()
    return {"status": "heartbeat_received"}

@app.get("/candidate/browser-status")
def candidate_browser_status(userId: str):
    """Get comprehensive status for candidate browser"""
    global CANDIDATE_SESSIONS, INTERVIEWER_SESSION
    
    candidate_info = CANDIDATE_SESSIONS.get(userId, {})
    interviewer_info = INTERVIEWER_SESSION
    
    return JSONResponse(content={
        "candidate": candidate_info,
        "interviewer": interviewer_info,
        "timestamp": datetime.utcnow().isoformat()
    })

# Instructions:
# 1. pip install fastapi uvicorn python-multipart requests torch torchvision ultralytics opencv-python
# 2. Replace YOUR_HF_TOKEN with your Hugging Face token for /detect-ai-text
# 3. Run: uvicorn main_api:app --reload
# 4. Visit http://127.0.0.1:8000/docs for interactive API documentation. 