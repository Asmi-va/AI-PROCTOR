# AI Interview Proctoring System

A comprehensive AI-powered interview proctoring system that provides real-time monitoring, candidate browser interface, and interviewer management capabilities.

## 🚀 Features

### Candidate Browser Interface
- **Dual Video Display**: Candidates can see both the interviewer and themselves simultaneously
- **Real-time Controls**: Camera and microphone toggle switches
- **Session Management**: Automatic connection tracking and heartbeat monitoring
- **Professional UI**: Modern glassmorphism design with responsive layout

### AI Proctoring Capabilities
- **Face Detection**: Real-time face and liveness detection
- **Object Detection**: Prohibited item detection using YOLOv5
- **Activity Detection**: Suspicious behavior detection using YOLOv8-pose
- **AI Audio Detection**: Detection of AI-generated audio
- **AI Text Detection**: Detection of AI-generated text responses
- **Tab Switch Detection**: Monitor when candidates switch browser tabs

### Interviewer Management
- **Live View**: Real-time monitoring of candidate activities
- **Video Upload**: Interviewers can share their video with candidates
- **Session Logs**: Comprehensive logging of all activities
- **Candidate Status**: Real-time status of all connected candidates

## 📁 Project Structure

```
AI-PROCTOR/
├── ai-backend/
│   └── model/
│       ├── main_api.py              # FastAPI backend with AI endpoints
│       ├── ai_audio_detection.py    # AI audio detection module
│       ├── face_detection_realtime.py # Face detection module
│       ├── activity_gesture_detection.py # Activity detection module
│       └── requirements.txt         # Python dependencies
├── backend/
│   ├── server.js                   # Node.js backend server
│   └── package.json               # Node.js dependencies
├── ai-interview-proctoring/
│   ├── index.html                 # Login page
│   ├── next.html                  # Welcome page with view options
│   ├── camera.html                # Standard candidate view
│   ├── candidate_browser.html     # 🆕 Candidate browser interface
│   ├── interviewer_live.html      # Interviewer live monitoring
│   ├── interviewer_upload.html    # 🆕 Interviewer video upload
│   └── interviewer.html           # Interviewer logs view
├── start_backend.py               # 🆕 Backend startup script
└── README.md                      # This file
```

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- Node.js 14+
- npm or yarn

### Quick Start

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd AI-PROCTOR
   ```

2. **Install Python dependencies**
   ```bash
   pip install -r ai-backend/model/requirements.txt
   ```

3. **Install Node.js dependencies**
   ```bash
   cd backend
   npm install
   cd ..
   ```

4. **Start the backends**
   ```bash
   python start_backend.py
   ```

## 🎯 Usage

### For Candidates

1. **Access the system**: Navigate to `http://localhost:3001/ai-interview-proctoring/`
2. **Login**: Click "Login and Allow Camera & Microphone"
3. **Choose view**: Select between "Standard View" or "Candidate Browser"
4. **Interview**: Use the candidate browser to see both interviewer and yourself

### For Interviewers

1. **Live Monitoring**: Access `http://localhost:3001/ai-interview-proctoring/interviewer_live.html`
2. **Upload Video**: Use `http://localhost:3001/ai-interview-proctoring/interviewer_upload.html`
3. **View Logs**: Check `http://localhost:3001/ai-interview-proctoring/interviewer.html`

## 🔧 API Endpoints

### Python FastAPI Backend (Port 8000)

#### AI Detection Endpoints
- `POST /detect-face` - Face and liveness detection
- `POST /detect-object` - Prohibited object detection
- `POST /detect-activity` - Suspicious activity detection
- `POST /detect-ai-audio` - AI audio detection
- `POST /detect-ai-text` - AI text detection

#### Candidate Browser Endpoints
- `POST /candidate/connect` - Register candidate session
- `POST /candidate/disconnect` - Disconnect candidate session
- `POST /candidate/update-status` - Update camera/microphone status
- `GET /candidate/sessions` - Get all active candidate sessions
- `POST /candidate/heartbeat` - Update candidate last seen
- `GET /candidate/browser-status` - Get comprehensive status

#### Interviewer Endpoints
- `POST /interviewer/connect` - Register interviewer as online
- `POST /interviewer/disconnect` - Register interviewer as offline
- `POST /interviewer/upload-frame` - Upload interviewer video frame
- `GET /interviewer/latest-frame` - Get latest interviewer frame
- `GET /interviewer/status` - Get interviewer online status

#### General Endpoints
- `POST /upload-frame` - Upload candidate frame for monitoring
- `GET /latest-frame` - Get latest candidate frame
- `POST /tab-switch` - Report tab switch events
- `GET /logs` - Get all system logs

### Node.js Backend (Port 3001)

- `POST /api/store-userid` - Store candidate user ID
- Static file serving for frontend

## 🎨 Candidate Browser Features

### Dual Video Interface
- **Interviewer Video**: Real-time feed from interviewer
- **Candidate Video**: Local camera feed with controls
- **Responsive Design**: Works on desktop and mobile

### Real-time Controls
- **Camera Toggle**: Enable/disable camera with backend sync
- **Microphone Toggle**: Enable/disable audio with backend sync
- **Session Timer**: Real-time interview duration
- **Connection Quality**: Automatic quality assessment

### Backend Integration
- **Session Management**: Automatic connection/disconnection
- **Heartbeat Monitoring**: Regular status updates
- **Status Synchronization**: Real-time status updates
- **Cleanup**: Automatic cleanup on page unload

## 🔍 AI Proctoring Features

### Face Detection
- Real-time face detection using OpenCV
- Liveness detection to prevent photo spoofing
- Multiple face detection for suspicious activity

### Object Detection
- YOLOv5-based object detection
- Prohibited items detection (phones, laptops, books)
- Real-time alerts for suspicious objects

### Activity Detection
- YOLOv8-pose for pose estimation
- Suspicious behavior detection:
  - Leaning out of frame
  - Hand out of frame
  - Talking to someone off-camera
- Real-time activity monitoring

### Audio Analysis
- AI-generated audio detection
- Real-time audio processing
- Suspicious audio pattern detection

### Text Analysis
- AI-generated text detection
- Real-time text analysis
- Suspicious response pattern detection

## 📊 Monitoring and Logging

### Real-time Logs
- All activities are logged with timestamps
- User ID tracking for each session
- Event categorization (face, object, activity, etc.)

### Interviewer Dashboard
- Live candidate video feed
- Real-time activity logs
- Connection status monitoring
- Session management

## 🔒 Security Features

### Tab Switch Detection
- Monitors when candidates switch browser tabs
- Logs all tab switch events
- Real-time alerts for suspicious behavior

### Session Management
- Automatic session tracking
- Connection status monitoring
- Cleanup on disconnection

### Privacy Protection
- Local video processing
- Secure data transmission
- Minimal data retention

## 🚀 Deployment

### Development
```bash
python start_backend.py
```

### Production
1. Set up reverse proxy (nginx)
2. Configure SSL certificates
3. Use process manager (PM2 for Node.js, systemd for Python)
4. Set up monitoring and logging

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For support and questions:
1. Check the API documentation at `http://localhost:8000/docs`
2. Review the logs for error messages
3. Ensure all dependencies are properly installed
4. Check that both backends are running

## 🔄 Recent Updates

### v2.0 - Candidate Browser Integration
- ✅ Added candidate browser interface with dual video display
- ✅ Integrated backend session management
- ✅ Added interviewer video upload interface
- ✅ Enhanced real-time controls and status monitoring
- ✅ Improved UI/UX with modern design
- ✅ Added comprehensive logging and monitoring

---

**Note**: This system is designed for educational and research purposes. Ensure compliance with local privacy laws and regulations when deploying in production environments. 