# 🔍 Real-Time Face Recognition System

This project is a **threaded real-time face recognition pipeline** built with Python, OpenCV, and PyTorch. It captures live camera feed, detects faces, extracts embeddings, and matches them against a pre-defined gallery using multiple concurrent threads for efficiency.

---

## 📦 Features

- Real-time face detection and recognition  
- Multi-threaded pipeline architecture  
- Modular structure: camera input, detection, recognition, and gallery management  
- Built on PyTorch and OpenCV  

---

## 🗂️ Project Structure

```
project/
│
├── camera/
│   └── camera_stream.py        # Camera input handler
├── detectors/
│   └── detection.py            # Face detection logic
├── recognizers/
│   └── recognition.py          # Face embedding and recognition
├── gallery/
│   └── gallery_db.py           # Gallery database for identity matching
├── main.py                     # Entry point with multi-threaded pipeline
└── README.md                   # Project documentation
```

---

## 🚀 Getting Started

### 1. Clone the repository

    git clone https://github.com/yourusername/face-recognition-pipeline.git
    cd face-recognition-pipeline

### 2. Install dependencies

    pip install -r requirements.txt

### 3. Run the system

    python main.py

---

## ⚙️ Components

- **CameraStream** – Captures frames from a video stream (e.g., webcam).
- **Detector** – Detects face bounding boxes from each frame.
- **Recognizer** – Converts face crops into vector embeddings using a neural network.
- **Gallery** – Stores and compares face embeddings for identity matching.

---

## 📌 Notes

- Ensure your camera is connected and accessible via OpenCV.
- PyTorch-compatible GPU is recommended for faster inference.
- Modify the `camera_stream.py`, `detection.py`, and `recognition.py` files to plug in your custom models.

---

## 🧠 TODOs

- [ ] Add face enrollment API for dynamic gallery updates
- [ ] Add GUI or web dashboard
- [ ] Save recognition logs or results
- [ ] Add support for video files or RTSP streams


