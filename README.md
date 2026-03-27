# Smart Bus Vision

A Low-Latency Passive Vision-Based Assistive Infrastructure for Accessible Public Transportation

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![YOLO](https://img.shields.io/badge/YOLO-v8-orange.svg)](https://github.com/ultralytics/ultralytics)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📋 Overview

**Smart Bus Vision** is an intelligent, infrastructure-side assistive system designed for deployment at bus stops. It automatically detects arriving buses, identifies their route numbers and destinations using computer vision and OCR, and announces the information through real-time audio output. The system is designed to improve accessibility for visually impaired and elderly passengers without requiring smartphones, apps, or any user interaction.

### Key Features

- **🚌 Real-Time Bus Detection**: Uses YOLOv8 for accurate bus detection and tracking
- **🎯 ROI Extraction**: Precisely localizes front display panels to isolate route and destination information
- **📝 OCR Text Recognition**: Leverages EasyOCR for robust text extraction from LED displays
- **⏱️ Temporal Consistency**: Verifies text across multiple frames to reduce transient errors
- **✅ Text Validation**: Corrects OCR errors using Levenshtein distance matching against known routes
- **🔊 Audio Announcements**: Converts recognized text to speech using Text-to-Speech (TTS)
- **📊 Queue Management**: Prevents overlapping announcements when multiple buses arrive simultaneously
- **📝 Logging System**: Records all announcements with timestamps for monitoring and analysis

## 🎯 Problem Statement

Public bus transportation systems primarily convey route numbers and destination information through visual displays mounted on the front of buses. While this approach works for sighted passengers, it creates significant accessibility barriers for:

- Visually impaired individuals
- Elderly passengers with reduced visual acuity
- People with limited vision in challenging conditions (poor lighting, fast-moving vehicles, adverse weather)

Traditional solutions often require:
- Personal smart devices (smartphones, wearables)
- Active user interaction
- Stable internet connectivity
- Technical familiarity

**Smart Bus Vision** eliminates these dependencies by providing a passive, infrastructure-side solution that operates autonomously.

## 🏗️ System Architecture

The system follows a modular pipeline architecture:

```
Video Input → Bus Detection → ROI Extraction → Text Recognition → 
Temporal Verification → Text Validation → Queue Management → Audio Announcement
```

### Core Modules

1. **Video Input Module**: Acquires and preprocesses video frames from a fixed camera
2. **Bus Detection Module**: Uses YOLOv8 to detect and track buses in real-time
3. **Display Region Localization (ROI Extraction)**: Isolates front display panels containing route and destination information
4. **Text Extraction Module**: Applies EasyOCR with LED-specific preprocessing for text recognition
5. **Temporal Consistency Verification**: Validates text across multiple frames to ensure stability
6. **Text Validation and Correction**: Matches OCR output against known destinations using fuzzy string matching
7. **Queue Management Module**: Manages announcement order to prevent overlapping audio output
8. **Audio Announcement Module**: Converts validated text to speech using pyttsx3

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-enabled GPU (optional, recommended for faster inference)
- Webcam or video file for testing

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/Smart-Bus-Vision.git
cd Smart-Bus-Vision
```

### Step 2: Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Download Model Weights

Place your trained YOLOv8 model weights (`best.pt`) in the `models/` directory.

### Step 5: Configure Routes

Add valid bus destinations to `data/routes.txt` (one per line).

## 📖 Usage

### Running the Main Pipeline

The main system processes video streams and generates audio announcements:

```bash
cd scripts
python main.py
```

**Configuration** (in `scripts/main.py`):
- `MODEL_PATH`: Path to YOLOv8 model weights
- `VIDEO_SOURCE`: Video file path or camera index (0 for webcam)
- `CONF_THRESH`: Detection confidence threshold (default: 0.4)
- `TEMPORAL_WINDOW`: Number of frames for temporal verification (default: 7)
- `STABLE_THRESHOLD`: Minimum consistent detections for announcement (default: 4)

### Image Processing Pipeline

For processing single images:

```bash
cd scripts
python image_pipeline.py
```

### OCR Testing

Test OCR on individual images:

```bash
cd scripts
python ocr_pipeline.py
```

## 📁 Project Structure

```
Smart-Bus-Vision/
├── data/
│   ├── data.yaml          # YOLO dataset configuration
│   ├── routes.txt          # Valid bus destinations list
│   ├── train/              # Training images
│   ├── val/                # Validation images
│   └── videos/             # Test videos
├── models/
│   └── best.pt             # Trained YOLOv8 model weights
├── scripts/
│   ├── main.py             # Main video processing pipeline
│   ├── image_pipeline.py   # Single image processing
│   ├── detect_and_crop.py  # Bus detection and ROI extraction
│   └── ocr_pipeline.py      # OCR text extraction
├── notebooks/
│   └── smart_bus_vision.ipynb  # Jupyter notebook for experimentation
├── requirements.txt        # Python dependencies
├── bus_announcements.log   # Announcement logs
└── README.md              # This file
```

## 🔧 Technical Details

### Bus Detection

- **Model**: YOLOv8 (You Only Look Once version 8)
- **Classes Detected**: 
  - `bus_front`: Front view of buses
  - `route_number`: Route number display region
  - `destination`: Destination display region
- **Tracking**: Uses YOLO's built-in tracking for multi-bus scenarios

### Text Recognition

- **OCR Engine**: EasyOCR with English language support
- **Preprocessing**:
  - Grayscale conversion
  - CLAHE (Contrast Limited Adaptive Histogram Equalization)
  - Morphological operations
  - 2x upscaling for better recognition
- **Confidence Threshold**: 0.3 (configurable)

### Temporal Verification

- Maintains a sliding window of recent OCR results
- Uses majority voting to determine stable text
- Requires consistent detection across multiple frames before announcement

### Text Correction

- Uses RapidFuzz library for fuzzy string matching
- Levenshtein distance-based similarity scoring
- Matches against predefined list of valid destinations
- Threshold: 80% similarity (configurable)

### Audio Announcement

- **TTS Engine**: pyttsx3 (offline, no internet required)
- **Format**: "Attention please. Bus number {route} to {destination} has arrived."
- **Queue System**: FIFO queue prevents overlapping announcements
- **Logging**: All announcements logged with timestamps

## 📊 Performance Metrics

The system is evaluated based on:

- **Bus Detection Accuracy**: Precision and recall of bus detection
- **OCR Reliability**: Text recognition accuracy on display panels
- **End-to-End Latency**: Time from bus detection to audio announcement
- **False Announcement Reduction**: Effectiveness of temporal verification and validation

## 🔬 Experimental Evaluation

The system has been tested on:
- Publicly available bus image datasets
- Real-world bus stop scenarios
- Various lighting conditions (daylight, low-light, glare)
- Different bus display formats and fonts

## 🌟 Key Advantages

1. **Passive Operation**: No user interaction required
2. **No Device Dependency**: Works without smartphones or wearables
3. **Offline Capable**: No internet connectivity needed
4. **Context-Aware**: Can be extended with passenger presence detection
5. **Low Latency**: Real-time processing suitable for bus stop deployment
6. **Robust**: Handles motion blur, lighting variations, and display inconsistencies
7. **Scalable**: Infrastructure-side deployment benefits all passengers

## 🔮 Future Enhancements

- [ ] Passenger presence detection for context-aware announcements
- [ ] Multi-language support for destinations
- [ ] Integration with bus tracking APIs for enhanced accuracy
- [ ] Web dashboard for monitoring and analytics
- [ ] Edge device optimization for deployment
- [ ] Support for multiple camera angles

## 📝 License

This project is developed as part of academic coursework. Please refer to the project report for detailed methodology and references.

## 📚 References

1. Shafique, S. (2025). Enhancing Mobility for the Blind: An AI-Powered Bus Route Recognition System
2. Maina, H. J., & Sánchez, J. A. (2020). Stop the Bus: Computer vision for automatic recognition of urban bus lines
3. Wang, A., et al. (2024). YOLOv10: Real-Time End-to-End Object Detection. NeurIPS 2024
4. Wongta, P., et al. An automatic bus route number recognition
5. Jeeva, C. (2022). Intelligent Image Text Reader using Easy OCR, NRCLex & NLTK

## 🤝 Contributing

This is an academic project. For questions or suggestions, please contact us.

## 📧 Contact

For inquiries about this project, please reach out to the team members.

---

**Note**: This system is designed as a proof-of-concept for infrastructure-side assistive technology in public transportation. Deployment in real-world scenarios would require additional testing, optimization, and compliance with local regulations.
