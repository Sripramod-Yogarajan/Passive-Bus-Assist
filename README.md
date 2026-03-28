# 🚌 PassiveBusAssist — Low-Latency Passive Vision-Based Assistive Infrastructure for Accessible Public Transportation

> A fully passive, infrastructure-side assistive system that enables visually impaired and mobility-limited passengers to independently identify approaching buses — no personal device required.

---

## 📌 Overview

Urban bus networks communicate route and destination information almost exclusively through front-mounted visual display panels. This creates a significant accessibility barrier for the estimated **253 million people** worldwide living with moderate-to-severe visual impairment.

**PassiveBusAssist** is a continuous video-based announcement pipeline deployed directly at the bus stop. Unlike existing static-image or user-centric approaches, this system:

- Requires **no user device**, no active participation, and no internet connectivity
- Operates **persistently in real time** using a single fixed surveillance-grade camera
- Makes announcements **only when a passenger is present** and a bus is confirmed — eliminating noise pollution

---

## 🏗️ System Architecture

```
Fixed Camera
     │
     ▼
┌─────────────────────────────────────────────────────────┐
│              CONTINUOUS VIDEO PIPELINE (1 fps)          │
│                                                         │
│  ┌──────────────┐    ┌──────────────┐    ┌───────────┐ │
│  │  YOLOv8-m    │───▶│  ByteTrack   │───▶│    ROI    │ │
│  │  Detection   │    │  (bus_id)    │    │ Extraction│ │
│  └──────────────┘    └──────────────┘    └─────┬─────┘ │
│                                                 │       │
│  ┌──────────────┐    ┌──────────────┐    ┌─────▼─────┐ │
│  │   Context-   │◀───│   Sliding    │◀───│  EasyOCR  │ │
│  │  Aware Gate  │    │  Window Vote │    │  + Lev.   │ │
│  └──────┬───────┘    └──────────────┘    └───────────┘ │
│         │                                               │
│  ┌──────▼───────┐    ┌──────────────┐                  │
│  │ FIFO Announce│───▶│   TTS Audio  │                  │
│  │    Queue     │    │ Announcement │                  │
│  └──────────────┘    └──────────────┘                  │
└─────────────────────────────────────────────────────────┘
         │
         ▼
  Monocular Distance
     Estimation
```

---

## ✨ Key Innovations

### 1. Video Pipeline with Identity Tracking
The first deployment of a **continuous video pipeline** for transit accessibility. YOLOv8-m with integrated **ByteTrack** assigns a persistent `bus_id` to each detected bus across frames, enabling:
- Multi-bus management at a single stop
- Per-bus sliding windows and state tracking
- Duplicate announcement elimination

### 2. Lexicon-Guided Temporal Stabilisation
A two-stage pipeline that dramatically reduces announcement latency:

| Stage | Mechanism | Effect |
|-------|-----------|--------|
| Stage 1 | Levenshtein correction (τ = 0.5) against route lexicon | Maps OCR errors to canonical labels (e.g., `Scrtaford → Stratford`) |
| Stage 2 | Sliding-window majority voting (k = 7 frames) | Declares stable route label for announcement |

**Result: 40.7% reduction in announcement decision latency** (30.0 → 17.8 frames)

### 3. Context-Aware Announcement Gate
```
G = (c_bus ≥ δ) ∧ (r* ≠ ∅) ∧ (p = 1)
```
Announcements are triggered **only when all three conditions are simultaneously satisfied**:
- ✅ Bus detection confidence ≥ threshold
- ✅ Stable route label confirmed
- ✅ Passenger present at the stop

### 4. Monocular Distance Estimation
Real-time bus distance using a single fixed camera — no stereo rig, no LiDAR:
```
D = (W_known × f) / W_pixel
```
Provides spatial awareness of approaching buses unavailable in any prior assistive transit system.

---

## 📊 Performance Results

### Detection Performance

| Condition | Precision | Recall | mAP50 |
|-----------|-----------|--------|-------|
| Daytime | 0.9517 | 0.9747 | 0.9759 |
| Night-time | 0.9412 | 0.9634 | 0.9621 |
| High LED Glare | 0.9283 | 0.9501 | 0.9498 |
| **Overall** | **0.9517** | **0.9747** | **0.9759** |

### OCR & Stabilisation Performance

| Condition | OCR w/o Lev. | OCR w/ Lev. | Stab. Frames (raw) | Stab. Frames (corrected) | Latency Reduction |
|-----------|-------------|------------|-------------------|------------------------|-------------------|
| Day | 83.40% | 94.20% | 30 | 17 | 43.3% |
| Night | 76.50% | 91.60% | 32 | 18 | 43.8% |
| High LED Glare | 72.80% | 89.10% | 35 | 19 | 45.7% |
| **Overall** | **80.35%** | **92.00%** | **30.0** | **17.8** | **40.7%** |

### System-Level Summary

| Metric | Value |
|--------|-------|
| Detection mAP50 | 0.9759 |
| OCR Accuracy (raw) | 80.35% |
| OCR Accuracy (Levenshtein-corrected) | ~92% |
| Announcement Latency Reduction | 40.7% |
| False Announcement Rate | 16.67% |
| Estimated Bus Distance Range | 5.53 – 7.48 m |
| Peak CPU RAM | 1,289.5 MB |
| Peak GPU VRAM | 255.1 MB |

---

## 🖼️ Sample Scenarios

### Daytime Detection
Route OCR accuracy: **99.44%** | Destination accuracy after correction: **100%** (+25%) | Estimated distance: **7.48 m**

### Night-time Detection
Route OCR remains high (96–97%) due to short alphanumeric strings. Destination strings benefit substantially from Levenshtein correction (up to +11.83%). Distance estimates remain physically plausible.

### High LED Glare
Most challenging condition. Route OCR drops to **75.44%**, but Levenshtein correction recovers destination accuracy to **91.80%** (+7.90%). The stabilisation window achieves its **largest latency reduction (45.7%)** under this condition.

---

## 🛠️ Technical Stack

| Component | Technology |
|-----------|-----------|
| Object Detection | YOLOv8-m (Ultralytics) |
| Multi-Object Tracking | ByteTrack (integrated in YOLOv8) |
| OCR Engine | EasyOCR (CNN + RNN + Attention) |
| String Correction | RapidFuzz (Levenshtein similarity) |
| Video Acquisition | OpenCV |
| Annotation Tool | CVAT |
| Target Edge Platform | NVIDIA Jetson AGX Orin |

---

## 📋 Dataset

A custom dataset was constructed from publicly available footage covering three condition categories:

- **Daytime** — normal ambient light
- **Night-time** — artificial and low illumination
- **High LED Glare** — oversaturated display panels

| Split | Images |
|-------|--------|
| Train | 400 |
| Validation | 200 |
| Test | 200 |
| **Total** | **800** |

Three annotated object classes: `bus_front`, `route_number`, `destination`

---

## 🏋️ Training Configuration

| Parameter | Value |
|-----------|-------|
| Epochs | 70 |
| Early stopping patience | 30 |
| Input resolution | 960 × 960 |
| Batch size | 8 |
| Validation metric | mAP50 |
| Pretrained weights | Enabled |
| Mosaic augmentation | Disabled (last 10 epochs) |

---

## ⚡ Edge Deployability

| Platform | RAM | CPU Headroom | GPU Headroom |
|----------|-----|-------------|-------------|
| Jetson Nano (4GB) | 4 GB | ~2.7 GB | ~3.7 GB |
| Jetson Orin NX (8GB) | 8 GB | ~6.7 GB | ~7.7 GB |
| Jetson AGX Orin (32GB) | 32 GB | ~30.7 GB | ~31.7 GB |
| **This system** | **1.29 GB** | — | — |

GPU VRAM remains stable at ~255 MB across inference — dominated by YOLOv8-m weights (~6.4 MB) and EasyOCR's CRNN model (~20 MB).

### Optimisation Opportunities
- **TensorRT INT8 quantisation** — reduces model size from ~6.4 MB to ~1.5 MB; inference latency from ~5 ms to ~3 ms
- **Lightweight CRNN** — replace EasyOCR with a domain-fine-tuned model targeting ~5 MB
- **Adaptive sampling** — increase from 1 fps to 3–5 fps during confirmed bus-approach events

---

## 🚀 Future Enhancements

- **Live RTSP stream deployment** with stream-recovery mechanisms and adaptive temporal stabilisation
- **TensorRT quantisation & model pruning** for higher effective inference rates
- **Multilingual TTS output** to serve linguistically diverse urban populations
- **Transformer-based OCR** for scrolling LED displays, multi-line panels, and non-Latin scripts
- **Adaptive majority-vote window** — dynamically tuned by OCR confidence, detection stability, or bus distance
- **Distance-triggered early alerts** and multi-camera centralised monitoring for large-scale deployment

---



*This system advances assistive transit technology from reactive, user-initiated recognition toward **proactive, infrastructure-integrated accessibility**.*
