# 🎯 ECA Presenter

> **On-Device AI Remote for Slide Control Using Hand Gestures**

**ECA Presenter** is a lightweight on-device AI remote that lets you control your presentation slides using **webcam-based hand gesture recognition** — no Bluetooth, smartphone, or network required.

Built on **ECA-Net (Efficient Channel Attention)**, it performs **real-time gesture inference on CPU-only environments** at up to **30 FPS**.

---

## ✋ Supported Gestures

| Gesture | Action | Model Label | Description |
| -------- | ------- | ------------ | ------------ |
| ✋ Palm | Next slide | `fist` | Palm and fist are unified under the same label (`fist`) and mapped to “Next Slide.” |
| 👌 OK Sign | Previous slide | `ok` | Thumb and index finger form a circle. |
| 👉 Index Up | Activate laser pointer | `index_up` | Triggers the pointer shortcut (e.g., `Ctrl + L` in PowerPoint). |
| ✌ V Sign | End presentation | `v_sign` | Ends the presentation and disables the pointer. |

> Compatible with **PowerPoint**, **Keynote**, and **Google Slides**.

---

## ECAGestureNet Architecture

```
──────────────────────────────────────────────────────────────
Input: 3 × 224 × 224 RGB
──────────────────────────────────────────────────────────────
Stage 1: Conv(3→32, k3, s2, p1) → BN → ReLU → ECA(32)
Output: 32 × 112 × 112
──────────────────────────────────────────────────────────────
Stage 2: Conv(32→64) → BN → ReLU → ECA(64)
Output: 64 × 56 × 56
──────────────────────────────────────────────────────────────
Stage 3: Conv(64→128) → BN → ReLU → ECA(128)
Output: 128 × 28 × 28
──────────────────────────────────────────────────────────────
Stage 4: Conv(128→256) → BN → ReLU → ECA(256)
Output: 256 × 14 × 14
──────────────────────────────────────────────────────────────
Global AvgPool → FC(256 → num_classes)
Output: logits (4 classes)
──────────────────────────────────────────────────────────────
```

### Highlights

- Each stage uses **Conv-BN-ReLU + ECA Block**
- ECA (Efficient Channel Attention) applies **1D convolution-based channel attention**
- Lightweight alternative to SE/CBAM with minimal overhead
- Global Average Pooling + FC for classification (`ok`, `fist`, `index_up`, `v_sign`)

> **Summary:** “Four Conv-ECA stages + Global Pool + FC” = compact yet powerful gesture recognition CNN.

---

## How to Use

### 1) Clone the repository

```bash
git clone https://github.com/USER/eca_presenter.git
cd eca_presenter
```

### 2) Create a virtual environment

```bash
python -m venv .venv
# Windows
.\.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

### 3) Install PyTorch

Use the official PyTorch installer for your system:  
🔗 [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)

Example (CPU only):

```bash
pip install torch torchvision
```

### 4) Install remaining dependencies

```bash
pip install -r requirements.txt
```

---

## 2. Run the Application

```bash
python runtime/main.py
```

**Runtime behavior:**
- Displays the recognized gesture and confidence score.
- Sends keyboard events directly to the active presentation window.
- Works fully offline using ONNX Runtime.

**Included models:**
```
models/gesture_eca.onnx
assets/labels.txt
```

---

## 3. Training & Model Conversion

### Dataset structure

```
data/
  train/
    ok/
    fist/
    index_up/
    v_sign/
  val/
    ok/
    fist/
    index_up/
    v_sign/
```

### Train the model

```bash
python model/train_eca_gesture.py
```

Output:
```
model/eca_gesture.pth
assets/labels.txt
```

### Convert to ONNX

```bash
python model/export_onnx.py
```

Output:
```
models/gesture_eca.onnx
```

---

## 4. Project Structure

```text
eca_presenter/
├── model/
│   ├── train_eca_gesture.py        # Training script
│   └── export_onnx.py              # ONNX exporter
├── runtime/
│   └── main.py                     # Webcam runtime + slide control
├── models/
│   └── gesture_eca.onnx            # Trained ONNX model
├── assets/
│   └── labels.txt                  # Class labels
├── requirements.txt
└── README.md
```

---

## 5. Design Motivation

### ① Inefficiency of smartphone remotes  
Presenters often can’t use both hands freely during talks.  
Using a smartphone to swipe slides interrupts the flow.

### ② Limitations of Bluetooth clickers  
- Battery drain or pairing failure  
- Compatibility issues  
- Easy to lose  
- May disconnect unexpectedly  

### ③ On-device AI advantages  
- No internet required  
- No data sent externally (privacy-safe)  
- Runs in real time on CPU using ONNX Runtime  
- Minimal latency and stable slide control  

---

## 6. System Pipeline

1. **MediaPipe Hands** detects the hand region.  
2. Crop and resize ROI to 224×224.  
3. **ONNX Runtime** performs gesture inference via ECAGestureNet.  
4. Apply **stability filtering** (confidence & consistent frames).  
5. Send key events using **pyautogui/keyboard** to control slides.

> Achieves ~30 FPS on CPU with < 50 ms end-to-end latency.

---

## 7. Research Contribution

| Goal | Description |
| ---- | ------------ |
| **ECA validation in real HCI** | Demonstrates ECA’s effectiveness in real-time, on-device gesture recognition. |
| **Lightweight attention** | Achieves similar accuracy to SE/CBAM with fewer FLOPs. |
| **Realtime performance** | Runs on CPU with no perceptible delay. |
| **Applied prototype** | Integrates ECA-Net into a functional presentation-control application. |

> This project bridges **academic model design** and **practical on-device AI applications** in HCI.

---

## 8. Development Environment

- Python 3.10  
- PyTorch / ONNX / ONNX Runtime  
- OpenCV  
- MediaPipe (optional)  
- keyboard / pyautogui  

---

## 9. Example Use Cases

- Gesture-controlled **slide navigation** during live talks  
- **Online teaching** with natural pointer control  
- **Interactive media art** installations  
- **Conference rooms** without physical remotes  

---

## Reference

> Wang Q., Wu B., Zhu P., Li P., Zuo W., Hu Q.  
> **ECA-Net: Efficient Channel Attention for Deep Convolutional Neural Networks.**  
> *Proceedings of CVPR 2020.*
