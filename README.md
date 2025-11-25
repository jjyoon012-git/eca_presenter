# 🎯 ECA Presenter

**손 제스처로 슬라이드를 넘기는 온디바이스 프레젠테이션 리모컨**

웹캠으로 손 제스처를 인식해

* ➡️ **다음 슬라이드**
* ⬅️ **이전 슬라이드**
* 🔴 **레이저 포인터 켜기/끄기**

를 조작할 수 있는 초경량 온디바이스 AI 리모컨입니다.
모델은 **ECA-Net(Efficient Channel Attention)** 기반으로 매우 가볍고 빠릅니다.

---

## ✋ 지원 제스처

| 제스처      | 클래스        | 동작                           |
| -------- | ---------- | ---------------------------- |
| 👌 O     | `ok`       | 다음 슬라이드 (Right Arrow)        |
| ✊ 주먹     | `fist`     | 이전 슬라이드 (Left Arrow)         |
| 👉 검지 위로 | `index_up` | 레이저 포인터 켜기 (Ctrl + L)        |
| ✌ V      | `v_sign`   | 레이저 포인터 끄기 (Ctrl + L 또는 ESC) |

> 슬라이드 종류(PowerPoint, Google Slides, Keynote 모두 지원)

---

## 🚀 1. 설치 방법

### 1) 저장소 클론

```bash
git clone https://github.com/USER/eca_presenter.git
cd eca_presenter
```

### 2) 가상환경 생성 (선택)

```bash
python -m venv .venv

# Windows
.\.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate
```

### 3) PyTorch 설치

각자 환경에 맞는 명령을 PyTorch 공식사이트에서 복사해 설치하는 것을 권장합니다.

🔗 [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)

**예시 (CPU 전용):**

```bash
pip install torch torchvision
```

### 4) 나머지 패키지 설치

```bash
pip install -r requirements.txt
```

---

## 🎥 2. 실행 방법 (바로 사용)

이미 학습된 모델(`models/gesture_eca.onnx`)과
라벨(`assets/labels.txt`)이 포함되어 있으므로
**웹캠 있는 PC라면 바로 실행 가능!**

```bash
python runtime/main.py
```

실행되면:

* 상단에 인식된 제스처 + confidence 표시
* 슬라이드 창을 활성화해두면 자동으로 키 입력 전송

---

## 🧠 3. 모델 재학습 (원하면)

### 1) 데이터셋 구조

아래 폴더에 제스처 이미지를 넣습니다:

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

### 2) 학습 실행

```bash
python model/train_eca_gesture.py
```

학습 완료 후 다음이 생성됨:

* `model/eca_gesture.pth` (PyTorch weights)
* `assets/labels.txt`

### 3) ONNX로 변환

```bash
python model/export_onnx.py
```

변환된 ONNX 모델:

```
models/gesture_eca.onnx
```

이제 runtime에서 자동으로 사용됩니다.

---

## 🧩 4. 프로젝트 구조

```text
eca_presenter/
├── model/
│   ├── train_eca_gesture.py        # 학습 코드
│   └── export_onnx.py              # ONNX 변환기
├── runtime/
│   └── main.py                     # 웹캠 + 슬라이드 제어 실행코드
├── models/
│   └── gesture_eca.onnx            # 학습된 ONNX 모델
├── assets/
│   └── labels.txt                  # 클래스 라벨
├── requirements.txt
└── README.md
```

---

## 🔧 5. 개발 환경

* Python 3.10
* PyTorch (CPU 또는 GPU 선택)
* OpenCV
* ONNX / ONNX Runtime
* keyboard 라이브러리 (키 입력)

모두 `requirements.txt`에 포함되어 있습니다.

---

## 💡 6. 실제 사용 예시

* **발표 중 리모컨 없이 슬라이드 넘기기**
* **온라인 수업 중 손 제스처로 화면 제어**
* **스마트 미디어 아트 전시 제스처 인터랙션**
* **회의실 PC에서 손으로 슬라이드 조작**