import time
import platform
from pathlib import Path
from collections import deque

import cv2
import numpy as np
import onnxruntime as ort
import mediapipe as mp

# 경로 설정
ROOT = Path(__file__).resolve().parent.parent  # eca_presenter/
MODEL_PATH = ROOT / "models" / "gesture_eca.onnx"
LABELS_PATH = ROOT / "assets" / "labels.txt"


# 키 입력 (Win / macOS)

SYSTEM = platform.system()
IS_WIN = SYSTEM == "Windows"
IS_MAC = SYSTEM == "Darwin"

if IS_WIN:
    import keyboard 
elif IS_MAC:
    import pyautogui

def send_key(key: str):
    """플랫폼에 맞게 키 입력 전송."""
    if IS_WIN:
        try:
            keyboard.send(key)
        except Exception as e:
            print(f"[WARN] keyboard.send 실패: {e}")
    elif IS_MAC:
        try:
            pyautogui.press(key)
        except Exception as e:
            print(f"[WARN] pyautogui.press 실패: {e}")
    else:
        print(f"[INFO] (시뮬) 키 입력: {key}")


# 설정값 (필요 시 여기만 조정)
INPUT_SIZE = (224, 224)
CONF_THRESH = 0.5           # 이 값 이상일 때만 유효 판정
STABLE_FRAMES = 3            # 동일 결과가 N프레임 연속 나와야 확정
COOLDOWN_SEC = 0.7            # 같은 키 연타 방지
MAX_NUM_HANDS = 1             # 한 손 기준
PAD_PX = 24                   # bbox 주변 여백
DRAW_VIS = True               # 시각화 박스/텍스트 그리기

# 카메라 인덱스 (None이면 실행 시 선택 모드)
CAMERA_INDEX = 1         # 예: 맥북 카메라가 1번이면 1로 고정해도 됨

# 라벨→키 매핑 (labels.txt 라벨과 이름을 맞춰주세요!)
LABEL2KEY = {
    # ✋ 손바닥 (라벨은 fist) → 다음 슬라이드
    "fist": "right",

    # 👌 ok 사인 → 이전 슬라이드
    "ok": "left",

    # 👉 검지 위로 → 레이저 포인터 토글 (켜기/끄기용)
    "index_up": "command+l",

    # ✌ V자 → 레이저 포인터 토글 (끄기/켜기 동일 키)
    "v_sign": "esc",
}

# 유틸
def load_labels(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        labs = [ln.strip() for ln in f if ln.strip()]
    return labs

def softmax(x: np.ndarray):
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)

def crop_square_with_pad(img, x1, y1, x2, y2, pad=0):
    h, w = img.shape[:2]
    x1 = max(0, x1 - pad); y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad); y2 = min(h, y2 + pad)
    # 정사각형 맞추기
    bw, bh = (x2 - x1), (y2 - y1)
    side = max(bw, bh)
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    sx1 = max(0, cx - side // 2)
    sy1 = max(0, cy - side // 2)
    sx2 = min(w, sx1 + side)
    sy2 = min(h, sy1 + side)
    return img[sy1:sy2, sx1:sx2], (sx1, sy1, sx2, sy2)

def open_camera(index: int):
    """플랫폼별로 카메라를 연다."""
    if IS_MAC:
        cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
    else:
        cap = cv2.VideoCapture(index)
    return cap

def select_camera(max_index: int = 4) -> cv2.VideoCapture:
    """여러 카메라 중에서 사용자가 선택하도록 함."""
    print("[INFO] 카메라 선택 모드: 맥북 카메라 화면에서 's'를 눌러 선택하세요.")
    chosen_cap = None
    for idx in range(max_index + 1):
        cap = open_camera(idx)
        if not cap.isOpened():
            cap.release()
            continue

        ok, frame = cap.read()
        if not ok:
            cap.release()
            continue

        txt = f"Camera {idx} - 's' 선택, 다른 키: 다음으로"
        cv2.putText(frame, txt, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow("Select Camera", frame)
        key = cv2.waitKey(0) & 0xFF
        cv2.destroyWindow("Select Camera")

        if key == ord('s'):
            print(f"[INFO] Camera {idx} 선택됨")
            chosen_cap = cap
            break
        else:
            cap.release()

    if chosen_cap is None:
        raise RuntimeError("사용 가능한 카메라를 선택하지 못했습니다.")
    return chosen_cap

# ONNX 로드
assert MODEL_PATH.exists(), f"모델이 없습니다: {MODEL_PATH}"
labels = load_labels(LABELS_PATH)
print(f"[INFO] labels: {labels}")

providers = (
    ["CUDAExecutionProvider", "CPUExecutionProvider"]
    if "CUDAExecutionProvider" in ort.get_available_providers()
    else ["CPUExecutionProvider"]
)
sess = ort.InferenceSession(str(MODEL_PATH), providers=providers)
in_name = sess.get_inputs()[0].name
out_name = sess.get_outputs()[0].name
print(f"[INFO] ONNX loaded with providers={providers}")
print(f"[INFO] inputs={in_name}, outputs={out_name}")

# MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=MAX_NUM_HANDS,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# 안정화/쿨다운 상태
recent = deque(maxlen=STABLE_FRAMES)
last_confirmed = None
last_sent_time = 0.0


# 카메라 열기
if CAMERA_INDEX is None:
    cap = select_camera(max_index=4)  # 필요하면 최대 인덱스 조정
else:
    cap = open_camera(CAMERA_INDEX)

if not cap.isOpened():
    raise RuntimeError("카메라를 열 수 없습니다.")

fps_t0 = time.time()
fps_cnt = 0
fps_val = 0.0

print("[INFO] 시작: 'q'로 종료")
while True:
    ok, frame = cap.read()
    if not ok:
        print("[WARN] 프레임 읽기 실패")
        break

    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = hands.process(rgb)

    roi = None
    roi_box = None
    # hand_present = False

    if res.multi_hand_landmarks:
        # 모든 랜드마크 좌표를 이용해 바운딩 박스 계산
        lm = res.multi_hand_landmarks[0]  # 첫 번째 손만
        xs = [int(pt.x * w) for pt in lm.landmark]
        ys = [int(pt.y * h) for pt in lm.landmark]
        x1, x2 = min(xs), max(xs)
        y1, y2 = min(ys), max(ys)

        # pad + 정사각 crop
        roi, roi_box = crop_square_with_pad(frame, x1, y1, x2, y2, PAD_PX)
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

    # ROI 없으면 전체 프레임에서 중앙 정사각 크롭 (fallback)
    if roi is None:
        side = min(h, w)
        sx1 = (w - side) // 2
        sy1 = (h - side) // 2
        roi = frame[sy1:sy1 + side, sx1:sx1 + side]
        roi_box = (sx1, sy1, sx1 + side, sy1 + side)
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

    # 전처리
    inp = cv2.resize(roi, INPUT_SIZE, interpolation=cv2.INTER_LINEAR)
    inp = inp.astype(np.float32) / 255.0          # [0, 1]
    inp = (inp - 0.5) / 0.5                       # [-1, 1]  == Normalize(0.5,0.5)
    inp = np.transpose(inp, (2, 0, 1))[None, ...] # (1, 3, H, W)


    # 추론
    probs = sess.run([out_name], {in_name: inp})[0].squeeze()  # (C,)
    if probs.ndim == 0:
        probs = np.array([1.0], dtype=np.float32)
    if probs.ndim == 1 and probs.shape[0] == len(labels):
        pred_prob = softmax(probs)
    else:
        # 이미 softmax 상태일 수도 있으니 normalize
        x = probs.astype(np.float32)
        pred_prob = x / max(1e-9, x.sum())

    pred_idx = int(np.argmax(pred_prob))
    pred_label = labels[pred_idx]
    pred_conf = float(pred_prob[pred_idx])

    # 안정화 버퍼 업데이트
    recent.append(pred_label)
    confirmed = None
    if len(recent) == STABLE_FRAMES and all(x == recent[0] for x in recent) and pred_conf >= CONF_THRESH:
        confirmed = recent[0]

    # 키 입력 (쿨다운)
    now = time.time()
    if (
        confirmed
        # and hand_present
        and confirmed != last_confirmed and (now - last_sent_time) >= COOLDOWN_SEC):
        key = LABEL2KEY.get(confirmed)
        if key:
            print(f"[ACT] {confirmed} ({pred_conf:.2f}) -> key='{key}'")
            send_key(key)
            last_sent_time = now
            last_confirmed = confirmed
        else:
            print(f"[INFO] 매핑 없음: '{confirmed}' (labels.txt와 LABEL2KEY 확인)")

    # 시각화
    if DRAW_VIS and roi_box is not None:
        x1, y1, x2, y2 = roi_box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        txt = f"{pred_label}:{pred_conf:.2f}"
        if confirmed:
            txt = f"[OK]{txt}"
        cv2.putText(frame, txt, (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 0) if confirmed else (0, 200, 255), 2)

    # FPS
    fps_cnt += 1
    if time.time() - fps_t0 >= 1.0:
        fps_val = fps_cnt / (time.time() - fps_t0)
        fps_cnt = 0
        fps_t0 = time.time()
    cv2.putText(frame, f"FPS: {fps_val:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow("ECA Gesture Presenter (Hand ROI)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("[INFO] 종료")
