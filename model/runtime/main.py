import sys
import time
import platform
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

# ==========================
# 0. 플랫폼 분기 설정
# ==========================

SYSTEM = platform.system()
IS_MAC = SYSTEM == "Darwin"
IS_WIN = SYSTEM == "Windows"
CAM_INDEX = 0

if IS_MAC:
    import pyautogui
    print("[INFO] macOS 감지: pyautogui로 키 입력을 전송합니다.")
elif IS_WIN:
    import keyboard
    print("[INFO] Windows 감지: keyboard 라이브러리로 키 입력을 전송합니다.")
else:
    print(f"[WARN] 지원되지 않는 OS: {SYSTEM}. 키 입력은 콘솔 로그로만 표시됩니다.")


# ==========================
# 1. 경로 설정
# ==========================

# 이 파일: model/runtime/main.py 기준
RUNTIME_DIR = Path(__file__).resolve().parent
ROOT_DIR = RUNTIME_DIR.parent.parent  # eca_presenter/

MODEL_PATH = ROOT_DIR / "models" / "gesture_eca.onnx"
LABELS_PATH = ROOT_DIR / "assets" / "labels.txt"

if not MODEL_PATH.exists():
    print(f"[ERROR] ONNX 모델이 없습니다: {MODEL_PATH}")
    sys.exit(1)

if not LABELS_PATH.exists():
    print(f"[ERROR] 라벨 파일이 없습니다: {LABELS_PATH}")
    sys.exit(1)


# ==========================
# 2. 라벨 로드
# ==========================

def load_labels(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        labels = [line.strip() for line in f.readlines() if line.strip()]
    return labels


LABELS = load_labels(LABELS_PATH)
print("[INFO] 클래스 라벨:", LABELS)


# ==========================
# 3. ONNX Runtime 세션 생성
# ==========================

def create_session(model_path: Path):
    providers = ["CPUExecutionProvider"]
    sess = ort.InferenceSession(str(model_path), providers=providers)
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    print("[INFO] ONNX 입력 이름:", input_name)
    print("[INFO] ONNX 출력 이름:", output_name)
    return sess, input_name, output_name


session, input_name, output_name = create_session(MODEL_PATH)


# ==========================
# 4. 키 입력 함수 (OS별 분기)
# ==========================

def send_action(action: str):
    """
    action: "next", "prev", "laser_on", "laser_off"
    """
    print(f"[ACTION] {action}")

    if IS_MAC:
        # macOS: pyautogui 사용 (손쉬운 사용 > 키보드/입력 권한 필요)
        if action == "next":
            pyautogui.press("right")
        elif action == "prev":
            pyautogui.press("left")
        elif action == "laser_on":
            pyautogui.hotkey("ctrl", "l")
        elif action == "laser_off":
            pyautogui.hotkey("ctrl", "l")  # 또는 pyautogui.press("esc")

    elif IS_WIN:
        # Windows: keyboard 사용
        if action == "next":
            keyboard.press_and_release("right")
        elif action == "prev":
            keyboard.press_and_release("left")
        elif action == "laser_on":
            keyboard.press_and_release("ctrl+l")
        elif action == "laser_off":
            keyboard.press_and_release("ctrl+l")  # 또는 keyboard.press_and_release("esc")

    else:
        # 기타 OS: 일단 콘솔에만 출력
        pass


# ==========================
# 5. 제스처 → 액션 매핑
# ==========================

GESTURE_TO_ACTION = {
    "ok": "next",
    "fist": "prev",
    "index_up": "laser_on",
    "v_sign": "laser_off",
}


# ==========================
# 6. 전처리 함수
# ==========================

def preprocess_frame(frame, img_size=224):
    """
    frame: BGR (OpenCV)
    return: (1, 3, H, W) float32 numpy array
    """
    if frame is None or frame.size == 0:
        return None

    h, w, _ = frame.shape

    # 정사각형 중심 크롭
    side = min(h, w)
    cy, cx = h // 2, w // 2
    y1 = max(0, cy - side // 2)
    y2 = y1 + side
    x1 = max(0, cx - side // 2)
    x2 = x1 + side

    crop = frame[y1:y2, x1:x2]

    if crop.size == 0:
        return None

    # 모델 입력용 크기 변경
    resized = cv2.resize(crop, (img_size, img_size))

    # BGR -> RGB
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    # [0,1] 스케일 후 Normalize (train과 동일)
    img = rgb.astype(np.float32) / 255.0
    img = (img - 0.5) / 0.5  # mean=0.5, std=0.5

    # HWC -> CHW
    img = np.transpose(img, (2, 0, 1))  # (C, H, W)

    # 배치 차원 추가
    img = np.expand_dims(img, axis=0)  # (1, 3, H, W)

    return img.astype(np.float32)


def softmax(x):
    x = np.array(x, dtype=np.float32)
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.sum(e)


# ==========================
# 7. 메인 루프
# ==========================

def main():
    # ---- 카메라 열기 ----
    if IS_MAC:
        cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_AVFOUNDATION)
    else:
        cap = cv2.VideoCapture(CAM_INDEX)

    if not cap.isOpened():
        print("[ERROR] 웹캠을 열 수 없습니다.")
        return

    print("[INFO] 웹캠 시작. 종료하려면 'q'를 누르세요.")

    last_action_time = 0.0
    action_cooldown = 1.5   # 한 번 제스처 실행 후 최소 1.5초 대기
    stable_required = 0.25  # 같은 제스처가 0.25초 이상 유지될 때만 실행

    last_pred = None
    last_action_label = None
    stable_start = None

    while True:
        ret, frame = cap.read()
        if not ret or frame is None or frame.size == 0:
            print("[WARN] 프레임을 읽어올 수 없습니다.")
            break

        # === 디버깅용: 프레임이 회색/고정인지 체크 ===
        if IS_MAC:
            # 프레임의 표준편차가 너무 작으면(거의 단색) 카메라 권한/Continuity 이슈일 수 있음
            std_val = frame.std()
            if std_val < 3:
                # 너무 많이 찍히면 시끄러우니까 가끔만 보고 싶으면 주석 처리 가능
                print(f"[WARN] 프레임 표준편차가 매우 낮음 (std={std_val:.2f}) - 거의 단색 화면일 수 있습니다.")

        # 모델 입력 전처리
        input_blob = preprocess_frame(frame, img_size=224)
        if input_blob is None:
            continue

        # ONNX 추론
        ort_inputs = {input_name: input_blob}
        ort_outs = session.run([output_name], ort_inputs)
        logits = ort_outs[0][0]              # (num_classes,)
        probs = softmax(logits)
        pred_idx = int(np.argmax(probs))
        pred_label = LABELS[pred_idx] if pred_idx < len(LABELS) else "unknown"
        pred_conf = float(probs[pred_idx])

        now = time.time()

        # === 제스처 안정화 로직 ===
        if pred_label == last_pred and pred_conf >= 0.7:
            if stable_start is None:
                stable_start = now
        else:
            stable_start = None

        last_pred = pred_label

        # === 화면에 내 모습 + 예측 결과 표시 (원본 프레임 사용) ===
        vis_frame = frame.copy()
        text = f"{pred_label} ({pred_conf:.2f})"
        cv2.putText(
            vis_frame,
            text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
        )
        cv2.imshow("ECA Presenter (Gesture View)", vis_frame)

        # === 실제 액션 실행 조건 ===
        CONF_THRESH = 0.7

        if (
            stable_start is not None
            and (now - stable_start) >= stable_required        # 일정 시간 유지
            and pred_conf >= CONF_THRESH                      # confidence 충분히 높음
            and (now - last_action_time) >= action_cooldown   # 쿨다운 지남
            and pred_label in GESTURE_TO_ACTION               # 정의된 제스처
            and pred_label != last_action_label               # 같은 제스처 연속으로 재발동 금지
        ):
            action = GESTURE_TO_ACTION[pred_label]
            send_action(action)
            last_action_time = now
            last_action_label = pred_label
            stable_start = None  # 다시 안정기간 측정

        # 종료 키
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()