import cv2
import time
import pathlib
import mediapipe as mp

# ===========================================
# 저장할 루트 폴더
# ===========================================
DATA_ROOT = pathlib.Path("data")  # 원하는 경로
DATA_ROOT.mkdir(parents=True, exist_ok=True)

# 모델 라벨과 동일하게!
CLASS_KEYS = {
    "1": "fist",       # 손바닥(라벨 fist)
    "2": "ok",         # OK 사인
    "3": "index_up",   # 검지 위로
    "4": "v_sign",     # V 사인
}

# 각 클래스 폴더 생성
for cls in CLASS_KEYS.values():
    (DATA_ROOT / cls).mkdir(parents=True, exist_ok=True)

# ===========================================
# MediaPipe Hands
# ===========================================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

# ===========================================
# ROI 크롭 (main.py와 동일하게 맞추는 게 중요)
# ===========================================
def crop_square_with_pad(img, x1, y1, x2, y2, pad=24):
    h, w = img.shape[:2]
    x1 = max(0, x1 - pad); y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad); y2 = min(h, y2 + pad)
    bw, bh = (x2 - x1), (y2 - y1)
    side = max(bw, bh)
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    sx1 = max(0, cx - side // 2)
    sy1 = max(0, cy - side // 2)
    sx2 = min(w, sx1 + side)
    sy2 = min(h, sy1 + side)
    return img[sy1:sy2, sx1:sx2]

# ===========================================
# 카메라 열기 (맥북 – main.py랑 같은 index 사용)
# ===========================================
CAMERA_INDEX = 1  # main.py에서 잘 되던 값 사용
cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_AVFOUNDATION)
if not cap.isOpened():
    raise RuntimeError("카메라를 열 수 없습니다.")

print("[INFO] 제스처 연속 캡처 시작")
print("1:fist  2:ok  3:index_up  4:v_sign  0:stop  q:quit")

# 현재 어떤 클래스를 연속 캡처 중인지
current_cls = None
last_save_time = 0.0
SAVE_INTERVAL = 0.2  # 몇 초마다 한 장씩 저장할지 (0.2초면 초당 5장 정도)

# 클래스별 현재까지 저장 개수
img_counts = {
    cls: len(list((DATA_ROOT / cls).glob("*.jpg")))
    for cls in CLASS_KEYS.values()
}

while True:
    ok, frame = cap.read()
    if not ok:
        print("[WARN] 프레임 읽기 실패")
        break

    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = hands.process(rgb)

    roi = None

    if res.multi_hand_landmarks:
        lm = res.multi_hand_landmarks[0]
        xs = [int(pt.x * w) for pt in lm.landmark]
        ys = [int(pt.y * h) for pt in lm.landmark]
        x1, x2 = min(xs), max(xs)
        y1, y2 = min(ys), max(ys)

        roi = crop_square_with_pad(frame, x1, y1, x2, y2, pad=24)
        cv2.rectangle(frame, (x1-24, y1-24), (x2+24, y2+24), (0, 255, 0), 2)
    else:
        # 손이 안 잡히면 그냥 중앙 크롭
        side = min(h, w)
        sx1 = (w - side) // 2
        sy1 = (h - side) // 2
        roi = frame[sy1:sy1+side, sx1:sx1+side]

    # ROI 디버그
    roi_show = cv2.resize(roi, (224, 224))
    cv2.imshow("ROI", roi_show)

    # 안내 텍스트
    txt1 = "1:fist  2:ok  3:index_up  4:v_sign  0:stop  q:quit"
    cv2.putText(frame, txt1, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 255, 255), 2)

    if current_cls is not None:
        txt2 = f"[CAPTURING] {current_cls}  count={img_counts[current_cls]}"
        cv2.putText(frame, txt2, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 0), 2)
    else:
        txt2 = "[IDLE] press 1-4 to start capturing"
        cv2.putText(frame, txt2, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (200, 200, 200), 2)

    cv2.imshow("Capture Gestures (Continuous)", frame)

    # 키 입력 처리
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('0'):
        # 캡처 중단
        if current_cls is not None:
            print(f"[INFO] stop capturing {current_cls}")
        current_cls = None
    elif chr(key) in CLASS_KEYS:
        # 해당 클래스 연속 캡처 시작 (또는 전환)
        cls = CLASS_KEYS[chr(key)]
        current_cls = cls
        print(f"[INFO] start capturing {current_cls}")

    # 연속 저장 로직
    now = time.time()
    if current_cls is not None and (now - last_save_time) >= SAVE_INTERVAL:
        img_counts[current_cls] += 1
        fname = f"{current_cls}_{int(now*1000)}_{img_counts[current_cls]:04d}.jpg"
        out_path = DATA_ROOT / current_cls / fname
        cv2.imwrite(str(out_path), roi)
        last_save_time = now
        print(f"[SAVE] {current_cls}: {out_path}")

cap.release()
cv2.destroyAllWindows()
print("[INFO] 종료")
