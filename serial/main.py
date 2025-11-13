import cv2
import torch
import numpy as np
from ultralytics import YOLO
from PID import Control
import time

# ------------------ 参数设置 ------------------
YOLO_WEIGHTS = "F:/python_project/badminton_detect/yolo_project/runs/train/yolov8_custom/weights/best.pt"
YOLO_CONF = 0.5

REDETECT_EVERY_N = 5

# 摄像头参数
CAMERA_INDEX = 1
CAMERA_WIDTH = 641
CAMERA_HEIGHT = 479
CAMERA_FPS = 120
# ------------------------------------------------

# ---- 初始化 ----
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"[INFO] 使用设备: {device}")

model = YOLO(YOLO_WEIGHTS)
model.to(device)
PID_Control = Control()

# ---- 打开摄像头 ----
cap = cv2.VideoCapture(CAMERA_INDEX + cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)

cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))

if not cap.isOpened():
    raise RuntimeError("无法打开外接摄像头")

# 检查摄像头是否真的设置成功
actual_fps = cap.get(cv2.CAP_PROP_FPS)
actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
print(f"[INFO] 摄像头请求: {CAMERA_WIDTH}x{CAMERA_HEIGHT} @ {CAMERA_FPS} FPS")
print(f"[INFO] 摄像头实际: {int(actual_width)}x{int(actual_height)} @ {int(actual_fps)} FPS")
print("[INFO] 摄像头已打开，按 'q' 退出。")


def create_kcf_tracker():
    # 兼容不同版本的OpenCV
    if hasattr(cv2, 'legacy') and hasattr(cv2.legacy, 'TrackerKCF_create'):
        return cv2.legacy.TrackerKCF_create()
    elif hasattr(cv2, 'TrackerKCF_create'):
        return cv2.TrackerKCF_create()
    else:
        raise RuntimeError("当前 OpenCV 不支持 KCF 跟踪器")


def yolo_detect_get_best_bbox(img, conf=YOLO_CONF):
    results = model.predict(img, conf=conf, device=device, verbose=False)
    boxes = results[0].boxes
    annotated = results[0].plot() # 获取YOLO绘制后的图像
    if boxes is None or len(boxes) == 0:
        return None, None, annotated
    confidences = boxes.conf.cpu().numpy()
    best_idx = np.argmax(confidences)
    best_box_xyxy = boxes.xyxy.cpu().numpy()[best_idx]
    best_confidence = confidences[best_idx]
    x1, y1, x2, y2 = best_box_xyxy
    bbox = (int(x1), int(y1), int(x2 - x1), int(y2 - y1))
    return bbox, best_confidence, annotated


tracker = None
tracking = False
frame_count = 0
# 初始化时，让云台指向画面中心。之后如果目标丢失，它会保持在这个变量记录的位置。
last_known_coords = (CAMERA_WIDTH // 2, CAMERA_HEIGHT // 2)

# ------------------ 主循环 ------------------
try:
    while True:
        ret, img = cap.read()
        if not ret:
            print("[WARN] 无法读取摄像头帧")
            time.sleep(0.1)
            continue

        img = np.ascontiguousarray(img, dtype=np.uint8)
        frame_count += 1
        display_frame = img.copy()
        h_img, w_img = img.shape[:2]
        
        current_frame_coords = None
        
        # ========== 阶段 1: 目标位置更新 ==========
        
        # 条件：1. 还未开始跟踪  2. 到了周期性校准的帧
        if not tracking or (frame_count % REDETECT_EVERY_N == 0):
            # === YOLO 检测模式 (包括校准) ===
            bbox_detect, confidence, annotated = yolo_detect_get_best_bbox(img, conf=YOLO_CONF)
            display_frame = annotated # 直接使用YOLO绘制的帧作为显示基础

            if bbox_detect is not None:
                x, y, w, h = bbox_detect
                current_frame_coords = (x + w // 2, y + h // 2)
                
                # 边界安全检查，确保bbox在图像内
                safe_bbox = (max(0, x), max(0, y), min(w_img-1-x, w), min(h_img-1-y, h))

                if safe_bbox[2] > 0 and safe_bbox[3] > 0:
                    tracker = create_kcf_tracker()
                    ok = tracker.init(img, safe_bbox)
                    if ok:
                        tracking = True
                        if frame_count > 1:
                             print(f"[INFO] 周期性校正成功 bbox={safe_bbox}, conf={confidence:.2f}")
                        else:
                             print(f"[INFO] 初始化跟踪成功 bbox={safe_bbox}, conf={confidence:.2f}")
                    else:
                        tracking = False # 初始化失败，下一帧继续YOLO
                        print(f"[ERROR] KCF 跟踪器初始化失败")
            else:
                # YOLO没找到目标，停止跟踪
                tracking = False
                tracker = None
                if frame_count > REDETECT_EVERY_N: # 避免启动时就打印
                    print("[INFO] 校正失败: 未检测到目标，回退YOLO检测")

        else:
            # === KCF 快速跟踪模式 ===
            ok, bbox = tracker.update(img)
            if ok and bbox is not None:
                x, y, w, h = [int(v) for v in bbox]
                current_frame_coords = (x + w // 2, y + h // 2)
                # 在画面上绘制KCF跟踪框
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.circle(display_frame, current_frame_coords, 5, (0, 255, 0), -1)
            else:
                tracking = False
                tracker = None
                print("[WARN] KCF 跟踪失败，回退 YOLO 检测")

        # ========== 阶段 2: 决策与控制 (已修改) ==========
        
        if current_frame_coords is not None:
            # 如果在当前帧找到了目标(无论是YOLO还是KCF)，就更新 "最后一次有效坐标"
            last_known_coords = current_frame_coords
            
            # 并命令云台追踪这个新找到的目标
            PID_Control.send_to_stm32(last_known_coords[0], last_known_coords[1])
        else:
            # 如果目标丢失 (current_frame_coords is None)，就命令云台追踪画面的正中心
            # 这将导致PID计算出的误差为0，从而让舵机停止运动
            center_x = CAMERA_WIDTH // 2
            center_y = CAMERA_HEIGHT // 2
            PID_Control.send_to_stm32(center_x, center_y)
        
        # ========== 阶段 3: 显示 ==========
        
        # 在最终要显示的画面上绘制一个醒目的黄色准星，表示云台当前锁定的目标点
        cv2.circle(display_frame, last_known_coords, 15, (0, 255, 255), 2)
        cv2.putText(display_frame, f"TARGET", (last_known_coords[0] + 15, last_known_coords[1] + 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        # 如果当前处于“搜索”状态（即未在跟踪），在画面中心画一个红色十字辅助
        if not tracking:
             center_x, center_y = w_img // 2, h_img // 2
             cv2.line(display_frame, (center_x - 15, center_y), (center_x + 15, center_y), (0, 0, 255), 2)
             cv2.line(display_frame, (center_x, center_y - 15), (center_x, center_y + 15), (0, 0, 255), 2)

        cv2.imshow("YOLO + KCF Tracking @ 120fps (Simple)", display_frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

finally:
    print("[INFO] 程序退出")
    cap.release()
    cv2.destroyAllWindows()
