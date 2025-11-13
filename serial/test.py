import cv2
import torch
import numpy as np
from ultralytics import YOLO
from PID import Control
import time

# ------------------ 参数设置 ------------------
YOLO_WEIGHTS = "F:/python_project/badminton_detect/yolo_project/runs/train/yolov8_custom/weights/best.pt"
YOLO_CONF = 0.5

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


def yolo_detect_get_best_bbox(img, conf=YOLO_CONF):
    """
    使用YOLO模型进行检测，并返回置信度最高的目标边界框。
    """
    results = model.predict(img, conf=conf, device=device, verbose=False)
    boxes = results[0].boxes
    annotated_frame = results[0].plot() # 获取YOLO绘制后的图像

    if boxes is None or len(boxes) == 0:
        return None, annotated_frame # 如果没有检测到目标，返回None和绘制后的图

    confidences = boxes.conf.cpu().numpy()
    best_idx = np.argmax(confidences)
    best_box_xyxy = boxes.xyxy.cpu().numpy()[best_idx]
    
    x1, y1, x2, y2 = best_box_xyxy
    bbox = (int(x1), int(y1), int(x2 - x1), int(y2 - y1)) # (x, y, w, h) 格式
    
    return bbox, annotated_frame


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
        
        h_img, w_img = img.shape[:2]
        
        # ========== 阶段 1: 目标位置更新 (纯YOLO) ==========
        
        bbox, display_frame = yolo_detect_get_best_bbox(img, conf=YOLO_CONF)
        
        target_found_this_frame = False
        
        if bbox is not None:
            target_found_this_frame = True
            x, y, w, h = bbox
            current_frame_coords = (x + w // 2, y + h // 2)
            last_known_coords = current_frame_coords
        
        # ========== 阶段 2: 决策与控制 (已修改) ==========
        
        if target_found_this_frame:
            # 如果在当前帧找到了目标，就命令云台追踪这个目标
            # last_known_coords 已经被更新为目标的最新坐标
            PID_Control.send_to_stm32(last_known_coords[0], last_known_coords[1])
        else:
            # 如果目标丢失，就命令云台追踪画面的正中心
            # 这将导致PID计算出的误差为0，从而让舵机停止运动
            center_x = CAMERA_WIDTH // 2
            center_y = CAMERA_HEIGHT // 2
            PID_Control.send_to_stm32(center_x, center_y)
        
        # ========== 阶段 3: 显示 ==========
        
        # 在最终要显示的画面上绘制一个醒目的黄色准星，表示云台当前锁定的目标点
        # 注意：这里的 last_known_coords 仍然会画在目标最后出现的位置，这是一个很好的视觉反馈
        cv2.circle(display_frame, last_known_coords, 15, (0, 255, 255), 2)
        cv2.putText(display_frame, f"TARGET", (last_known_coords[0] + 15, last_known_coords[1] + 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        # 如果当前处于“搜索”状态（即本帧未找到目标），在画面中心画一个红色十字辅助
        if not target_found_this_frame:
             center_x, center_y = w_img // 2, h_img // 2
             cv2.line(display_frame, (center_x - 15, center_y), (center_x + 15, center_y), (0, 0, 255), 2)
             cv2.line(display_frame, (center_x, center_y - 15), (center_x, center_y + 15), (0, 0, 255), 2)

        cv2.imshow("Pure YOLO Tracking", display_frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

finally:
    print("[INFO] 程序退出")
    cap.release()
    cv2.destroyAllWindows()