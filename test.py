import cv2
import time


# 打开编号为 1 的外接摄像头
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

# 设置分辨率和帧率
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 641)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 479)
cap.set(cv2.CAP_PROP_FPS, 120)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))

if not cap.isOpened():
    raise RuntimeError("外接摄像头(编号1)无法打开")

prev_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print("捕获失败，检查摄像头连接")
        break

    # 显示画面
    cv2.imshow("外接摄像头 - 原始", frame)
    current_time = time.time()
    print(f"fps:{1 / (current_time -prev_time):.2f}")
    prev_time = current_time

    # 按 ESC 退出
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
