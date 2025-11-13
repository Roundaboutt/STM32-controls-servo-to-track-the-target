from ultralytics import YOLO

# 1. 加载你训练好的模型
model = YOLO("runs/train/yolov8_custom/weights/best.pt")

# 2. 对指定图片进行预测
results = model.predict(
    source=r"yolo_project\test_img",  # 你的图片路径
    save=True,                       # 保存带框结果
    imgsz=320,                       # 图片尺寸
    conf=0.5,                       # 置信度阈值
    device=0
)

# 3. 打印检测结果
for r in results:
    print(r.boxes.xyxy)  # 边界框坐标 [x1, y1, x2, y2]
    print(r.boxes.conf)  # 每个框的置信度
    print(r.boxes.cls)   # 类别编号
