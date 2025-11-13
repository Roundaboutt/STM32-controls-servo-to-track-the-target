import torch
from ultralytics import YOLO
import os

def train():

    # ======================
    # 1. 写入 YOLOv8 data.yaml
    # ======================
    os.makedirs("datasets/yolov8", exist_ok=True)

    yaml_path = "datasets/yolov8/data.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(
            "train: F:/python_project/badminton_detect/yolo_project/datasets/images\n"
            "val: F:/python_project/badminton_detect/yolo_project/datasets/images\n"
            "nc: 1\n"
            "names: ['badminton']\n"
        )

    print(f"data.yaml 已生成 -> {yaml_path}")

    # ======================
    # 2. 加载 YOLOv8 模型
    # ======================
    model = YOLO("yolov8s.pt")
    # ======================
    # 3. 训练
    # ======================
    results = model.train(
        data=yaml_path,                    # 数据配置文件
        epochs=100,
        imgsz=640,
        batch=16,
        device=0 if torch.cuda.is_available() else "cpu",
        project="runs/train",
        name="yolov8_custom",
        exist_ok=True,
    )

if __name__ == '__main__':
    train()