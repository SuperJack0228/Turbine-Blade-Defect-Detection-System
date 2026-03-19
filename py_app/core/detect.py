# core/detect.py
from ultralytics import YOLO

# 检测类别名称，需与训练集顺序一致
YOLO_CLASSES = ['表面划伤', '表面磕伤', '表面污渍', '表面压痕']

# 权重路径改为你自己的best.pt
MODEL_PATH = "/Users/superjack/test/turbo_detector/train22/weights/best.pt"

# 加载模型只需一次
_yolo_model = None

def get_yolo_model():
    global _yolo_model
    if _yolo_model is None:
        _yolo_model = YOLO(MODEL_PATH)
    return _yolo_model

def detect_image(image_path):
    """
    输入图片路径，返回检测目标列表，每个目标为dict：
    {bbox: [x1, y1, x2, y2], label: '缺陷类型', confidence: 置信度}
    """
    model = get_yolo_model()
    results = model.predict(source=image_path, save=False, imgsz=640)
    detections = []
    for r in results:
        for box in r.boxes.data.tolist():
            x1, y1, x2, y2, conf, cls_id = box
            label = YOLO_CLASSES[int(cls_id)] if int(cls_id) < len(YOLO_CLASSES) else f"类别{int(cls_id)}"
            detections.append({
                "bbox": [round(x1,1), round(y1,1), round(x2,1), round(y2,1)],
                "label": label,
                "confidence": round(conf, 4)
            })
    return detections

# 仅用于命令行单独测试
if __name__ == "__main__":
    import sys, json
    if len(sys.argv) != 2:
        print("用法: python detect.py /path/to/image.jpg")
        sys.exit(1)
    result = detect_image(sys.argv[1])
    print(json.dumps(result, ensure_ascii=False, indent=2))
