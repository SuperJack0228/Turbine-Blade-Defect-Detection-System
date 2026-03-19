# model/detector.py

import os
import json
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont

class BladeDetector:
    def __init__(self, model_name="yolov8s.pt"):
        """
        推理用检测器，加载 YOLOv8 预训练模型
        参数:
            model_name: 模型文件名或官方模型标识，如 "yolov8s.pt" 或 "yolov8s"
        """
        # 自动下载并加载模型
        self.model = YOLO(model_name)

    def predict(self, image_path, save_img=True, out_dir="output"):
        """
        对单张图片运行检测
        参数:
            image_path: 图像路径
            save_img: 是否把带框图片保存到 out_dir
            out_dir: 保存目录
        返回:
            列表，元素为 {"bbox":[x1,y1,x2,y2],"confidence":float,"class_id":int}
        """
        # 1）推理
        results = self.model.predict(source=image_path, save=False, imgsz=640)

        # 2）解析结果列表
        dets = results[0].boxes.data.tolist()  # 每个 box: [x1,y1,x2,y2,conf,cls_id]
        output = []
        for x1, y1, x2, y2, conf, cls_id in dets:
            output.append({
                "bbox": [round(x1,1), round(y1,1), round(x2,1), round(y2,1)],
                "confidence": round(conf,4),
                "class_id": int(cls_id)
            })

        # 3）保存可视化图
        if save_img:
            os.makedirs(out_dir, exist_ok=True)
            img = Image.open(image_path)
            draw = ImageDraw.Draw(img)
            font = ImageFont.load_default()
            for obj in output:
                x1, y1, x2, y2 = obj["bbox"]
                draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
                label = f"{obj['class_id']}:{obj['confidence']}"
                draw.text((x1, y1 - 10), label, fill="red", font=font)
            save_path = os.path.join(out_dir, "det_" + os.path.basename(image_path))
            img.save(save_path)
            print(f"✅ 已保存检测图 → {save_path}")

        return output

if __name__ == "__main__":
    # 终端测试示例
    import sys
    if len(sys.argv) != 2:
        print("用法: python detector.py /path/to/image.jpg")
        sys.exit(1)
    img = sys.argv[1]
    det = BladeDetector()
    res = det.predict(img)
    print(json.dumps(res, ensure_ascii=False))
