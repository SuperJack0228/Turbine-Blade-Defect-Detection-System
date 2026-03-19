# infer.py

import sys
import json
from model.classifier import BladeClassifier

def run_classification(image_path):
    """
    加载分类模型，预测单张图像，打印 JSON 结果
    """
    clf = BladeClassifier(weights_path="output/best_model.pth")
    res = clf.predict(image_path)
    print(json.dumps(res, ensure_ascii=False))

if __name__ == "__main__":
    # 命令行用法: python infer.py cls /path/to/image.jpg
    if len(sys.argv) != 3:
        print("用法: python infer.py cls /path/to/image.jpg")
        sys.exit(1)
    mode, img_path = sys.argv[1], sys.argv[2]
    if mode.lower() == "cls":
        run_classification(img_path)
    else:
        print(f"不支持的模式: {mode}")
        sys.exit(1)
