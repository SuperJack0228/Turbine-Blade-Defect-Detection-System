# infer.py
# infer.py
import sys, os
# 确保能导入 core 目录下的 classifier.py
sys.path.insert(0, os.path.dirname(__file__))
# 确保权重路径能从项目根访问
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

import sys
import json
from classifier import BladeClassifier


def run_classification(image_path):
    """
    加载分类模型，预测单张图像，打印 JSON 结果
    """
    clf = BladeClassifier(weights_path=os.path.join(ROOT, "output", "best_model.pth"))
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
