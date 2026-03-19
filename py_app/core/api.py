#!/usr/bin/env python3
import os
import sys

# 确保能导入同目录下的 classifier.py 和 detect.py
CORE_DIR = os.path.dirname(os.path.abspath(__file__))
if CORE_DIR not in sys.path:
    sys.path.insert(0, CORE_DIR)

from classifier import BladeClassifier
from detect import detect_image as _detect

def classify_image(image_path: str) -> dict:
    """
    调用分类器进行图像分类，返回包含 defect, confidence 的字典
    """
    # 定位到项目根目录（core -> py_app/core -> ../../ => turbo_detector）
    project_root = os.path.abspath(os.path.join(CORE_DIR, "..", ".."))
    weights_path = os.path.join(project_root, "output", "best_model.pth")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"权重文件未找到: {weights_path}")
    clf = BladeClassifier(weights_path=weights_path)
    return clf.predict(image_path)


def detect_image(image_path: str) -> list:
    """
    调用检测器进行目标检测，返回检测结果列表
    """
    return _detect(image_path)
