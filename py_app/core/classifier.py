# model/classifier.py

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import json
import os

# 这里填写你的缺陷类别名称，顺序需和训练时一致
CLASS_NAMES = ["划痕", "擦伤", "污渍", "表面压痕", "墨点"]

class BladeClassifier:
    def __init__(self,
                 weights_path="output/best_model.pth",
                 num_classes=len(CLASS_NAMES),
                 device=None):
        """
        推理用分类器，加载训练好的权重做单张图预测
        """
        # 1. 设备优先 GPU
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # 2. 构建 ResNet18 网络骨架，并替换最后一层
        self.model = models.resnet18(pretrained=True)
        in_feats = self.model.fc.in_features
        self.model.fc = nn.Linear(in_feats, num_classes)

        # 3. 加载权重文件
        if os.path.exists(weights_path):
            state_dict = torch.load(weights_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            print(f"✅ 已加载分类权重：{weights_path}")
        else:
            raise FileNotFoundError(f"找不到权重文件：{weights_path}")

        # 4. 切到指定设备 & 推理模式
        self.model = self.model.to(self.device)
        self.model.eval()

        # 5. 图像预处理，与训练时保持一致
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def predict(self, image_path):
        """
        预测一张图片
        返回：{"defect": 类别名, "confidence": 置信度}
        """
        img = Image.open(image_path).convert("RGB")
        x = self.transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1)[0]
        conf, idx = torch.max(probs, 0)
        return {
            "defect": CLASS_NAMES[idx.item()],
            "confidence": round(conf.item(), 4)
        }

if __name__ == "__main__":
    # 测试用示例
    import sys
    if len(sys.argv) != 2:
        print("用法: python classifier.py /path/to/image.jpg")
        sys.exit(1)
    img = sys.argv[1]
    clf = BladeClassifier(weights_path="output/best_model.pth")
    res = clf.predict(img)
    print(json.dumps(res, ensure_ascii=False))
