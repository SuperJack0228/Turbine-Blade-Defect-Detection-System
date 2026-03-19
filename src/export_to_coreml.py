# export_to_coreml.py

import torch
import torch.nn as nn
import coremltools as ct
from model.classifier import BladeClassifier, CLASS_NAMES

class WrappedModel(nn.Module):
    """
    在前面加上 Normalize(mean,std)，
    后面接原来的分类网络（不含 softmax）。
    """
    def __init__(self, base_model: BladeClassifier):
        super().__init__()
        # 注意：base_model.model 已经是 resnet18(fc 已改)
        self.model = base_model.model
        # 把均值和 std 变成 [1,3,1,1] 形状的常量
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1)
        std  = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1)
        # 注册为 buffer，让它随模型一起保存和 trace
        self.register_buffer("mean", mean)
        self.register_buffer("std",  std)

    def forward(self, x):
        # x 输入是 [0,1] 之间的像素（Vision 送来的已归一化图像）
        x = (x - self.mean) / self.std
        # 得到 logits
        logits = self.model(x)
        return logits

def main():
    # 1. 加载基础分类器
    base = BladeClassifier(weights_path="output/best_model.pth")
    base.model.eval()
    base.model.cpu()

    # 2. 包裹成加了 Normalize 的模型
    wrapped = WrappedModel(base)
    wrapped.eval()
    wrapped.cpu()

    # 3. Trace 成 TorchScript
    example = torch.rand(1, 3, 224, 224)  # Vision 会传 [0,1] 的 float 图
    traced = torch.jit.trace(wrapped, example)

    # 4. 转成 Core ML，输入就是像素范围[0,1]，不需额外 scale、bias
    mlmodel = ct.convert(
        traced,
        inputs=[ct.ImageType(name="input_1", shape=example.shape)],
        classifier_config=ct.ClassifierConfig(class_labels=CLASS_NAMES)
    )

    # 5. 保存
    mlmodel.save("BladeClassifier.mlpackage")
    print("✅ 已生成带预处理的 BladeClassifier.mlpackage")

if __name__ == "__main__":
    main()
