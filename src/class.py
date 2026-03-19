# train_classifier.py

import os
import time
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets
import matplotlib.pyplot as plt

# --------- 配置区 ----------
TRAIN_DIR    = '/Users/superjack/Downloads/Data/T-LV/train_dataroot'
TEST_DIR     = '/Users/superjack/Downloads/Data/T-LV/test_dataroot'
OUTPUT_DIR   = 'output'
WEIGHTS_PATH = os.path.join(OUTPUT_DIR, 'best_model.pth')

BATCH_SIZE   = 4
LR           = 1e-3
NUM_CLASSES  = 5
NUM_EPOCHS   = 20
# --------------------------

# 1. 图像预处理，保持和推理一致
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

# 2. 创建数据集与加载器
train_ds = datasets.ImageFolder(root=TRAIN_DIR, transform=transform)
test_ds  = datasets.ImageFolder(root=TEST_DIR,  transform=transform)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)

# 3. 构建模型、损失函数、优化器
model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

best_acc = 0.0
loss_history = []

# 4. 训练循环
for epoch in range(NUM_EPOCHS):
    epoch_start = time.time()
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / len(train_loader)
    epoch_acc  = correct / total * 100
    loss_history.append(epoch_loss)

    # 验证
    model.eval()
    val_correct = 0
    val_total   = 0
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            preds = outputs.argmax(dim=1)
            val_correct += (preds == labels).sum().item()
            val_total   += labels.size(0)
    val_acc = val_correct / val_total * 100

    # 保存最优权重
    if val_acc > best_acc:
        best_acc = val_acc
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        torch.save(model.state_dict(), WEIGHTS_PATH)

    print(f"Epoch {epoch+1}/{NUM_EPOCHS}  "
          f"Train Loss: {epoch_loss:.4f}  "
          f"Train Acc: {epoch_acc:.2f}%  "
          f"Val Acc: {val_acc:.2f}%  "
          f"time: {time.time()-epoch_start:.1f}s")

# 5. 绘制并保存损失曲线
plt.plot(range(1, NUM_EPOCHS+1), loss_history, marker='o')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
os.makedirs(OUTPUT_DIR, exist_ok=True)
plt.savefig(os.path.join(OUTPUT_DIR, 'training_loss.png'))
plt.show()

print(f"\n🎉 训练完成，最佳验证准确率：{best_acc:.2f}%")
print(f"权重保存在：{WEIGHTS_PATH}")