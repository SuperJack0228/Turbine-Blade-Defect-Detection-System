import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets
import time
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# 自动创建output目录
os.makedirs('output', exist_ok=True)

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# T-LV数据集路径（按你实际路径填写）
train_data_TLV = datasets.ImageFolder(root='/Users/superjack/Downloads/Data/T-LV/train_dataroot', transform=transform)
test_data_TLV = datasets.ImageFolder(root='/Users/superjack/Downloads/Data/T-LV/test_dataroot', transform=transform)

trainloader_TLV = DataLoader(train_data_TLV, batch_size=4, shuffle=True, num_workers=0)
testloader_TLV = DataLoader(test_data_TLV, batch_size=4, shuffle=False, num_workers=0)

# 检查类别数量
num_classes = len(train_data_TLV.classes)
print("类别列表：", train_data_TLV.classes)
print("类别数：", num_classes)

# 检查数据加载是否正常
for images, labels in trainloader_TLV:
    print(f"Image batch dimensions: {images.shape}")
    print(f"Labels batch dimensions: {labels.shape}")
    break

def train_and_test(model, trainloader, testloader, optimizer, criterion, num_epochs=80):
    loss_history = []
    accuracy_history = []
    best_acc = 0

    for epoch in range(num_epochs):
        print(f"Starting Epoch [{epoch + 1}/{num_epochs}]...")
        epoch_start_time = time.time()

        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        for i, (inputs, labels) in enumerate(trainloader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total_predictions += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

            if i % 100 == 0:
                print(f"  Batch [{i + 1}/{len(trainloader)}], Loss: {loss.item():.4f}")

        epoch_loss = running_loss / len(trainloader)
        epoch_accuracy = correct_predictions / total_predictions * 100
        loss_history.append(epoch_loss)
        accuracy_history.append(epoch_accuracy)
        epoch_end_time = time.time()

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%")
        print(f"Epoch [{epoch + 1}] completed in {epoch_end_time - epoch_start_time:.2f} seconds.\n")

        # 只要有更高的训练准确率就保存
        if epoch_accuracy > best_acc:
            best_acc = epoch_accuracy
            torch.save(model.state_dict(), 'output/best_model_TLV.pth')

    # 返回训练损失、准确率
    return loss_history, accuracy_history

def plot_curves(loss_history, accuracy_history):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(loss_history)+1), loss_history, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss vs. Epoch')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(accuracy_history)+1), accuracy_history, label='Training Accuracy', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training Accuracy vs. Epoch')
    plt.legend()

    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(model, dataloader, class_names):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in dataloader:
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.numpy())
            all_labels.extend(labels.numpy())

    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.show()

def main():
    # 构建ResNet18，最后一层自动根据类别数调整
    model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 训练
    loss_history, accuracy_history = train_and_test(
        model, trainloader_TLV, testloader_TLV, optimizer, criterion, num_epochs=80)

    # 保存损失和准确率
    np.save('loss_history_TLV.npy', np.array(loss_history))
    np.save('accuracy_history_TLV.npy', np.array(accuracy_history))

    # 绘制损失和准确率曲线
    plot_curves(loss_history, accuracy_history)

    # === 加载最佳权重后再画混淆矩阵 ===
    model.load_state_dict(torch.load('output/best_model_TLV.pth', map_location='cpu'))
    model.eval()
    plot_confusion_matrix(model, testloader_TLV, train_data_TLV.classes)

if __name__ == '__main__':
    main()
