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

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 数据集路径请按实际修改
train_data_STIE = datasets.ImageFolder(root='/Users/superjack/Downloads/Data/S-TIE/train-dataroot', transform=transform)
test_data_STIE = datasets.ImageFolder(root='/Users/superjack/Downloads/Data/S-TIE/test-dataroot', transform=transform)

trainloader_STIE = DataLoader(train_data_STIE, batch_size=4, shuffle=True, num_workers=0)
testloader_STIE = DataLoader(test_data_STIE, batch_size=4, shuffle=False, num_workers=0)

# 检查数据加载是否正常
for images, labels in trainloader_STIE:
    print(f"Image batch dimensions: {images.shape}")
    print(f"Labels batch dimensions: {labels.shape}")
    break

def train_and_test(model, trainloader, testloader, optimizer, criterion, num_epochs=18):
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
            torch.save(model.state_dict(), 'output/best_model_5class.pth')

    # 测试阶段
    print("Starting Testing...")
    model.eval()
    correct_predictions = 0
    total_predictions = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in testloader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total_predictions += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
            all_preds.extend(predicted.numpy())
            all_labels.extend(labels.numpy())

    test_accuracy = correct_predictions / total_predictions * 100
    print(f"Test Accuracy: {test_accuracy:.2f}%")

    return loss_history, accuracy_history, all_labels, all_preds

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

def plot_confusion_matrix(all_labels, all_preds, class_names):
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.show()

def main():
    model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 5)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 训练+测试
    loss_history, accuracy_history, all_labels, all_preds = train_and_test(
        model, trainloader_STIE, testloader_STIE, optimizer, criterion, num_epochs=100) # 20可以改大

    # 保存历史记录
    np.save('loss_history.npy', np.array(loss_history))
    np.save('accuracy_history.npy', np.array(accuracy_history))

    # 曲线绘制
    plot_curves(loss_history, accuracy_history)

    # 混淆矩阵绘制
    class_names = train_data_STIE.classes
    plot_confusion_matrix(all_labels, all_preds, class_names)

if __name__ == '__main__':
    main()
