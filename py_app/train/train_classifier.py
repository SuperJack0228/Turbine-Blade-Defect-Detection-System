#!/usr/bin/env python3
import os
import argparse
import time
print("start1", flush=True)
import torch
print("start2", flush=True)
import torchvision
print("start3", flush=True)
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets
import matplotlib.pyplot as plt

def train_and_evaluate(model, trainloader, testloader, optimizer, criterion, num_epochs):
    best_acc = 0.0
    loss_history = []
    acc_history  = []

    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total   = 0
        start = time.time()

        for i, (inputs, labels) in enumerate(trainloader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)
            if (i == 0) or ((i + 1) % 50 == 0) or ((i + 1) == len(trainloader)):
                print(f"Batch [{i+1}/{len(trainloader)}], Loss: {loss.item():.4f}", flush=True)

        
        epoch_loss = running_loss / total
        epoch_acc  = correct / total * 100
        loss_history.append(epoch_loss)
        acc_history.append(epoch_acc)

        elapsed = time.time() - start
        print(f"Epoch [{epoch}/{num_epochs}]  "
              f"Loss: {epoch_loss:.4f}  "
              f"Acc: {epoch_acc:.2f}%  "
              f"Time: {elapsed:.1f}s",flush=True)

        # 验证集准确率，用于选最优权重
        model.eval()
        correct_val = 0
        total_val   = 0
        with torch.no_grad():
            for inputs, labels in testloader:
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                correct_val += (preds == labels).sum().item()
                total_val   += labels.size(0)
        val_acc = correct_val / total_val * 100
        if val_acc > best_acc:
            best_acc = val_acc
            # 保存最优权重
            torch.save(model.state_dict(), os.path.join(args.output_dir, "best_model.pth"))

        print(f" → Val Acc: {val_acc:.2f}%  (Best: {best_acc:.2f}%)\n",flush=True)

    return loss_history, acc_history

if __name__ == "__main__":
    print("Start", flush=True)
    parser = argparse.ArgumentParser(description="Train ResNet-18 on defect classification")
    parser.add_argument("--train-dir", type=str, required=True,
                        help="Path to training data folder (ImageFolder format)")
    parser.add_argument("--test-dir", type=str, required=True,
                        help="Path to test data folder (ImageFolder format)")
    parser.add_argument("--epochs",    type=int,   default=15,  help="Number of epochs")
    parser.add_argument("--batch-size",type=int,   default=4,   help="Batch size")
    parser.add_argument("--lr",        type=float, default=0.001, help="Learning rate")
    parser.add_argument("--output-dir",type=str,   default="../output",
                        help="Directory to save best_model.pth")
    args = parser.parse_args()

    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)
    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std =[0.229, 0.224, 0.225])
    ])
    print("transform ok", flush=True)
    print("准备加载训练集", flush=True)
    # 加载数据
    train_dataset = datasets.ImageFolder(root=args.train_dir, transform=transform)
    print("train_dataset ok", flush=True)
    test_dataset  = datasets.ImageFolder(root=args.test_dir,  transform=transform)
    print("test_dataset ok", flush=True)
    trainloader = DataLoader(train_dataset, batch_size=args.batch_size,
                             shuffle=True, num_workers=0)
    print("trainloader ok", flush=True)
    testloader  = DataLoader(test_dataset,  batch_size=args.batch_size,
                             shuffle=False, num_workers=0)
    print("testloader ok", flush=True)

    # 构建模型
    model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
    print("model ok", flush=True)
    model.fc = nn.Linear(model.fc.in_features, len(train_dataset.classes))
    print("fc ok", flush=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # 训练与评估
    print("准备开始训练", flush=True)
    loss_hist, acc_hist = train_and_evaluate(
        model, trainloader, testloader, optimizer, criterion, args.epochs
    )
    print("训练完成", flush=True)
    # 绘制并保存曲线
    # 曲线保存到 data 目录
    data_dir = os.path.abspath(os.path.join(args.output_dir, "..", "data"))
    os.makedirs(data_dir, exist_ok=True)

    # 损失曲线
    plt.figure()
    plt.plot(range(1, args.epochs+1), loss_hist, marker='o')
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig(os.path.join(data_dir, "training_loss.png"))

    # 准确率曲线
    plt.figure()
    plt.plot(range(1, args.epochs+1), acc_hist, marker='o')
    plt.title("Training Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.grid(True)
    plt.savefig(os.path.join(data_dir, "training_accuracy.png"))

    print(f"✅ Training complete. Best model saved to {args.output_dir}/best_model.pth",flush=True)
    print(f"✅ Plots saved to {data_dir}/training_loss.png and training_accuracy.png",flush=True)
