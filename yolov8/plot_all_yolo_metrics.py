import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

df = pd.read_csv("/Users/superjack/test/turbo_detector_副本/yolov8/results.csv")
epochs = df['epoch']

# 1. 损失曲线
plt.figure(figsize=(10, 6))
plt.plot(epochs, df['train/box_loss'], label='训练集框损失')
plt.plot(epochs, df['train/cls_loss'], label='训练集分类损失')
plt.plot(epochs, df['train/dfl_loss'], label='训练集分布损失')
plt.plot(epochs, df['val/box_loss'], label='验证集框损失', linestyle='--')
plt.plot(epochs, df['val/cls_loss'], label='验证集分类损失', linestyle='--')
plt.plot(epochs, df['val/dfl_loss'], label='验证集分布损失', linestyle='--')
plt.xlabel('训练轮数')
plt.ylabel('损失值')
plt.title('损失函数随训练轮数变化曲线')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# 2. 验证集主要指标
plt.figure(figsize=(10, 6))
plt.plot(epochs, df['metrics/precision(B)'], label='精确率')
plt.plot(epochs, df['metrics/recall(B)'], label='召回率')
plt.plot(epochs, df['metrics/mAP50(B)'], label='mAP@0.5')
plt.plot(epochs, df['metrics/mAP50-95(B)'], label='mAP@[0.5:0.95]')
plt.xlabel('训练轮数')
plt.ylabel('指标值')
plt.title('验证集指标随训练轮数变化曲线')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# 3. 学习率变化
plt.figure(figsize=(10, 5))
plt.plot(epochs, df['lr/pg0'], label='学习率组0')
plt.plot(epochs, df['lr/pg1'], label='学习率组1')
plt.plot(epochs, df['lr/pg2'], label='学习率组2')
plt.xlabel('训练轮数')
plt.ylabel('学习率')
plt.title('学习率随训练轮数变化曲线')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
