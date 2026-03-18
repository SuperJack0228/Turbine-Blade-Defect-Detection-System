# Turbine-Blade-Defect-Detection-System
项目概述： 本项目设计并实现了一套两级视觉检测算法，用于涡轮叶片表面的精密缺陷检测 。系统集成工业六轴机械臂与传送带工站，实现了从自动上料到检测回流的全自动化闭环 。
# Blade-Vision-Inspector: 工业级涡轮叶片表面缺陷检测系统

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg) 
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-orange.svg) 
![YOLOv8](https://img.shields.io/badge/Model-YOLOv8%20%7C%20ResNet18-green.svg) 
![Status](https://img.shields.io/badge/Project-Industrial--Ready-red.svg)

## 📌 项目概述 (Project Overview)
本项目是一套集成 **六轴工业机器人**、**高分辨率视觉传感器** 与 **深度学习算法** 的端到端自动化质检方案。旨在解决汽车发动机涡轮叶片在高温高压工况下，人工目检效率低、易漏检微小缺陷的痛点。

通过构建“**采集-识别-交互**”三层架构，系统实现了从自动化路径规划拍摄、实时 AI 推理到结果可视化展示的全流程闭环。

---

## 🛠️ 技术架构 (Technical Architecture)

### 1. 硬件集成 (Hardware Integration)
本项目采用了工业级标准的硬件选型，确保了在生产线环境下的重复精度：
* **执行机构**：KUKA KR10R1100-2 六轴工业机器人（重复定位精度 $\pm 0.03mm$)。
* **视觉传感器**：Hikrobot MV-CE050-10UM 工业相机（500 万像素，Sony IMX264 传感器。
* **光学系统**：定制 LED 环形光源及 16mm 工业定焦镜头，针对金属曲面反光特性进行了光路优化。
* **物流系统**：皮带驱动式滚筒输送线，支持光电触发自动上料

### 2. AI 算法链路 (AI Pipeline)
设计了两条模型路径以兼顾 **分类精度** 与 **空间定位时效性**：
* **缺陷分类路径 (ResNet-18)**：针对五类缺陷（划痕、擦伤、污渍、压痕、墨点）进行高精度判定。
* **目标检测路径 (YOLOv8m)**：实现缺陷区域的动态回归预测，支持边缘端多目标并发识别。
* **优化策略**：采用 **迁移学习 (Transfer Learning)** 加快收敛，利用数据增强（旋转、亮度扰动等）提升模型鲁棒性。

---

## 📂 核心功能 (Core Features)

* **自动化采集流程**：通过离线示教与路径规划，机械臂实现“上俯-侧扫-翻转”多姿态成像，覆盖复杂曲面盲区。
* **工业化数据集构建**：包含 5000+ 张原始缺陷图像，经过严格的归一化处理及清洗。
* **集成式 GUI 平台**：基于 PyQt5 开发，支持模型加载、推理调用及检测日志导出。

---

## 📊 性能评估 (Performance Evaluation)

| 指标 (Metric) | 分类准确率 (ResNet) | 检测 mAP@0.5 (YOLOv8) | 推理时延 (Single Image) |
| :--- | :--- | :--- | :--- |
| **数值 (Value)** | **>95.0%** | **~71.1%** | **<1.5s** |
注：对于表面划伤和磕伤等特征明显的缺陷，置信度通常稳定在 0.8 以上
---
## 🚀 快速开始 (Getting Started)
**环境要求**
* Python 3.9+
* PyTorch 2.0 (with MPS or CUDA support)
* OpenCV-Python
* PyQt5
  
---

## 安装部署
* 克隆仓库
git clone https://github.com/YourUsername/Blade-Vision-Inspector.git
* 安装依赖
pip install -r requirements.txt
* 启动 GUI 平台
python main.py
