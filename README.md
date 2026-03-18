# Turbine-Blade-Defect-Detection-System
项目概述： 本项目设计并实现了一套两级视觉检测算法，用于涡轮叶片表面的精密缺陷检测 。系统集成工业六轴机械臂与传送带工站，实现了从自动上料到检测回流的全自动化闭环 。
# Blade-Vision-Inspector: 工业级涡轮叶片表面缺陷检测系统

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg) 
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-orange.svg) 
![YOLOv8](https://img.shields.io/badge/Model-YOLOv8%20%7C%20ResNet18-green.svg) 
![Status](https://img.shields.io/badge/Project-Industrial--Ready-red.svg)

## 📌 项目概述 (Project Overview)
[cite_start]本项目是一套集成 **六轴工业机器人**、**高分辨率视觉传感器** 与 **深度学习算法** 的端到端自动化质检方案 [cite: 50, 56, 168, 188][cite_start]。旨在解决汽车发动机涡轮叶片在高温高压工况下，人工目检效率低、易漏检微小缺陷的痛点 [cite: 50, 161, 163, 205]。

[cite_start]通过构建“**采集-识别-交互**”三层架构，系统实现了从自动化路径规划拍摄、实时 AI 推理到结果可视化展示的全流程闭环 [cite: 51, 193, 223, 224, 237]。

---

## 🛠️ 技术架构 (Technical Architecture)

### 1. 硬件集成 (Hardware Integration)
[cite_start]本项目采用了工业级标准的硬件选型，确保了在生产线环境下的重复精度 [cite: 191, 245, 247, 321]：
* [cite_start]**执行机构**：KUKA KR10R1100-2 六轴工业机器人（重复定位精度 $\pm 0.03mm$）[cite: 100, 232, 253, 255, 265]。
* [cite_start]**视觉传感器**：Hikrobot MV-CE050-10UM 工业相机（500 万像素，Sony IMX264 传感器）[cite: 101, 232, 275, 276, 279]。
* [cite_start]**光学系统**：定制 LED 环形光源及 16mm 工业定焦镜头，针对金属曲面反光特性进行了光路优化 [cite: 232, 249]。
* [cite_start]**物流系统**：皮带驱动式滚筒输送线，支持光电触发自动上料 [cite: 191, 250]。

### 2. AI 算法链路 (AI Pipeline)
[cite_start]设计了两条模型路径以兼顾 **分类精度** 与 **空间定位时效性** [cite: 54, 188, 214, 224, 233]：
* [cite_start]**缺陷分类路径 (ResNet-18)**：针对五类缺陷（划痕、擦伤、污渍、压痕、墨点）进行高精度判定 [cite: 54, 110, 112, 192, 233, 326]。
* [cite_start]**目标检测路径 (YOLOv8m)**：实现缺陷区域的动态回归预测，支持边缘端多目标并发识别 [cite: 54, 127, 192, 233, 411, 489]。
* [cite_start]**优化策略**：采用 **迁移学习 (Transfer Learning)** 加快收敛，利用数据增强（旋转、亮度扰动等）提升模型鲁棒性 [cite: 53, 113, 184, 192, 340, 365, 370]。

---

## 📂 核心功能 (Core Features)

* [cite_start]**自动化采集流程**：通过离线示教与路径规划，机械臂实现“上俯-侧扫-翻转”多姿态成像，覆盖复杂曲面盲区 [cite: 52, 67, 191, 232, 262]。
* [cite_start]**工业化数据集构建**：包含 5000+ 张原始缺陷图像，经过严格的归一化处理及清洗 [cite: 53, 70, 333, 437, 452, 458]。
* [cite_start]**集成式 GUI 平台**：基于 PyQt5 开发，支持模型加载、推理调用及检测日志导出 [cite: 32, 55, 577, 579, 587, 590, 641]。

---

## 📊 性能评估 (Performance Evaluation)

| 指标 (Metric) | 分类准确率 (ResNet) | 检测 mAP@0.5 (YOLOv8) | 推理时延 (Single Image) |
| :--- | :--- | :--- | :--- |
| **数值 (Value)** | [cite_start]**>95.0%** [cite: 216, 391] | [cite_start]**~71.1%** [cite: 411, 548] | [cite_start]**<1.5s** [cite: 217] |

---

## 💡 为什么这个项目适合特斯拉/智能制造？

作为电池先进工艺工程师（视觉方向）的候选人，此项目证明了以下核心能力：
1. [cite_start]**视觉系统开发**：具备相机选型、打光调试及机械臂视觉随动的实战经验 [cite: 232, 245, 274, 321]。
2. [cite_start]**AI 工业落地**：熟悉 YOLO/ResNet 模型的训练、清洗、评估及在 MacOS(MPS) 环境下的硬件加速 [cite: 192, 411, 492, 498, 575]。
3. [cite_start]**工程标准化**：建立了标准化的图像标注规范与日志归档系统，符合制造流程智能化升级的需求 [cite: 452, 458, 486, 649, 660]。

---
