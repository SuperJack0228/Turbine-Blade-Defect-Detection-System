#!/usr/bin/env python3
# ui/ui_main.py 版本1.2：完善训练模式

import sys
import os
import signal
import time
import re
import subprocess
import time 
from threading import Thread
from PyQt5.QtGui import QPixmap, QPainter, QPen, QFont
from PyQt5.QtCore import Qt

# 将 py_app/core 目录加入模块搜索路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CORE_DIR   = os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'core'))
if CORE_DIR not in sys.path:
    sys.path.insert(0, CORE_DIR)

# 导入核心接口
from api import classify_image, detect_image

# PyQt5 相关
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QPushButton, QVBoxLayout, QHBoxLayout,
    QLabel, QComboBox, QSpinBox,
    QProgressBar, QPlainTextEdit, QFileDialog,
    QStackedWidget
)
from PyQt5.QtGui import QPixmap, QPainter, QPen
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtCore import pyqtSignal


class MainWindow(QMainWindow):
    log_signal = pyqtSignal(str)
    def __init__(self):
        super().__init__()
        self.setWindowTitle('涡轮叶片缺陷检测系统')
        self.setFixedSize(800, 600)

        # 左侧菜单
        self.btnAnalyzeMode = QPushButton('分析模式')
        self.btnTrainMode   = QPushButton('训练模式')
        for btn in (self.btnAnalyzeMode, self.btnTrainMode):
            btn.setFixedHeight(40)

        menuLayout = QVBoxLayout()
        menuLayout.addWidget(self.btnAnalyzeMode)
        menuLayout.addWidget(self.btnTrainMode)
        menuLayout.addStretch()
        menuWidget = QWidget()
        menuWidget.setLayout(menuLayout)
        menuWidget.setFixedWidth(140)

        # 右侧页面堆栈
        self.stack = QStackedWidget()
        self.pageAnalyze = self.buildAnalyzePage()
        self.pageTrain   = self.buildTrainPage()
        self.stack.addWidget(self.pageAnalyze)
        self.stack.addWidget(self.pageTrain)

        self.log_signal.connect(self.textLog.appendPlainText)
        # 主布局
        # ------ 1. 主内容区（左菜单 + 页面堆栈） ------
        mainContentLayout = QHBoxLayout()
        mainContentLayout.addWidget(menuWidget)
        mainContentLayout.addWidget(self.stack, stretch=1)

# ------ 2. 底部footer ------
        footer = QWidget()
        footerLayout = QHBoxLayout(footer)
        footerLayout.setContentsMargins(15, 5, 15, 5)

        logoLabel = QLabel()
        logoPixmap = QPixmap("ui/school.png").scaledToHeight(40, Qt.SmoothTransformation)
        logoLabel.setPixmap(logoPixmap)
        footerLayout.addWidget(logoLabel)

        footerText = QLabel("版本号 v2.1 | 版权所有 © 2025 SuperJack")
        footerText.setStyleSheet("color: gray; font-size: 11px; margin-left: 20px;")
        footerLayout.addWidget(footerText)

        footerLayout.addStretch()

# ------ 3. 整体主布局 ------
        mainVLayout = QVBoxLayout()
        mainVLayout.addLayout(mainContentLayout)  # 只加一次
        mainVLayout.addWidget(footer)             # 底部footer

        container = QWidget()
        container.setLayout(mainVLayout)
        self.setCentralWidget(container)



        # 切换事件
        self.btnAnalyzeMode.clicked.connect(lambda: self.stack.setCurrentWidget(self.pageAnalyze))
        self.btnTrainMode.clicked.connect(lambda: self.stack.setCurrentWidget(self.pageTrain))

        # 训练控制属性
        self.train_proc = None
        self.paused = False

    def buildAnalyzePage(self):
        # 保持原来分析页面
        page = QWidget()
        self.comboPart  = QComboBox(); self.comboPart.addItems(['叶轮','中间壳'])
        self.comboMode  = QComboBox(); self.comboMode.addItems(['分类','检测'])
        self.comboMode.currentTextChanged.connect(self.onModeChanged)
        self.btnImport  = QPushButton('导入图片')
        self.btnAnalyze = QPushButton('开始分析'); self.btnAnalyze.setEnabled(False)
        self.labelImage = QLabel('暂无图片'); self.labelImage.setFixedSize(700,350); self.labelImage.setAlignment(Qt.AlignCenter); self.labelImage.setStyleSheet('border:1px solid gray;')
        self.progress   = QProgressBar()
        self.textResult = QPlainTextEdit(); self.textResult.setReadOnly(True)
        self.btnSaveImage = QPushButton('保存结果图片')
        self.btnSaveImage.setEnabled(False)  # 没有结果前不能点
        self.btnSaveImage.clicked.connect(self.saveResultImage)
        hl = QHBoxLayout(); hl.addWidget(self.comboPart); hl.addWidget(self.comboMode); hl.addStretch(); hl.addWidget(self.btnImport); hl.addWidget(self.btnAnalyze);hl.addWidget(self.btnSaveImage)
        layout = QVBoxLayout(page); layout.addLayout(hl); layout.addWidget(self.labelImage); layout.addWidget(self.progress); layout.addWidget(self.textResult)
        self.btnImport.clicked.connect(self.importImage)
        self.btnAnalyze.clicked.connect(self.startAnalysis)
        # 初始化控件显隐
        self.onModeChanged(self.comboMode.currentText())
        return page
    
    def onModeChanged(self, mode):
        if mode == '分类':
            self.comboPart.show()
        else:  # 检测
            self.comboPart.hide()

    def saveResultImage(self):
        if not hasattr(self, 'currentPixmapWithBoxes'):
            self.textResult.appendPlainText("没有检测结果图片可保存")
            return
        filePath, _ = QFileDialog.getSaveFileName(self, "保存检测结果图片", "", "PNG Image (*.png);;JPEG Image (*.jpg *.jpeg)")
        if filePath:
            self.currentPixmapWithBoxes.save(filePath)
            self.textResult.appendPlainText(f"已保存图片到: {filePath}")

    def buildTrainPage(self):
        # 完整训练页面布局
        page = QWidget()
        self.comboDatasetSource = QComboBox()
        self.comboDatasetSource.addItems([
            '默认数据集（目标检测）',
            '默认数据集（分类）',
            '导入数据集'
        ])
        # 检测参数区的 epoch
        labelEpochDet = QLabel('迭代次数：')
        self.spinEpochDet = QSpinBox(); self.spinEpochDet.setRange(1,100); self.spinEpochDet.setValue(15)

        # 分类参数区的 epoch
        labelEpochCls = QLabel('迭代次数：')
        self.spinEpochCls = QSpinBox(); self.spinEpochCls.setRange(1,100); self.spinEpochCls.setValue(15)


            # 目标检测专用参数
        labelBatch = QLabel('批次大小：')
        self.comboBatch = QComboBox()
        self.comboBatch.addItems(['4', '8', '16'])  # 只允许这三种
        self.comboBatch.setCurrentText('8')         # 默认8

        labelYolo = QLabel('YOLO模型：')
        self.comboYolo = QComboBox(); self.comboYolo.addItems(['yolov8n', 'yolov8s', 'yolov8m', 'yolov8l'])

            # 导入数据集按钮
        self.btnImportData = QPushButton('导入数据集')
        self.datasetLabel  = QLabel('未选择数据集')
            # 检测参数区
        self.widgetDetParam = QWidget()
        detMainLayout = QVBoxLayout(self.widgetDetParam)

# 行1：epoch
        rowEpoch = QHBoxLayout()
        rowEpoch.addStretch()
        rowEpoch.addWidget(labelEpochDet)
        rowEpoch.addWidget(self.spinEpochDet)
        rowEpoch.addStretch()
        detMainLayout.addLayout(rowEpoch)

# 行2：batch和yolo
        rowBatchYolo = QHBoxLayout()
        rowBatchYolo.addStretch()
        rowBatchYolo.addWidget(labelBatch)
        rowBatchYolo.addWidget(self.comboBatch)
        rowBatchYolo.addWidget(labelYolo)
        rowBatchYolo.addWidget(self.comboYolo)
        rowBatchYolo.addStretch()
        detMainLayout.addLayout(rowBatchYolo)


            # 分类参数区
        self.widgetClsParam = QWidget()
        clsLayout = QHBoxLayout(self.widgetClsParam)
        clsLayout.addWidget(labelEpochCls)
        clsLayout.addWidget(self.spinEpochCls)

            # 导入数据集区
        self.widgetImport = QWidget()
        importLayout = QHBoxLayout(self.widgetImport)
        importLayout.addWidget(self.btnImportData)
        importLayout.addWidget(self.datasetLabel)

        mainLayout = QVBoxLayout(page)
        mainLayout.addWidget(self.comboDatasetSource)      # 数据集来源下拉框
        mainLayout.addWidget(self.widgetDetParam)          # 检测参数区
        mainLayout.addWidget(self.widgetClsParam)          # 分类参数区
        mainLayout.addWidget(self.widgetImport)            # 导入数据集区
        

# 训练和中止按钮
        btnLayout = QHBoxLayout()
        self.btnTrain = QPushButton('开始训练'); self.btnTrain.setEnabled(True)
        self.btnStop  = QPushButton('中止'); self.btnStop.setEnabled(False)
        btnLayout.addWidget(self.btnTrain)
        btnLayout.addWidget(self.btnStop)
        mainLayout.addLayout(btnLayout)

# 进度条和日志
        self.progressT = QProgressBar()
        self.textLog   = QPlainTextEdit(); self.textLog.setReadOnly(True)
        self.progressT.setMaximumWidth(300)
        self.progressT.setMinimumWidth(150)
        self.progressT.setFixedHeight(18)
        self.progressT.setMinimum(0)
        self.progressT.setMaximum(100)

        self.labelPct = QLabel("0%")
        progressLayout = QHBoxLayout()
        progressLayout.addWidget(self.progressT)
        progressLayout.addWidget(self.labelPct)
        progressLayout.addStretch()
        mainLayout.addLayout(progressLayout)
        mainLayout.addWidget(self.progressT)
        mainLayout.addWidget(self.textLog)
        page.setLayout(mainLayout)

        # 事件绑定
        self.comboDatasetSource.currentTextChanged.connect(self.onDatasetSourceChanged)
        self.btnImportData.clicked.connect(self.importDataset)
        self.btnTrain.clicked.connect(self.startTraining)
        self.btnStop.clicked.connect(self.stopTraining)
        self.labelElapsedTime = QLabel("已用时: 0秒")
        mainLayout.addWidget(self.labelElapsedTime)
# 计时器！加到训练界面底部布局
        

# 初始化界面参数显隐
        self.onDatasetSourceChanged(self.comboDatasetSource.currentText())
        return page

    def onDatasetSourceChanged(self, text):
        if text == '默认数据集（目标检测）':
            self.widgetDetParam.show()
            self.widgetClsParam.hide()
            self.widgetImport.hide()
            self.btnTrain.setEnabled(True)
            self.btnStop.setEnabled(True)
            self.trainMode = 'detection'
            self.detDatasetPath = '/Users/superjack/Downloads'
        elif text == '默认数据集（分类）':
            self.trainMode = 'classification'
            self.clsDatasetPath = '/Users/superjack/Downloads/Data/T-LV'
            self.widgetDetParam.hide()
            self.widgetClsParam.show()
            self.widgetImport.hide()
            self.btnTrain.setEnabled(True)
            self.btnStop.setEnabled(True)
        else:  # 导入数据集
            self.widgetDetParam.hide()
            self.widgetClsParam.hide()
            self.widgetImport.show()
            self.btnTrain.setEnabled(False)
            self.btnStop.setEnabled(False)

    def importImage(self):
        path,_ = QFileDialog.getOpenFileName(self,'选择图片','','Images (*.png *.jpg *.jpeg)')
        if not path: return
        self.currentImagePath=path
        pix=QPixmap(path).scaled(self.labelImage.size(),Qt.KeepAspectRatio,Qt.SmoothTransformation)
        self.labelImage.setPixmap(pix)
        self.btnAnalyze.setEnabled(True)
        self.progress.setValue(0)
        self.textResult.clear()

    def importDataset(self):
        dir_path=QFileDialog.getExistingDirectory(self,'选择数据集文件夹')
        if not dir_path: return
        self.datasetPath=dir_path
        self.datasetLabel.setText(os.path.basename(dir_path))
        self.btnTrain.setEnabled(True)
        self.btnStop.setEnabled(False)
        self.progressT.setValue(0)
        self.textLog.clear()

    def startAnalysis(self):
        self.btnImport.setEnabled(False)
        self.btnAnalyze.setEnabled(False)
        self.progress.setValue(50)
        mode = self.comboMode.currentText()
        txt = ""
        try:
            if mode == '分类':
                res = classify_image(self.currentImagePath)
                txt = f"{res['defect']}（置信度: {res['confidence']:.2f}）"
                pix = QPixmap(self.currentImagePath).scaled(self.labelImage.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.labelImage.setPixmap(pix)
                self.btnSaveImage.setEnabled(False)
            else:  # 检测
                results = detect_image(self.currentImagePath)
                if not results:
                    txt = "未检测到缺陷"
                    pix = QPixmap(self.currentImagePath).scaled(self.labelImage.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                    self.labelImage.setPixmap(pix)
                    self.btnSaveImage.setEnabled(False)
                    self.currentPixmapWithBoxes = None
                else:
                    lines = []
                    for i, obj in enumerate(results, 1):
                        bbox = obj['bbox']
                        label = obj['label']
                        conf = obj['confidence']
                        lines.append(
                            f"{i}. {label} | 置信度: {conf:.2f}\n   位置: [{bbox[0]:.1f}, {bbox[1]:.1f}, {bbox[2]:.1f}, {bbox[3]:.1f}]"
                        )
                txt = "\n".join(lines)
                pix = self.draw_boxes_on_pixmap(self.currentImagePath, results)
                self.labelImage.setPixmap(pix)
                self.currentPixmapWithBoxes = pix
                self.btnSaveImage.setEnabled(True)
        except Exception as e:
            txt = f"推理出错：{e}"
            self.btnSaveImage.setEnabled(False)
        finally:
        # 统一刷新文本框、进度条和按钮状态
            self.textResult.setPlainText(txt)
            self.progress.setValue(100)
            self.btnImport.setEnabled(True)
            self.btnAnalyze.setEnabled(True)

    def updateElapsedTime(self):
        self.elapsedSeconds = int(time.time() - self.train_start_time)
        self.labelElapsedTime.setText(f"已用时: {self.elapsedSeconds}秒")

    def startTraining(self):
        import time, subprocess, re
        from threading import Thread

        self.train_start_time = time.time()
        self.progressT.setValue(0)
        self.btnTrain.setEnabled(False)
        self.btnStop.setEnabled(True)
        self.textLog.appendPlainText("==== 开始训练 ====")
        #计时器
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.updateElapsedTime)
        self.timer.start(1000)
        self.elapsedSeconds = 0
        self.labelElapsedTime.setText("已用时: 0秒")


        if getattr(self, "trainMode", None) == "classification":
            dataset_path = self.clsDatasetPath
            train_dir = os.path.join(self.clsDatasetPath, "train_dataroot")
            test_dir = os.path.join(self.clsDatasetPath, "test_dataroot")
            epochs = self.spinEpochCls.value() if hasattr(self, 'spinEpochCls') else self.spinEpoch.value()
            self.textLog.appendPlainText(f'[分类] 数据集: {dataset_path} | 轮数: {epochs}')
            script = '/Users/superjack/test/turbo_detector/py_app/train/train_classifier.py'

            batch_pattern = re.compile(r"Batch\s*\[(\d+)/(\d+)\]")
            epoch_pattern = re.compile(r"Epoch\s*\[(\d+)/(\d+)\].*Loss: ([0-9.]+).*Acc: ([0-9.]+)%")

            best_acc = 0.0

            cmd = [
                sys.executable, script,
                '--train-dir', train_dir,
                '--test-dir', test_dir,
                '--epochs', str(epochs)
            ]
            pattern = re.compile(r"Epoch\s*\[(\d+)/(\d+)\].*Loss: ([0-9.]+), Accuracy?: ([0-9.]+)%")
        elif getattr(self, "trainMode", None) == "detection":
            dataset_path = self.detDatasetPath
        # 检测模式参数
            epochs = self.spinEpochDet.value() if hasattr(self, 'spinEpochDet') else self.spinEpoch.value()
            batch = int(self.comboBatch.currentText())
            yolo_model = self.comboYolo.currentText()
            self.textLog.appendPlainText(f'[检测] 数据集: {dataset_path} | 轮数: {epochs} | batch: {batch} | YOLO模型: {yolo_model}')
            script = os.path.abspath(os.path.join(__file__, '..', '..', 'train', 'train_detector.py'))
            cmd = [
                sys.executable, script,
                '--data-dir', dataset_path,
                '--epochs', str(epochs),
                '--batch', str(batch),
                '--model', yolo_model
            ]
            pattern = re.compile(r"Epoch\s*\[(\d+)/(\d+)\].*")  # 后续按你的检测脚本实际输出调整
        else:
            self.textLog.appendPlainText("未选择训练模式或数据集，请先在上方选择")
            self.btnTrain.setEnabled(True)
            self.btnStop.setEnabled(False)
            return

        def task():
            best_acc = 0.0
            try:
                proc = subprocess.Popen(
                    cmd, cwd=os.path.dirname(script),
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, bufsize=1
                )
                self.train_proc = proc
            except Exception as e:
                QTimer.singleShot(0, lambda: self.textLog.appendPlainText(f'训练启动失败: {e}'))
                QTimer.singleShot(0, lambda: self.btnTrain.setEnabled(True))
                QTimer.singleShot(0, lambda: self.btnStop.setEnabled(False))
                return

            for line in proc.stdout:
                t = line.rstrip()
                self.log_signal.emit(t)  # 只发信号

                m = batch_pattern.search(line)
                if m:
                    batch_idx = int(m.group(1))
                    batch_total = int(m.group(2))
                    pct = int(batch_idx / batch_total * 100)
                    print(f"[SET PROGRESS] {pct}%")
                    self.progressT.setValue(pct)
                    self.labelPct.setText(f"{pct}%")
                    self.progressT.show()

                m2 = epoch_pattern.search(line)
                if m2:
                    ep = int(m2.group(1))
                    tot = int(m2.group(2))
                    loss = float(m2.group(3))
                    acc = float(m2.group(4))
                    self.log_signal.emit(f'第{ep}轮: Loss={loss:.4f}, Acc={acc:.2f}%')


            code = proc.wait()
            train_time = time.time() - self.train_start_time
            QTimer.singleShot(0, self.timer.stop)
            if code == 0:
                msg = (f'训练完成，最高准确率={best_acc:.2f}%，总用时={train_time:.1f}秒'
                    if self.trainMode == 'classification'
                    else f'训练完成，总用时={train_time:.1f}秒')
            else:
                msg = f'训练异常退出，返回码={code}'
            QTimer.singleShot(0, lambda: self.textLog.appendPlainText(msg))
            QTimer.singleShot(0, lambda: self.btnTrain.setEnabled(True))
            QTimer.singleShot(0, lambda: self.btnStop.setEnabled(False))

        Thread(target=task, daemon=True).start()
  
    def draw_boxes_on_pixmap(self, image_path, detections):
    # 原图按labelImage大小缩放，便于画框
        target_size = self.labelImage.size()
        pixmap = QPixmap(image_path).scaled(target_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        painter = QPainter(pixmap)
        pen = QPen(Qt.red)
        pen.setWidth(2)
        painter.setPen(pen)
        font = QFont()
        font.setPointSize(12)
        painter.setFont(font)
        # 计算缩放比例（避免画框跑偏）
        original_pix = QPixmap(image_path)
        scale_x = pixmap.width() / original_pix.width()
        scale_y = pixmap.height() / original_pix.height()
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            x1, x2 = x1 * scale_x, x2 * scale_x
            y1, y2 = y1 * scale_y, y2 * scale_y
            painter.drawRect(int(x1), int(y1), int(x2 - x1), int(y2 - y1))
            painter.drawText(int(x1), int(y1) - 5, f"{det['label']} {det['confidence']:.2f}")
        painter.end()
        return pixmap

    def stopTraining(self):
        if hasattr(self, 'train_proc') and self.train_proc and self.train_proc.poll() is None:
            self.train_proc.kill()
            self.textLog.appendPlainText('训练已中止')
        if hasattr(self, 'timer'):
            self.timer.stop()
        self.btnTrain.setEnabled(True)
        self.btnStop.setEnabled(False)

    def resetTrainButtons(self):
        QTimer.singleShot(0,lambda:self.btnImportData.setEnabled(True))
        QTimer.singleShot(0,lambda:self.btnTrain.setEnabled(True))
        QTimer.singleShot(0,lambda:self.btnStop.setEnabled(False))
        self.paused=False

if __name__=='__main__':
    app=QApplication(sys.argv)
    win=MainWindow()
    win.show()
    sys.exit(app.exec_())
