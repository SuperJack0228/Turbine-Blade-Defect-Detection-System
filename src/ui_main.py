# ui_main.py
import os, sys
from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QLabel, QFileDialog,
    QVBoxLayout, QHBoxLayout, QComboBox, QMessageBox, QProgressBar
)
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt, QTimer
from model.classifier import BladeClassifier
from model.detector   import BladeDetector

class MainUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("涡轮叶片缺陷检测系统")
        self.setFixedSize(600, 760)

        self.image_path = None

        # 1. 图像预览区
        self.img_label = QLabel("未选择图片", self)
        self.img_label.setFixedSize(560, 360)
        self.img_label.setStyleSheet("border: 1px solid gray;")
        self.img_label.setAlignment(Qt.AlignCenter)

        # 2. 导入图像按钮
        self.btn_import = QPushButton("导入图像")
        self.btn_import.clicked.connect(self.open_image)

        # 3. 缺陷类型下拉（叶轮 vs 中间壳）
        self.combo_type = QComboBox()
        self.combo_type.addItems(["叶轮", "中间壳"])

        # 4. 算法模式下拉（分类 vs 检测）
        self.combo_mode = QComboBox()
        self.combo_mode.addItems(["分类", "目标检测"])

        # 5. 开始分析按钮
        self.btn_run = QPushButton("开始分析")
        self.btn_run.clicked.connect(self.start_analysis)
        self.btn_run.setEnabled(False)

        # 6. 查看训练曲线按钮
        self.btn_curve = QPushButton("查看训练曲线")
        self.btn_curve.clicked.connect(self.show_curve)

        # 7. 进度条（默认隐藏）
        self.progress = QProgressBar()
        self.progress.setFixedWidth(560)
        self.progress.setValue(0)
        self.progress.setVisible(False)

        # 8. 结果展示区
        self.result_label = QLabel("结果将在此显示")
        self.result_label.setWordWrap(True)
        self.result_label.setFixedHeight(60)
        self.result_label.setStyleSheet("border: 1px solid gray; padding: 5px;")

        # 布局：顶部按钮和下拉
        top_layout = QHBoxLayout()
        top_layout.addWidget(self.btn_import)
        top_layout.addWidget(self.combo_type)
        top_layout.addWidget(self.combo_mode)
        top_layout.addWidget(self.btn_run)
        top_layout.addWidget(self.btn_curve)

        # 主布局
        main_layout = QVBoxLayout()
        main_layout.addWidget(self.img_label)
        main_layout.addLayout(top_layout)
        main_layout.addWidget(self.progress)
        main_layout.addWidget(self.result_label)
        self.setLayout(main_layout)

    def open_image(self):
        """导入图像并预览"""
        fname, _ = QFileDialog.getOpenFileName(
            self, "选择图片", "", "Image files (*.jpg *.png)"
        )
        if not fname:
            return
        self.image_path = fname
        pix = QPixmap(fname).scaled(
            self.img_label.width(), self.img_label.height(), Qt.KeepAspectRatio
        )
        self.img_label.setPixmap(pix)
        self.btn_run.setEnabled(True)
        self.result_label.setText("已选择图片，点击“开始分析”")

    def start_analysis(self):
        """点击开始，显示进度条，1s 后真正执行分析"""
        if not self.image_path:
            QMessageBox.warning(self, "错误", "请先导入图片")
            return

        # 禁用按钮，重置进度
        self.btn_run.setEnabled(False)
        self.progress.setValue(0)
        self.progress.setVisible(True)
        self.result_label.setText("分析中，请稍候…")
        QApplication.processEvents()

        # 半秒后进度到 50%
        QTimer.singleShot(500, lambda: self.progress.setValue(50))
        # 一秒后执行真正分析
        QTimer.singleShot(1000, self.run_analysis)

    def run_analysis(self):
        """真正的分类或检测逻辑"""
        try:
            mode = self.combo_mode.currentText()
            # 加载分类或检测模型
            if mode == "分类":
                clf = BladeClassifier(weights_path="output/best_model.pth")
                res = clf.predict(self.image_path)
                txt = f"【{self.combo_type.currentText()} 分类结果】{res['defect']}，置信度：{res['confidence']}"
                self.result_label.setText(txt)
            else:
                det = BladeDetector(model_name="yolov8s.pt")
                res = det.predict(
                    self.image_path, save_img=True, out_dir="output"
                )
                out_img = os.path.join(
                    "output", "det_" + os.path.basename(self.image_path)
                )
                if os.path.exists(out_img):
                    pix = QPixmap(out_img).scaled(
                        self.img_label.width(), self.img_label.height(), Qt.KeepAspectRatio
                    )
                    self.img_label.setPixmap(pix)
                txt = f"【{self.combo_type.currentText()} 目标检测】检测到 {len(res)} 个缺陷"
                self.result_label.setText(txt)
        except Exception as e:
            QMessageBox.critical(self, "分析失败", str(e))
        finally:
            # 隐藏进度条，恢复按钮
            self.progress.setVisible(False)
            self.btn_run.setEnabled(True)

    def show_curve(self):
        """弹窗展示训练曲线"""
        curve_path = "output/training_loss.png"
        if not os.path.exists(curve_path):
            QMessageBox.warning(self, "未找到", "请先完成训练并生成曲线")
            return
        dlg = QWidget()
        dlg.setWindowTitle("训练损失曲线")
        dlg.setFixedSize(600, 400)
        lbl = QLabel(dlg)
        pix = QPixmap(curve_path).scaled(580, 380, Qt.KeepAspectRatio)
        lbl.setPixmap(pix)
        dlg.show()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = MainUI()
    win.show()
    sys.exit(app.exec_())
