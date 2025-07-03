from PyQt5.QtWidgets import (
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QFileDialog,
    QPushButton,
    QLabel,
    QSlider,
    QDoubleSpinBox,
    QSpinBox,
    QGroupBox,
    QCheckBox,
    QProgressBar,
    QComboBox,
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
import cv2
import numpy as np


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Пакетный редактор изображений")
        self.setGeometry(100, 100, 1000, 700)
        self._create_ui()
        self._connect_signals()
        from core.image_processor import ImageProcessor

        self.processor = ImageProcessor()
        self.current_image = None
        self.current_image_path = None
        self.input_dir = None
        self.output_dir = None

    def _create_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout()
        main_widget.setLayout(layout)
        self.settings_panel = self._create_settings_panel()
        layout.addWidget(self.settings_panel, stretch=1)
        right_panel = QVBoxLayout()
        self.preview_widget = QLabel()
        self.preview_widget.setAlignment(Qt.AlignCenter)
        self.preview_widget.setMinimumSize(600, 400)
        right_panel.addWidget(self.preview_widget)
        control_panel = QHBoxLayout()
        self.load_image_btn = QPushButton("Загрузить изображение")
        self.select_input_btn = QPushButton("Выбрать исходную папку")
        self.select_output_btn = QPushButton("Выбрать выходную папку")
        self.process_btn = QPushButton("Обработать")
        self.process_btn.setEnabled(False)
        control_panel.addWidget(self.load_image_btn)
        control_panel.addWidget(self.select_input_btn)
        control_panel.addWidget(self.select_output_btn)
        control_panel.addWidget(self.process_btn)
        right_panel.addLayout(control_panel)
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        right_panel.addWidget(self.progress_bar)
        layout.addLayout(right_panel, stretch=2)

    def _create_settings_panel(self):
        panel = QWidget()
        layout = QVBoxLayout()
        panel.setLayout(layout)
        brightness_group = QGroupBox("Яркость")
        brightness_layout = QHBoxLayout()
        self.brightness_slider = QSlider(Qt.Horizontal)
        self.brightness_slider.setRange(-100, 100)
        self.brightness_spinbox = QSpinBox()
        self.brightness_spinbox.setRange(-100, 100)
        brightness_layout.addWidget(self.brightness_slider)
        brightness_layout.addWidget(self.brightness_spinbox)
        brightness_group.setLayout(brightness_layout)
        layout.addWidget(brightness_group)
        contrast_group = QGroupBox("Контраст")
        contrast_layout = QHBoxLayout()
        self.contrast_slider = QSlider(Qt.Horizontal)
        self.contrast_slider.setRange(0, 200)
        self.contrast_slider.setValue(100)
        self.contrast_spinbox = QDoubleSpinBox()
        self.contrast_spinbox.setRange(0.0, 2.0)
        self.contrast_spinbox.setValue(1.0)
        self.contrast_spinbox.setSingleStep(0.1)
        contrast_layout.addWidget(self.contrast_slider)
        contrast_layout.addWidget(self.contrast_spinbox)
        contrast_group.setLayout(contrast_layout)
        layout.addWidget(contrast_group)
        saturation_group = QGroupBox("Насыщенность")
        saturation_layout = QHBoxLayout()
        self.saturation_slider = QSlider(Qt.Horizontal)
        self.saturation_slider.setRange(0, 200)
        self.saturation_slider.setValue(100)
        self.saturation_spinbox = QDoubleSpinBox()
        self.saturation_spinbox.setRange(0.0, 2.0)
        self.saturation_spinbox.setValue(1.0)
        self.saturation_spinbox.setSingleStep(0.1)
        saturation_layout.addWidget(self.saturation_slider)
        saturation_layout.addWidget(self.saturation_spinbox)
        saturation_group.setLayout(saturation_layout)
        layout.addWidget(saturation_group)
        hue_group = QGroupBox("Цветовой баланс (HUE)")
        hue_layout = QHBoxLayout()
        self.hue_slider = QSlider(Qt.Horizontal)
        self.hue_slider.setRange(-180, 180)
        self.hue_spinbox = QSpinBox()
        self.hue_spinbox.setRange(-180, 180)
        hue_layout.addWidget(self.hue_slider)
        hue_layout.addWidget(self.hue_spinbox)
        hue_group.setLayout(hue_layout)
        layout.addWidget(hue_group)
        wb_group = QGroupBox("Точка белого")
        wb_layout = QVBoxLayout()
        red_layout = QHBoxLayout()
        red_layout.addWidget(QLabel("Красный:"))
        self.red_balance = QDoubleSpinBox()
        self.red_balance.setRange(0.1, 3.0)
        self.red_balance.setValue(1.0)
        self.red_balance.setSingleStep(0.1)
        red_layout.addWidget(self.red_balance)
        wb_layout.addLayout(red_layout)
        green_layout = QHBoxLayout()
        green_layout.addWidget(QLabel("Зеленый:"))
        self.green_balance = QDoubleSpinBox()
        self.green_balance.setRange(0.1, 3.0)
        self.green_balance.setValue(1.0)
        self.green_balance.setSingleStep(0.1)
        green_layout.addWidget(self.green_balance)
        wb_layout.addLayout(green_layout)
        blue_layout = QHBoxLayout()
        blue_layout.addWidget(QLabel("Синий:"))
        self.blue_balance = QDoubleSpinBox()
        self.blue_balance.setRange(0.1, 3.0)
        self.blue_balance.setValue(1.0)
        self.blue_balance.setSingleStep(0.1)
        blue_layout.addWidget(self.blue_balance)
        wb_layout.addLayout(blue_layout)
        wb_group.setLayout(wb_layout)
        layout.addWidget(wb_group)
        gamma_group = QGroupBox("Гамма-коррекция")
        gamma_layout = QHBoxLayout()
        self.gamma_slider = QSlider(Qt.Horizontal)
        self.gamma_slider.setRange(30, 300)
        self.gamma_slider.setValue(100)
        self.gamma_spinbox = QDoubleSpinBox()
        self.gamma_spinbox.setRange(0.3, 3.0)
        self.gamma_spinbox.setValue(1.0)
        self.gamma_spinbox.setSingleStep(0.1)
        gamma_layout.addWidget(self.gamma_slider)
        gamma_layout.addWidget(self.gamma_spinbox)
        gamma_group.setLayout(gamma_layout)
        layout.addWidget(gamma_group)
        sharpness_group = QGroupBox("Резкость")
        sharpness_layout = QHBoxLayout()
        self.sharpness_slider = QSlider(Qt.Horizontal)
        self.sharpness_slider.setRange(0, 100)
        self.sharpness_spinbox = QSpinBox()
        self.sharpness_spinbox.setRange(0, 100)
        sharpness_layout.addWidget(self.sharpness_slider)
        sharpness_layout.addWidget(self.sharpness_spinbox)
        sharpness_group.setLayout(sharpness_layout)
        layout.addWidget(sharpness_group)
        temp_group = QGroupBox("Цветовая температура")
        temp_layout = QHBoxLayout()
        self.temp_slider = QSlider(Qt.Horizontal)
        self.temp_slider.setRange(-100, 100)
        self.temp_spinbox = QSpinBox()
        self.temp_spinbox.setRange(-100, 100)
        temp_layout.addWidget(self.temp_slider)
        temp_layout.addWidget(self.temp_spinbox)
        temp_group.setLayout(temp_layout)
        layout.addWidget(temp_group)
        exposure_group = QGroupBox("Экспозиция")
        exposure_layout = QHBoxLayout()
        self.exposure_slider = QSlider(Qt.Horizontal)
        self.exposure_slider.setRange(-100, 100)
        self.exposure_slider.setValue(0)
        self.exposure_spinbox = QDoubleSpinBox()
        self.exposure_spinbox.setRange(-1.0, 1.0)
        self.exposure_spinbox.setValue(0.0)
        self.exposure_spinbox.setSingleStep(0.1)
        exposure_layout.addWidget(self.exposure_slider)
        exposure_layout.addWidget(self.exposure_spinbox)
        exposure_group.setLayout(exposure_layout)
        layout.addWidget(exposure_group)
        options_group = QGroupBox("Опции обработки")
        options_layout = QVBoxLayout()
        self.recursive_cb = QCheckBox("Рекурсивная обработка подпапок")
        self.recursive_cb.setChecked(True)
        self.output_format_cb = QComboBox()
        self.output_format_cb.addItems(["JPEG", "PNG"])
        options_layout.addWidget(self.recursive_cb)
        options_layout.addWidget(QLabel("Формат вывода:"))
        options_layout.addWidget(self.output_format_cb)
        options_group.setLayout(options_layout)
        layout.addWidget(options_group)
        self.reset_btn = QPushButton("Сбросить настройки")
        layout.addWidget(self.reset_btn)
        layout.addStretch()
        return panel

    def _connect_signals(self):
        self.load_image_btn.clicked.connect(self._load_reference_image)
        self.select_input_btn.clicked.connect(self._select_input_directory)
        self.select_output_btn.clicked.connect(self._select_output_directory)
        self.process_btn.clicked.connect(self._process_batch)
        self.reset_btn.clicked.connect(self._reset_settings)
        controls = [
            (self.brightness_slider, self.brightness_spinbox, lambda v: v, lambda v: v),
            (
                self.contrast_slider,
                self.contrast_spinbox,
                lambda v: v / 100,
                lambda v: int(v * 100),
            ),
            (
                self.saturation_slider,
                self.saturation_spinbox,
                lambda v: v / 100,
                lambda v: int(v * 100),
            ),
            (self.hue_slider, self.hue_spinbox, lambda v: v, lambda v: v),
            (
                self.gamma_slider,
                self.gamma_spinbox,
                lambda v: v / 100,
                lambda v: int(v * 100),
            ),
            (self.sharpness_slider, self.sharpness_spinbox, lambda v: v, lambda v: v),
            (self.temp_slider, self.temp_spinbox, lambda v: v, lambda v: v),
            (
                self.exposure_slider,
                self.exposure_spinbox,
                lambda v: v / 100,
                lambda v: int(v * 100),
            ),
        ]
        for slider, spinbox, to_spin, to_slide in controls:
            slider.valueChanged.connect(
                lambda v, s=spinbox, f=to_spin: s.setValue(f(v))
            )
            spinbox.valueChanged.connect(
                lambda v, s=slider, f=to_slide: s.setValue(f(v))
            )
            slider.valueChanged.connect(self._update_preview)
            spinbox.valueChanged.connect(self._update_preview)
        self.red_balance.valueChanged.connect(self._update_preview)
        self.green_balance.valueChanged.connect(self._update_preview)
        self.blue_balance.valueChanged.connect(self._update_preview)

    def _load_reference_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Выберите изображение", "", "Images (*.jpg *.jpeg *.png *.bmp)"
        )
        if file_path:
            self.current_image_path = file_path
            self.current_image = cv2.imread(file_path)
            if self.current_image is not None:
                self.current_image = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB)
                self._update_preview()
                self.process_btn.setEnabled(True)

    def _update_preview(self):
        if self.current_image is not None:
            settings = {
                "brightness": self.brightness_spinbox.value() / 50.0,
                "contrast": self.contrast_spinbox.value(),
                "saturation": self.saturation_spinbox.value(),
                "hue": self.hue_spinbox.value(),
                "white_balance": [
                    self.red_balance.value(),
                    self.green_balance.value(),
                    self.blue_balance.value(),
                ],
                "gamma": self.gamma_spinbox.value(),
                "sharpness": self.sharpness_spinbox.value(),
                "color_temp": self.temp_spinbox.value(),
                "exposure": self.exposure_spinbox.value(),
            }
            try:
                processed_img = self.processor.apply_settings(
                    self.current_image, settings
                )
                height, width, _ = processed_img.shape
                bytes_per_line = 3 * width
                q_img = QImage(
                    processed_img.data,
                    width,
                    height,
                    bytes_per_line,
                    QImage.Format_RGB888,
                )
                self.preview_widget.setPixmap(
                    QPixmap.fromImage(q_img).scaled(
                        self.preview_widget.width(),
                        self.preview_widget.height(),
                        Qt.KeepAspectRatio,
                        Qt.SmoothTransformation,
                    )
                )
            except Exception as e:
                print(f"Ошибка при обработке изображения: {str(e)}")

    def _select_input_directory(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Выберите исходную папку")
        if dir_path:
            self.input_dir = dir_path

    def _select_output_directory(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Выберите выходную папку")
        if dir_path:
            self.output_dir = dir_path

    def _process_batch(self):
        if hasattr(self, "input_dir") and hasattr(self, "output_dir"):
            settings = {
                "brightness": self.brightness_spinbox.value() / 50.0,
                "contrast": self.contrast_spinbox.value(),
                "saturation": self.saturation_spinbox.value(),
                "hue": self.hue_spinbox.value(),
                "white_balance": [
                    self.red_balance.value(),
                    self.green_balance.value(),
                    self.blue_balance.value(),
                ],
                "gamma": self.gamma_spinbox.value(),
                "sharpness": self.sharpness_spinbox.value(),
                "color_temp": self.temp_spinbox.value(),
                "exposure": self.exposure_spinbox.value(),
            }
            from core.batch_processor import BatchProcessor

            batch_processor = BatchProcessor(self.processor)
            self.progress_bar.setVisible(True)
            self.progress_bar.setRange(0, 0)
            try:
                batch_processor.process_directory(
                    input_dir=self.input_dir,
                    output_dir=self.output_dir,
                    settings=settings,
                    recursive=self.recursive_cb.isChecked(),
                    output_format=self.output_format_cb.currentText(),
                )
            except Exception as e:
                print(f"Ошибка при пакетной обработке: {str(e)}")
            finally:
                self.progress_bar.setVisible(False)

    def _reset_settings(self):
        self.brightness_slider.setValue(0)
        self.contrast_slider.setValue(100)
        self.saturation_slider.setValue(100)
        self.hue_slider.setValue(0)
        self.red_balance.setValue(1.0)
        self.green_balance.setValue(1.0)
        self.blue_balance.setValue(1.0)
        self.gamma_slider.setValue(100)
        self.sharpness_slider.setValue(0)
        self.temp_slider.setValue(0)
        self.exposure_slider.setValue(0)
        if self.current_image is not None:
            self._update_preview()
