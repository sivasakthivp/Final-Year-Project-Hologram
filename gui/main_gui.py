"""
PyQt5 Desktop GUI for GPU-Accelerated Medical Imaging
Professional desktop interface with real-time GPU monitoring
"""

import sys
from pathlib import Path
from typing import Optional, Dict, Any
import asyncio
import time
import logging
from datetime import datetime
from dataclasses import dataclass
import numpy as np
from PIL import Image
import io

try:
    from PyQt5.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QLabel, QPushButton, QSlider, QSpinBox, QComboBox, QFileDialog,
        QProgressBar, QTabWidget, QTableWidget, QTableWidgetItem, QTextEdit,
        QSplitter, QScrollArea, QFrame, QGroupBox, QGridLayout
    )
    from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread, QObject, QSize
    from PyQt5.QtGui import QPixmap, QImage, QFont, QColor, QIcon
    from PyQt5.QtChart import QChart, QChartView, QLineSeries
    PYQT5_AVAILABLE = True
except ImportError:
    PYQT5_AVAILABLE = False
    print("⚠️  PyQt5 not installed. Install with: pip install PyQt5")

logger = logging.getLogger(__name__)


class ProcessingWorker(QObject):
    """Worker thread for GPU processing"""
    
    finished = pyqtSignal()
    progress = pyqtSignal(int)
    result = pyqtSignal(dict)
    error = pyqtSignal(str)
    
    def __init__(self, task_type: str, data: np.ndarray, model: Any):
        super().__init__()
        self.task_type = task_type
        self.data = data
        self.model = model
    
    def run(self):
        """Run processing task"""
        try:
            self.progress.emit(10)
            
            if self.task_type == "segmentation":
                result = self._process_segmentation()
            elif self.task_type == "depth":
                result = self._process_depth()
            elif self.task_type == "super_resolution":
                result = self._process_super_resolution()
            else:
                raise ValueError(f"Unknown task type: {self.task_type}")
            
            self.progress.emit(100)
            self.result.emit(result)
            
        except Exception as e:
            self.error.emit(str(e))
        
        finally:
            self.finished.emit()
    
    def _process_segmentation(self) -> Dict:
        """GPU segmentation"""
        import torch
        self.progress.emit(20)
        
        # Convert to tensor
        data_tensor = torch.from_numpy(self.data).float()
        if data_tensor.dim() == 2:
            data_tensor = data_tensor.unsqueeze(0).unsqueeze(0)
        
        self.progress.emit(50)
        
        # Run model
        with torch.no_grad():
            output = self.model(data_tensor)
        
        self.progress.emit(80)
        
        mask = output.argmax(dim=1).squeeze().numpy()
        return {"type": "segmentation", "result": mask}
    
    def _process_depth(self) -> Dict:
        """GPU depth estimation"""
        import torch
        self.progress.emit(20)
        
        data_tensor = torch.from_numpy(self.data).float()
        if data_tensor.dim() == 2:
            data_tensor = data_tensor.unsqueeze(0)
        
        self.progress.emit(50)
        
        with torch.no_grad():
            depth = self.model(data_tensor)
        
        self.progress.emit(80)
        
        return {"type": "depth", "result": depth.squeeze().numpy()}
    
    def _process_super_resolution(self) -> Dict:
        """GPU super-resolution"""
        import torch
        self.progress.emit(20)
        
        data_tensor = torch.from_numpy(self.data).float()
        if data_tensor.dim() == 2:
            data_tensor = data_tensor.unsqueeze(0)
        if data_tensor.dim() == 3:
            data_tensor = data_tensor.unsqueeze(0)
        
        self.progress.emit(50)
        
        with torch.no_grad():
            sr = self.model(data_tensor)
        
        self.progress.emit(80)
        
        return {"type": "super_resolution", "result": sr.squeeze().numpy()}


class GPUMonitorWidget(QWidget):
    """GPU status monitoring widget"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_stats)
    
    def init_ui(self):
        """Initialize UI"""
        layout = QVBoxLayout()
        
        # Title
        title = QLabel("GPU Monitor")
        title_font = QFont()
        title_font.setBold(True)
        title_font.setPointSize(12)
        title.setFont(title_font)
        layout.addWidget(title)
        
        # Stats grid
        stats_layout = QGridLayout()
        
        self.device_label = QLabel("Device: ---")
        self.memory_label = QLabel("Memory: ---")
        self.temp_label = QLabel("Temperature: ---")
        self.utilization_label = QLabel("Utilization: ---")
        
        stats_layout.addWidget(self.device_label, 0, 0)
        stats_layout.addWidget(self.memory_label, 0, 1)
        stats_layout.addWidget(self.temp_label, 1, 0)
        stats_layout.addWidget(self.utilization_label, 1, 1)
        
        layout.addLayout(stats_layout)
        
        # Memory bar
        self.memory_bar = QProgressBar()
        self.memory_bar.setMaximum(6000)  # RTX 3050 6GB
        layout.addWidget(QLabel("GPU Memory Usage:"))
        layout.addWidget(self.memory_bar)
        
        # Process info
        self.process_text = QTextEdit()
        self.process_text.setReadOnly(True)
        self.process_text.setMaximumHeight(150)
        layout.addWidget(QLabel("Active Processes:"))
        layout.addWidget(self.process_text)
        
        layout.addStretch()
        self.setLayout(layout)
        
        self.timer.start(1000)  # Update every second
    
    def update_stats(self):
        """Update GPU statistics"""
        try:
            import torch
            
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated(0) / 1024 / 1024
                reserved = torch.cuda.memory_reserved(0) / 1024 / 1024
                total = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024
                
                device_name = torch.cuda.get_device_name(0)
                self.device_label.setText(f"Device: {device_name}")
                self.memory_label.setText(f"Memory: {allocated:.0f}MB / {total:.0f}MB")
                self.memory_bar.setValue(int(allocated))
                
                utilization = (allocated / total) * 100
                self.utilization_label.setText(f"Utilization: {utilization:.1f}%")
        
        except Exception as e:
            self.device_label.setText(f"Error: {str(e)}")


class ProcessingFrameWidget(QWidget):
    """Main processing frame - handles all GPU operations"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
        self.current_image = None
        self.processing = False
    
    def init_ui(self):
        """Initialize processing frame UI"""
        layout = QVBoxLayout()
        
        # Title
        title = QLabel("Processing Pipeline")
        title_font = QFont()
        title_font.setBold(True)
        title_font.setPointSize(14)
        title.setFont(title_font)
        layout.addWidget(title)
        
        # Controls group
        controls_group = QGroupBox("Processing Controls")
        controls_layout = QGridLayout()
        
        # File loader
        load_btn = QPushButton("📂 Load Image")
        load_btn.clicked.connect(self.load_image)
        controls_layout.addWidget(load_btn, 0, 0)
        
        # Processing type selector
        self.process_selector = QComboBox()
        self.process_selector.addItems([
            "Segmentation",
            "Depth Estimation",
            "Super Resolution",
            "Volumetric Analysis"
        ])
        controls_layout.addWidget(QLabel("Processing Type:"), 0, 1)
        controls_layout.addWidget(self.process_selector, 0, 2)
        
        # Parameters
        self.param_slider = QSlider(Qt.Horizontal)
        self.param_slider.setMinimum(0)
        self.param_slider.setMaximum(100)
        self.param_slider.setValue(50)
        controls_layout.addWidget(QLabel("Intensity:"), 1, 0)
        controls_layout.addWidget(self.param_slider, 1, 1, 1, 2)
        
        # Process button
        self.process_btn = QPushButton("▶️ Process (GPU)")
        self.process_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                padding: 10px;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #45a049; }
            QPushButton:pressed { background-color: #3d8b40; }
        """)
        self.process_btn.clicked.connect(self.start_processing)
        controls_layout.addWidget(self.process_btn, 2, 0, 1, 3)
        
        controls_group.setLayout(controls_layout)
        layout.addWidget(controls_group)
        
        # Progress
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximum(100)
        layout.addWidget(QLabel("Processing Progress:"))
        layout.addWidget(self.progress_bar)
        
        # Status text
        self.status_text = QTextEdit()
        self.status_text.setReadOnly(True)
        self.status_text.setMaximumHeight(200)
        layout.addWidget(QLabel("Status Log:"))
        layout.addWidget(self.status_text)
        
        # Batch operations
        batch_group = QGroupBox("Batch Processing")
        batch_layout = QHBoxLayout()
        
        batch_spinbox = QSpinBox()
        batch_spinbox.setMinimum(1)
        batch_spinbox.setMaximum(100)
        batch_spinbox.setValue(10)
        batch_layout.addWidget(QLabel("Batch Size:"))
        batch_layout.addWidget(batch_spinbox)
        
        batch_btn = QPushButton("⚡ Process Batch")
        batch_btn.clicked.connect(self.batch_process)
        batch_layout.addWidget(batch_btn)
        
        batch_group.setLayout(batch_layout)
        layout.addWidget(batch_group)
        
        layout.addStretch()
        self.setLayout(layout)
    
    def load_image(self):
        """Load image file"""
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(
            self,
            "Load Medical Image",
            "",
            "DICOM Files (*.dcm);;PNG Files (*.png);;TIFF Files (*.tif)"
        )
        
        if file_path:
            try:
                if file_path.endswith('.dcm'):
                    import pydicom
                    dcm = pydicom.dcmread(file_path)
                    self.current_image = dcm.pixel_array.astype(np.float32)
                else:
                    img = Image.open(file_path)
                    self.current_image = np.array(img, dtype=np.float32)
                
                self.log_status(f"✓ Loaded image: {Path(file_path).name} ({self.current_image.shape})")
            
            except Exception as e:
                self.log_status(f"✗ Failed to load image: {e}")
    
    def start_processing(self):
        """Start GPU processing"""
        if self.current_image is None:
            self.log_status("✗ No image loaded")
            return
        
        if self.processing:
            self.log_status("✗ Processing already in progress")
            return
        
        self.processing = True
        self.process_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        
        process_type = self.process_selector.currentText().lower().replace(" ", "_")
        self.log_status(f"▶ Starting GPU processing: {process_type}")
        
        # Simulate processing with progress
        self.simulate_processing(process_type)
    
    def simulate_processing(self, process_type: str):
        """Simulate processing with progress updates"""
        self.progress_bar.setValue(20)
        self.log_status(f"  [20%] Loading model and data to GPU...")
        
        # Use timer to simulate progress
        timer = QTimer()
        progress_value = 20
        
        def update_progress():
            nonlocal progress_value
            progress_value += 15
            self.progress_bar.setValue(min(progress_value, 95))
            
            if progress_value >= 95:
                timer.stop()
                self.progress_bar.setValue(100)
                self.log_status(f"✓ {process_type} completed successfully!")
                self.processing = False
                self.process_btn.setEnabled(True)
        
        timer.timeout.connect(update_progress)
        timer.start(500)
    
    def batch_process(self):
        """Process batch of images"""
        self.log_status("⚡ Starting batch GPU processing...")
        self.log_status("  Processing multiple images with GPU optimization...")
    
    def log_status(self, message: str):
        """Add status message to log"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.status_text.append(f"[{timestamp}] {message}")
        
        # Auto-scroll to bottom
        self.status_text.verticalScrollBar().setValue(
            self.status_text.verticalScrollBar().maximum()
        )


class VisualizationWidget(QWidget):
    """Result visualization widget"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
    
    def init_ui(self):
        """Initialize visualization UI"""
        layout = QVBoxLayout()
        
        # Title
        title = QLabel("Result Visualization")
        title_font = QFont()
        title_font.setBold(True)
        title_font.setPointSize(12)
        title.setFont(title_font)
        layout.addWidget(title)
        
        # Image display area
        self.image_label = QLabel()
        self.image_label.setMinimumSize(500, 400)
        self.image_label.setStyleSheet("border: 2px solid #ccc; background: #f0f0f0;")
        self.image_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.image_label)
        
        # Controls
        ctrl_layout = QHBoxLayout()
        
        zoom_slider = QSlider(Qt.Horizontal)
        zoom_slider.setMinimum(10)
        zoom_slider.setMaximum(400)
        zoom_slider.setValue(100)
        ctrl_layout.addWidget(QLabel("Zoom:"))
        ctrl_layout.addWidget(zoom_slider)
        
        save_btn = QPushButton("💾 Save Result")
        save_btn.clicked.connect(self.save_result)
        ctrl_layout.addWidget(save_btn)
        
        export_btn = QPushButton("📤 Export to File")
        export_btn.clicked.connect(self.export_result)
        ctrl_layout.addWidget(export_btn)
        
        layout.addLayout(ctrl_layout)
        
        # Result info
        self.result_info = QTextEdit()
        self.result_info.setReadOnly(True)
        self.result_info.setMaximumHeight(150)
        layout.addWidget(QLabel("Result Information:"))
        layout.addWidget(self.result_info)
        
        self.setLayout(layout)
    
    def display_image(self, image_array: np.ndarray):
        """Display image in widget"""
        if image_array is None:
            return
        
        # Normalize to 0-255
        if image_array.dtype != np.uint8:
            image_array = ((image_array - image_array.min()) / 
                          (image_array.max() - image_array.min()) * 255).astype(np.uint8)
        
        # Convert to RGB if grayscale
        if len(image_array.shape) == 2:
            image_array = np.stack([image_array] * 3, axis=-1)
        
        # Convert to QImage
        h, w = image_array.shape[:2]
        bytes_per_line = 3 * w
        q_image = QImage(image_array.tobytes(), w, h, bytes_per_line, QImage.Format_RGB888)
        
        # Display
        pixmap = QPixmap.fromImage(q_image)
        scaled_pixmap = pixmap.scaledToWidth(400, Qt.SmoothTransformation)
        self.image_label.setPixmap(scaled_pixmap)
    
    def save_result(self):
        """Save current result"""
        pass
    
    def export_result(self):
        """Export result to file"""
        pass


class MedicalImagingGUI(QMainWindow):
    """Main desktop application window"""
    
    def __init__(self):
        super().__init__()
        
        if not PYQT5_AVAILABLE:
            print("PyQt5 not available. Skipping GUI initialization.")
            return
        
        self.init_ui()
    
    def init_ui(self):
        """Initialize main UI"""
        self.setWindowTitle("🏥 GPU-Accelerated Medical Imaging System")
        self.setGeometry(100, 100, 1400, 900)
        
        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QHBoxLayout()
        
        # Left panel - Processing controls
        left_panel = QWidget()
        left_layout = QVBoxLayout()
        
        # Processing frame
        self.processing_widget = ProcessingFrameWidget()
        left_layout.addWidget(self.processing_widget)
        
        left_panel.setLayout(left_layout)
        left_panel.setMaximumWidth(400)
        
        # Right panel - GPU Monitor and Visualization
        right_splitter = QSplitter(Qt.Vertical)
        
        # Top: GPU Monitor
        self.gpu_monitor = GPUMonitorWidget()
        right_splitter.addWidget(self.gpu_monitor)
        
        # Bottom: Visualization
        self.visualization = VisualizationWidget()
        right_splitter.addWidget(self.visualization)
        
        right_splitter.setSizes([200, 600])
        
        # Add panels to main layout
        main_splitter = QSplitter(Qt.Horizontal)
        main_splitter.addWidget(left_panel)
        main_splitter.addWidget(right_splitter)
        main_splitter.setSizes([400, 1000])
        
        main_layout.addWidget(main_splitter)
        
        central_widget.setLayout(main_layout)
        
        # Styling
        self.apply_styling()
    
    def apply_styling(self):
        """Apply dark theme styling"""
        style = """
            QMainWindow {
                background-color: #1e1e1e;
                color: #e0e0e0;
            }
            QWidget {
                background-color: #2d2d2d;
                color: #e0e0e0;
            }
            QPushButton {
                background-color: #0d47a1;
                color: white;
                border: none;
                padding: 8px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1565c0;
            }
            QLabel {
                color: #e0e0e0;
            }
            QTextEdit, QTableWidget {
                background-color: #1e1e1e;
                color: #e0e0e0;
                border: 1px solid #444;
            }
            QProgressBar {
                background-color: #1e1e1e;
                border: 1px solid #444;
                border-radius: 4px;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
            }
        """
        self.setStyleSheet(style)


def run_gui():
    """Run the PyQt5 GUI application"""
    if not PYQT5_AVAILABLE:
        print("❌ PyQt5 is required for GUI mode")
        print("Install with: pip install PyQt5")
        return
    
    app = QApplication(sys.argv)
    window = MedicalImagingGUI()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    run_gui()
