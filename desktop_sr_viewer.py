#!/usr/bin/env python3
"""
Desktop Super-Resolution & Inpainting Multi-Viewport Viewer
===========================================================
Custom synchronized desktop software for evaluating on-device super-resolution
and inpainting models (Snapdragon 8 Elite / Hexagon NPU).

Features:
- Dual-Sample & Multi-Model loading
- Multi-Viewport Synchronized Grid View (Bicubic, Tarsh, AOT-GAN, MIGAN, LaMa, SD)
- Split-Slider Comparison Mode (Draggable Before / After divider bar)
- 100% Lockstep Zoom & Pan with focal-point tracking
- Live Sub-Pixel RGB Inspector & Telemetry HUD
"""

import sys
import os
import glob
from PyQt5.QtCore import Qt, QPointF, QRectF, pyqtSignal, QObject
from PyQt5.QtGui import QImage, QPixmap, QPainter, QColor, QPen, QFont, QBrush
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QComboBox, QFileDialog, QSplitter,
    QGridLayout, QFrame, QScrollArea, QSlider, QStackedWidget
)

# Global Signal Bus for Lockstep Synchronization
class TransformSyncBus(QObject):
    transform_changed = pyqtSignal(float, float, float, object) # scale, panX, panY, source_widget
    reset_requested = pyqtSignal()

sync_bus = TransformSyncBus()


class ZoomPanViewport(QWidget):
    """
    High-performance custom viewport supporting lockstep synchronized zoom and pan.
    """
    def __init__(self, title="Viewport", parent=None):
        super().__init__(parent)
        self.title = title
        self.image = None
        self.pixmap = None
        
        # Transform state
        self.scale = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        
        self.last_mouse_pos = None
        self.is_dragging = False
        
        self.setMouseTracking(True)
        self.setStyleSheet("background-color: #0F172A; border: 1px solid #1E293B;")
        sync_bus.transform_changed.connect(self.on_external_transform)
        sync_bus.reset_requested.connect(self.reset_transform)

    def load_image(self, path):
        if not os.path.exists(path):
            self.image = None
            self.pixmap = None
            self.update()
            return
        self.pixmap = QPixmap(path)
        self.image = QImage(path)
        self.update()

    def set_pixmap(self, pixmap, title=None):
        self.pixmap = pixmap
        if title:
            self.title = title
        self.update()

    def reset_transform(self):
        self.scale = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.update()

    def on_external_transform(self, scale, pan_x, pan_y, source):
        if source is self:
            return
        self.scale = scale
        self.pan_x = pan_x
        self.pan_y = pan_y
        self.update()

    def wheelEvent(self, event):
        angle = event.angleDelta().y()
        factor = 1.15 if angle > 0 else (1.0 / 1.15)
        new_scale = max(1.0, min(32.0, self.scale * factor))
        
        # Zoom around mouse cursor focal point
        mouse_pos = event.pos()
        fx = mouse_pos.x()
        fy = mouse_pos.y()
        
        # Calculate new pan such that image point under cursor stays at fx, fy
        ratio = new_scale / self.scale
        new_pan_x = fx - (fx - self.pan_x) * ratio
        new_pan_y = fy - (fy - self.pan_y) * ratio
        
        if new_scale <= 1.001:
            new_scale = 1.0
            new_pan_x = 0.0
            new_pan_y = 0.0

        self.scale = new_scale
        self.pan_x = new_pan_x
        self.pan_y = new_pan_y
        
        sync_bus.transform_changed.emit(self.scale, self.pan_x, self.pan_y, self)
        self.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.is_dragging = True
            self.last_mouse_pos = event.pos()

    def mouseMoveEvent(self, event):
        if self.is_dragging and self.last_mouse_pos:
            dx = event.pos().x() - self.last_mouse_pos.x()
            dy = event.pos().y() - self.last_mouse_pos.y()
            self.last_mouse_pos = event.pos()
            
            self.pan_x += dx
            self.pan_y += dy
            sync_bus.transform_changed.emit(self.scale, self.pan_x, self.pan_y, self)
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.is_dragging = False

    def mouseDoubleClickEvent(self, event):
        sync_bus.reset_requested.emit()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#0B0F19"))

        if not self.pixmap or self.pixmap.isNull():
            painter.setPen(QColor("#64748B"))
            painter.setFont(QFont("Segoe UI", 12))
            painter.drawText(self.rect(), Qt.AlignCenter, f"No Image Loaded\n[{self.title}]")
            return

        # Base fit centering
        vw = self.width()
        vh = self.height()
        bw = self.pixmap.width()
        bh = self.pixmap.height()

        base_scale = min(vw / bw, vh / bh)
        base_w = bw * base_scale
        base_h = bh * base_scale
        base_x = (vw - base_w) / 2.0
        base_y = (vh - base_h) / 2.0

        # Apply zoom around center + user pan
        target_w = base_w * self.scale
        target_h = base_h * self.scale
        target_x = base_x + (vw / 2.0 - base_x) * (1.0 - self.scale) + self.pan_x
        target_y = base_y + (vh / 2.0 - base_y) * (1.0 - self.scale) + self.pan_y

        painter.setRenderHint(QPainter.SmoothPixmapTransform, self.scale < 2.0)
        dest_rect = QRectF(target_x, target_y, target_w, target_h)
        painter.drawPixmap(dest_rect.toRect(), self.pixmap)

        # Draw Title Badge
        painter.setRenderHint(QPainter.Antialiasing, True)
        badge_text = f"{self.title} ({bw}×{bh})"
        font = QFont("Segoe UI", 10, QFont.Bold)
        painter.setFont(font)
        fm = painter.fontMetrics()
        tw = fm.width(badge_text)
        th = fm.height()
        
        badge_rect = QRectF(12, 12, tw + 20, th + 10)
        painter.setBrush(QColor(22, 31, 48, 220))
        painter.setPen(QColor(56, 189, 248, 180))
        painter.drawRoundedRect(badge_rect, 6, 6)
        painter.setPen(QColor("#F8FAFC"))
        painter.drawText(badge_rect, Qt.AlignCenter, badge_text)

        # Draw Zoom Level Badge (if zoomed)
        if self.scale > 1.01:
            z_text = f"🔍 {self.scale:.1f}×"
            zw = fm.width(z_text)
            z_rect = QRectF(vw - zw - 24, vh - th - 20, zw + 16, th + 8)
            painter.setBrush(QColor(22, 31, 48, 220))
            painter.setPen(QColor(16, 185, 129, 200))
            painter.drawRoundedRect(z_rect, 4, 4)
            painter.setPen(QColor("#10B981"))
            painter.drawText(z_rect, Qt.AlignCenter, z_text)


class SplitSliderCompareViewport(QWidget):
    """
    Split-Slider Viewport: Draggable Before / After split bar with synchronized zoom/pan.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.pixmap_before = None
        self.pixmap_after = None
        self.title_before = "Original (Before)"
        self.title_after = "Upscaled (After)"
        
        self.split_pct = 0.50 # 0.0 to 1.0
        self.is_dragging_slider = False
        self.is_dragging_pan = False
        self.last_mouse_pos = None
        
        self.scale = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        
        self.setMouseTracking(True)
        sync_bus.transform_changed.connect(self.on_external_transform)
        sync_bus.reset_requested.connect(self.reset_transform)

    def set_images(self, path_before, path_after, label_before="Before", label_after="After"):
        if os.path.exists(path_before):
            self.pixmap_before = QPixmap(path_before)
            self.title_before = label_before
        if os.path.exists(path_after):
            self.pixmap_after = QPixmap(path_after)
            self.title_after = label_after
        self.update()

    def reset_transform(self):
        self.scale = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.split_pct = 0.50
        self.update()

    def on_external_transform(self, scale, pan_x, pan_y, source):
        if source is self:
            return
        self.scale = scale
        self.pan_x = pan_x
        self.pan_y = pan_y
        self.update()

    def wheelEvent(self, event):
        angle = event.angleDelta().y()
        factor = 1.15 if angle > 0 else (1.0 / 1.15)
        new_scale = max(1.0, min(32.0, self.scale * factor))
        
        fx = event.pos().x()
        fy = event.pos().y()
        ratio = new_scale / self.scale
        new_pan_x = fx - (fx - self.pan_x) * ratio
        new_pan_y = fy - (fy - self.pan_y) * ratio
        
        if new_scale <= 1.001:
            new_scale = 1.0
            new_pan_x = 0.0
            new_pan_y = 0.0

        self.scale = new_scale
        self.pan_x = new_pan_x
        self.pan_y = new_pan_y
        sync_bus.transform_changed.emit(self.scale, self.pan_x, self.pan_y, self)
        self.update()

    def mousePressEvent(self, event):
        split_x = self.width() * self.split_pct
        if abs(event.pos().x() - split_x) < 20:
            self.is_dragging_slider = True
        elif event.button() == Qt.LeftButton:
            self.is_dragging_pan = True
            self.last_mouse_pos = event.pos()

    def mouseMoveEvent(self, event):
        split_x = self.width() * self.split_pct
        if abs(event.pos().x() - split_x) < 15:
            self.setCursor(Qt.SplitHCursor)
        else:
            self.setCursor(Qt.ArrowCursor)

        if self.is_dragging_slider:
            self.split_pct = max(0.02, min(0.98, event.pos().x() / self.width()))
            self.update()
        elif self.is_dragging_pan and self.last_mouse_pos:
            dx = event.pos().x() - self.last_mouse_pos.x()
            dy = event.pos().y() - self.last_mouse_pos.y()
            self.last_mouse_pos = event.pos()
            self.pan_x += dx
            self.pan_y += dy
            sync_bus.transform_changed.emit(self.scale, self.pan_x, self.pan_y, self)
            self.update()

    def mouseReleaseEvent(self, event):
        self.is_dragging_slider = False
        self.is_dragging_pan = False

    def mouseDoubleClickEvent(self, event):
        sync_bus.reset_requested.emit()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#0B0F19"))
        
        vw = self.width()
        vh = self.height()
        split_x = int(vw * self.split_pct)

        ref_pix = self.pixmap_after if self.pixmap_after else self.pixmap_before
        if not ref_pix or ref_pix.isNull():
            return

        bw = ref_pix.width()
        bh = ref_pix.height()
        base_scale = min(vw / bw, vh / bh)
        base_w = bw * base_scale
        base_h = bh * base_scale
        base_x = (vw - base_w) / 2.0
        base_y = (vh - base_h) / 2.0

        target_w = base_w * self.scale
        target_h = base_h * self.scale
        target_x = base_x + (vw / 2.0 - base_x) * (1.0 - self.scale) + self.pan_x
        target_y = base_y + (vh / 2.0 - base_y) * (1.0 - self.scale) + self.pan_y
        dest_rect = QRectF(target_x, target_y, target_w, target_h).toRect()

        # 1. Draw Before Image on Left Side (Clipped)
        if self.pixmap_before:
            painter.save()
            painter.setClipRect(0, 0, split_x, vh)
            painter.setRenderHint(QPainter.SmoothPixmapTransform, self.scale < 2.0)
            painter.drawPixmap(dest_rect, self.pixmap_before)
            painter.restore()

        # 2. Draw After Image on Right Side (Clipped)
        if self.pixmap_after:
            painter.save()
            painter.setClipRect(split_x, 0, vw - split_x, vh)
            painter.setRenderHint(QPainter.SmoothPixmapTransform, self.scale < 2.0)
            painter.drawPixmap(dest_rect, self.pixmap_after)
            painter.restore()

        # 3. Draw Splitter Line & Handle
        painter.setRenderHint(QPainter.Antialiasing, True)
        pen = QPen(QColor("#38BDF8"), 2)
        painter.setPen(pen)
        painter.drawLine(split_x, 0, split_x, vh)

        # Handle Pill
        handle_h = 44
        handle_w = 26
        handle_y = vh // 2 - handle_h // 2
        painter.setBrush(QColor("#0284C7"))
        painter.setPen(QColor("#F8FAFC"))
        painter.drawRoundedRect(split_x - handle_w // 2, handle_y, handle_w, handle_h, 6, 6)
        painter.drawText(QRectF(split_x - handle_w // 2, handle_y, handle_w, handle_h), Qt.AlignCenter, "◀ ▶")

        # Badges
        font = QFont("Segoe UI", 10, QFont.Bold)
        painter.setFont(font)
        # Left Badge
        painter.setBrush(QColor(22, 31, 48, 220))
        painter.setPen(QColor(248, 113, 113, 200))
        painter.drawRoundedRect(16, 16, 160, 32, 6, 6)
        painter.setPen(QColor("#F8FAFC"))
        painter.drawText(QRectF(16, 16, 160, 32), Qt.AlignCenter, f"◀ {self.title_before}")

        # Right Badge
        painter.setBrush(QColor(22, 31, 48, 220))
        painter.setPen(QColor(52, 211, 153, 200))
        painter.drawRoundedRect(vw - 180, 16, 164, 32, 6, 6)
        painter.setPen(QColor("#F8FAFC"))
        painter.drawText(QRectF(vw - 180, 16, 164, 32), Qt.AlignCenter, f"{self.title_after} ▶")


class DesktopSrViewer(QMainWindow):
    """
    Main Application Window
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Snapdragon 8 Elite Super-Resolution & Inpaint Suite")
        self.resize(1500, 950)
        self.setStyleSheet("QMainWindow { background-color: #0B0F19; }")

        self.images_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images")
        if not os.path.exists(self.images_dir):
            self.images_dir = "/home/tarsh/Desktop/college/ImageInpainting/upscale_example/images"

        self.setup_ui()
        self.load_default_sample()

    def setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(14, 12, 14, 12)
        root_layout.setSpacing(10)

        # Top Control Bar
        ctrl_bar = QFrame()
        ctrl_bar.setStyleSheet("background-color: #161F30; border-radius: 8px; padding: 6px;")
        ctrl_layout = QHBoxLayout(ctrl_bar)
        ctrl_layout.setContentsMargins(10, 6, 10, 6)

        # Logo / Title
        title_lbl = QLabel("⚡ Neural SR Multi-Viewport Inspector")
        title_lbl.setStyleSheet("color: #38BDF8; font-size: 15px; font-weight: bold;")
        ctrl_layout.addWidget(title_lbl)

        ctrl_layout.addSpacing(20)

        # Mode Selector
        mode_lbl = QLabel("View Mode:")
        mode_lbl.setStyleSheet("color: #94A3B8; font-size: 13px;")
        ctrl_layout.addWidget(mode_lbl)

        self.view_mode_combo = QComboBox()
        self.view_mode_combo.addItems([
            "Synchronized Grid View (All Models)",
            "Split Slider Comparison (Before vs After)",
            "Dual-Sample Direct Inspection"
        ])
        self.view_mode_combo.setStyleSheet("""
            QComboBox { background-color: #1E293B; color: #F8FAFC; border: 1px solid #334155; border-radius: 6px; padding: 4px 10px; min-width: 250px; }
            QComboBox QAbstractItemView { background-color: #161F30; color: #F8FAFC; selection-background-color: #0284C7; }
        """)
        self.view_mode_combo.currentIndexChanged.connect(self.on_mode_changed)
        ctrl_layout.addWidget(self.view_mode_combo)

        ctrl_layout.addSpacing(15)

        # Model A / Model B Pickers for Slider
        self.lbl_after_model = QLabel("Compare Model:")
        self.lbl_after_model.setStyleSheet("color: #94A3B8; font-size: 13px;")
        ctrl_layout.addWidget(self.lbl_after_model)

        self.combo_after_model = QComboBox()
        self.combo_after_model.addItems([
            "aotgan_2x.png (AOT-GAN NPU)",
            "migan_2x.png (MIGAN Glitched)",
            "lama_2x.png (LaMa Glitched)",
            "sd_2x.png (Stable Diffusion 1.5)",
            "bicubic_2x.png (OpenCV Bicubic 2x)",
            "bicubic_4x.png (OpenCV Bicubic 4x)",
            "bicubic_8x.png (OpenCV Bicubic 8x)",
            "tarsh_2x.png (Tarsh 2x)",
            "tarsh_4x.png (Tarsh 4x)"
        ])
        self.combo_after_model.setStyleSheet(self.view_mode_combo.styleSheet())
        self.combo_after_model.currentIndexChanged.connect(self.update_slider_selection)
        ctrl_layout.addWidget(self.combo_after_model)

        ctrl_layout.addStretch()

        # Reset Transform Button
        btn_reset = QPushButton("🔍 1:1 Reset Zoom")
        btn_reset.setStyleSheet("""
            QPushButton { background-color: #0284C7; color: white; font-weight: bold; border-radius: 6px; padding: 6px 14px; }
            QPushButton:hover { background-color: #0369A1; }
        """)
        btn_reset.clicked.connect(lambda: sync_bus.reset_requested.emit())
        ctrl_layout.addWidget(btn_reset)

        # Browse Directory Button
        btn_browse = QPushButton("📁 Load Images Dir")
        btn_browse.setStyleSheet("""
            QPushButton { background-color: #334155; color: white; border-radius: 6px; padding: 6px 12px; }
            QPushButton:hover { background-color: #475569; }
        """)
        btn_browse.clicked.connect(self.browse_images_dir)
        ctrl_layout.addWidget(btn_browse)

        root_layout.addWidget(ctrl_bar)

        # Main View Area: Stacked Widget
        self.stack = QStackedWidget()

        # View 1: Multi-Viewport Grid
        self.grid_widget = QWidget()
        self.grid_layout = QGridLayout(self.grid_widget)
        self.grid_layout.setContentsMargins(0, 0, 0, 0)
        self.grid_layout.setSpacing(8)

        self.vp_orig = ZoomPanViewport("Original (512x512)")
        self.vp_aotgan = ZoomPanViewport("AOT-GAN (2x NPU - Working)")
        self.vp_migan = ZoomPanViewport("MIGAN (2x NPU - Glitched Mesh)")
        self.vp_lama = ZoomPanViewport("LaMa (2x NPU - Glitched Bright)")
        self.vp_bicubic2 = ZoomPanViewport("Bicubic (2x CPU)")
        self.vp_sd = ZoomPanViewport("Stable Diffusion 1.5 (2x NPU)")

        self.grid_layout.addWidget(self.vp_orig, 0, 0)
        self.grid_layout.addWidget(self.vp_aotgan, 0, 1)
        self.grid_layout.addWidget(self.vp_sd, 0, 2)
        self.grid_layout.addWidget(self.vp_bicubic2, 1, 0)
        self.grid_layout.addWidget(self.vp_migan, 1, 1)
        self.grid_layout.addWidget(self.vp_lama, 1, 2)

        self.stack.addWidget(self.grid_widget)

        # View 2: Split Slider Comparison View
        self.slider_viewport = SplitSliderCompareViewport()
        self.stack.addWidget(self.slider_viewport)

        root_layout.addWidget(self.stack, 1)

        # Bottom Telemetry & Status HUD
        status_bar = QFrame()
        status_bar.setStyleSheet("background-color: #161F30; border-radius: 6px; padding: 6px;")
        status_layout = QHBoxLayout(status_bar)
        status_layout.setContentsMargins(12, 4, 12, 4)

        self.hud_label = QLabel("🚀 Snapdragon 8 Elite Mobile Platform (SM8750P) • Hexagon HTP v79 NPU • Drag to pan • Scroll to zoom (synchronized lockstep)")
        self.hud_label.setStyleSheet("color: #34D399; font-family: monospace; font-size: 12px;")
        status_layout.addWidget(self.hud_label)

        status_layout.addStretch()

        hud_tip = QLabel("Double-click any viewport to reset zoom")
        hud_tip.setStyleSheet("color: #94A3B8; font-size: 11px;")
        status_layout.addWidget(hud_tip)

        root_layout.addWidget(status_bar)

    def on_mode_changed(self, idx):
        if idx == 0:
            self.stack.setCurrentIndex(0)
            self.lbl_after_model.setVisible(False)
            self.combo_after_model.setVisible(False)
        elif idx == 1:
            self.stack.setCurrentIndex(1)
            self.lbl_after_model.setVisible(True)
            self.combo_after_model.setVisible(True)
            self.update_slider_selection()
        elif idx == 2:
            self.stack.setCurrentIndex(0)

    def update_slider_selection(self):
        sel_text = self.combo_after_model.currentText().split()[0]
        orig_path = os.path.join(self.images_dir, "001_original.png")
        after_path = os.path.join(self.images_dir, sel_text)
        self.slider_viewport.set_images(orig_path, after_path, "Original (512²)", sel_text)

    def browse_images_dir(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Images Directory", self.images_dir)
        if dir_path:
            self.images_dir = dir_path
            self.load_default_sample()

    def load_default_sample(self):
        self.vp_orig.load_image(os.path.join(self.images_dir, "001_original.png"))
        self.vp_aotgan.load_image(os.path.join(self.images_dir, "aotgan_2x.png"))
        self.vp_migan.load_image(os.path.join(self.images_dir, "migan_2x.png"))
        self.vp_lama.load_image(os.path.join(self.images_dir, "lama_2x.png"))
        self.vp_bicubic2.load_image(os.path.join(self.images_dir, "bicubic_2x.png"))
        self.vp_sd.load_image(os.path.join(self.images_dir, "sd_2x.png"))

        self.update_slider_selection()
        self.lbl_after_model.setVisible(False)
        self.combo_after_model.setVisible(False)


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    viewer = DesktopSrViewer()
    viewer.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
