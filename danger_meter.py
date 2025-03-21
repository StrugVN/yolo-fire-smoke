from PyQt5.QtWidgets import QProgressBar, QHBoxLayout, QLabel, QFrame
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QPalette
import numpy as np

class DangerMeter:
    """
    A class to calculate and display the danger level based on
    the coverage of fire and smoke in the frame.
    """
    def __init__(self, parent_widget):
        # Create UI components
        self.container = QFrame(parent_widget)
        self.container.setFrameShape(QFrame.StyledPanel)
        self.container.setMaximumHeight(60)
        self.container.setStyleSheet("background-color: #f5f5f5; border-radius: 5px;")
        
        # Create layout
        self.layout = QHBoxLayout(self.container)
        self.layout.setContentsMargins(10, 5, 10, 5)
        
        # Create title label
        self.title_label = QLabel("Danger-O-Meter:")
        self.title_label.setStyleSheet("font-weight: bold;")
        self.layout.addWidget(self.title_label)
        
        # Create progress bar for danger level
        self.danger_bar = QProgressBar()
        self.danger_bar.setRange(0, 100)
        self.danger_bar.setValue(0)
        self.danger_bar.setTextVisible(True)
        self.danger_bar.setFormat("%v% Danger")
        self.danger_bar.setMinimumWidth(200)
        self.layout.addWidget(self.danger_bar)
        
        # Create status label
        self.status_label = QLabel("Safe")
        self.status_label.setStyleSheet("font-weight: bold; color: green;")
        self.status_label.setMinimumWidth(100)
        self.layout.addWidget(self.status_label)
        
        # Add stretch to keep components aligned
        self.layout.addStretch(1)
        
        # Set initial values
        self.danger_level = 0
        self.fire_coverage = 0
        self.smoke_coverage = 0
        self.update_meter(0)
        
        # Make container initially visible
        self.container.setVisible(True)

    def calculate_danger(self, results):
        """
        Calculate the danger level based on YOLO detection results.
        
        Args:
            results: YOLO detection results
            
        Returns:
            float: Danger level percentage (0-100)
        """
        # Default return if no results
        if not results or not hasattr(results, 'boxes') or results.boxes is None:
            self.fire_coverage = 0
            self.smoke_coverage = 0
            return 0
        
        # Get image dimensions
        img_height, img_width = results.orig_shape
        total_area = img_height * img_width
        
        # Initialize areas
        fire_area = 0
        smoke_area = 0
        
        # Extract bounding boxes and classes
        boxes = results.boxes
        if len(boxes) == 0:
            self.fire_coverage = 0
            self.smoke_coverage = 0
            return 0
            
        # Process each detection
        for i in range(len(boxes)):
            # Get box coordinates and class
            box = boxes[i].xyxy.cpu().numpy()[0] if hasattr(boxes[i], 'xyxy') else boxes[i].cpu().numpy()
            cls = int(boxes[i].cls.cpu().numpy()[0]) if hasattr(boxes[i], 'cls') else 0
            
            # Calculate box area
            x1, y1, x2, y2 = box[:4]
            box_area = (x2 - x1) * (y2 - y1)
            
            # Add to respective area based on class
            if cls == 0:  # Fire class
                fire_area += box_area
            elif cls == 1:  # Smoke class
                smoke_area += box_area
        
        # Calculate coverage percentages (limit to 100%)
        self.fire_coverage = min(100, (fire_area / total_area) * 100)
        self.smoke_coverage = min(100, (smoke_area / total_area) * 100)
        
        # Calculate danger level - weighted sum with fire having higher weight
        # Fire is 3x more dangerous than smoke
        danger_level = min(100, (self.fire_coverage * 3 + self.smoke_coverage) / 4)
        
        return danger_level
        
    def update_meter(self, danger_level):
        """
        Update the danger meter UI based on the danger level.
        
        Args:
            danger_level: Calculated danger level (0-100)
        """
        self.danger_level = danger_level
        self.danger_bar.setValue(int(danger_level))
        
        # Set color based on danger level
        palette = QPalette()
        if danger_level < 20:
            color = QColor(0, 255, 0)  # Green - Safe
            status = "Safe"
            status_color = "green"
        elif danger_level < 40:
            color = QColor(150, 255, 0)  # Light Green-Yellow - Low
            status = "Low Risk"
            status_color = "#689F38"
        elif danger_level < 60:
            color = QColor(255, 255, 0)  # Yellow - Moderate
            status = "Moderate"
            status_color = "#FBC02D"
        elif danger_level < 80:
            color = QColor(255, 150, 0)  # Orange - High
            status = "High Risk"
            status_color = "#EF6C00"
        else:
            color = QColor(255, 0, 0)  # Red - Extreme
            status = "EXTREME DANGER"
            status_color = "red"
        
        palette.setColor(QPalette.Highlight, color)
        self.danger_bar.setPalette(palette)
        
        # Update status label
        self.status_label.setText(status)
        self.status_label.setStyleSheet(f"font-weight: bold; color: {status_color};")
        
        # Return detailed info for display
        return {
            'danger_level': danger_level,
            'fire_coverage': self.fire_coverage,
            'smoke_coverage': self.smoke_coverage,
            'status': status
        }
        
    def get_coverage_text(self):
        """Returns formatted text showing fire and smoke coverage percentages"""
        return f"Fire: {self.fire_coverage:.1f}% | Smoke: {self.smoke_coverage:.1f}%"