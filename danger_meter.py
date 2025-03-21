import time
from PyQt5.QtWidgets import QProgressBar, QHBoxLayout, QLabel, QFrame
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QPalette

class DangerMeter:
    """
    A class to calculate and display the danger level based on
    the coverage of fire and smoke in the frame and duration.
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
        self.layout.setSpacing(5)  
        
        # Create title label (changed to "Risk")
        self.title_label = QLabel("Risk:")
        self.title_label.setStyleSheet("font-weight: bold;")
        self.title_label.setFixedWidth(40)  # Fix width to save space
        self.layout.addWidget(self.title_label)
        
        # Create custom progress bar
        self.danger_bar = CustomProgressBar()
        self.danger_bar.setRange(0, 100)
        self.danger_bar.setValue(0)
        self.danger_bar.setMinimumWidth(500)
        self.layout.addWidget(self.danger_bar, 10)
        
        # Add stretch to keep components aligned
        self.layout.addStretch(1)
        
        # Set initial values
        self.danger_level = 0
        self.fire_coverage = 0
        self.smoke_coverage = 0
        
        # Time-based danger calculation variables
        self.first_fire_time = 0
        self.first_smoke_time = 0
        self.last_fire_time = 0
        self.last_smoke_time = 0
        self.last_clear_time = 0
        self.fire_duration_factor = 0
        self.smoke_duration_factor = 0
        
        # Define time thresholds for max danger
        self.fire_max_duration = 5.0
        self.smoke_max_duration = 10.0
        self.error_margin = 1.0
        self.clear_required = 3.0
        
        # Initialize the meter
        self.update_meter(0)
        
        # Make container initially visible
        self.container.setVisible(True)

    def calculate_danger(self, results):
        """
        Calculate the danger level based on YOLO detection results and detection duration.
        
        Args:
            results: YOLO detection results
            
        Returns:
            float: Danger level percentage (0-100)
        """
        current_time = time.time()
        has_fire = False
        has_smoke = False
        
        # Default return if no results
        if not results or not hasattr(results, 'boxes') or results.boxes is None:
            self.fire_coverage = 0
            self.smoke_coverage = 0
            # Handle clearance if there's no detections
            self._handle_clearance(current_time)
            return self._calculate_final_danger_level()
        
        # Get image dimensions
        img_height, img_width = results.orig_shape
        total_screen_area = img_height * img_width
        
        # Initialize areas
        fire_area = 0
        smoke_area = 0
        
        # Extract bounding boxes and classes
        boxes = results.boxes
        if len(boxes) == 0:
            self.fire_coverage = 0
            self.smoke_coverage = 0
            # Handle clearance if there's no detections
            self._handle_clearance(current_time)
            return self._calculate_final_danger_level()
            
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
                has_fire = True
            elif cls == 1:  # Smoke class
                smoke_area += box_area
                has_smoke = True
        
        # Calculate individual coverage percentages
        self.fire_coverage = (fire_area / total_screen_area) * 100
        self.smoke_coverage = (smoke_area / total_screen_area) * 100
        
        # Process fire duration
        if has_fire:
            if self.first_fire_time == 0 or (current_time - self.last_fire_time) > self.error_margin:
                self.first_fire_time = current_time
            
            # Calculate duration factor (0-100)
            elapsed = current_time - self.first_fire_time
            self.fire_duration_factor = min(100, (elapsed / self.fire_max_duration) * 100)
            self.last_fire_time = current_time
        elif (current_time - self.last_fire_time) > self.error_margin:
            # Reset if no fire detected beyond error margin
            self.first_fire_time = 0
        
        # Process smoke duration
        if has_smoke:
            if self.first_smoke_time == 0 or (current_time - self.last_smoke_time) > self.error_margin:
                self.first_smoke_time = current_time
            
            # Calculate duration factor (0-100)
            elapsed = current_time - self.first_smoke_time
            self.smoke_duration_factor = min(100, (elapsed / self.smoke_max_duration) * 100)
            self.last_smoke_time = current_time
        elif (current_time - self.last_smoke_time) > self.error_margin:
            # Reset if no smoke detected beyond error margin
            self.first_smoke_time = 0
            
        # Handle clearance if neither fire nor smoke is detected
        if not has_fire and not has_smoke:
            self._handle_clearance(current_time)
        else:
            self.last_clear_time = 0
            
        # Calculate final danger level
        return self._calculate_final_danger_level()
        
    def _handle_clearance(self, current_time):
        """Handle clearance of time-based factors after period of no detection"""
        if self.last_clear_time == 0:
            self.last_clear_time = current_time
        elif (current_time - self.last_clear_time) >= self.clear_required:
            # Reset time factors after required clear duration
            self.fire_duration_factor = max(0, self.fire_duration_factor - 20)  # Gradually reduce
            self.smoke_duration_factor = max(0, self.smoke_duration_factor - 20)  # Gradually reduce
            
    def _calculate_final_danger_level(self):
        """Calculate final danger level combining area coverage and duration factors"""
        # Combined area coverage
        combined_coverage = min(100, self.fire_coverage + self.smoke_coverage)
        
        # Duration increases danger - weight fire duration higher than smoke
        duration_factor = (self.fire_duration_factor * 0.7 + self.smoke_duration_factor * 0.3) / 100
        
        # Final danger level formula:
        # Base level from combined coverage, plus up to 50% more from duration
        # This ensures area coverage remains the primary factor but duration adds significant weight
        danger_bonus = min(50, combined_coverage * duration_factor)
        final_danger = min(100, combined_coverage + danger_bonus)
        
        return final_danger
        
    def update_meter(self, danger_level):
        """
        Update the danger meter UI based on the danger level.
        
        Args:
            danger_level: Calculated danger level (0-100)
        """
        self.danger_level = danger_level
        self.danger_bar.setValue(int(danger_level))
        
        # Set color and status based on danger level
        if danger_level < 20:
            color = QColor(0, 255, 0)  # Green - Safe
            status = "Safe"
            text_color = "black"
        elif danger_level < 40:
            color = QColor(150, 255, 0)  # Light Green-Yellow - Low
            status = "Low Risk"
            text_color = "black"
        elif danger_level < 60:
            color = QColor(255, 255, 0)  # Yellow - Moderate
            status = "Moderate"
            text_color = "black"
        elif danger_level < 80:
            color = QColor(255, 150, 0)  # Orange - High
            status = "High Risk"
            text_color = "white"
        else:
            color = QColor(255, 0, 0)  # Red - Extreme
            status = "EXTREME DANGER"
            text_color = "white"
        
        # Update the custom progress bar
        self.danger_bar.setStatusText(status)
        self.danger_bar.setStatusColor(text_color)
        self.danger_bar.setBarColor(color)
        
    def get_coverage_text(self):
        """Returns formatted text showing combined fire and smoke coverage percentages"""
        combined = min(100, self.fire_coverage + self.smoke_coverage)
        return f"Fire: {self.fire_coverage:.1f}% | Smoke: {self.smoke_coverage:.1f}% | Combined: {combined:.1f}%"
    

class CustomProgressBar(QProgressBar):
    """
    Custom progress bar with text centered in the filled portion.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.status_text = "Safe"
        self.text_color = "black"
        self.bar_color = QColor(0, 255, 0)  # Default green
        
    def setStatusText(self, text):
        self.status_text = text
        self.update()
        
    def setStatusColor(self, color):
        self.text_color = color
        self.update()
        
    def setBarColor(self, color):
        # Store the color
        self.bar_color = color
        
        # Convert QColor to CSS color string
        css_color = f"rgb({color.red()}, {color.green()}, {color.blue()})"
        
        # Apply stylesheet directly - this is more reliable for changing colors
        self.setStyleSheet(f"""
            QProgressBar {{
                border: 1px solid #888;
                border-radius: 0px;
                text-align: center;
                background-color: #f0f0f0;
            }}
            QProgressBar::chunk {{
                background-color: {css_color};
            }}
        """)
        self.update()
        
    def paintEvent(self, event):
        # First, draw the normal progress bar
        super().paintEvent(event)
        
        # Then overlay our custom text
        from PyQt5.QtGui import QPainter, QColor
        from PyQt5.QtCore import QRectF
        
        painter = QPainter(self)
        
        # Calculate the width of the filled portion
        progress_width = int((self.width() * self.value()) / self.maximum())
        
        # Only draw text if there's enough space
        if progress_width > 50:  # Minimum width to show text
            # Set up the font
            font = painter.font()
            font.setBold(True)
            painter.setFont(font)
            
            # Draw the text centered in the filled portion
            painter.setPen(QColor(self.text_color))
            text_rect = QRectF(0, 0, progress_width, self.height())
            painter.drawText(text_rect, Qt.AlignCenter, self.status_text)