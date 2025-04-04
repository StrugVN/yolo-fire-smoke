import time
from PyQt5.QtWidgets import QProgressBar, QHBoxLayout, QLabel, QFrame
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QPalette
import math

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

        self.last_risk_level = 0
        self.last_risk_time = 0
        self.highest_recent_risk = 0
        self.decay_start_time = 0
        self.decay_start_value = 0
        self.decay_duration = 6.0  
        self.decay_drop_threshold = 4.0
        
        # Initialize the meter
        self.update_meter(0)
        
        # Make container initially visible
        self.container.setVisible(True)

    def calculate_danger(self, results):
        """
        Calculate the danger level based on YOLO detection results, accounting for overlapping boxes.
        
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
            self._handle_clearance(current_time)
            # Clear detection data when no detections
            self.detection_data = []
            self.detection_boxes = []
            return self._calculate_final_danger_level()

        # Get image dimensions
        img_height, img_width = results.orig_shape
        total_screen_area = max(1, img_height * img_width)

        # Create mask images for fire and smoke to account for overlaps
        import numpy as np
        fire_mask = np.zeros((img_height, img_width), dtype=np.uint8)
        smoke_mask = np.zeros((img_height, img_width), dtype=np.uint8)

        boxes = results.boxes
        if len(boxes) == 0:
            self.fire_coverage = 0
            self.smoke_coverage = 0
            self._handle_clearance(current_time)
            # Clear detection data if no boxes
            self.detection_data = []
            self.detection_boxes = []
            return self._calculate_final_danger_level()

        # Initialize lists to store detection data and positions for drawing
        self.detection_data = []   # List of dicts: each with label, cls, index, conf, area, x, y.
        self.detection_boxes = []  # For drawing labels on the frame: list of (x, y, label)
        fire_index = 0
        smoke_index = 0

        # Process each detection
        for i in range(len(boxes)):
            # Get box coordinates and class
            box = boxes[i].xyxy[0].cpu().numpy() if hasattr(boxes[i], 'xyxy') else boxes[i].cpu().numpy()
            cls = int(boxes[i].cls.cpu().numpy()[0]) if hasattr(boxes[i], 'cls') else 0

            # Get confidence (if available)
            if hasattr(boxes[i], 'conf'):
                conf = float(boxes[i].conf.cpu().numpy()[0])
            else:
                conf = 0.0

            # Get integer coordinates for the bounding box
            x1, y1, x2, y2 = [int(coord) for coord in box[:4]]

            # Ensure coordinates are within image bounds
            x1 = max(0, min(x1, img_width - 1))
            x2 = max(0, min(x2, img_width - 1))
            y1 = max(0, min(y1, img_height - 1))
            y2 = max(0, min(y2, img_height - 1))

            # Process fire and smoke detections separately and assign index label
            if cls == 0:  # Fire class
                fire_mask[y1:y2, x1:x2] = 1
                has_fire = True
                label = f"fire_{fire_index}"
                index = fire_index
                fire_index += 1
            elif cls == 1:  # Smoke class
                smoke_mask[y1:y2, x1:x2] = 1
                has_smoke = True
                label = f"smoke_{smoke_index}"
                index = smoke_index
                smoke_index += 1
            else:
                continue  # Skip any other classes

            # Calculate bounding box area
            area = (x2 - x1) * (y2 - y1)
            # Save detailed detection data
            self.detection_data.append({
                "label": label,
                "cls": cls,
                "index": index,
                "conf": conf,
                "area": area,
                "x": x1,
                "y": y1
            })
            # Save position for drawing label on the frame
            self.detection_boxes.append((x1, y1, label))

        # Calculate coverage percentages from masks
        fire_area = np.sum(fire_mask)
        smoke_area = np.sum(smoke_mask)
        self.fire_coverage = (fire_area / total_screen_area) * 100
        self.smoke_coverage = (smoke_area / total_screen_area) * 100
        
        # Process duration factors as before
        if has_fire:
            if self.first_fire_time == 0 or (current_time - self.last_fire_time) > self.error_margin:
                self.first_fire_time = current_time
            
            elapsed = current_time - self.first_fire_time
            self.fire_duration_factor = min(100, (elapsed / self.fire_max_duration) * 100)
            self.last_fire_time = current_time
        elif (current_time - self.last_fire_time) > self.error_margin:
            self.first_fire_time = 0
        
        if has_smoke:
            if self.first_smoke_time == 0 or (current_time - self.last_smoke_time) > self.error_margin:
                self.first_smoke_time = current_time
            
            elapsed = current_time - self.first_smoke_time
            self.smoke_duration_factor = min(100, (elapsed / self.smoke_max_duration) * 100)
            self.last_smoke_time = current_time
        elif (current_time - self.last_smoke_time) > self.error_margin:
            self.first_smoke_time = 0
        
        # Handle clearance if neither fire nor smoke is detected
        if not has_fire and not has_smoke:
            self._handle_clearance(current_time)
        else:
            self.last_clear_time = 0
        
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
        """
        Calculate final danger level with improved decay mechanism
        """
        current_time = time.time()
        
        # Non-linear scaling for fire and smoke (unchanged)
        if self.fire_coverage > 0:
            a = 70 / (math.log(11) - math.log(2))
            b = 10 - a * math.log(2)
            fire_risk = min(100, a * math.log(self.fire_coverage + 1) + b)
            has_fire = True
        else:
            fire_risk = 0
            has_fire = False
        
        if self.smoke_coverage > 0:
            a = 35 / (math.log(11) - math.log(2))
            b = 5 - a * math.log(2)
            smoke_risk = min(100, a * math.log(self.smoke_coverage + 1) + b)
            has_smoke = True
        else:
            smoke_risk = 0
            has_smoke = False
        
        # Calculate base risk and apply growth factor (unchanged)
        base_risk = max(fire_risk, smoke_risk)
        
        fire_growth_rate = self.fire_duration_factor / 100
        smoke_growth_rate = self.smoke_duration_factor / 100
        
        if fire_growth_rate <= 0.5:
            fire_growth_effect = fire_growth_rate * 1.5
        else:
            fire_growth_effect = 0.75 + (fire_growth_rate - 0.5) * 0.5
        
        if smoke_growth_rate <= 0.5:
            smoke_growth_effect = smoke_growth_rate * 1.5
        else:
            smoke_growth_effect = 0.75 + (smoke_growth_rate - 0.5) * 0.5
        
        growth_multiplier = 1.0 + max(fire_growth_effect, smoke_growth_effect)
        current_calculated_risk = min(100, base_risk * growth_multiplier)
        
        # IMPROVED DECAY LOGIC
        # Check if current risk is significantly lower than previous (risk drop)
        is_significant_drop = (self.last_risk_level - current_calculated_risk) > 5.0  # 5% threshold
        
        # Case 1: Risk is increasing or stable
        if current_calculated_risk >= self.last_risk_level:
            # Use the new higher value
            final_risk = current_calculated_risk
            # Reset decay since risk is increasing
            self.decay_start_time = 0
            self.decay_start_value = 0
        
        # Case 2: Risk dropped significantly and decay not started yet
        elif is_significant_drop and self.decay_start_time == 0:
            # Start a new decay from the last risk level
            self.decay_start_time = current_time
            self.decay_start_value = self.last_risk_level
            final_risk = self.last_risk_level  # Start from previous value
        
        # Case 3: Already in decay process
        elif self.decay_start_time > 0:
            elapsed_decay_time = current_time - self.decay_start_time
            
            if elapsed_decay_time < self.decay_drop_threshold:
                # Linear decay for first threshold seconds
                initial_decay_factor = 1.0 - (elapsed_decay_time / self.decay_duration)
                decay_risk = self.decay_start_value * initial_decay_factor
                # Take the max of decay risk and current risk
                final_risk = max(decay_risk, current_calculated_risk)
            elif elapsed_decay_time < self.decay_duration:
                # After threshold, use current risk
                final_risk = current_calculated_risk
                # If we've reached current risk level, end decay
                if abs(final_risk - current_calculated_risk) < 1.0:
                    self.decay_start_time = 0
            else:
                # Decay complete
                final_risk = current_calculated_risk
                self.decay_start_time = 0
        
        # Case 4: Small risk decrease, not worth starting decay
        else:
            final_risk = current_calculated_risk
        
        # Update last risk level for next iteration
        self.last_risk_level = final_risk
        
        return final_risk
        
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
            status = "Caution"
            text_color = "black"
        elif danger_level < 40:
            color = QColor(150, 255, 0)  # Light Green-Yellow - Low
            status = "Hazardous"
            text_color = "black"
        elif danger_level < 60:
            color = QColor(255, 255, 0)  # Yellow - Moderate
            status = "Critical"
            text_color = "black"
        elif danger_level < 80:
            color = QColor(255, 150, 0)  # Orange - High
            status = "Danger!"
            text_color = "white"
        else:
            color = QColor(255, 0, 0)  # Red - Extreme
            status = "DISASTEROUS!"
            text_color = "white"
        
        # Update the custom progress bar
        # If you want to include the percentage in the status text:
        # self.danger_bar.setStatusText(f"{status} ({int(danger_level)}%)")
        
        # If you want to remove percentage completely:
        self.danger_bar.setStatusText(f"{status}")
        
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
        
        # Apply stylesheet directly - note the "text-align: right" is set to none
        # to prevent the percentage from showing on the right
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
        
        # This is the key line - make sure default text display is disabled
        self.setTextVisible(False)
        
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