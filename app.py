import sys
import cv2
import numpy as np
import threading
import queue
import time
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QLabel, QPushButton, QLineEdit, QFileDialog,
                            QSlider, QComboBox, QMessageBox)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QThread, pyqtSignal, pyqtSlot
from ultralytics import YOLO

class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)
    update_position_signal = pyqtSignal(int, int)  # current_frame, total_frames
    
    def __init__(self, source_type, source_path, model_path):
        super().__init__()
        self.source_type = source_type  # 'rtsp' or 'file'
        self.source_path = source_path
        self.model_path = model_path
        self.running = True
        self.paused = False
        self.frame_queue = queue.Queue(maxsize=2)
        self.stop_event = threading.Event()
        self.playback_speed = 1.0  # Normal speed
        self.seek_position = -1  # -1 means no seeking
        self.current_frame_position = 0
        self.total_frames = 0
        
        # Load YOLO model
        try:
            self.model = YOLO(model_path)
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.running = False
            return
        
    def run(self):
        if not self.running:
            return
            
        if self.source_type == 'rtsp':
            # Start a separate thread for RTSP streaming to handle network issues
            reader_thread = threading.Thread(
                target=self.frame_reader,
                args=(self.source_path, self.frame_queue, self.stop_event)
            )
            reader_thread.daemon = True
            reader_thread.start()
            
            # Process frames from the queue
            self.process_frames_from_queue()
            
            # Clean up
            self.stop_event.set()
            if reader_thread.is_alive():
                reader_thread.join(timeout=1.0)
                
        elif self.source_type == 'file':
            # Process video file directly
            self.process_video_file()
    
    def frame_reader(self, rtsp_url, frame_queue, stop_event):
        """Thread function to read frames from RTSP stream"""
        # Open RTSP stream
        cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        
        # Check if opened successfully
        if not cap.isOpened():
            print("Failed to connect to RTSP stream in reader thread")
            stop_event.set()
            # Signal connection failure to main thread
            self.change_pixmap_signal.emit(np.zeros((480, 640, 3), dtype=np.uint8))
            return
        
        print("RTSP connection established in reader thread")
        
        consecutive_failures = 0
        max_failures = 5
        
        while not stop_event.is_set():
            ret, frame = cap.read()
            
            if not ret:
                consecutive_failures += 1
                print(f"Frame read failure {consecutive_failures}/{max_failures}")
                
                if consecutive_failures >= max_failures:
                    print("Too many failures. Reconnecting...")
                    cap.release()
                    time.sleep(1)
                    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
                    consecutive_failures = 0
                    
                    if not cap.isOpened():
                        print("Failed to reconnect")
                        stop_event.set()
                        # Signal connection failure to main thread
                        self.change_pixmap_signal.emit(np.zeros((480, 640, 3), dtype=np.uint8))
                        break
                
                time.sleep(0.5)
                continue
                
            consecutive_failures = 0
            
            # Keep only the most recent frame
            while not frame_queue.empty():
                try:
                    frame_queue.get_nowait()
                    frame_queue.task_done()
                except queue.Empty:
                    break
            
            frame_queue.put(frame)
            
            # Small delay to prevent overwhelming CPU
            time.sleep(0.01)
        
        # Clean up
        cap.release()
        print("Frame reader thread stopped")
    
    def process_frames_from_queue(self):
        """Process frames from the queue with YOLO model"""
        fps = 0
        frame_count = 0
        start_time = time.time()
        
        while not self.stop_event.is_set() and self.running:
            try:
                # Get the latest frame
                frame = self.frame_queue.get(timeout=1.0)
                self.frame_queue.task_done()
                
                # Update FPS calculation
                frame_count += 1
                elapsed_time = time.time() - start_time
                if elapsed_time >= 1.0:
                    fps = frame_count / elapsed_time
                    frame_count = 0
                    start_time = time.time()
                
                # Process with YOLO
                results = self.model(frame, verbose=False)
                
                # Draw results
                if results and len(results) > 0:
                    annotated_frame = results[0].plot()
                    
                    # Add FPS counter
                    cv2.putText(
                        annotated_frame,
                        f"FPS: {fps:.1f}",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2
                    )
                    
                    self.change_pixmap_signal.emit(annotated_frame)
                else:
                    # Add FPS to original frame
                    cv2.putText(
                        frame,
                        f"FPS: {fps:.1f}",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2
                    )
                    
                    self.change_pixmap_signal.emit(frame)
                
            except queue.Empty:
                # No frames available
                continue
            except Exception as e:
                print(f"Error in processing: {e}")
    
    def process_video_file(self):
        """Process a video file with YOLO model"""
        cap = cv2.VideoCapture(self.source_path)
        
        # Check if opened successfully
        if not cap.isOpened():
            print(f"Error: Unable to open video file at {self.source_path}")
            return
        
        # Get video properties
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        if video_fps <= 0:
            video_fps = 30  # Default to 30 fps if detection fails
        
        fps = 0
        frame_count = 0
        start_time = time.time()
        last_frame_time = time.time()
        
        while self.running:
            # Handle seeking
            if self.seek_position >= 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, self.seek_position)
                self.current_frame_position = self.seek_position
                self.seek_position = -1
                last_frame_time = time.time()  # Reset timing after seeking
            
            # Handle pause
            if self.paused:
                time.sleep(0.05)  # Reduce CPU usage while paused
                continue
                
            # Timing control for playback speed
            current_time = time.time()
            target_frame_time = 1.0 / (video_fps * self.playback_speed)
            elapsed = current_time - last_frame_time
            
            # Only process next frame if enough time has passed
            if elapsed < target_frame_time:
                sleep_time = max(0.001, target_frame_time - elapsed)
                time.sleep(sleep_time)
                continue
                
            # Read next frame
            ret, frame = cap.read()
            if not ret:
                # If we reach the end, emit the last position update and break
                self.update_position_signal.emit(self.current_frame_position, self.total_frames)
                break
            
            # Update timing for next frame
            last_frame_time = time.time()
            
            # Update current position
            self.current_frame_position = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            self.update_position_signal.emit(self.current_frame_position, self.total_frames)
                
            # Update FPS calculation
            frame_count += 1
            elapsed_time = time.time() - start_time
            if elapsed_time >= 1.0:
                fps = frame_count / elapsed_time
                frame_count = 0
                start_time = time.time()
            
            try:
                # Process with YOLO
                results = self.model(frame, verbose=False)
                
                # Draw results
                if results and len(results) > 0:
                    annotated_frame = results[0].plot()
                    
                    # Add FPS counter and frame position
                    cv2.putText(
                        annotated_frame,
                        f"FPS: {fps:.1f} | Frame: {self.current_frame_position}/{self.total_frames}",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        2
                    )
                    
                    self.change_pixmap_signal.emit(annotated_frame)
                else:
                    # Add FPS to original frame
                    cv2.putText(
                        frame,
                        f"FPS: {fps:.1f} | Frame: {self.current_frame_position}/{self.total_frames}",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        2
                    )
                    
                    self.change_pixmap_signal.emit(frame)
                    
            except Exception as e:
                print(f"Error processing frame: {e}")
                self.change_pixmap_signal.emit(frame)
            
        # Release resources
        cap.release()
    
    def stop(self):
        self.running = False
        self.stop_event.set()
        self.wait()
        
    def toggle_pause(self):
        self.paused = not self.paused
        return self.paused
        
    def set_speed(self, speed):
        self.playback_speed = speed
        
    def seek(self, position):
        self.seek_position = position


class YOLODetectionApp(QMainWindow):
    def __init__(self, model_path):
        super().__init__()
        
        self.model_path = model_path
        self.video_thread = None
        self.current_source_type = None  # 'rtsp' or 'file'
        
        self.init_ui()
        
    def init_ui(self):
        # Set window properties
        self.setWindowTitle("Fire warning - Yolo 11 demo")
        self.setGeometry(100, 100, 900, 600)
        
        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)  # Reduce outer margins
        main_layout.setSpacing(5)  # Reduce spacing between elements
        
        # Create top control layout
        top_control_layout = QHBoxLayout()
        top_control_layout.setContentsMargins(0, 0, 0, 5)  # Reduce margins
        
        # RTSP URL input
        self.rtsp_input = QLineEdit()
        self.rtsp_input.setPlaceholderText("Enter URL...")
        top_control_layout.addWidget(self.rtsp_input)
        
        # Open Stream button
        self.open_stream_btn = QPushButton("Open Stream")
        self.open_stream_btn.clicked.connect(self.open_stream)
        self.open_stream_btn.setMaximumWidth(100)
        top_control_layout.addWidget(self.open_stream_btn)
        
        # Open File button
        self.open_file_btn = QPushButton("Open File")
        self.open_file_btn.clicked.connect(self.open_file)
        self.open_file_btn.setMaximumWidth(100)
        top_control_layout.addWidget(self.open_file_btn)
        
        # Add top control layout to main layout
        main_layout.addLayout(top_control_layout)
        
        # Create display widget container (to control visibility)
        self.display_container = QWidget()
        display_container_layout = QVBoxLayout(self.display_container)
        display_container_layout.setContentsMargins(0, 0, 0, 0)
        
        # Create display label for stream window within the container
        self.display_label = QLabel()
        self.display_label.setAlignment(Qt.AlignCenter)
        self.display_label.setStyleSheet("background-color: black;")
        self.display_label.setMinimumHeight(400)  # Set minimum height for display
        display_container_layout.addWidget(self.display_label)
        
        # Add the display container to main layout
        main_layout.addWidget(self.display_container)
        
        # Initially hide the display container
        self.display_container.setVisible(False)
        
        # Create "No Content" placeholder label
        self.no_content_label = QLabel("No stream or video loaded.\nUse the buttons above to open media.")
        self.no_content_label.setAlignment(Qt.AlignCenter)
        self.no_content_label.setStyleSheet("font-size: 14pt; color: gray;")
        self.no_content_label.setMinimumHeight(400)  # Match display height
        main_layout.addWidget(self.no_content_label)
        
        # Create compact video control layout (initially hidden)
        self.video_control_layout = QVBoxLayout()
        self.video_control_layout.setContentsMargins(0, 0, 0, 0)  # No margins
        self.video_control_layout.setSpacing(2)  # Minimal spacing
        
        # Progress slider - more compact
        slider_layout = QHBoxLayout()
        slider_layout.setContentsMargins(0, 0, 0, 0)
        self.position_slider = QSlider(Qt.Horizontal)
        self.position_slider.setEnabled(False)
        self.position_slider.valueChanged.connect(self.slider_value_changed)
        self.position_slider.sliderPressed.connect(self.slider_pressed)
        self.position_slider.sliderReleased.connect(self.slider_released)
        slider_layout.addWidget(self.position_slider)
        self.video_control_layout.addLayout(slider_layout)
        
        # Playback controls - horizontal and compact
        playback_controls = QHBoxLayout()
        playback_controls.setContentsMargins(0, 0, 0, 0)
        
        # Play/Pause button - smaller
        self.play_pause_btn = QPushButton("Pause")
        self.play_pause_btn.setEnabled(False)
        self.play_pause_btn.clicked.connect(self.toggle_play_pause)
        self.play_pause_btn.setMaximumWidth(120)
        playback_controls.addWidget(self.play_pause_btn)
        
        # Speed control - more compact
        playback_controls.addWidget(QLabel("Speed:"))
        self.speed_combo = QComboBox()
        self.speed_combo.addItems(["0.25x", "0.5x", "1.0x", "1.5x", "2.0x"])
        self.speed_combo.setCurrentIndex(2)  # Default to 1.0x
        self.speed_combo.setEnabled(False)
        self.speed_combo.currentIndexChanged.connect(self.change_speed)
        self.speed_combo.setMaximumWidth(100)
        playback_controls.addWidget(self.speed_combo)
        
        # Position label - right-aligned
        self.position_label = QLabel("0 / 0")
        self.position_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.position_label.setMinimumWidth(100)
        self.position_label.setMaximumWidth(150)
        playback_controls.addWidget(self.position_label)
        
        # Add stretching space to push controls to the edges
        playback_controls.addStretch(1)
        
        # Add playback controls to video control layout
        self.video_control_layout.addLayout(playback_controls)
        
        # Add video control layout to main layout (hidden initially)
        self.video_controls_widget = QWidget()
        self.video_controls_widget.setLayout(self.video_control_layout)
        self.video_controls_widget.setVisible(False)
        self.video_controls_widget.setMaximumHeight(70)  # Limit maximum height
        main_layout.addWidget(self.video_controls_widget)
        
        # Create bottom control layout - compact
        bottom_control_layout = QHBoxLayout()
        bottom_control_layout.setContentsMargins(0, 5, 0, 0)
        
        # Close Stream button
        self.close_stream_btn = QPushButton("Close Stream/Video")
        self.close_stream_btn.clicked.connect(self.close_stream)
        self.close_stream_btn.setEnabled(False)  # Initially disabled
        bottom_control_layout.addWidget(self.close_stream_btn)
        
        # Add bottom control layout to main layout
        main_layout.addLayout(bottom_control_layout)
        
        # Flag for slider updates
        self.slider_is_being_dragged = False
        
    def open_stream(self):
        # Get RTSP URL from input box
        rtsp_url = self.rtsp_input.text().strip()
        if not rtsp_url:
            return
            
        # Close any existing stream
        self.close_stream()
        
        # Create and start a new video thread
        self.video_thread = VideoThread('rtsp', rtsp_url, self.model_path)
        self.video_thread.change_pixmap_signal.connect(self.update_image)
        self.video_thread.start()
        
        # Remember source type
        self.current_source_type = 'rtsp'
        
        # Update UI state
        self.close_stream_btn.setEnabled(True)
        self.open_stream_btn.setEnabled(False)
        self.open_file_btn.setEnabled(False)
        
        # Hide placeholder and show display container
        self.no_content_label.setVisible(False)
        self.display_container.setVisible(True)
        
        # Hide video controls for RTSP stream
        self.video_controls_widget.setVisible(False)
        self.play_pause_btn.setEnabled(False)
        self.speed_combo.setEnabled(False)
        self.position_slider.setEnabled(False)
        
    def open_file(self):
        # Open file dialog to select a video file
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Video File", "", "Video Files (*.mp4 *.avi *.mov *.mkv)"
        )
        
        if not file_path:
            return
            
        # Close any existing stream
        self.close_stream()
        
        # Create and start a new video thread
        self.video_thread = VideoThread('file', file_path, self.model_path)
        self.video_thread.change_pixmap_signal.connect(self.update_image)
        self.video_thread.update_position_signal.connect(self.update_position)
        self.video_thread.start()
        
        # Remember source type
        self.current_source_type = 'file'
        
        # Update UI state
        self.close_stream_btn.setEnabled(True)
        self.open_stream_btn.setEnabled(False)
        self.open_file_btn.setEnabled(False)
        
        # Hide placeholder and show display container
        self.no_content_label.setVisible(False)
        self.display_container.setVisible(True)
        
        # Show and enable video controls for file playback
        self.video_controls_widget.setVisible(True)
        self.play_pause_btn.setEnabled(True)
        self.play_pause_btn.setText("Pause")
        self.speed_combo.setEnabled(True)
        self.position_slider.setEnabled(True)
        
    def close_stream(self):
        # Stop and delete the video thread if it exists
        if self.video_thread and self.video_thread.isRunning():
            self.video_thread.stop()
            self.video_thread = None
        
        # Hide display container and show placeholder instead
        self.display_container.setVisible(False)
        self.no_content_label.setVisible(True)
        
        # Update UI state
        self.close_stream_btn.setEnabled(False)
        self.open_stream_btn.setEnabled(True)
        self.open_file_btn.setEnabled(True)
        self.video_controls_widget.setVisible(False)
        self.position_label.setText("0 / 0")
        self.position_slider.setValue(0)
        
        # Reset source type
        self.current_source_type = None
        
    def toggle_play_pause(self):
        if self.video_thread and self.current_source_type == 'file':
            paused = self.video_thread.toggle_pause()
            self.play_pause_btn.setText("Play" if paused else "Pause")
            
    def change_speed(self, index):
        if self.video_thread and self.current_source_type == 'file':
            speed_options = [0.25, 0.5, 1.0, 1.5, 2.0]
            self.video_thread.set_speed(speed_options[index])
            
    def slider_pressed(self):
        """Called when slider is pressed - store current pause state"""
        if self.video_thread and self.current_source_type == 'file':
            # Remember if the video was playing or paused
            self.was_playing = not self.video_thread.paused
            self.slider_is_being_dragged = True
            
    def slider_released(self):
        """Called when slider is released - seek and restore playback state"""
        if self.video_thread and self.current_source_type == 'file':
            # Seek to the position
            position = self.position_slider.value()
            self.video_thread.seek(position)
            
            # If it was playing before, ensure it continues playing
            if hasattr(self, 'was_playing') and self.was_playing:
                self.video_thread.paused = False
                self.play_pause_btn.setText("Pause")
                
            self.slider_is_being_dragged = False
    
    def slider_value_changed(self, position):
        """Called when slider value changes - handle direct clicks and jumps"""
        if self.video_thread and self.current_source_type == 'file':
            # Update position label
            self.position_label.setText(f"{position} / {self.video_thread.total_frames}")
            
            # If this is a direct click (not a drag operation) and not from auto-updates
            if not self.slider_is_being_dragged and position != self.video_thread.current_frame_position:
                # Remember if the video was playing or paused
                self.was_playing = not self.video_thread.paused
                
                # Seek to the position
                self.video_thread.seek(position)
                
                # If it was playing before, ensure it continues playing
                if hasattr(self, 'was_playing') and self.was_playing:
                    self.video_thread.paused = False
                    self.play_pause_btn.setText("Pause")
        
    @pyqtSlot(np.ndarray)
    def update_image(self, cv_img):
        # Check if it's a black image indicating connection failure
        if cv_img.shape == (480, 640, 3) and np.sum(cv_img) == 0:
            # Show connection failure message and close stream
            QMessageBox.critical(
                self,
                "Connection Failed",
                "Failed to connect to the RTSP stream. Please check the URL and try again.",
                QMessageBox.Ok
            )
            self.close_stream()
            return
            
        # Convert from OpenCV BGR format to RGB format
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        
        # Convert to Qt format and update display
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        
        # Scale to fit display area while maintaining aspect ratio
        scaled_pixmap = pixmap.scaled(
            self.display_label.size(), 
            Qt.KeepAspectRatio, 
            Qt.SmoothTransformation
        )
        
        self.display_label.setPixmap(scaled_pixmap)
        
    @pyqtSlot(int, int)
    def update_position(self, current_frame, total_frames):
        # Only update slider if it's not being dragged
        if not hasattr(self, 'slider_is_being_dragged') or not self.slider_is_being_dragged:
            # Update position slider without triggering the valueChanged signal
            self.position_slider.blockSignals(True)
            self.position_slider.setRange(0, total_frames)
            self.position_slider.setValue(current_frame)
            self.position_slider.blockSignals(False)
            
            # Update position label (only if not being dragged)
            self.position_label.setText(f"{current_frame} / {total_frames}")


if __name__ == "__main__":
    # Path to your trained YOLO v11 model
    yolo_model_path = r"best.pt"
    
    # Create and run the application
    app = QApplication(sys.argv)
    window = YOLODetectionApp(yolo_model_path)
    window.show()
    sys.exit(app.exec_())