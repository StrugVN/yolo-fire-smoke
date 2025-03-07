import cv2
import time
import queue
import threading
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal
from ultralytics import YOLO

class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)
    update_position_signal = pyqtSignal(int, int)  # current_frame, total_frames
    
    def __init__(self, source_type, source_path, model_path, alarm_system=None):
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
        
        # Add alarm system reference
        self.alarm_system = alarm_system
        
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
                
                # Process alarm detection if alarm system is available
                if self.alarm_system and results:
                    self.alarm_system.process_detections(results[0])
                
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
                
                # Process alarm detection if alarm system is available
                if self.alarm_system and results:
                    self.alarm_system.process_detections(results[0])
                
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