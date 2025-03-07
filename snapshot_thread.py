from PyQt5.QtCore import QThread, pyqtSignal
import requests
import cv2
import numpy as np
import time
from ultralytics import YOLO


class SnapshotThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)
    
    def __init__(self, snapshot_url, model_path):
        super().__init__()
        self.snapshot_url = snapshot_url
        self.model_path = model_path
        self.running = True
        self.fps = 0
        self.frame_count = 0
        self.start_time = time.time()
        
        # Load YOLO model
        try:
            self.model = YOLO(model_path)
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.running = False
            return
    
    def run(self):
        while self.running:
            try:
                response = requests.get(self.snapshot_url, timeout=2)
                if response.status_code == 200:
                    img_array = np.frombuffer(response.content, np.uint8)
                    frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                    
                    if frame is not None:
                        results = self.model(frame, verbose=False)
                        annotated_frame = results[0].plot() if results else frame
                        
                        # FPS calculation
                        self.frame_count += 1
                        elapsed_time = time.time() - self.start_time
                        if elapsed_time >= 1.0:
                            self.fps = self.frame_count / elapsed_time
                            self.frame_count = 0
                            self.start_time = time.time()
                        
                        # Display FPS on the frame
                        cv2.putText(
                            annotated_frame,
                            f"FPS: {self.fps:.1f}",
                            (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 255, 0),
                            2
                        )
                        
                        self.change_pixmap_signal.emit(annotated_frame)
                
                time.sleep(0.2)  # 5 FPS
            except Exception as e:
                print(f"Snapshot stream error: {e}")
                time.sleep(1)
                
    def stop(self):
        self.running = False
        self.wait()

