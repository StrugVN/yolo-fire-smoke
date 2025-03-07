import time
import threading
import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal
import pygame

class AlarmSystem(QObject):
    """
    Alarm system for fire and smoke detection
    Handles detection sequences and alarm sounds
    """
    alarm_status_signal = pyqtSignal(str, bool)  # type, is_active
    
    def __init__(self):
        super().__init__()
        # Initialize pygame for sound
        pygame.mixer.init()
        
        # Detection parameters
        self.smoke_detection_required = 3  # 3 seconds
        self.fire_detection_required = 1   # 1 second
        self.error_margin = 1  # 1 second error margin
        self.clear_required = 3  # 3 seconds clear before stopping alarm
        
        # Detection state tracking
        self.smoke_detection_time = 0
        self.fire_detection_time = 0
        self.last_smoke_time = 0
        self.last_fire_time = 0
        self.last_clear_time = 0
        
        # Alarm states
        self.smoke_alarm_active = False
        self.fire_alarm_active = False
        self.alarm_thread = None
        self.stop_alarm_thread = threading.Event()
        self.pause_alarm = False
        
        # Sound parameters
        self.beep_interval = 2.0  # Start with 1 second between beeps
        self.min_beep_interval = 0.5  # Fastest beep rate 
        self.beep_reduction_rate = 0.05  # How much to reduce interval each beep
        
        # Load sound
        try:
            self.beep_sound = pygame.mixer.Sound("alarm_beep.wav")
        except:
            # Create a fallback beep sound if file not found
            self._create_fallback_beep()
    
    def _create_fallback_beep(self):
        """Create a simple beep sound if the sound file is not available"""
        sample_rate = 44100
        duration = 0.2  # 200ms beep
        
        # Generate a simple sine wave
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        tone = np.sin(2 * np.pi * 880 * t)  # 880 Hz tone
        
        # Normalize and convert to 16-bit integers
        tone = (tone * 32767).astype(np.int16)
        
        # Create sound from the numpy array
        self.beep_sound = pygame.mixer.Sound(pygame.sndarray.make_sound(tone))
    
    def process_detections(self, result):
        """
        Process detection results to trigger alarms
        
        Args:
            result: Results object from YOLO model
        """
        current_time = time.time()
        has_smoke = False
        has_fire = False
        
        # Check for smoke and fire in the detections
        if hasattr(result, 'boxes') and result.boxes is not None:
            boxes = result.boxes
            class_ids = boxes.cls.cpu().numpy() if hasattr(boxes, 'cls') else []
            
            # Check each detected class
            for class_id in class_ids:
                class_id = int(class_id)  # Convert tensor to int if needed
                if class_id == 0:  # Assuming class 0 is fire
                    has_fire = True
                elif class_id == 1:  # Assuming class 1 is smoke
                    has_smoke = True
        
        # Process smoke detection
        if has_smoke:
            # If this is the first smoke detection or within error margin
            if self.smoke_detection_time == 0 or (current_time - self.last_smoke_time) <= self.error_margin:
                if self.smoke_detection_time == 0:
                    self.smoke_detection_time = current_time
                
                # Check if detection threshold reached
                elapsed = current_time - self.smoke_detection_time
                if elapsed >= self.smoke_detection_required and not self.smoke_alarm_active:
                    self.activate_smoke_alarm()
            
            self.last_smoke_time = current_time
        else:
            # If no smoke detected for longer than error margin
            if (current_time - self.last_smoke_time) > self.error_margin:
                self.smoke_detection_time = 0
        
        # Process fire detection (similar logic)
        if has_fire:
            if self.fire_detection_time == 0 or (current_time - self.last_fire_time) <= self.error_margin:
                if self.fire_detection_time == 0:
                    self.fire_detection_time = current_time
                
                elapsed = current_time - self.fire_detection_time
                if elapsed >= self.fire_detection_required and not self.fire_alarm_active:
                    self.activate_fire_alarm()
            
            self.last_fire_time = current_time
        else:
            if (current_time - self.last_fire_time) > self.error_margin:
                self.fire_detection_time = 0
        
        # Check to stop the alarm
        if (not has_smoke and not has_fire) and (self.smoke_alarm_active or self.fire_alarm_active):
            if self.last_clear_time == 0:
                self.last_clear_time = current_time
            elif (current_time - self.last_clear_time) >= self.clear_required:
                self.deactivate_alarms()
        else:
            self.last_clear_time = 0
    
    def activate_smoke_alarm(self):
        """Activate the smoke alarm"""
        self.smoke_alarm_active = True
        self.alarm_status_signal.emit("smoke", True)
        self._start_alarm_thread()
    
    def activate_fire_alarm(self):
        """Activate the fire alarm"""
        self.fire_alarm_active = True
        self.alarm_status_signal.emit("fire", True)
        self._start_alarm_thread()
    
    def deactivate_alarms(self):
        """Deactivate all alarms"""
        if self.smoke_alarm_active:
            self.smoke_alarm_active = False
            self.alarm_status_signal.emit("smoke", False)
        
        if self.fire_alarm_active:
            self.fire_alarm_active = False
            self.alarm_status_signal.emit("fire", False)
        
        # Stop the alarm thread
        if self.alarm_thread and self.alarm_thread.is_alive():
            self.stop_alarm_thread.set()
    
    def _start_alarm_thread(self):
        """Start the alarm thread if not already running"""
        if not self.alarm_thread or not self.alarm_thread.is_alive():
            self.stop_alarm_thread.clear()
            self.alarm_thread = threading.Thread(target=self._alarm_loop)
            self.alarm_thread.daemon = True
            self.alarm_thread.start()
    
    def _alarm_loop(self):
        """Alarm sound loop with intensifying beep rate"""
        current_interval = self.beep_interval
        
        while not self.stop_alarm_thread.is_set() and (self.smoke_alarm_active or self.fire_alarm_active):
            # Only play sound if not paused
            if not self.pause_alarm:
                # Play the beep sound
                self.beep_sound.play()
                
                # Gradually increase beep frequency (decrease interval)
                current_interval = max(self.min_beep_interval, current_interval - self.beep_reduction_rate)
            
            # Wait for the next beep (even when paused, to avoid CPU hogging)
            time.sleep(current_interval)
        
        # Reset beep interval for next alarm
        current_interval = self.beep_interval
        
    def set_pause(self, paused):
        """Pause or resume the alarm sounds"""
        self.pause_alarm = paused
    
    def stop(self):
        """Stop the alarm system"""
        self.deactivate_alarms()
        pygame.mixer.quit()
