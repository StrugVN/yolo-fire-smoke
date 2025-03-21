import streamlit as st
import cv2
import numpy as np
import time
import tempfile
import os
from threading import Thread
import requests
from PIL import Image
from io import BytesIO
import pygame

# Set page configuration
st.set_page_config(
    page_title="Fire Detection System",
    page_icon="🔥",
    layout="wide"
)

# Initialize pygame for sound (optional - streamlit doesn't handle sound well)
try:
    pygame.mixer.init()
    has_sound = True
    try:
        beep_sound = pygame.mixer.Sound("alarm_beep.wav")
    except:
        # Create a fallback beep
        sample_rate = 44100
        duration = 0.2
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        tone = np.sin(2 * np.pi * 880 * t)
        tone = (tone * 32767).astype(np.int16)
        beep_sound = pygame.mixer.Sound(pygame.sndarray.make_sound(tone))
except:
    has_sound = False
    st.warning("Sound initialization failed - alarm sounds will be unavailable")

# Make the model loading more robust to handle file paths
@st.cache_resource
def load_model(model_path):
    """Load YOLO model with caching"""
    try:
        import os
        from ultralytics import YOLO
        
        # Check if file exists
        if not os.path.isfile(model_path):
            st.error(f"Model file not found: {model_path}")
            return None
            
        # Check if it's a valid model file
        valid_extensions = ['.pt', '.pth', '.weights']
        if not any(model_path.endswith(ext) for ext in valid_extensions):
            st.warning(f"Model file may not be a valid YOLO model: {model_path}")
        
        # Try to load the model
        model = YOLO(model_path)
        st.success(f"YOLO model loaded successfully: {model_path}")
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Ensure YOLO package is imported at the top
try:
    from ultralytics import YOLO
except ImportError:
    st.error("Error: ultralytics package not found. Please install it with 'pip install ultralytics'")
    st.stop()

# Danger calculation logic (simplified from original DangerMeter class)
def calculate_danger(results, img_width, img_height):
    """Calculate danger level based on detection results"""
    if not results or not hasattr(results, 'boxes') or results.boxes is None:
        return 0, 0, 0
    
    # Get image dimensions
    total_screen_area = max(1, img_height * img_width)
    
    # Create mask images for fire and smoke to account for overlaps
    fire_mask = np.zeros((img_height, img_width), dtype=np.uint8)
    smoke_mask = np.zeros((img_height, img_width), dtype=np.uint8)
    
    has_fire = False
    has_smoke = False
    
    # Extract bounding boxes and classes
    boxes = results.boxes
    if len(boxes) == 0:
        return 0, 0, 0
    
    # Process each detection
    for i in range(len(boxes)):
        # Get box coordinates and class
        box = boxes[i].xyxy[0].cpu().numpy() if hasattr(boxes[i], 'xyxy') else boxes[i].cpu().numpy()
        cls = int(boxes[i].cls.cpu().numpy()[0]) if hasattr(boxes[i], 'cls') else 0
        
        # Get integer coordinates for mask
        x1, y1, x2, y2 = [int(coord) for coord in box[:4]]
        
        # Keep coordinates within image bounds
        x1 = max(0, min(x1, img_width-1))
        x2 = max(0, min(x2, img_width-1))
        y1 = max(0, min(y1, img_height-1))
        y2 = max(0, min(y2, img_height-1))
        
        # Add to appropriate mask
        if cls == 0:  # Fire class
            fire_mask[y1:y2, x1:x2] = 1
            has_fire = True
        elif cls == 1:  # Smoke class
            smoke_mask[y1:y2, x1:x2] = 1
            has_smoke = True
    
    # Calculate areas from masks (accounting for overlaps)
    fire_area = np.sum(fire_mask)
    smoke_area = np.sum(smoke_mask)
    
    # Calculate coverage percentages
    fire_coverage = (fire_area / total_screen_area) * 100
    smoke_coverage = (smoke_area / total_screen_area) * 100
    
    # Simple danger calculation formula (simplified from original)
    if fire_coverage > 0:
        fire_risk = min(100, 20 + 30 * np.log1p(fire_coverage))
    else:
        fire_risk = 0
    
    if smoke_coverage > 0:
        smoke_risk = min(100, 10 + 20 * np.log1p(smoke_coverage))
    else:
        smoke_risk = 0
    
    danger_level = max(fire_risk, smoke_risk)
    
    return danger_level, fire_coverage, smoke_coverage

# Function to process detections for alarm
def process_detections(results):
    """Process detection results to trigger alarms"""
    has_smoke = False
    has_fire = False
    smoke_count = 0
    fire_count = 0
    
    # Check for smoke and fire in the detections
    if hasattr(results, 'boxes') and results.boxes is not None:
        boxes = results.boxes
        class_ids = boxes.cls.cpu().numpy() if hasattr(boxes, 'cls') else []
        
        # Check each detected class
        for class_id in class_ids:
            class_id = int(class_id)  # Convert tensor to int if needed
            if class_id == 0:  # Assuming class 0 is fire
                has_fire = True
                fire_count += 1
            elif class_id == 1:  # Assuming class 1 is smoke
                has_smoke = True
                smoke_count += 1
    
    return has_fire, has_smoke, fire_count, smoke_count

# Function to capture frames from RTSP stream
def capture_rtsp_frames(rtsp_url, stop_flag):
    """Capture frames from RTSP stream"""
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
    
    if not cap.isOpened():
        st.session_state.connection_error = True
        return
    
    consecutive_failures = 0
    max_failures = 5
    
    while not stop_flag():
        ret, frame = cap.read()
        
        if not ret:
            consecutive_failures += 1
            if consecutive_failures >= max_failures:
                cap.release()
                time.sleep(1)
                cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
                consecutive_failures = 0
                
                if not cap.isOpened():
                    st.session_state.connection_error = True
                    break
            
            time.sleep(0.5)
            continue
        
        consecutive_failures = 0
        
        # Ensure frame is BGR (3 channels)
        if len(frame.shape) == 2:  # Grayscale
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif frame.shape[2] == 4:  # RGBA
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
        
        # Store the frame in session state
        st.session_state.current_frame = frame.copy()
        
        # Small delay to prevent overwhelming CPU
        time.sleep(0.05)
    
    cap.release()

# Function to capture frames from MJPEG stream
def capture_mjpeg_frames(url, stop_flag):
    """Capture frames from MJPEG/snapshot stream"""
    while not stop_flag():
        try:
            response = requests.get(url, timeout=2)
            if response.status_code == 200:
                img_array = np.frombuffer(response.content, np.uint8)
                frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                
                if frame is not None:
                    # Ensure frame is BGR (3 channels)
                    if len(frame.shape) == 2:  # Grayscale
                        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                    elif frame.shape[2] == 4:  # RGBA
                        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
                        
                    st.session_state.current_frame = frame.copy()
            
            time.sleep(0.2)  # 5 FPS
        except Exception as e:
            st.session_state.connection_error = True
            time.sleep(1)

# Function to play alarm sound (this will not work well in Streamlit)
def play_alarm_sound():
    """Play alarm sound in a loop"""
    if not has_sound:
        return
        
    current_interval = 1.0
    min_interval = 0.3
    
    while st.session_state.alarm_active and not st.session_state.stop_alarm_thread:
        if not st.session_state.alarm_muted:
            beep_sound.play()
            current_interval = max(min_interval, current_interval - 0.05)
        
        time.sleep(current_interval)

# Initialize session state variables
if 'started' not in st.session_state:
    st.session_state.started = False
    st.session_state.model_path = "best.pt"
    st.session_state.model = None
    st.session_state.rtsp_url = ""
    st.session_state.current_frame = None
    st.session_state.annotated_frame = None
    st.session_state.current_source_type = None
    st.session_state.stop_capture = False
    st.session_state.capture_thread = None
    st.session_state.connection_error = False
    st.session_state.smoke_alarm_active = False
    st.session_state.fire_alarm_active = False
    st.session_state.alarm_active = False
    st.session_state.alarm_muted = False
    st.session_state.stop_alarm_thread = False
    st.session_state.alarm_thread = None
    st.session_state.smoke_count = 0
    st.session_state.fire_count = 0
    st.session_state.danger_level = 0
    st.session_state.fire_coverage = 0
    st.session_state.smoke_coverage = 0
    st.session_state.video_position = 0
    st.session_state.video_total_frames = 0
    st.session_state.video_fps = 30
    st.session_state.video_paused = False
    st.session_state.last_update_time = time.time()
    
    # Detection parameters
    st.session_state.smoke_detection_time = 0
    st.session_state.fire_detection_time = 0
    st.session_state.last_smoke_time = 0
    st.session_state.last_fire_time = 0
    st.session_state.last_clear_time = 0
    st.session_state.smoke_detection_required = 3  # 3 seconds
    st.session_state.fire_detection_required = 1   # 1 second
    st.session_state.error_margin = 1  # 1 second error margin
    st.session_state.clear_required = 3  # 3 seconds clear before stopping alarm

# Main app layout
st.title("Fire Detection System")

# Sidebar for controls
with st.sidebar:
    st.header("Controls")
    
    # Model selection
    uploaded_model = st.file_uploader("Upload YOLO model (.pt file)", type=['pt', 'pth', 'weights'])
    if uploaded_model:
        # Save uploaded model to temp file
        temp_model = tempfile.NamedTemporaryFile(delete=False, suffix='.pt')
        temp_model.write(uploaded_model.getvalue())
        temp_model.close()
        st.session_state.model_path = temp_model.name
        st.session_state.model = None  # Reset model to force reload
    
    # Input source controls
    st.subheader("Input Source")
    source_type = st.radio("Select source type", ["RTSP Stream", "MJPEG Stream", "Video File", "Image File"])
    
    if source_type in ["RTSP Stream", "MJPEG Stream"]:
        st.session_state.rtsp_url = st.text_input("Enter URL", st.session_state.rtsp_url)
    elif source_type == "Video File":
        uploaded_video = st.file_uploader("Upload video file", type=['mp4', 'avi', 'mov', 'mkv'])
    elif source_type == "Image File":
        uploaded_image = st.file_uploader("Upload image file", type=['jpg', 'jpeg', 'png', 'bmp'])
    
    # Start/Stop button
    col1, col2 = st.columns(2)
    with col1:
        start_button = st.button("Start Detection")
    with col2:
        stop_button = st.button("Stop Detection")

# Main content area
col1, col2 = st.columns([3, 1])

with col1:
    # Display area
    if st.session_state.current_frame is None and not st.session_state.started:
        st.info("No stream or video loaded. Use the controls to start detection.")
    
    # Create a placeholder for the video display
    video_display = st.empty()
    
    # Progress bar for video files
    if st.session_state.current_source_type == "video" and st.session_state.started:
        video_progress = st.slider(
            "Video progress", 
            0, 
            max(1, st.session_state.video_total_frames), 
            st.session_state.video_position,
            key="video_progress"
        )
        
        # Handle seeking when slider is moved
        if video_progress != st.session_state.video_position:
            st.session_state.video_position = video_progress
        
        # Play/Pause button for video
        if st.button("Play/Pause Video"):
            st.session_state.video_paused = not st.session_state.video_paused

with col2:
    # Alarm status display
    st.subheader("Alarm Status")
    
    # Danger meter
    st.write("Risk Level:")
    danger_progress = st.progress(0)
    danger_text = st.empty()
    
    # Fire and smoke indicators
    fire_indicator = st.empty()
    smoke_indicator = st.empty()
    
    # Mute button
    if st.session_state.alarm_active:
        if st.button("Mute Alarm"):
            st.session_state.alarm_muted = True
            # In a real app, you'd stop the sound here

# Fix the YOLO model loading and error handling
if start_button:
    # Clear any previous capture
    if st.session_state.capture_thread and st.session_state.capture_thread.is_alive():
        st.session_state.stop_capture = True
        st.session_state.capture_thread.join(timeout=1.0)
    
    # Reset error flags
    st.session_state.connection_error = False
    
    # Load model if needed
    if st.session_state.model is None:
        with st.spinner("Loading YOLO model..."):
            st.session_state.model = load_model(st.session_state.model_path)
        
        if st.session_state.model is None:
            st.error(f"Failed to load YOLO model from {st.session_state.model_path}. Please check if the file exists and is a valid YOLO model.")
            st.stop()
    
    # Handle different source types
    if source_type == "RTSP Stream" and st.session_state.rtsp_url:
        st.session_state.current_source_type = "rtsp"
        st.session_state.stop_capture = False
        st.session_state.capture_thread = Thread(
            target=capture_rtsp_frames,
            args=(st.session_state.rtsp_url, lambda: st.session_state.stop_capture)
        )
        st.session_state.capture_thread.daemon = True
        st.session_state.capture_thread.start()
        st.session_state.started = True
        
    elif source_type == "MJPEG Stream" and st.session_state.rtsp_url:
        st.session_state.current_source_type = "mjpeg"
        st.session_state.stop_capture = False
        st.session_state.capture_thread = Thread(
            target=capture_mjpeg_frames,
            args=(st.session_state.rtsp_url, lambda: st.session_state.stop_capture)
        )
        st.session_state.capture_thread.daemon = True
        st.session_state.capture_thread.start()
        st.session_state.started = True
        
# Add better error handling for video file loading
    elif source_type == "Video File" and 'uploaded_video' in locals() and uploaded_video is not None:
        # Save uploaded video to temp file
        try:
            st.session_state.current_source_type = "video"
            temp_video = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            temp_video.write(uploaded_video.getvalue())
            temp_video.close()
            
            # Get video properties
            cap = cv2.VideoCapture(temp_video.name)
            if not cap.isOpened():
                st.error(f"Failed to open video file. The file may be corrupted or in an unsupported format.")
                st.stop()
                
            st.session_state.video_total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            st.session_state.video_fps = cap.get(cv2.CAP_PROP_FPS)
            if st.session_state.video_fps <= 0:
                st.session_state.video_fps = 30
            cap.release()
            
            # Start a thread to process the video
            st.session_state.video_path = temp_video.name
            st.session_state.video_position = 0
            st.session_state.video_paused = False
            st.session_state.started = True
            
        except Exception as e:
            st.error(f"Error processing video file: {str(e)}")
            st.stop()
        
    elif source_type == "Image File" and 'uploaded_image' in locals() and uploaded_image is not None:
        # Process a single image
        st.session_state.current_source_type = "image"
        
        # Read image
        image = Image.open(uploaded_image)
        image_np = np.array(image)
        
        # Handle image channels properly
        if len(image_np.shape) == 3:
            if image_np.shape[2] == 4:  # RGBA
                # Convert RGBA to RGB
                image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)
            # Convert RGB to BGR for OpenCV
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        elif len(image_np.shape) == 2:  # Grayscale
            # Convert grayscale to BGR
            image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2BGR)
        
        st.session_state.current_frame = image_np
        st.session_state.started = True

# Handle stop button
if stop_button:
    # Stop capture thread
    st.session_state.stop_capture = True
    if st.session_state.capture_thread and st.session_state.capture_thread.is_alive():
        st.session_state.capture_thread.join(timeout=1.0)
    
    # Reset state
    st.session_state.current_frame = None
    st.session_state.annotated_frame = None
    st.session_state.started = False
    st.session_state.current_source_type = None
    st.session_state.smoke_alarm_active = False
    st.session_state.fire_alarm_active = False
    st.session_state.alarm_active = False
    st.session_state.danger_level = 0
    
    # Update UI
    danger_progress.progress(0)
    danger_text.write("Risk Level: Safe (0%)")
    fire_indicator.warning("FIRE (0)")
    smoke_indicator.info("SMOKE (0)")

# Process video file frame-by-frame
if st.session_state.started and st.session_state.current_source_type == "video" and not st.session_state.video_paused:
    try:
        cap = cv2.VideoCapture(st.session_state.video_path)
        
        if not cap.isOpened():
            st.error("Failed to open video file for processing.")
            st.session_state.started = False
            st.stop()
        
        # Seek to the current position
        cap.set(cv2.CAP_PROP_POS_FRAMES, st.session_state.video_position)
        
        # Read the current frame
        ret, frame = cap.read()
        if ret:
            # Ensure frame is BGR (3 channels)
            if len(frame.shape) == 2:  # Grayscale
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            elif frame.shape[2] == 4:  # RGBA
                frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
                
            st.session_state.current_frame = frame
            st.session_state.video_position += 1
        else:
            # End of video
            st.session_state.video_paused = True
        
        cap.release()
    except Exception as e:
        st.error(f"Error processing video frame: {str(e)}")
        st.session_state.video_paused = True

# Process current frame with YOLO model
if st.session_state.started and st.session_state.current_frame is not None:
    current_time = time.time()
    
    # Only process every 100ms to limit resource usage
    if current_time - st.session_state.last_update_time > 0.1:
        st.session_state.last_update_time = current_time
        
        # Make sure the image has 3 channels (RGB) not 4 (RGBA)
        frame = st.session_state.current_frame
        if len(frame.shape) == 3 and frame.shape[2] == 4:
            # Convert RGBA to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
        elif len(frame.shape) == 2:
            # Convert grayscale to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            
        # Run inference
        results = st.session_state.model(frame, verbose=False)
        
        if results and len(results) > 0:
            # Get frame dimensions
            h, w = st.session_state.current_frame.shape[:2]
            
            # Calculate danger level
            danger, fire_coverage, smoke_coverage = calculate_danger(results[0], w, h)
            st.session_state.danger_level = danger
            st.session_state.fire_coverage = fire_coverage
            st.session_state.smoke_coverage = smoke_coverage
            
            # Process detections for alarm
            has_fire, has_smoke, fire_count, smoke_count = process_detections(results[0])
            st.session_state.fire_count = fire_count
            st.session_state.smoke_count = smoke_count
            
            # Process smoke detection for alarm
            if has_smoke:
                # If this is the first smoke detection or within error margin
                if st.session_state.smoke_detection_time == 0 or (current_time - st.session_state.last_smoke_time) <= st.session_state.error_margin:
                    if st.session_state.smoke_detection_time == 0:
                        st.session_state.smoke_detection_time = current_time
                    
                    # Check if detection threshold reached
                    elapsed = current_time - st.session_state.smoke_detection_time
                    if elapsed >= st.session_state.smoke_detection_required and not st.session_state.smoke_alarm_active:
                        st.session_state.smoke_alarm_active = True
                        st.session_state.alarm_active = True
                
                st.session_state.last_smoke_time = current_time
            else:
                # If no smoke detected for longer than error margin
                if (current_time - st.session_state.last_smoke_time) > st.session_state.error_margin:
                    st.session_state.smoke_detection_time = 0
            
            # Process fire detection (similar logic)
            if has_fire:
                if st.session_state.fire_detection_time == 0 or (current_time - st.session_state.last_fire_time) <= st.session_state.error_margin:
                    if st.session_state.fire_detection_time == 0:
                        st.session_state.fire_detection_time = current_time
                    
                    elapsed = current_time - st.session_state.fire_detection_time
                    if elapsed >= st.session_state.fire_detection_required and not st.session_state.fire_alarm_active:
                        st.session_state.fire_alarm_active = True
                        st.session_state.alarm_active = True
                
                st.session_state.last_fire_time = current_time
            else:
                if (current_time - st.session_state.last_fire_time) > st.session_state.error_margin:
                    st.session_state.fire_detection_time = 0
            
            # Check to stop the alarm
            if (not has_smoke and not has_fire) and (st.session_state.smoke_alarm_active or st.session_state.fire_alarm_active):
                if st.session_state.last_clear_time == 0:
                    st.session_state.last_clear_time = current_time
                elif (current_time - st.session_state.last_clear_time) >= st.session_state.clear_required:
                    st.session_state.smoke_alarm_active = False
                    st.session_state.fire_alarm_active = False
                    st.session_state.alarm_active = False
                    st.session_state.alarm_muted = False
            else:
                st.session_state.last_clear_time = 0
            
            # Draw results
            annotated_frame = results[0].plot()
            
            # Add FPS and other info
            frame_type = st.session_state.current_source_type.upper()
            cv2.putText(
                annotated_frame,
                f"{frame_type} Mode",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )
            
            # Add coverage info
            coverage_text = f"Fire: {fire_coverage:.1f}% | Smoke: {smoke_coverage:.1f}%"
            cv2.putText(
                annotated_frame,
                coverage_text,
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 255),
                2
            )
            
            st.session_state.annotated_frame = annotated_frame
        else:
            # No detections
            frame = st.session_state.current_frame.copy()
            cv2.putText(
                frame,
                "No detections",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )
            st.session_state.annotated_frame = frame

# Update UI elements
if st.session_state.started:
    # Display the annotated frame
    if st.session_state.annotated_frame is not None:
        # Ensure frame is BGR (3 channels) before conversion
        frame = st.session_state.annotated_frame
        if len(frame.shape) == 2:  # Grayscale
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif frame.shape[2] == 4:  # RGBA
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
        
        # Convert from BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video_display.image(rgb_frame, channels="RGB", use_container_width=True)
    
    # Update danger meter
    danger_level = st.session_state.danger_level
    danger_progress.progress(int(danger_level))
    
    # Set danger text based on level
    if danger_level < 20:
        status = "Safe"
        danger_text.info(f"Risk Level: {status} ({int(danger_level)}%)")
    elif danger_level < 40:
        status = "Caution"
        danger_text.info(f"Risk Level: {status} ({int(danger_level)}%)")
    elif danger_level < 60:
        status = "Hazardous"
        danger_text.warning(f"Risk Level: {status} ({int(danger_level)}%)")
    elif danger_level < 80:
        status = "Critical"
        danger_text.warning(f"Risk Level: {status} ({int(danger_level)}%)")
    else:
        status = "DISASTEROUS!"
        danger_text.error(f"Risk Level: {status} ({int(danger_level)}%)")
    
    # Update fire and smoke indicators
    if st.session_state.fire_alarm_active:
        fire_indicator.error(f"FIRE ({st.session_state.fire_count})")
    else:
        fire_indicator.info(f"FIRE ({st.session_state.fire_count})")
    
    if st.session_state.smoke_alarm_active:
        smoke_indicator.warning(f"SMOKE ({st.session_state.smoke_count})")
    else:
        smoke_indicator.info(f"SMOKE ({st.session_state.smoke_count})")

# Check for connection error
if st.session_state.connection_error:
    st.error("Failed to connect to the stream. Please check the URL and try again.")
    # Reset flag after showing
    st.session_state.connection_error = False
    # Stop current capture
    st.session_state.stop_capture = True

# Display instructions in the sidebar
with st.sidebar:
    st.markdown("---")
    st.subheader("Instructions")
    st.markdown("""
    1. Select your input source type
    2. Provide the URL or upload a file
    3. Click 'Start Detection' to begin
    4. The system will detect fire and smoke
    5. Alarms will activate based on detections
    6. Use 'Stop Detection' to end the session
    """)
    
    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    This fire detection system uses YOLO to detect fire and smoke in video streams, 
    files, or images. When detections persist for a threshold period, alarms are activated.
    """)