import streamlit as st
import cv2
import numpy as np
import time
import tempfile
import os
import requests
from ultralytics import YOLO

# ------------------------------
# Helper function: Calculate danger level
# ------------------------------
def calculate_danger(results):
    """
    Calculate the danger level based on YOLO detection results.
    Returns a tuple (danger_level, text) where danger_level is 0-100.
    This is a simplified version based on your DangerMeter code.
    """
    if not results or not hasattr(results, 'boxes') or results.boxes is None or len(results.boxes) == 0:
        return 0, "Fire: 0.0% | Smoke: 0.0% | Combined: 0.0%"

    # Get image dimensions from original image shape (assumes [height, width])
    img_height, img_width = results.orig_shape
    total_area = max(1, img_height * img_width)
    
    fire_mask = np.zeros((img_height, img_width), dtype=np.uint8)
    smoke_mask = np.zeros((img_height, img_width), dtype=np.uint8)
    
    for box in results.boxes:
        # Get bounding box coordinates and class
        coords = box.xyxy[0].cpu().numpy() if hasattr(box, 'xyxy') else box.cpu().numpy()
        cls = int(box.cls.cpu().numpy()[0]) if hasattr(box, 'cls') else 0
        x1, y1, x2, y2 = [int(coord) for coord in coords[:4]]
        # Clamp coordinates to image bounds
        x1 = max(0, min(x1, img_width - 1))
        x2 = max(0, min(x2, img_width - 1))
        y1 = max(0, min(y1, img_height - 1))
        y2 = max(0, min(y2, img_height - 1))
        if cls == 0:  # Fire detection
            fire_mask[y1:y2, x1:x2] = 1
        elif cls == 1:  # Smoke detection
            smoke_mask[y1:y2, x1:x2] = 1

    fire_area = np.sum(fire_mask)
    smoke_area = np.sum(smoke_mask)
    fire_coverage = (fire_area / total_area) * 100
    smoke_coverage = (smoke_area / total_area) * 100
    combined = min(100, fire_coverage + smoke_coverage)
    text = f"Fire: {fire_coverage:.1f}% | Smoke: {smoke_coverage:.1f}% | Combined: {combined:.1f}%"
    return combined, text

# ------------------------------
# Streamlit App UI
# ------------------------------
st.set_page_config(page_title="Fire Detection System", layout="wide")
st.title("Fire Detection System")

# Preload YOLO model into session state.
if 'model' not in st.session_state:
    st.session_state.model = None
    st.session_state.model_path = None

# Sidebar: Allow user to upload a model file.
uploaded_model = st.sidebar.file_uploader("Upload YOLO Model", type=["pt", "pth", "weights"])
if uploaded_model is not None:
    # Save uploaded file to a temporary file and load the model from it.
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as tmp_file:
        tmp_file.write(uploaded_model.read())
        tmp_file.flush()
        model_path = tmp_file.name
    try:
        st.session_state.model = YOLO(model_path)
        st.session_state.model_path = model_path
        st.sidebar.success("Model loaded successfully from uploaded file.")
    except Exception as e:
        st.sidebar.error(f"Error loading uploaded model: {e}")
else:
    # If no model is uploaded and no model loaded yet, load the default.
    if st.session_state.model is None:
        try:
            st.session_state.model = YOLO("best.pt")
            st.session_state.model_path = "best.pt"
            st.sidebar.info("Loaded default model (best.pt).")
        except Exception as e:
            st.sidebar.error(f"Error loading default model: {e}")

# Sidebar: Other controls.
st.sidebar.header("Settings")
source_type = st.sidebar.selectbox("Select Source", 
    ["RTSP Stream", "MJPEG Stream", "Video File", "Image"])

if source_type in ["RTSP Stream", "MJPEG Stream"]:
    url = st.sidebar.text_input("Stream URL", value="")
else:
    uploaded_file = st.sidebar.file_uploader("Upload File", 
                      type=["mp4", "avi", "mov", "mkv", "jpg", "jpeg", "png"])

# Start/Stop buttons.
if st.sidebar.button("Start"):
    st.session_state.run_stream = True
if st.sidebar.button("Stop"):
    st.session_state.run_stream = False

# Placeholders for image and metrics.
image_placeholder = st.empty()
danger_placeholder = st.empty()
fps_placeholder = st.empty()

# ------------------------------
# Main Loop: Stream Processing
# ------------------------------
def process_stream():
    cap = None
    tfile = None  # For temporary video file if needed

    if source_type == "RTSP Stream":
        if url == "":
            st.error("Please provide a valid URL.")
            return
        cap = cv2.VideoCapture(url)
    elif source_type == "MJPEG Stream":
        if url == "":
            st.error("Please provide a valid URL.")
            return
        # MJPEG stream will be handled using requests.
    elif source_type == "Video File":
        if uploaded_file is None:
            st.error("Please upload a video file.")
            return
        # Save uploaded file to a temporary file for cv2.VideoCapture.
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        cap = cv2.VideoCapture(tfile.name)

    if source_type == "Image":
        if uploaded_file is None:
            st.error("Please upload an image file.")
            return
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if frame is None:
            st.error("Error reading image.")
            return
        results = st.session_state.model(frame, verbose=False)
        annotated_frame = results[0].plot() if results and len(results) > 0 else frame
        danger_level, danger_text = calculate_danger(results[0]) if results and len(results) > 0 else (0, "")
        image_placeholder.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))
        danger_placeholder.progress(int(danger_level))
        danger_placeholder.text(danger_text)
        return

    prev_time = time.time()
    frame_count = 0

    # Loop to read frames and update display.
    while st.session_state.run_stream:
        if source_type == "MJPEG Stream":
            try:
                r = requests.get(url, timeout=2)
                if r.status_code == 200:
                    img_array = np.frombuffer(r.content, np.uint8)
                    frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                    if frame is None:
                        st.warning("Failed to decode image.")
                        continue
                else:
                    st.warning("Received non-200 response. Ending stream...")
                    break
            except Exception as e:
                st.error(f"Error fetching MJPEG frame: {e}")
                break
        else:
            ret, frame = cap.read()
            if not ret:
                st.warning("No frame received. Ending stream...")
                break

        # Run YOLO detection.
        try:
            results = st.session_state.model(frame, verbose=False)
        except Exception as e:
            st.error(f"Error during detection: {e}")
            break

        # Annotate frame if detections exist.
        if results and len(results) > 0:
            annotated_frame = results[0].plot()
        else:
            annotated_frame = frame.copy()

        # Calculate and display FPS.
        frame_count += 1
        curr_time = time.time()
        elapsed = curr_time - prev_time
        if elapsed > 1:
            fps = frame_count / elapsed
            frame_count = 0
            prev_time = curr_time
        else:
            fps = 0
        cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Calculate danger level (if detections available).
        if results and len(results) > 0:
            danger_level, danger_text = calculate_danger(results[0])
        else:
            danger_level, danger_text = 0, "No detections"

        # Update the image and danger meter.
        image_placeholder.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))
        danger_placeholder.progress(int(danger_level))
        danger_placeholder.text(danger_text)

        time.sleep(0.05)

    # Cleanup.
    if cap is not None:
        cap.release()
    if tfile is not None:
        try:
            os.unlink(tfile.name)
        except Exception:
            pass

if st.session_state.get("run_stream", False):
    process_stream()
