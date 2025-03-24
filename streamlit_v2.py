import streamlit as st
import cv2
import numpy as np
import time
import tempfile
import os
import requests
from ultralytics import YOLO

@st.cache_resource
def load_model(model_path="/mount/src/yolo-fire-smoke/best.pt"):    # for deploy
    return YOLO(model_path)

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
    img_height, img_width = results.orig_shape
    total_area = max(1, img_height * img_width)
    
    fire_mask = np.zeros((img_height, img_width), dtype=np.uint8)
    smoke_mask = np.zeros((img_height, img_width), dtype=np.uint8)
    
    for box in results.boxes:
        coords = box.xyxy[0].cpu().numpy() if hasattr(box, 'xyxy') else box.cpu().numpy()
        cls = int(box.cls.cpu().numpy()[0]) if hasattr(box, 'cls') else 0
        x1, y1, x2, y2 = [int(coord) for coord in coords[:4]]
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

# Create a row with two columns: one for the title, one for the styled download button
col1, col2 = st.columns([3, 1])
with col1:
    st.title("Fire Detection System")
with col2:
    # We create a container that pushes the button to the bottom using flexbox.
    st.markdown(
        """
        <div class="bottom-container">
            <a href="https://drive.google.com/file/d/1tWsjRH6mMC1XvkdPuweDh2b4SxHhbVrA/view?usp=drive_link" 
               target="_blank" class="download-button">
                Download the app
            </a>
        </div>
        <style>
        .bottom-container {
            display: flex;
            flex-direction: column;
            justify-content: flex-end;
            margin-top: 20px;
        }
        .download-button {
            border: none;                  /* No border */
            background-color: #c3e6cb;     /* Light green background */
            color: #28a745;               /* Green text */
            padding: 10px 16px;
            border-radius: 12px;          /* Rounded corners */
            font-weight: bold;
            cursor: pointer;
            text-decoration: none;        /* Remove link underline */
            display: inline-block;        /* Keep button shape */
            text-align: center;
            width: 100%;
        }
        .download-button:hover {
            background-color: #a5d6a7;     /* Slightly darker on hover */
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# 1) If no model is in session state, set defaults
if 'model' not in st.session_state:
    st.session_state.model = None
    st.session_state.model_path = None

# 2) Radio buttons for source selection
st.sidebar.header("Settings")
source_type = st.sidebar.radio(
    "Select Source", 
    ["RTSP Stream", "MJPEG Stream", "Video File", "Image"],
    index=0  # Example default index
)

# 3) Stop any ongoing stream if the user changes source
if 'last_source' not in st.session_state:
    st.session_state.last_source = source_type
if source_type != st.session_state.last_source:
    st.session_state.run_stream = False
    st.session_state.last_source = source_type

# 4) Depending on the source, get URL or file
if source_type in ["RTSP Stream", "MJPEG Stream"]:
    url = st.sidebar.text_input("Stream URL", value="")
else:
    uploaded_file = st.sidebar.file_uploader(
        "Upload File (Video/Image)",
        type=["mp4", "avi", "mov", "mkv", "jpg", "jpeg", "png"]
    )

# 5) Model file uploader (just above Start/Stop)
uploaded_model = st.sidebar.file_uploader("Upload YOLO Model", type=["pt", "pth", "weights"])

# Attempt to load the model if a new model is uploaded
if uploaded_model is not None:
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
    # If no model is uploaded and we still have no model, load best.pt by default
    if st.session_state.model is None:
        try:
            st.session_state.model = model = load_model()
            st.session_state.model_path = "best.pt"
            st.sidebar.info("Loaded default model (best.pt).")
        except Exception as e:
            st.sidebar.error(f"Error loading default model 'best.pt': {e}")

# 6) Start/Stop buttons in the same row
start_col, stop_col = st.sidebar.columns(2)
with start_col:
    if st.button("Start"):
        st.session_state.run_stream = True
with stop_col:
    if st.button("Stop"):
        st.session_state.run_stream = False

# 7) Placeholders for Fire/Smoke counts, image, danger, fps
fire_smoke_placeholder = st.empty()  # Will show Fire(x) Smoke(x)
image_placeholder = st.empty()       # Main video frame
danger_placeholder = st.empty()      # Danger bar
fps_placeholder = st.empty()         # If you want to display FPS outside the frame

st.sidebar.markdown("---")
st.sidebar.markdown("**Version 0.0.0**")

# ------------------------------
# Main Loop: Stream Processing
# ------------------------------
def process_stream():
    cap = None
    tfile = None  # For temporary video file if needed

    # --- Setup capture based on source_type ---
    if source_type == "RTSP Stream":
        if url == "":
            st.error("Please provide a valid URL.")
            return
        # Use FFMPEG backend and reduce buffering.
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    elif source_type == "MJPEG Stream":
        if url == "":
            st.error("Please provide a valid URL.")
            return
        # MJPEG stream handled via HTTP requests.

    elif source_type == "Video File":
        if uploaded_file is None:
            st.error("Please upload a video file.")
            return
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        cap = cv2.VideoCapture(tfile.name)

    elif source_type == "Image":
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
        
        # Count Fire/Smoke for single image
        fire_count, smoke_count = 0, 0
        if results and len(results) > 0 and results[0].boxes is not None:
            for box in results[0].boxes:
                cls = int(box.cls.cpu().numpy()[0])
                if cls == 0:
                    fire_count += 1
                elif cls == 1:
                    smoke_count += 1
        fire_color = "#FF0000" if fire_count > 0 else "#000000"
        smoke_color = "#FFA500" if smoke_count > 0 else "#000000"

        fire_smoke_placeholder.markdown(f"""
        <div style="font-size:16px;">
          <b>Fire</b> (<span style="color:{fire_color};">{fire_count}</span>) &nbsp;&nbsp;
          <b>Smoke</b> (<span style="color:{smoke_color};">{smoke_count}</span>)
        </div>
        """, unsafe_allow_html=True)

        danger_level, danger_text = calculate_danger(results[0]) if results and len(results) > 0 else (0, "")
        image_placeholder.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))
        danger_placeholder.progress(int(danger_level))
        danger_placeholder.text(danger_text)
        return

    # --- Streaming loop for RTSP, MJPEG, or Video File ---
    prev_time = time.time()
    frame_count = 0

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
            # Flush stale frames to reduce lag (for RTSP or Video File)
            for _ in range(5):
                cap.grab()
            ret, frame = cap.read()
            if not ret:
                st.warning("No frame received. Ending stream...")
                break

        # YOLO detection
        try:
            results = st.session_state.model(frame, verbose=False)
        except Exception as e:
            st.error(f"Error during detection: {e}")
            break

        # Annotate frame if detections exist
        if results and len(results) > 0:
            annotated_frame = results[0].plot()
        else:
            annotated_frame = frame.copy()

        # Count Fire/Smoke
        fire_count, smoke_count = 0, 0
        if results and len(results) > 0 and results[0].boxes is not None:
            for box in results[0].boxes:
                cls = int(box.cls.cpu().numpy()[0])
                if cls == 0:
                    fire_count += 1
                elif cls == 1:
                    smoke_count += 1

        # Color the labels only if > 0
        fire_color = "#FF0000" if fire_count > 0 else "#000000"
        smoke_color = "#FFA500" if smoke_count > 0 else "#000000"
        fire_smoke_placeholder.markdown(f"""
        <div style="font-size:16px;">
          <b>Fire</b> (<span style="color:{fire_color};">{fire_count}</span>) &nbsp;&nbsp;
          <b>Smoke</b> (<span style="color:{smoke_color};">{smoke_count}</span>)
        </div>
        """, unsafe_allow_html=True)

        # Calculate and overlay FPS
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

        # Danger meter
        if results and len(results) > 0:
            danger_level, danger_text = calculate_danger(results[0])
        else:
            danger_level, danger_text = 0, "No detections"

        # Update UI
        image_placeholder.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))
        danger_placeholder.progress(int(danger_level))
        danger_placeholder.text(danger_text)

        time.sleep(0.02)

    # Cleanup
    if cap is not None:
        cap.release()
    if tfile is not None:
        try:
            os.unlink(tfile.name)
        except Exception:
            pass

# 8) If user clicked Start, run the stream loop
if st.session_state.get("run_stream", False):
    process_stream()
