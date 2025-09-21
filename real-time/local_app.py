import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
import time

# Page configuration
st.set_page_config(page_title="Real-Time Spitting Detection", layout="wide")
st.title("🎥 Real-Time Spitting Detection with YOLO")
st.markdown("Live feed from your webcam with detection results side by side.")

# Load the YOLO model
@st.cache_resource
def load_model():
    model = YOLO("spit-detection.pt")
    return model

model = load_model()

# Start detection button
start_detection = st.button("🔴 Start Detection")

# Create layout with two columns
col1, col2 = st.columns(2)

# Placeholder for frames
original_frame_placeholder = col1.empty()
detected_frame_placeholder = col2.empty()

if start_detection:
    cap = cv2.VideoCapture(0)
    st.info("Click 'Stop' to end detection.")
    stop = st.button("⏹ Stop")

    while cap.isOpened() and not stop:
        ret, frame = cap.read()
        if not ret:
            st.error("❌ Failed to read from webcam.")
            break

        # Resize frame for consistent output
        frame_resized = cv2.resize(frame, (640, 480))

        # Run YOLO detection
        results = model(frame_resized)

        # Get result image with boxes
        detected_img = results[0].plot()

        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        detected_rgb = cv2.cvtColor(detected_img, cv2.COLOR_BGR2RGB)

        # Update Streamlit display
        original_frame_placeholder.image(frame_rgb, channels="RGB")
        detected_frame_placeholder.image(detected_rgb, channels="RGB")

        # Optional: reduce CPU usage
        time.sleep(0.05)

    cap.release()
    st.success("Detection stopped.")
