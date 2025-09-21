import streamlit as st
import cv2
import tempfile
import numpy as np
from ultralytics import YOLO

# Load the YOLO model
model = YOLO("best.pt")

st.title("Spitting Detection System")
st.markdown("Upload an image, video, or use webcam to see spitting detection in action!")

# Sidebar for Video/Image
option = st.sidebar.radio("Choose input type", ["Image", "Video", "Webcam"])


# -------------------- IMAGE UPLOAD --------------------
if option == "Image":
    image_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if image_file:
        file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        st.image(img, caption="Uploaded Image", channels="BGR")

        st.markdown("### Detection Result")
        results = model.predict(img)
        annotated_img = results[0].plot()
        st.image(annotated_img, channels="BGR")


# -------------------- VIDEO UPLOAD --------------------
elif option == "Video":
    video_file = st.file_uploader("Upload a video", type=["mp4", "mov", "avi", "mkv"])

    if video_file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video_file.read())
        
        cap = cv2.VideoCapture(tfile.name)
        stframe = st.empty()
        st.markdown("### Detection Preview")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model.predict(frame)
            annotated_frame = results[0].plot()
            stframe.image(annotated_frame, channels="BGR", use_container_width=True)

        cap.release()


# -------------------- WEBCAM CAPTURE --------------------
elif option == "Webcam":
    st.markdown("### Live Webcam Capture")
    st.info("Click the button below to capture an image from your webcam.")

    # Webcam capture
    img_file_buffer = st.camera_input("Capture Image")

    if img_file_buffer is not None:
        # Convert captured image to OpenCV format
        bytes_data = img_file_buffer.getvalue()
        np_arr = np.frombuffer(bytes_data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        st.image(frame, caption="Captured Image", channels="BGR")

        st.markdown("### Detection Result")
        results = model.predict(frame)
        annotated_frame = results[0].plot()
        st.image(annotated_frame, channels="BGR")
