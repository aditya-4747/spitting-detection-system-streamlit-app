import streamlit as st
import cv2
import tempfile
import numpy as np
from ultralytics import YOLO

model = YOLO("best.pt")

st.title("Spitting Detection System")
st.markdown("Upload an image or video to see spitting detection in action!")

# Sidebar for selection
option = st.sidebar.radio("Choose input type", ["Image", "Video"])

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
            stframe.image(annotated_frame, channels="BGR", use_column_width=True)

        cap.release()
