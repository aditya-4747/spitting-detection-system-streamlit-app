# Spitting Detection System

An AI-powered computer vision system that detects spitting behavior in images and videos using YOLOv8. The project demonstrates the end-to-end machine learning pipeline — from dataset curation and manual annotation to model training, evaluation, and deployment via a Streamlit app. Designed as a proof of concept for enhancing public hygiene monitoring, it explores how object detection can be applied to real-world societal challenges.

---

## ✨ Key highlights
- **Dataset:** 755 images, **983 annotated instances** (492 spitting, 491 non-spitting) after augmentation.
- **Model:** Ultralytics **YOLOv8-L** (final), trained for **50 epochs** on Google Colab (T4 GPU).
- **Metrics:** **Precision 99.20%**, **Recall 99.90%**, **F1 99.54%**, **mAP@50 99.50%**, **Accuracy (instance-level) 98.19%**.
  - *Note:* mAP@50 / precision / recall are primary object-detection metrics. The reported accuracy is an additional instance-level binary metric derived from the YOLO confusion matrix for interpretability.
- **Deployment:** Streamlit Cloud (image/video upload or webcam capture interface).

---

## 📂 Repository structure
```
.
├── real-time
  ├── local_app.py         # File to run local inference (use this instead of app.py for real-time inference)
├── README.md              # Project documentation
├── app.py                 # Streamlit app (UI + inference) (only for deployment)
├── model.pt               # Trained YOLOv8 model weights
├── packages.txt           # system deps for Streamlit Cloud
├── requirements.txt       # python libs
```

---

## 🧰 Tech stack
- **Model & Training:** Ultralytics YOLOv8 (PyTorch)
- **Inference / CV:** OpenCV, NumPy
- **Frontend / Demo:** Streamlit
- **Annotation:** LabelImg (manual annotations)
- **Platform:** Google Colab (T4 GPU)
- **Other:** Python, pip

---

## 📥 Dataset & preprocessing (what I did)
- Collected and curated **755 images** with **983 instances** (spitting vs non-spitting).
- Manual annotation for every instance using **LabelImg** (including augmented images).
- Image normalization to **640 × 640 px** for training consistency.
- Augmentation strategy:
  - Horizontal flipping for most images.
  - Rotation ±15° for specifically selected images (used when the subject appeared infrequently).
- Data split: custom Python script to rename images with convention (`spitting-[n]`, `non-spitting-[n]`) and randomly allocate **10% test / 10% validation**.

---

## 🧪 Model training (summary)
- Experiments conducted across YOLO variants (v5, v8, v9, v11), multiple annotation strategies and dataset variants.
- Final model: **YOLOv8-L** selected based on validation performance.
- Training setup (final run):
  - Epochs: **50**
  - Optimizer / params: default Ultralytics settings (AdamW, lr=0.001, batch=16)
  - Environment: Google Colab (T4 GPU)

---

## 📈 Evaluation
- Primary detection metrics used: **Precision, Recall, F1, mAP@50**. These are directly reported from the YOLO evaluation pipeline.
- For interpretability, an additional **binary classification accuracy (98.19%)** was computed from the YOLO confusion matrix.

---

## ⚙️ Run locally
1. Clone the repo
```bash
git clone https://github.com/aditya-4747/spitting-detection-system.git
cd spitting-detection-system
```
2. (Optional) create and activate a virtual environment
```bash
python -m venv venv
# mac/linux
source venv/bin/activate
# windows
venv\Scripts\activate
```
3. Install dependencies
```bash
pip install -r requirements.txt
```
4. Run the Streamlit app
```bash
streamlit run ./real-time/local_app.py
```

---

## 🌐 Streamlit Cloud | [**Live demo**](https://spitting-detection-system.streamlit.app/)
- The app is deployed on **Streamlit Cloud** (image/video upload or Webcam capture).

(This live demo does not features real-time detection. Run locally with suitable GPU to unleash real-time capabilities.)

---

## 🧭 Team & role
- **Team size:** 4
- **Role:** Team lead — responsible for annotation strategy experimentation, model selection and training, inference pipeline, and Streamlit deployment. Team members supported documentation, dataset collection, annotations & training runs.

---

## 🔮 Future Direction
- Expand dataset diversity (more subjects, varied backgrounds, and lighting).
- Optimize model for **edge deployment** (model quantization / pruning) for integration with CCTV or embedded devices.
- Improve inference throughput for real-time deployment (model optimization, batching, or edge acceleration).

---

## ✉️ Contact
[**Aditya Maddeshiya**](https://github.com/aditya-4747)

---




