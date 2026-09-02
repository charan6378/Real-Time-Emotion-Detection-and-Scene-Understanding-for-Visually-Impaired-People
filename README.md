# 👁️ Real-Time Emotion Detection and Scene Understanding for Visually Impaired People

An AI-powered computer vision system designed to assist visually impaired people by providing **real-time object detection, emotion recognition, spatial awareness, scene understanding, and audio feedback**.

The system uses **YOLOv8-based computer vision models** to analyze images and live camera frames, identify objects and human emotions, determine the approximate position of detected objects, and convert the resulting scene information into speech.

---

## 🌟 Project Overview

Visually impaired people may face difficulties understanding their surrounding environment, identifying nearby objects, and interpreting the emotions of people around them.

This project addresses these challenges through a software-based **AI vision assistant** that analyzes visual information and communicates the detected information through audio.

The system provides:

* 🔍 Real-time object detection
* 😊 Human emotion detection
* 📍 Approximate object position awareness
* 🧠 Stable scene understanding across multiple frames
* 🔊 Automatic audio descriptions
* 📷 Live camera analysis
* 🖼️ Image upload and analysis
* 📊 Model evaluation using ROC curves and mAP
* 💻 Command-line image inference

---

## ✨ Key Features

### 🔍 Object Detection

The system uses **YOLOv8s** to detect objects in an image or live camera feed.

Detected objects are displayed with bounding boxes and labels.

### 😊 Emotion Detection

A custom-trained **YOLOv8 emotion detection model** is used to identify emotions from people in the scene.

The detected emotion is incorporated into the generated scene description.

### 📍 Position Awareness

For every detected object, the system estimates its position within the image.

The image is divided into approximate regions:

* Top Left
* Top Center
* Top Right
* Middle Left
* Center
* Middle Right
* Bottom Left
* Bottom Center
* Bottom Right

This allows the system to generate descriptions such as:

> "A person at the center"

### 🧠 Stable Scene Understanding

Instead of immediately speaking every detection from every camera frame, the application maintains a short buffer of recent frames.

Objects and emotions that appear consistently across multiple frames are considered stable and included in the scene description.

This reduces unnecessary or repetitive audio feedback.

### 🔊 Audio Feedback

The Streamlit application uses **Google Text-to-Speech (gTTS)** to convert scene descriptions into audio.

During live detection, audio feedback is controlled using a time interval and comparison with the previous spoken description to reduce repeated announcements.

### 📷 Live Camera Detection

The application can access the computer's camera and continuously analyze incoming frames.

The live camera interface provides:

* Real-time annotated video
* Stable scene description
* Automatic audio feedback
* Start/Stop controls

The camera is configured at **640 × 480 resolution** with a reduced buffer size to help minimize frame lag.

### 🖼️ Image Upload

Users can upload:

* JPG
* JPEG
* PNG
* WEBP

The uploaded image is processed by both detection models and the annotated results are displayed along with the generated audio description.

### 💻 Command-Line Detection

The project also provides `cli_detect.py` for running detection on an individual image without using the Streamlit interface.

The CLI provides:

* Detected objects
* Confidence scores
* Object positions
* Approximate distance estimation
* Detected emotions
* Audio output using `pyttsx3`

---

## 🏗️ System Architecture

```text
                  Input
                    │
          ┌─────────┴─────────┐
          │                   │
     Image Upload        Live Camera
          │                   │
          └─────────┬─────────┘
                    │
                    ▼
             Image Processing
                    │
          ┌─────────┴─────────┐
          │                   │
          ▼                   ▼
     YOLOv8s Model      Custom YOLOv8
     Object Detection   Emotion Detection
          │                   │
          ▼                   ▼
     Object Labels       Emotions
     Confidence          Confidence
     Position
     Distance
          │                   │
          └─────────┬─────────┘
                    ▼
           Stable Scene Builder
                    │
                    ▼
          Natural Language Scene
              Description
                    │
                    ▼
             Text-to-Speech
                    │
                    ▼
              🔊 Audio Feedback
```

---

## 🧠 Models Used

### Object Detection Model

**YOLOv8s**

The pretrained YOLOv8 small model is used for general-purpose object detection.

Model file:

```text
yolov8s.pt
```

### Emotion Detection Model

A custom-trained YOLOv8 model is used for emotion detection.

Model file:

```text
emotion_detection_model.pt
```

The model was trained using a YOLO-format emotion dataset.

> The trained `.pt` model files are intentionally excluded from this repository through `.gitignore` because of their size.

---

## 🧪 Emotion Model Training

The emotion detection model was developed using the YOLOv8 framework.

The training workflow includes:

1. Dataset acquisition
2. Dataset inspection
3. Exploratory data analysis
4. Class distribution analysis
5. YOLO-format dataset preparation
6. YOLOv8 model initialization
7. Model training
8. Validation
9. Test evaluation
10. Single-image inference
11. mAP analysis
12. ROC curve analysis
13. Model interpretability experiments using EigenCAM

The training process is documented in:

```text
emotion-detection-using-yolo.ipynb
```

---

## 📊 Model Evaluation

The project includes evaluation and visualization scripts for analyzing model performance.

### Mean Average Precision

The emotion detection model is evaluated using:

* mAP@0.5
* mAP@0.5:0.95

### ROC Curves

The repository contains generated ROC curve visualizations:

```text
roc_curve.png
detection_roc_curve.png
detection_roc_curve_fixed.png
coco_detection_roc_curve.png
```

The project also includes scripts for generating ROC curves:

```text
plot_roc_curve.py
plot_detection_roc.py
```

The evaluation scripts use confidence scores and, for object detection evaluation, bounding-box IoU matching.

---

## 🔬 Model Interpretability

The training notebook also contains an **EigenCAM-based visualization experiment**.

This helps visualize regions of an input image that contribute to the model's predictions and provides additional insight into the model's behavior.

---

## 🖥️ Application Interface

The Streamlit application provides three main pages:

### 🏠 Home

Provides an overview of the system and its capabilities.

### 📤 Upload Image

Users can upload an image and receive:

* Original image
* Detection results
* Object information
* Emotion information
* Scene description
* Audio feedback

### 📹 Live Camera

Users can start the computer camera and receive continuous:

* Object detection
* Emotion detection
* Position information
* Stable scene descriptions
* Audio feedback

---

## 📁 Project Structure

```text
Real-Time-Emotion-Detection-and-Scene-Understanding-for-Visually-Impaired-People/
│
├── app2.py
│   └── Main Streamlit application
│
├── cli_detect.py
│   └── Command-line image detection and audio output
│
├── emotion-detection-using-yolo.ipynb
│   └── Emotion model training, evaluation and analysis
│
├── plot_roc_curve.py
│   └── Emotion detection ROC curve generation
│
├── plot_detection_roc.py
│   └── Object detection ROC curve generation
│
├── roc_curve.png
├── detection_roc_curve.png
├── detection_roc_curve_fixed.png
├── coco_detection_roc_curve.png
│   └── Evaluation visualizations
│
├── usecases/
│   └── Example images and application screenshots
│
└── .gitignore
```

---

## ⚙️ Technologies Used

| Technology   | Purpose                                 |
| ------------ | --------------------------------------- |
| Python       | Core programming language               |
| YOLOv8       | Object and emotion detection            |
| Ultralytics  | YOLO model framework                    |
| PyTorch      | Deep learning framework                 |
| OpenCV       | Image and camera processing             |
| Streamlit    | Web application interface               |
| Pillow       | Image processing                        |
| gTTS         | Text-to-speech in Streamlit application |
| pyttsx3      | Offline/local speech output in CLI      |
| scikit-learn | ROC curve and AUC calculation           |
| NumPy        | Numerical processing                    |
| Matplotlib   | Visualization                           |
| Pandas       | Data analysis                           |
| Seaborn      | Exploratory data analysis               |
| Roboflow     | Emotion dataset acquisition             |
| EigenCAM     | Model interpretability                  |

---

## 🚀 Installation

### 1. Clone the repository

```bash
git clone https://github.com/charan6378/Real-Time-Emotion-Detection-and-Scene-Understanding-for-Visually-Impaired-People.git
```

```bash
cd Real-Time-Emotion-Detection-and-Scene-Understanding-for-Visually-Impaired-People
```

### 2. Create a virtual environment

```bash
python -m venv sightassist_env
```

Activate it on Windows:

```bash
sightassist_env\Scripts\activate
```

### 3. Install dependencies

Install the required packages:

```bash
pip install streamlit ultralytics torch torchvision opencv-python pillow gTTS pyttsx3 numpy pandas matplotlib seaborn scikit-learn pyyaml
```

For the training/interpretability notebook, additional packages such as Roboflow and Grad-CAM may be required.

```bash
pip install roboflow grad-cam
```

---

## 📦 Model Files

The application expects the following model files in the project directory:

```text
yolov8s.pt
emotion_detection_model.pt
```

These files are **not included in the GitHub repository** because they are large binary model files.

Place the model files in the project root:

```text
project/
│
├── app2.py
├── cli_detect.py
├── yolov8s.pt
└── emotion_detection_model.pt
```

---

## ▶️ Running the Streamlit Application

After installing the dependencies and placing the required model files in the project directory:

```bash
streamlit run app2.py
```

The Streamlit application will open in your browser.

From the interface you can select:

```text
Home
Upload Image
Live Camera
```

---

## 💻 Running Command-Line Detection

To analyze an individual image:

```bash
python cli_detect.py path/to/image.jpg
```

Example:

```bash
python cli_detect.py usecases/image_2.jpg
```

The program displays detected objects and emotions in the terminal and generates an audio description using the local text-to-speech engine.

---

## 🔊 Example Scene Description

The system combines visual detections into an audio-friendly description.

A generated description can contain information such as:

```text
Scene description: person at middle center, chair at bottom right.
Emotions detected: happy.
```

The exact output depends on the input image and model predictions.

---

## 🖼️ Application Screenshots

Example application screenshots are available in:

```text
usecases/
```

These include examples of the application's home page, detection page, detected objects, and audio feedback.

---

## ⚠️ Important Notes

### Distance Estimation

The CLI implementation calculates an **approximate distance value from the detected bounding-box area**.

This is not a physical depth sensor measurement and should therefore be treated as an approximate visual estimate rather than a precise real-world distance.

### Internet Connection

The Streamlit application uses **gTTS**, which requires network access to generate speech.

### Camera

Live detection requires a working camera accessible by the computer.

### Model Files

The `.pt` model files are excluded from GitHub through `.gitignore` and must be provided separately before running the complete application.

---

## 🔮 Future Enhancements

Potential improvements include:

* More accurate monocular depth estimation
* Better distance estimation
* Additional emotion classes
* Improved emotion recognition in challenging lighting conditions
* Object tracking across frames
* Voice commands
* Offline text-to-speech integration
* Mobile deployment
* Edge-device optimization
* GPS/location-aware assistance
* Obstacle prioritization
* Multi-language audio feedback
* Improved low-light performance

---

## 🎯 Applications

This system can potentially support:

* Assistive vision applications
* Indoor navigation assistance
* Scene awareness
* Object identification
* Social/emotional awareness
* Accessibility-focused computer vision
* AI-based visual assistance

---

## 👥 Project

**Project Title:**
Real-Time Emotion Detection and Scene Understanding for Visually Impaired People

**Project Type:**
Artificial Intelligence / Deep Learning / Computer Vision

**Primary Technologies:**
Python, YOLOv8, PyTorch, OpenCV, Streamlit

---

## 📄 License

This project is intended for educational and research purposes.

---

## ⭐ Acknowledgements

This project makes use of open-source technologies and frameworks including:

* Ultralytics YOLO
* PyTorch
* OpenCV
* Streamlit
* scikit-learn
* Roboflow
* Google Text-to-Speech

If you find this project useful, consider giving the repository a ⭐.
