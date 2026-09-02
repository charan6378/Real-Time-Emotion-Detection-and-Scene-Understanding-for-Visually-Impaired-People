import sys
from ultralytics import YOLO
from PIL import Image
import pyttsx3
import os

def speak_text(text):
    try:
        tts_engine = pyttsx3.init()
        tts_engine.setProperty('rate', 150)
        tts_engine.say(text)
        tts_engine.runAndWait()
    except Exception as e:
        print("[TTS ERROR]", e)

def get_position(bbox, img_width, img_height):
    x_min, y_min, x_max, y_max = bbox
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    if x_center < img_width / 3:
        horizontal = "left"
    elif x_center > 2 * img_width / 3:
        horizontal = "right"
    else:
        horizontal = "center"
    if y_center < img_height / 3:
        vertical = "top"
    elif y_center > 2 * img_height / 3:
        vertical = "bottom"
    else:
        vertical = "middle"
    return f"{vertical} {horizontal}"

def detect_objects(image_pil, model):
    results = model(image_pil, conf=0.25)
    detections = []
    for det in results[0].boxes:
        x_min, y_min, x_max, y_max = map(float, det.xyxy[0])
        conf = float(det.conf[0])
        cls_id = int(det.cls[0])
        label = model.names[cls_id]
        position = get_position([x_min, y_min, x_max, y_max], image_pil.width, image_pil.height)
        area = (x_max - x_min) * (y_max - y_min)
        distance = max(20, 500 / (area ** 0.5)) if area > 0 else 100
        detections.append({
            'label': label,
            'confidence': conf,
            'position': position,
            'distance': int(distance),
            'bbox': [x_min, y_min, x_max, y_max]
        })
    return detections

def detect_emotions(image_pil, model):
    try:
        results = model(image_pil, conf=0.3)
        emotions = []
        for det in results[0].boxes:
            x_min, y_min, x_max, y_max = map(float, det.xyxy[0])
            conf = float(det.conf[0])
            cls_id = int(det.cls[0])
            label = model.names[cls_id]
            emotions.append({
                'label': label,
                'confidence': conf,
                'bbox': [x_min, y_min, x_max, y_max]
            })
        return emotions
    except Exception as e:
        print(f"Emotion detection error: {e}")
        return []

def main(img_path):
    if not os.path.isfile("yolov8s.pt") or not os.path.isfile("emotion_detection_model.pt"):
        print("Missing model file(s)!")
        return
    try:
        # Patch torch.load to always set weights_only=False
        import torch
        import builtins
        orig_torch_load = torch.load
        def patched_load(*args, **kwargs):
            kwargs["weights_only"] = False
            return orig_torch_load(*args, **kwargs)
        torch.load = patched_load
        object_model = YOLO('yolov8s.pt')
        emotion_model = YOLO('emotion_detection_model.pt')
        torch.load = orig_torch_load
    except Exception as e:
        print("Model loading failed:", e)
        return
    try:
        image = Image.open(img_path).convert("RGB")
    except Exception as e:
        print("Failed to load image:", e)
        return
    print("[INFO] Detecting objects...")
    objects = detect_objects(image, object_model)
    print("[INFO] Detecting emotions...")
    emotions = detect_emotions(image, emotion_model)
    # Print results
    print("\nDetected objects:")
    if objects:
        for obj in objects:
            print(f"- {obj['label']} | Position: {obj['position']} | Distance: {obj['distance']}cm | Confidence: {obj['confidence']:.0%}")
    else:
        print("None")
    print("\nDetected emotions:")
    if emotions:
        for em in emotions:
            print(f"- {em['label']} | Confidence: {em['confidence']:.0%}")
    else:
        print("None")
    # Audio summary
    speech_parts = []
    if objects:
        for obj in objects[:3]:
            speech_parts.append(f"{obj['label']} at {obj['position']}, about {obj['distance']} centimeters away")
    if emotions:
        emotion_str = ", ".join([f"{e['label']}" for e in emotions[:2]])
        speech_parts.append(f"Emotions detected: {emotion_str}")
    if speech_parts:
        speech_text = "Scene description: " + ". ".join(speech_parts) + "."
    else:
        speech_text = "No significant objects or emotions detected."
    print("\nDescription:\n", speech_text)
    print("Playing audio...")
    speak_text(speech_text)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python cli_detect.py <image_path>")
    else:
        main(sys.argv[1])

