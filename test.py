import cv2
import mediapipe as mp
import torch
import torch.nn.functional as F
from torchvision import transforms, models
import time
import numpy as np
import torch.nn as nn
from PIL import Image

# -----------------------------
# Load Models
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# EYE MODEL (ResNet50)
eye_model = models.resnet50(weights=None)
eye_model.fc = nn.Linear(eye_model.fc.in_features, 2)
eye_model.load_state_dict(torch.load("model_eyes_weights.pth", map_location=device))
eye_model = eye_model.to(device)
eye_model.eval()

# MOUTH MODEL (ResNet18)
mouth_model = models.resnet18(weights=None)
mouth_model.fc = nn.Linear(mouth_model.fc.in_features, 2)
mouth_model.load_state_dict(torch.load("model_mouth_weights.pth", map_location=device))
mouth_model = mouth_model.to(device)
mouth_model.eval()

# -----------------------------
# Landmark Constants
# -----------------------------
LEFT_EYE = [33, 133, 160, 159, 158, 153, 144, 145, 153]
RIGHT_EYE = [362, 263, 387, 386, 385, 380, 373, 374, 380]
MOUTH_LANDMARKS = list(range(78, 88)) + list(range(308, 318))

# -----------------------------
# Transform for Model Input
# -----------------------------
val_test_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# -----------------------------
# Initialize MediaPipe
# -----------------------------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False,
                                  max_num_faces=1,
                                  min_detection_confidence=0.5,
                                  min_tracking_confidence=0.5)

# -----------------------------
# Utility Functions
# -----------------------------
def extract_region(image, landmarks, points, scale=1.2):
    """Extracts a rectangular ROI around a set of landmark points."""
    h, w, _ = image.shape
    pts = np.array([(int(landmarks[p].x * w), int(landmarks[p].y * h)) for p in points])
    x, y, w_, h_ = cv2.boundingRect(pts)
    cx, cy = x + w_ // 2, y + h_ // 2
    size = int(max(w_, h_) * scale)
    x1, y1 = max(cx - size // 2, 0), max(cy - size // 2, 0)
    x2, y2 = min(cx + size // 2, w), min(cy + size // 2, h)
    return image[y1:y2, x1:x2], (x1, y1, x2, y2)

# -----------------------------
# Start Webcam Feed
# -----------------------------
cap = cv2.VideoCapture(0)
prev_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(rgb)

    eye_state, mouth_state = "N/A", "N/A"

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark

        # Extract both eyes + bounding boxes
        left_eye, left_box = extract_region(frame, landmarks, LEFT_EYE)
        right_eye, right_box = extract_region(frame, landmarks, RIGHT_EYE)

        # Draw boxes if found
        if left_box:
            cv2.rectangle(frame, (left_box[0], left_box[1]),
                          (left_box[2], left_box[3]), (255, 0, 0), 1)
        if right_box:
            cv2.rectangle(frame, (right_box[0], right_box[1]),
                          (right_box[2], right_box[3]), (255, 0, 0), 1)

        # Eye prediction
        if left_eye is not None and right_eye is not None:
            # Convert both to RGB if needed
            left_eye = cv2.cvtColor(left_eye, cv2.COLOR_BGR2RGB)
            right_eye = cv2.cvtColor(right_eye, cv2.COLOR_BGR2RGB)

            # Resize both to same dimensions
            h, w = min(left_eye.shape[0], right_eye.shape[0]), min(left_eye.shape[1], right_eye.shape[1])
            left_eye_resized = cv2.resize(left_eye, (w, h))
            right_eye_resized = cv2.resize(right_eye, (w, h))

            # Combine both eyes
            eye_combined = cv2.addWeighted(left_eye_resized, 0.5, right_eye_resized, 0.5, 0)

            # Transform + inference
            eye_input = val_test_transform(Image.fromarray(eye_combined)).unsqueeze(0).to(device)
            with torch.no_grad():
                eye_out = eye_model(eye_input)
                eye_prob = torch.softmax(eye_out, dim=1)[0, 1].item()  # class 1 = closed
                eye_state = "Closed" if eye_prob > 0.5 else "Open"

        # Mouth region
        mouth, mouth_box = extract_region(frame, landmarks, MOUTH_LANDMARKS)
        if mouth is not None:
            cv2.rectangle(frame, (mouth_box[0], mouth_box[1]),
                          (mouth_box[2], mouth_box[3]), (0, 0, 255), 1)

            mouth_rgb = cv2.cvtColor(mouth, cv2.COLOR_BGR2RGB)
            mouth_pil = Image.fromarray(mouth_rgb)
            mouth_input = val_test_transform(mouth_pil).unsqueeze(0).to(device)

            with torch.no_grad():
                mouth_out = mouth_model(mouth_input)
                mouth_prob = F.softmax(mouth_out, dim=1)[0, 1].item()  # class 1 = yawn
                mouth_state = "Yawning" if mouth_prob > 0.5 else "Normal"

    # -----------------------------
    # FPS Calculation
    # -----------------------------
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time + 1e-6)
    prev_time = curr_time

    # -----------------------------
    # Display
    # -----------------------------
    cv2.putText(frame, f"Eye: {eye_state}", (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1)
    cv2.putText(frame, f"Mouth: {mouth_state}", (30, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1)
    cv2.putText(frame, f"FPS: {fps:.1f}", (30, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 1)

    cv2.imshow("Drowsiness Detection (with Boxes)", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
