import torch
import torch.nn as nn
from torchvision import models, transforms
import cv2
import mediapipe as mp
from PIL import Image
import numpy as np
import time

# -------------------- Device Setup --------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -------------------- Load Models --------------------
# Eyes
model_eyes_loaded = models.convnext_tiny(weights=None)
model_eyes_loaded.classifier[2] = nn.Linear(model_eyes_loaded.classifier[2].in_features, 2)
model_eyes_loaded.load_state_dict(torch.load("model_eyes_tiny_weights(1).pth", map_location=device))
model_eyes_loaded = model_eyes_loaded.to(device).eval()

# Mouth
model_mouth_loaded = models.convnext_tiny(weights=None)
model_mouth_loaded.classifier[2] = nn.Linear(model_mouth_loaded.classifier[2].in_features, 2)
model_mouth_loaded.load_state_dict(torch.load("model_mouth_tiny_weights(3).pth", map_location=device))
model_mouth_loaded = model_mouth_loaded.to(device).eval()

# -------------------- Mediapipe Setup --------------------
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(max_num_faces=1, refine_landmarks=True)

# -------------------- Landmark Indices --------------------
LEFT_EYE = [33, 133, 160, 159, 158, 153, 144, 145, 153]
RIGHT_EYE = [362, 263, 387, 386, 385, 380, 373, 374, 380]
MOUTH_LANDMARKS = list(range(78, 88)) + list(range(308, 318))

# -------------------- Transform --------------------
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# -------------------- ROI Extraction --------------------
def extract_roi(image, landmarks, indices, expand_x=5, expand_y=5):
    h, w, _ = image.shape
    x_coords = [int(landmarks[i].x * w) for i in indices]
    y_coords = [int(landmarks[i].y * h) for i in indices]

    x_min = max(min(x_coords) - expand_x, 0)
    x_max = min(max(x_coords) + expand_x, w)
    y_min = max(min(y_coords) - expand_y, 0)
    y_max = min(max(y_coords) + expand_y, h)

    if x_max <= x_min or y_max <= y_min:
        return None, None  # 🔥 prevents crashes

    roi = image[y_min:y_max, x_min:x_max]
    if roi.size == 0:
        return None, None

    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    return Image.fromarray(roi), (x_min, y_min, x_max, y_max)


# -------------------- Prediction --------------------
def predict(model, roi):
    roi = transform(roi).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(roi)
        _, pred = torch.max(output, 1)
    return int(pred.item())

# -------------------- Video Capture --------------------
cap = cv2.VideoCapture(0)
prev_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    left_eye_img, right_eye_img, mouth_img = None, None, None
    left_eye_state = right_eye_state = mouth_state = None

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark

        # Eyes
        left_eye_img, left_eye_box = extract_roi(frame, landmarks, LEFT_EYE, expand_x=5, expand_y=20)
        right_eye_img, right_eye_box = extract_roi(frame, landmarks, RIGHT_EYE, expand_x=5, expand_y=20)
        left_eye_state = predict(model_eyes_loaded, left_eye_img)
        right_eye_state = predict(model_eyes_loaded, right_eye_img)

        # Mouth
        mouth_img, mouth_box = extract_roi(frame, landmarks, MOUTH_LANDMARKS, expand_x=5, expand_y=5)
        mouth_state = predict(model_mouth_loaded, mouth_img)

        # Draw bounding boxes
        for (x_min, y_min, x_max, y_max) in [left_eye_box, right_eye_box, mouth_box]:
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)

        # Display text
        cv2.putText(frame, f"Left Eye: {'Closed' if left_eye_state == 1 else 'Open'}", (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Right Eye: {'Closed' if right_eye_state == 1 else 'Open'}", (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Mouth: {'Yawn' if mouth_state == 1 else 'No Yawn'}", (30, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # -------------------- FPS --------------------
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
    prev_time = curr_time
    cv2.putText(frame, f"FPS: {fps:.2f}", (30, 130),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    # -------------------- ROI Strip (side display) --------------------
    roi_previews = []
    for roi_img, label in [(left_eye_img, "Left Eye"), (right_eye_img, "Right Eye"), (mouth_img, "Mouth")]:
        if roi_img is not None:
            roi_bgr = cv2.cvtColor(np.array(roi_img), cv2.COLOR_RGB2BGR)
            roi_bgr = cv2.resize(roi_bgr, (100, 100))
            cv2.putText(roi_bgr, label, (5, 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            roi_previews.append(roi_bgr)

    if roi_previews:
        roi_strip = np.vstack(roi_previews)
        # Match height to frame
        roi_strip = cv2.resize(roi_strip, (roi_strip.shape[1], frame.shape[0]))
        combined_frame = np.hstack((frame, roi_strip))
    else:
        combined_frame = frame

    cv2.imshow("Drowsiness Detection", combined_frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
