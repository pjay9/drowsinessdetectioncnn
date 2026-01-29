
import torch
import torch.nn as nn
from torchvision import models, transforms
import cv2
import mediapipe as mp
from PIL import Image
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Recreate the model architecture
model_eyes = models.mobilenet_v2(weights=None)
model_eyes.classifier[1] = nn.Linear(model_eyes.last_channel, 2)  # 2 classes: open/closed
model_eyes.load_state_dict(torch.load("model_eyes_tiny_weights(1).pth", map_location=device))
model_eyes = model_eyes.to(device)
model_eyes.eval()

# Mouth model
model_mouth = models.mobilenet_v2(weights=None)
model_mouth.classifier[1] = nn.Linear(model_mouth.last_channel, 2)  # 2 classes: yawn/no yawn
model_mouth.load_state_dict(torch.load("model_mouth_tiny_weights(3).pth", map_location=device))
model_mouth = model_mouth.to(device)
model_mouth.eval()

mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(max_num_faces=1, refine_landmarks=True)

# Landmark indices
LEFT_EYE = [33, 133, 160, 159, 158, 153, 144, 145, 153]
RIGHT_EYE = [362, 263, 387, 386, 385, 380, 373, 374, 380]
MOUTH_LANDMARKS = list(range(78, 88)) + list(range(308, 318))  # exactly like training

transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

def extract_roi(image, landmarks, indices, expand_x=5, expand_y=5):
    """
    Extract ROI around landmarks, with adjustable horizontal (expand_x)
    and vertical (expand_y) padding.
    """
    h, w, _ = image.shape
    x_coords = [int(landmarks[i].x * w) for i in indices]
    y_coords = [int(landmarks[i].y * h) for i in indices]
    x_min, x_max = max(min(x_coords)-expand_x,0), min(max(x_coords)+expand_x,w)
    y_min, y_max = max(min(y_coords)-expand_y,0), min(max(y_coords)+expand_y,h)
    roi = image[y_min:y_max, x_min:x_max]
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    return Image.fromarray(roi), (x_min, y_min, x_max, y_max)

def predict(model, roi):
    roi = transform(roi).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(roi)
        _, pred = torch.max(output, 1)
    return int(pred.item())

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark

        # Eyes (taller ROI to include eyebrows)
        left_eye_img, left_eye_box = extract_roi(frame, landmarks, LEFT_EYE, expand_x=5, expand_y=20)
        right_eye_img, right_eye_box = extract_roi(frame, landmarks, RIGHT_EYE, expand_x=5, expand_y=20)
        left_eye_state = predict(model_eyes, left_eye_img)
        right_eye_state = predict(model_eyes, right_eye_img)

        # Mouth (use training landmarks, default padding)
        mouth_img, mouth_box = extract_roi(frame, landmarks, MOUTH_LANDMARKS, expand_x=5, expand_y=5)
        mouth_state = predict(model_mouth, mouth_img)

        # Draw bounding boxes
        for (x_min, y_min, x_max, y_max) in [left_eye_box, right_eye_box, mouth_box]:
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)

        # Display states
        cv2.putText(frame, f"Left Eye: {'Open' if left_eye_state else 'Closed'}", (30,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0),2)
        cv2.putText(frame, f"Right Eye: {'Open' if right_eye_state else 'Closed'}", (30,60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0),2)
        cv2.putText(frame, f"Mouth: {'Yawn' if mouth_state else 'No Yawn'}", (30,90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255),2)

        # Debug ROI previews
        cv2.imshow("Left Eye ROI", cv2.cvtColor(np.array(left_eye_img), cv2.COLOR_RGB2BGR))
        cv2.imshow("Right Eye ROI", cv2.cvtColor(np.array(right_eye_img), cv2.COLOR_RGB2BGR))
        cv2.imshow("Mouth ROI", cv2.cvtColor(np.array(mouth_img), cv2.COLOR_RGB2BGR))

    cv2.imshow("Drowsiness Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
