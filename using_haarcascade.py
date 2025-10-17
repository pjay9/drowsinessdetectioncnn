import cv2
import mediapipe as mp
import numpy as np
import time

mp_face_mesh = mp.solutions.face_mesh

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH_VERT = [13, 14]      
MOUTH_HORZ = [78, 308]    

def calculate_eye_aspect_ratio(points, landmarks):
    if len(points) < 6:
        return 0
    vertical_1 = np.linalg.norm(np.array([landmarks[points[1]].x, landmarks[points[1]].y]) -
                                np.array([landmarks[points[5]].x, landmarks[points[5]].y]))
    vertical_2 = np.linalg.norm(np.array([landmarks[points[2]].x, landmarks[points[2]].y]) -
                                np.array([landmarks[points[4]].x, landmarks[points[4]].y]))
    horizontal = np.linalg.norm(np.array([landmarks[points[0]].x, landmarks[points[0]].y]) -
                                np.array([landmarks[points[3]].x, landmarks[points[3]].y]))
    return (vertical_1 + vertical_2) / (2.0 * horizontal)

def calculate_mouth_aspect_ratio(landmarks):
    vertical = np.linalg.norm(np.array([landmarks[MOUTH_VERT[0]].x, landmarks[MOUTH_VERT[0]].y]) -
                              np.array([landmarks[MOUTH_VERT[1]].x, landmarks[MOUTH_VERT[1]].y]))
    horizontal = np.linalg.norm(np.array([landmarks[MOUTH_HORZ[0]].x, landmarks[MOUTH_HORZ[0]].y]) -
                                np.array([landmarks[MOUTH_HORZ[1]].x, landmarks[MOUTH_HORZ[1]].y]))
    return vertical / horizontal

def get_label(aspect_ratio, threshold, open_label="Open", closed_label="Closed"):
    return open_label if aspect_ratio > threshold else closed_label

EYE_AR_THRESH = 0.2
MOUTH_AR_THRESH = 0.3  
EYE_DROWSY_TIME = 0.5   
MOUTH_DROWSY_TIME = 1.0 

eyes_closed_start = None
mouth_open_start = None
drowsy = False

cap = cv2.VideoCapture(0)

with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
) as face_mesh:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)
        frame_height, frame_width, _ = frame.shape

        drowsy = False

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                landmarks = face_landmarks.landmark

                left_ear = calculate_eye_aspect_ratio(LEFT_EYE, landmarks)
                right_ear = calculate_eye_aspect_ratio(RIGHT_EYE, landmarks)

                left_eye_label = get_label(left_ear, EYE_AR_THRESH)
                right_eye_label = get_label(right_ear, EYE_AR_THRESH)

                cv2.putText(frame, f"Left Eye: {left_eye_label}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                cv2.putText(frame, f"Right Eye: {right_eye_label}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

                mar = calculate_mouth_aspect_ratio(landmarks)
                mouth_label = get_label(mar, MOUTH_AR_THRESH)

                cv2.putText(frame, f"Mouth: {mouth_label}", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                if left_eye_label == "Closed" and right_eye_label == "Closed":
                    if eyes_closed_start is None:
                        eyes_closed_start = time.time()
                    elif time.time() - eyes_closed_start > EYE_DROWSY_TIME:
                        drowsy = True
                else:
                    eyes_closed_start = None

                if mouth_label == "Open":
                    if mouth_open_start is None:
                        mouth_open_start = time.time()
                    elif time.time() - mouth_open_start > MOUTH_DROWSY_TIME:
                        drowsy = True
                else:
                    mouth_open_start = None

                for idx in LEFT_EYE + RIGHT_EYE + MOUTH_VERT + MOUTH_HORZ:
                    x = int(landmarks[idx].x * frame_width)
                    y = int(landmarks[idx].y * frame_height)
                    cv2.circle(frame, (x, y), 2, (0, 255, 255), -1)

        status_text = "DROWSY" if drowsy else "Not Drowsy"
        color = (0, 0, 255) if drowsy else (0, 255, 0)
        cv2.putText(frame, status_text, (10, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

        cv2.imshow('Drowsiness Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
