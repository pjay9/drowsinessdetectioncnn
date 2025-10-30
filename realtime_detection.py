import argparse, time, os, sys, threading, collections
import cv2
import numpy as np
import torch
import mediapipe as mp

from queue import Queue

# -------------------- Utilities --------------------
def build_argparser():
    p = argparse.ArgumentParser(description="Real-time detection (fast)")
    p.add_argument("--camera", type=int, default=0, help="Camera device index")
    p.add_argument("--eyes_model", type=str, default="model_eyes_weights.pth", help="Path to eyes model (.pth)")
    p.add_argument("--mouth_model", type=str, default="model_mouth_weights.pth", help="Path to mouth model (.pth)")
    p.add_argument("--img_size", type=int, default=64, help="Input size for models (square)")
    p.add_argument("--use_cuda", action="store_true", help="Force use CUDA if available")
    p.add_argument("--half", action="store_true", help="Use half precision (FP16) on CUDA for speed")
    return p

# Threaded video capture to avoid blocking the main loop
class ThreadedCamera:
    def __init__(self, src=0, width=640, height=480, queue_size=3):
        self.cap = cv2.VideoCapture(src, cv2.CAP_DSHOW if os.name == 'nt' else 0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.q = Queue(maxsize=queue_size)
        self.stopped = False
        t = threading.Thread(target=self._reader, daemon=True)
        t.start()

    def _reader(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            if not ret:
                self.stop()
                break
            if not self.q.full():
                self.q.put(frame)

    def read(self):
        if self.q.empty():
            return None
        return self.q.get()

    def stop(self):
        self.stopped = True
        try:
            self.cap.release()
        except:
            pass

# -------------------- Model loader --------------------
def try_load_model(path, device, half=False):
    if not os.path.exists(path):
        print(f"[WARN] Model file not found: {path}")
        return None
    # Attempt to load state_dict or scripted model
    try:
        # Try scripted or state dict
        loaded = torch.load(path, map_location=device)
        # If it's a dict of state_dict, return it as raw (user must match architecture)
        # We will try to detect a simple classifier structure dynamically during inference
        print(f"[INFO] Loaded PyTorch object from {path}. Type: {type(loaded)}")
        # If it's a scripted module
        if isinstance(loaded, torch.jit.ScriptModule) or isinstance(loaded, torch.nn.Module):
            model = loaded
        elif isinstance(loaded, dict):
            # We'll create a tiny conv-net matching common training scripts if user didn't provide architecture
            # But it's risky to invent architecture; instead we wrap the state dict into a small conv model if possible
            model = make_default_model()
            model.load_state_dict(loaded)
        else:
            # unknown type - try to wrap
            model = loaded
        model.to(device)
        model.eval()
        if half and device.type=='cuda':
            try:
                model.half()
            except Exception as e:
                print("[WARN] Failed to convert model to half precision:", e)
        return model
    except Exception as e:
        print(f"[ERROR] Failed to load model {path}: {e}")
        return None

def make_default_model(num_classes=2, in_channels=3, img_size=64):
    # Very small conv net similar to many toy classifiers
    import torch.nn as nn
    class TinyNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(in_channels, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
            )
            self.fc = nn.Linear(64, num_classes)
        def forward(self, x):
            x = self.net(x)
            x = x.view(x.size(0), -1)
            return self.fc(x)
    return TinyNet()

# -------------------- Preprocessing --------------------
def preprocess_roi(roi_bgr, img_size=64, device=torch.device("cpu"), half=False):
    # roi_bgr: HxWx3 BGR image from OpenCV
    # Convert to RGB, resize, scale to [0,1], to tensor (C,H,W)
    img = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    # To tensor
    t = torch.from_numpy(arr).permute(2,0,1).unsqueeze(0)  # 1,C,H,W
    t = t.to(device)
    if half and device.type=='cuda':
        t = t.half()
    return t

# -------------------- Landmark helpers (MediaPipe Face Mesh indices) --------------------
mp_face_mesh = mp.solutions.face_mesh
# Landmarks for left and right eyes (approximate) and outer mouth
LEFT_EYE_IDX = [33, 133, 159, 145, 153, 154]   # example set
RIGHT_EYE_IDX = [362, 263, 386, 382, 388, 387] # example set
MOUTH_IDX = [61, 291, 0, 17, 13, 14]           # example set

def bbox_from_landmarks(landmarks, idxs, frame_w, frame_h, pad=0.25):
    xs = [landmarks[i].x*frame_w for i in idxs]
    ys = [landmarks[i].y*frame_h for i in idxs]
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    w = x2-x1; h = y2-y1
    cx = x1 + w/2; cy = y1 + h/2
    size = max(w,h) * (1+pad)
    x1n = int(max(0, cx - size/2)); y1n = int(max(0, cy - size/2))
    x2n = int(min(frame_w-1, cx + size/2)); y2n = int(min(frame_h-1, cy + size/2))
    if x2n<=x1n or y2n<=y1n:
        return None
    return x1n, y1n, x2n, y2n

# -------------------- Main loop --------------------
def main(args):
    device = torch.device("cuda" if (torch.cuda.is_available() and args.use_cuda) else "cpu")
    print(f"[INFO] Using device: {device}")

    eyes_model = try_load_model(args.eyes_model, device, half=args.half)
    mouth_model = try_load_model(args.mouth_model, device, half=args.half)

    cam = ThreadedCamera(src=args.camera, width=640, height=480, queue_size=4)
    time.sleep(0.2)

    # MediaPipe face mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1,
                                      refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)

    fps_deque = collections.deque(maxlen=30)
    last = time.time()

    try:
        while True:
            frame = cam.read()
            if frame is None:
                # small sleep to wait for frames
                time.sleep(0.005)
                continue
            h, w = frame.shape[:2]
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(img_rgb)

            eyes_pred_text = ""
            mouth_pred_text = ""

            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark
                # Left eye bbox & right eye bbox
                left_bbox = bbox_from_landmarks(landmarks, LEFT_EYE_IDX, w, h, pad=0.6)
                right_bbox = bbox_from_landmarks(landmarks, RIGHT_EYE_IDX, w, h, pad=0.6)
                mouth_bbox = bbox_from_landmarks(landmarks, MOUTH_IDX, w, h, pad=0.8)

                # Draw landmarks (optional light)
                for lm in [33, 263, 1, 61, 291]:
                    x = int(landmarks[lm].x*w); y = int(landmarks[lm].y*h)
                    cv2.circle(frame, (x,y), 1, (0,255,0), -1)

                # Eyes: prepare and infer if model available
                if eyes_model is not None and (left_bbox or right_bbox):
                    rois = []
                    if left_bbox:
                        x1,y1,x2,y2 = left_bbox
                        left_roi = frame[y1:y2, x1:x2]
                        rois.append(("L", left_roi))
                    if right_bbox:
                        x1,y1,x2,y2 = right_bbox
                        right_roi = frame[y1:y2, x1:x2]
                        rois.append(("R", right_roi))

                    preds = []
                    for side, roi in rois:
                        try:
                            t = preprocess_roi(roi, img_size=args.img_size, device=device, half=args.half)
                            with torch.no_grad():
                                out = eyes_model(t)
                                if isinstance(out, (tuple,list)):
                                    out = out[0]
                                out = out.softmax(dim=1) if out.shape[-1]>1 else out.sigmoid()
                                val, idx = out.max(dim=1)
                                preds.append((side, float(val.item()), int(idx.item())))
                        except Exception as e:
                            preds.append((side, None, None))
                    # Build text
                    if preds:
                        eyes_pred_text = "Eyes: " + ", ".join([f"{p[0]}[{p[2]}:{p[1]:.2f}]" for p in preds])

                # Mouth: prepare and infer if available
                if mouth_model is not None and mouth_bbox:
                    x1,y1,x2,y2 = mouth_bbox
                    roi = frame[y1:y2, x1:x2]
                    try:
                        t = preprocess_roi(roi, img_size=args.img_size, device=device, half=args.half)
                        with torch.no_grad():
                            out = mouth_model(t)
                            if isinstance(out, (tuple,list)):
                                out = out[0]
                            out = out.softmax(dim=1) if out.shape[-1]>1 else out.sigmoid()
                            val, idx = out.max(dim=1)
                            mouth_pred_text = f"Mouth[{int(idx.item())}:{float(val.item()):.2f}]"
                    except Exception as e:
                        mouth_pred_text = f"Mouth: err"

                # Draw boxes and text
                if left_bbox:
                    x1,y1,x2,y2 = left_bbox
                    cv2.rectangle(frame, (x1,y1),(x2,y2),(255,0,0),1)
                if right_bbox:
                    x1,y1,x2,y2 = right_bbox
                    cv2.rectangle(frame, (x1,y1),(x2,y2),(255,0,0),1)
                if mouth_bbox:
                    x1,y1,x2,y2 = mouth_bbox
                    cv2.rectangle(frame, (x1,y1),(x2,y2),(0,0,255),1)

            if eyes_pred_text or mouth_pred_text:
                print(f"Eyes: {eyes_pred_text} | Mouth: {mouth_pred_text}")

            # FPS calc
            now = time.time()
            fps_deque.append(1.0/(now-last) if now!=last else 0.0)
            last = now
            fps = sum(fps_deque)/len(fps_deque) if fps_deque else 0.0

            # Overlay
            y0 = 20
            cv2.putText(frame, f"FPS: {fps:.1f}", (10,y0), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0),2)
            if eyes_pred_text:
                cv2.putText(frame, eyes_pred_text, (10,y0+25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,0),1)
            if mouth_pred_text:
                cv2.putText(frame, mouth_pred_text, (10,y0+45), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,255),1)

            cv2.imshow("Realtime Detection (q to quit)", frame)
            key = cv2.waitKey(1) & 0xFF
            if key==ord('q'):
                break
    finally:
        cam.stop()
        cv2.destroyAllWindows()
        face_mesh.close()

if __name__ == "__main__":
    args = build_argparser().parse_args()
    main(args)
