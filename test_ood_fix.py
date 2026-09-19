import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf

OUTPUT_DIR = Path("outputs")
TRAIN_CSV = OUTPUT_DIR / "train_split.csv"

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_face_or_features(img_np):
    """Detects human faces in the uploaded image with strict thresholds."""
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=8, minSize=(60, 60))
    if len(faces) > 0:
        return True, "Human Face / Selfie detected by Cascade Classifier."
        
    return False, "No face detected."

# Test Face Detection Guardrail on synthetic face vs skin lesion
face_img = np.zeros((224, 224, 3), dtype=np.uint8)
cv2.ellipse(face_img, (112, 112), (70, 90), 0, 0, 360, (200, 200, 220), -1) # face oval
cv2.circle(face_img, (90, 90), 12, (50, 50, 50), -1) # left eye
cv2.circle(face_img, (134, 90), 12, (50, 50, 50), -1) # right eye

is_face, msg = detect_face_or_features(face_img)
print("=" * 60)
print("       REAL-WORLD OOD & FACE GUARDRAIL TEST               ")
print("=" * 60)
print(f" Synthetic Face Test  -> Detected: {is_face} | Msg: {msg}")

# Test on real skin lesion image
train_df = pd.read_csv(TRAIN_CSV)
real_lesion_path = train_df.iloc[0]['filepath']
real_img = cv2.cvtColor(cv2.imread(real_lesion_path), cv2.COLOR_BGR2RGB)
is_lesion_face, msg_l = detect_face_or_features(real_img)
print(f" Real Skin Lesion Test -> Detected: {is_lesion_face} | Msg: {msg_l}")
print("=" * 60)
