# app.py
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
from PIL import Image
from ultralytics import YOLO
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from passporteye import read_mrz
import os

app = FastAPI(title="Fully Rotation-Proof MRZ Extraction API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------
# Load models once
# ---------------------------
yolo_model = YOLO("yolov8n.pt")  # MRZ-finetuned YOLO recommended
trocr_processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-stage1")
trocr_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-stage1")

# ---------------------------
# Helper: OpenCV rotate
# ---------------------------
def rotate_image_cv2(image, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(255,255,255))
    return rotated

# ---------------------------
# Helper: Deskew MRZ
# ---------------------------
def deskew_mrz(mrz_crop):
    gray = cv2.cvtColor(mrz_crop, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    coords = np.column_stack(np.where(bw > 0))
    if len(coords) == 0:
        return mrz_crop
    rect = cv2.minAreaRect(coords)
    angle = rect[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = mrz_crop.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(mrz_crop, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

# ---------------------------
# Helper: Enhance MRZ for OCR
# ---------------------------
def enhance_for_ocr(mrz_crop):
    h, w = mrz_crop.shape[:2]
    if h > w:
        mrz_crop = cv2.rotate(mrz_crop, cv2.ROTATE_90_CLOCKWISE)
    mrz_resized = cv2.resize(mrz_crop, (512, 256))
    gray = cv2.cvtColor(mrz_resized, cv2.COLOR_BGR2GRAY)
    enhanced = cv2.equalizeHist(gray)
    mrz_final = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    mrz_final = cv2.copyMakeBorder(mrz_final, 20, 20, 0, 0, cv2.BORDER_CONSTANT, value=[255,255,255])
    return mrz_final

# ---------------------------
# Helper: OCR MRZ with rotation trials
# ---------------------------
def ocr_mrz_with_rotations(mrz_crop):
    best_text = ""
    best_crop = mrz_crop.copy()
    for angle in [0, 90, 180, 270]:
        crop_rot = rotate_image_cv2(mrz_crop, angle)
        crop_rot = enhance_for_ocr(crop_rot)
        mrz_image = Image.fromarray(cv2.cvtColor(crop_rot, cv2.COLOR_BGR2RGB))
        pixel_values = trocr_processor(images=mrz_image, return_tensors="pt").pixel_values
        generated_ids = trocr_model.generate(pixel_values)
        text = trocr_processor.batch_decode(generated_ids, skip_special_tokens=True)[0].replace("\n","").strip()
        if len(text) > len(best_text):
            best_text = text
            best_crop = crop_rot
    return best_text, best_crop

# ---------------------------
# MRZ Extraction Endpoint
# ---------------------------
@app.post("/extract_mrz/")
async def extract_mrz(file: UploadFile = File(...)):
    img_array = np.frombuffer(await file.read(), np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    mrz_crop = None
    mrz_text = ""
    mrz_dict = {}

    # Try full image rotations 0,90,180,270 until YOLO detects MRZ
    found = False
    for angle in [0, 90, 180, 270]:
        rotated_img = rotate_image_cv2(img, angle) if angle != 0 else img
        results = yolo_model(rotated_img)
        if len(results[0].boxes) > 0:
            img = rotated_img
            x1, y1, x2, y2 = map(int, results[0].boxes.xyxy[0].cpu().numpy())
            mrz_crop = img[y1:y2, x1:x2]
            found = True
            break

    if found and mrz_crop is not None:
        mrz_crop = deskew_mrz(mrz_crop)
        mrz_text, mrz_crop = ocr_mrz_with_rotations(mrz_crop)

    # PassportEye fallback on full image
    if not mrz_text or len(mrz_text) < 30:
        temp_path = "/tmp/full_mrz_temp.jpg"
        cv2.imwrite(temp_path, img)
        mrz_obj = read_mrz(temp_path)
        if mrz_obj is not None:
            mrz_dict = mrz_obj.to_dict()
            mrz_text = mrz_dict.get("raw_text", "")
        else:
            mrz_text = "MRZ not detected"
            mrz_dict = {}
        os.remove(temp_path)
    elif mrz_crop is not None:
        temp_path = "/tmp/mrz_crop_temp.jpg"
        mrz_image = Image.fromarray(cv2.cvtColor(mrz_crop, cv2.COLOR_BGR2RGB))
        mrz_image.save(temp_path)
        mrz_obj = read_mrz(temp_path)
        if mrz_obj is not None:
            mrz_dict = mrz_obj.to_dict()
        else:
            mrz_dict = {"raw": mrz_text}
        os.remove(temp_path)

    return {"mrz_text": mrz_text, "mrz_fields": mrz_dict}

# ---------------------------
# Run:
# uvicorn app:app --reload --host 0.0.0.0 --port 8000
# ---------------------------
