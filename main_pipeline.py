import cv2
import os
import numpy as np
import pandas as pd
from skimage.restoration import denoise_tv_chambolle
from skimage.metrics import structural_similarity as ssim
from skimage.measure import label, regionprops

# === CONFIGURATION ===
root_dir = "radar_project"
input_folder = os.path.join(root_dir, "radar_dataset")
enhanced_folder = os.path.join(root_dir, "enhanced_outputs")
detection_folder = os.path.join(root_dir, "detections")

# Create necessary folders
os.makedirs(input_folder, exist_ok=True)
os.makedirs(enhanced_folder, exist_ok=True)
os.makedirs(detection_folder, exist_ok=True)

# === PIPELINE FUNCTIONS ===
def enhance_radar_image(image):
    """Apply TV denoising, CLAHE, and upscaling"""
    tv_img = denoise_tv_chambolle(image / 255.0, weight=0.1)
    tv_img = (tv_img * 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(tv_img)
    upscaled_img = cv2.resize(clahe_img, (image.shape[1]*2, image.shape[0]*2), interpolation=cv2.INTER_LANCZOS4)
    return upscaled_img

def detect_objects(enhanced_img):
    """Detect blobs using Otsu threshold and connected components"""
    _, binary = cv2.threshold(enhanced_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    labeled = label(binary)
    bboxes = [region.bbox for region in regionprops(labeled) if region.area >= 20]
    return bboxes, binary

# === MAIN PIPELINE ===
def run_pipeline():
    image_files = sorted([f for f in os.listdir(input_folder) if f.lower().endswith((".png", ".jpg", ".jpeg"))])

    ssim_scores = []
    detection_counts = []

    for filename in image_files:
        orig_path = os.path.join(input_folder, filename)
        orig = cv2.imread(orig_path, cv2.IMREAD_GRAYSCALE)
        if orig is None:
            print(f"Skipped unreadable image: {filename}")
            continue

        # --- Enhancement ---
        enhanced = enhance_radar_image(orig)
        enhanced_path = os.path.join(enhanced_folder, f"enh_{filename}")
        cv2.imwrite(enhanced_path, enhanced)

        # --- SSIM ---
        resized_orig = cv2.resize(orig, (enhanced.shape[1], enhanced.shape[0]))
        ssim_score = ssim(resized_orig, enhanced)

        # --- Detection ---
        boxes, _ = detect_objects(enhanced)
        detection_counts.append(len(boxes))
        ssim_scores.append(ssim_score)

        # --- Draw and save detections ---
        output = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        for box in boxes:
            minr, minc, maxr, maxc = box
            cv2.rectangle(output, (minc, minr), (maxc, maxr), (0, 0, 255), 2)
        cv2.imwrite(os.path.join(detection_folder, f"detect_{filename}"), output)

    # --- Save results ---
    df = pd.DataFrame({
        "Image Name": image_files,
        "SSIM (Original vs Enhanced)": ssim_scores,
        "Detected Objects": detection_counts
    })
    df.to_csv(os.path.join(root_dir, "results.csv"), index=False)
    print(df)

# Run it
if __name__ == "__main__":
    run_pipeline()
