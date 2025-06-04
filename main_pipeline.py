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
detection_folder = os.path.join(root_dir, "detection")

os.makedirs(input_folder, exist_ok=True)
os.makedirs(enhanced_folder, exist_ok=True)
os.makedirs(detection_folder, exist_ok=True)

# === STEP 1: Image Enhancement ===
def enhance_radar_image(image):
    """TV denoising + CLAHE + upscaling"""
    tv_img = denoise_tv_chambolle(image / 255.0, weight=0.03)
    tv_img = (tv_img * 255).astype(np.uint8)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
    clahe_img = clahe.apply(tv_img)

    upscaled = cv2.resize(clahe_img, (image.shape[1]*2, image.shape[0]*2), interpolation=cv2.INTER_LANCZOS4)
    return upscaled

# === STEP 2: Object Detection with Filtering ===
def detect_objects(enhanced_img):
    """Balanced detection — includes cars, filters clutter"""
    blur = cv2.GaussianBlur(enhanced_img, (5, 5), 0)
    binary = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        blockSize=21,
        C=-5
    )

    # Clean up tree/sky zone
    binary[:100, :] = 0

    labeled = label(binary)
    bboxes = []
    for region in regionprops(labeled):
        minr, minc, maxr, maxc = region.bbox
        area = region.area
        height = maxr - minr
        width = maxc - minc
        aspect_ratio = width / height if height != 0 else 0

        # ✅ Balanced filtering
        if 300 <= area <= 8000 and 0.3 <= aspect_ratio <= 4.5:
            if height > 10 and width > 10:  # remove micro-rects
                bboxes.append((minr, minc, maxr, maxc))

    return bboxes, binary

# === STEP 3: Full Pipeline Execution ===
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

        # Step 1: Enhance
        enhanced = enhance_radar_image(orig)
        cv2.imwrite(os.path.join(enhanced_folder, f"enh_{filename}"), enhanced)

        # Step 2: SSIM
        resized_orig = cv2.resize(orig, (enhanced.shape[1], enhanced.shape[0]))
        ssim_score = ssim(resized_orig, enhanced)

        # Step 3: Detect
        boxes, _ = detect_objects(enhanced)
        detection_counts.append(len(boxes))
        ssim_scores.append(ssim_score)

        # Step 4: Save output
        output = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        for minr, minc, maxr, maxc in boxes:
            cv2.rectangle(output, (minc, minr), (maxc, maxr), (0, 0, 255), 2)
        cv2.imwrite(os.path.join(detection_folder, f"detect_{filename}"), output)

    # Step 5: Save results
    df = pd.DataFrame({
        "Image Name": image_files,
        "SSIM (Original vs Enhanced)": ssim_scores,
        "Detected Objects": detection_counts
    })
    df.to_csv(os.path.join(root_dir, "results.csv"), index=False)
    print("✅ DONE! Cleaned detections + results.csv saved")

# Run script
if __name__ == "__main__":
    run_pipeline()
