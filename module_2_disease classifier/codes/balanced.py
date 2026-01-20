import os
import random
import shutil

# ===================== CONFIG =====================

# Disease datasets (this is where your dataset folder REALLY is)
DISEASE_BASE = "/content/drive/MyDrive/Final_year_project/module_2_disease_classifier/dataset"

# Normal split (this is where common_normal REALLY is)
NORMAL_BASE = "/content/drive/MyDrive/Final_year_project/diseases_Data/module_2_disease_classifier/common_normal"

# Output balanced dataset (we keep it with training code)
OUTPUT_BASE = "/content/drive/MyDrive/Final_year_project/module_2_disease_classifier/balanced_dataset"

DISEASES = [
    "Moyamoya_Disease_with_Intraventricular_Hemorrhage",
    "Neurofibromatosis_Type_1_NF1",
    "Optic_Glioma",
    "Tuberous_Sclerosis"
]

SPLITS = ["train", "val", "test"]
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp")
SEED = 42

# =================================================
random.seed(SEED)

def get_images_recursive(folder):
    images = []
    if not os.path.exists(folder):
        return images
    for root, _, files in os.walk(folder):
        for f in files:
            if f.lower().endswith(IMAGE_EXTS):
                images.append(os.path.join(root, f))
    return images


for disease in DISEASES:
    print(f"\n========== PROCESSING: {disease} ==========")

    for split in SPLITS:

        # -------- collect disease images (real + synthetic) --------
        disease_images = []

        for src in ["real", "synthetic"]:
            src_path = os.path.join(
                DISEASE_BASE,
                disease,
                src,
                split,
                "disease"
            )
            disease_images.extend(get_images_recursive(src_path))

        # -------- collect normal images --------
        normal_pool = get_images_recursive(os.path.join(NORMAL_BASE, split))

        if len(disease_images) == 0:
            print(f"[ERROR] No disease images for {disease} ({split})")
            continue

        if len(normal_pool) == 0:
            print(f"[ERROR] No normal images for {split}")
            continue

        # -------- SAFE BALANCING --------
        final_count = min(len(disease_images), len(normal_pool))

        disease_images = random.sample(disease_images, final_count)
        normal_images  = random.sample(normal_pool, final_count)

        # -------- output dirs --------
        out_disease = os.path.join(OUTPUT_BASE, disease, split, "disease")
        out_normal  = os.path.join(OUTPUT_BASE, disease, split, "normal")

        os.makedirs(out_disease, exist_ok=True)
        os.makedirs(out_normal, exist_ok=True)

        # -------- copy images --------
        for img in disease_images:
            shutil.copy2(img, os.path.join(out_disease, os.path.basename(img)))

        for img in normal_images:
            shutil.copy2(img, os.path.join(out_normal, os.path.basename(img)))

        print(
            f"[{split.upper()}] "
            f"Disease used: {len(disease_images)} | "
            f"Normal used: {len(normal_images)}"
        )

    print(f"✅ DONE: {disease}")

print("\n🎯 ALL BALANCED DATASETS CREATED SUCCESSFULLY")
