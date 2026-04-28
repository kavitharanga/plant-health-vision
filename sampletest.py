"""
Randomly picks one image per class from the dataset folder
and copies them into test_images/ for quick inference testing.
"""

import random
import shutil
from pathlib import Path

DATA_DIR = Path("data/raw/PlantVillage/PlantVillage")  # adjust if needed
OUTPUT_DIR = Path("test_images")
IMAGES_PER_CLASS = 1  # increase if you want more

OUTPUT_DIR.mkdir(exist_ok=True)

classes_found = []

for class_dir in sorted(DATA_DIR.iterdir()):
    if not class_dir.is_dir():
        continue

    images = (
        list(class_dir.glob("*.jpg"))
        + list(class_dir.glob("*.JPG"))
        + list(class_dir.glob("*.png"))
        + list(class_dir.glob("*.PNG"))
    )

    if not images:
        print(f"  ⚠️  No images found in {class_dir.name}, skipping")
        continue

    sampled = random.sample(images, min(IMAGES_PER_CLASS, len(images)))

    for img_path in sampled:
        # Rename to class_name__original_filename.jpg for clarity
        dest_name = f"{class_dir.name}__{img_path.name}"
        dest = OUTPUT_DIR / dest_name
        shutil.copy(img_path, dest)
        print(f"  ✅  {class_dir.name} → {dest_name}")

    classes_found.append(class_dir.name)

print(f"\n📁 Sampled {len(classes_found)} classes → {OUTPUT_DIR}/")
print(f"   Total images: {len(list(OUTPUT_DIR.iterdir()))}")
