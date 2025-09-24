import json
import os

# Verify that image paths in the detection results match actual files
detection_file = '/mnt/c/Users/dmorr/postprocessing/snapshot-safari/snapshot-safari-2025-09-19-v1000.0.0-redwood/combined_api_outputs/snapshot-safari-2025-09-19-v1000.0.0-redwood-ensemble_output_modular_image-level.md-format.within_image_smoothing.seqsmoothing.json'
image_base_dir = '/mnt/g/snapshot_safari_2024_expansion/SER'

print("Verifying image path structure...")

# Load just a small sample to verify paths
with open(detection_file, 'r') as f:
    data = json.load(f)

print(f"Total images in detection results: {len(data['images'])}")

# Check first 10 image paths
sample_images = data['images'][:10]
found_count = 0
missing_count = 0

for img in sample_images:
    file_path = img['file']
    full_path = os.path.join(image_base_dir, file_path)

    if os.path.exists(full_path):
        found_count += 1
        print(f"✓ Found: {file_path}")
    else:
        missing_count += 1
        print(f"✗ Missing: {file_path}")

print(f"\nPath verification summary:")
print(f"  Found: {found_count}/10")
print(f"  Missing: {missing_count}/10")

# Skip the slow enumeration of all files
print(f"\nImages in detection results: {len(data['images']):,}")
print("(Skipping full directory enumeration to avoid slow HDD access)")

# Show the structure we're working with
print(f"\nImage path structure example:")
print(f"  Base directory: {image_base_dir}")
print(f"  Sample relative path: {sample_images[0]['file']}")
print(f"  Full path: {os.path.join(image_base_dir, sample_images[0]['file'])}")