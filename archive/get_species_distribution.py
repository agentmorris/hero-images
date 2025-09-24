import json
from collections import defaultdict, Counter

# Analyze the full dataset to understand species distribution and data characteristics
detection_file = '/mnt/c/Users/dmorr/postprocessing/snapshot-safari/snapshot-safari-2025-09-19-v1000.0.0-redwood/combined_api_outputs/snapshot-safari-2025-09-19-v1000.0.0-redwood-ensemble_output_modular_image-level.md-format.within_image_smoothing.seqsmoothing.json'

print("Analyzing complete dataset for species distribution and characteristics...")

# Data collection
species_counts = Counter()
confidence_ranges = {'detection': [], 'classification': []}
bbox_sizes = []
total_images = 0
images_with_animals = 0
images_with_high_conf_animals = 0
images_with_classifications = 0
multi_animal_images = 0
multi_species_images = 0

# Load and parse the JSON file more efficiently
with open(detection_file, 'r') as f:
    data = json.load(f)

print(f"Total images in dataset: {len(data['images'])}")
print("Processing images...")

for i, image in enumerate(data['images']):
    if i % 100000 == 0:
        print(f"Processed {i} images...")

    total_images += 1

    if 'detections' in image and image['detections']:
        animal_detections = [d for d in image['detections'] if d['category'] == '1']

        if animal_detections:
            images_with_animals += 1

            # Check for high confidence animals
            high_conf_animals = [d for d in animal_detections if d['conf'] >= 0.3]
            if high_conf_animals:
                images_with_high_conf_animals += 1

            # Check for multiple animals
            if len(animal_detections) > 1:
                multi_animal_images += 1

            # Process each animal detection
            image_species = set()
            has_classifications = False

            for det in animal_detections:
                confidence_ranges['detection'].append(det['conf'])

                # Calculate bbox size
                bbox = det['bbox']
                bbox_size = bbox[2] * bbox[3]  # width * height
                bbox_sizes.append(bbox_size)

                # Process classifications
                if 'classifications' in det and det['classifications']:
                    has_classifications = True
                    for cls in det['classifications']:
                        # Classifications are in format [species_id, confidence]
                        species_id, cls_conf = cls[0], cls[1]
                        confidence_ranges['classification'].append(cls_conf)

                        if cls_conf >= 0.5:  # Only count high-confidence species
                            species_counts[species_id] += 1
                            image_species.add(species_id)

            if has_classifications:
                images_with_classifications += 1

            # Check for multiple species in one image
            if len(image_species) > 1:
                multi_species_images += 1

print(f"\nDataset Summary:")
print(f"Total images: {total_images:,}")
print(f"Images with animals: {images_with_animals:,} ({images_with_animals/total_images*100:.1f}%)")
print(f"Images with high-conf animals (>0.3): {images_with_high_conf_animals:,} ({images_with_high_conf_animals/total_images*100:.1f}%)")
print(f"Images with classifications: {images_with_classifications:,} ({images_with_classifications/total_images*100:.1f}%)")
print(f"Multi-animal images: {multi_animal_images:,} ({multi_animal_images/images_with_animals*100:.1f}% of animal images)")
print(f"Multi-species images: {multi_species_images:,} ({multi_species_images/images_with_animals*100:.1f}% of animal images)")

print(f"\nTop 20 Species (by high-confidence detections):")
for species_id, count in species_counts.most_common(20):
    print(f"  Species {species_id}: {count:,} detections")

print(f"\nBounding Box Size Distribution:")
if bbox_sizes:
    bbox_sizes.sort()
    n = len(bbox_sizes)
    print(f"  Min: {min(bbox_sizes):.4f}")
    print(f"  25th percentile: {bbox_sizes[n//4]:.4f}")
    print(f"  Median: {bbox_sizes[n//2]:.4f}")
    print(f"  75th percentile: {bbox_sizes[3*n//4]:.4f}")
    print(f"  Max: {max(bbox_sizes):.4f}")

print(f"\nDetection Confidence Distribution:")
if confidence_ranges['detection']:
    det_conf = confidence_ranges['detection']
    det_conf.sort()
    n = len(det_conf)
    print(f"  Min: {min(det_conf):.3f}")
    print(f"  25th percentile: {det_conf[n//4]:.3f}")
    print(f"  Median: {det_conf[n//2]:.3f}")
    print(f"  75th percentile: {det_conf[3*n//4]:.3f}")
    print(f"  Max: {max(det_conf):.3f}")

print(f"\nClassification Confidence Distribution:")
if confidence_ranges['classification']:
    cls_conf = confidence_ranges['classification']
    cls_conf.sort()
    n = len(cls_conf)
    print(f"  Min: {min(cls_conf):.3f}")
    print(f"  25th percentile: {cls_conf[n//4]:.3f}")
    print(f"  Median: {cls_conf[n//2]:.3f}")
    print(f"  75th percentile: {cls_conf[3*n//4]:.3f}")
    print(f"  Max: {max(cls_conf):.3f}")

# Identify potential "hero image" candidates based on heuristics
print(f"\nPotential Hero Image Heuristics:")
large_animals = len([s for s in bbox_sizes if s > 0.1])  # Animals taking up >10% of image
centered_threshold = 0.1  # Distance from center
very_high_conf = len([c for c in confidence_ranges['detection'] if c > 0.8])

print(f"  Large animals (>10% of image): {large_animals:,}")
print(f"  Very high confidence detections (>0.8): {very_high_conf:,}")
print(f"  Multi-animal images: {multi_animal_images:,}")
print(f"  Multi-species images: {multi_species_images:,}")