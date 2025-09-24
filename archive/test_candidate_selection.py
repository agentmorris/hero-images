"""
Test the candidate selection system on a subset of data.
"""

import json
from candidate_selector import HeuristicSelector, RandomSelector, load_detection_data
import time

# Test with a smaller subset first to verify the system works
detection_file = '/mnt/c/Users/dmorr/postprocessing/snapshot-safari/snapshot-safari-2025-09-19-v1000.0.0-redwood/combined_api_outputs/snapshot-safari-2025-09-19-v1000.0.0-redwood-ensemble_output_modular_image-level.md-format.within_image_smoothing.seqsmoothing.json'

print("Loading detection data...")
start_time = time.time()
data = load_detection_data(detection_file)
load_time = time.time() - start_time
print(f"Loaded {len(data['images']):,} images in {load_time:.1f} seconds")

# Test with first 50,000 images to avoid memory issues
print("\nTesting with subset of data...")
subset_data = {
    'info': data['info'],
    'detection_categories': data['detection_categories'],
    'images': data['images'][:50000]
}

print(f"Testing on {len(subset_data['images']):,} images")

# Test heuristic selector
print("\n=== Testing Heuristic Selector ===")
start_time = time.time()
heuristic_selector = HeuristicSelector()
heuristic_candidates = heuristic_selector.select_candidates(subset_data, num_candidates=20)
heuristic_time = time.time() - start_time

print(f"Selected {len(heuristic_candidates)} candidates in {heuristic_time:.1f} seconds")
print("\nTop 10 heuristic candidates:")
for i, candidate in enumerate(heuristic_candidates[:10]):
    criteria = candidate.selection_criteria
    print(f"{i+1:2d}. Score: {candidate.score:6.2f} | {candidate.file_path}")
    print(f"     Animals: {criteria.get('num_animals', 0)}, "
          f"Max size: {criteria.get('max_animal_size', 0):.4f}, "
          f"Species: {criteria.get('species_count', 0)}, "
          f"Max conf: {criteria.get('max_confidence', 0):.3f}")

# Test random selector for comparison
print("\n=== Testing Random Selector ===")
start_time = time.time()
random_selector = RandomSelector()
random_candidates = random_selector.select_candidates(subset_data, num_candidates=10)
random_time = time.time() - start_time

print(f"Selected {len(random_candidates)} candidates in {random_time:.1f} seconds")
print("\nRandom candidates (for comparison):")
for i, candidate in enumerate(random_candidates[:5]):
    print(f"{i+1}. {candidate.file_path}")

# Analyze the characteristics of heuristic vs random selections
print("\n=== Selection Analysis ===")

def analyze_candidates(candidates, name):
    if not candidates:
        return

    sizes = []
    confidences = []
    animal_counts = []
    species_counts = []

    for candidate in candidates:
        criteria = candidate.selection_criteria
        if criteria:
            sizes.append(criteria.get('max_animal_size', 0))
            confidences.append(criteria.get('max_confidence', 0))
            animal_counts.append(criteria.get('num_animals', 0))
            species_counts.append(criteria.get('species_count', 0))

    if sizes:
        print(f"\n{name} Selection Characteristics:")
        print(f"  Avg max animal size: {sum(sizes)/len(sizes):.4f}")
        print(f"  Avg max confidence: {sum(confidences)/len(confidences):.3f}")
        print(f"  Avg animals per image: {sum(animal_counts)/len(animal_counts):.1f}")
        print(f"  Avg species per image: {sum(species_counts)/len(species_counts):.1f}")
        print(f"  Images with multiple animals: {sum(1 for c in animal_counts if c > 1)}/{len(animal_counts)}")
        print(f"  Images with multiple species: {sum(1 for c in species_counts if c > 1)}/{len(species_counts)}")

analyze_candidates(heuristic_candidates, "Heuristic")
analyze_candidates(random_candidates, "Random")

# Test performance on different sizes
print("\n=== Performance Testing ===")
test_sizes = [1000, 5000, 10000]

for size in test_sizes:
    test_data = {
        'info': data['info'],
        'detection_categories': data['detection_categories'],
        'images': data['images'][:size]
    }

    start_time = time.time()
    candidates = heuristic_selector.select_candidates(test_data, num_candidates=100)
    elapsed = time.time() - start_time

    print(f"  {size:,} images -> {len(candidates)} candidates in {elapsed:.2f}s ({size/elapsed:.0f} images/sec)")

print("\nCandidate selection system test completed successfully!")