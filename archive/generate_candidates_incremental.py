"""
Generate hero image candidates with progress tracking and chunked processing.
"""

import json
import os
import shutil
from stratified_selector import StratifiedSelector
import time

def process_in_chunks(data, chunk_size=100000):
    """Process large dataset in chunks to show progress and manage memory."""

    print(f"Processing {len(data['images']):,} images in chunks of {chunk_size:,}")

    # Split data into chunks
    images = data['images']
    chunks = []

    for i in range(0, len(images), chunk_size):
        chunk_data = {
            'info': data['info'],
            'detection_categories': data['detection_categories'],
            'images': images[i:i + chunk_size]
        }
        chunks.append(chunk_data)

    print(f"Created {len(chunks)} chunks")
    return chunks

def main():
    # Configuration
    detection_file = '/mnt/c/Users/dmorr/postprocessing/snapshot-safari/snapshot-safari-2025-09-19-v1000.0.0-redwood/combined_api_outputs/snapshot-safari-2025-09-19-v1000.0.0-redwood-ensemble_output_modular_image-level.md-format.within_image_smoothing.seqsmoothing.json'
    source_image_dir = '/mnt/g/snapshot_safari_2024_expansion/SER'
    output_dir = '/mnt/c/temp/hero-images/candidates'

    # Start with smaller number for initial testing
    total_candidates = 1000

    print("=== Hero Image Candidate Generation (Incremental) ===")
    print(f"Target candidates: {total_candidates:,}")

    # Load detection data
    print("\nLoading detection data...")
    start_time = time.time()
    with open(detection_file, 'r') as f:
        data = json.load(f)
    load_time = time.time() - start_time
    print(f"Loaded {len(data['images']):,} images in {load_time:.1f} seconds")

    # Run selection on smaller subset first to test
    print(f"\nTesting on first 200K images...")
    test_data = {
        'info': data['info'],
        'detection_categories': data['detection_categories'],
        'images': data['images'][:200000]
    }

    start_time = time.time()
    selector = StratifiedSelector()
    candidates = selector.select_candidates(test_data, total_candidates=total_candidates)
    selection_time = time.time() - start_time
    print(f"Selection completed in {selection_time:.1f} seconds")

    # Show what we got
    category_counts = {}
    for candidate in candidates:
        category = candidate.category
        category_counts[category] = category_counts.get(category, 0) + 1

    print(f"\nSelected {len(candidates)} candidates:")
    for category, count in category_counts.items():
        percentage = count / len(candidates) * 100
        print(f"  {category}: {count:,} ({percentage:.1f}%)")

    # Copy first 10 images as test
    print(f"\nCopying first 10 candidate images as test...")
    test_candidates = candidates[:10]

    copied_count = 0
    for i, candidate in enumerate(test_candidates):
        original_path = candidate.file_path
        flattened_name = original_path.replace('/', '#').replace('\\', '#')

        source_path = os.path.join(source_image_dir, original_path)
        dest_path = os.path.join(output_dir, flattened_name)

        print(f"  {i+1}. {original_path} -> {flattened_name}")

        try:
            if os.path.exists(source_path):
                shutil.copy2(source_path, dest_path)
                copied_count += 1
                print(f"     ✓ Copied successfully")
            else:
                print(f"     ✗ Source file not found")
        except Exception as e:
            print(f"     ✗ Error: {e}")

    print(f"\nTest copy complete: {copied_count}/10 images copied")

    if copied_count > 0:
        print("\nSystem is working! Ready to process full candidate set.")
        print("Run with larger candidate count when ready.")
    else:
        print("\nIssues found with file copying. Please check paths.")

if __name__ == "__main__":
    main()