"""
Generate test batch of 100 candidates with full spectrum including negatives.
"""

import json
import os
import shutil
from stratified_selector_training import StratifiedSelector, create_timestamped_folder
import time
from datetime import datetime
from collections import defaultdict


def copy_candidates_to_folder(candidates, source_base_dir, output_dir):
    """Copy candidate images to timestamped folder."""

    copied_count = 0
    failed_count = 0

    print(f"\nCopying {len(candidates)} candidates to {output_dir}")

    for i, candidate in enumerate(candidates):
        original_path = candidate.file_path
        flattened_name = original_path.replace('/', '#').replace('\\', '#')

        source_path = os.path.join(source_base_dir, original_path)
        dest_path = os.path.join(output_dir, flattened_name)

        try:
            if os.path.exists(source_path):
                shutil.copy2(source_path, dest_path)
                copied_count += 1
            else:
                print(f"  Warning: Source file not found: {source_path}")
                failed_count += 1

        except Exception as e:
            print(f"  Error copying {original_path}: {e}")
            failed_count += 1

    print(f"Copy complete: {copied_count} succeeded, {failed_count} failed")
    return copied_count, failed_count


def save_candidate_metadata(candidates, metadata_path, folder_name):
    """Save candidate metadata to JSON file."""

    metadata = {
        'generation_info': {
            'total_candidates': len(candidates),
            'generation_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'selector_type': 'StratifiedSelector_Training',
            'folder_name': folder_name,
            'description': 'Training dataset candidates including negative examples'
        },
        'category_distribution': {},
        'size_distribution': {},
        'candidates': []
    }

    # Count by category
    category_counts = {}
    all_sizes = []

    for candidate in candidates:
        category = candidate.category
        category_counts[category] = category_counts.get(category, 0) + 1
        all_sizes.append(candidate.metadata['max_size'])

    metadata['category_distribution'] = category_counts

    # Size distribution statistics
    if all_sizes:
        all_sizes.sort()
        n = len(all_sizes)
        metadata['size_distribution'] = {
            'min': min(all_sizes),
            'max': max(all_sizes),
            'median': all_sizes[n//2],
            'p10': all_sizes[n//10],
            'p90': all_sizes[9*n//10],
            'mean': sum(all_sizes) / len(all_sizes)
        }

    # Store candidate details
    for candidate in candidates:
        candidate_info = {
            'original_path': candidate.file_path,
            'flattened_filename': candidate.file_path.replace('/', '#').replace('\\', '#'),
            'category': candidate.category,
            'quality_score': candidate.quality_score,
            'metadata': candidate.metadata
        }
        metadata['candidates'].append(candidate_info)

    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved metadata to {metadata_path}")


def main():
    # Configuration
    detection_file = '/mnt/c/Users/dmorr/postprocessing/snapshot-safari/snapshot-safari-2025-09-19-v1000.0.0-redwood/combined_api_outputs/snapshot-safari-2025-09-19-v1000.0.0-redwood-ensemble_output_modular_image-level.md-format.within_image_smoothing.seqsmoothing.json'
    source_image_dir = '/mnt/g/snapshot_safari_2024_expansion/SER'
    base_candidates_dir = '/mnt/c/temp/hero-images/candidates'

    # Create timestamped subfolder
    output_dir = create_timestamped_folder(base_candidates_dir)
    folder_name = os.path.basename(output_dir)

    # Test with 100 candidates first
    total_candidates = 100

    print("=== Hero Image Candidate Generation (Test - 100 candidates) ===")
    print(f"Output directory: {output_dir}")
    print(f"Target candidates: {total_candidates}")

    # Load detection data
    print("\nLoading detection data...")
    start_time = time.time()
    with open(detection_file, 'r') as f:
        data = json.load(f)
    load_time = time.time() - start_time
    print(f"Loaded {len(data['images']):,} images in {load_time:.1f} seconds")

    # Use subset for testing
    print(f"\nTesting on first 200K images...")
    test_data = {
        'info': data['info'],
        'detection_categories': data['detection_categories'],
        'images': data['images'][:200000]
    }

    # Run stratified selection
    print(f"\nRunning stratified selection (including negatives)...")
    start_time = time.time()
    selector = StratifiedSelector()
    candidates = selector.select_candidates(test_data, total_candidates=total_candidates)
    selection_time = time.time() - start_time
    print(f"Selection completed in {selection_time:.1f} seconds")

    # Analyze selection
    by_category = defaultdict(list)
    all_sizes = []

    for candidate in candidates:
        by_category[candidate.category].append(candidate)
        all_sizes.append(candidate.metadata['max_size'])

    print(f"\nSelection Analysis:")
    print(f"Total candidates: {len(candidates)}")

    print(f"\nBy category:")
    for category, cands in by_category.items():
        sizes = [c.metadata['max_size'] for c in cands]
        print(f"  {category}: {len(cands)} candidates")
        if sizes:
            print(f"    Size range: {min(sizes):.4f} - {max(sizes):.4f}")

    # Overall size distribution
    if all_sizes:
        all_sizes.sort()
        n = len(all_sizes)
        print(f"\nOverall size distribution:")
        print(f"  Range: {min(all_sizes):.4f} - {max(all_sizes):.4f}")
        print(f"  10th percentile: {all_sizes[n//10]:.4f}")
        print(f"  Median: {all_sizes[n//2]:.4f}")
        print(f"  90th percentile: {all_sizes[9*n//10]:.4f}")

    # Copy images
    print(f"\nCopying candidate images...")
    start_time = time.time()
    copied, failed = copy_candidates_to_folder(candidates, source_image_dir, output_dir)
    copy_time = time.time() - start_time
    print(f"Image copying completed in {copy_time:.1f} seconds")

    # Save metadata
    metadata_file = os.path.join(output_dir, 'candidates_metadata.json')
    print(f"\nSaving candidate metadata...")
    save_candidate_metadata(candidates, metadata_file, folder_name)

    # Final summary
    print(f"\n=== Test Generation Summary ===")
    print(f"Output folder: {folder_name}")
    print(f"Total candidates: {len(candidates)}")
    print(f"Images copied: {copied}/{len(candidates)}")
    print(f"Ready for visual review!")

    if copied > 0:
        print(f"\nNext step: Review images in {output_dir}")
        print("If distribution looks good, proceed with full candidate generation.")


if __name__ == "__main__":
    main()