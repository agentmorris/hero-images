"""
Generate full 5000 candidates with training-oriented selection.
"""

import json
import os
import shutil
from stratified_selector_training import StratifiedSelector, create_timestamped_folder
import time
from datetime import datetime
from collections import defaultdict


def copy_candidates_to_folder(candidates, source_base_dir, output_dir):
    """Copy candidate images to timestamped folder with progress tracking."""

    copied_count = 0
    failed_count = 0

    print(f"\nCopying {len(candidates)} candidates to {output_dir}")

    for i, candidate in enumerate(candidates):
        if (i + 1) % 500 == 0:
            print(f"  Progress: {i + 1}/{len(candidates)} images copied...")

        original_path = candidate.file_path
        flattened_name = original_path.replace('/', '#').replace('\\', '#')

        source_path = os.path.join(source_base_dir, original_path)
        dest_path = os.path.join(output_dir, flattened_name)

        try:
            if os.path.exists(source_path):
                shutil.copy2(source_path, dest_path)
                copied_count += 1
            else:
                failed_count += 1
                if failed_count <= 10:  # Only show first 10 failures
                    print(f"  Warning: Source file not found: {source_path}")

        except Exception as e:
            failed_count += 1
            if failed_count <= 10:
                print(f"  Error copying {original_path}: {e}")

    if failed_count > 10:
        print(f"  ... and {failed_count - 10} more failures")

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
            'description': 'Full training dataset candidates including negative examples for hero image classification'
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
            'p25': all_sizes[n//4],
            'p75': all_sizes[3*n//4],
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

    # Generate 5000 candidates
    total_candidates = 5000

    print("=== Hero Image Candidate Generation (Full - 5000 candidates) ===")
    print(f"Output directory: {output_dir}")
    print(f"Target candidates: {total_candidates:,}")

    # Load detection data
    print("\nLoading detection data...")
    start_time = time.time()
    with open(detection_file, 'r') as f:
        data = json.load(f)
    load_time = time.time() - start_time
    print(f"Loaded {len(data['images']):,} images in {load_time:.1f} seconds")

    # Run stratified selection on full dataset
    print(f"\nRunning stratified selection on full dataset...")
    start_time = time.time()
    selector = StratifiedSelector()
    candidates = selector.select_candidates(data, total_candidates=total_candidates)
    selection_time = time.time() - start_time
    print(f"Selection completed in {selection_time:.1f} seconds")

    # Analyze selection
    by_category = defaultdict(list)
    all_sizes = []

    for candidate in candidates:
        by_category[candidate.category].append(candidate)
        all_sizes.append(candidate.metadata['max_size'])

    print(f"\nSelection Analysis:")
    print(f"Total candidates: {len(candidates):,}")

    print(f"\nBy category:")
    for category, cands in by_category.items():
        sizes = [c.metadata['max_size'] for c in cands]
        percentage = len(cands) / len(candidates) * 100
        print(f"  {category}: {len(cands):,} candidates ({percentage:.1f}%)")
        if sizes:
            print(f"    Size range: {min(sizes):.4f} - {max(sizes):.4f}")
            print(f"    Size mean: {sum(sizes)/len(sizes):.4f}")

    # Overall size distribution
    if all_sizes:
        all_sizes.sort()
        n = len(all_sizes)
        print(f"\nOverall size distribution:")
        print(f"  Range: {min(all_sizes):.4f} - {max(all_sizes):.4f}")
        print(f"  10th percentile: {all_sizes[n//10]:.4f}")
        print(f"  25th percentile: {all_sizes[n//4]:.4f}")
        print(f"  Median: {all_sizes[n//2]:.4f}")
        print(f"  75th percentile: {all_sizes[3*n//4]:.4f}")
        print(f"  90th percentile: {all_sizes[9*n//10]:.4f}")

    # Copy images
    print(f"\nCopying candidate images...")
    start_time = time.time()
    copied, failed = copy_candidates_to_folder(candidates, source_image_dir, output_dir)
    copy_time = time.time() - start_time
    print(f"Image copying completed in {copy_time:.1f} seconds ({copied/copy_time:.0f} images/sec)")

    # Save metadata
    metadata_file = os.path.join(output_dir, 'candidates_metadata.json')
    print(f"\nSaving candidate metadata...")
    save_candidate_metadata(candidates, metadata_file, folder_name)

    # Final summary
    print(f"\n=== Full Generation Summary ===")
    print(f"Output folder: {folder_name}")
    print(f"Total candidates generated: {len(candidates):,}")
    print(f"Images successfully copied: {copied:,}")
    print(f"Copy failures: {failed:,}")

    if failed > 0:
        success_rate = copied / (copied + failed) * 100
        print(f"Success rate: {success_rate:.1f}%")

    print(f"\nData ready for labeling phase!")
    print(f"Candidates available in: {output_dir}")
    print(f"Metadata saved in: {metadata_file}")

    # Display summary statistics
    print(f"\nDataset characteristics:")
    print(f"  Full spectrum coverage: {min(all_sizes):.4f} - {max(all_sizes):.4f} relative size")
    print(f"  Negative examples: {by_category.get('single_tiny', []) + by_category.get('single_huge', [])} candidates")
    print(f"  Optimal candidates: {len(by_category.get('single_optimal', []))} candidates")
    print(f"  Multi-animal scenes: {len(by_category.get('multi_same_species', [])) + len(by_category.get('multi_different_species', []))} candidates")
    print(f"  Rare species: {len(by_category.get('rare_species', []))} candidates")


if __name__ == "__main__":
    main()