"""
Generate hero image candidates from full dataset and copy to labeling folder.
"""

import json
import os
import shutil
from stratified_selector import StratifiedSelector
import time

def copy_candidates_to_folder(candidates, source_base_dir, output_dir):
    """
    Copy candidate images to output folder with flattened naming.
    Replace / with # in filenames to maintain reference to original path.
    """

    copied_count = 0
    failed_count = 0

    print(f"\nCopying {len(candidates)} candidates to {output_dir}")

    for i, candidate in enumerate(candidates):
        if (i + 1) % 100 == 0:
            print(f"  Copied {i + 1}/{len(candidates)} images...")

        # Create flattened filename
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

    print(f"\nCopy complete: {copied_count} succeeded, {failed_count} failed")
    return copied_count, failed_count


def save_candidate_metadata(candidates, metadata_path):
    """Save candidate metadata to JSON file."""

    metadata = {
        'generation_info': {
            'total_candidates': len(candidates),
            'generation_time': time.strftime('%Y-%m-%d %H:%M:%S'),
            'selector_type': 'StratifiedSelector'
        },
        'category_distribution': {},
        'candidates': []
    }

    # Count by category
    category_counts = {}
    for candidate in candidates:
        category = candidate.category
        category_counts[category] = category_counts.get(category, 0) + 1

    metadata['category_distribution'] = category_counts

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
    output_dir = '/mnt/c/temp/hero-images/candidates'
    metadata_file = '/mnt/c/temp/hero-images/candidates_metadata.json'

    # Target number of candidates for labeling
    total_candidates = 5000

    # Custom category distribution - adjust based on your preferences
    category_distribution = {
        'single_large': 0.30,        # 30% - single large animals (most important for hero shots)
        'single_medium': 0.20,       # 20% - single medium animals
        'multi_same_species': 0.15,  # 15% - multiple same species (herds, etc.)
        'multi_different_species': 0.15, # 15% - multiple different species
        'rare_species': 0.10,        # 10% - rare species (always interesting)
        'high_action': 0.10          # 10% - potential action shots
    }

    print("=== Hero Image Candidate Generation ===")
    print(f"Target candidates: {total_candidates:,}")
    print(f"Source directory: {source_image_dir}")
    print(f"Output directory: {output_dir}")

    # Load detection data
    print("\nLoading detection data...")
    start_time = time.time()
    with open(detection_file, 'r') as f:
        data = json.load(f)
    load_time = time.time() - start_time
    print(f"Loaded {len(data['images']):,} images in {load_time:.1f} seconds")

    # Run stratified selection
    print(f"\nRunning stratified selection...")
    print("Category distribution:")
    for category, proportion in category_distribution.items():
        target_count = int(total_candidates * proportion)
        print(f"  {category}: {proportion:.1%} ({target_count:,} candidates)")

    start_time = time.time()
    selector = StratifiedSelector()
    candidates = selector.select_candidates(
        data,
        total_candidates=total_candidates,
        category_distribution=category_distribution
    )
    selection_time = time.time() - start_time
    print(f"\nSelection completed in {selection_time:.1f} seconds")

    # Copy images to output folder
    print(f"\nCopying candidate images...")
    start_time = time.time()
    copied, failed = copy_candidates_to_folder(candidates, source_image_dir, output_dir)
    copy_time = time.time() - start_time
    print(f"Image copying completed in {copy_time:.1f} seconds")

    # Save metadata
    print(f"\nSaving candidate metadata...")
    save_candidate_metadata(candidates, metadata_file)

    # Final summary
    print(f"\n=== Generation Summary ===")
    print(f"Total candidates generated: {len(candidates):,}")
    print(f"Images successfully copied: {copied:,}")
    print(f"Images failed to copy: {failed:,}")
    print(f"Metadata saved to: {metadata_file}")
    print(f"Images available in: {output_dir}")

    # Show category breakdown
    category_counts = {}
    for candidate in candidates:
        category = candidate.category
        category_counts[category] = category_counts.get(category, 0) + 1

    print(f"\nActual category distribution:")
    for category, count in category_counts.items():
        percentage = count / len(candidates) * 100
        print(f"  {category}: {count:,} ({percentage:.1f}%)")

    print(f"\nReady for labeling phase!")


if __name__ == "__main__":
    main()