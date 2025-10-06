"""
Generate sequence-aware hero image candidates - OPTIMIZED VERSION
"""

import json
import os
import shutil
from stratified_selector_sequence_aware import SequenceAwareStratifiedSelector, create_timestamped_folder
import time
from datetime import datetime
from collections import defaultdict
from typing import Dict, Any, List


def copy_candidates_to_folder(candidates, source_base_dir, output_dir):
    """Copy candidate images to timestamped folder with progress tracking."""

    copied_count = 0
    failed_count = 0

    print(f"\nCopying {len(candidates)} candidates to {output_dir}")

    for i, candidate in enumerate(candidates):
        if (i + 1) % 250 == 0:
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
                if failed_count <= 5:  # Reduced from 10
                    print(f"  Warning: Source file not found: {source_path}")

        except Exception as e:
            failed_count += 1
            if failed_count <= 5:
                print(f"  Error copying {original_path}: {e}")

    if failed_count > 5:
        print(f"  ... and {failed_count - 5} more failures")

    print(f"Copy complete: {copied_count} succeeded, {failed_count} failed")
    return copied_count, failed_count


def save_candidate_metadata_optimized(candidates, metadata_path, folder_name):
    """Save candidate metadata - OPTIMIZED to write less data."""

    print("Preparing metadata...")

    # Calculate statistics first
    category_counts = {}
    all_sizes = []
    seq_ids = []
    seq_sizes = []

    for candidate in candidates:
        category = candidate.category
        category_counts[category] = category_counts.get(category, 0) + 1
        all_sizes.append(candidate.metadata['max_size'])
        seq_ids.append(candidate.seq_id)
        seq_sizes.append(candidate.metadata.get('sequence_size', 1))

    # Size distribution statistics
    all_sizes.sort()
    n = len(all_sizes)
    size_stats = {
        'min': min(all_sizes),
        'max': max(all_sizes),
        'median': all_sizes[n//2],
        'p10': all_sizes[n//10],
        'p25': all_sizes[n//4],
        'p75': all_sizes[3*n//4],
        'p90': all_sizes[9*n//10],
        'mean': sum(all_sizes) / len(all_sizes)
    }

    # Sequence diversity statistics
    unique_sequences = len(set(seq_ids))
    sequence_stats = {
        'total_candidates': len(candidates),
        'unique_sequences': unique_sequences,
        'sequence_diversity_percent': unique_sequences / len(candidates) * 100,
        'avg_sequence_size': sum(seq_sizes) / len(seq_sizes),
        'min_sequence_size': min(seq_sizes),
        'max_sequence_size': max(seq_sizes)
    }

    metadata = {
        'generation_info': {
            'total_candidates': len(candidates),
            'generation_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'selector_type': 'SequenceAwareStratifiedSelector',
            'folder_name': folder_name,
            'description': 'Sequence-aware training dataset - one per camera trap sequence'
        },
        'category_distribution': category_counts,
        'size_distribution': size_stats,
        'sequence_stats': sequence_stats,
        'candidates': []
    }

    # Store ONLY essential candidate details to reduce file size
    print("Writing candidate data...")
    for i, candidate in enumerate(candidates):
        if (i + 1) % 1000 == 0:
            print(f"  Processed {i + 1}/{len(candidates)} candidates...")

        # Only store essential metadata, not the full analysis
        candidate_info = {
            'original_path': candidate.file_path,
            'flattened_filename': candidate.file_path.replace('/', '#').replace('\\', '#'),
            'category': candidate.category,
            'quality_score': candidate.quality_score,
            'seq_id': candidate.seq_id,
            'frame_num': candidate.frame_num,
            'seq_num_frames': candidate.seq_num_frames,
            # Essential metadata only
            'max_size': candidate.metadata['max_size'],
            'max_confidence': candidate.metadata['max_confidence'],
            'num_animals': candidate.metadata['num_animals'],
            'num_species': candidate.metadata['num_species']
        }
        metadata['candidates'].append(candidate_info)

    print("Writing metadata file...")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved optimized metadata to {metadata_path}")


class OptimizedSequenceAwareSelector(SequenceAwareStratifiedSelector):
    """Optimized version with progress tracking and efficient metadata lookup."""

    def _build_metadata_lookup(self, metadata_data: Dict[str, Any]) -> Dict[str, Dict]:
        """Build lookup from file_name to metadata - OPTIMIZED."""
        print("Building metadata lookup...")
        lookup = {}

        total_images = len(metadata_data['images'])
        for i, image_info in enumerate(metadata_data['images']):
            if (i + 1) % 100000 == 0:
                print(f"  Processed {i + 1}/{total_images} metadata entries...")

            file_name = image_info['file_name']
            # Store only what we need
            lookup[file_name] = {
                'seq_id': image_info['seq_id'],
                'frame_num': image_info['frame_num'],
                'seq_num_frames': image_info['seq_num_frames']
            }

        print(f"Metadata lookup complete: {len(lookup):,} entries")
        return lookup

    def _categorize_images(self, detection_data: Dict[str, Any], metadata_lookup: Dict[str, Dict]) -> Dict[str, List]:
        """Categorize images with progress tracking."""

        print("Analyzing species rarity...")
        species_counts = self._count_species(detection_data)
        total_detections = sum(species_counts.values())

        species_frequencies = {sp: count/total_detections for sp, count in species_counts.items()}
        rare_threshold = sorted(species_frequencies.values())[len(species_frequencies) // 5] if species_frequencies else 0
        rare_species = {sp for sp, freq in species_frequencies.items() if freq <= rare_threshold}

        print(f"Categorizing {len(detection_data['images']):,} images...")
        categorized = defaultdict(list)

        images_processed = 0
        images_with_metadata = 0
        images_with_animals = 0

        for i, image_data in enumerate(detection_data['images']):
            if (i + 1) % 50000 == 0:
                print(f"  Processed {i + 1}/{len(detection_data['images'])} images...")

            images_processed += 1
            file_path = image_data['file']

            # Get sequence information from metadata
            seq_info = metadata_lookup.get(file_path)
            if not seq_info:
                continue  # Skip images without sequence info

            images_with_metadata += 1

            animal_detections = [
                d for d in image_data.get('detections', [])
                if d['category'] == '1' and d['conf'] >= self.detection_conf_threshold
            ]

            if not animal_detections:
                continue

            images_with_animals += 1

            analysis = self._analyze_image(animal_detections, rare_species)
            categories = self._assign_categories(analysis)

            for category in categories:
                quality_score = self._calculate_quality_score(analysis, category)

                candidate = ImageCandidate(
                    file_path=file_path,
                    detections=animal_detections,
                    category=category,
                    quality_score=quality_score,
                    metadata=analysis,
                    seq_id=seq_info['seq_id'],
                    frame_num=seq_info['frame_num'],
                    seq_num_frames=seq_info['seq_num_frames']
                )

                categorized[category].append(candidate)

        print(f"Categorization complete:")
        print(f"  Images processed: {images_processed:,}")
        print(f"  Images with metadata: {images_with_metadata:,}")
        print(f"  Images with animals: {images_with_animals:,}")

        return categorized


# Import the ImageCandidate class
from stratified_selector_sequence_aware import ImageCandidate


def main():
    # Configuration
    detection_file = '/mnt/c/Users/dmorr/postprocessing/snapshot-safari/snapshot-safari-2025-09-19-v1000.0.0-redwood/combined_api_outputs/snapshot-safari-2025-09-19-v1000.0.0-redwood-ensemble_output_modular_image-level.md-format.within_image_smoothing.seqsmoothing.json'
    metadata_file = '/mnt/c/temp/hero-images/snapshot_safari_2024_metadata.ser.json'
    source_image_dir = '/mnt/g/snapshot_safari_2024_expansion/SER'
    base_candidates_dir = '/mnt/c/temp/hero-images/candidates'

    # Create timestamped subfolder
    output_dir = create_timestamped_folder(base_candidates_dir)
    folder_name = os.path.basename(output_dir)

    # Generate 5000 candidates
    total_candidates = 5000

    print("=== OPTIMIZED Sequence-Aware Hero Image Candidate Generation ===")
    print(f"Output directory: {output_dir}")
    print(f"Target candidates: {total_candidates:,}")

    # Load detection data (READ ONCE)
    print("\nStep 1: Loading detection data...")
    start_time = time.time()
    with open(detection_file, 'r') as f:
        detection_data = json.load(f)
    load_time = time.time() - start_time
    print(f"✓ Loaded {len(detection_data['images']):,} detection results in {load_time:.1f} seconds")

    # Load metadata (READ ONCE)
    print("\nStep 2: Loading sequence metadata...")
    start_time = time.time()
    with open(metadata_file, 'r') as f:
        metadata_data = json.load(f)
    metadata_load_time = time.time() - start_time
    print(f"✓ Loaded {len(metadata_data['images']):,} metadata entries in {metadata_load_time:.1f} seconds")

    # Run sequence-aware stratified selection
    print(f"\nStep 3: Running sequence-aware selection...")
    start_time = time.time()
    selector = OptimizedSequenceAwareSelector()
    candidates = selector.select_candidates(
        detection_data,
        metadata_data,
        total_candidates=total_candidates
    )
    selection_time = time.time() - start_time
    print(f"✓ Selection completed in {selection_time:.1f} seconds")

    # Quick analysis
    seq_ids = [c.seq_id for c in candidates]
    unique_sequences = len(set(seq_ids))
    seq_diversity = unique_sequences / len(candidates) * 100

    print(f"\nStep 4: Selection Summary:")
    print(f"✓ Total candidates: {len(candidates):,}")
    print(f"✓ Unique sequences: {unique_sequences:,}")
    print(f"✓ Sequence diversity: {seq_diversity:.1f}%")

    # Copy images
    print(f"\nStep 5: Copying images...")
    start_time = time.time()
    copied, failed = copy_candidates_to_folder(candidates, source_image_dir, output_dir)
    copy_time = time.time() - start_time
    print(f"✓ Copying completed in {copy_time:.1f} seconds")

    # Save optimized metadata
    print(f"\nStep 6: Saving metadata...")
    metadata_file_path = os.path.join(output_dir, 'candidates_metadata.json')
    start_time = time.time()
    save_candidate_metadata_optimized(candidates, metadata_file_path, folder_name)
    metadata_time = time.time() - start_time
    print(f"✓ Metadata saved in {metadata_time:.1f} seconds")

    # Final summary
    total_time = load_time + metadata_load_time + selection_time + copy_time + metadata_time
    print(f"\n=== SUCCESS: Sequence-Aware Generation Complete ===")
    print(f"Output folder: {folder_name}")
    print(f"Total time: {total_time:.1f} seconds")
    print(f"Candidates: {len(candidates):,} images from {unique_sequences:,} unique sequences")
    print(f"Sequence diversity: {seq_diversity:.1f}%")
    print(f"Copy success: {copied:,}/{len(candidates):,} images")

    print(f"\n🎉 Dataset ready for labeling!")


if __name__ == "__main__":
    main()