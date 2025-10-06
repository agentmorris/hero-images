"""
Sequence-Aware Stratified Candidate Selection for Training Dataset

Ensures only one image per camera trap sequence (burst) is selected to maximize diversity.
"""

import json
import random
import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from collections import defaultdict
import math
from datetime import datetime


@dataclass
class ImageCandidate:
    """Represents a candidate image with metadata for selection."""
    file_path: str
    detections: List[Dict[str, Any]]
    category: str
    quality_score: float = 0.0
    metadata: Dict[str, Any] = None
    seq_id: str = ""
    frame_num: int = 0
    seq_num_frames: int = 1

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class SequenceAwareStratifiedSelector:
    """
    Selects candidates ensuring only one image per camera trap sequence.
    """

    def __init__(self,
                 detection_conf_threshold: float = 0.3,
                 classification_conf_threshold: float = 0.5,
                 optimal_size_center: float = 0.25,
                 size_variance: float = 0.15):
        self.detection_conf_threshold = detection_conf_threshold
        self.classification_conf_threshold = classification_conf_threshold
        self.optimal_size_center = optimal_size_center
        self.size_variance = size_variance

    def select_candidates(self,
                         detection_data: Dict[str, Any],
                         metadata_data: Dict[str, Any],
                         total_candidates: int,
                         category_distribution: Optional[Dict[str, float]] = None) -> List[ImageCandidate]:
        """Select candidates ensuring only one per sequence."""

        if category_distribution is None:
            category_distribution = {
                'single_optimal': 0.20,
                'single_medium': 0.15,
                'single_tiny': 0.10,
                'single_huge': 0.05,
                'multi_same_species': 0.15,
                'multi_different_species': 0.15,
                'rare_species': 0.10,
                'high_action': 0.10
            }

        # Build metadata lookup
        metadata_lookup = self._build_metadata_lookup(metadata_data)

        # Categorize all eligible images
        print("Categorizing images with sequence awareness...")
        categorized_images = self._categorize_images(detection_data, metadata_lookup)

        # Group by sequence within each category
        print("Grouping by sequences within categories...")
        sequence_groups = self._group_by_sequences(categorized_images)

        # Print category and sequence statistics
        print("\nCategory and sequence statistics:")
        for category, seq_groups in sequence_groups.items():
            total_images = sum(len(candidates) for candidates in seq_groups.values())
            print(f"  {category}: {len(seq_groups):,} sequences, {total_images:,} total images")

        # Sample one image per sequence from each category
        print("\nSampling one image per sequence from categories...")
        selected_candidates = []

        for category, proportion in category_distribution.items():
            target_count = int(total_candidates * proportion)
            seq_groups = sequence_groups.get(category, {})

            if not seq_groups:
                print(f"  {category}: No sequences available")
                continue

            # Select best sequence candidates, one per sequence
            sequence_representatives = []
            for seq_id, candidates in seq_groups.items():
                # Sort candidates in this sequence by quality score
                candidates.sort(key=lambda x: x.quality_score, reverse=True)
                best_in_sequence = candidates[0]
                best_in_sequence.metadata['sequence_size'] = len(candidates)
                sequence_representatives.append(best_in_sequence)

            # Sort sequences by their best representative's score
            sequence_representatives.sort(key=lambda x: x.quality_score, reverse=True)

            # Take top sequences up to target count
            selected_from_category = sequence_representatives[:target_count]

            print(f"  {category}: Selected {len(selected_from_category)} sequences / {target_count} requested from {len(seq_groups)} available")
            selected_candidates.extend(selected_from_category)

        # Fill remaining slots if under target
        remaining_slots = total_candidates - len(selected_candidates)
        if remaining_slots > 0:
            print(f"\nFilling {remaining_slots} remaining slots from all categories...")

            selected_seq_ids = {c.seq_id for c in selected_candidates}
            all_remaining_sequences = []

            for category, seq_groups in sequence_groups.items():
                for seq_id, candidates in seq_groups.items():
                    if seq_id not in selected_seq_ids:
                        candidates.sort(key=lambda x: x.quality_score, reverse=True)
                        best_in_sequence = candidates[0]
                        best_in_sequence.metadata['sequence_size'] = len(candidates)
                        all_remaining_sequences.append(best_in_sequence)

            all_remaining_sequences.sort(key=lambda x: x.quality_score, reverse=True)
            selected_candidates.extend(all_remaining_sequences[:remaining_slots])

        print(f"\nTotal selected: {len(selected_candidates)} candidates from {len(set(c.seq_id for c in selected_candidates))} unique sequences")
        return selected_candidates

    def _build_metadata_lookup(self, metadata_data: Dict[str, Any]) -> Dict[str, Dict]:
        """Build lookup from file_name to metadata."""
        lookup = {}
        for image_info in metadata_data['images']:
            file_name = image_info['file_name']
            lookup[file_name] = {
                'seq_id': image_info['seq_id'],
                'frame_num': image_info['frame_num'],
                'seq_num_frames': image_info['seq_num_frames'],
                'location': image_info['location']
            }
        return lookup

    def _gaussian_size_score(self, size: float) -> float:
        """Calculate size score using Gaussian distribution."""
        exponent = -((size - self.optimal_size_center) ** 2) / (2 * self.size_variance ** 2)
        gaussian_score = math.exp(exponent)
        baseline = 0.3
        return baseline + gaussian_score * (1.0 - baseline)

    def _categorize_images(self, detection_data: Dict[str, Any], metadata_lookup: Dict[str, Dict]) -> Dict[str, List[ImageCandidate]]:
        """Categorize images including sequence information."""

        # Analyze species rarity
        species_counts = self._count_species(detection_data)
        total_detections = sum(species_counts.values())

        species_frequencies = {sp: count/total_detections for sp, count in species_counts.items()}
        rare_threshold = sorted(species_frequencies.values())[len(species_frequencies) // 5] if species_frequencies else 0
        rare_species = {sp for sp, freq in species_frequencies.items() if freq <= rare_threshold}

        categorized = defaultdict(list)

        for image_data in detection_data['images']:
            file_path = image_data['file']

            # Get sequence information from metadata
            seq_info = metadata_lookup.get(file_path, {})
            if not seq_info:
                continue  # Skip images without sequence info

            animal_detections = [
                d for d in image_data.get('detections', [])
                if d['category'] == '1' and d['conf'] >= self.detection_conf_threshold
            ]

            if not animal_detections:
                continue

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

        return categorized

    def _group_by_sequences(self, categorized_images: Dict[str, List[ImageCandidate]]) -> Dict[str, Dict[str, List[ImageCandidate]]]:
        """Group images by sequence within each category."""

        sequence_groups = {}

        for category, candidates in categorized_images.items():
            seq_groups = defaultdict(list)

            for candidate in candidates:
                seq_groups[candidate.seq_id].append(candidate)

            sequence_groups[category] = dict(seq_groups)

        return sequence_groups

    def _analyze_image(self, detections: List[Dict], rare_species: set) -> Dict[str, Any]:
        """Analyze image characteristics."""

        num_animals = len(detections)

        # Size analysis
        sizes = [d['bbox'][2] * d['bbox'][3] for d in detections]
        max_size = max(sizes)
        avg_size = sum(sizes) / len(sizes)
        max_size_gaussian_score = self._gaussian_size_score(max_size)

        # Confidence analysis
        confidences = [d['conf'] for d in detections]
        max_confidence = max(confidences)
        avg_confidence = sum(confidences) / len(confidences)

        # Centrality analysis
        largest_detection = detections[sizes.index(max_size)]
        bbox = largest_detection['bbox']
        center_x, center_y = bbox[0] + bbox[2]/2, bbox[1] + bbox[3]/2
        centrality = 1.0 - math.sqrt((center_x - 0.5)**2 + (center_y - 0.5)**2)

        # Species analysis
        species_in_image = set()
        has_rare_species = False

        for detection in detections:
            for classification in detection.get('classifications', []):
                species_id, conf = classification[0], classification[1]
                if conf >= self.classification_conf_threshold:
                    species_in_image.add(species_id)
                    if species_id in rare_species:
                        has_rare_species = True

        return {
            'num_animals': num_animals,
            'max_size': max_size,
            'avg_size': avg_size,
            'max_size_gaussian_score': max_size_gaussian_score,
            'max_confidence': max_confidence,
            'avg_confidence': avg_confidence,
            'centrality': centrality,
            'num_species': len(species_in_image),
            'species_list': list(species_in_image),
            'has_rare_species': has_rare_species,
            'sizes': sizes,
            'confidences': confidences
        }

    def _assign_categories(self, analysis: Dict[str, Any]) -> List[str]:
        """Assign categories including explicit size-based negative categories."""

        categories = []

        num_animals = analysis['num_animals']
        max_size = analysis['max_size']
        num_species = analysis['num_species']
        has_rare_species = analysis['has_rare_species']
        max_confidence = analysis['max_confidence']
        centrality = analysis['centrality']

        # Single animal categories with explicit size ranges
        if num_animals == 1:
            if max_size >= 0.80:
                categories.append('single_huge')
            elif max_size >= 0.10:
                categories.append('single_optimal')
            elif max_size >= 0.005:
                categories.append('single_medium')
            else:
                categories.append('single_tiny')

        # Multiple animal categories
        elif num_animals > 1:
            if num_species <= 1:
                categories.append('multi_same_species')
            else:
                categories.append('multi_different_species')

        # Rare species category (can overlap)
        if has_rare_species:
            categories.append('rare_species')

        # High action category
        if (max_confidence >= 0.85 and centrality >= 0.7 and
            0.05 <= max_size <= 0.50):
            categories.append('high_action')

        return categories if categories else ['single_medium']

    def _calculate_quality_score(self, analysis: Dict[str, Any], category: str) -> float:
        """Calculate quality score with reduced Gaussian dominance."""

        size_score = analysis['max_size_gaussian_score']
        confidence_score = analysis['max_confidence']
        centrality_score = analysis['centrality']

        # Category-specific weighting (same as before)
        if category in ['single_tiny', 'single_huge']:
            score = confidence_score * 0.5 + centrality_score * 0.3 + size_score * 0.2
        elif category == 'single_optimal':
            score = size_score * 0.5 + centrality_score * 0.3 + confidence_score * 0.2
        elif category == 'single_medium':
            score = size_score * 0.3 + centrality_score * 0.4 + confidence_score * 0.3
        elif category == 'multi_same_species':
            multi_bonus = min(analysis['num_animals'] / 5.0, 1.0)
            score = (size_score * 0.3 + confidence_score * 0.3 +
                    centrality_score * 0.2 + multi_bonus * 0.2)
        elif category == 'multi_different_species':
            species_bonus = min(analysis['num_species'] / 4.0, 1.0)
            score = (size_score * 0.25 + confidence_score * 0.25 +
                    centrality_score * 0.2 + species_bonus * 0.3)
        elif category == 'rare_species':
            score = confidence_score * 0.4 + size_score * 0.4 + centrality_score * 0.2
        elif category == 'high_action':
            score = (confidence_score * 0.3 + centrality_score * 0.3 +
                    size_score * 0.3 + min(analysis['avg_confidence'], 1.0) * 0.1)
        else:
            score = (size_score + confidence_score + centrality_score) / 3

        return score

    def _count_species(self, detection_data: Dict[str, Any]) -> Dict[str, int]:
        """Count species occurrences for rarity analysis."""
        species_counts = defaultdict(int)

        for image_data in detection_data['images']:
            for detection in image_data.get('detections', []):
                if detection['category'] == '1':
                    for classification in detection.get('classifications', []):
                        species_id, conf = classification[0], classification[1]
                        if conf >= self.classification_conf_threshold:
                            species_counts[species_id] += 1

        return dict(species_counts)


def create_timestamped_folder(base_path: str) -> str:
    """Create timestamped subfolder for this candidate generation run."""
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    folder_name = f"heuristics-{timestamp}"
    folder_path = os.path.join(base_path, folder_name)
    os.makedirs(folder_path, exist_ok=True)
    return folder_path


if __name__ == "__main__":
    # Test with small subset first
    detection_file = '/mnt/c/Users/dmorr/postprocessing/snapshot-safari/snapshot-safari-2025-09-19-v1000.0.0-redwood/combined_api_outputs/snapshot-safari-2025-09-19-v1000.0.0-redwood-ensemble_output_modular_image-level.md-format.within_image_smoothing.seqsmoothing.json'
    metadata_file = '/mnt/c/temp/hero-images/snapshot_safari_2024_metadata.ser.json'

    print("Loading detection data...")
    with open(detection_file, 'r') as f:
        data = json.load(f)

    print("Loading metadata...")
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)

    # Test on subset
    subset_data = {
        'info': data['info'],
        'detection_categories': data['detection_categories'],
        'images': data['images'][:50000]
    }

    print(f"Testing sequence-aware selection on {len(subset_data['images']):,} images")

    selector = SequenceAwareStratifiedSelector()
    candidates = selector.select_candidates(subset_data, metadata, total_candidates=100)

    # Analyze sequence diversity
    seq_ids = [c.seq_id for c in candidates]
    unique_sequences = len(set(seq_ids))

    print(f"\nSequence diversity check:")
    print(f"  Total candidates: {len(candidates)}")
    print(f"  Unique sequences: {unique_sequences}")
    print(f"  Sequence diversity: {unique_sequences/len(candidates)*100:.1f}%")

    if unique_sequences < len(candidates):
        print("  WARNING: Some sequences have multiple images selected!")

    # Show sequence size distribution
    seq_sizes = [c.metadata.get('sequence_size', 1) for c in candidates]
    print(f"\nSequence size distribution:")
    print(f"  Min sequence size: {min(seq_sizes)}")
    print(f"  Max sequence size: {max(seq_sizes)}")
    print(f"  Average sequence size: {sum(seq_sizes)/len(seq_sizes):.1f}")

    # Show some examples
    print(f"\nExample candidates:")
    for i, candidate in enumerate(candidates[:5]):
        meta = candidate.metadata
        print(f"  {i+1}. {candidate.file_path}")
        print(f"     Seq: {candidate.seq_id}, Frame: {candidate.frame_num}/{candidate.seq_num_frames}")
        print(f"     Category: {candidate.category}, Score: {candidate.quality_score:.3f}")
        print(f"     Size: {meta['max_size']:.4f}, Conf: {meta['max_confidence']:.3f}")