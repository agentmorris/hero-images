"""
Stratified Candidate Selection for Training Dataset

Samples across full spectrum including negative examples for robust training.
Over-represents promising images but ensures coverage of likely poor images.
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

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class StratifiedSelector:
    """
    Selects candidates for training dataset including negative examples.
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

    def select_candidates(self, detection_data: Dict[str, Any],
                         total_candidates: int,
                         category_distribution: Optional[Dict[str, float]] = None) -> List[ImageCandidate]:
        """Select candidates ensuring coverage of full quality spectrum."""

        if category_distribution is None:
            # Adjusted distribution to include more negative examples
            category_distribution = {
                'single_optimal': 0.20,      # 20% - well-sized single animals (likely positive)
                'single_medium': 0.15,       # 15% - medium single animals
                'single_tiny': 0.10,         # 10% - very small animals (likely negative)
                'single_huge': 0.05,         # 5% - very large animals (likely negative)
                'multi_same_species': 0.15,  # 15% - multiple same species
                'multi_different_species': 0.15, # 15% - multiple different species
                'rare_species': 0.10,        # 10% - rare species (mix of good/bad)
                'high_action': 0.10          # 10% - potential action shots
            }

        # Categorize all eligible images
        print("Categorizing images for training dataset (including negatives)...")
        categorized_images = self._categorize_images(detection_data)

        # Print category statistics
        print("\nCategory statistics:")
        for category, images in categorized_images.items():
            print(f"  {category}: {len(images):,} images")

        # Sample from each category
        print("\nSampling from categories...")
        selected_candidates = []

        for category, proportion in category_distribution.items():
            target_count = int(total_candidates * proportion)
            category_images = categorized_images.get(category, [])

            if not category_images:
                print(f"  {category}: No images available")
                continue

            # For negative categories, use more random sampling
            if category in ['single_tiny', 'single_huge']:
                # Mix top quality with random sampling for negative examples
                category_images.sort(key=lambda x: x.quality_score, reverse=True)
                top_half = category_images[:len(category_images)//2]
                bottom_half = category_images[len(category_images)//2:]
                random.shuffle(bottom_half)

                # Take half from top quality, half random
                num_top = target_count // 2
                num_random = target_count - num_top
                selected_from_category = top_half[:num_top] + bottom_half[:num_random]
            else:
                # Standard quality-based selection for other categories
                category_images.sort(key=lambda x: x.quality_score, reverse=True)
                selected_from_category = category_images[:target_count]

            print(f"  {category}: Selected {len(selected_from_category)} / {target_count} requested from {len(category_images)} available")
            selected_candidates.extend(selected_from_category)

        # Fill remaining slots
        remaining_slots = total_candidates - len(selected_candidates)
        if remaining_slots > 0:
            print(f"\nFilling {remaining_slots} remaining slots from all categories...")

            selected_paths = {c.file_path for c in selected_candidates}
            all_remaining = []

            for category_images in categorized_images.values():
                for img in category_images:
                    if img.file_path not in selected_paths:
                        all_remaining.append(img)

            all_remaining.sort(key=lambda x: x.quality_score, reverse=True)
            selected_candidates.extend(all_remaining[:remaining_slots])

        print(f"\nTotal selected: {len(selected_candidates)} candidates")
        return selected_candidates

    def _gaussian_size_score(self, size: float) -> float:
        """Calculate size score using Gaussian distribution, but less dominant."""
        exponent = -((size - self.optimal_size_center) ** 2) / (2 * self.size_variance ** 2)
        gaussian_score = math.exp(exponent)

        # Larger baseline to reduce Gaussian dominance
        baseline = 0.3  # Increased from 0.1
        return baseline + gaussian_score * (1.0 - baseline)

    def _categorize_images(self, detection_data: Dict[str, Any]) -> Dict[str, List[ImageCandidate]]:
        """Categorize images including explicit size-based negative categories."""

        # Analyze species rarity
        species_counts = self._count_species(detection_data)
        total_detections = sum(species_counts.values())

        species_frequencies = {sp: count/total_detections for sp, count in species_counts.items()}
        rare_threshold = sorted(species_frequencies.values())[len(species_frequencies) // 5] if species_frequencies else 0
        rare_species = {sp for sp, freq in species_frequencies.items() if freq <= rare_threshold}

        categorized = defaultdict(list)

        for image_data in detection_data['images']:
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
                    file_path=image_data['file'],
                    detections=animal_detections,
                    category=category,
                    quality_score=quality_score,
                    metadata=analysis
                )

                categorized[category].append(candidate)

        return categorized

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
            if max_size >= 0.80:  # Very large animals (likely negative - too close)
                categories.append('single_huge')
            elif max_size >= 0.10:  # Well-sized animals (likely positive)
                categories.append('single_optimal')
            elif max_size >= 0.005:  # Medium animals
                categories.append('single_medium')
            else:  # Very small animals (likely negative - too distant)
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
            0.05 <= max_size <= 0.50):  # Size constraint for action
            categories.append('high_action')

        return categories if categories else ['single_medium']

    def _calculate_quality_score(self, analysis: Dict[str, Any], category: str) -> float:
        """Calculate quality score with reduced Gaussian dominance."""

        size_score = analysis['max_size_gaussian_score']
        confidence_score = analysis['max_confidence']
        centrality_score = analysis['centrality']

        # For negative categories, reduce size score influence
        if category in ['single_tiny', 'single_huge']:
            # For negatives, prioritize confidence and centrality over size
            score = confidence_score * 0.5 + centrality_score * 0.3 + size_score * 0.2

        elif category == 'single_optimal':
            # For optimal, size matters more
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

    print("Loading detection data...")
    with open(detection_file, 'r') as f:
        data = json.load(f)

    # Test on subset
    subset_data = {
        'info': data['info'],
        'detection_categories': data['detection_categories'],
        'images': data['images'][:50000]
    }

    print(f"Testing training-oriented selection on {len(subset_data['images']):,} images")

    selector = StratifiedSelector()
    candidates = selector.select_candidates(subset_data, total_candidates=100)

    # Analyze size distribution including extremes
    by_category = defaultdict(list)
    for candidate in candidates:
        by_category[candidate.category].append(candidate)

    print("\nSize distribution analysis (including negatives):")
    all_sizes = []
    for category, cands in by_category.items():
        sizes = [c.metadata['max_size'] for c in cands]
        all_sizes.extend(sizes)
        if sizes:
            print(f"\n{category} ({len(cands)} candidates):")
            print(f"  Size range: {min(sizes):.4f} - {max(sizes):.4f}")
            print(f"  Average size: {sum(sizes)/len(sizes):.4f}")

    if all_sizes:
        all_sizes.sort()
        print(f"\nOverall size distribution:")
        print(f"  Full range: {min(all_sizes):.4f} - {max(all_sizes):.4f}")
        print(f"  10th percentile: {all_sizes[len(all_sizes)//10]:.4f}")
        print(f"  50th percentile: {all_sizes[len(all_sizes)//2]:.4f}")
        print(f"  90th percentile: {all_sizes[9*len(all_sizes)//10]:.4f}")