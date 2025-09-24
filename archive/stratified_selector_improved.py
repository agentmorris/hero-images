"""
Improved Stratified Candidate Selection System

Uses bell curve distribution for animal size scoring to favor optimal sizes
while still sampling across the full spectrum.
"""

import json
import random
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from collections import defaultdict
import math


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
    Selects candidates using stratified sampling with improved size scoring.
    """

    def __init__(self,
                 detection_conf_threshold: float = 0.3,
                 classification_conf_threshold: float = 0.5,
                 optimal_size_center: float = 0.25,  # 25% of image area is optimal
                 size_variance: float = 0.15):       # Standard deviation for bell curve
        self.detection_conf_threshold = detection_conf_threshold
        self.classification_conf_threshold = classification_conf_threshold
        self.optimal_size_center = optimal_size_center
        self.size_variance = size_variance

    def select_candidates(self, detection_data: Dict[str, Any],
                         total_candidates: int,
                         category_distribution: Optional[Dict[str, float]] = None) -> List[ImageCandidate]:
        """Select candidates using stratified sampling with improved size scoring."""

        if category_distribution is None:
            category_distribution = {
                'single_large': 0.25,
                'single_medium': 0.15,
                'multi_same_species': 0.20,
                'multi_different_species': 0.15,
                'rare_species': 0.15,
                'high_action': 0.10
            }

        # Categorize all eligible images
        print("Categorizing images with improved size scoring...")
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

            # Sort by quality score and take top candidates
            category_images.sort(key=lambda x: x.quality_score, reverse=True)
            selected_from_category = category_images[:target_count]

            print(f"  {category}: Selected {len(selected_from_category)} / {target_count} requested from {len(category_images)} available")
            selected_candidates.extend(selected_from_category)

        # Fill remaining slots if under target
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
        """
        Calculate size score using Gaussian distribution.
        Favors optimal size range but doesn't exclude extremes.
        """
        # Gaussian bell curve centered at optimal_size_center
        exponent = -((size - self.optimal_size_center) ** 2) / (2 * self.size_variance ** 2)
        gaussian_score = math.exp(exponent)

        # Add small baseline to ensure no size gets zero score
        baseline = 0.1
        return baseline + gaussian_score * (1.0 - baseline)

    def _categorize_images(self, detection_data: Dict[str, Any]) -> Dict[str, List[ImageCandidate]]:
        """Categorize images based on their characteristics with improved size scoring."""

        # Analyze species rarity
        species_counts = self._count_species(detection_data)
        total_detections = sum(species_counts.values())

        # Define rare species (bottom 20% by frequency)
        species_frequencies = {sp: count/total_detections for sp, count in species_counts.items()}
        rare_threshold = sorted(species_frequencies.values())[len(species_frequencies) // 5] if species_frequencies else 0
        rare_species = {sp for sp, freq in species_frequencies.items() if freq <= rare_threshold}

        categorized = defaultdict(list)

        for image_data in detection_data['images']:
            # Filter for high-confidence animal detections
            animal_detections = [
                d for d in image_data.get('detections', [])
                if d['category'] == '1' and d['conf'] >= self.detection_conf_threshold
            ]

            if not animal_detections:
                continue

            # Analyze image characteristics with improved size scoring
            analysis = self._analyze_image(animal_detections, rare_species)

            # Categorize based on analysis
            categories = self._assign_categories(analysis)

            # Create candidate for each applicable category
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
        """Analyze image characteristics with improved size scoring."""

        num_animals = len(detections)

        # Size analysis with bell curve scoring
        sizes = [d['bbox'][2] * d['bbox'][3] for d in detections]
        max_size = max(sizes)
        avg_size = sum(sizes) / len(sizes)

        # New: Calculate Gaussian size score for the largest animal
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
            'max_size_gaussian_score': max_size_gaussian_score,  # New field
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
        """Assign image to appropriate categories based on analysis."""

        categories = []

        num_animals = analysis['num_animals']
        max_size = analysis['max_size']
        num_species = analysis['num_species']
        has_rare_species = analysis['has_rare_species']
        max_confidence = analysis['max_confidence']
        centrality = analysis['centrality']

        # Single animal categories - use broader size thresholds
        if num_animals == 1:
            if max_size >= 0.05:  # Large animal (5%+ of image) - broadened from 15%
                categories.append('single_large')
            elif max_size >= 0.001:  # Medium animal (0.1%+ of image) - broadened from 2%
                categories.append('single_medium')

        # Multiple animal categories
        elif num_animals > 1:
            if num_species <= 1:
                categories.append('multi_same_species')
            else:
                categories.append('multi_different_species')

        # Rare species category
        if has_rare_species:
            categories.append('rare_species')

        # High action/quality category
        if (max_confidence >= 0.85 and centrality >= 0.7 and
            max_size >= 0.005):  # Lowered from 0.05
            categories.append('high_action')

        return categories if categories else ['single_medium']

    def _calculate_quality_score(self, analysis: Dict[str, Any], category: str) -> float:
        """Calculate quality score using Gaussian size scoring."""

        # Use the new Gaussian size score instead of log scaling
        size_score = analysis['max_size_gaussian_score']
        confidence_score = analysis['max_confidence']
        centrality_score = analysis['centrality']

        # Category-specific weighting (same as before)
        if category == 'single_large':
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


if __name__ == "__main__":
    # Test the improved stratified selector
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

    print(f"Testing improved stratified selection on {len(subset_data['images']):,} images")

    selector = StratifiedSelector()
    candidates = selector.select_candidates(subset_data, total_candidates=100)

    print(f"\nSelected {len(candidates)} candidates")

    # Analyze size distribution in results
    by_category = defaultdict(list)
    for candidate in candidates:
        by_category[candidate.category].append(candidate)

    print("\nSize distribution analysis:")
    for category, cands in by_category.items():
        sizes = [c.metadata['max_size'] for c in cands]
        if sizes:
            print(f"\n{category} ({len(cands)} candidates):")
            print(f"  Size range: {min(sizes):.4f} - {max(sizes):.4f}")
            print(f"  Average size: {sum(sizes)/len(sizes):.4f}")

            # Show top 3 with size info
            for i, cand in enumerate(cands[:3]):
                meta = cand.metadata
                print(f"  {i+1}. Score: {cand.quality_score:.3f}, Size: {meta['max_size']:.4f} | {cand.file_path}")