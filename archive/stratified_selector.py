"""
Stratified Candidate Selection System

Instead of ranking all images by a single score, this approach categorizes images
into different aesthetic categories and samples from each to ensure diversity.
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
    Selects candidates using stratified sampling across different image characteristics.

    Categories:
    - single_large: Single animal taking up significant portion of image
    - single_medium: Single animal, medium size
    - multi_same_species: Multiple animals, same species
    - multi_different_species: Multiple animals, different species
    - rare_species: Images with rare/interesting species
    - high_action: High confidence, well-positioned animals (potential action/behavior)
    """

    def __init__(self,
                 detection_conf_threshold: float = 0.3,
                 classification_conf_threshold: float = 0.5):
        self.detection_conf_threshold = detection_conf_threshold
        self.classification_conf_threshold = classification_conf_threshold

    def select_candidates(self, detection_data: Dict[str, Any],
                         total_candidates: int,
                         category_distribution: Optional[Dict[str, float]] = None) -> List[ImageCandidate]:
        """
        Select candidates using stratified sampling.

        Args:
            detection_data: MegaDetector output data
            total_candidates: Total number of candidates to select
            category_distribution: Dict of category -> proportion (0-1). If None, uses default.
        """

        if category_distribution is None:
            category_distribution = {
                'single_large': 0.25,      # 25% - single large animals
                'single_medium': 0.15,     # 15% - single medium animals
                'multi_same_species': 0.20, # 20% - multiple same species
                'multi_different_species': 0.15, # 15% - multiple different species
                'rare_species': 0.15,      # 15% - rare or interesting species
                'high_action': 0.10        # 10% - potential action/behavior shots
            }

        # First pass: categorize all eligible images
        print("Categorizing images...")
        categorized_images = self._categorize_images(detection_data)

        # Print category statistics
        print("\nCategory statistics:")
        for category, images in categorized_images.items():
            print(f"  {category}: {len(images):,} images")

        # Second pass: sample from each category
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

        # Fill remaining slots if we're under target
        remaining_slots = total_candidates - len(selected_candidates)
        if remaining_slots > 0:
            print(f"\nFilling {remaining_slots} remaining slots from all categories...")

            # Collect all non-selected high-quality images
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

    def _categorize_images(self, detection_data: Dict[str, Any]) -> Dict[str, List[ImageCandidate]]:
        """Categorize images based on their characteristics."""

        # First, analyze species rarity
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

            # Analyze image characteristics
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
        """Analyze image characteristics for categorization."""

        num_animals = len(detections)

        # Size analysis
        sizes = [d['bbox'][2] * d['bbox'][3] for d in detections]
        max_size = max(sizes)
        avg_size = sum(sizes) / len(sizes)

        # Confidence analysis
        confidences = [d['conf'] for d in detections]
        max_confidence = max(confidences)
        avg_confidence = sum(confidences) / len(confidences)

        # Centrality analysis (how close to center is the largest animal)
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

        # Single animal categories
        if num_animals == 1:
            if max_size >= 0.15:  # Large animal (15%+ of image)
                categories.append('single_large')
            elif max_size >= 0.02:  # Medium animal (2%+ of image)
                categories.append('single_medium')

        # Multiple animal categories
        elif num_animals > 1:
            if num_species <= 1:  # Same species (or unclassified)
                categories.append('multi_same_species')
            else:  # Different species
                categories.append('multi_different_species')

        # Rare species category (can overlap with others)
        if has_rare_species:
            categories.append('rare_species')

        # High action/quality category (well-positioned, high confidence)
        if (max_confidence >= 0.85 and centrality >= 0.7 and
            max_size >= 0.05):  # High conf, centered, reasonably sized
            categories.append('high_action')

        return categories if categories else ['single_medium']  # Default fallback

    def _calculate_quality_score(self, analysis: Dict[str, Any], category: str) -> float:
        """Calculate quality score for an image within its category."""

        # Base components
        size_score = min(math.log10(analysis['max_size'] + 0.0001) + 4, 10) / 10  # Normalize to 0-1
        confidence_score = analysis['max_confidence']
        centrality_score = analysis['centrality']

        # Category-specific weighting
        if category == 'single_large':
            # For single large animals, prioritize size and centrality
            score = size_score * 0.5 + centrality_score * 0.3 + confidence_score * 0.2

        elif category == 'single_medium':
            # For single medium animals, balance size, centrality, and confidence
            score = size_score * 0.3 + centrality_score * 0.4 + confidence_score * 0.3

        elif category == 'multi_same_species':
            # For multiple same species, consider animal count and average quality
            multi_bonus = min(analysis['num_animals'] / 5.0, 1.0)  # Bonus up to 5 animals
            score = (size_score * 0.3 + confidence_score * 0.3 +
                    centrality_score * 0.2 + multi_bonus * 0.2)

        elif category == 'multi_different_species':
            # For multiple species, prioritize species diversity
            species_bonus = min(analysis['num_species'] / 4.0, 1.0)  # Bonus up to 4 species
            score = (size_score * 0.25 + confidence_score * 0.25 +
                    centrality_score * 0.2 + species_bonus * 0.3)

        elif category == 'rare_species':
            # For rare species, prioritize confidence and size
            score = confidence_score * 0.4 + size_score * 0.4 + centrality_score * 0.2

        elif category == 'high_action':
            # For high action, balance all factors
            score = (confidence_score * 0.3 + centrality_score * 0.3 +
                    size_score * 0.3 + min(analysis['avg_confidence'], 1.0) * 0.1)

        else:
            # Default scoring
            score = (size_score + confidence_score + centrality_score) / 3

        return score

    def _count_species(self, detection_data: Dict[str, Any]) -> Dict[str, int]:
        """Count species occurrences for rarity analysis."""
        species_counts = defaultdict(int)

        for image_data in detection_data['images']:
            for detection in image_data.get('detections', []):
                if detection['category'] == '1':  # Animal
                    for classification in detection.get('classifications', []):
                        species_id, conf = classification[0], classification[1]
                        if conf >= self.classification_conf_threshold:
                            species_counts[species_id] += 1

        return dict(species_counts)


if __name__ == "__main__":
    # Test the stratified selector
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

    print(f"Testing stratified selection on {len(subset_data['images']):,} images")

    selector = StratifiedSelector()
    candidates = selector.select_candidates(subset_data, total_candidates=100)

    print(f"\nSelected {len(candidates)} candidates")

    # Analyze results by category
    by_category = defaultdict(list)
    for candidate in candidates:
        by_category[candidate.category].append(candidate)

    print("\nResults by category:")
    for category, cands in by_category.items():
        print(f"\n{category} ({len(cands)} candidates):")
        for i, cand in enumerate(cands[:3]):  # Show top 3 in each category
            meta = cand.metadata
            print(f"  {i+1}. Score: {cand.quality_score:.3f} | {cand.file_path}")
            print(f"     Animals: {meta['num_animals']}, Size: {meta['max_size']:.4f}, "
                  f"Species: {meta['num_species']}, Conf: {meta['max_confidence']:.3f}")