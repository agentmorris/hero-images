"""
Hero Image Candidate Selection System

This module provides a flexible system for selecting candidate images
for aesthetic rating/labeling from camera trap datasets.
"""

import json
import os
import random
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import math


@dataclass
class ImageCandidate:
    """Represents a candidate image with metadata for selection."""
    file_path: str
    detections: List[Dict[str, Any]]
    score: float = 0.0
    selection_criteria: Dict[str, Any] = None

    def __post_init__(self):
        if self.selection_criteria is None:
            self.selection_criteria = {}


class CandidateSelector(ABC):
    """Abstract base class for image candidate selection methods."""

    @abstractmethod
    def select_candidates(self, detection_data: Dict[str, Any],
                         num_candidates: int) -> List[ImageCandidate]:
        """Select candidate images from detection data."""
        pass


class HeuristicSelector(CandidateSelector):
    """Heuristic-based candidate selection using camera trap detection data."""

    def __init__(self,
                 detection_conf_threshold: float = 0.3,
                 classification_conf_threshold: float = 0.5,
                 image_base_dir: str = "/mnt/g/snapshot_safari_2024_expansion/SER"):
        self.detection_conf_threshold = detection_conf_threshold
        self.classification_conf_threshold = classification_conf_threshold
        self.image_base_dir = image_base_dir

    def select_candidates(self, detection_data: Dict[str, Any],
                         num_candidates: int) -> List[ImageCandidate]:
        """
        Select candidates using heuristic scoring based on:
        - Animal size (larger = better)
        - Image centrality (more centered = better)
        - Multiple animals (more = better, up to a point)
        - Multiple species (bonus)
        - Rare species (bonus)
        - Detection confidence (higher = better)
        """

        candidates = []

        # First, identify species rarity from the full dataset
        species_counts = self._count_species(detection_data)
        total_species_detections = sum(species_counts.values())

        for image_data in detection_data['images']:
            # Filter for images with high-confidence animal detections
            animal_detections = [
                d for d in image_data.get('detections', [])
                if d['category'] == '1' and d['conf'] >= self.detection_conf_threshold
            ]

            if not animal_detections:
                continue

            # Calculate heuristic score
            score = self._calculate_heuristic_score(
                animal_detections, species_counts, total_species_detections
            )

            candidate = ImageCandidate(
                file_path=image_data['file'],
                detections=animal_detections,
                score=score,
                selection_criteria=self._get_selection_criteria(animal_detections)
            )

            candidates.append(candidate)

        # Sort by score (descending) and return top candidates
        candidates.sort(key=lambda x: x.score, reverse=True)
        return candidates[:num_candidates]

    def _count_species(self, detection_data: Dict[str, Any]) -> Dict[str, int]:
        """Count occurrences of each species for rarity assessment."""
        species_counts = {}

        for image_data in detection_data['images']:
            for detection in image_data.get('detections', []):
                if detection['category'] == '1':  # Animal
                    for classification in detection.get('classifications', []):
                        species_id, conf = classification[0], classification[1]
                        if conf >= self.classification_conf_threshold:
                            species_counts[species_id] = species_counts.get(species_id, 0) + 1

        return species_counts

    def _calculate_heuristic_score(self, detections: List[Dict],
                                 species_counts: Dict[str, int],
                                 total_species_detections: int) -> float:
        """Calculate composite heuristic score for an image."""

        if not detections:
            return 0.0

        # Base score components
        size_score = 0.0
        centrality_score = 0.0
        confidence_score = 0.0
        multi_animal_bonus = 0.0
        multi_species_bonus = 0.0
        rarity_bonus = 0.0

        # Track unique species in this image
        image_species = set()

        for detection in detections:
            bbox = detection['bbox']
            x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]

            # Size score: larger animals are better (log scale to avoid extreme dominance)
            animal_size = w * h
            size_score = max(size_score, math.log10(animal_size + 0.0001) + 4)  # Shift to positive

            # Centrality score: animals closer to center are better
            center_x, center_y = x + w/2, y + h/2
            distance_from_center = math.sqrt((center_x - 0.5)**2 + (center_y - 0.5)**2)
            centrality_score = max(centrality_score, 1.0 - distance_from_center)

            # Confidence score
            confidence_score = max(confidence_score, detection['conf'])

            # Species analysis
            for classification in detection.get('classifications', []):
                species_id, conf = classification[0], classification[1]
                if conf >= self.classification_conf_threshold:
                    image_species.add(species_id)

                    # Rarity bonus: rarer species get higher scores
                    if species_id in species_counts and total_species_detections > 0:
                        species_frequency = species_counts[species_id] / total_species_detections
                        rarity_bonus = max(rarity_bonus, 1.0 - species_frequency)

        # Multi-animal bonus (diminishing returns)
        num_animals = len(detections)
        if num_animals > 1:
            multi_animal_bonus = min(math.log2(num_animals), 2.0)  # Cap at 4 animals

        # Multi-species bonus
        if len(image_species) > 1:
            multi_species_bonus = min(len(image_species) * 0.5, 2.0)  # Cap bonus

        # Weighted composite score
        total_score = (
            size_score * 3.0 +           # Size is very important
            centrality_score * 2.0 +     # Centrality matters
            confidence_score * 1.0 +     # Confidence matters
            multi_animal_bonus * 1.5 +   # Multiple animals bonus
            multi_species_bonus * 2.0 +  # Multiple species rare and valuable
            rarity_bonus * 1.0           # Rare species bonus
        )

        return total_score

    def _get_selection_criteria(self, detections: List[Dict]) -> Dict[str, Any]:
        """Extract selection criteria metadata for analysis."""
        if not detections:
            return {}

        # Calculate summary statistics
        sizes = [d['bbox'][2] * d['bbox'][3] for d in detections]
        confidences = [d['conf'] for d in detections]

        # Get species information
        species_list = []
        for detection in detections:
            for classification in detection.get('classifications', []):
                species_id, conf = classification[0], classification[1]
                if conf >= self.classification_conf_threshold:
                    species_list.append(species_id)

        return {
            'num_animals': len(detections),
            'max_animal_size': max(sizes),
            'avg_animal_size': sum(sizes) / len(sizes),
            'max_confidence': max(confidences),
            'avg_confidence': sum(confidences) / len(confidences),
            'species_count': len(set(species_list)),
            'species_list': list(set(species_list))
        }


class RandomSelector(CandidateSelector):
    """Random baseline selector for comparison."""

    def __init__(self, detection_conf_threshold: float = 0.3):
        self.detection_conf_threshold = detection_conf_threshold

    def select_candidates(self, detection_data: Dict[str, Any],
                         num_candidates: int) -> List[ImageCandidate]:
        """Randomly select from images with animal detections."""

        candidates = []

        for image_data in detection_data['images']:
            animal_detections = [
                d for d in image_data.get('detections', [])
                if d['category'] == '1' and d['conf'] >= self.detection_conf_threshold
            ]

            if animal_detections:
                candidate = ImageCandidate(
                    file_path=image_data['file'],
                    detections=animal_detections,
                    score=random.random()
                )
                candidates.append(candidate)

        # Random selection
        random.shuffle(candidates)
        return candidates[:num_candidates]


class HybridSelector(CandidateSelector):
    """Combines multiple selection methods."""

    def __init__(self, selectors: List[Tuple[CandidateSelector, float]]):
        """
        Initialize with list of (selector, weight) tuples.
        Weights determine how many candidates each method contributes.
        """
        self.selectors = selectors

    def select_candidates(self, detection_data: Dict[str, Any],
                         num_candidates: int) -> List[ImageCandidate]:
        """Combine results from multiple selectors."""

        all_candidates = []
        total_weight = sum(weight for _, weight in self.selectors)

        for selector, weight in self.selectors:
            num_from_this_selector = int((weight / total_weight) * num_candidates)
            candidates = selector.select_candidates(detection_data, num_from_this_selector)

            # Add source information
            for candidate in candidates:
                candidate.selection_criteria['source_selector'] = selector.__class__.__name__

            all_candidates.extend(candidates)

        # Remove duplicates and return
        seen_paths = set()
        unique_candidates = []

        for candidate in all_candidates:
            if candidate.file_path not in seen_paths:
                seen_paths.add(candidate.file_path)
                unique_candidates.append(candidate)

        return unique_candidates[:num_candidates]


def load_detection_data(json_path: str) -> Dict[str, Any]:
    """Load detection data from JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)


def save_candidates(candidates: List[ImageCandidate], output_path: str):
    """Save candidate list to JSON file."""
    candidate_data = []
    for candidate in candidates:
        candidate_data.append({
            'file_path': candidate.file_path,
            'score': candidate.score,
            'selection_criteria': candidate.selection_criteria,
            'num_detections': len(candidate.detections)
        })

    with open(output_path, 'w') as f:
        json.dump(candidate_data, f, indent=2)


if __name__ == "__main__":
    # Example usage
    detection_file = '/mnt/c/Users/dmorr/postprocessing/snapshot-safari/snapshot-safari-2025-09-19-v1000.0.0-redwood/combined_api_outputs/snapshot-safari-2025-09-19-v1000.0.0-redwood-ensemble_output_modular_image-level.md-format.within_image_smoothing.seqsmoothing.json'

    print("Loading detection data...")
    data = load_detection_data(detection_file)

    print("Selecting candidates...")
    selector = HeuristicSelector()
    candidates = selector.select_candidates(data, num_candidates=100)

    print(f"Selected {len(candidates)} candidates")
    for i, candidate in enumerate(candidates[:5]):
        print(f"{i+1}: {candidate.file_path} (score: {candidate.score:.2f})")
        print(f"   Criteria: {candidate.selection_criteria}")