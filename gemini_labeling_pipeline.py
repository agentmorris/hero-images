"""
Gemini 2.5 Flash Batch Labeling Pipeline for Hero Image Aesthetic Rating

Processes camera trap images through Gemini 2.5 Flash to generate 0-10 aesthetic ratings
with batch mode for cost optimization.
"""

import json
import os
import base64
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from PIL import Image
import io
import google.generativeai as genai
from datetime import datetime


@dataclass
class LabelingResult:
    """Result of labeling a single image."""
    image_path: str
    aesthetic_score: float
    reasoning: str
    processing_time: float
    success: bool = True
    error_message: str = ""


class ImageProcessor:
    """Handles image loading and resizing."""

    @staticmethod
    def resize_image_to_768_long_side(image_path: str) -> bytes:
        """
        Resize image to 768px on long side while preserving aspect ratio.
        Returns JPEG bytes for API submission.
        """
        try:
            with Image.open(image_path) as img:
                # Convert to RGB if necessary
                if img.mode != 'RGB':
                    img = img.convert('RGB')

                # Get current dimensions
                width, height = img.size

                # Calculate new dimensions (768px on long side)
                if width > height:
                    new_width = 768
                    new_height = int((height * 768) / width)
                else:
                    new_height = 768
                    new_width = int((width * 768) / height)

                # Resize with high-quality resampling
                resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

                # Convert to bytes
                img_byte_arr = io.BytesIO()
                resized_img.save(img_byte_arr, format='JPEG', quality=85, optimize=True)
                return img_byte_arr.getvalue()

        except Exception as e:
            raise Exception(f"Failed to process image {image_path}: {str(e)}")


class GeminiLabeler:
    """Handles Gemini API interactions for image labeling."""

    def __init__(self, api_key: str, model_name: str = "gemini-2.5-flash"):
        """
        Initialize Gemini labeler.

        Args:
            api_key: Google AI API key
            model_name: Gemini model to use
        """
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        self.model_name = model_name

        # Create the aesthetic rating prompt
        self.prompt = self._create_aesthetic_prompt()

    def _create_aesthetic_prompt(self) -> str:
        """Create the prompt for aesthetic rating."""
        return """You are an expert wildlife photography curator evaluating camera trap images for aesthetic appeal.

Rate this image on a scale of 0-10 for its potential as a "hero image" - meaning how visually compelling and aesthetically pleasing it would be to a general audience.

Consider these factors:
- Animal positioning and framing (centered, well-composed vs. edge-cropped or partially obscured)
- Image clarity and focus (sharp vs. blurry)
- Lighting quality (well-lit vs. too dark/bright/harsh shadows)
- Animal behavior and pose (interesting/natural vs. awkward/unnatural)
- Overall visual appeal (would this catch someone's attention?)

Rating scale:
- 0-2: Poor (blurry, poorly framed, or not visually appealing)
- 3-4: Below average (some issues with composition, lighting, or clarity)
- 5-6: Average (acceptable but unremarkable)
- 7-8: Good (well-composed, clear, visually appealing)
- 9-10: Excellent (exceptional composition, lighting, and visual impact)

Respond with ONLY a JSON object in this exact format:
{
  "score": [number from 0-10],
  "reasoning": "[2-3 sentence explanation of the score]"
}

Do not include any text before or after the JSON."""

    def label_image(self, image_bytes: bytes, image_path: str) -> LabelingResult:
        """
        Label a single image using Gemini.

        Args:
            image_bytes: Resized image as bytes
            image_path: Original image path for reference

        Returns:
            LabelingResult with score and reasoning
        """
        start_time = time.time()

        try:
            # Prepare the image for the API
            image_data = {
                'mime_type': 'image/jpeg',
                'data': image_bytes
            }

            # Generate response
            response = self.model.generate_content([
                image_data,
                self.prompt
            ])

            processing_time = time.time() - start_time

            # Parse the JSON response (handle markdown code blocks)
            try:
                response_text = response.text.strip()

                # Remove markdown code blocks if present
                if response_text.startswith('```json'):
                    response_text = response_text[7:]  # Remove ```json
                if response_text.endswith('```'):
                    response_text = response_text[:-3]  # Remove ```

                response_text = response_text.strip()

                result_json = json.loads(response_text)
                score = float(result_json['score'])
                reasoning = result_json['reasoning']

                # Validate score range
                if not (0 <= score <= 10):
                    raise ValueError(f"Score {score} outside valid range 0-10")

                return LabelingResult(
                    image_path=image_path,
                    aesthetic_score=score,
                    reasoning=reasoning,
                    processing_time=processing_time,
                    success=True
                )

            except (json.JSONDecodeError, KeyError, ValueError) as e:
                return LabelingResult(
                    image_path=image_path,
                    aesthetic_score=0.0,
                    reasoning="",
                    processing_time=processing_time,
                    success=False,
                    error_message=f"Failed to parse response: {str(e)}. Response: {response.text[:200]}"
                )

        except Exception as e:
            processing_time = time.time() - start_time
            return LabelingResult(
                image_path=image_path,
                aesthetic_score=0.0,
                reasoning="",
                processing_time=processing_time,
                success=False,
                error_message=f"API error: {str(e)}"
            )

    def label_batch(self, image_paths: List[str], batch_size: int = 10) -> List[LabelingResult]:
        """
        Label a batch of images with rate limiting.

        Args:
            image_paths: List of image file paths
            batch_size: Number of images to process before brief pause

        Returns:
            List of LabelingResult objects
        """
        results = []

        for i, image_path in enumerate(image_paths):
            print(f"Processing image {i+1}/{len(image_paths)}: {os.path.basename(image_path)}")

            try:
                # Load and resize image
                image_bytes = ImageProcessor.resize_image_to_768_long_side(image_path)

                # Get Gemini rating
                result = self.label_image(image_bytes, image_path)
                results.append(result)

                if result.success:
                    print(f"  ✓ Score: {result.aesthetic_score:.1f} ({result.processing_time:.1f}s)")
                else:
                    print(f"  ✗ Failed: {result.error_message}")

                # Rate limiting: brief pause every batch_size images
                if (i + 1) % batch_size == 0 and i < len(image_paths) - 1:
                    print(f"  Pausing briefly after {batch_size} images...")
                    time.sleep(2)

            except Exception as e:
                result = LabelingResult(
                    image_path=image_path,
                    aesthetic_score=0.0,
                    reasoning="",
                    processing_time=0.0,
                    success=False,
                    error_message=f"Processing error: {str(e)}"
                )
                results.append(result)
                print(f"  ✗ Error: {str(e)}")

        return results


class LabelingPipeline:
    """Main pipeline for labeling hero image candidates."""

    def __init__(self, api_key: str, candidates_dir: str, output_dir: str):
        """
        Initialize labeling pipeline.

        Args:
            api_key: Google AI API key
            candidates_dir: Directory containing candidate images
            output_dir: Directory to save labeling results
        """
        self.labeler = GeminiLabeler(api_key)
        self.candidates_dir = candidates_dir
        self.output_dir = output_dir

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

    def get_candidate_images(self) -> List[str]:
        """Get list of candidate image paths."""
        image_files = []
        for filename in os.listdir(self.candidates_dir):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                full_path = os.path.join(self.candidates_dir, filename)
                image_files.append(full_path)

        return sorted(image_files)

    def save_results(self, results: List[LabelingResult], output_filename: str):
        """Save labeling results to JSON file."""

        # Calculate statistics
        successful_results = [r for r in results if r.success]
        failed_results = [r for r in results if not r.success]

        if successful_results:
            scores = [r.aesthetic_score for r in successful_results]
            avg_score = sum(scores) / len(scores)
            min_score = min(scores)
            max_score = max(scores)
            processing_times = [r.processing_time for r in successful_results]
            avg_processing_time = sum(processing_times) / len(processing_times)
        else:
            avg_score = min_score = max_score = avg_processing_time = 0.0

        output_data = {
            'labeling_info': {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'model_used': self.labeler.model_name,
                'total_images': len(results),
                'successful_labels': len(successful_results),
                'failed_labels': len(failed_results),
                'success_rate': len(successful_results) / len(results) * 100 if results else 0,
                'statistics': {
                    'avg_score': avg_score,
                    'min_score': min_score,
                    'max_score': max_score,
                    'avg_processing_time': avg_processing_time
                }
            },
            'results': []
        }

        # Add individual results
        for result in results:
            result_data = {
                'image_path': result.image_path,
                'image_filename': os.path.basename(result.image_path),
                'aesthetic_score': result.aesthetic_score,
                'reasoning': result.reasoning,
                'processing_time': result.processing_time,
                'success': result.success
            }

            if not result.success:
                result_data['error_message'] = result.error_message

            output_data['results'].append(result_data)

        # Save to file
        output_path = os.path.join(self.output_dir, output_filename)
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)

        print(f"\nResults saved to: {output_path}")
        return output_path

    def run_labeling(self, max_images: Optional[int] = None, batch_size: int = 10) -> str:
        """
        Run the complete labeling pipeline.

        Args:
            max_images: Maximum number of images to process (None for all)
            batch_size: Batch size for rate limiting

        Returns:
            Path to results file
        """
        print("=== Gemini 2.5 Flash Hero Image Labeling Pipeline ===")

        # Get candidate images
        image_paths = self.get_candidate_images()

        if max_images:
            image_paths = image_paths[:max_images]

        print(f"Found {len(image_paths)} candidate images to process")

        if not image_paths:
            print("No images found to process!")
            return ""

        # Process images
        start_time = time.time()
        results = self.labeler.label_batch(image_paths, batch_size=batch_size)
        total_time = time.time() - start_time

        # Print summary
        successful = len([r for r in results if r.success])
        failed = len([r for r in results if not r.success])

        print(f"\n=== Labeling Complete ===")
        print(f"Total time: {total_time:.1f} seconds")
        print(f"Successful: {successful}/{len(results)} ({successful/len(results)*100:.1f}%)")
        print(f"Failed: {failed}/{len(results)}")

        if successful > 0:
            scores = [r.aesthetic_score for r in results if r.success]
            print(f"Score range: {min(scores):.1f} - {max(scores):.1f}")
            print(f"Average score: {sum(scores)/len(scores):.1f}")

        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"gemini_labels_{timestamp}.json"
        return self.save_results(results, output_filename)


def load_api_key() -> str:
    """Load API key from GEMINI_API_KEY.txt file."""
    try:
        with open("GEMINI_API_KEY.txt", "r") as f:
            api_key = f.read().strip()
            if not api_key:
                raise ValueError("API key file is empty")
            return api_key
    except FileNotFoundError:
        raise FileNotFoundError(
            "GEMINI_API_KEY.txt not found. Please create this file with your Google AI API key."
        )
    except Exception as e:
        raise Exception(f"Error loading API key: {str(e)}")


def main():
    """Example usage of the labeling pipeline."""

    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Gemini synchronous labeling pipeline")
    parser.add_argument(
        'candidates_dir',
        help='Directory containing candidate images to label'
    )
    parser.add_argument(
        '--output-dir', '-o',
        required=True,
        help='Output directory for results'
    )
    parser.add_argument(
        '--max-images', '-n',
        type=int,
        help='Maximum number of images to process (for testing)'
    )
    args = parser.parse_args()

    # Configuration
    CANDIDATES_DIR = args.candidates_dir
    OUTPUT_DIR = args.output_dir

    # Use command line max_images if provided
    MAX_IMAGES = args.max_images

    print("Starting Gemini labeling pipeline...")

    try:
        # Load API key from file
        print("Loading API key...")
        API_KEY = load_api_key()
        print("✓ API key loaded successfully")

        pipeline = LabelingPipeline(API_KEY, CANDIDATES_DIR, OUTPUT_DIR)
        results_file = pipeline.run_labeling(max_images=MAX_IMAGES, batch_size=5)

        if results_file:
            print(f"Labeling completed successfully!")
            print(f"Results saved to: {results_file}")
        else:
            print("Labeling failed - no results generated")

    except Exception as e:
        print(f"Pipeline error: {str(e)}")
        print("\nTroubleshooting:")
        print("1. Make sure GEMINI_API_KEY.txt exists in the current directory")
        print("2. Ensure the API key is valid and has the correct permissions")
        print("3. Check your internet connection")


if __name__ == "__main__":
    main()