"""
Gemini Labeling Script for Hero Images

This script supports both batch and synchronous labeling workflows:

Batch mode (default):
1. Prepares requests for all images
2. Submits batch job to Gemini
3. Polls for completion
4. Downloads and processes results
5. Generates final JSON output

Synchronous mode (--sync):
1. Processes images one-by-one in real-time
2. Immediate results with progress updates
3. Good for smaller jobs or when you need results quickly

Usage:
    # Batch mode (default, recommended for large jobs)
    python3 gemini_labeling.py /path/to/candidates --output-dir /path/to/results

    # Synchronous mode (good for smaller jobs)
    python3 gemini_labeling.py /path/to/candidates --output-dir /path/to/results --sync

Run this script independently - it will handle everything automatically.
"""

import json
import os
import time
import base64
import sys
import argparse

import google.generativeai as genai

from typing import List, Dict, Any, Optional
from google import genai as batch_genai
from datetime import datetime

from hero_images.image_processor import ImageProcessor


# Gemini API Pricing (as of January 2025, per https://ai.google.dev/gemini-api/docs/pricing)
#
# BATCH API (50% discount vs. standard API):
# Gemini 2.5 Flash Batch:
#   - Input: $0.15 per 1M tokens
#   - Output: $1.25 per 1M tokens
#
# Gemini 2.5 Pro Batch:
#   - Input: $0.625 per 1M tokens (prompts ≤200k tokens)
#   - Output: $5.00 per 1M tokens (prompts ≤200k tokens)
#
# SYNCHRONOUS API (standard pricing):
# Gemini 2.5 Flash:
#   - Input: $0.30 per 1M tokens
#   - Output: $2.50 per 1M tokens
#
# Gemini 2.5 Pro:
#   - Input: $1.25 per 1M tokens (prompts ≤200k tokens)
#   - Output: $10.00 per 1M tokens (prompts ≤200k tokens)
#
# Image token calculation:
#   - Images are tiled into 768x768 pixel tiles
#   - Each tile = 258 tokens
#   - For 768x768 or smaller: 1 tile = 258 tokens
#   - Calculation: tiles = ((width+767)//768) * ((height+767)//768)
#
# Per-image cost estimate (for 768px images with ~300 token prompt and ~200 token response):
#   - Image input: 258 tokens
#   - Text prompt: ~300 tokens
#   - Text output: ~200 tokens
#   - Total input: ~558 tokens
#   - Total output: ~200 tokens
#
# Batch API costs:
GEMINI_25_FLASH_BATCH_COST_PER_IMAGE = (558 * 0.15 / 1_000_000) + (200 * 1.25 / 1_000_000)  # ~$0.00033 per image
GEMINI_25_PRO_BATCH_COST_PER_IMAGE = (558 * 0.625 / 1_000_000) + (200 * 5.00 / 1_000_000)  # ~$0.00135 per image

# Synchronous API costs (2x batch API):
GEMINI_25_FLASH_SYNC_COST_PER_IMAGE = (558 * 0.30 / 1_000_000) + (200 * 2.50 / 1_000_000)  # ~$0.00067 per image
GEMINI_25_PRO_SYNC_COST_PER_IMAGE = (558 * 1.25 / 1_000_000) + (200 * 10.00 / 1_000_000)  # ~$0.00270 per image


def sanitize_model_name(model_name: str) -> str:
    """Sanitize model name for use in filenames by replacing problematic characters."""
    # Remove "models/" prefix if present
    name = model_name.replace('models/', '')
    # Replace problematic characters
    return name.replace(':', '-').replace('/', '-')


def enumerate_image_files(source: str, recursive: bool = False) -> List[str]:
    """
    Enumerate image files from a source (directory, text file, or JSON file).

    Args:
        source: Path to directory, text file with image paths, or JSON file with list of paths
        recursive: If source is a directory, search recursively

    Returns:
        List of absolute image file paths
    """
    image_files = []

    if os.path.isdir(source):
        # Source is a directory - enumerate image files
        if recursive:
            for root, dirs, files in os.walk(source):
                for filename in files:
                    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                        full_path = os.path.join(root, filename)
                        image_files.append(full_path)
        else:
            for filename in os.listdir(source):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    full_path = os.path.join(source, filename)
                    image_files.append(full_path)

    elif os.path.isfile(source):
        # Source is a file - could be text or JSON
        if source.lower().endswith('.json'):
            # JSON file with list of paths
            with open(source, 'r') as f:
                data = json.load(f)
                if isinstance(data, list):
                    image_files = [path for path in data if isinstance(path, str)]
                else:
                    raise ValueError(f"JSON file must contain a list of image paths, got {type(data)}")
        else:
            # Text file with one path per line
            with open(source, 'r') as f:
                for line in f:
                    path = line.strip()
                    if path and not path.startswith('#'):  # Skip empty lines and comments
                        image_files.append(path)
    else:
        raise FileNotFoundError(f"Source does not exist: {source}")

    return image_files


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


class GeminiBatchProcessor:
    """Handles Gemini Batch API workflow."""

    def __init__(self, api_key: str, model_name: str = "gemini-2.5-flash", image_size: int = 768):
        """Initialize with API key, model name, and image size."""
        # Configure old API for synchronous operations (if needed)
        genai.configure(api_key=api_key)
        # Initialize new batch API client
        self.client = batch_genai.Client(api_key=api_key)
        # Ensure model name starts with "models/"
        if not model_name.startswith("models/"):
            model_name = f"models/{model_name}"
        self.model_name = model_name
        self.image_size = image_size

        # Create the aesthetic rating prompt
        self.prompt = """You are an expert wildlife photography curator evaluating camera trap images for aesthetic appeal.

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
  "reasoning": "[2-3 sentence explanation of the score]",
  "image_filename": "[the filename of the image you are analyzing]"
}

Do not include any text before or after the JSON."""

    def prepare_batch_requests(self, image_paths: List[str]) -> List[Dict[str, Any]]:
        """
        Prepare batch requests using the new API structure.

        Returns:
            List of prepared requests
        """
        print(f"Preparing batch requests for {len(image_paths)} images...")

        batch_requests = []

        for i, image_path in enumerate(image_paths):
            try:
                if (i + 1) % 100 == 0:
                    print(f"  Prepared {i + 1}/{len(image_paths)} requests...")

                # Load and resize image
                image_bytes = ImageProcessor.resize_image_to_bytes(image_path, self.image_size)
                image_b64 = base64.b64encode(image_bytes).decode('utf-8')

                # Create custom prompt with filename for cross-validation
                filename = os.path.basename(image_path)
                custom_prompt = self.prompt + f"\n\nNote: You are analyzing the image file named '{filename}'. Please include this exact filename in the image_filename field of your JSON response."

                # Create request with proper structure for batch API
                request = {
                    'contents': [{
                        'parts': [
                            {
                                'inline_data': {
                                    'mime_type': 'image/jpeg',
                                    'data': image_b64
                                }
                            },
                            {
                                'text': custom_prompt
                            }
                        ],
                        'role': 'user'
                    }]
                }

                batch_requests.append((request, image_path))

            except Exception as e:
                print(f"  Warning: Failed to prepare request for {image_path}: {e}")

        print(f"✓ Prepared {len(batch_requests)} requests")
        return batch_requests

    def save_batch_metadata(self, batch_job: Any, image_paths: List[str], output_dir: str) -> str:
        """Save batch job metadata immediately after submission."""

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        metadata_filename = f"gemini_batch_metadata_{timestamp}.json"
        metadata_path = os.path.join(output_dir, metadata_filename)

        metadata = {
            'batch_info': {
                'job_name': batch_job.name,
                'job_id': batch_job.name.split('/')[-1] if '/' in batch_job.name else batch_job.name,
                'display_name': getattr(batch_job, 'display_name', 'N/A'),
                'status': batch_job.state.name,
                'model': self.model_name,
                'submission_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'total_images': len(image_paths)
            },
            'image_list': [
                {
                    'index': i,
                    'path': path,
                    'filename': os.path.basename(path)
                }
                for i, path in enumerate(image_paths)
            ]
        }

        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"✓ Batch metadata saved to: {metadata_path}")
        return metadata_path

    def save_raw_responses_debug(self, results: List[Any], image_paths: List[str]) -> None:
        """Save raw API responses to debug file for troubleshooting alignment issues."""

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        debug_filename = f"gemini_batch_debug_responses_{timestamp}.json"
        debug_path = os.path.join("/mnt/c/temp/hero-images/labels/", debug_filename)

        debug_data = {
            "timestamp": timestamp,
            "total_results": len(results),
            "total_expected": len(image_paths),
            "expected_image_order": [
                {
                    "index": i,
                    "path": path,
                    "filename": os.path.basename(path)
                }
                for i, path in enumerate(image_paths)
            ],
            "raw_responses": []
        }

        for i, result in enumerate(results):
            response_debug = {
                "response_index": i,
                "has_response": bool(result.response),
                "has_text": bool(result.response and hasattr(result.response, 'text') and result.response.text),
                "response_text": "",
                "parsed_json": None,
                "parsing_error": None
            }

            if result.response and hasattr(result.response, 'text') and result.response.text:
                response_text = result.response.text.strip()
                response_debug["response_text"] = response_text

                # Try to parse JSON
                try:
                    if response_text.startswith('```json'):
                        response_text = response_text[7:]
                    if response_text.endswith('```'):
                        response_text = response_text[:-3]
                    response_text = response_text.strip()

                    parsed = json.loads(response_text)
                    response_debug["parsed_json"] = parsed
                except Exception as e:
                    response_debug["parsing_error"] = str(e)

            debug_data["raw_responses"].append(response_debug)

        with open(debug_path, 'w', encoding='utf-8') as f:
            json.dump(debug_data, f, indent=2)

        print(f"✓ Raw responses saved for debugging: {debug_path}")

    def submit_batch_job(self, batch_requests_and_paths: List[tuple]) -> tuple:
        """
        Submit batch job to Gemini using new API.

        Returns:
            (Batch job object, List of image paths in order)
        """
        print("Submitting batch job to Gemini...")

        try:
            # Separate requests from paths
            batch_requests = [req for req, _ in batch_requests_and_paths]
            image_paths = [path for _, path in batch_requests_and_paths]

            # Create batch job with inline requests
            print("  Creating batch job...")
            batch_job = self.client.batches.create(
                model=self.model_name,
                src=batch_requests,
                config={
                    'display_name': f"hero-images-batch-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
                }
            )

            print(f"  ✓ Batch job created: {batch_job.name}")
            print(f"  ✓ Status: {batch_job.state.name}")
            print(f"  ✓ Model: {self.model_name}")
            print(f"  ✓ Request count: {len(batch_requests)}")

            # Debug: Print job details
            if hasattr(batch_job, 'create_time'):
                print(f"  ✓ Created at: {batch_job.create_time}")
            if hasattr(batch_job, 'error') and batch_job.error:
                print(f"  ⚠️  Initial error: {batch_job.error}")

            return batch_job, image_paths

        except Exception as e:
            raise Exception(f"Failed to submit batch job: {str(e)}")

    def poll_batch_completion(self, batch_job: Any, poll_interval: int = 60) -> Any:
        """
        Poll batch job until completion.

        Args:
            batch_job: Batch job object from submit_batch_job
            poll_interval: Seconds between polls (default 60 seconds)

        Returns:
            Final batch job object
        """
        print(f"Polling batch job: {batch_job.name}")
        print(f"Poll interval: {poll_interval} seconds")

        start_time = time.time()
        poll_count = 0

        completed_states = {
            'JOB_STATE_SUCCEEDED',
            'JOB_STATE_FAILED',
            'JOB_STATE_CANCELLED',
            'JOB_STATE_EXPIRED',
        }

        while True:
            try:
                # Get updated batch job status
                current_job = self.client.batches.get(name=batch_job.name)
                poll_count += 1
                elapsed = time.time() - start_time

                # Extract additional status information that's actually available
                status_info = f"Status = {current_job.state.name}"

                # Add timing information if available
                if hasattr(current_job, 'start_time') and current_job.start_time:
                    # Job has started processing
                    if current_job.state.name == 'JOB_STATE_RUNNING':
                        status_info += f", Started at {current_job.start_time}"
                elif hasattr(current_job, 'create_time') and current_job.create_time and current_job.state.name == 'JOB_STATE_PENDING':
                    # Job is still pending, show how long it's been queued
                    from datetime import datetime
                    try:
                        # Parse ISO timestamp and calculate wait time
                        create_time_str = str(current_job.create_time)
                        # Simple parsing - just extract minutes for rough estimate
                        queue_minutes = elapsed / 60
                        status_info += f", Queued for {queue_minutes:.1f}m"
                    except:
                        pass

                # Add last update time
                if hasattr(current_job, 'update_time') and current_job.update_time:
                    status_info += f", Last update = {current_job.update_time}"

                # Check for error information
                if hasattr(current_job, 'error') and current_job.error:
                    status_info += f", Error = {current_job.error}"

                print(f"  Poll #{poll_count} ({elapsed/3600:.1f}h elapsed): {status_info}")

                # Debug: Print all available attributes and their values (remove this after testing)
                if poll_count == 1:  # Only on first poll to avoid spam
                    available_attrs = [attr for attr in dir(current_job) if not attr.startswith('_')]
                    print(f"  Debug: Available job attributes = {available_attrs}")

                    # Print key attribute values
                    for attr in ['state', 'name', 'created_time', 'update_time', 'progress']:
                        if hasattr(current_job, attr):
                            value = getattr(current_job, attr)
                            print(f"  Debug: {attr} = {value}")

                            # If it's progress, drill down further
                            if attr == 'progress' and value:
                                progress_attrs = [a for a in dir(value) if not a.startswith('_')]
                                print(f"  Debug: progress attributes = {progress_attrs}")
                                for progress_attr in progress_attrs[:5]:  # Limit to first 5
                                    try:
                                        prog_value = getattr(value, progress_attr)
                                        print(f"  Debug: progress.{progress_attr} = {prog_value}")
                                    except:
                                        pass

                if current_job.state.name in completed_states:
                    if current_job.state.name == "JOB_STATE_SUCCEEDED":
                        print(f"  ✓ Batch job completed successfully!")
                        return current_job
                    elif current_job.state.name == "JOB_STATE_FAILED":
                        raise Exception(f"Batch job failed: {getattr(current_job, 'error', 'Unknown error')}")
                    elif current_job.state.name == "JOB_STATE_CANCELLED":
                        print(f"  🚫 Batch job was cancelled")
                        raise Exception("Batch job was cancelled")
                    elif current_job.state.name == "JOB_STATE_EXPIRED":
                        print(f"  ⏰ Batch job expired")
                        raise Exception("Batch job expired")

                print(f"    Job is {current_job.state.name.lower().replace('job_state_', '')}... waiting {poll_interval} seconds")
                time.sleep(poll_interval)

            except KeyboardInterrupt:
                print("\n⚠️  Polling interrupted by user")
                print(f"Batch job name: {batch_job.name}")
                print(f"To cancel this job, run:")
                print(f"  python -m hero_images.gemini_labeling --cancel {batch_job.name}")
                print("💡 The job will continue running on Google's servers until cancelled")
                raise

            except Exception as e:
                error_msg = str(e).lower()
                # Don't retry for final states - these are permanent
                if any(keyword in error_msg for keyword in ['cancelled', 'expired', 'failed']):
                    print(f"  ❌ Final state reached: {e}")
                    raise  # Re-raise to exit the polling loop
                else:
                    # Temporary errors - retry
                    print(f"  ⚠️  Error during polling: {e}")
                    print(f"  Will retry in {poll_interval} seconds...")
                    time.sleep(poll_interval)

    def download_and_process_results(self, batch_job: Any, image_paths: List[str]) -> List[Dict[str, Any]]:
        """
        Download and process batch results.

        Returns:
            List of processed results
        """
        print("Processing batch results...")

        try:
            # Get results from inline responses
            if not (batch_job.dest and batch_job.dest.inlined_responses):
                raise Exception("No results found in batch job")

            results = batch_job.dest.inlined_responses
            print(f"✓ Found {len(results)} results")

            # Validate response count matches expected count
            expected_count = len(image_paths)
            actual_count = len(results)
            if actual_count != expected_count:
                raise Exception(f"Response count mismatch: expected {expected_count} results but got {actual_count}. This indicates a problem with batch processing order.")

            print(f"✓ Response count validation passed: {actual_count} results match {expected_count} expected")

            # Save raw responses for debugging
            self.save_raw_responses_debug(results, image_paths)

            # Create lookup mapping of expected filenames to paths
            expected_filenames = {}
            for path in image_paths:
                filename = os.path.basename(path)
                expected_filenames[filename] = path

            print(f"✓ Created filename lookup for {len(expected_filenames)} expected images")

            # Process results using filename-based matching (no order assumptions)
            processed_results = []
            returned_filenames = []

            for result_index, result in enumerate(results):
                try:
                    if result.response and hasattr(result.response, 'text') and result.response.text:
                        response_text = result.response.text.strip()

                        # Parse JSON response (handle markdown)
                        if response_text.startswith('```json'):
                            response_text = response_text[7:]
                        if response_text.endswith('```'):
                            response_text = response_text[:-3]

                        response_text = response_text.strip()
                        response_json = json.loads(response_text)

                        # Get filename from response (trust what Gemini returned)
                        returned_filename = response_json.get('image_filename', '')

                        if not returned_filename:
                            print(f"⚠️  Warning: Response {result_index} has no image_filename field - skipping")
                            continue

                        # Check for filename uniqueness
                        if returned_filename in returned_filenames:
                            print(f"❌ Error: Duplicate filename detected: '{returned_filename}'")
                            print(f"   This filename was already returned in a previous response")
                            print("   Aborting processing due to non-unique filenames")
                            raise Exception(f"Duplicate filename in responses: '{returned_filename}'")

                        returned_filenames.append(returned_filename)

                        # Check whether filename matches any expected filename
                        if returned_filename not in expected_filenames:
                            print(f"⚠️  Warning: Response {result_index} returned unexpected filename '{returned_filename}' - skipping")
                            continue

                        # Create successful result using filename-based lookup
                        image_path = expected_filenames[returned_filename]
                        processed_result = {
                            'image_path': image_path,
                            'image_filename': returned_filename,
                            'aesthetic_score': float(response_json['score']),
                            'reasoning': response_json['reasoning'],
                            'success': True
                        }

                        processed_results.append(processed_result)

                    else:
                        # Handle error responses with no text
                        print(f"⚠️  Warning: Response {result_index} has no valid text - skipping")

                except Exception as e:
                    # Check whether this is the duplicate filename error that should abort
                    if "Duplicate filename in responses" in str(e):
                        raise  # Re-raise to abort processing

                    # For other errors, just log and skip this response
                    print(f"⚠️  Warning: Error processing response {result_index}: {e} - skipping")

            # Summary of filename-based processing
            successful_count = len([r for r in processed_results if r['success']])
            print(f"✓ Filename-based processing complete:")
            print(f"   • Total API responses: {len(results)}")
            print(f"   • Valid responses processed: {len(processed_results)}")
            print(f"   • Successful labels: {successful_count}")
            print(f"   • Unique filenames returned: {len(returned_filenames)}")

            return processed_results

        except Exception as e:
            raise Exception(f"Failed to process results: {str(e)}")

    def save_results(self, results: List[Dict[str, Any]], output_path: str, batch_info: Dict[str, Any]):
        """Save processed results to JSON file."""

        successful_results = [r for r in results if r['success']]
        failed_results = [r for r in results if not r['success']]

        if successful_results:
            scores = [r['aesthetic_score'] for r in successful_results]
            avg_score = sum(scores) / len(scores)
            min_score = min(scores)
            max_score = max(scores)
        else:
            avg_score = min_score = max_score = 0.0

        output_data = {
            'labeling_info': {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'model_used': self.model_name,
                'processing_method': 'batch_api',
                'batch_job_name': batch_info.get('name', 'unknown'),
                'total_images': len(results),
                'successful_labels': len(successful_results),
                'failed_labels': len(failed_results),
                'success_rate': len(successful_results) / len(results) * 100 if results else 0,
                'statistics': {
                    'avg_score': avg_score,
                    'min_score': min_score,
                    'max_score': max_score
                }
            },
            'results': results
        }

        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)

        print(f"✓ Results saved to: {output_path}")
        return output_path


class GeminiSyncProcessor:
    """Handles synchronous Gemini API processing for real-time labeling."""

    def __init__(self, api_key: str, model_name: str = "gemini-2.5-flash", image_size: int = 768):
        """Initialize with API key, model name, and image size."""
        genai.configure(api_key=api_key)
        # Ensure model name does NOT start with "models/" for synchronous API
        if model_name.startswith("models/"):
            model_name = model_name.replace("models/", "")
        self.model = genai.GenerativeModel(model_name)
        self.model_name = model_name
        self.image_size = image_size

        # Create the aesthetic rating prompt
        self.prompt = """You are an expert wildlife photography curator evaluating camera trap images for aesthetic appeal.

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

    def label_image(self, image_path: str) -> Dict[str, Any]:
        """
        Label a single image using synchronous API.

        Args:
            image_path: Path to the image file

        Returns:
            Dict with result information
        """
        start_time = time.time()

        try:
            # Load and resize image
            image_bytes = ImageProcessor.resize_image_to_bytes(image_path, self.image_size)

            # Prepare the image for the API
            image_data = {
                'mime_type': 'image/jpeg',
                'data': image_bytes
            }

            # Generate response
            response = self.model.generate_content([image_data, self.prompt])

            processing_time = time.time() - start_time

            # Parse the JSON response (handle markdown code blocks)
            try:
                response_text = response.text.strip()

                # Remove markdown code blocks if present
                if response_text.startswith('```json'):
                    response_text = response_text[7:]
                if response_text.endswith('```'):
                    response_text = response_text[:-3]

                response_text = response_text.strip()
                result_json = json.loads(response_text)
                score = float(result_json['score'])
                reasoning = result_json['reasoning']

                # Validate score range
                if not (0 <= score <= 10):
                    raise ValueError(f"Score {score} outside valid range 0-10")

                return {
                    'image_path': image_path,
                    'image_filename': os.path.basename(image_path),
                    'aesthetic_score': score,
                    'reasoning': reasoning,
                    'processing_time': processing_time,
                    'success': True
                }

            except (json.JSONDecodeError, KeyError, ValueError) as e:
                return {
                    'image_path': image_path,
                    'image_filename': os.path.basename(image_path),
                    'aesthetic_score': 0.0,
                    'reasoning': "",
                    'processing_time': processing_time,
                    'success': False,
                    'error_message': f"Failed to parse response: {str(e)}. Response: {response.text[:200]}"
                }

        except Exception as e:
            processing_time = time.time() - start_time
            return {
                'image_path': image_path,
                'image_filename': os.path.basename(image_path),
                'aesthetic_score': 0.0,
                'reasoning': "",
                'processing_time': processing_time,
                'success': False,
                'error_message': f"API error: {str(e)}"
            }

    def label_images(self, image_paths: List[str], rate_limit_pause: int = 2, batch_size: int = 10) -> List[Dict[str, Any]]:
        """
        Label multiple images synchronously with rate limiting.

        Args:
            image_paths: List of image file paths
            rate_limit_pause: Seconds to pause after each batch
            batch_size: Number of images to process before pausing

        Returns:
            List of result dictionaries
        """
        results = []

        print(f"Processing {len(image_paths)} images synchronously...")

        for i, image_path in enumerate(image_paths):
            print(f"  [{i+1}/{len(image_paths)}] {os.path.basename(image_path)}...", end=" ", flush=True)

            result = self.label_image(image_path)
            results.append(result)

            if result['success']:
                print(f"✓ Score: {result['aesthetic_score']:.1f} ({result['processing_time']:.1f}s)")
            else:
                print(f"✗ Failed: {result.get('error_message', 'Unknown error')}")

            # Rate limiting: brief pause every batch_size images
            if (i + 1) % batch_size == 0 and i < len(image_paths) - 1:
                print(f"  Pausing {rate_limit_pause}s for rate limiting...")
                time.sleep(rate_limit_pause)

        return results

    def save_checkpoint(self, results: List[Dict[str, Any]], checkpoint_path: str):
        """Save checkpoint file with atomic write."""
        successful_results = [r for r in results if r['success']]
        failed_results = [r for r in results if not r['success']]

        if successful_results:
            scores = [r['aesthetic_score'] for r in successful_results]
            avg_score = sum(scores) / len(scores)
            min_score = min(scores)
            max_score = max(scores)
        else:
            avg_score = min_score = max_score = 0.0

        checkpoint_data = {
            'checkpoint_info': {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'model_used': self.model_name,
                'processing_method': 'synchronous_api',
                'total_images': len(results),
                'successful_labels': len(successful_results),
                'failed_labels': len(failed_results),
                'success_rate': len(successful_results) / len(results) * 100 if results else 0,
                'statistics': {
                    'avg_score': avg_score,
                    'min_score': min_score,
                    'max_score': max_score
                },
                'is_checkpoint': True
            },
            'results': results
        }

        # Atomic write: write to backup file first, then rename
        backup_path = checkpoint_path.replace('.tmp.json', '.bk.tmp.json')
        try:
            with open(backup_path, 'w') as f:
                json.dump(checkpoint_data, f, indent=2)

            # Remove old checkpoint if it exists
            if os.path.exists(checkpoint_path):
                os.remove(checkpoint_path)

            # Rename backup to checkpoint
            os.rename(backup_path, checkpoint_path)

        except Exception as e:
            # Clean up backup file if something went wrong
            if os.path.exists(backup_path):
                os.remove(backup_path)
            raise e

    def save_results(self, results: List[Dict[str, Any]], output_path: str):
        """Save processed results to JSON file."""

        successful_results = [r for r in results if r['success']]
        failed_results = [r for r in results if not r['success']]

        if successful_results:
            scores = [r['aesthetic_score'] for r in successful_results]
            avg_score = sum(scores) / len(scores)
            min_score = min(scores)
            max_score = max(scores)
            processing_times = [r['processing_time'] for r in successful_results]
            avg_processing_time = sum(processing_times) / len(processing_times)
        else:
            avg_score = min_score = max_score = avg_processing_time = 0.0

        output_data = {
            'labeling_info': {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'model_used': self.model_name,
                'processing_method': 'synchronous_api',
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
            'results': results
        }

        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)

        print(f"✓ Results saved to: {output_path}")
        return output_path


def load_checkpoint(checkpoint_path: str) -> List[Dict[str, Any]]:
    """Load results from a checkpoint file."""
    try:
        with open(checkpoint_path, 'r') as f:
            data = json.load(f)

        if 'results' in data:
            print(f"✅ Loaded {len(data['results'])} results from checkpoint")
            return data['results']
        else:
            print("⚠️  Checkpoint file format not recognized, starting fresh")
            return []
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")
        print("Starting fresh...")
        return []


def main():
    """Main labeling workflow (supports both batch and synchronous modes)."""

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Gemini Hero Image Labeling (Batch or Synchronous)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Batch mode (default, recommended for large jobs)
  python gemini_labeling.py /path/to/candidates --output-dir /path/to/output

  # Synchronous mode (good for smaller jobs, real-time results)
  python gemini_labeling.py /path/to/candidates --output-dir /path/to/output --sync

  # Synchronous mode with custom checkpoint interval
  python gemini_labeling.py /path/to/candidates --output-dir /path/to/output --sync --checkpoint-interval 500

  # Resume from synchronous checkpoint
  python gemini_labeling.py /path/to/candidates --output-dir /path/to/output --sync --resume /path/to/output/gemini_sync_labels_*.tmp.json

  # Disable checkpointing for short sync jobs
  python gemini_labeling.py /path/to/candidates --output-dir /path/to/output --sync --checkpoint-interval 0

  # Test with 5 images (batch mode)
  python gemini_labeling.py /path/to/candidates --output-dir /path/to/output --max-images 5

  # Test with 5 images (synchronous mode)
  python gemini_labeling.py /path/to/candidates --output-dir /path/to/output --max-images 5 --sync

  # Cancel a running batch job
  python gemini_labeling.py --cancel batches/xyz789

  # Resume batch job with metadata file
  python gemini_labeling.py --resume /path/to/gemini_batch_metadata_20250923_143022.json
        """
    )
    parser.add_argument(
        'source',
        nargs='?',
        help='Directory containing candidate images, text file with image paths (one per line), or JSON file with list of image paths'
    )
    parser.add_argument(
        '--output-dir', '-o',
        help='Output directory for results (required unless using --cancel)'
    )
    parser.add_argument(
        '--max-images', '-n',
        type=int,
        help='Maximum number of images to process (for testing)'
    )
    parser.add_argument(
        '--auto-confirm', '-y',
        action='store_true',
        help='Automatically confirm batch submission without prompting'
    )
    parser.add_argument(
        '--poll-interval', '-p',
        type=int,
        default=60,
        help='Seconds between polling checks (default: 60)'
    )
    parser.add_argument(
        '--cancel',
        type=str,
        help='Cancel a running batch job by providing its name/ID'
    )
    parser.add_argument(
        '--resume',
        type=str,
        help='Resume from checkpoint file (*.tmp.json for sync mode) or retrieve batch job results (metadata file or job ID for batch mode)'
    )
    parser.add_argument(
        '--recursive', '-r',
        action='store_true',
        help='Search for images recursively in subdirectories'
    )
    parser.add_argument(
        '--model', '-m',
        default='gemini-2.5-flash',
        help='Gemini model name to use (default: gemini-2.5-flash). Can optionally include "models/" prefix.'
    )
    parser.add_argument(
        '--image-size',
        type=int,
        default=768,
        help='Maximum dimension for resized images (default: 768)'
    )
    parser.add_argument(
        '--sync',
        action='store_true',
        help='Use synchronous API instead of batch API (2x cost, but real-time results)'
    )
    parser.add_argument(
        '--checkpoint-interval',
        type=int,
        default=1000,
        help='Save checkpoint every N images in sync mode (default: 1000, use 0 to disable)'
    )

    args = parser.parse_args()

    # Validate incompatible arguments
    # Note: --sync with --resume is now allowed for checkpoint files (*.tmp.json)
    if args.sync and args.resume:
        # Check if resume argument is a checkpoint file or batch job
        if not (args.resume.endswith('.tmp.json') or 'checkpoint' in args.resume.lower()):
            print("❌ Error: --sync with --resume requires a checkpoint file (*.tmp.json), not a batch job ID/metadata file.")
            sys.exit(1)
    if args.sync and args.cancel:
        print("❌ Error: --sync and --cancel are incompatible. Cancel is only for batch jobs.")
        sys.exit(1)

    # Configuration from arguments
    SOURCE = args.source
    OUTPUT_DIR = args.output_dir

    mode_str = "Synchronous" if args.sync else "Batch"
    print(f"=== Gemini {mode_str} API Hero Image Labeling ===")
    if not args.cancel:
        print(f"Source: {SOURCE}")
        print(f"Output directory: {OUTPUT_DIR}")
        if args.sync:
            print(f"Mode: Synchronous (real-time processing, 2x cost)")
        else:
            print(f"Mode: Batch (async processing, 50% discount)")

    # Validate arguments (unless cancelling or resuming)
    if not args.cancel and not args.resume:
        if not SOURCE:
            parser.print_help()
            sys.exit(1)
        if not OUTPUT_DIR:
            print("❌ Error: --output-dir is required unless using --cancel or --resume with metadata file")
            sys.exit(1)
        if not os.path.exists(SOURCE):
            print(f"❌ Error: Source does not exist: {SOURCE}")
            sys.exit(1)

    try:
        # Load API key
        print("\n1. Loading API key...")
        api_key = load_api_key()
        print("✓ API key loaded")

        # Handle cancellation if requested
        if args.cancel:
            print(f"\n🚫 Cancelling batch job: {args.cancel}")
            client = batch_genai.Client(api_key=api_key)
            try:
                client.batches.cancel(name=args.cancel)
                print("✓ Cancellation request sent")
                print("💡 It may take a few minutes for the job to fully stop")
            except Exception as e:
                print(f"❌ Failed to cancel job: {str(e)}")
            return

        # Handle resume if requested (batch mode only - sync mode is handled above)
        if args.resume:
            # Check if this is a sync checkpoint file
            if args.resume.endswith('.tmp.json') or 'checkpoint' in args.resume.lower():
                print("❌ Error: Synchronous checkpoint files can only be used with --sync mode")
                sys.exit(1)

            processor = GeminiBatchProcessor(api_key, args.model, args.image_size)

            # Determine if input is a batch ID or metadata file
            if args.resume.endswith('.json') and 'metadata' in args.resume:
                print(f"\n🔄 Loading batch info from metadata file: {args.resume}")
                try:
                    with open(args.resume, 'r') as f:
                        metadata = json.load(f)
                    batch_id = metadata['batch_info']['job_name']
                    image_paths = [item['path'] for item in metadata['image_list']]
                    print(f"✓ Found batch job: {batch_id}")
                    print(f"✓ Found {len(image_paths)} images in original submission")
                except Exception as e:
                    print(f"❌ Failed to load metadata file: {str(e)}")
                    return
            else:
                batch_id = args.resume
                image_paths = []  # We'll try to reconstruct from results
                print(f"\n🔄 Resuming batch job: {batch_id}")

            try:
                # Get current job status
                client = batch_genai.Client(api_key=api_key)
                batch_job = client.batches.get(name=batch_id)
                print(f"✓ Job status: {batch_job.state.name}")

                # Handle based on current status
                if batch_job.state.name in ['JOB_STATE_PENDING', 'JOB_STATE_RUNNING']:
                    print("💡 Job is still processing, resuming polling...")
                    batch_job = processor.poll_batch_completion(batch_job, args.poll_interval)
                elif batch_job.state.name == 'JOB_STATE_SUCCEEDED':
                    print("✓ Job already completed, retrieving results...")
                elif batch_job.state.name in ['JOB_STATE_FAILED', 'JOB_STATE_CANCELLED', 'JOB_STATE_EXPIRED']:
                    print(f"❌ Job is in final state: {batch_job.state.name}")
                    if batch_job.state.name == 'JOB_STATE_FAILED' and hasattr(batch_job, 'error'):
                        print(f"Error details: {batch_job.error}")
                    return
                else:
                    print(f"⚠️ Unknown job state: {batch_job.state.name}")
                    return

                # Process results if job succeeded
                if batch_job.state.name == 'JOB_STATE_SUCCEEDED':
                    print(f"\n6. Processing results...")


                    # If we don't have image paths from metadata, try to reconstruct
                    if not image_paths:
                        print("⚠️ No image paths available - results will have limited path information")
                        # Create dummy paths based on result count
                        if batch_job.dest and batch_job.dest.inlined_responses:
                            result_count = len(batch_job.dest.inlined_responses)
                            image_paths = [f"unknown_image_{i}.jpg" for i in range(result_count)]

                    results = processor.download_and_process_results(batch_job, image_paths)

                    # Save results
                    if not OUTPUT_DIR:
                        OUTPUT_DIR = os.path.dirname(args.resume) if args.resume.endswith('.json') else "."

                    print(f"\n7. Saving results to {OUTPUT_DIR}...")
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    sanitized_model = sanitize_model_name(processor.model_name)
                    output_filename = f"gemini_batch_labels_{sanitized_model}_{timestamp}.json"
                    output_path = os.path.join(OUTPUT_DIR, output_filename)

                    # Create batch_info dict from BatchJob object
                    batch_info = {
                        'name': batch_job.name,
                        'display_name': getattr(batch_job, 'display_name', 'N/A'),
                        'state': batch_job.state.name if hasattr(batch_job.state, 'name') else str(batch_job.state)
                    }

                    processor.save_results(results, output_path, batch_info)

                    # Summary
                    successful = len([r for r in results if r['success']])
                    failed = len([r for r in results if not r['success']])

                    print(f"\n🎉 Resume complete!")
                    print(f"✓ Processed: {len(results)} images")
                    print(f"✓ Successful: {successful}")
                    print(f"✗ Failed: {failed}")
                    print(f"✓ Results: {output_path}")

                    if successful > 0:
                        scores = [r['aesthetic_score'] for r in results if r['success']]
                        print(f"📊 Score range: {min(scores):.1f} - {max(scores):.1f}")
                        print(f"📊 Average score: {sum(scores)/len(scores):.1f}")

            except Exception as e:
                print(f"❌ Failed to resume job: {str(e)}")
            return

        # Get image files first (common to both modes)
        print("\n2. Finding candidate images...")
        image_files = enumerate_image_files(SOURCE, args.recursive)
        image_files.sort()

        # Limit number of images if specified
        if args.max_images:
            image_files = image_files[:args.max_images]
            print(f"✓ Found {len(image_files)} images to process (limited to {args.max_images})")
        else:
            print(f"✓ Found {len(image_files)} images to process")

        if not image_files:
            print("❌ No images found!")
            return

        # ===== SYNCHRONOUS MODE =====
        if args.sync:
            # Initialize synchronous processor
            processor = GeminiSyncProcessor(api_key, args.model, args.image_size)

            # Ensure output directory exists
            os.makedirs(OUTPUT_DIR, exist_ok=True)

            # Handle resume logic
            existing_results = []
            processed_filenames = set()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            sanitized_model = sanitize_model_name(processor.model_name)
            output_filename = f"gemini_sync_labels_{sanitized_model}_{timestamp}.json"
            output_path = os.path.join(OUTPUT_DIR, output_filename)
            checkpoint_path = output_path.replace('.json', '.tmp.json')

            if args.resume:
                if not os.path.exists(args.resume):
                    print(f"❌ Error: Checkpoint file does not exist: {args.resume}")
                    sys.exit(1)

                print(f"\n3. Loading checkpoint from {args.resume}...")
                existing_results = load_checkpoint(args.resume)
                processed_filenames = {r['image_filename'] for r in existing_results}

                # Use the same timestamp/naming as the checkpoint for consistency
                checkpoint_basename = os.path.basename(args.resume)
                if checkpoint_basename.endswith('.tmp.json'):
                    output_filename = checkpoint_basename.replace('.tmp.json', '.json')
                    output_path = os.path.join(OUTPUT_DIR, output_filename)
                    checkpoint_path = args.resume  # Continue using the same checkpoint file

            # Filter images to only process those not already completed
            images_to_process = []
            for image_path in image_files:
                filename = os.path.basename(image_path)
                if filename not in processed_filenames:
                    images_to_process.append(image_path)

            # Estimate cost for synchronous mode (only for remaining images)
            model_name_lower = processor.model_name.lower()
            if 'gemini-2.5-flash' in model_name_lower or 'gemini-2-5-flash' in model_name_lower:
                cost_per_image = GEMINI_25_FLASH_SYNC_COST_PER_IMAGE
                model_info = "Gemini 2.5 Flash Sync"
            elif 'gemini-2.5-pro' in model_name_lower or 'gemini-2-5-pro' in model_name_lower:
                cost_per_image = GEMINI_25_PRO_SYNC_COST_PER_IMAGE
                model_info = "Gemini 2.5 Pro Sync"
            else:
                cost_per_image = None
                model_info = processor.model_name

            if cost_per_image:
                estimated_cost = len(images_to_process) * cost_per_image
                print(f"\nEstimated cost ({model_info}): ${estimated_cost:.4f}")
                if args.image_size != 768:
                    print(f"⚠️  Warning: Cost estimate assumes 768px images. Your --image-size is {args.image_size}px.")
                    print(f"   Actual cost may vary depending on image dimensions.")
            else:
                print(f"\n⚠️  Warning: Cost estimate unavailable for model {model_info}")
                print(f"   Cost constants only defined for Gemini 2.5 Flash and Gemini 2.5 Pro")

            if args.resume:
                print(f"📥 Resuming: {len(existing_results)} already processed, {len(images_to_process)} remaining")
            else:
                print(f"🆕 Starting fresh: {len(images_to_process)} images to process")

            if not images_to_process:
                print("✅ All images already processed!")
                final_results = existing_results
            else:
                # Ask for confirmation
                if args.auto_confirm:
                    print("Auto-confirming synchronous processing...")
                else:
                    response = input("Continue with synchronous processing? (y/N): ")
                    if response.lower() != 'y':
                        print("Cancelled by user")
                        return

                # Process images synchronously with checkpointing
                step_num = 4 if args.resume else 3
                print(f"\n{step_num}. Processing images...")
                start_time = time.time()

                new_results = []
                all_results = existing_results.copy()

                for i, image_path in enumerate(images_to_process):
                    print(f"  [{i+1}/{len(images_to_process)}] {os.path.basename(image_path)}...", end=" ", flush=True)

                    result = processor.label_image(image_path)
                    new_results.append(result)
                    all_results.append(result)

                    if result['success']:
                        print(f"✓ Score: {result['aesthetic_score']:.1f} ({result['processing_time']:.1f}s)")
                    else:
                        print(f"✗ Failed: {result.get('error_message', 'Unknown error')}")

                    # Save checkpoint periodically
                    if args.checkpoint_interval > 0 and (i + 1) % args.checkpoint_interval == 0:
                        print(f"💾 Saving checkpoint after {len(all_results)} total images...")
                        processor.save_checkpoint(all_results, checkpoint_path)

                    # Rate limiting: brief pause every 10 images
                    if (i + 1) % 10 == 0 and i < len(images_to_process) - 1:
                        print(f"  Pausing 2s for rate limiting...")
                        time.sleep(2)

                total_time = time.time() - start_time
                final_results = all_results

            # Save final results
            step_num = 5 if args.resume else 4
            print(f"\n{step_num}. Saving final results...")
            processor.save_results(final_results, output_path)

            # Clean up checkpoint file on successful completion
            if args.checkpoint_interval > 0 and os.path.exists(checkpoint_path):
                try:
                    os.remove(checkpoint_path)
                    print(f"🧹 Cleaned up checkpoint file: {checkpoint_path}")
                except Exception as e:
                    print(f"⚠️  Could not remove checkpoint file: {e}")

            # Summary
            successful = len([r for r in final_results if r['success']])
            failed = len([r for r in final_results if not r['success']])

            print(f"\n🎉 Synchronous labeling complete!")
            if 'total_time' in locals():
                print(f"✓ Total time: {total_time:.1f}s")
            print(f"✓ Processed: {len(final_results)} images")
            print(f"✓ Successful: {successful}")
            print(f"✗ Failed: {failed}")
            print(f"✓ Results: {output_path}")

            if successful > 0:
                scores = [r['aesthetic_score'] for r in final_results if r['success']]
                print(f"📊 Score range: {min(scores):.1f} - {max(scores):.1f}")
                print(f"📊 Average score: {sum(scores)/len(scores):.1f}")

            return

        # ===== BATCH MODE (DEFAULT) =====
        # Initialize batch processor
        processor = GeminiBatchProcessor(api_key, args.model, args.image_size)

        # Ask for confirmation (unless auto-confirm is set)
        # Estimate cost based on model type and mode
        model_name_lower = processor.model_name.lower()
        if 'gemini-2.5-flash' in model_name_lower or 'gemini-2-5-flash' in model_name_lower:
            cost_per_image = GEMINI_25_FLASH_BATCH_COST_PER_IMAGE
            model_info = "Gemini 2.5 Flash Batch"
        elif 'gemini-2.5-pro' in model_name_lower or 'gemini-2-5-pro' in model_name_lower:
            cost_per_image = GEMINI_25_PRO_BATCH_COST_PER_IMAGE
            model_info = "Gemini 2.5 Pro Batch"
        else:
            cost_per_image = None
            model_info = processor.model_name

        if cost_per_image:
            estimated_cost = len(image_files) * cost_per_image
            print(f"\nEstimated cost ({model_info}): ${estimated_cost:.4f}")
            if args.image_size != 768:
                print(f"⚠️  Warning: Cost estimate assumes 768px images. Your --image-size is {args.image_size}px.")
                print(f"   Actual cost may vary depending on image dimensions.")
        else:
            print(f"\n⚠️  Warning: Cost estimate unavailable for model {model_info}")
            print(f"   Cost constants only defined for Gemini 2.5 Flash and Gemini 2.5 Pro")

        if args.auto_confirm:
            print("Auto-confirming batch submission...")
        else:
            response = input("Continue with batch submission? (y/N): ")
            if response.lower() != 'y':
                print("Cancelled by user")
                return

        # Prepare batch requests
        print(f"\n3. Preparing batch requests...")
        batch_requests = processor.prepare_batch_requests(image_files)

        if not batch_requests:
            print("❌ No requests prepared!")
            return

        # Submit batch job
        print(f"\n4. Submitting batch job...")
        batch_job, ordered_image_paths = processor.submit_batch_job(batch_requests)

        # Save batch metadata immediately after submission
        print(f"\n4b. Saving batch metadata...")
        processor.save_batch_metadata(batch_job, ordered_image_paths, OUTPUT_DIR)

        # Poll for completion
        print(f"\n5. Polling for completion...")
        print("💡 This may take several hours. You can safely interrupt and resume later.")
        try:
            batch_job = processor.poll_batch_completion(batch_job, args.poll_interval)
        except Exception as e:
            error_msg = str(e).lower()
            if any(keyword in error_msg for keyword in ['cancelled', 'expired', 'failed']):
                print(f"\n❌ Batch job terminated: {str(e)}")
                print("No results will be processed.")
                return
            else:
                raise  # Re-raise unexpected errors

        # Download and process results
        print(f"\n6. Processing results...")
        results = processor.download_and_process_results(batch_job, ordered_image_paths)

        # Save results
        print(f"\n7. Saving results...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        sanitized_model = sanitize_model_name(processor.model_name)
        output_filename = f"gemini_batch_labels_{sanitized_model}_{timestamp}.json"
        output_path = os.path.join(OUTPUT_DIR, output_filename)

        # Create batch_info dict from BatchJob object
        batch_info = {
            'name': batch_job.name,
            'display_name': getattr(batch_job, 'display_name', 'N/A'),
            'state': batch_job.state.name if hasattr(batch_job.state, 'name') else str(batch_job.state)
        }

        processor.save_results(results, output_path, batch_info)

        # Summary
        successful = len([r for r in results if r['success']])
        failed = len([r for r in results if not r['success']])

        print(f"\n🎉 Batch labeling complete!")
        print(f"✓ Processed: {len(results)} images")
        print(f"✓ Successful: {successful}")
        print(f"✗ Failed: {failed}")
        print(f"✓ Results: {output_path}")

        if successful > 0:
            scores = [r['aesthetic_score'] for r in results if r['success']]
            print(f"📊 Score range: {min(scores):.1f} - {max(scores):.1f}")
            print(f"📊 Average score: {sum(scores)/len(scores):.1f}")

    except KeyboardInterrupt:
        print(f"\n⚠️  Process interrupted by user")
        sys.exit(1)

    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
