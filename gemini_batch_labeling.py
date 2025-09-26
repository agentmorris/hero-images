"""
Gemini Batch API Labeling Script for Hero Images

This script handles the complete asynchronous batch workflow:
1. Prepares JSONL file with all image requests
2. Submits batch job to Gemini
3. Polls for completion
4. Downloads and processes results
5. Generates final JSON output

Usage:
    python3 gemini_batch_labeling.py <candidates_directory>
    python3 gemini_batch_labeling.py /path/to/candidates --output-dir /path/to/results

Run this script independently - it will handle everything automatically.
"""

import json
import os
import time
import base64
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from PIL import Image
import io
import google.generativeai as genai
from google import genai as batch_genai
from datetime import datetime
import sys
import argparse


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


class ImageProcessor:
    """Handles image loading and resizing."""

    @staticmethod
    def resize_image_to_768_long_side(image_path: str) -> bytes:
        """Resize image to 768px on long side while preserving aspect ratio."""
        try:
            with Image.open(image_path) as img:
                if img.mode != 'RGB':
                    img = img.convert('RGB')

                width, height = img.size

                if width > height:
                    new_width = 768
                    new_height = int((height * 768) / width)
                else:
                    new_height = 768
                    new_width = int((width * 768) / height)

                resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

                img_byte_arr = io.BytesIO()
                resized_img.save(img_byte_arr, format='JPEG', quality=85, optimize=True)
                return img_byte_arr.getvalue()

        except Exception as e:
            raise Exception(f"Failed to process image {image_path}: {str(e)}")


class GeminiBatchProcessor:
    """Handles Gemini Batch API workflow."""

    def __init__(self, api_key: str):
        """Initialize with API key."""
        # Configure old API for synchronous operations (if needed)
        genai.configure(api_key=api_key)
        # Initialize new batch API client
        self.client = batch_genai.Client(api_key=api_key)
        self.model_name = "models/gemini-2.5-flash"

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
                image_bytes = ImageProcessor.resize_image_to_768_long_side(image_path)
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
                print(f"  python gemini_batch_labeling.py --cancel {batch_job.name}")
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

                        # Check if filename matches any expected filename
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
                    # Check if this is the duplicate filename error that should abort
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


def main():
    """Main batch processing workflow."""

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Gemini Batch API Hero Image Labeling",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test with 5 images
  python gemini_batch_labeling.py /path/to/candidates --output-dir /path/to/output --max-images 5

  # Process full dataset
  python gemini_batch_labeling.py /path/to/candidates --output-dir /path/to/output

  # Cancel a running job
  python gemini_batch_labeling.py --cancel batches/xyz789

  # Resume with batch ID (for running or completed jobs)
  python gemini_batch_labeling.py --resume batches/xyz789 --output-dir /path/to/output

  # Resume with metadata file (automatically finds batch ID and image paths)
  python gemini_batch_labeling.py --resume /path/to/gemini_batch_metadata_20250923_143022.json
        """
    )
    parser.add_argument(
        'candidates_dir',
        nargs='?',
        help='Directory containing candidate images to label'
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
        help='Retrieve results from a completed batch job by providing its name/ID'
    )

    args = parser.parse_args()

    # Configuration from arguments
    CANDIDATES_DIR = args.candidates_dir
    OUTPUT_DIR = args.output_dir

    print("=== Gemini Batch API Hero Image Labeling ===")
    if not args.cancel:
        print(f"Candidates directory: {CANDIDATES_DIR}")
        print(f"Output directory: {OUTPUT_DIR}")

    # Validate arguments (unless cancelling or resuming)
    if not args.cancel and not args.resume:
        if not CANDIDATES_DIR:
            print("❌ Error: candidates_dir is required unless using --cancel or --resume")
            sys.exit(1)
        if not OUTPUT_DIR:
            print("❌ Error: --output-dir is required unless using --cancel or --resume with metadata file")
            sys.exit(1)
        if not os.path.exists(CANDIDATES_DIR):
            print(f"❌ Error: Candidates directory does not exist: {CANDIDATES_DIR}")
            sys.exit(1)
        if not os.path.isdir(CANDIDATES_DIR):
            print(f"❌ Error: Path is not a directory: {CANDIDATES_DIR}")
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

        # Handle resume if requested
        if args.resume:
            processor = GeminiBatchProcessor(api_key)

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
                    output_filename = f"gemini_batch_labels_{timestamp}.json"
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

        # Initialize processor
        processor = GeminiBatchProcessor(api_key)

        # Get image files
        print("\n2. Finding candidate images...")
        image_files = []
        for filename in os.listdir(CANDIDATES_DIR):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                full_path = os.path.join(CANDIDATES_DIR, filename)
                image_files.append(full_path)

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

        # Ask for confirmation (unless auto-confirm is set)
        estimated_cost = len(image_files) * 0.003  # Rough estimate
        print(f"\nEstimated cost: ${estimated_cost:.2f}")
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
        output_filename = f"gemini_batch_labels_{timestamp}.json"
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