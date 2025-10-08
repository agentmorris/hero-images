"""
Local VLM Labeling Script for Hero Images via vLLM

This script processes camera trap images using a locally-hosted vision-language model
via vLLM, producing JSON output compatible with the existing visualization pipeline.
Defaults to Qwen2.5-VL models but supports any vLLM-compatible VLM.

Prerequisites:
1. Install vLLM:
   pip install vllm

2. Start vLLM server with a vision-language model (default: Qwen2.5-VL):
   # For 7B model (recommended for dual RTX 4090):
   vllm serve Qwen/Qwen2.5-VL-7B-Instruct --host 0.0.0.0 --port 8000 --data-parallel-size 2

   # For 3B model (if 7B doesn't fit):
   vllm serve Qwen/Qwen2.5-VL-3B-Instruct --host 0.0.0.0 --port 8000 --data-parallel-size 2

Usage:
    python vllm_local_labeling.py <candidates_directory> --output-dir <output_dir>
    python vllm_local_labeling.py /path/to/candidates --output-dir /path/to/results --max-images 10
"""

import json
import os
import time
from typing import List, Dict, Any, Optional
import requests
from datetime import datetime
import sys
import argparse
import subprocess
import psutil

from hero_images.image_processor import ImageProcessor


def sanitize_model_name(model_name: str) -> str:
    """Sanitize model name for use in filenames by replacing problematic characters."""
    return model_name.replace(':', '-').replace('/', '-')


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


class GPUChecker:
    """Check GPU memory and suggest appropriate model."""

    @staticmethod
    def get_gpu_memory() -> List[Dict[str, Any]]:
        """Get GPU memory information."""
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,memory.free',
                                   '--format=csv,noheader,nounits'],
                                  capture_output=True, text=True, check=True)

            gpus = []
            for line in result.stdout.strip().split('\n'):
                if line.strip():
                    name, total, free = line.split(', ')
                    gpus.append({
                        'name': name.strip(),
                        'total_mb': int(total.strip()),
                        'free_mb': int(free.strip())
                    })
            return gpus
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("⚠️  Warning: Could not detect GPU information via nvidia-smi")
            return []

    @staticmethod
    def suggest_model() -> str:
        """Suggest the best model based on available GPU memory."""
        gpus = GPUChecker.get_gpu_memory()

        if not gpus:
            print("⚠️  No GPU information available, defaulting to 3B model")
            return "Qwen/Qwen2.5-VL-3B-Instruct"

        total_memory = sum(gpu['total_mb'] for gpu in gpus)
        free_memory = sum(gpu['free_mb'] for gpu in gpus)

        print(f"💾 GPU Information:")
        for i, gpu in enumerate(gpus):
            print(f"   GPU {i}: {gpu['name']} - {gpu['total_mb']/1024:.1f}GB total, {gpu['free_mb']/1024:.1f}GB free")
        print(f"   Total VRAM: {total_memory/1024:.1f}GB, Free: {free_memory/1024:.1f}GB")

        # Model memory requirements (rough estimates)
        if total_memory >= 45000:  # ~45GB+ total
            print("✅ Sufficient memory for Qwen2.5-VL-7B-Instruct")
            return "Qwen/Qwen2.5-VL-7B-Instruct"
        elif total_memory >= 20000:  # ~20GB+ total
            print("✅ Sufficient memory for Qwen2.5-VL-3B-Instruct")
            return "Qwen/Qwen2.5-VL-3B-Instruct"
        else:
            print("⚠️  Limited VRAM, using Qwen2.5-VL-3B-Instruct")
            return "Qwen/Qwen2.5-VL-3B-Instruct"


class VLLMProcessor:
    """Handles VLM inference via vLLM server."""

    def __init__(self, server_url: str = "http://localhost:8000", image_size: int = 768):
        """Initialize with vLLM server URL and image size."""
        self.server_url = server_url
        self.model_name = "unknown"  # Will be updated when server is detected
        self.image_size = image_size

        # Create the aesthetic rating prompt (same as Gemini script)
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

    def check_server_health(self) -> bool:
        """Check if vLLM server is running and responsive."""
        try:
            response = requests.get(f"{self.server_url}/health", timeout=5)
            if response.status_code == 200:
                print(f"✅ vLLM server is running at {self.server_url}")
                return True
            else:
                print(f"❌ vLLM server returned status {response.status_code}")
                return False
        except requests.exceptions.RequestException as e:
            print(f"❌ Could not connect to vLLM server at {self.server_url}: {e}")
            return False

    def get_server_info(self) -> Dict[str, Any]:
        """Get information about the running model."""
        try:
            response = requests.get(f"{self.server_url}/v1/models", timeout=10)
            if response.status_code == 200:
                models_data = response.json()
                if 'data' in models_data and models_data['data']:
                    model_info = models_data['data'][0]
                    self.model_name = model_info.get('id', 'unknown')
                    print(f"✅ Connected to model: {self.model_name}")
                    return model_info
            return {}
        except Exception as e:
            print(f"⚠️  Could not get server info: {e}")
            return {}

    def label_image(self, image_path: str) -> Dict[str, Any]:
        """
        Send image to vLLM server for labeling.

        Returns:
            Dictionary with labeling results
        """
        try:
            # Process image
            image_b64 = ImageProcessor.resize_image_to_base64(image_path, self.image_size)
            filename = os.path.basename(image_path)

            # Create custom prompt with filename
            custom_prompt = self.prompt + f"\n\nNote: You are analyzing the image file named '{filename}'. Please include this exact filename in the image_filename field of your JSON response."

            # Prepare request for vLLM OpenAI-compatible API
            payload = {
                "model": self.model_name,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": custom_prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_b64}"
                                }
                            }
                        ]
                    }
                ],
                "max_tokens": 300,
                "temperature": 0.1
            }

            # Send request
            response = requests.post(
                f"{self.server_url}/v1/chat/completions",
                json=payload,
                timeout=60
            )

            if response.status_code != 200:
                return {
                    'image_path': image_path,
                    'image_filename': filename,
                    'success': False,
                    'error': f"HTTP {response.status_code}: {response.text}"
                }

            # Parse response
            response_data = response.json()

            if 'choices' not in response_data or not response_data['choices']:
                return {
                    'image_path': image_path,
                    'image_filename': filename,
                    'success': False,
                    'error': "No response choices returned"
                }

            content = response_data['choices'][0]['message']['content'].strip()

            # Parse JSON response (handle markdown formatting)
            if content.startswith('```json'):
                content = content[7:]
            if content.endswith('```'):
                content = content[:-3]
            content = content.strip()

            try:
                result_json = json.loads(content)

                return {
                    'image_path': image_path,
                    'image_filename': filename,
                    'aesthetic_score': float(result_json.get('score', 0)),
                    'reasoning': result_json.get('reasoning', 'No reasoning provided'),
                    'success': True
                }

            except json.JSONDecodeError as e:
                return {
                    'image_path': image_path,
                    'image_filename': filename,
                    'success': False,
                    'error': f"Failed to parse JSON response: {e}. Raw content: {content[:200]}..."
                }

        except Exception as e:
            return {
                'image_path': image_path,
                'image_filename': filename,
                'success': False,
                'error': str(e)
            }

    def process_images(self, image_paths: List[str], progress_callback=None) -> List[Dict[str, Any]]:
        """Process multiple images with progress tracking."""
        results = []
        total = len(image_paths)

        print(f"Processing {total} images...")

        for i, image_path in enumerate(image_paths):
            if progress_callback:
                progress_callback(i, total)

            result = self.label_image(image_path)
            results.append(result)

            # Progress update
            if (i + 1) % 10 == 0 or i == total - 1:
                successful = len([r for r in results if r['success']])
                print(f"  Processed {i + 1}/{total} images ({successful} successful)")

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
                'processing_method': 'local_vllm',
                'server_url': self.server_url,
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
        """Save results in format compatible with existing visualization."""
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
                'processing_method': 'local_vllm',
                'server_url': self.server_url,
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

        print(f"✅ Results saved to: {output_path}")
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


def print_setup_instructions():
    """Print setup instructions for vLLM server."""
    print("\n📋 Setup Instructions:")
    print("1. Install vLLM:")
    print("   pip install vllm")
    print("\n2. Start vLLM server with recommended model:")

    recommended_model = GPUChecker.suggest_model()

    if "7B" in recommended_model:
        print(f"   vllm serve {recommended_model} --host 0.0.0.0 --port 8000 --data-parallel-size 2")
    else:
        print(f"   vllm serve {recommended_model} --host 0.0.0.0 --port 8000 --data-parallel-size 2")

    print("\n3. Wait for server to load the model (this may take several minutes)")
    print("4. Run this script again")


def main():
    """Main processing workflow."""

    parser = argparse.ArgumentParser(
        description="Local VLM Hero Image Labeling via vLLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test with 5 images
  python vllm_local_labeling.py /path/to/candidates --output-dir /path/to/output --max-images 5

  # Process full dataset with checkpointing
  python vllm_local_labeling.py /path/to/candidates --output-dir /path/to/output

  # Resume from checkpoint
  python vllm_local_labeling.py /path/to/candidates --output-dir /path/to/output --resume /path/to/checkpoint.tmp.json

  # Disable checkpointing
  python vllm_local_labeling.py /path/to/candidates --output-dir /path/to/output --checkpoint-interval 0

  # Custom checkpoint interval
  python vllm_local_labeling.py /path/to/candidates --output-dir /path/to/output --checkpoint-interval 500

  # Check GPU memory and get setup instructions
  python vllm_local_labeling.py --setup-help
        """
    )

    parser.add_argument(
        'source',
        nargs='?',
        help='Directory containing candidate images, text file with image paths (one per line), or JSON file with list of image paths'
    )
    parser.add_argument(
        '--output-dir', '-o',
        help='Output directory for results'
    )
    parser.add_argument(
        '--max-images', '-n',
        type=int,
        help='Maximum number of images to process (for testing)'
    )
    parser.add_argument(
        '--server-url',
        default='http://localhost:8000',
        help='vLLM server URL (default: http://localhost:8000)'
    )
    parser.add_argument(
        '--setup-help',
        action='store_true',
        help='Show setup instructions and exit'
    )
    parser.add_argument(
        '--checkpoint-interval',
        type=int,
        default=1000,
        help='Save checkpoint every N images (default: 1000, use 0 to disable)'
    )
    parser.add_argument(
        '--resume',
        help='Resume from a checkpoint file (path to .tmp.json file)'
    )
    parser.add_argument(
        '--recursive', '-r',
        action='store_true',
        help='Search for images recursively in subdirectories'
    )
    parser.add_argument(
        '--image-size',
        type=int,
        default=768,
        help='Maximum dimension for resized images (default: 768)'
    )

    args = parser.parse_args()

    print("=== Local VLM Hero Image Labeling via vLLM ===")

    # Show setup help if requested
    if args.setup_help:
        print_setup_instructions()
        return

    # Validate arguments
    if not args.source:
        parser.print_help()
        sys.exit(1)

    if not args.output_dir:
        print("❌ Error: --output-dir is required")
        sys.exit(1)

    if not os.path.exists(args.source):
        print(f"❌ Error: Source does not exist: {args.source}")
        sys.exit(1)

    try:
        # Initialize processor
        processor = VLLMProcessor(args.server_url, args.image_size)

        # Check server health
        print("\n1. Checking vLLM server...")
        if not processor.check_server_health():
            print("\n💡 It looks like the vLLM server is not running.")
            print_setup_instructions()
            sys.exit(1)

        # Get server info
        processor.get_server_info()

        # Get image files
        print("\n2. Finding candidate images...")
        image_files = enumerate_image_files(args.source, args.recursive)
        image_files.sort()

        # Limit number of images if specified
        if args.max_images:
            image_files = image_files[:args.max_images]
            print(f"✅ Found {len(image_files)} images to process (limited to {args.max_images})")
        else:
            print(f"✅ Found {len(image_files)} images to process")

        if not image_files:
            print("❌ No images found!")
            return

        # Ensure output directory exists
        os.makedirs(args.output_dir, exist_ok=True)

        # Handle resume logic
        existing_results = []
        processed_filenames = set()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        sanitized_model = sanitize_model_name(processor.model_name)
        output_filename = f"vllm_local_labels_{sanitized_model}_{timestamp}.json"
        output_path = os.path.join(args.output_dir, output_filename)
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
                output_path = os.path.join(args.output_dir, output_filename)
                checkpoint_path = args.resume  # Continue using the same checkpoint file

        # Filter images to only process those not already completed
        images_to_process = []
        for image_path in image_files:
            filename = os.path.basename(image_path)
            if filename not in processed_filenames:
                images_to_process.append(image_path)

        print(f"\n4. Processing images...")
        if args.resume:
            print(f"📥 Resuming: {len(existing_results)} already processed, {len(images_to_process)} remaining")
        else:
            print(f"🆕 Starting fresh: {len(images_to_process)} images to process")

        if not images_to_process:
            print("✅ All images already processed!")
            final_results = existing_results
        else:
            start_time = time.time()

            # Process remaining images with checkpointing
            new_results = []
            all_results = existing_results.copy()

            for i, image_path in enumerate(images_to_process):
                result = processor.label_image(image_path)
                new_results.append(result)
                all_results.append(result)

                # Progress update
                if (i + 1) % 10 == 0 or i == len(images_to_process) - 1:
                    successful = len([r for r in new_results if r['success']])
                    print(f"  Processed {i + 1}/{len(images_to_process)} new images ({successful} successful)")

                # Save checkpoint periodically
                if args.checkpoint_interval > 0 and (i + 1) % args.checkpoint_interval == 0:
                    print(f"💾 Saving checkpoint after {len(all_results)} total images...")
                    processor.save_checkpoint(all_results, checkpoint_path)

            processing_time = time.time() - start_time
            final_results = all_results

        # Save final results
        print(f"\n5. Saving final results...")
        processor.save_results(final_results, output_path)

        # Clean up checkpoint file on successful completion
        if args.checkpoint_interval > 0 and os.path.exists(checkpoint_path):
            try:
                os.remove(checkpoint_path)
                print(f"🧹 Cleaned up checkpoint file: {checkpoint_path}")
            except Exception as e:
                print(f"⚠️  Could not remove checkpoint file: {e}")

        processing_time = processing_time if 'processing_time' in locals() else 0

        # Summary
        successful = len([r for r in final_results if r['success']])
        failed = len([r for r in final_results if not r['success']])

        print(f"\n🎉 Local VLM labeling complete!")
        print(f"✅ Total processed: {len(final_results)} images")
        print(f"✅ Successful: {successful}")
        print(f"❌ Failed: {failed}")
        if processing_time > 0:
            images_processed = len(images_to_process) if 'images_to_process' in locals() else len(final_results)
            if images_processed > 0:
                print(f"⏱️  Processing time: {processing_time:.1f}s ({processing_time/images_processed:.1f}s per image)")
        print(f"📁 Results: {output_path}")

        if successful > 0:
            scores = [r['aesthetic_score'] for r in final_results if r['success']]
            print(f"📊 Score range: {min(scores):.1f} - {max(scores):.1f}")
            print(f"📊 Average score: {sum(scores)/len(scores):.1f}")

        if failed > 0:
            print(f"\n⚠️  {failed} images failed to process. Check the output file for error details.")

    except KeyboardInterrupt:
        print(f"\n⚠️  Process interrupted by user")
        sys.exit(1)

    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()