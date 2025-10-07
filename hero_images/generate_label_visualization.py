"""
Generate HTML visualization of Gemini labeling results.
Shows images alongside their aesthetic scores and reasoning.
"""

import json
import os
import sys
import base64
import random
from typing import Dict, Any, List, Union
from datetime import datetime


def copy_and_resize_image(source_path: str, output_folder: str, filename: str, max_size: int = 400) -> str:
    """Copy and resize image to output folder, return relative path."""
    try:
        from PIL import Image

        # Create safe filename
        safe_filename = filename.replace('#', '_').replace(':', '_')
        output_path = os.path.join(output_folder, safe_filename)

        # Open and resize image
        with Image.open(source_path) as img:
            # Convert to RGB if necessary
            if img.mode != 'RGB':
                img = img.convert('RGB')

            # Calculate new size maintaining aspect ratio
            width, height = img.size
            if width > height:
                new_width = min(max_size, width)
                new_height = int((height * new_width) / width)
            else:
                new_height = min(max_size, height)
                new_width = int((width * new_height) / height)

            # Resize and save
            resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            resized_img.save(output_path, 'JPEG', quality=85, optimize=True)

        # Return relative path for HTML
        return os.path.join(os.path.basename(output_folder), safe_filename)

    except Exception as e:
        print(f"Warning: Could not process image {source_path}: {e}")
        return ""


def get_sample_filenames(sample_from: Union[str, List[str]], sample: int, random_seed: int, recursive: bool = False) -> List[str]:
    """Get a fixed list of filenames to sample from."""
    if isinstance(sample_from, str):
        if os.path.isdir(sample_from):
            # Directory: get all image files
            filenames = []
            if recursive:
                for root, dirs, files in os.walk(sample_from):
                    for f in files:
                        if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                            filenames.append(f)
            else:
                for f in os.listdir(sample_from):
                    if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                        filenames.append(f)
        elif os.path.isfile(sample_from) and sample_from.endswith('.json'):
            # JSON file: extract filenames
            with open(sample_from, 'r') as f:
                data = json.load(f)
            filenames = [r['image_filename'] for r in data['results']]
        else:
            raise ValueError(f"sample_from must be a directory or JSON file: {sample_from}")
    else:
        # List of files
        filenames = [os.path.basename(f) for f in sample_from]

    # Sort for consistency
    filenames.sort()

    # Apply sampling with fixed random seed
    if sample > 0 and len(filenames) > sample:
        random.seed(random_seed)
        filenames = random.sample(filenames, sample)
        # Re-sort after sampling for consistent ordering
        filenames.sort()

    return filenames


def generate_html_visualization_with_shared_images(json_file_path: str, output_html_path: str, shared_images_path: str, shared_images_folder: str, sample: int = 500, top_only: bool = False, sort_by: str = 'filename', random_seed: int = 0, sample_from: Union[str, List[str], None] = None, create_images: bool = True, recursive: bool = False):
    """Generate HTML file using a shared images folder for efficiency."""

    print(f"Loading results from {json_file_path}...")

    # Load the JSON results
    with open(json_file_path, 'r') as f:
        data = json.load(f)

    labeling_info = data.get('labeling_info') or data.get('checkpoint_info', {})
    results = data['results']

    print(f"Found {len(results)} labeled images")

    # If sample_from is specified, filter results to only include those filenames
    if sample_from is not None:
        target_filenames = get_sample_filenames(sample_from, sample, random_seed, recursive)
        print(f"Filtering to {len(target_filenames)} target filenames from sample_from")

        # Create a lookup for existing results
        results_by_filename = {r['image_filename']: r for r in results}

        # Create new results list with target filenames
        filtered_results = []
        for filename in target_filenames:
            if filename in results_by_filename:
                filtered_results.append(results_by_filename[filename])
            else:
                # Create a "missing" result entry
                filtered_results.append({
                    'image_filename': filename,
                    'image_path': '',  # Unknown path
                    'success': False,
                    'error': f'No results available for this image in {os.path.basename(json_file_path)}',
                    'aesthetic_score': 0,
                    'reasoning': f'Image not found in {labeling_info.get("model_used", "this model")} results'
                })

        results = filtered_results
        sample = 0  # Don't apply additional sampling since we already filtered
        print(f"After filtering: {len(results)} results ({len([r for r in results if r['success']])} successful, {len([r for r in results if not r['success']])} missing/failed)")

    # Create shared image subfolder only on first iteration
    if create_images:
        os.makedirs(shared_images_path, exist_ok=True)
        print(f"Creating shared image folder: {shared_images_path}")

    # Sort results by score (highest first) for better viewing
    successful_results = [r for r in results if r['success']]
    failed_results = [r for r in results if not r['success']]

    # Check whether results were sampled
    display_note = ""
    if sample > 0 and len(results) > sample:
        display_note = f" (showing random sample of {sample} from {len(results)})"
    elif top_only and failed_results:
        display_note = f" (showing {len(successful_results)} successful results only)"

    # Start building HTML
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Hero Image Labeling Results{display_note}</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
                background-color: #f5f5f5;
            }}
            .header {{
                background: white;
                padding: 20px;
                border-radius: 8px;
                margin-bottom: 20px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            .stats {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 10px;
                margin-top: 15px;
            }}
            .stat {{
                background: #f8f9fa;
                padding: 10px;
                border-radius: 4px;
                text-align: center;
            }}
            .stat-value {{
                font-size: 24px;
                font-weight: bold;
                color: #2c3e50;
            }}
            .stat-label {{
                font-size: 12px;
                color: #666;
                text-transform: uppercase;
            }}
            .image-result {{
                background: white;
                border-radius: 8px;
                margin-bottom: 20px;
                padding: 20px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                display: grid;
                grid-template-columns: 400px 1fr;
                gap: 20px;
                align-items: start;
            }}
            .image-container {{
                text-align: center;
            }}
            .image-container img {{
                max-width: 100%;
                max-height: 300px;
                border-radius: 4px;
                border: 1px solid #ddd;
            }}
            .image-filename {{
                font-size: 12px;
                color: #666;
                margin-top: 8px;
                word-break: break-all;
            }}
            .result-info {{
                padding: 10px 0;
            }}
            .score {{
                font-size: 48px;
                font-weight: bold;
                margin-bottom: 10px;
                padding: 20px;
                text-align: center;
                border-radius: 8px;
                color: white;
            }}
            .score-0 {{ background: #e74c3c; }}
            .score-1 {{ background: #e67e22; }}
            .score-2 {{ background: #f39c12; }}
            .score-3 {{ background: #f1c40f; color: #333; }}
            .score-4 {{ background: #9b59b6; }}
            .score-5 {{ background: #3498db; }}
            .score-6 {{ background: #1abc9c; }}
            .score-7 {{ background: #2ecc71; }}
            .score-8 {{ background: #27ae60; }}
            .score-9 {{ background: #229954; }}
            .score-10 {{ background: #1e8449; }}
            .reasoning {{
                background: #f8f9fa;
                padding: 15px;
                border-radius: 4px;
                border-left: 4px solid #3498db;
                line-height: 1.5;
            }}
            .processing-time {{
                font-size: 12px;
                color: #666;
                margin-top: 10px;
            }}
            .failed {{
                background: #ffebee;
                border-left-color: #e74c3c;
            }}
            .failed .score {{
                background: #e74c3c;
                font-size: 24px;
            }}
            .missing {{
                background: #f8f9fa;
                border-left-color: #6c757d;
            }}
            .missing .score {{
                background: #6c757d;
                font-size: 18px;
            }}
            .error-message {{
                color: #c0392b;
                font-style: italic;
            }}
            .missing .error-message {{
                color: #6c757d;
            }}
            @media (max-width: 768px) {{
                .image-result {{
                    grid-template-columns: 1fr;
                }}
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🎯 Hero Image Labeling Results</h1>
            <p><strong>Generated:</strong> {labeling_info['timestamp']}</p>
            <p><strong>Model:</strong> {labeling_info['model_used']}</p>

            <div class="stats">
                <div class="stat">
                    <div class="stat-value">{labeling_info['total_images']}</div>
                    <div class="stat-label">Total Images</div>
                </div>
                <div class="stat">
                    <div class="stat-value">{labeling_info['successful_labels']}</div>
                    <div class="stat-label">Successful</div>
                </div>
                <div class="stat">
                    <div class="stat-value">{labeling_info['success_rate']:.1f}%</div>
                    <div class="stat-label">Success Rate</div>
                </div>
                <div class="stat">
                    <div class="stat-value">{labeling_info['statistics']['avg_score']:.1f}</div>
                    <div class="stat-label">Avg Score</div>
                </div>
                <div class="stat">
                    <div class="stat-value">{labeling_info['statistics']['min_score']:.0f} - {labeling_info['statistics']['max_score']:.0f}</div>
                    <div class="stat-label">Score Range</div>
                </div>
            </div>
        </div>
    """

    # Add processing time stat if available
    if 'avg_processing_time' in labeling_info['statistics']:
        processing_time_stat = f"""
                <div class="stat">
                    <div class="stat-value">{labeling_info['statistics']['avg_processing_time']:.1f}s</div>
                    <div class="stat-label">Avg Time</div>
                </div>"""
        # Insert before the closing tags
        html_content = html_content.replace("            </div>\n        </div>\n    \"\"\"", f"            {processing_time_stat}\n            </div>\n        </div>\n    \"\"\"")

    # Sort results based on user preference
    if sort_by == 'score':
        print(f"Sorting by score (highest to lowest)")
        # For score sorting: successful first (by score), then failed (by filename)
        successful_results.sort(key=lambda x: x['aesthetic_score'], reverse=True)
        failed_results.sort(key=lambda x: x['image_filename'])
        # Apply filtering based on arguments
        if top_only:
            sorted_results = successful_results
            print(f"Filtering to top-scoring images only: {len(sorted_results)} results")
        else:
            sorted_results = successful_results + failed_results
    else:  # sort_by == 'filename'
        print(f"Sorting by filename (alphabetical)")
        # For filename sorting: sort ALL results together by filename, regardless of success
        all_results = successful_results + failed_results
        all_results.sort(key=lambda x: x['image_filename'])
        # Apply filtering based on arguments
        if top_only:
            sorted_results = [r for r in all_results if r['success']]
            print(f"Filtering to top-scoring images only: {len(sorted_results)} results")
        else:
            sorted_results = all_results

    # Apply random sampling if specified (only if sample_from wasn't used)
    if sample > 0 and len(sorted_results) > sample and sample_from is None:
        # Set random seed for reproducible sampling
        random.seed(random_seed)
        total_count = len(sorted_results)
        sorted_results = random.sample(sorted_results, sample)
        print(f"Randomly sampling {sample} out of {total_count} results (seed: {random_seed})")

        # Re-apply sorting to the sampled results
        if sort_by == 'score':
            print(f"Re-sorting sampled results by score (highest to lowest)")
            # Separate successful and failed in the sample
            sampled_successful = [r for r in sorted_results if r['success']]
            sampled_failed = [r for r in sorted_results if not r['success']]
            sampled_successful.sort(key=lambda x: x['aesthetic_score'], reverse=True)
            sampled_failed.sort(key=lambda x: x['image_filename'])
            sorted_results = sampled_successful + sampled_failed
        else:  # sort_by == 'filename'
            print(f"Re-sorting sampled results by filename (alphabetical)")
            sorted_results.sort(key=lambda x: x['image_filename'])

    # Add each image result
    for i, result in enumerate(sorted_results):
        print(f"Processing image {i+1}/{len(sorted_results)}: {result['image_filename']}")

        # Copy and resize image to shared folder (only if create_images is True)
        image_rel_path = ""
        if create_images:
            image_rel_path = copy_and_resize_image(result['image_path'], shared_images_path, result['image_filename'])
        else:
            # Use existing image in shared folder - construct path same way as copy_and_resize_image
            safe_filename = result['image_filename'].replace('#', '_').replace(':', '_')
            expected_image_path = os.path.join(shared_images_path, safe_filename)

            # Check whether image exists in shared folder, if not create it
            if os.path.exists(expected_image_path):
                image_rel_path = os.path.join(os.path.basename(shared_images_path), safe_filename)
            else:
                # Image missing from shared folder - create it now
                print(f"  Creating missing image: {result['image_filename']}")
                image_rel_path = copy_and_resize_image(result['image_path'], shared_images_path, result['image_filename'])

        if result['success']:
            score = result['aesthetic_score']
            score_class = f"score-{int(score)}"
            score_display = f"{score:.1f}"
            reasoning = result['reasoning']
            additional_class = ""
        else:
            score_class = "score-0"
            if 'error' in result:
                score_display = "Unavailable"
                reasoning = result['error']
                additional_class = "missing"
            else:
                score_display = "FAILED"
                reasoning = result.get('error_message', 'Processing failed')
                additional_class = "failed"

        processing_time = result.get('processing_time', 0.0)

        html_content += f"""
        <div class="image-result">
            <div class="image-container">
        """

        if image_rel_path and (create_images or os.path.exists(os.path.join(os.path.dirname(output_html_path), image_rel_path))):
            html_content += f'<img src="{image_rel_path}" alt="{result["image_filename"]}">'
        else:
            html_content += '<div style="background: #f0f0f0; height: 200px; display: flex; align-items: center; justify-content: center; color: #666;">Image not found</div>'

        html_content += f"""
                <div class="image-filename">{result['image_filename']}</div>
            </div>
            <div class="result-info">
                <div class="score {score_class}">{score_display}</div>
                <div class="reasoning {additional_class}">
        """

        if result['success']:
            html_content += f"<strong>Reasoning:</strong><br>{reasoning}"
        elif 'error' in result:
            html_content += f'<strong>Missing:</strong><br><span class="error-message">{reasoning}</span>'
        else:
            html_content += f'<strong>Error:</strong><br><span class="error-message">{reasoning}</span>'

        html_content += f"""
                </div>
                {"<div class=\"processing-time\">Processing time: " + f"{processing_time:.1f}s" + "</div>" if processing_time > 0 else ""}
            </div>
        </div>
        """

    # Close HTML
    html_content += """
    </body>
    </html>
    """

    # Write HTML file
    with open(output_html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"\nHTML visualization saved to: {output_html_path}")
    return output_html_path


def generate_html_visualization(json_file_path: str, output_html_path: str, sample: int = 500, top_only: bool = False, sort_by: str = 'filename', random_seed: int = 0, sample_from: Union[str, List[str], None] = None, recursive: bool = False):
    """Generate HTML file showing images with their Gemini ratings."""

    print(f"Loading results from {json_file_path}...")

    # Load the JSON results
    with open(json_file_path, 'r') as f:
        data = json.load(f)

    labeling_info = data['labeling_info']
    results = data['results']

    print(f"Found {len(results)} labeled images")

    # If sample_from is specified, filter results to only include those filenames
    if sample_from is not None:
        target_filenames = get_sample_filenames(sample_from, sample, random_seed, recursive)
        print(f"Filtering to {len(target_filenames)} target filenames from sample_from")

        # Create a lookup for existing results
        results_by_filename = {r['image_filename']: r for r in results}

        # Create new results list with target filenames
        filtered_results = []
        for filename in target_filenames:
            if filename in results_by_filename:
                filtered_results.append(results_by_filename[filename])
            else:
                # Create a "missing" result entry
                filtered_results.append({
                    'image_filename': filename,
                    'image_path': '',  # Unknown path
                    'success': False,
                    'error': f'No results available for this image in {os.path.basename(json_file_path)}',
                    'aesthetic_score': 0,
                    'reasoning': f'Image not found in {labeling_info.get("model_used", "this model")} results'
                })

        results = filtered_results
        sample = 0  # Don't apply additional sampling since we already filtered
        print(f"After filtering: {len(results)} results ({len([r for r in results if r['success']])} successful, {len([r for r in results if not r['success']])} missing/failed)")

    # Create image subfolder
    html_basename = os.path.splitext(os.path.basename(output_html_path))[0]
    images_folder_name = f"{html_basename}_images"
    images_folder_path = os.path.join(os.path.dirname(output_html_path), images_folder_name)
    os.makedirs(images_folder_path, exist_ok=True)

    print(f"Creating image folder: {images_folder_path}")

    # Sort results by score (highest first) for better viewing
    successful_results = [r for r in results if r['success']]
    failed_results = [r for r in results if not r['success']]

    # Check whether results were sampled
    display_note = ""
    if sample > 0 and len(results) > sample:
        display_note = f" (showing random sample of {sample} from {len(results)})"
    elif top_only and failed_results:
        display_note = f" (showing {len(successful_results)} successful results only)"

    # Start building HTML
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Hero Image Labeling Results{display_note}</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
                background-color: #f5f5f5;
            }}
            .header {{
                background: white;
                padding: 20px;
                border-radius: 8px;
                margin-bottom: 20px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            .stats {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 10px;
                margin-top: 15px;
            }}
            .stat {{
                background: #f8f9fa;
                padding: 10px;
                border-radius: 4px;
                text-align: center;
            }}
            .stat-value {{
                font-size: 24px;
                font-weight: bold;
                color: #2c3e50;
            }}
            .stat-label {{
                font-size: 12px;
                color: #666;
                text-transform: uppercase;
            }}
            .image-result {{
                background: white;
                border-radius: 8px;
                margin-bottom: 20px;
                padding: 20px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                display: grid;
                grid-template-columns: 400px 1fr;
                gap: 20px;
                align-items: start;
            }}
            .image-container {{
                text-align: center;
            }}
            .image-container img {{
                max-width: 100%;
                max-height: 300px;
                border-radius: 4px;
                border: 1px solid #ddd;
            }}
            .image-filename {{
                font-size: 12px;
                color: #666;
                margin-top: 8px;
                word-break: break-all;
            }}
            .result-info {{
                padding: 10px 0;
            }}
            .score {{
                font-size: 48px;
                font-weight: bold;
                margin-bottom: 10px;
                padding: 20px;
                text-align: center;
                border-radius: 8px;
                color: white;
            }}
            .score-0 {{ background: #e74c3c; }}
            .score-1 {{ background: #e67e22; }}
            .score-2 {{ background: #f39c12; }}
            .score-3 {{ background: #f1c40f; color: #333; }}
            .score-4 {{ background: #9b59b6; }}
            .score-5 {{ background: #3498db; }}
            .score-6 {{ background: #1abc9c; }}
            .score-7 {{ background: #2ecc71; }}
            .score-8 {{ background: #27ae60; }}
            .score-9 {{ background: #229954; }}
            .score-10 {{ background: #1e8449; }}
            .reasoning {{
                background: #f8f9fa;
                padding: 15px;
                border-radius: 4px;
                border-left: 4px solid #3498db;
                line-height: 1.5;
            }}
            .processing-time {{
                font-size: 12px;
                color: #666;
                margin-top: 10px;
            }}
            .failed {{
                background: #ffebee;
                border-left-color: #e74c3c;
            }}
            .failed .score {{
                background: #e74c3c;
                font-size: 24px;
            }}
            .missing {{
                background: #f8f9fa;
                border-left-color: #6c757d;
            }}
            .missing .score {{
                background: #6c757d;
                font-size: 18px;
            }}
            .error-message {{
                color: #c0392b;
                font-style: italic;
            }}
            .missing .error-message {{
                color: #6c757d;
            }}
            @media (max-width: 768px) {{
                .image-result {{
                    grid-template-columns: 1fr;
                }}
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🎯 Hero Image Labeling Results</h1>
            <p><strong>Generated:</strong> {labeling_info['timestamp']}</p>
            <p><strong>Model:</strong> {labeling_info['model_used']}</p>

            <div class="stats">
                <div class="stat">
                    <div class="stat-value">{labeling_info['total_images']}</div>
                    <div class="stat-label">Total Images</div>
                </div>
                <div class="stat">
                    <div class="stat-value">{labeling_info['successful_labels']}</div>
                    <div class="stat-label">Successful</div>
                </div>
                <div class="stat">
                    <div class="stat-value">{labeling_info['success_rate']:.1f}%</div>
                    <div class="stat-label">Success Rate</div>
                </div>
                <div class="stat">
                    <div class="stat-value">{labeling_info['statistics']['avg_score']:.1f}</div>
                    <div class="stat-label">Avg Score</div>
                </div>
                <div class="stat">
                    <div class="stat-value">{labeling_info['statistics']['min_score']:.0f} - {labeling_info['statistics']['max_score']:.0f}</div>
                    <div class="stat-label">Score Range</div>
                </div>
            </div>
        </div>
    """

    # Add processing time stat if available
    if 'avg_processing_time' in labeling_info['statistics']:
        processing_time_stat = f"""
                <div class="stat">
                    <div class="stat-value">{labeling_info['statistics']['avg_processing_time']:.1f}s</div>
                    <div class="stat-label">Avg Time</div>
                </div>"""
        # Insert before the closing tags
        html_content = html_content.replace("            </div>\n        </div>\n    \"\"\"", f"            {processing_time_stat}\n            </div>\n        </div>\n    \"\"\"")

    # Sort results based on user preference
    if sort_by == 'score':
        print(f"Sorting by score (highest to lowest)")
        # For score sorting: successful first (by score), then failed (by filename)
        successful_results.sort(key=lambda x: x['aesthetic_score'], reverse=True)
        failed_results.sort(key=lambda x: x['image_filename'])
        # Apply filtering based on arguments
        if top_only:
            sorted_results = successful_results
            print(f"Filtering to top-scoring images only: {len(sorted_results)} results")
        else:
            sorted_results = successful_results + failed_results
    else:  # sort_by == 'filename'
        print(f"Sorting by filename (alphabetical)")
        # For filename sorting: sort ALL results together by filename, regardless of success
        all_results = successful_results + failed_results
        all_results.sort(key=lambda x: x['image_filename'])
        # Apply filtering based on arguments
        if top_only:
            sorted_results = [r for r in all_results if r['success']]
            print(f"Filtering to top-scoring images only: {len(sorted_results)} results")
        else:
            sorted_results = all_results

    # Apply random sampling if specified (only if sample_from wasn't used)
    if sample > 0 and len(sorted_results) > sample and sample_from is None:
        # Set random seed for reproducible sampling
        random.seed(random_seed)
        total_count = len(sorted_results)
        sorted_results = random.sample(sorted_results, sample)
        print(f"Randomly sampling {sample} out of {total_count} results (seed: {random_seed})")

        # Re-apply sorting to the sampled results
        if sort_by == 'score':
            print(f"Re-sorting sampled results by score (highest to lowest)")
            # Separate successful and failed in the sample
            sampled_successful = [r for r in sorted_results if r['success']]
            sampled_failed = [r for r in sorted_results if not r['success']]
            sampled_successful.sort(key=lambda x: x['aesthetic_score'], reverse=True)
            sampled_failed.sort(key=lambda x: x['image_filename'])
            sorted_results = sampled_successful + sampled_failed
        else:  # sort_by == 'filename'
            print(f"Re-sorting sampled results by filename (alphabetical)")
            sorted_results.sort(key=lambda x: x['image_filename'])

    # Add each image result
    for i, result in enumerate(sorted_results):
        print(f"Processing image {i+1}/{len(sorted_results)}: {result['image_filename']}")

        # Copy and resize image to subfolder
        image_rel_path = copy_and_resize_image(result['image_path'], images_folder_path, result['image_filename'])

        if result['success']:
            score = result['aesthetic_score']
            score_class = f"score-{int(score)}"
            score_display = f"{score:.1f}"
            reasoning = result['reasoning']
            additional_class = ""
        else:
            score_class = "score-0"
            if 'error' in result:
                score_display = "Unavailable"
                reasoning = result['error']
                additional_class = "missing"
            else:
                score_display = "FAILED"
                reasoning = result.get('error_message', 'Processing failed')
                additional_class = "failed"

        processing_time = result.get('processing_time', 0.0)

        html_content += f"""
        <div class="image-result">
            <div class="image-container">
        """

        if image_rel_path:
            html_content += f'<img src="{image_rel_path}" alt="{result["image_filename"]}">'
        else:
            html_content += '<div style="background: #f0f0f0; height: 200px; display: flex; align-items: center; justify-content: center; color: #666;">Image not found</div>'

        html_content += f"""
                <div class="image-filename">{result['image_filename']}</div>
            </div>
            <div class="result-info">
                <div class="score {score_class}">{score_display}</div>
                <div class="reasoning {additional_class}">
        """

        if result['success']:
            html_content += f"<strong>Reasoning:</strong><br>{reasoning}"
        elif 'error' in result:
            html_content += f'<strong>Missing:</strong><br><span class="error-message">{reasoning}</span>'
        else:
            html_content += f'<strong>Error:</strong><br><span class="error-message">{reasoning}</span>'

        html_content += f"""
                </div>
                {"<div class=\"processing-time\">Processing time: " + f"{processing_time:.1f}s" + "</div>" if processing_time > 0 else ""}
            </div>
        </div>
        """

    # Close HTML
    html_content += """
    </body>
    </html>
    """

    # Write HTML file
    with open(output_html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"\nHTML visualization saved to: {output_html_path}")
    return output_html_path


def generate_index_html(index_path: str, html_files_info: List[Dict[str, Any]], args):
    """Generate index HTML page with links to all individual visualizations."""

    # Calculate some summary stats
    total_files = len(html_files_info)

    # Sample information for display
    sample_info = ""
    if args.sample_from:
        sample_info += f"<strong>Sample source:</strong> {args.sample_from}<br>"
    if args.sample > 0:
        sample_info += f"<strong>Sample size:</strong> {args.sample} images<br>"
    if args.random_seed is not None:
        sample_info += f"<strong>Random seed:</strong> {args.random_seed}<br>"
    sample_info += f"<strong>Sort by:</strong> {args.sort_by}<br>"
    if args.top_only:
        sample_info += f"<strong>Filter:</strong> Top-scoring images only<br>"

    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Hero Image Labeling Results - Model Comparison</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                max-width: 1000px;
                margin: 0 auto;
                padding: 20px;
                background-color: #f5f5f5;
            }}
            .header {{
                background: white;
                padding: 30px;
                border-radius: 8px;
                margin-bottom: 30px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                text-align: center;
            }}
            .subtitle {{
                color: #666;
                font-size: 18px;
                margin-top: 10px;
            }}
            .info-section {{
                background: white;
                padding: 20px;
                border-radius: 8px;
                margin-bottom: 20px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            .models-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                gap: 20px;
                margin-top: 20px;
            }}
            .model-card {{
                background: white;
                border-radius: 8px;
                padding: 20px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                transition: transform 0.2s, box-shadow 0.2s;
            }}
            .model-card:hover {{
                transform: translateY(-2px);
                box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            }}
            .model-title {{
                font-size: 20px;
                font-weight: bold;
                margin-bottom: 10px;
                color: #2c3e50;
            }}
            .model-stats {{
                display: grid;
                grid-template-columns: repeat(3, 1fr);
                gap: 10px;
                margin: 15px 0;
            }}
            .stat {{
                text-align: center;
                padding: 8px;
                background: #f8f9fa;
                border-radius: 4px;
            }}
            .stat-value {{
                font-size: 18px;
                font-weight: bold;
                color: #2c3e50;
            }}
            .stat-label {{
                font-size: 12px;
                color: #666;
                text-transform: uppercase;
            }}
            .view-button {{
                display: inline-block;
                background: #3498db;
                color: white;
                padding: 12px 24px;
                text-decoration: none;
                border-radius: 6px;
                font-weight: bold;
                transition: background 0.2s;
                text-align: center;
                width: 100%;
                box-sizing: border-box;
            }}
            .view-button:hover {{
                background: #2980b9;
            }}
            .processing-method {{
                font-size: 12px;
                color: #666;
                margin-bottom: 10px;
            }}
            .instructions {{
                background: #e8f4fd;
                border: 1px solid #bee5eb;
                border-radius: 6px;
                padding: 15px;
                margin: 20px 0;
            }}
            .instructions h3 {{
                margin-top: 0;
                color: #0c5460;
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🎯 Hero Image Labeling Results</h1>
            <div class="subtitle">Model Comparison Dashboard</div>
            <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>

        <div class="info-section">
            <h2>📊 Comparison Parameters</h2>
            {sample_info}
            <strong>Models compared:</strong> {total_files}
        </div>

        <div class="instructions">
            <h3>💡 How to Compare Models</h3>
            <p>Click the links below to open each model's results in a new tab. Since all models used identical sampling parameters, you can compare them side-by-side by scrolling through the results in each tab.</p>
        </div>

        <div class="models-grid">
    """

    # Add a card for each model
    for file_info in html_files_info:
        labeling_info = file_info['labeling_info']

        total_images = labeling_info.get('total_images', 0)
        successful_labels = labeling_info.get('successful_labels', 0)
        success_rate = labeling_info.get('success_rate', 0)
        avg_score = labeling_info.get('statistics', {}).get('avg_score', 0)

        html_content += f"""
            <div class="model-card">
                <div class="model-title">{file_info['link_text']}</div>
                <div class="processing-method">
                    Timestamp: {labeling_info.get('timestamp', 'Unknown')}
                </div>

                <div class="model-stats">
                    <div class="stat">
                        <div class="stat-value">{total_images}</div>
                        <div class="stat-label">Total Images</div>
                    </div>
                    <div class="stat">
                        <div class="stat-value">{success_rate:.1f}%</div>
                        <div class="stat-label">Success Rate</div>
                    </div>
                    <div class="stat">
                        <div class="stat-value">{avg_score:.1f}</div>
                        <div class="stat-label">Avg Score</div>
                    </div>
                </div>

                <a href="{file_info['html_filename']}" target="_blank" class="view-button">
                    📋 View Results
                </a>
            </div>
        """

    html_content += """
        </div>
    </body>
    </html>
    """

    # Write HTML file
    with open(index_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"📄 Index page saved to: {index_path}")


def main():
    """Generate HTML visualization for the most recent Gemini results."""

    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Generate HTML visualization of Gemini labeling results")
    parser.add_argument(
        'input_path',
        nargs='?',
        help='JSON file or directory containing label files'
    )
    parser.add_argument(
        '--labels-dir', '-d',
        help='Directory containing label JSON files (legacy option, use input_path instead)'
    )
    parser.add_argument(
        '--sample', '-s',
        type=int,
        default=500,
        help='Number of images to randomly sample for visualization (default: 500, use 0 for no sampling)'
    )
    parser.add_argument(
        '--top-only',
        action='store_true',
        help='Show only highest-scoring images (ignores failed results)'
    )
    parser.add_argument(
        '--sort-by',
        choices=['filename', 'score'],
        default='filename',
        help='Sort results by filename (default) or score (highest to lowest)'
    )
    parser.add_argument(
        '--random-seed',
        type=int,
        default=0,
        help='Random seed for reproducible sampling (default: 0)'
    )
    parser.add_argument(
        '--sample-from',
        help='Directory or JSON file to sample filenames from (for consistent comparison across models)'
    )
    parser.add_argument(
        '--recursive', '-r',
        action='store_true',
        help='Search for images recursively in subdirectories (only used with --sample-from)'
    )
    parser.add_argument(
        '--output-dir', '-o',
        help='Output directory for HTML files and images (default: same directory as input JSON files)'
    )
    args = parser.parse_args()

    # Determine input source
    input_path = args.input_path or args.labels_dir

    if not input_path:
        parser.print_help()
        sys.exit(1)

    # Check whether input is a file or directory
    if os.path.isfile(input_path):
        # Direct file specified
        if not input_path.endswith('.json'):
            print("❌ Input file must be a JSON file")
            return

        json_path = input_path
        # Use output_dir if specified, otherwise use same directory as input
        output_dir = args.output_dir if args.output_dir else os.path.dirname(input_path)
        # Create output directory if it doesn't exist
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        json_filename = os.path.basename(input_path)

        print(f"Using specific JSON file: {json_filename}")
        if args.output_dir:
            print(f"Output directory: {output_dir}")

    elif os.path.isdir(input_path):
        # Directory specified - find all label files and generate index
        labels_dir = input_path

        # Find all label JSON files
        json_files = [f for f in os.listdir(labels_dir) if
                      '_labels_' in f and f.endswith('.json')]

        if not json_files:
            print(f"❌ No label files (*_labels_*.json) found in {labels_dir}!")
            return

        json_files.sort()  # Sort alphabetically for consistent ordering
        print(f"Found {len(json_files)} label files to process")

        # Use output_dir if specified, otherwise use same directory as input
        output_dir = args.output_dir if args.output_dir else labels_dir
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        if args.output_dir:
            print(f"Output directory: {output_dir}")

        # Create shared timestamp and images folder for all visualizations
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        shared_images_folder = f"batch_labels_{timestamp}_images"
        shared_images_path = os.path.join(output_dir, shared_images_folder)

        # Generate individual HTML files for each JSON file
        html_files_info = []
        for i, json_filename in enumerate(json_files):
            json_path = os.path.join(labels_dir, json_filename)
            html_filename = json_filename.replace('.json', '.html')
            html_path = os.path.join(output_dir, html_filename)

            print(f"Generating visualization for: {json_filename}")

            # Use shared images folder for all visualizations
            generate_html_visualization_with_shared_images(
                json_path, html_path, shared_images_path, shared_images_folder,
                args.sample, args.top_only, args.sort_by, args.random_seed,
                args.sample_from, create_images=(i == 0), recursive=args.recursive  # Only create images for first iteration
            )

            # Extract model info for index
            with open(json_path, 'r') as f:
                data = json.load(f)

            labeling_info = data.get('labeling_info') or data.get('checkpoint_info', {})
            model_used = labeling_info.get('model_used', 'Unknown')
            processing_method = labeling_info.get('processing_method', 'unknown')

            # Format link text: "Method: model"
            if processing_method == 'local_ollama':
                link_text = f"Ollama: {model_used}"
            elif processing_method == 'local_vllm':
                link_text = f"vLLM: {model_used}"
            elif processing_method == 'gemini_batch':
                link_text = f"Gemini Batch: {model_used}"
            elif processing_method == 'gemini_sync':
                link_text = f"Gemini: {model_used}"
            else:
                link_text = f"{processing_method}: {model_used}"

            html_files_info.append({
                'html_filename': html_filename,
                'link_text': link_text,
                'labeling_info': labeling_info
            })

        # Generate index HTML file (using same timestamp)
        index_filename = f"index.{timestamp}.html"
        index_path = os.path.join(output_dir, index_filename)

        generate_index_html(index_path, html_files_info, args)

        print(f"\n✅ Generated {len(json_files)} visualizations and index")
        print(f"📁 Open index file: {index_path}")
        return

    else:
        print(f"❌ Input path does not exist: {input_path}")
        return

    # Generate corresponding HTML filename
    html_filename = json_filename.replace('.json', '.html')
    html_path = os.path.join(output_dir, html_filename)

    print(f"Creating HTML visualization for: {json_filename}")

    generate_html_visualization(json_path, html_path, args.sample, args.top_only, args.sort_by, args.random_seed, args.sample_from, args.recursive)

    print(f"\n✅ Visualization complete!")
    print(f"📁 Open in browser: {html_path}")


if __name__ == "__main__":
    main()
