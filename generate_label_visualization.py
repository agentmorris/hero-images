"""
Generate HTML visualization of Gemini labeling results.
Shows images alongside their aesthetic scores and reasoning.
"""

import json
import os
import base64
import random
from typing import Dict, Any


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


def generate_html_visualization(json_file_path: str, output_html_path: str, sample: int = 500, top_only: bool = False, sort_by: str = 'filename'):
    """Generate HTML file showing images with their Gemini ratings."""

    print(f"Loading results from {json_file_path}...")

    # Load the JSON results
    with open(json_file_path, 'r') as f:
        data = json.load(f)

    labeling_info = data['labeling_info']
    results = data['results']

    print(f"Found {len(results)} labeled images")

    # Create image subfolder
    html_basename = os.path.splitext(os.path.basename(output_html_path))[0]
    images_folder_name = f"{html_basename}_images"
    images_folder_path = os.path.join(os.path.dirname(output_html_path), images_folder_name)
    os.makedirs(images_folder_path, exist_ok=True)

    print(f"Creating image folder: {images_folder_path}")

    # Sort results by score (highest first) for better viewing
    successful_results = [r for r in results if r['success']]
    failed_results = [r for r in results if not r['success']]

    # Check if results were sampled
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
        <title>Gemini Hero Image Labeling Results{display_note}</title>
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
            .error-message {{
                color: #c0392b;
                font-style: italic;
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
            <h1>🎯 Gemini Hero Image Labeling Results</h1>
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
        successful_results.sort(key=lambda x: x['aesthetic_score'], reverse=True)
        failed_results.sort(key=lambda x: x['image_filename'])  # Failed by filename for consistency
    else:  # sort_by == 'filename'
        print(f"Sorting by filename (alphabetical)")
        successful_results.sort(key=lambda x: x['image_filename'])
        failed_results.sort(key=lambda x: x['image_filename'])

    # Apply filtering based on arguments
    if top_only:
        # Show only successful results
        sorted_results = successful_results
        print(f"Filtering to top-scoring images only: {len(sorted_results)} results")
    else:
        # Combine: successful first, then failed (both sorted according to sort_by preference)
        sorted_results = successful_results + failed_results

    # Apply random sampling if specified
    if sample > 0 and len(sorted_results) > sample:
        total_count = len(sorted_results)
        sorted_results = random.sample(sorted_results, sample)
        print(f"Randomly sampling {sample} out of {total_count} results")

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
    args = parser.parse_args()

    # Determine input source
    input_path = args.input_path or args.labels_dir or "/mnt/c/temp/hero-images/labels"

    # Check if input is a file or directory
    if os.path.isfile(input_path):
        # Direct file specified
        if not input_path.endswith('.json'):
            print("❌ Input file must be a JSON file")
            return

        json_path = input_path
        output_dir = os.path.dirname(input_path)
        json_filename = os.path.basename(input_path)

        print(f"Using specific JSON file: {json_filename}")

    elif os.path.isdir(input_path):
        # Directory specified - find most recent file
        labels_dir = input_path

        # Find the most recent JSON file (both synchronous and batch results)
        json_files = [f for f in os.listdir(labels_dir) if
                      (f.startswith('gemini_labels_') or f.startswith('gemini_batch_labels_')) and f.endswith('.json')]

        if not json_files:
            print(f"No Gemini label files found in {labels_dir}!")
            return

        # Sort by filename (timestamp) and get the most recent
        json_files.sort(reverse=True)
        json_filename = json_files[0]
        json_path = os.path.join(labels_dir, json_filename)
        output_dir = labels_dir

        print(f"Using most recent JSON file: {json_filename}")

    else:
        print(f"❌ Input path does not exist: {input_path}")
        return

    # Generate corresponding HTML filename
    html_filename = json_filename.replace('.json', '.html')
    html_path = os.path.join(output_dir, html_filename)

    print(f"Creating HTML visualization for: {json_filename}")

    generate_html_visualization(json_path, html_path, args.sample, args.top_only, args.sort_by)

    print(f"\n✅ Visualization complete!")
    print(f"📁 Open in browser: {html_path}")


if __name__ == "__main__":
    main()