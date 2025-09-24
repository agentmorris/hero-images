import json
import sys

# Read the first 1000 lines to understand structure without loading entire file
with open('/mnt/c/Users/dmorr/postprocessing/snapshot-safari/snapshot-safari-2025-09-19-v1000.0.0-redwood/combined_api_outputs/snapshot-safari-2025-09-19-v1000.0.0-redwood-ensemble_output_modular_image-level.md-format.within_image_smoothing.seqsmoothing.json', 'r') as f:
    lines = []
    for i, line in enumerate(f):
        lines.append(line)
        if i > 1000:
            break
    partial_json = ''.join(lines)

# Try to find a complete image entry
start_idx = partial_json.find('"file"')
if start_idx > 0:
    # Find the complete first image entry
    brace_count = 0
    in_image = False
    image_start = None

    for i, char in enumerate(partial_json[start_idx-20:]):
        if char == '{':
            if not in_image:
                image_start = i + start_idx - 20
                in_image = True
            brace_count += 1
        elif char == '}':
            brace_count -= 1
            if brace_count == 0 and in_image:
                image_end = i + start_idx - 20 + 1
                sample_image = partial_json[image_start:image_end]
                print('Sample image entry:')
                print(sample_image)
                break