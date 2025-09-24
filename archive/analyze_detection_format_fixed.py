import json

# Let's look for entries with classifications and higher confidence detections
detection_file = '/mnt/c/Users/dmorr/postprocessing/snapshot-safari/snapshot-safari-2025-09-19-v1000.0.0-redwood/combined_api_outputs/snapshot-safari-2025-09-19-v1000.0.0-redwood-ensemble_output_modular_image-level.md-format.within_image_smoothing.seqsmoothing.json'

print("Searching for entries with animal detections and classifications...")

entries_found = 0
high_conf_animals = 0
with_classifications = 0

with open(detection_file, 'r') as f:
    # Skip to the images section
    for line in f:
        if '"images"' in line:
            break

    current_entry = ""
    brace_count = 0
    in_entry = False

    for line in f:
        if entries_found >= 50:  # Look at first 50 entries
            break

        for char in line:
            if char == '{':
                if not in_entry:
                    in_entry = True
                    current_entry = "{"
                else:
                    current_entry += char
                brace_count += 1
            elif char == '}':
                current_entry += char
                brace_count -= 1
                if brace_count == 0 and in_entry:
                    # Complete entry found
                    try:
                        entry = json.loads(current_entry)
                        entries_found += 1

                        if 'detections' in entry:
                            for det in entry['detections']:
                                if det['category'] == '1' and det['conf'] >= 0.3:  # Animal with good confidence
                                    high_conf_animals += 1
                                    print(f"\nEntry {entries_found}: {entry['file']}")
                                    print(f"  Animal detection conf: {det['conf']:.3f}")
                                    bbox_size = det['bbox'][2] * det['bbox'][3]  # width * height
                                    print(f"  Bbox: {det['bbox']} (relative size: {bbox_size:.3f})")

                                    if 'classifications' in det:
                                        with_classifications += 1
                                        print(f"  Classifications: {det['classifications']}")
                                        # Let's inspect the structure first
                                        if det['classifications']:
                                            print(f"  First classification type: {type(det['classifications'][0])}")
                                    break
                    except json.JSONDecodeError:
                        pass
                    except Exception as e:
                        print(f"Error processing entry: {e}")

                    # Reset for next entry
                    current_entry = ""
                    in_entry = False
            else:
                if in_entry:
                    current_entry += char

print(f"\nSummary from first {entries_found} entries:")
print(f"High confidence animals (>0.3): {high_conf_animals}")
print(f"Entries with classifications: {with_classifications}")