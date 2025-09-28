# Hero Image Classification Project

This is an exploratory project comparing methods for identifying "hero images" from camera trap datasets, i.e., aesthetically pleasing wildlife photos.


## Project overview

The system will process large camera trap collections to identify candidates with aesthetic appeal through a two-stage pipeline:

1. **Candidate selection**: Heuristic-based filtering using AI detection results (typically MegaDetector and SpeciesNet) to identify promising images
2. **Labeling**: LLM aesthetic rating using Gemini 2.5 Flash or local VLMs


### Scripts

- **`generate_sequence_aware_candidates_optimized.py`** - Generate candidates for labeling using heuristics
- **`gemini_batch_labeling.py`** - Asynchronous API labeling using the Gemini batch API
- **`vllm_local_labeling.py`** - Local VLM labeling via vLLM (supports Qwen2.5-VL and other models, no API costs)
- **`generate_label_visualization.py`** - Create HTML visualizations compatible with all labeling results


### Modules

- **`stratified_selector_sequence_aware.py`** - Main candidate selection system with sequence awareness



## Usage

### Setup

**For Gemini labeling**

```bash
pip install -r requirements.txt
echo "your-gemini-api-key" > GEMINI_API_KEY.txt
```

**For local VLM labeling**

```bash
pip install -r requirements.txt
pip install vllm
```

### Select candidates for LLM labeling

```bash
python3 generate_sequence_aware_candidates_optimized.py
```

### Label images

** Local VLM **

```bash
# Check GPU memory and get setup instructions
python vllm_local_labeling.py --setup-help

# Start vLLM server (example for dual RTX 4090)
vllm serve Qwen/Qwen2.5-VL-7B-Instruct \
  --host 0.0.0.0 \
  --port 8000 \
  --data-parallel-size 2 \
  --gpu-memory-utilization 0.9 \
  --max-model-len 60000

# Run labeling (in another terminal)
python vllm_local_labeling.py /path/to/candidates --output-dir /path/to/output
```

** Gemini Batch API **

```bash
python gemini_batch_labeling.py /path/to/candidates --output-dir /path/to/output
```

### 4. Visualize results

```bash
python generate_label_visualization.py /path/to/gemini_batch_labels_20250923_143022.json
python generate_label_visualization.py /path/to/vllm_local_labels_20250927_143022.json
```


## Gemini batch job management

### Cancel a running job

If you need to stop a batch job (e.g., if it's taking too long or you made an error):

```bash
# When you interrupt polling with Ctrl+C, the script shows the cancel command:
python gemini_batch_labeling.py --cancel batches/xyz789
```

Ctrl+C only stops the local script - the job continues running on Google's servers until cancelled.

### Resume jobs (running or completed)

If your script was interrupted or you want to retrieve results from a completed job:

```bash
python gemini_batch_labeling.py --resume /path/to/gemini_batch_metadata_YYYYMMDD_HHMMSS.json
```

Resume behavior:

- **Running jobs**: Continues polling until completion
- **Completed jobs**: Immediately retrieves and saves results
- **Failed/cancelled jobs**: Shows status and exits


## Data pipeline

```
Raw images
    ↓ (MegaDetector + SpeciesNet)
Detection and classification results
    ↓ (Sequence-aware stratified sampling)
Candidates (N diverse images)
    ↓ (Gemini 2.5 Flash or Local VLM Labeling)
Labeled dataset (0-10 aesthetic scores)
```


## Cost and performance comparison

| Method | Cost | Speed | GPU Required | Quality |
|--------|------|-------|--------------|---------|
| **Local VLM (Qwen2.5-VL-7B)** | Free | ~2-5s per image | Yes (24GB+ VRAM) | High |
| **Gemini Batch API** | ~$0.003 per image | Async (hours) | No | High |
| **Gemini Sync API** | ~$0.006 per image | ~1-2s per image | No | High |


## Future work

- **Heuristic improvement**: Revisit sampling heuristics, which were originally designed to get a range of image quality for training, but I'm using the pipeline now with the intention of just finding good images
- **Prompt engineering**: Try a variety of prompts, consider few-shot training, add more wildlife-specific criteria (e.g. "eye contact with camera", "different species interacting", etc.)
- **VLM comparison**: Compare quality between Gemini models and local VLMs, e.g. highlighting images with significant disagreement
- **Human labeling**: Implement Labelme integration for human validation
- **Production deployment**: Scale to operational camera trap processing, e.g. include checkpoints to handle the case where large jobs are interrupted

## Technical notes

- **Image preprocessing**: All methods resize to 768px max dimension
- **Output compatibility**: All labeling scripts produce identical JSON format for visualization
- **GPU requirements**: Local VLM requires ~20-30GB VRAM for 7B model, 10-15GB for 3B model
