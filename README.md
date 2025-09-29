# Hero Image Classification Project

This is an exploratory project comparing methods for identifying "hero images" from camera trap datasets, i.e., aesthetically pleasing wildlife photos.


## Project overview

The system will process large camera trap collections to identify candidates with aesthetic appeal through a two-stage pipeline:

1. **Candidate selection**: Heuristic-based filtering using AI detection results (typically MegaDetector and SpeciesNet) to identify promising images
2. **Labeling**: LLM aesthetic rating using Gemini 2.5 Flash or local VLMs


### Scripts

- **`generate_sequence_aware_candidates_optimized.py`** - Generate candidates for labeling using heuristics
- **`gemini_batch_labeling.py`** - Asynchronous API labeling using the Gemini batch API
- **`vllm_local_labeling.py`** - Local VLM labeling via vLLM
- **`ollama_local_labeling.py`** - Local VLM labeling via Ollama models)
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

**For local VLM labeling (vLLM)**

```bash
pip install -r requirements.txt
pip install vllm
pip install huggingface-hub
```

**For local VLM labeling (Ollama)**

```bash
pip install -r requirements.txt
curl -fsSL https://ollama.com/install.sh | sh
```

### Select candidates for LLM labeling

```bash
python3 generate_sequence_aware_candidates_optimized.py
```

### Label images

#### Label images with Gemini 2.5 Flash (via the Gemini Batch API)

```bash
python gemini_batch_labeling.py /path/to/candidates --output-dir /path/to/output
```


#### Label images with a local VLM

##### vLLM vs. ollama

My experience was precisely consistent with Internet summaries: vLLM is way faster when running the same model, but basically requires a PhD in Linux to get things working, and in some cases, I just couldn't get some models working, even when they eventually worked on Ollama.  Ollama is a little slower, but "just works", including much more hassle-free management of VRAM.

##### Label images with vLLM

```bash
# Check GPU memory and get setup instructions
python vllm_local_labeling.py --setup-help

# Start vLLM server (example for Qwen-2.5-VL-7B on 2x4090)
vllm serve Qwen/Qwen2.5-VL-7B-Instruct \
  --host 0.0.0.0 \
  --port 8000 \
  --data-parallel-size 2 \
  --gpu-memory-utilization 0.9 \
  --max-model-len 60000

# Start vLLM server (example for Gemma3-12B on 2x4090)
#
# Gemma requires acknowledging the license agreement, so you have to
# sign in to Hugging Face before starting vLLM.
hf auth login
vllm serve google/gemma-3-12b-it \
    --host 0.0.0.0 \
    --port 8000 \
    --data-parallel-size 2 \
    --gpu-memory-utilization 0.9 \
    --max-model-len 5000 \
	--max-num-seqs 1 \
	--max-num-batched-tokens 2048

# Note to self: we're using a smaller value for --max-model-len (maximum context size)
# for Gemma because the model weights are larger, leaving less VRAM for context. Estimated
# context size for my queries is <5000 (200-300 for text, 1k-2k per image, 100-200 for response),
# so even 32k is plenty.

# Run labeling (in another terminal)
python vllm_local_labeling.py /path/to/candidates --output-dir /path/to/output
```

##### Label images with Ollama

```bash
# Start Ollama server (a bind error likely indicates the server is already running)
ollama serve

# Pull vision model (in another terminal)
ollama pull gemma3:12b

# Run labeling
python ollama_local_labeling.py /path/to/candidates --output-dir /path/to/output
```

Alternative models to try:

* gemma3:27b (17GB)
* qwen2.5vl:32b (21GB)
* llava:34b (20GB)
* qwen2.5vl:72b (49GB)

For example:

```bash
export MODEL_NAME=qwen2.5vl:72b
ollama pull ${MODEL_NAME}
python ollama_local_labeling.py /path/to/candidates --output-dir /path/to/output --model ${MODEL_NAME}
```

Other Ollama notes:

* To bind separate servers to separate GPUs:

```bash
# First instance (GPU 0)
CUDA_VISIBLE_DEVICES=0 OLLAMA_HOST=localhost:11434 ollama serve

# Second instance (GPU 1) 
CUDA_VISIBLE_DEVICES=1 OLLAMA_HOST=localhost:11435 ollama serve
```

* If ollama is running as a service, kill it via `sudo systemctl stop ollama`

* Models are stored in ~/.ollama/models

* List models with `ollama list`

* Remove models with `ollama rm`

### Visualize results

**Single model visualization**

```bash
python generate_label_visualization.py /path/to/gemini_batch_labels_20250923_143022.json
python generate_label_visualization.py /path/to/vllm_local_labels_20250927_143022.json
```

**Side-by-side model comparison**

```bash
# Generate visualizations for the same set of images across different models
# Use the first result file to define the sample set
python generate_label_visualization.py gemini_results.json --sample-from gemini_results.json --random-seed 42
python generate_label_visualization.py ollama_results.json --sample-from gemini_results.json --random-seed 42
python generate_label_visualization.py vllm_results.json --sample-from gemini_results.json --random-seed 42

# Or sample from a directory of candidate images
python generate_label_visualization.py model1_results.json --sample-from /path/to/candidates --sample 100 --random-seed 42
python generate_label_visualization.py model2_results.json --sample-from /path/to/candidates --sample 100 --random-seed 42
```

**Visualization options**

- `--sample N` - Show N randomly sampled images (default: 500)
- `--random-seed N` - Fix random seed for reproducible sampling (default: 0)
- `--sample-from PATH` - Sample from specific directory or JSON file for consistent comparison
- `--sort-by {filename,score}` - Sort by filename (default) or aesthetic score
- `--top-only` - Show only successful results (exclude failed images)


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

- **Parallelization**: Currently most of the models I'm using via ollama don't quite max out 1 GPU, and the other is siting idle.  Allow request submission across two threads.
- **Heuristic improvement**: Revisit sampling heuristics, which were originally designed to get a range of image quality for training, but I'm using the pipeline now with the intention of just finding good images
- **Prompt engineering**: Try a variety of prompts, consider few-shot training, add more wildlife-specific criteria (e.g. "eye contact with camera", "different species interacting", etc.)
- **VLM comparison**: Compare quality between Gemini models and local VLMs, e.g. highlighting images with significant disagreement
- **Human labeling**: Implement Labelme integration for human validation
- **Production deployment**: Scale to operational camera trap processing, e.g. include checkpoints to handle the case where large jobs are interrupted

## Technical notes

- **Image preprocessing**: All methods resize to 768px max dimension
- **Output compatibility**: All labeling scripts produce identical JSON format for visualization
- **GPU requirements**: Local VLM requires ~20-30GB VRAM for 7B model, 10-15GB for 3B model
