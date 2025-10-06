# Finding "hero images" in camera trap image collections

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

Install the package in development mode (recommended):

```bash
pip install -e .
```

Or install just the requirements:

```bash
pip install -r requirements.txt
```

**For Gemini labeling:** Create an API key file:

```bash
echo "your-gemini-api-key" > GEMINI_API_KEY.txt
```

**For local VLM labeling (Ollama):** Install Ollama:

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

### Select candidates for LLM labeling

```bash
python3 generate_sequence_aware_candidates_optimized.py
```

### Label images

#### Label images with Gemini 2.5 Flash (via the Gemini Batch API)

```bash
python gemini_batch_labeling.py /path/to/candidates --output-dir /path/to/output --recursive --model gemini-2.5-flash
```

The `--model` argument is optional; the default is `gemini-2.5-flash`, you can also use `gemini-2.5-pro`.

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
python vllm_local_labeling.py /path/to/candidates --output-dir /path/to/output --recursive
```

##### Label images with Ollama

```bash
# Start Ollama server (a bind error likely indicates the server is already running)
ollama serve

# Pull vision model (in another terminal)
ollama pull gemma3:12b

# Run labeling (with automatic checkpointing every 1000 images)
python ollama_local_labeling.py /path/to/candidates --output-dir /path/to/output --recursive

# Resume from checkpoint if interrupted
python ollama_local_labeling.py /path/to/candidates --output-dir /path/to/output --resume /path/to/output/ollama_local_labels_YYYYMMDD_HHMMSS.tmp.json --recursive

# Disable checkpointing for short jobs
python ollama_local_labeling.py /path/to/candidates --output-dir /path/to/output --checkpoint-interval 0 --recursive
```

Models to try:

* gemma3:4b (3.3GB)
* gemma3:12b (8.1GB)
* gemma3:27b (17GB)

* llava:7b (4.7GB)
* llava:13b (8.0GB)
* llava:34b (20GB)

* qwen2.5vl:3b (3.2GB)
* qwen2.5vl:7b (6.0GB)
* qwen2.5vl:32b (21GB)
* qwen2.5vl:72b (49GB)

For example:

```bash
export MODEL_NAME=qwen2.5vl:72b
ollama pull ${MODEL_NAME}
python ollama_local_labeling.py /path/to/candidates --output-dir /path/to/output --model ${MODEL_NAME} --recursive
```

Other Ollama notes:

* To bind separate servers to separate GPUs:

```bash
# First instance (GPU 0)
CUDA_VISIBLE_DEVICES=0 OLLAMA_HOST=localhost:11434 ollama serve

# Second instance (GPU 1) 
CUDA_VISIBLE_DEVICES=1 OLLAMA_HOST=localhost:11435 ollama serve
```

* If ollama is running as a service, kill it via `sudo systemctl stop ollama`, re-start it with `sudo systemctl start ollama`.  Disable service auto-start with `sudo systemctl disable ollama`.

* Models are stored in ~/.ollama/models, unless you change the OLLAMA_MODELS environment variable.  If you run ollama via `ollama serve`, you need to set this variable in the shell where you run `ollama serve`.  If ollama is running as a service, follow [these instructions](https://github.com/ollama/ollama/issues/680#issuecomment-2880768673) to change the model download folder.

* List models with `ollama list`

* Remove models with `ollama rm`

* With some models (not even particularly large models) I was getting timeouts during model loading.  This seems sporadic and unrelated to model size, system load, etc.  The following might help, in the shell where you're going to run `ollama serve`:

```bash
export OLLAMA_KEEP_ALIVE=1h
export OLLAMA_LOAD_TIMEOUT=30m
```

#### Shared parameters

- `--recursive` or `-r` - Search for images recursively in subdirectories
- `--image-size N` - Maximum dimension for resized images (default: 768)


#### Checkpoint/resume options for both local VLM labeling Scripts

- `--checkpoint-interval N` - Save progress every N images (default: 1000, use 0 to disable)
- `--resume FILE.tmp.json` - Resume from specific checkpoint file
- Checkpoint files are automatically cleaned up on successful completion
- If process crashes, resume with `--resume /path/to/output/filename.tmp.json`


### Visualize results

**Single model visualization**

```bash
python generate_label_visualization.py /path/to/batch_labels_file_name_20250923_143022.json
```

**Multi-model comparison dashboard**

```bash
# Generate comparison dashboard for all models in a directory
python generate_label_visualization.py /path/to/results_directory --sample-from /path/to/candidates --sample 100 --random-seed 42
```

Notes to self:

```bash
python generate_label_visualization.py /mnt/c/temp/hero-images/labels/ --sample 1000 --random-seed 0 --sample-from /mnt/c/temp/hero-images/candidates/heuristics-20250923162520/
python generate_label_visualization.py /mnt/c/temp/hero-images/labels/ --sample 1000 --random-seed 0 --sample-from /mnt/c/temp/hero-images/candidates/heuristics-20250923162520/ --sort-by score
```

This creates:
- Individual HTML files for each *_labels_*.json file found
- An index.YYYYMMDD_HHMMSS.html dashboard with links to all results
- A single shared batch_labels_YYYYMMDD_HHMMSS_images/ folder

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
- **Prompt engineering**: Try a variety of prompts, consider few-shot training, add more wildlife-specific criteria (e.g. "eye contact with camera", "different species interacting", etc.).
- **Hyperparameter tuning**: Experiment with temperature, max_tokens
- **VLM comparison**: Compare quality between Gemini models and local VLMs, e.g. highlighting images with significant disagreement
- **Human labeling**: Implement Labelme integration for human validation
- **Production deployment**: Scale to operational camera trap processing, e.g. include checkpoints to handle the case where large jobs are interrupted


## Technical notes

- **Image preprocessing**: All methods resize to 768px max dimension by default
- **Output compatibility**: All labeling scripts produce identical JSON format for visualization
