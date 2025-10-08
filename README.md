# Finding "hero images" in camera trap image collections

This is an exploratory project comparing methods for identifying "hero images" from camera trap datasets, i.e., aesthetically pleasing wildlife photos.


## Project overview

The system will process large camera trap collections to identify candidates with aesthetic appeal through a two-stage pipeline:

1. **Candidate selection**: Heuristic-based filtering using AI detection results (typically MegaDetector and SpeciesNet) to identify promising images
2. **Labeling**: LLM aesthetic rating using Gemini 2.5 Flash or local VLMs

[This page](https://lila.science/public/misc/hero-image-samples/by-score/index.html) shows examples of the output of this system on a collection of images from [Snapshot Serengeti](https://lila.science/datasets/snapshot-serengeti/).


### Scripts

- **`generate_sequence_aware_candidates_optimized.py`** - Generate candidates for labeling using heuristics
- **`gemini_labeling.py`** - Gemini API labeling (supports both batch and synchronous modes)
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
python -m hero_images.generate_sequence_aware_candidates_optimized
```

### Label images

All labeling scripts accept flexible input sources for the first positional argument:
- **Directory**: Process all images in a directory (use `--recursive` for subdirectories)
- **Text file**: One absolute image path per line (lines starting with `#` are treated as comments)
- **JSON file**: A JSON array containing absolute image paths

#### Label images with Gemini 2.5 Flash

The Gemini labeling script supports two modes:

**Batch mode (default, recommended for large jobs):**
- 50% cost discount vs. synchronous API
- Asynchronous processing (takes hours but runs on Google's servers)
- Can resume/cancel jobs

```bash
# From directory
python -m hero_images.gemini_labeling /path/to/candidates --output-dir /path/to/output --recursive

# From text file with image paths
python -m hero_images.gemini_labeling /path/to/image_list.txt --output-dir /path/to/output

# From JSON file with image paths
python -m hero_images.gemini_labeling /path/to/image_list.json --output-dir /path/to/output
```

**Synchronous mode (good for smaller jobs):**
- 2x cost vs. batch mode
- Real-time processing with immediate results
- Progress updates as images are processed
- Supports checkpointing and resume (like Ollama labeling)

```bash
# Basic synchronous processing (with automatic checkpointing every 1000 images)
python -m hero_images.gemini_labeling /path/to/candidates --output-dir /path/to/output --recursive --sync

# Resume from checkpoint if interrupted
python -m hero_images.gemini_labeling /path/to/candidates --output-dir /path/to/output --sync --resume /path/to/output/gemini_sync_labels_YYYYMMDD_HHMMSS.tmp.json --recursive

# Custom checkpoint interval
python -m hero_images.gemini_labeling /path/to/candidates --output-dir /path/to/output --sync --checkpoint-interval 500 --recursive

# Disable checkpointing for short jobs
python -m hero_images.gemini_labeling /path/to/candidates --output-dir /path/to/output --sync --checkpoint-interval 0 --recursive
```

The `--model` argument is optional; the default is `gemini-2.5-flash`, you can also use `gemini-2.5-pro`.

#### Label images with a local VLM

##### vLLM vs. ollama

My experience was precisely consistent with Internet summaries: vLLM is way faster when running the same model, but basically requires a PhD in Linux to get things working, and in some cases, I just couldn't get some models working, even when they eventually worked on Ollama.  Ollama is a little slower, but "just works", including much more hassle-free management of VRAM.

##### Label images with vLLM

```bash
# Check GPU memory and get setup instructions
python -m hero_images.vllm_local_labeling --setup-help

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
python -m hero_images.vllm_local_labeling /path/to/candidates --output-dir /path/to/output --recursive
```

##### Label images with Ollama

```bash
# Start Ollama server (a bind error likely indicates the server is already running)
ollama serve

# Pull vision model (in another terminal)
ollama pull gemma3:12b

# Run labeling (with automatic checkpointing every 1000 images)
python -m hero_images.ollama_local_labeling /path/to/candidates --output-dir /path/to/output --recursive

# Resume from checkpoint if interrupted
python -m hero_images.ollama_local_labeling /path/to/candidates --output-dir /path/to/output --resume /path/to/output/ollama_local_labels_YYYYMMDD_HHMMSS.tmp.json --recursive

# Disable checkpointing for short jobs
python -m hero_images.ollama_local_labeling /path/to/candidates --output-dir /path/to/output --checkpoint-interval 0 --recursive
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
python -m hero_images.ollama_local_labeling /path/to/candidates --output-dir /path/to/output --model ${MODEL_NAME} --recursive
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

#### Shared parameters for Gemini labeling

- `--recursive` or `-r` - Search for images recursively in subdirectories
- `--image-size N` - Maximum dimension for resized images (default: 768)
- `--model MODEL` - Gemini model to use (default: gemini-2.5-flash)
- `--auto-confirm` or `-y` - Skip cost confirmation prompt
- `--sync` - Use synchronous API instead of batch (2x cost, real-time results)
- `--checkpoint-interval N` - Save progress every N images in sync mode (default: 1000, use 0 to disable)
- `--resume FILE` - Resume from checkpoint file (*.tmp.json for sync mode) or batch job metadata

#### Shared parameters for local VLM labeling

- `--recursive` or `-r` - Search for images recursively in subdirectories
- `--image-size N` - Maximum dimension for resized images (default: 768)


#### Checkpoint/resume options for local VLM labeling and Gemini sync mode

- `--checkpoint-interval N` - Save progress every N images (default: 1000, use 0 to disable)
- `--resume FILE.tmp.json` - Resume from specific checkpoint file
- Checkpoint files are automatically cleaned up on successful completion
- If process crashes, resume with `--resume /path/to/output/filename.tmp.json`
- Available for: Ollama labeling, vLLM labeling, and Gemini synchronous mode


### Visualize results

**Single model visualization**

```bash
python -m hero_images.generate_label_visualization /path/to/batch_labels_file_name_20250923_143022.json
```

**Multi-model comparison dashboard**

```bash
# Generate comparison dashboard for all models in a directory
python -m hero_images.generate_label_visualization /path/to/results_directory --sample-from /path/to/candidates --sample 100 --random-seed 42
```

Notes to self:

```bash
python -m hero_images.generate_label_visualization /mnt/c/temp/hero-images/labels/ --sample 1000 --random-seed 0 --sample-from /mnt/c/temp/hero-images/candidates/heuristics-20250923162520/
python -m hero_images.generate_label_visualization /mnt/c/temp/hero-images/labels/ --sample 1000 --random-seed 0 --sample-from /mnt/c/temp/hero-images/candidates/heuristics-20250923162520/ --sort-by score
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

### Cancel a running batch job

If you need to stop a batch job (e.g., if it's taking too long or you made an error):

```bash
# When you interrupt polling with Ctrl+C, the script shows the cancel command:
python -m hero_images.gemini_labeling --cancel batches/xyz789
```

Ctrl+C only stops the local script - the job continues running on Google's servers until cancelled.

Cancellation is only available for batch jobs, not synchronous processing.

### Resume batch jobs (running or completed)

If your script was interrupted or you want to retrieve results from a completed job:

```bash
python -m hero_images.gemini_labeling --resume /path/to/gemini_batch_metadata_YYYYMMDD_HHMMSS.json
```

Resume behavior:

- **Running jobs**: Continues polling until completion
- **Completed jobs**: Immediately retrieves and saves results
- **Failed/cancelled jobs**: Shows status and exits

### Resume synchronous jobs from checkpoint

Synchronous mode now supports checkpointing (like Ollama labeling):

```bash
python -m hero_images.gemini_labeling /path/to/candidates --output-dir /path/to/output --sync --resume /path/to/output/gemini_sync_labels_YYYYMMDD_HHMMSS.tmp.json --recursive
```

Checkpoint behavior:
- Automatically saves progress every 1000 images (configurable with `--checkpoint-interval`)
- Skips already-processed images when resuming
- Cleans up checkpoint file on successful completion
- Use `--checkpoint-interval 0` to disable for short jobs


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


## Future work

- **Parallelization**: Currently most of the models I'm using via ollama don't quite max out 1 GPU, and the other is siting idle.  Allow request submission across two threads.
- **Heuristic improvement**: Revisit sampling heuristics, which were originally designed to get a range of image quality for training, but I'm using the pipeline now with the intention of just finding good images
- **Prompt engineering**: Try a variety of prompts, consider few-shot training, add more wildlife-specific criteria (e.g. "eye contact with camera", "different species interacting", etc.).
- **Hyperparameter tuning**: Experiment with temperature, max_tokens
- **VLM comparison**: Compare quality between Gemini models and local VLMs, e.g. highlighting images with significant disagreement
- **Human labeling**: Implement Labelme integration for human validation


## Technical notes

- **Image preprocessing**: All methods resize to 768px max dimension by default
- **Output compatibility**: All labeling scripts produce identical JSON format for visualization
