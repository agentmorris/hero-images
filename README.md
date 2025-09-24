# Hero image classification project

This is an exploratory project comparing methods for identifying "hero images" from camera trap datasets, i.e., aesthetically pleasing wildlife photos.


## Project overview

The system will process large camera trap collections to identify candidates with aesthetic appeal through a three-stage pipeline:

1. **Candidate selection**: Heuristic-based filtering using AI detection results to identify promising images
2. **Labeling**: Hybrid human/LLM aesthetic rating using Gemini 2.5 Flash
3. **Model training**: Supervised learning for automated hero image classification


## Production scripts

### Core pipeline

- **`stratified_selector_sequence_aware.py`** - Main candidate selection system with sequence awareness
- **`generate_sequence_aware_candidates_optimized.py`** - Generate 5K candidates from full dataset
- **`gemini_labeling_pipeline.py`** - Synchronous Gemini 2.5 Flash labeling pipeline
- **`gemini_batch_labeling.py`** - Asynchronous batch API labeling with cost optimization
- **`generate_label_visualization.py`** - Create HTML visualizations compatible with both sync and batch labeling results


## Scrap/exploration scripts

### Data exploration

- **`explore_detection_results.py`** - Initial exploration of MegaDetector output format
- **`analyze_detection_format.py`** - Parse detection JSON structure (initial version)
- **`analyze_detection_format_fixed.py`** - Fixed detection format analysis
- **`get_species_distribution.py`** - Comprehensive dataset statistics and species analysis
- **`verify_image_paths.py`** - Verify detection results match actual image files

### Candidate selection evolution

- **`candidate_selector.py`** - Original heuristic selector (pre-sequence-aware)
- **`stratified_selector.py`** - Early stratified sampling approach
- **`stratified_selector_improved.py`** - Improved size scoring with Gaussian distribution
- **`stratified_selector_training.py`** - Training-oriented version with negative examples
- **`test_candidate_selection.py`** - Test original candidate selection system

### Generation scripts (development)

- **`generate_candidates.py`** - Early candidate generation script
- **`generate_candidates_incremental.py`** - Incremental processing version
- **`generate_full_candidates.py`** - Full dataset processing (pre-sequence-aware)
- **`generate_test_candidates.py`** - Generate small test batches for validation


## Usage

### 1. Setup

```bash
pip install -r requirements.txt
echo "your-gemini-api-key" > GEMINI_API_KEY.txt
```

### 2. Generate candidates (if needed)

```bash
python3 generate_sequence_aware_candidates_optimized.py
```

### 3. Label images

**Option A: Synchronous (immediate results)**

```bash
python gemini_labeling_pipeline.py /path/to/candidates --output-dir /path/to/output
```

**Option B: Batch API (50% cheaper, asynchronous)**

```bash
python gemini_batch_labeling.py /path/to/candidates --output-dir /path/to/output
```

### 4. Visualize results

```bash
python generate_label_visualization.py /path/to/gemini_batch_labels_20250923_143022.json
```


## Batch job management

### Cancel a running job

If you need to stop a batch job (e.g., if it's taking too long or you made an error):

```bash
# When you interrupt polling with Ctrl+C, the script shows the cancel command:
python gemini_batch_labeling.py --cancel batches/xyz789
```

**Important**: Ctrl+C only stops the local script - the job continues running on Google's servers until cancelled.

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
Raw Images
    ↓ (MegaDetector + SpeciesNet)
Detection and classification results
    ↓ (Sequence-Aware Stratified Sampling)
Candidates (5K diverse images)
    ↓ (Gemini 2.5 Flash Labeling)
Labeled Dataset (0-10 aesthetic scores)
    ↓ (Future: Model Training)
Hero Image Classifier
```


## Future work

- **Model training**: Train CNN classifier on labeled dataset
- **VLM integration**: Add open-weights VLM candidate selection
- **Human labeling**: Implement Labelme integration for human validation
- **Production deployment**: Scale to operational camera trap processing


## Cost optimization

- **Image resizing**: 768px max dimension
- **Batch processing**: 50% cost reduction vs synchronous API calls
