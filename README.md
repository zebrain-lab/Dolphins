# Dolphin Whistle Detection and Extraction

A Python package for automatically detecting and extracting dolphin whistles from audio recordings using deep learning models and template matching techniques.

## Overview

This package provides a comprehensive solution for analyzing dolphin vocalizations by:

1. **Detection**: Using pre-trained deep learning models (VGG-based) to identify whistle segments in audio recordings
2. **Extraction**: Extracting frequency contours from detected whistles using spectrogram analysis
3. **Template Matching**: Matching extracted whistles against known templates using Dynamic Time Warping (DTW)
4. **Clustering**: Grouping similar whistles together for analysis

## Features

- **Deep Learning Detection**: Pre-trained VGG-based models for accurate whistle detection in audio spectrograms
- **Automatic Contour Extraction**: Extracts frequency contours from detected whistle segments
- **Template Matching**: Uses DTW (Dynamic Time Warping) to match whistles against known templates
- **Batch Processing**: Efficiently processes multiple audio files with parallel processing
- **Clustering Analysis**: Groups similar whistles using distance-based clustering
- **Flexible CLI**: Command-line interface for easy integration into workflows

## Installation

### Requirements

- Python 3.7+
- TensorFlow 2.15.0
- NumPy, Pandas, SciPy
- scikit-learn
- fastdtw
- OpenCV (for image processing)
- Other dependencies listed in `requirements.txt`

### Install Dependencies

```bash
pip install -r requirements.txt
```

## Package Structure

```
AutomaticExtraction/
├── src/
│   ├── audio/              # Audio processing and spectrogram generation
│   ├── detection/          # Deep learning model for whistle detection
│   ├── extraction/         # Contour extraction and template matching
│   ├── cli/                # Command-line interface
│   ├── training/           # Model training utilities
│   └── utils/              # Helper functions
├── models/                 # Pre-trained detection models
└── examples/               # Example audio files and templates
```

## Usage

### Command-Line Interface

#### Detect Whistles in Audio Files

Detect whistles in audio recordings using a pre-trained model:

```bash
python run_detection.py \
    --model_path AutomaticExtraction/models/model_vgg.h5 \
    --recordings /path/to/audio/files \
    --saving_folder /path/to/results \
    --batch_size 50 \
    --save_p \
    --max_workers 8
```

**Parameters:**
- `--model_path`: Path to the pre-trained model file (.h5)
- `--recordings`: Directory containing audio files (.wav)
- `--saving_folder`: Directory to save detection results
- `--start_time`: Start time in seconds (default: 0)
- `--end_time`: End time in seconds (default: None, processes entire file)
- `--batch_size`: Number of spectrograms per batch (default: 50)
- `--save_p`: Save positive detections as images
- `--max_workers`: Number of parallel workers (default: 8)
- `--file_list`: Optional text file listing specific files to process

#### Extract Whistle Contours

Extract frequency contours from detection results:

```bash
python -m AutomaticExtraction.src.cli.extract_command \
    --audio_path /path/to/audio/file.wav \
    --predictions_path /path/to/predictions.csv \
    --output_dir /path/to/output \
    --cluster \
    --distance_threshold 0.3 \
    --visualize
```

**Parameters:**
- `--audio_path`: Path to audio file or directory
- `--predictions_path`: Path to predictions CSV file or directory
- `--output_dir`: Directory to save extracted contours
- `--cluster`: Enable clustering of similar contours
- `--distance_threshold`: DTW distance threshold for clustering (default: 0.3)
- `--min_samples`: Minimum samples per cluster (default: 2)
- `--visualize`: Generate cluster visualization

### Python API

#### Detection Example

```python
from AutomaticExtraction.src.detection.predict import process_predict_extract
from AutomaticExtraction.src.detection.model import load_detection_model

# Load model
model = load_detection_model('AutomaticExtraction/models/model_vgg.h5')

# Process recordings
process_predict_extract(
    recording_folder_path='path/to/recordings',
    saving_folder='path/to/results',
    start_time=0,
    end_time=1800,
    batch_size=50,
    save_p=True,
    model_path='AutomaticExtraction/models/model_vgg.h5',
    max_workers=8
)
```

#### Extraction Example

```python
from AutomaticExtraction.src.extraction.contour import extract_whistle_contours_from_file
from AutomaticExtraction.src.extraction.dtw import compute_normalized_dtw_distance
import pandas as pd

# Extract contours from a file
times, freqs = extract_whistle_contours_from_file(
    audio_path='path/to/audio.wav',
    predictions_path='path/to/predictions.csv',
    output_csv='path/to/contours.csv'
)

# Load template
template = pd.read_csv('examples/templates/SW_Neo_1.csv')

# Match against template using DTW
distance = compute_normalized_dtw_distance(
    contour1=contour,
    contour2=template[['time', 'frequency']].values
)
```

## Output Format

### Detection Results

Detection results are saved as CSV files with the following columns:
- `file_name`: Name of the audio file
- `initial_point`: Start time of detection (seconds)
- `finish_point`: End time of detection (seconds)
- `confidence`: Confidence score (0-1)

### Extraction Results

Extracted contours are saved as CSV files with:
- `time`: Time points (seconds)
- `frequency`: Frequency values (Hz)

## Template Matching

The package supports matching extracted whistles against known templates using Dynamic Time Warping (DTW). Templates should be provided as CSV files with `time` and `frequency` columns, or as MATLAB (.mat) files.

Example templates are provided in `examples/templates/`.

## Models

Pre-trained models are located in `AutomaticExtraction/models/`:
- `model_vgg.h5`: VGG-based detection model
- `model_finetuned_vgg_alexis.h5`: Fine-tuned VGG model

## Examples

See the example notebooks:
- `example_notebook_1.ipynb`: Basic detection and extraction workflow
- `example_notebook_2.ipynb`: Advanced clustering and analysis

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## Contact

- Maintainer / Lab: Zebrain Lab
- Issues: mustun@bio.ens.psl.eu
