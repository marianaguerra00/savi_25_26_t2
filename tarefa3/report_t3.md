# MNIST Digit Detection System

A real-time digit detection and evaluation system using a CNN-based sliding window approach with interactive visualization.

## Overview

This system detects handwritten digits in 128x128 images using a trained CNN model. It employs a sliding window technique with configurable detection parameters and provides comprehensive evaluation metrics with an interactive visualization interface.

## Features

- **Sliding Window Detection**: Scans images with configurable window size and stride
- **Smart Window Rejection**: Filters out empty or low-confidence regions
- **Bounding Box Grouping**: Merges overlapping detections intelligently
- **Comprehensive Evaluation**: Calculates precision, recall, and F1 score with IoU-based matching
- **Interactive Visualization**: Browse detection results with ground truth comparison
- **Performance Metrics**: Visual display of model confidence distribution

## Requirements

```
torch
torchvision
numpy
opencv-python (cv2)
matplotlib
tqdm
```

## Installation

```bash
pip install torch torchvision numpy opencv-python matplotlib tqdm
```

## Usage

### Basic Usage

```bash
python detect.py
```

### Advanced Options

```bash
python detect.py --version versionD \
                 --modelPath ./mnist_cnn.pth \
                 --numImages 30 \
                 --maxImages 1000
```

### Arguments

- `--version`: Dataset version folder name (default: `versionD`)
- `--modelPath`: Path to trained model weights (default: `./mnist_cnn.pth`)
- `--numImages`: Number of images to store for visualization (default: `30`)
- `--maxImages`: Maximum images to process, -1 for all (default: `-1`)

## Detection Configuration

All detection parameters are configured at the top of `main()`:

```python
windowSize = 36              # Sliding window size
stride = 4                   # Window stride
minForegroundRatio = 0.015   # Minimum foreground pixel ratio
foregroundThreshold = 30     # Pixel intensity threshold
minMaxIntensityFactor = 2.0  # Min/max intensity ratio
confidenceThreshold = 0.7    # Minimum prediction confidence
confidenceMargin = 0.25      # Top-2 confidence margin
borderMarginRatio = 0.05     # Border exclusion ratio
groupingDistance = 25.2      # Bounding box grouping distance
iouThreshold = 0.3           # IoU threshold for matching
```

## Detection Pipeline

1. **Image Loading**: Loads 128x128 images from ubyte format
2. **Sliding Window**: Scans image with configurable window and stride
3. **Window Rejection**: Filters low-quality windows using multiple heuristics
4. **Digit Recognition**: Processes accepted windows through CNN model
5. **Confidence Filtering**: Rejects low-confidence and ambiguous predictions
6. **Bounding Box Grouping**: Merges nearby detections into single predictions
7. **Evaluation**: Matches predictions to ground truth using IoU

## Visualization Interface
<img src="images\slidingwindow.png" width="600" alt="Description">

### Components

**Image Display** (Top Left)
- Green boxes: True Positives (TP) - correct detection and classification
- Orange boxes: False Positives (FP-WR) - correct location, wrong digit
- Red boxes: False Positives (FP) - incorrect detection
- Blue dashed boxes: False Negatives (FN) - missed ground truth
- Yellow labels: Ground truth digit for matched predictions

**Metrics Chart** (Bottom Left)
- Bar chart showing Precision, Recall, and F1 Score
- Displays overall detection performance

**Navigation** (Right Side)
- Next/Previous buttons to browse through processed images
- Processing time displayed in top-right corner

## Evaluation Metrics

The system calculates standard object detection metrics:

- **Precision**: `TP / (TP + FP)` - accuracy of positive predictions
- **Recall**: `TP / (TP + FN)` - coverage of ground truth objects
- **F1 Score**: `2 × (Precision × Recall) / (Precision + Recall)` - harmonic mean

Detection matching uses IoU (Intersection over Union) with a configurable threshold.

## Data Format

### Images
- Binary format: ubyte
- Size: 128x128 pixels
- Grayscale intensity values

### Labels
- Binary format with custom structure
- Format per image: `[num_objects][digit, x, y, w, h]...`
- Coordinates stored as uint16

## Algorithm Details

### Window Rejection Criteria

A window is rejected if:
- Maximum intensity too close to mean (low contrast)
- Foreground pixel ratio below threshold
- No foreground pixels detected
- Digit touches window border

### Bounding Box Grouping

Boxes are grouped if their centers are within `groupingDistance`. The final box is the average of all grouped boxes, and the label is determined by majority vote.

### Detection Matching

Predictions are matched to ground truth using:
1. IoU calculation between predicted and ground truth boxes
2. Greedy matching: each prediction matched to best available GT
3. Classification correctness checked after spatial matching

## Performance

Typical performance on test dataset:
- Processing time: ~15 minutes for 10,000 images (CPU)
- Precision: ~0.97
- Recall: ~0.96
- F1 Score: ~0.96

## Customization

### Adjusting Detection Sensitivity

**More Detections** (higher recall, lower precision):
- Decrease `confidenceThreshold` (e.g., 0.5)
- Decrease `confidenceMargin` (e.g., 0.15)
- Decrease `foregroundThreshold` (e.g., 20)

**Fewer False Positives** (higher precision, lower recall):
- Increase `confidenceThreshold` (e.g., 0.8)
- Increase `confidenceMargin` (e.g., 0.35)
- Increase `minForegroundRatio` (e.g., 0.02)

### Changing Detection Resolution

- Smaller `stride`: More thorough scanning, slower processing
- Larger `stride`: Faster processing, may miss small digits
- Adjust `windowSize`: Must match digit size in images

## Qualitative Analysis

### Main Limitations

**Processing Speed**: Sliding window approach is slow (~15 minutes for 10,000 images) due to dense scanning generating ~1,000 windows per image without batch processing.

**Confidence Filtering**: Not fully effective - causes valid low-confidence digits to become False Negatives, creating precision-recall trade-off.

**Common Errors**: 
- FP from background noise, partial/overlapping digits, misclassification
- FN from low confidence, poor contrast, border-touching, extreme scaling
- Grouping issues: over-merging distinct digits or under-merging duplicates

### Conclusion

The system achieves strong performance (F1: 0.96) but faces fundamental trade-offs:
- **Speed vs. Accuracy**: Dense scanning ensures coverage but is computationally expensive
- **Precision vs. Recall**: Confidence filtering reduces false positives but increases false negatives
- **Generalization**: Fixed parameters work well on test set but may not adapt to varying conditions

Suitable for offline batch processing where accuracy is prioritized over speed. Real-time applications would require significant architectural changes (batch inference, NMS, adaptive thresholds, or Region Proposal Networks).


## Author

João Freitas & Mariana Guerra