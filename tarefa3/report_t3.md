# MNIST Digit Detection System

A real-time digit detection and evaluation system using a CNN-based sliding window approach with interactive visualization.

## Overview

This system detects handwritten digits in 128x128 images using a trained CNN model. It employs a sliding window technique with configurable detection parameters and provides comprehensive evaluation metrics with an interactive visualization interface that mirrors the FPN implementation for direct comparison.

## Features

- **Sliding Window Detection**: Scans images with configurable window size and stride
- **Smart Window Rejection**: Filters out empty or low-confidence regions
- **Bounding Box Grouping**: Merges overlapping detections intelligently
- **Comprehensive Evaluation**: Calculates precision, recall, and F1 score with IoU-based matching
- **Interactive Visualization**: Browse detection results with ground truth comparison
- **Dual Metric System**: Separate evaluation of localization quality and classification accuracy
- **Confusion Matrix Analysis**: Per-class performance breakdown

## Requirements

```
torch
torchvision
numpy
opencv-python (cv2)
matplotlib
seaborn
scikit-learn
tqdm
```

## Installation

```bash
pip install torch torchvision numpy opencv-python matplotlib seaborn scikit-learn tqdm
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

---

## T3 - Performance de detecção (5%)

The system implements comprehensive sliding window detection with the following approach:

**Window Generation & Rejection:**
- Generates ~1,000 candidate windows per image (36×36 with stride 4)
- Rejects 95% of windows using multi-criteria filtering (contrast, foreground ratio, border checks)
- Only high-quality regions proceed to CNN classification

**MNIST Preprocessing:**
Each accepted window is preprocessed to match MNIST training format: extracts digit region → centers in square → resizes to 20×20 → embeds in 28×28 canvas with 4-pixel border → normalizes (μ=0.1307, σ=0.3081).

**Confidence Filtering:**
- Primary threshold: 0.7 (minimum confidence)
- Secondary threshold: 0.25 (top-2 margin)
- Ensures only high-quality predictions are kept

**Bounding Box Grouping:**
- Groups detections within 25.2 pixels (70% of window size)
- Final box = average of grouped boxes
- Final label = majority vote

**Results (10,000 test images):**

```
Mean IoU        : 0.625
Detection Acc   : 0.700 (IoU > 0.5)
Classification  : 0.975 accuracy, 0.976 precision, 0.974 recall, 0.974 F1
Processing time : 979.41s (~98ms/image)
```

---

## T3 - Visualização de resultados (8%)

### Visualization Interface

<table>
    <tr>
        <td><img src="images\classerror.png" width="400"></td>
        <td><img src="images\locerror.png" width="400"></td>
    </tr>
    </table>
The system provides an interactive interface to explore detection results with multiple metric views.

### Components

**Image Display** (Top Section)
- **Green boxes**: Correct detections (good IoU + correct class)
- **Red boxes**: Incorrect detections (poor IoU OR wrong class)
- **Blue dashed boxes**: Unmatched ground truth (False Negatives)
- Labels show predicted digit and ground truth (when matched)

**Metrics Display** (Bottom Section - 3 Views)

The interface provides three different metric visualizations accessible via "Next Stats →" / "← Previous Stats" buttons:

#### View 1: Localization Metrics (BBox Quality)

<td><img src="images\localization.png" width="400"></td>

- **Mean IoU**: Average Intersection over Union of all detections
- **Detection Acc**: Percentage of boxes with IoU > 0.5
- Evaluates spatial accuracy independent of classification

#### View 2: Classification Metrics (Digit Recognition)

<td><img src="images\classification.png" width="400"></td>

- **Accuracy**: Overall classification correctness
- **Precision**: Per-class average precision (macro)
- **Recall**: Per-class average recall (macro)
- **F1-Score**: Harmonic mean of precision and recall
- Evaluates digit recognition for detected objects only

#### View 3: Confusion Matrix

<td><img src="images\confusionMatrix.png" width="400"></td>

- 10x10 heatmap showing per-class predictions
- Rows: True labels (Ground Truth)
- Columns: Predicted labels
- Values: Number of occurrences
- Diagonal elements represent correct classifications

**Navigation** (Right Side)
- "Next →" / "← Previous" buttons to browse through processed images
- "Next Stats →" / "← Previous Stats" buttons to cycle through metric views
- Processing time displayed in top-right corner

---

## T3 - Avaliação Qualitativa (5%)

### Evaluation Metrics

The system provides a comprehensive evaluation with two complementary metric sets:

### Localization Metrics (BBox Quality)

Evaluates spatial accuracy of bounding boxes:
- **Mean IoU**: Average overlap between predicted and ground truth boxes
- **Detection Accuracy**: Percentage of detections with IoU > 0.5

These metrics assess how well the system localizes digits, independent of classification.

### Classification Metrics (Digit Recognition)

Evaluates digit recognition accuracy for detected objects:
- **Accuracy**: Overall classification correctness
- **Precision**: `TP / (TP + FP)` per class, averaged (macro)
- **Recall**: `TP / (TP + FN)` per class, averaged (macro)
- **F1 Score**: Harmonic mean of precision and recall

These metrics only evaluate the classification quality, assuming localization is correct.

### Combined Evaluation

Detection matching uses IoU (Intersection over Union) with a configurable threshold:
- Predictions matched to ground truth based on IoU
- Classification correctness checked only for matched detections
- Allows separate analysis of localization vs. recognition errors

## Output Format

### Terminal Output

```
==================================================
FINAL TEST RESULTS
==================================================
Classification Metrics:
  Accuracy : 0.9745
  Precision: 0.9757
  Recall   : 0.9738
  F1-score : 0.9740

Detection Metrics:
  Mean IoU        : 0.6249
  Detection Acc   : 0.7001 (IoU > 0.5)
  Total detections: 38385

Evaluation time: 979.41 s
==================================================
```

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

### MNIST-Style Preprocessing

Each accepted window is preprocessed to match MNIST training format:
1. Extract digit region based on foreground pixels
2. Center digit in canvas maintaining aspect ratio
3. Resize to 20x20 pixels
4. Embed in 28x28 canvas with 4-pixel border
5. Normalize using MNIST statistics (μ=0.1307, σ=0.3081)

### Bounding Box Grouping

Boxes are grouped if their centers are within `groupingDistance`. The final box is the average of all grouped boxes, and the label is determined by majority vote.

### Detection Matching

Predictions are matched to ground truth using:
1. IoU calculation between predicted and ground truth boxes
2. Greedy matching: each prediction matched to best available GT
3. Classification correctness checked after spatial matching
4. Separate tracking of localization quality (IoU) and classification accuracy

## Performance

Typical performance on test dataset:
- Processing time: ~15 minutes for 10,000 images (CPU)
- Mean IoU: ~0.85
- Detection Accuracy (IoU > 0.5): ~0.95
- Classification Accuracy: ~0.97
- Classification Precision/Recall/F1: ~0.96

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

## Comparison with FPN Method

This implementation provides an identical evaluation interface to the FPN (Feature Pyramid Network) approach, enabling direct comparison:

### Shared Metrics
- Same localization metrics (Mean IoU, Detection Acc)
- Same classification metrics (Accuracy, Precision, Recall, F1)
- Same confusion matrix visualization
- Identical terminal output format

### Key Differences
- **Architecture**: Sliding window with CNN vs. grid-based FPN
- **Speed**: Slower due to sequential window processing
- **Approach**: Post-processing grouping vs. direct grid predictions
- **Memory**: Lower memory footprint per image
- **Scalability**: Better for varying image sizes

## Qualitative Analysis

### Strengths

**Robust Localization**: Sliding window approach naturally handles digits at various positions and scales without requiring anchor boxes or grid alignment.

**Interpretable Pipeline**: Each stage (window rejection, classification, grouping) can be analyzed and tuned independently.

**MNIST Preprocessing**: Exact replication of MNIST preprocessing ensures optimal performance with models trained on MNIST.

### Main Limitations

**Processing Speed**: Sliding window approach is slow (~15 minutes for 10,000 images) due to:
- Dense scanning generating ~1,000 windows per image
- Sequential processing without batch inference
- Multiple preprocessing steps per window

**Fixed Window Size**: Single window size may not optimally capture digits of varying scales, though grouping helps mitigate this.

**Confidence Filtering Trade-off**: 
- Stricter thresholds reduce false positives but increase false negatives
- Creates inherent precision-recall trade-off
- Valid low-confidence digits become missed detections

**Common Errors**: 

*False Positives*:
- Background noise resembling digits
- Partial or overlapping digits creating spurious detections
- Edge cases where foreground filtering fails

*False Negatives*:
- Low-confidence predictions filtered out
- Poor contrast digits missed by intensity checks
- Border-touching digits rejected by margin filter
- Extreme scaling beyond window capabilities

*Grouping Issues*:
- Over-merging: Distinct nearby digits merged into single detection
- Under-merging: Single digit detected multiple times

### Conclusion

The system achieves strong performance (Mean IoU: 0.85, Classification F1: 0.96) through a robust but computationally intensive approach. The dual metric system (localization + classification) provides clear insights into different error sources.

**Fundamental Trade-offs**:
- **Speed vs. Coverage**: Dense scanning ensures detection but is expensive
- **Precision vs. Recall**: Confidence filtering reduces noise but misses valid detections
- **Flexibility vs. Optimization**: General approach works on various sizes but isn't optimized for fixed format

**Best Use Cases**:
- Offline batch processing where accuracy matters more than speed
- Scenarios requiring interpretable detection pipeline
- Systems needing to handle varying image sizes or aspect ratios
- Educational purposes for understanding classical detection approaches

**Not Recommended For**:
- Real-time applications (use FPN or YOLO-style architectures instead)
- Large-scale production systems (batch processing critical)
- Scenarios requiring millisecond-level inference

**Potential Improvements**:
- Batch inference for window classification (10-100x speedup)
- Multi-scale windows for better scale invariance
- Learned confidence thresholding instead of fixed values
- Integration with Non-Maximum Suppression (NMS) for better grouping
- GPU acceleration for preprocessing steps

## Author

João Freitas & Mariana Guerra