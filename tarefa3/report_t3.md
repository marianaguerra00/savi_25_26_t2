# Digit Detection and Recognition with Sliding Windows

This repository contains a complete **digit detection and recognition system** based on a **sliding window approach** combined with a **CNN classifier** trained on MNIST-style digits.  
The project includes detection, evaluation, and an interactive visualization interface.

---


## How to Run
```bash
python main_sliding_window.py![alt text](image-1.png)
```

## How it Run

### 1. Sliding Window

- Window size: **36 × 36**
- Stride: **4**
- Dense scanning maximizes recall

### 2. Window Rejection (Early Filtering)

Windows are discarded if they:

- Have low contrast
- Contain too little foreground
- Touch the image borders
- Are likely background-only

This significantly reduces computation.

### 3. CNN Classification

Each remaining window is:

- Resized to MNIST style (28×28)
- Normalized
- Classified using a CNN

Predictions are filtered by:

- Confidence threshold
- Confidence margin (top-1 vs top-2)

### 4. Spatial Grouping (Soft NMS)

Instead of classical Non-Maximum Suppression (NMS), we use:

- Spatial clustering
- Bounding box averaging
- Majority voting on labels

This approach is robust for dense sliding windows with heavy overlap.

```bash
We use spatial clustering followed by averaging instead of classical NMS, since the sliding windows are dense and strongly overlapping.
```

### 5. Final Bounding Boxes

For each cluster:

- Bounding box = average of windows
- Label = majority vote

### 6. Intersection over Union (IoU)
- IoU threshold: 0.3
- Lower threshold improves recall for imperfect localization

### Matching Strategy

- Ground-truth driven matching
- One-to-one matching
- Class must match
- Each prediction and GT can be used only once

### Metrics

- True Positives (TP): correct digit + sufficient IoU
- False Positives (FP): prediction without valid GT match
- False Negatives (FN): GT digit with no matched prediction

Computed globally:
```bash
Precision = TP / (TP + FP)
Recall    = TP / (TP + FN)
F1-score  = harmonic mean
```

### Visualization Interface

The visualization provides:
- Bounding boxes overlaid on images
- Color-coded detections:

    - 🟢 Green: True Positive (TP)

    - 🔴 Red: False Positive (FP)

    - 🔵 Blue dashed: False Negative (FN)

- Predicted digit labels
- Interactive buttons:
    - Next / Previous image
    - Next / Previous statistics

- Detection metrics plot
- Confidence distribution histogram
- Total processing time displayed (top-right)