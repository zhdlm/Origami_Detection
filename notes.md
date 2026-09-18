# Project
```
Raw AFM Image 
      │
      ▼
┌─────────────────────────────────────────────────────┐
│ Instance Segmentation Model (e.g., YOLO, Mask-R-CNN)│
└─────────────────────────────────────────────────────┘
      │
      ├──> Instance Mask  (Per-object pixel boundaries)
      ├──> Shape Class    (Triangle, Star, Smiley...)
      └──> Detection Confidence Score
```
- Segmentation: The network outputs isolated binary masks for each touching or overlapping object.

- Classification: The network assigns class probabilities directly to each segmented mask.

- Completeness Score: Handle this using a hybrid approach (Model + Post-Processing), which is significantly easier to scale and debug:

    - Approach A (Algorithmic / Post-processing): Train the neural network to segment the shape. Once you have the predicted binary mask, compute its completeness mathematically by comparing it against an ideal geometric template or calculating shape descriptors (e.g., convex hull ratio, area-to-perimeter ratio, or structural similarity index).

    - Approach B (Class-based): Treat completeness as sub-classes inside the network (e.g., triangle_complete, triangle_broken).

**Start with YOLO and may move to Mask-R-CNN if underperorm on high density images, overcrowing, and/or imbalanced data sets.**

## Summary & Architecture plan
**Core Objective**
Build a fine-tuned deep learning pipeline to analyze 2D Atomic Force Microscopy (AFM) heightmaps. The pipeline must:
1. Segment individual, potentially touching, or incomplete shapes as distinct instances.
2. Classify each instance (e.g., triangle, rectangle, star, smiley).
3. Determine a completeness score ($0$ to $1$) for each detected shape.
4. Scale easily to new 2D shapes or heightmaps with greater 3D topographic variation without rebuilding the core structure.

**Selected Architecture & Strategy**
Primary Architecture: Single-Stage Instance Segmentation (YOLOv8-seg or YOLO11-seg)
- Model Type: Single-stage Convolutional Neural Network (CNN) configured for instance segmentation (-seg variants).
- Outputs: Predicts pixel-accurate binary masks, bounding boxes, and shape class labels in a single execution pass.
- Compatibility with 3D/Topographic Data: Standard $2.5\text{D}$ AFM heightmaps ($N \times M$ grids of $Z$-height values) require no model architecture changes—even when moving to structures with larger vertical height variations.
- Completeness Score: Handled post-segmentation via Python/OpenCV algorithms comparing the output binary mask against an ideal geometric template (e.g., area ratio, convex hull analysis).

**Contingency Plan: Two-Stage Segmentation (Mask R-CNN)** \
If future datasets present extreme particle overcrowding, severe class imbalances, or loss of sharp corners, transition to Mask R-CNN (via Detectron2 or PyTorch) to utilize its sub-pixel RoIAlign spatial refinement.

**Dataset Strategy & Training Setup**
- Synthetic Data Generation + Real Data Fine-Tuning
    - Data Needs: 1,000–5,000 synthetically generated AFM images with automated polygon annotations, supplemented by 50–100 manually annotated real AFM images for fine-tuning.
    - Class Distribution: Balanced annotations across target shapes (~22.5% of total instances per class), plus 5%–10% empty background images (empty label files) to train the model to suppress AFM scan-line artifacts and substrate noise.
- Hardware & Execution
    - Model Variant: yolov8n-seg.pt or yolo11n-seg.pt (Nano variants) for lightweight execution.
    - CPU Local Execution: Fast inference (~50–150 ms/image). CPU training is viable for small synthetic tests.
    - Cloud GPU Alternative: Use free cloud platforms (Google Colab / Kaggle T4 GPUs) for fast fine-tuning.
- Handling Future Shapes: When adding new shapes, fine-tune the existing trained weights (best.pt) on a combined dataset containing both previous and new shape classes. This prevents catastrophic forgetting while leveraging learned low-level AFM features.

## Classigication

Starting with the following classification:
- rectangle
- triangle

Soon:
- stars
- smiley

## Models

- YOLOv8 / YOLOv11:
    - Grid Division: YOLO divides an input image into a grid (for example, 16x16 or 32x32 cells).
    - Feature Extraction: It uses Convolutional Layers (filters that detect lines, edges, surface gradients, and textures) to extract spatial features across the grid.
    - Parallel Predictions: For each grid cell, the network predicts:
        - Bounding Box coordinates (center point, width, height)
        - Class Probabilities (e.g., 95% triangle, 3% star, 2% smiley)
        - Instance Masks (for -seg variants): Generates the exact pixel-by-pixel outline for objects located within those cells.

- Mask R-CNN:
    - Feature Extraction (Backbone): The input image passes through a Convolutional Neural Network (such as ResNet) to extract high-level spatial feature maps (edges, gradients, textures).
    - Region Proposal Network (RPN) — Stage 1: A lightweight sub-network slides over the feature maps and proposes candidate regions (called Regions of Interest or RoIs) that likely contain objects. It filters out millions of empty background pixels right away.
    - Region ROI Pooling: The proposed candidate boxes are cropped from the feature map and resized to a fixed matrix size.
    - Classification & Box Refinement — Stage 2: Fully connected layers process each cropped region to output:
        - Class Label: What object is inside this box? (e.g., triangle, star).
        - Box Offset: Fine-tuned adjustments to make the bounding box fit tightly around the object.
        - Instance Segmentation: adds a small Fully Convolutional Network (FCN) branch. This branch takes each proposed box region and generates a binary pixel mask (0 or 1) to outline the exact object boundary.

## Data Sets

### Aim

Size:
- Training: 1000-5000 synthetic images + 50-100 real images
- Validation: 30-50 real images
- Testing: 30-50 real images

Balance:
- Measure balance by instance count, not image count: If one image contains 10 triangles and another contains 1 star, balancing the images (50/50) will still result in an imbalanced model (10 triangles vs. 1 star). Ensure the total number of labeled masks per class is balanced
- Tolerable variance: Neural networks do not require exact equality. A 20-30% spread between classes is usually fine. Major imbalances (e.g., 70% rectangles, 5% stars) cause the network to default to predicting the dominant class whenever it encounters ambiguous or degraded shapes.
- Negative images: 5-10%. Include images containing only AFM substrate noise, scan-line artifacts, or empty space, and leave their .txt label files completely empty.
- Balance the instances equaly in between the different shapes.


### Steps to Generate Synthetic AFM Data

1. Programmatically generate 2D shape masks (triangles, stars, smileys) with random rotations, scales, missing chunks (to simulate incompleteness), and overlaps.
2. Apply synthetic AFM noise (Gaussian noise, background height gradients, scan-line artifacts).
3. Export the images alongside their exact ground-truth mask coordinates automatically.

### Adding new shapes to the network

1. Start with model fine tuned from previous training on previous shapes
2. Merge the data sets of old and new shapes
3. Train for Fewer Epochs: Because the network already understands AFM topography, scan lines, and old shape geometries, fine-tuning on the combined dataset will converge in a fraction of the time compared to your initial training run.   
4. (Optional) Freeze Early Backbone Layers: If you want extra training speed or have very few images of the new shape, freeze the first few layers of the network (freeze=10). This forces the network to lock its low-level feature extraction filters while only adjusting the higher-level classification/segmentation heads.

# Publications using NNs for AFM images segmentation

1. Machine learning-based multidomain processing for texture-based image segmentation and analysis
Borodinov, N., Tsai, W.-Y., Korolkov, V. V., Balke, N., Kalinin, S. V., & Ovchinnikova, O. S.
(2020), Applied Physics Letters, 116(4), 044103
https://doi.org/10.1063/1.5135328

2.  (2021). Cell detection and segmentation in microscopy images with improved Mask R-CNN.
Fujita, S., & Han, X.-H. 
(2021) Lecture Notes in Computer Science, 58–70.
https://doi.org/10.1007/978-3-030-69756-3_5
Cited by: 70

3. Mask R-CNN.
He, K., Gkioxari, G., Dollár, P., & Girshick, R.
(2017) IEEE International Conference on Computer Vision (ICCV).
https://doi.org/10.1109/iccv.2017.322
Cited by: 52362

4. Deep learning for live cell shape detection and automated AFM navigation.
Rade, J., Zhang, J., Sarkar, S., Krishnamurthy, A., Ren, J., & Sarkar, A.
(2022) Bioengineering, 9(10), 522.
https://doi.org/10.3390/bioengineering9100522
Cited by: 40

5. Cellpose: a generalist algorithm for cellular segmentation.
Stringer, C., Wang, T., Michaelos, M., & Pachitariu, M.
(2020)
https://doi.org/10.1101/2020.02.02.931238
Cited by: 5364

6. Deep learning strategy for small dataset from atomic force microscopy mechano-imaging on macrophages phenotypes.
Wu, H., Zhang, L., Zhao, B., Yang, W., & Galluzzi, M.
(2023) Frontiers in Bioengineering and Biotechnology, 11.
https://doi.org/10.3389/fbioe.2023.1259979
Cited by: 13