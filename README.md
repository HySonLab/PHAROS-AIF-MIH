# PHAROS-AIF-MIH

## Abstract
We present a deep learning framework for COVID-19 detection and severity assessment from chest CT scans. Our method integrates both 2.5D and 3D representations to capture complementary slice-level and volumetric information. The 2.5D branch processes multi-view CT slices (axial, coronal, sagittal) using a DINOv3 backbone to extract robust visual features, while the 3D branch employs a ResNet-18 architecture to model volumetric context. To improve cross-source robustness, the 3D model is pretrained using Variance Risk Extrapolation (VREx) and further refined with supervised contrastive learning. Predictions from both branches are combined through an ensemble strategy. Experiments on the PHAROS-AIF-MIH benchmark demonstrate that the proposed approach achieves strong performance while maintaining robustness across multiple data sources and fairness-aware evaluation settings. Code is available at \url{https://github.com/HySonLab/PHAROS-AIF-MIH}.


## Dataset file structure

Unzip the competition datasets and place them in this structure below:

```
- data
    - task_1
        - 1st_challenge_test_set
            - test
        - covid1
        - covid2
        - non-covid1
        - non-covid2
        - non-covid3
        - Validation
            - val
                - covid
                - non-covid
        - train_covid.csv
        - train_non_covid.csv
        - validation_non_covid.csv
        - validation_non_covid.csv
    - task_2
        - 2nd_challenge_test_set
            - test_for_participants
        - train1
            - A
            - G
        - train2
            - covid
            - normal
        - Validation
            - val
                - A
                - G
                - covid
                - normal
```

## Preprocessing

`task1_preprocess.py` and `task2_preprocess.py` reads the task datasets and processes them by reconstructing a 3D CT scan from Axial view slices as such:
- Read scans, remove duplicates
- Concatinate the slices on top on of each other
- Resize into `128 x 128 x 128` volumes
- 3D Gaussian denoising and mask sharpen
- Normalize into [0, 255] grayscale values

The resulting volume provides the Coronal and Sagittal views that weren't previously available.

![](assets/multiview.png)

`make_task1_csv.py` and `make_task2_csv.py` generates CSVs for indexing the volumes and labels for fast implementation.

## 3D ResNet-18 (R3D-18) Approach

### Architecture
We implemented a 3D ResNet-18 model adapted for single-channel CT scan input:
- **Backbone**: Pretrained R3D-18 with modified first convolution layer (3→1 channels)
- **Input**: 128×128×128 volumetric CT scans
- **Output**: Multi-class classification head

### Training Strategy

#### Two-Stage Training Pipeline

**Stage 1: Domain Pretraining with VRE**
- **Objective**: Variance Risk Extrapolation (VRE) for domain generalization
- **Loss**: Cross-Entropy + λ×variance(per-domain losses)
- **Targets**: Source domain labels (gender for Task 2, data source for Task 1)
- **Duration**: 5 epochs
- **Learning Rate**: 1e-4 with Cosine Annealing

**Stage 2: Task-Specific Fine-tuning**
- **Objective**: Supervised classification with contrastive learning
- **Augmentation**: MixUp (α=0.4) for improved generalization
- **Loss**: Cross-Entropy + Supervised Contrastive Loss
- **Duration**: 20 epochs
- **Learning Rate**: 1e-5 with Cosine Annealing

#### Key Techniques
- **MixUp Data Augmentation**: Blends samples and labels for robustness
- **Supervised Contrastive Learning**: Enhances feature separation
- **Gradient Accumulation**: Enables effective training with limited memory
- **AMP (Automatic Mixed Precision)**: Accelerates training

### Results

#### Task 1: COVID-19 Detection (Binary Classification)
- **Validation Accuracy**: 87.01%
- **Macro F1-Score**: 0.7648
- **Per-Source Performance**:
  - Source 0: F1=0.8630
  - Source 1: F1=0.8408
  - Source 2: F1=0.4828
  - Source 3: F1=0.8725

#### Task 2: Multi-Class Classification (A, G, COVID, Normal)
- **Validation Accuracy**: 76.77%
- **Macro F1-Score**: 0.66.77
- **Per-Gender Performance**:
  - Male: F1=0.7249
  - Female: F1=0.6104

#### Class-wise Performance (Task 2)
- **Class A**: Precision=0.6901, Recall=0.9800, F1=0.8099
- **Class G**: Precision=0.7500, Recall=0.1200, F1=0.2069
- **COVID**: Precision=0.8462, Recall=0.8250, F1=0.8354
- **Normal**: Precision=0.8293, Recall=0.8500, F1=0.8395

### Analysis
- The VRE pretraining significantly improved cross-domain generalization
- MixUp augmentation enhanced robustness to domain shifts
- Strong performance on COVID detection but challenges with Class G (Ground Glass) classification
- Gender bias observed in Task 2, with better performance on male scans

### Technical Implementation
- **Framework**: PyTorch with torchvision.models.video.r3d_18
- **Hardware**: GPU training with CUDA AMP acceleration
- **Batch Size**: 32 samples per batch
- **Optimization**: AdamW optimizer with weight decay 1e-5

## 2.5D DINOv3 Approach

### Architecture
We implemented a 2.5D approach leveraging multi-view CT slice representations with DINOv3 backbone:
- **Backbone**: DINOv3 (Vision Transformer) pretrained on large-scale image datasets
- **Input**: Multi-view slices from CT volumes (Axial, Coronal, Sagittal planes)
- **Slice Selection**: Strategic sampling from key anatomical planes
- **Output**: Classification head for task-specific predictions

### Multi-View Strategy

#### View Extraction
- **Axial View**: Original CT slices (primary diagnostic view)
- **Coronal View**: Reconstructed from volume (frontal plane)
- **Sagittal View**: Reconstructed from volume (lateral plane)
- **Slice Sampling**: Uniform sampling with attention to lung regions

#### View Processing Pipeline
1. **Volume Reconstruction**: 128×128×128 CT volumes from preprocessed slices
2. **Multi-plane Extraction**: Generate orthogonal views from 3D volume
3. **Slice Selection**: Strategic sampling of representative slices per view
4. **Individual Processing**: Each view processed independently through DINOv3
5. **Feature Fusion**: Combine multi-view features for final classification

### DINOv3 Adaptation

#### Training Methodology
- **Transfer Learning**: Leverage large-scale visual pretraining
- **Multi-view Consistency**: Encourage consistent predictions across views
- **Attention Mechanisms**: Focus on relevant lung regions
- **Domain Adaptation**: Specific fine-tuning for medical imaging

### Data Augmentation
- **Spatial Augmentations**: Rotation, flipping, scaling for each view
- **Intensity Augmentations**: Contrast adjustment, noise injection
- **View-specific Augmentations**: Tailored transformations per anatomical plane
- **Multi-view Consistency**: Ensure coherent augmentations across views

### Feature Fusion Strategies

#### Early Fusion
- **Pixel-level**: Combine multi-view slices before backbone processing
- **Channel Stacking**: Treat views as different input channels
- **Spatial Alignment**: Ensure proper geometric correspondence

#### Late Fusion
- **Feature-level**: Combine DINOv3 embeddings from each view
- **Attention-weighted**: Learn importance weights for each view
- **Ensemble Methods**: Combine predictions from view-specific classifiers

#### Hierarchical Fusion
- **Multi-scale**: Features at different transformer layers
- **Cross-view Attention**: Learn inter-view relationships
- **Progressive Integration**: Gradual fusion of multi-view information

### Implementation Details

#### Technical Specifications
- **Framework**: PyTorch with transformers library
- **DINOv3 Variant**: Base/large model depending on computational budget
- **Input Resolution**: 224×224 per slice (standard ImageNet preprocessing)
- **Batch Processing**: Multiple views processed in parallel
- **Memory Management**: Efficient handling of multi-view data

#### Training Pipeline
- **Stage 1**: Freeze backbone, train classification heads only
- **Stage 2**: Progressive unfreezing of transformer layers
- **Stage 3**: End-to-end fine-tuning with reduced learning rates
- **Validation**: Multi-view consistency checks and performance monitoring

#### Optimization Strategy
- **Learning Rate Schedule**: Cosine annealing with warmup
- **Weight Decay**: Regularization for transformer parameters
- **Gradient Clipping**: Stabilize training of deep networks
- **Mixed Precision**: Accelerate training while maintaining accuracy

### Hyperparameters

#### Model Architecture
- **DINOv3 Variant**: Base model (12 layers, 12 heads, 768 hidden dim)
- **Input Resolution**: 224×224 pixels per slice
- **Patch Size**: 16×16 pixels (14×14 patches per image)
- **Number of Views**: 3 (Axial, Coronal, Sagittal)
- **Slices per View**: 8-12 representative slices
- **Classification Head**: Linear layer(s) with dropout

#### Training Hyperparameters
- **Batch Size**: 32 samples (8-10 views per sample)
- **Learning Rate**: 1e-4 (backbone), 1e-3 (classification head)
- **Learning Rate Schedule**: Cosine annealing with 5% warmup
- **Weight Decay**: 1e-4 for backbone, 1e-5 for classification layers
- **Gradient Clipping**: 1.0 max norm
- **Training Epochs**: 
  - Stage 1 (frozen backbone): 10 epochs
  - Stage 2 (partial unfreeze): 15 epochs  
  - Stage 3 (full fine-tune): 20 epochs