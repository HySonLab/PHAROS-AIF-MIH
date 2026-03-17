# PHAROS-AIF-MIH

## Abstract
We present a deep learning framework for COVID-19 detection and disease classification from chest CT scans using both 2.5D and 3D representations. The 2.5D branch extracts multi-view slice features (axial, coronal, sagittal) using a DINOv3 vision transformer, while the 3D branch uses a ResNet-18 architecture to model volumetric context. The 3D model is pretrained with Variance Risk Extrapolation (VREx) and further refined using supervised contrastive learning to improve cross-source robustness. Predictions from both branches are combined using logit-level ensembling. On the PHAROS-AIF-MIH benchmark, the ensemble achieves 95.13% accuracy and 0.8321 Macro F1 for COVID-19 detection. For multi-class disease classification, the 2.5D DINOv3 model achieves the best performance with 76.77% accuracy and 0.7230 Macro F1. These results highlight the benefits of combining pretrained slice-based representations with volumetric modeling for multi-source medical imaging analysis.

Repository: https://github.com/HySonLab/PHAROS-AIF-MIH


## Dataset Structure

After downloading the competition datasets, organize them as follows:

```

data/
    task_1/
        1st_challenge_test_set/test
    covid1/
    covid2/
    non-covid1/
    non-covid2/
    non-covid3/
    Validation/val/
        covid/
        non-covid/
    train_covid.csv
    train_non_covid.csv
    validation_non_covid.csv

    task_2/
        2nd_challenge_test_set/test_for_participants
        train1/
            A/
            G/
        train2/
            covid/
            normal/
        Validation/val/
            A/
            G/
            covid/
            normal/
```


## Preprocessing

`task1_preprocess.py` and `task2_preprocess.py` reconstruct 3D CT volumes from axial slices.

Processing steps:
- remove duplicate slices
- stack slices into a 3D volume
- resize to 128 × 128 × 128
- apply 3D Gaussian denoising and mask sharpening
- normalize to grayscale [0, 255]

The reconstructed volume enables extraction of coronal and sagittal views.

`make_task1_csv.py` and `make_task2_csv.py` generate CSV files indexing volumes and labels.

![](assets/multiview.png)


## 3D ResNet-18 Approach

### Training Pipeline

| Stage | Objective | Loss | Details |
|---|---|---|---|
| VREx Pretraining | Domain generalization | Cross-entropy + variance of domain losses | Domains: data source (Task 1) or gender (Task 2); 5 epochs; LR = 1e-4 |
| Task Fine-tuning | Task-specific learning | Cross-entropy + supervised contrastive loss | MixUp (α = 0.4); 20 epochs; LR = 1e-5 |


### Results

Task 1 (COVID detection)

| Metric | Value |
|---|---|
| Accuracy | 87.01% |
| Macro F1 | 0.7648 |

Per-source F1

| Source | F1 |
|---|---|
| Source 0 | 0.8630 |
| Source 1 | 0.8408 |
| Source 2 | 0.4828 |
| Source 3 | 0.8725 |

Task 2 (A, G, COVID, Normal)

| Metric | Value |
|---|---|
| Accuracy | 76.77% |
| Macro F1 | 0.6677 |


## 2.5D DINOv3 Approach

### Training Strategy

| Stage | Description |
|---|---|
| Stage 1 | Train classification head with frozen backbone |
| Stage 2 | Partially unfreeze transformer layers |
| Stage 3 | End-to-end fine-tuning |

### Results

Task 1

| Metric | Value |
|---|---|
| Accuracy | 93.51% |
| Macro F1 | 0.8221 |

Per-source F1

| Source | F1 |
|---|---|
| Source 0 | 0.9430 |
| Source 1 | 0.9431 |
| Source 2 | 0.4828 |
| Source 3 | 0.9194 |

Task 2

| Metric | Value |
|---|---|
| Accuracy | 76.77% |
| Macro F1 | 0.7229 |


## Ensemble

Predictions from the 2.5D and 3D models are combined using logit-level averaging.

| Task | Accuracy | Macro F1 |
|---|---|---|
| Task 1 (COVID detection) | 95.13% | 0.8321 |

## If our work is useful, please cite us!
```bibtex
@misc{yang2026halfway3densembling25d,
      title={Halfway to 3D: Ensembling 2.5D and 3D Models for Robust COVID-19 CT Diagnosis}, 
      author={Tuan-Anh Yang and Bao V. Q. Bui and Chanh-Quang Vo-Van and Truong-Son Hy},
      year={2026},
      eprint={2603.14832},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2603.14832}, 
}
```
