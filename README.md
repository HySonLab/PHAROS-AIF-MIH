# PHAROS-AIF-MIH

## Abstract
We propose a deep learning framework for COVID-19 detection and disease classification from chest CT scans that integrates both 2.5D and 3D representations to capture complementary slice-level and volumetric information. The 2.5D branch processes multi-view CT slices (axial, coronal, sagittal) using a DINOv3 vision transformer, while the 3D branch employs a ResNet-18 architecture pretrained with Variance Risk Extrapolation (VREx) and further refined with supervised contrastive learning to improve cross-source robustness. Predictions from both branches are combined via logit-level ensembling. 

Experiments on the PHAROS-AIF-MIH benchmark demonstrate the effectiveness of the proposed approach. On the test set, our method achieved runner-up in the Multi-Source COVID-19 Detection Challenge, with the best ensemble reaching a Macro F1-score of 0.751. For the Fair Disease Diagnosis Challenge, our approach ranked third place, achieving a best Macro F1-score of 0.633 with improved performance balance across genders. These results highlight the benefits of combining pretrained slice-based representations with volumetric modeling, as well as the importance of ensemble strategies for improving robustness and fairness in multi-source medical imaging tasks.

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

## Test set results

### Multi-Source COVID-19 Detection (Task 1)

| Method         | Macro F1  | H1        | H2        | H3    | H4    |
| -------------- | --------- | --------- | --------- | ----- | ----- |
| 2.5D           | 0.741     | 0.910     | 0.691     | 0.493 | 0.869 |
| 3D             | 0.699     | 0.825     | 0.608     | 0.495 | 0.868 |
| Ensemble (0.3) | 0.739     | 0.895     | 0.673     | 0.496 | 0.892 |
| Ensemble (0.5) | **0.751** | **0.914** | **0.712** | 0.495 | 0.883 |
| Ensemble (0.7) | 0.744     | 0.912     | 0.696     | 0.495 | 0.873 |

### Fair Disease Diagnosis (Task 2)

| Method         | Macro F1  | Female    | Male      |
| -------------- | --------- | --------- | --------- |
| 2.5D           | 0.555     | 0.709     | 0.402     |
| 3D             | 0.572     | **0.756** | 0.388     |
| Ensemble (0.3) | 0.568     | 0.732     | 0.403     |
| Ensemble (0.5) | 0.561     | 0.718     | 0.403     |
| Ensemble (0.7) | **0.633** | 0.709     | **0.557** |

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
