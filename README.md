# PHAROS-AIF-MIH

## Abstract
We propose an approach for COVID-19 detection and severity assessment from chest CT scans. Our method leverages both 2.5D and 3D representations to capture both local and global patterns in the data. The 2.5D approach uses multiview slices of a CT scan and uses a DINOv3 backbone for feature extraction and downstream training. The 3D approach uses the 3D ResNet-18 model to learn feature representations using a Variance Risk Extrapolation pretraining and contrastive supervised finetuning. Benchmarking on datasets focused on multi-source robustness and gender bias, we present an ensemble of both approaches - reaching a MacroF1 score of [insert value here].

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

