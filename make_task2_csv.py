import csv
import os
from pathlib import Path
from typing import List

DATA_DIR = Path(__file__).parent / "data"


def build_task2_npy_csv(output_path: Path) -> None:
    """Create a CSV index for Task 2 npy files using folder names as labels.

    Infers for each `.npy` file:
        - ct_scan_name: stem of the filename (e.g. ct_scan_0)
        - npy_path: path relative to the task_2 folder
        - class_label: one of {A, G, covid, normal} from the folder path
        - gender_label: one of {male, female} from the folder path
        - split: {train, val} inferred from whether path contains "Validation/val"
    """

    task2_root = Path(__file__).parent
    all_rows: List[dict] = []

    for dirpath, _, filenames in os.walk(DATA_DIR):
        dir_path = Path(dirpath)
        rel_dir = dir_path.relative_to(task2_root)

        # Infer split
        if "Validation" in rel_dir.parts or "val" in rel_dir.parts:
            split = "val"
        elif "2nd_challenge_test_set" in rel_dir.parts:
            split = "test"
        else:
            split = "train"

        # Infer class label from path parts in priority order
        parts = {p.lower() for p in rel_dir.parts}
        if "covid_npy" in parts:
            class_label = "covid"
        elif "normal_npy" in parts:
            class_label = "normal"
        elif "a_npy" in parts or "a" in parts:
            class_label = "A"
        elif "g_npy" in parts or "g" in parts:
            class_label = "G"
        else:
            class_label = "unknown"

        # Infer gender
        if "male" in parts:
            gender_label = "male"
        elif "female" in parts:
            gender_label = "female"
        else:
            gender_label = "unknown"

        for fname in filenames:
            if not fname.lower().endswith(".npy"):
                continue

            scan_name = Path(fname).stem  # e.g. ct_scan_0
            full_path = dir_path / fname
            rel_path = full_path.relative_to(task2_root)

            all_rows.append(
                {
                    "ct_scan_name": scan_name,
                    "npy_path": str(rel_path).replace(os.sep, "/"),
                    "class_label": class_label,
                    "gender_label": gender_label,
                    "split": split,
                }
            )
    
    # Write CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["ct_scan_name", "npy_path", "class_label", "gender_label", "split"]
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)


if __name__ == "__main__":
    out_csv = Path(__file__).parent / "task2_npy_index.csv"
    build_task2_npy_csv(out_csv)
    print(f"Wrote {out_csv}")
