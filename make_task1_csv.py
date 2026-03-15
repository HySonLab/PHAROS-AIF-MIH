import csv
import os
from pathlib import Path
from typing import Dict, List

DATA_DIR = Path(__file__).parent / "data"


def index_npy_paths(root: Path) -> Dict[str, str]:
    """Walk `root` and map ct_scan_name -> relative npy path.

    Assumes files are named like ct_scan_X/ct_scan_X.npy somewhere under root.
    """
    mapping: Dict[str, str] = {}
    for dirpath, _, filenames in os.walk(root):
        for fname in filenames:
            if not fname.lower().endswith(".npy"):
                continue
            scan_name = Path(fname).stem  # e.g. ct_scan_0
            full_path = Path(dirpath) / fname
            rel_path = full_path.relative_to(root.parent)  # relative to task_1/
            # If duplicates exist, keep the first one we see.
            split = "val" if "val" in dirpath else "train"
            class_label = "non-covid" if "non-covid" in dirpath else "covid"

            mapping[f"{scan_name}_{class_label}_{split}"] = str(rel_path).replace(os.sep, "/")

    return mapping


def read_label_csv(path: Path, class_label: str, split: str) -> List[dict]:
    rows: List[dict] = []
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ct_name = row["ct_scan_name"]
            source = row["data_centre"]
            rows.append(
                {
                    "ct_scan_name": ct_name,
                    "source_label": source,
                    "class_label": class_label,
                    "split": split,
                }
            )
    return rows


def build_task1_npy_csv(output_path: Path) -> None:
    # Index all npy files under data/
    npy_index = index_npy_paths(DATA_DIR)

    all_rows: List[dict] = []

    # Train CSVs
    train_covid_csv = DATA_DIR / "train_covid.csv"
    train_non_covid_csv = DATA_DIR / "train_non_covid.csv"
    val_covid_csv = DATA_DIR / "validation_covid.csv"
    val_non_covid_csv = DATA_DIR / "validation_non_covid.csv"

    for csv_path, class_label, split in [
        (train_covid_csv, "covid", "train"),
        (train_non_covid_csv, "non-covid", "train"),
        (val_covid_csv, "covid", "val"),
        (val_non_covid_csv, "non-covid", "val"),
    ]:
        if not csv_path.exists():
            continue
        rows = read_label_csv(csv_path, class_label=class_label, split=split)
        for row in rows:
            scan_name = row["ct_scan_name"]
            npy_path = npy_index.get(f"{scan_name}_{class_label}_{split}")
            if npy_path is None:
                # Skip if we cannot find a corresponding npy file
                continue
            row["npy_path"] = npy_path
            all_rows.append(row)

    test_path = DATA_DIR / "1st_challenge_test_set" / "test_npy"
    task1_root = Path(__file__).parent
    
    if test_path.exists():
        # test path has folders of scan name, inside is npy file
        for folder in test_path.iterdir():
            if folder.is_dir():
                for file in folder.iterdir():
                    if file.suffix == ".npy":
                        scan_name = folder.name
                        dir_path = Path(file)
                        rel_dir = dir_path.relative_to(task1_root)
                        all_rows.append(
                            {
                                "ct_scan_name": scan_name,
                                "npy_path": str(rel_dir).replace(os.sep, "/"),
                                "source_label": "unknown",
                                "class_label": "unknown",
                                "split": "test",
                            }
                        )

    # Write consolidated CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["ct_scan_name", "npy_path", "source_label", "class_label", "split"]
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)


if __name__ == "__main__":
    out_csv = Path(__file__).parent / "task1_npy_index.csv"
    build_task1_npy_csv(out_csv)
    print(f"Wrote {out_csv}")
