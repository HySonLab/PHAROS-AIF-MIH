import os
import numpy as np
from PIL import Image
from scipy.ndimage import zoom
from scipy.ndimage import gaussian_filter

def load_jpeg_volume(folder_path):
    # Sort files numerically (0.jpg, 1.jpg, 2.jpg, ...)
    files = sorted(
        [f for f in os.listdir(folder_path) 
            if f.lower().endswith(".jpg") and not f.startswith(".")],
        key=lambda x: int(os.path.splitext(x)[0])
    )

    slices = []
    target_hw = None  # (H, W)
    seen_slices = set()  # Track unique slice hashes to prevent duplicates
    
    for f in files:
        try:
            img_path = os.path.join(folder_path, f)
            img = Image.open(img_path).convert("L")  # grayscale
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            continue

        # Ensure all slices in this volume have the same 2D size
        if target_hw is None:
            arr = np.array(img, dtype=np.float32)
            target_hw = arr.shape  # (H, W)
        else:
            # PIL uses (W, H), so reverse when providing (H, W)
            img = img.resize((target_hw[1], target_hw[0]))
            arr = np.array(img, dtype=np.float32)

        # Check for duplicate slices using hash
        slice_hash = hash(arr.tobytes())
        if slice_hash in seen_slices:
            print(f"Skipping duplicate slice: {img_path}")
            continue
        
        seen_slices.add(slice_hash)
        slices.append(arr)

    volume = np.stack(slices, axis=0)  # (D, H, W)
    return volume


def resize_volume(volume, target_shape=(128, 128, 128)):
    D, H, W = volume.shape
    target_D, target_H, target_W = target_shape

    # Compute zoom factors
    zoom_factors = (
        target_D / D,
        target_H / H,
        target_W / W
    )

    # Trilinear interpolation (order=1)
    resized = zoom(volume, zoom_factors, order=1)
    return resized


def normalize_volume(volume):
    vmin, vmax = volume.min(), volume.max()
    if vmax > vmin:
        volume = (volume - vmin) / (vmax - vmin)
    else:
        volume = np.zeros_like(volume, dtype=np.float32)

    volume = (volume * 255.0).clip(0, 255)
    return volume.astype(np.uint8)

def sharpen_and_denoise(volume):
    volume = volume.astype(np.float32)

    # 1. Denoise with 3D Gaussian smoothing
    smooth = gaussian_filter(volume, sigma=1)

    # 2. Sharpen with unsharp masking
    sharpen_strength = 1.5
    volume = volume + sharpen_strength * (volume - smooth)

    # 3. Clip intensity range
    volume = np.clip(volume, 0, 255)

    return volume.astype(np.uint8)

def jpeg_folder_to_numpy(folder_path, target_shape=(128, 128, 128)):
    vol = load_jpeg_volume(folder_path)
    vol = resize_volume(vol, target_shape)
    vol = sharpen_and_denoise(vol)
    vol = normalize_volume(vol)
    return vol


def convert_all_scans_to_numpy(input_root, output_root, target_shape=(128, 128, 128)):
    """Walk input_root and convert every JPEG scan folder to a .npy file.

    Keeps the relative folder structure under output_root and saves each
    volume using the scan folder name, e.g.:

        input_root/class1/scanA/*.jpg -> output_root/class1/scanA/scanA.npy
    """

    input_root = os.path.abspath(input_root)
    output_root = os.path.abspath(output_root)

    for dirpath, dirnames, filenames in os.walk(input_root):
        # Check if this directory looks like a scan folder (contains .jpg files)
        jpg_files = [
            f for f in filenames
            if f.lower().endswith(".jpg") and not f.startswith(".")
        ]

        if not jpg_files:
            continue

        # Relative path from the input root to this scan folder
        rel_dir = os.path.relpath(dirpath, input_root)
        scan_name = os.path.basename(dirpath)

        # Corresponding output directory
        out_dir = os.path.join(output_root, rel_dir)
        os.makedirs(out_dir, exist_ok=True)

        # Full path for the .npy file, named after the scan folder
        out_path = os.path.join(out_dir, f"{scan_name}.npy")

        if os.path.exists(out_path):
            continue
        
        volume_np = jpeg_folder_to_numpy(dirpath, target_shape=target_shape)
        np.save(out_path, volume_np)


if __name__ == "__main__":
    # Example: adjust these paths to your actual task_1 scan root
    # input_root could be something like "covid1" or the folder that
    # contains all your scan subfolders.
    
    # input_root = "covid1"  # TODO: change to your actual input root
    # output_root = "covid1_npy"  # TODO: change to your desired output root

    # convert_all_scans_to_numpy(input_root, output_root, target_shape=(128, 256, 256))

    folders = [
        "data/covid1",
        "data/covid2",
        "data/non-covid1",
        "data/non-covid2",
        "data/non-covid3",
        "data/Validation/val/covid",
        "data/Validation/val/non-covid",
        "data/1st_challenge_test_set/test"
    ]
    
    for folder in folders:
        input_root = folder
        output_root = f"{folder}_npy"
        convert_all_scans_to_numpy(input_root, output_root, target_shape=(128, 128, 128))