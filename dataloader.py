"""
python -m pip install --upgrade huggingface_hub
pip install torchmil
https://torchmil.readthedocs.io/en/latest/api/datasets/camelyon16mil_dataset/
"""
import glob
import os
import tarfile
from huggingface_hub import snapshot_download
import matplotlib.pyplot as plt
from torchmil.datasets import CAMELYON16MILDataset
from tqdm import tqdm
import shutil
import torch
import numpy as np
import torch.utils.data as data_utils

import glob
import os
import tarfile
from huggingface_hub import snapshot_download
import matplotlib.pyplot as plt
from torchmil.datasets import CAMELYON16MILDataset
from tqdm import tqdm

def download_camelyon16(root_dir: str = "./datasets"):
    """Download Camelyon16 dataset from Hugging Face."""
    print("Downloading dataset from Hugging Face...")
    snapshot_download(
        repo_id="torchmil/Camelyon16_MIL",
        repo_type="dataset",
        local_dir=root_dir,
    )
    print("Download completed.")


def extract_dataset_archives(root_dir: str = "./datasets"):
    """Extract all downloaded .tar.gz files into their respective folders."""
    tar_files = glob.glob(
        os.path.join(root_dir, "**", "*.tar.gz"), recursive=True
    )
    if not tar_files:
        print("No .tar.gz files found to extract.")
        return

    for tar_path in tqdm(tar_files, desc="Extracting dataset"):
        extract_path = os.path.dirname(tar_path)
        tqdm.write(f"Extracting {tar_path} to {extract_path}...")
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(path=extract_path)
    print("Extraction complete!")

def repair_and_extract_camelyon16(search_dir="./datasets"):
    print("--- Searching for compressed archives ---")
    tar_files = glob.glob(
        os.path.join(search_dir, "**", "*.tar.gz"), recursive=True
    )

    if not tar_files:
        print("No .tar.gz archives found.")
        return

    for tar_path in tar_files:
        parent_dir = os.path.dirname(tar_path)
        archive_name = os.path.basename(tar_path)

        # Determine output folder name (e.g., 'features_UNI' from 'features_UNI.tar.gz')
        folder_name = archive_name.replace(".tar.gz", "")
        extract_target_dir = os.path.join(parent_dir, folder_name)

        print(f"\nExtracting {archive_name} -> {extract_target_dir}...")
        os.makedirs(extract_target_dir, exist_ok=True)

        with tarfile.open(tar_path, "r:gz") as tar:
            members = tar.getmembers()
            for member in tqdm(members, desc=f"Extracting {folder_name}"):
                tar.extract(member, path=extract_target_dir)

        # Fix Windows nested folder duplication (e.g., features_UNI/features_UNI/*.npy)
        nested_dir = os.path.join(extract_target_dir, folder_name)
        if os.path.exists(nested_dir) and os.path.isdir(nested_dir):
            print(f"Fixing nested directory structure for {folder_name}...")
            for item in os.listdir(nested_dir):
                src = os.path.join(nested_dir, item)
                dst = os.path.join(extract_target_dir, item)
                shutil.move(src, dst)
            os.rmdir(nested_dir)

    print("\n[SUCCESS] All archives extracted and directory structure repaired!")

"""Locates the parent directory containing BOTH 'splits.csv' and a 'patches_*' directory."""
def get_correct_root(search_dir="./datasets"):
    splits = glob.glob(
        os.path.join(search_dir, "**", "splits.csv"), recursive=True
    )

    for split_path in splits:
        candidate_root = os.path.dirname(split_path)
        # Check if there is a patches directory inside
        subdirs = [
            d
            for d in os.listdir(candidate_root)
            if os.path.isdir(os.path.join(candidate_root, d))
        ]
        if any(d.startswith("patches_") for d in subdirs):
            return candidate_root

    # Fallback to first splits.csv parent if patches directory naming differs
    if splits:
        return os.path.dirname(splits[0])

    raise FileNotFoundError("Could not locate splits.csv!")

"""Visualize dataset"""
def visualize_dataset(root: str, target_label: str= None):
    if target_label is not None and target_label not in [0, 1]:
        raise Exception(f"Input only 0 or 1 for target label")
    
    dataset_root = get_correct_root(root)
    # Load dataset with instance-level labels requested (y_inst)
    dataset = CAMELYON16MILDataset(
        root=dataset_root,
        features="UNI",
        partition="train",
        verbose= True,
        bag_keys=["X", "Y", "y_inst", "coords"],
    )

    print(f"\nTotal bags loaded: {len(dataset)}")

    for idx, sample in enumerate(dataset):
        features = sample["X"]  # Shape: (Num_Patches, Feature_Dim)
        label = sample["Y"]  # Slide-level binary label (1 = Tumor)
        coords = sample["coords"]  # Spatial coordinates (Num_Patches, 2)
        y_inst = sample["y_inst"]  # Instance/Patch labels (0 = Normal, 1 = Tumor)

        print(f"\n--- Visualizing Positive Slide Sample #{idx} ---")
        print(f"Features shape: {features.shape}")
        print(f"Slide Label: {label}")
        print(f"Coordinates shape: {coords.shape}")
        print(f"y_inst size: {y_inst.size()}")
        print(f"y_inst unique classes: {torch.unique(y_inst)}")

        # Convert Tensors to NumPy
        if isinstance(features, torch.Tensor):
            features = features.numpy()
        if isinstance(coords, torch.Tensor):
            coords = coords.numpy()
        if isinstance(y_inst, torch.Tensor):
            y_inst = y_inst.numpy()

        label_val = label.item() if isinstance(label, torch.Tensor) else label

        # Filter by target_label if specified ---
        if target_label is not None and label_val != target_label:
            continue

        label_text = ("Tumor / Metastasis" if label_val == 1 else "Normal Tissue")

        # Separate coordinates by patch label class
        normal_mask = y_inst == 0
        tumor_mask = y_inst == 1

        # Calculate feature magnitude (L2 norm) for right-hand heatmap
        patch_intensity = np.linalg.norm(features, axis=1)

        # Create Side-by-Side Figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # --- PANEL 1: Ground-Truth Patch Map with Red Dots for Tumor ---
        # Plot Normal Tissue Patches (y_inst = 0)
        ax1.scatter(
            coords[normal_mask, 0],
            coords[normal_mask, 1],
            c="lightslategrey",
            s=2,
            alpha=0.3,
            label="Normal Patch (y_inst=0)",
        )

        # Plot Tumor / Metastatic Patches (y_inst = 1) in Red
        ax1.scatter(
            coords[tumor_mask, 0],
            coords[tumor_mask, 1],
            c="crimson",
            s=4,
            alpha=0.8,
            label="Tumor Patch (y_inst=1)",
        )

        ax1.invert_yaxis()  # Pathology grid origin (0,0) is top-left
        ax1.set_title(
            f"Slide #{idx} Patch Annotations\nSlide Status: {label_val} ({label_text})",
            fontsize=11,
            fontweight="bold",
        )
        ax1.set_xlabel("X Coordinate")
        ax1.set_ylabel("Y Coordinate")
        ax1.grid(True, linestyle="--", alpha=0.3)

        # Add Legend to Left Plot
        ax1.legend(
            loc="upper right", frameon=True, facecolor="white", framealpha=0.9
        )

        # --- PANEL 2: Feature Density Heatmap ---
        sc = ax2.scatter(
            coords[:, 0],
            coords[:, 1],
            c=patch_intensity,
            cmap="magma",
            s=3,
            alpha=0.8,
        )
        ax2.invert_yaxis()
        ax2.set_title(
            f"Feature Vector Magnitude Heatmap\n({features.shape[0]} Patches × {features.shape[1]}-D UNI Embeddings)",
            fontsize=11,
            fontweight="bold",
        )
        ax2.set_xlabel("X Coordinate")
        ax2.set_ylabel("Y Coordinate")
        ax2.grid(True, linestyle="--", alpha=0.3)

        cbar = plt.colorbar(sc, ax=ax2)
        cbar.set_label("UNI Feature L2 Norm", rotation=270, labelpad=15)

        plt.suptitle(
            f"Camelyon16 Slide Inspection — Sample Index #{idx}",
            fontsize=13,
            fontweight="bold",
        )
        plt.tight_layout()
        plt.show()

"""
Camelyon16 PyTorch Dataset designed for ABMIL model training.
Returns:
    bag: Feature matrix tensor of shape (N, feature_dim) where N is variable patch count.
    label: A 2-element list [slide_label, instance_labels] mimicking MnistBags:
            - slide_label: Tensor scalar containing slide label (0: Normal, 1: Tumor)
            - instance_labels: Tensor containing per-patch binary labels (0 or 1)
"""
class Camelyon16Bags(data_utils.Dataset):
    def __init__(self, root="./datasets", train=True, features="UNI"):
        self.root = get_correct_root(root)
        self.train = train
        self.partition = "train" if self.train else "test"

        # Load underlying torchmil dataset
        self.dataset = CAMELYON16MILDataset(
            root=self.root,
            features=features,
            partition=self.partition,
            bag_keys=["X", "Y", "y_inst", "coords"],
            verbose=False,
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = self.dataset[index]
        
        bag = sample["X"]         # Features Tensor (N_patches, Feature_Dim)
        slide_label = sample["Y"] # Slide level binary label (0 or 1)
        y_inst = sample["y_inst"] # Instance level binary labels
        coords = sample["coords"] # Spatial coordinates (N_patches, 2)

        # Ensure slide_label is a scalar tensor
        if not isinstance(slide_label, torch.Tensor):
            slide_label = torch.tensor(slide_label, dtype=torch.long)
        else:
            slide_label = slide_label.squeeze().long()

        # Format label array exactly matching the MnistBags interface:
        # [max_label_of_bag (slide label), instance_labels_array]
        label = [slide_label, y_inst, coords]

        return bag, label
    

if __name__ == "__main__":
    '''
    # 1. Download dataset (UNCOMMENT if downloading for the first time)
    download_camelyon16(root_dir="./datasets")

    # 2. Extract dataset archives (MUST RUN AT LEAST ONCE to unpack .npy files!)
    extract_dataset_archives("./datasets")
    # 3. Repair dataset
    repair_and_extract_camelyon16("./datasets")
    '''
    # 4. Visualize the dataset
    visualize_dataset(root= "./datasets", target_label= 1)
    
    # Test DataLoaders matching the MnistBags structure
    train_loader = data_utils.DataLoader(
        Camelyon16Bags(root="./datasets", train=True, features="UNI"),
        batch_size=1,  # Must be 1 due to varying patch sizes (N) across slides
        shuffle=True,
    )

    test_loader = data_utils.DataLoader(
        Camelyon16Bags(root="./datasets", train=False, features="UNI"),
        batch_size=1,
        shuffle=False,
    )

    # Calculate Train Statistics
    len_bag_list_train = []
    camelyon_bags_train = 0
    for batch_idx, (bag, label) in enumerate(train_loader):
        # Unpack batch dimension added by DataLoader (batch_size=1)
        bag = bag.squeeze(0)  # Shape: (N, 1024)
        slide_label = label[0].item()
        
        len_bag_list_train.append(bag.size(0))
        camelyon_bags_train += slide_label

    print('Number of positive train bags: {}/{}\n'
          'Number of instances per bag, mean: {:.2f}, max: {}, min: {}\n'.format(
          camelyon_bags_train, len(train_loader),
          np.mean(len_bag_list_train), np.max(len_bag_list_train), np.min(len_bag_list_train)))

    # Calculate Test Statistics
    len_bag_list_test = []
    camelyon_bags_test = 0
    for batch_idx, (bag, label) in enumerate(test_loader):
        bag = bag.squeeze(0)
        slide_label = label[0].item()
        
        len_bag_list_test.append(bag.size(0))
        camelyon_bags_test += slide_label

    print('Number of positive test bags: {}/{}\n'
          'Number of instances per bag, mean: {:.2f}, max: {}, min: {}\n'.format(
          camelyon_bags_test, len(test_loader),
          np.mean(len_bag_list_test), np.max(len_bag_list_test), np.min(len_bag_list_test)))