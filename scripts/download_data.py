import os
import shutil
from sklearn.model_selection import train_test_split
import subprocess
import kagglehub

def download_dataset(dataset_name="ayuraj/asl-dataset", data_dir="data/raw"):
    """
    Downloads a Kaggle dataset using kagglehub and extracts it.
    Args:
        dataset_name (str): Kaggle dataset name (e.g., 'username/dataset-name').
        data_dir (str): Path to store the raw dataset.
    """
    # Create data directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)

    # Download dataset using kagglehub
    print(f"Downloading dataset '{dataset_name}'...")
    path = kagglehub.dataset_download(dataset_name)
    
    # Move the downloaded dataset to the desired directory
    shutil.move(path, data_dir)
    print(f"Dataset downloaded and stored in '{data_dir}'")

    # If dataset is compressed, extract it
    zip_path = os.path.join(data_dir, f"{dataset_name.split('/')[-1]}.zip")
    if zip_path.endswith(".zip"):
        print(f"Extracting dataset to '{data_dir}'...")
        subprocess.run(f"unzip {zip_path} -d {data_dir}", shell=True, check=True)
        os.remove(zip_path)
        print("Dataset extracted successfully!")

def split_dataset(source_dir="data/raw", train_dir="data/train", val_dir="data/val", test_dir="data/test"):
    """
    Splits the dataset into training, validation, and test sets.
    Args:
        source_dir (str): Directory containing raw dataset organized in class-wise subfolders.
        train_dir (str): Directory to store training data.
        val_dir (str): Directory to store validation data.
        test_dir (str): Directory to store test data.
    """
    # Ratios
    train_ratio = 0.7
    val_ratio = 0.2
    test_ratio = 0.1

    # Create target directories
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Process each class
    print(f"Splitting dataset from '{source_dir}'...")
    for class_name in os.listdir(source_dir):
        class_path = os.path.join(source_dir, class_name)
        if os.path.isdir(class_path):
            # Gather images in the class directory
            images = [os.path.join(class_path, img) for img in os.listdir(class_path) if img.endswith(('.jpg', '.png'))]

            # Split images into train, val, and test sets
            train_imgs, temp_imgs = train_test_split(images, test_size=(1 - train_ratio), random_state=42)
            val_imgs, test_imgs = train_test_split(temp_imgs, test_size=(test_ratio / (val_ratio + test_ratio)), random_state=42)

            # Create subdirectories for the class in each target folder
            os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
            os.makedirs(os.path.join(val_dir, class_name), exist_ok=True)
            os.makedirs(os.path.join(test_dir, class_name), exist_ok=True)

            # Copy images to respective directories
            for img in train_imgs:
                shutil.copy(img, os.path.join(train_dir, class_name))
            for img in val_imgs:
                shutil.copy(img, os.path.join(val_dir, class_name))
            for img in test_imgs:
                shutil.copy(img, os.path.join(test_dir, class_name))

    print("Dataset split into train, val, and test successfully!")


if __name__ == "__main__":
    
    DATASET_NAME = "ayuraj/asl-dataset"

    # Paths
    RAW_DATA_DIR = "data/raw"
    TRAIN_DIR = "data/train"
    VAL_DIR = "data/val"
    TEST_DIR = "data/test"

    # Step 1: Download the dataset
    download_dataset(dataset_name=DATASET_NAME, data_dir=RAW_DATA_DIR)

    # Step 2: Split the dataset into train, val, and test sets
    split_dataset(source_dir=RAW_DATA_DIR, train_dir=TRAIN_DIR, val_dir=VAL_DIR, test_dir=TEST_DIR)
