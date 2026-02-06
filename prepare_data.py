import os
import shutil
import subprocess
import sys

# --- AUTO-INSTALLER ---
try:
    import kagglehub
except ImportError:
    print("Kagglehub not found. Installing now...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "kagglehub"])
    import kagglehub

# 1. AUTHENTICATION
os.environ['KAGGLE_API_TOKEN'] = 'KGAT_62e550101568108988a6cb1fee981796'

# 2. CONFIGURATION
datasets = [
    "arindamsaha07/skin-diesease-image-dataset",
    "mohamedjadallah/skin-disease-image-dataset",
    "subirbiswas19/skin-disease-dataset"
]

# Use absolute paths for the Windows long-path prefix to work
BASE_OUTPUT_DIR = os.path.abspath("data")
# Windows magic prefix for long paths
WIN_PREFIX = "\\\\?\\" if os.name == 'nt' else ""

DISEASE_MAPPING = {
    "mel": "melanoma",
    "bcc": "basal_cell_carcinoma",
    "akiec": "actinic_keratosis",
    "nv": "nevus",
    "bkl": "benign_keratosis",
    "df": "dermatofibroma",
    "vasc": "vascular_lesion"
}

def prepare_and_merge():
    for ds in datasets:
        print(f"--- Downloading {ds} ---")
        # download() returns the path to the cached dataset
        cache_path = kagglehub.dataset_download(ds)
        
        for root, dirs, files in os.walk(cache_path):
            if files:
                # 1. Determine if this is part of a 'train' or 'test' split
                path_parts = root.lower().split(os.sep)
                split_type = "train_data" # Default
                if "test" in path_parts or "val" in path_parts or "validation" in path_parts:
                    split_type = "test_data"
                
                # 2. Identify the Disease Category
                raw_folder = os.path.basename(root).lower().strip()
                clean_name = DISEASE_MAPPING.get(raw_folder, "".join(e for e in raw_folder if e.isalnum()))
                
                # 3. Construct Target Path
                target_dir = os.path.join(BASE_OUTPUT_DIR, split_type, clean_name)
                os.makedirs(WIN_PREFIX + target_dir, exist_ok=True)
                
                # 4. Copy Files
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        source_file = os.path.join(root, file)
                        # Shortened unique name to prevent path length issues
                        ds_short = ds.split('/')[-1][:5]
                        unique_filename = f"{ds_short}_{file[:40]}" 
                        
                        dest_file = os.path.join(target_dir, unique_filename)
                        
                        try:
                            shutil.copy2(WIN_PREFIX + source_file, WIN_PREFIX + dest_file)
                        except Exception as e:
                            print(f"Error copying {file}: {e}")

    print(f"\nSUCCESS: Data organized in {BASE_OUTPUT_DIR}/train_data and {BASE_OUTPUT_DIR}/test_data")

if __name__ == "__main__":
    prepare_and_merge()