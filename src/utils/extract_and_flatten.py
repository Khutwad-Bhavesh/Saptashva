import os
import zipfile
import shutil
from pathlib import Path

def extract_and_flatten(base_dir):
    base_path = Path(base_dir)
    
    print("1. Extracting all .zip files...")
    # Find all zip files recursively
    for zip_file in base_path.rglob('*.zip'):
        print(f"Extracting: {zip_file.name}")
        try:
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                # Extract in the same directory as the zip file temporarily
                zip_ref.extractall(zip_file.parent)
        except Exception as e:
            print(f"Failed to extract {zip_file.name}: {e}")

    print("\n2. Flattening directories and moving data files to the root of datasets/...")
    # Move all relevant files to the root of base_dir
    extensions_to_keep = ['.fits', '.lc.gz', '.pi.gz', '.gti.gz']
    
    for ext in extensions_to_keep:
        # Search for files with this extension recursively
        for file_path in base_path.rglob(f'*{ext}'):
            # Avoid moving files that are already in the base directory
            if file_path.parent != base_path:
                dest_path = base_path / file_path.name
                
                # Handle naming collisions
                counter = 1
                while dest_path.exists():
                    dest_path = base_path / f"{file_path.stem}_{counter}{file_path.suffix}"
                    counter += 1
                    
                print(f"Moving {file_path.name} to {base_path}")
                shutil.move(str(file_path), str(dest_path))

    print("\n3. Cleaning up empty folders and original zip files...")
    # Remove all the zip files to save space
    for zip_file in base_path.rglob('*.zip'):
        os.remove(zip_file)
        
    # Recursively remove empty directories (bottom-up)
    for root, dirs, files in os.walk(base_dir, topdown=False):
        for name in dirs:
            dir_path = os.path.join(root, name)
            try:
                os.rmdir(dir_path) # rmdir only works if the directory is empty
            except OSError:
                pass # Directory not empty, which means it might contain other files we didn't move
                
    print("\nExtraction and flattening complete!")

if __name__ == '__main__':
    dataset_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'datasets'))
    if os.path.exists(dataset_dir):
        extract_and_flatten(dataset_dir)
    else:
        print(f"Could not find datasets directory at {dataset_dir}")
