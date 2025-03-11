#!/usr/bin/env python3
"""
Script to download RAM++ model file directly using the recommended approach.
"""
import os
import sys
import time
import warnings
from pathlib import Path

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

def main():
    """Download RAM++ model file using the recommended approach."""
    print("Setting up RAM++ model...")
    sys.setrecursionlimit(15000)
    
    # Force CPU usage
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["FORCE_CPU"] = "1"
    
    # Set up paths
    cache_dir = "/app/data/ram_models"
    os.makedirs(cache_dir, exist_ok=True)
    
    try:
        print("Downloading RAM++ model using huggingface_hub...")
        
        # Import the necessary modules
        from huggingface_hub import hf_hub_download
        
        # Download the model file
        model_path = hf_hub_download(
            repo_id="xinyu1205/recognize-anything-plus-model",
            filename="ram_plus_swin_large_14m.pth",
            cache_dir=cache_dir
        )
        
        print(f"RAM++ model downloaded successfully to: {model_path}")
        
        # Create a symlink or copy to the expected location if needed
        destination = os.path.join(cache_dir, "ram_plus_swin_large_14m.pth")
        if not os.path.exists(destination):
            print(f"Creating symlink from {model_path} to {destination}")
            if os.path.islink(destination):
                os.unlink(destination)
            os.symlink(model_path, destination)
        
        # Verify the download
        if os.path.exists(destination):
            file_size = os.path.getsize(destination)
            print(f"RAM++ model file size: {file_size / (1024*1024):.1f} MB")
            print("RAM++ model setup successfully")
            return True
        
    except Exception as e:
        print(f"Error downloading RAM++ model: {str(e)}")
        
    # If we get here, something went wrong
    print("WARNING: Failed to download the RAM++ model file")
    # Create a placeholder file to indicate we tried
    with open(os.path.join(cache_dir, "DOWNLOAD_FAILED.txt"), "w") as f:
        f.write("Model download failed. Please download manually.")
    
    return False

if __name__ == "__main__":
    main()