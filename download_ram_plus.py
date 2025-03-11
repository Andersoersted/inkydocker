#!/usr/bin/env python3
"""
Script to download RAM++ model file directly using requests.
"""
import os
import sys
import requests
import time
from pathlib import Path

def download_file(url, destination):
    """Download a file from a URL to a destination path."""
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    # Get the total file size
    total_size = int(response.headers.get('content-length', 0))
    print(f"File size: {total_size/(1024*1024):.1f} MB")
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    
    # Download the file
    with open(destination, 'wb') as f:
        downloaded = 0
        start_time = time.time()
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                
                # Print progress less frequently to avoid log clipping
                if total_size > 0 and downloaded % (50 * 1024 * 1024) == 0:  # Print every 50MB
                    percent = int(100 * downloaded / total_size)
                    elapsed = time.time() - start_time
                    if elapsed > 0:
                        speed = downloaded / (1024 * 1024 * elapsed)
                        print(f"Progress: {percent}% ({downloaded/(1024*1024):.1f}/{total_size/(1024*1024):.1f} MB) - {speed:.2f} MB/s")
        print()  # New line after progress

def main():
    """Download RAM++ model file directly."""
    print("Setting up RAM++ model...")
    sys.setrecursionlimit(15000)
    
    # Force CPU usage
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["FORCE_CPU"] = "1"
    
    # Set up paths
    cache_dir = "/app/data/ram_models"
    os.makedirs(cache_dir, exist_ok=True)
    
    # Model file to download
    filename = "ram_plus_swin_large_14m.pth"
    destination = os.path.join(cache_dir, filename)
    
    # URLs to try (in order)
    urls = [
        "https://huggingface.co/xinyu1205/recognize-anything-plus-model/resolve/main/ram_plus_swin_large_14m.pth",
        "https://huggingface.co/xinyu1205/recognize-anything-plus-model/blob/main/ram_plus_swin_large_14m.pth",
        "https://huggingface.co/xinyu1205/recognize-anything-plus-model/raw/main/ram_plus_swin_large_14m.pth"
    ]
    
    # Try each URL until one works
    success = False
    for i, url in enumerate(urls):
        try:
            print(f"Downloading RAM++ model (attempt {i+1})...")
            download_file(url, destination)
            success = True
            print(f"RAM++ model download complete")
            break
        except Exception as e:
            print(f"Download attempt {i+1} failed: {str(e)[:100]}...")
    
    # Verify the download
    if success and os.path.exists(destination):
        file_size = os.path.getsize(destination)
        print(f"RAM++ model file size: {file_size / (1024*1024):.1f} MB")
        print("RAM++ model setup successfully")
    else:
        print("WARNING: Failed to download the RAM++ model file")
        # Create a placeholder file to indicate we tried
        with open(os.path.join(cache_dir, "DOWNLOAD_FAILED.txt"), "w") as f:
            f.write("Model download failed. Please download manually.")

if __name__ == "__main__":
    main()