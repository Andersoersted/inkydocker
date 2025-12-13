"""
Image utility functions for InkyDocker.
Handles image file validation and conversion operations.
"""
import os
from PIL import Image

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}


def allowed_file(filename):
    """
    Check if a file has an allowed image extension.

    Args:
        filename: The filename to check

    Returns:
        bool: True if the file extension is allowed, False otherwise
    """
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def convert_to_jpeg(file, base, image_folder):
    """
    Convert an image file to JPEG format.

    Args:
        file: File object to convert
        base: Base filename (without extension)
        image_folder: Directory to save the converted image

    Returns:
        str: New filename if successful, None if conversion failed
    """
    try:
        image = Image.open(file)
        new_filename = f"{base}.jpg"
        filepath = os.path.join(image_folder, new_filename)
        image.convert("RGB").save(filepath, "JPEG")
        return new_filename
    except Exception as e:
        return None
