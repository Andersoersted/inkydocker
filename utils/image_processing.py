"""
Image processing utilities for InkyDocker.
Handles common image operations: cropping, resizing, rotation, and preparation for e-ink displays.
"""
import os
from PIL import Image
from flask import current_app


def calculate_crop_box(orig_width, orig_height, target_width, target_height, crop_data=None):
    """
    Calculate the crop box for an image.

    Args:
        orig_width: Original image width
        orig_height: Original image height
        target_width: Target device width
        target_height: Target device height
        crop_data: Optional dict with x, y, width, height (normalized 0-1)

    Returns:
        tuple: (left, top, right, bottom) crop box coordinates
    """
    if crop_data and all(k in crop_data for k in ['x', 'y', 'width', 'height']):
        # Use provided crop data (normalized coordinates)
        x = crop_data['x']
        y = crop_data['y']
        w = crop_data['width']
        h = crop_data['height']

        # Convert normalized coordinates to pixel coordinates
        left = int(x * orig_width)
        top = int(y * orig_height)
        right = int((x + w) * orig_width)
        bottom = int((y + h) * orig_height)

        return (left, top, right, bottom)

    # Auto-center crop based on aspect ratios
    orig_ratio = orig_width / orig_height
    target_ratio = target_width / target_height

    if orig_ratio > target_ratio:
        # Image is wider - crop width
        new_width = int(orig_height * target_ratio)
        left = (orig_width - new_width) // 2
        top = 0
        right = left + new_width
        bottom = orig_height
    else:
        # Image is taller - crop height
        new_height = int(orig_width / target_ratio)
        left = 0
        top = (orig_height - new_height) // 2
        right = orig_width
        bottom = top + new_height

    return (left, top, right, bottom)


def prepare_image_for_device(image_path, device_width, device_height, orientation='landscape', crop_data=None):
    """
    Prepare an image for display on an e-ink device.
    Handles cropping, resizing, and rotation.

    Args:
        image_path: Path to the source image
        device_width: Device display width in pixels
        device_height: Device display height in pixels
        orientation: Device orientation ('portrait' or 'landscape')
        crop_data: Optional crop information dict

    Returns:
        PIL.Image: Processed image ready for device
    """
    with Image.open(image_path) as img:
        orig_w, orig_h = img.size

        # Determine if device is in portrait mode
        is_portrait = orientation.lower() == 'portrait'

        # Calculate target dimensions based on orientation
        if is_portrait:
            target_w, target_h = device_height, device_width
        else:
            target_w, target_h = device_width, device_height

        # Calculate crop box
        crop_box = calculate_crop_box(orig_w, orig_h, target_w, target_h, crop_data)

        # Crop image
        cropped = img.crop(crop_box)

        # Resize to exact device dimensions
        resized = cropped.resize((target_w, target_h), Image.Resampling.LANCZOS)

        # Rotate if portrait mode
        if is_portrait:
            resized = resized.rotate(90, expand=True)

        # Convert to RGB if needed (e-ink displays typically use RGB)
        if resized.mode != 'RGB':
            resized = resized.convert('RGB')

        return resized


def save_temp_image(image, prefix='temp'):
    """
    Save a PIL Image to a temporary file.

    Args:
        image: PIL.Image object
        prefix: Filename prefix

    Returns:
        str: Path to saved temporary file
    """
    import tempfile
    import uuid

    temp_id = str(uuid.uuid4())[:8]
    temp_filename = os.path.join(tempfile.gettempdir(), f"{prefix}_{temp_id}.jpg")

    image.save(temp_filename, "JPEG", quality=95)

    return temp_filename


def create_thumbnail(image_path, thumbnail_path, size=(200, 200)):
    """
    Create a thumbnail from an image.

    Args:
        image_path: Source image path
        thumbnail_path: Destination thumbnail path
        size: Thumbnail size tuple (width, height)
    """
    with Image.open(image_path) as img:
        # Handle EXIF orientation
        try:
            exif = img._getexif()
            if exif:
                orientation_tag = 274
                if orientation_tag in exif:
                    orientation = exif[orientation_tag]
                    # Apply orientation correction
                    if orientation == 2:
                        img = img.transpose(Image.FLIP_LEFT_RIGHT)
                    elif orientation == 3:
                        img = img.transpose(Image.ROTATE_180)
                    elif orientation == 4:
                        img = img.transpose(Image.FLIP_TOP_BOTTOM)
                    elif orientation == 5:
                        img = img.transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.ROTATE_90)
                    elif orientation == 6:
                        img = img.transpose(Image.ROTATE_270)
                    elif orientation == 7:
                        img = img.transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.ROTATE_270)
                    elif orientation == 8:
                        img = img.transpose(Image.ROTATE_90)
        except (AttributeError, KeyError, IndexError):
            pass

        # Create thumbnail
        img.thumbnail(size)

        # Convert to RGB if needed
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")

        # Save JPEG thumbnail
        img.save(thumbnail_path, "JPEG", quality=85)

        # Optionally save WebP version for better performance
        webp_path = os.path.splitext(thumbnail_path)[0] + '.webp'
        img.save(webp_path, "WEBP", quality=80)
