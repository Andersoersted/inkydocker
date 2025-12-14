"""
Device communication service.
Handles sending images to e-ink display devices.
"""
import os
import httpx
from flask import current_app
from utils.image_processing import prepare_image_for_device, save_temp_image
from utils.database import add_send_log_entry
from models import db


async def send_image_to_device(image_path, device, crop_data=None):
    """
    Send an image to an e-ink display device.

    Args:
        image_path: Path to image file
        device: Device model object
        crop_data: Optional crop information dict

    Returns:
        dict: Status dict with 'status' and 'message'
    """
    try:
        # Parse device resolution
        parts = device.resolution.split("x")
        dev_width = int(parts[0])
        dev_height = int(parts[1])

        # Prepare image for device
        processed_image = prepare_image_for_device(
            image_path,
            dev_width,
            dev_height,
            orientation=device.orientation,
            crop_data=crop_data
        )

        # Save to temporary file
        temp_filename = save_temp_image(processed_image, prefix='device_send')

        # Prepare device address
        addr = device.address
        if not (addr.startswith("http://") or addr.startswith("https://")):
            addr = "http://" + addr

        # Send to device
        url = f"{addr}/display"

        async with httpx.AsyncClient(timeout=120.0) as client:
            with open(temp_filename, 'rb') as f:
                files = {'file': f}
                response = await client.post(url, files=files)

        # Clean up temp file
        try:
            os.remove(temp_filename)
        except Exception as e:
            current_app.logger.warning(f"Error deleting temp file: {e}")

        # Check response
        if response.status_code != 200:
            return {
                "status": "error",
                "message": f"Device returned status {response.status_code}: {response.text}"
            }

        return {"status": "success", "message": "Image sent successfully"}

    except httpx.TimeoutException:
        return {"status": "error", "message": "Request timed out"}
    except httpx.RequestError as e:
        return {"status": "error", "message": f"Network error: {str(e)}"}
    except Exception as e:
        current_app.logger.error(f"Error sending image: {e}")
        return {"status": "error", "message": f"Error: {str(e)}"}


def send_image_to_device_sync(image_path, device, crop_data=None, update_device=True, filename=None):
    """
    Synchronous wrapper for send_image_to_device.
    Updates device last_sent and creates send log entry.

    Args:
        image_path: Path to image file
        device: Device model object
        crop_data: Optional crop information dict
        update_device: Whether to update device.last_sent
        filename: Image filename for logging

    Returns:
        dict: Status dict with 'status' and 'message'
    """
    import asyncio

    # Send image
    result = asyncio.run(send_image_to_device(image_path, device, crop_data))

    # Update device and create log entry if successful
    if result['status'] == 'success' and update_device:
        if filename:
            device.last_sent = filename
            db.session.commit()
            add_send_log_entry(filename)

    return result
