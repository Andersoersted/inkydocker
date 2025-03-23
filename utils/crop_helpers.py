from models import CropInfo, SendLog, db

def load_crop_info_from_db(filename, device_address=None):
    """
    Load crop information from the database with enhanced logging.
    
    Args:
        filename: The filename to load crop info for
        device_address: The device address to load crop info for (optional)
        
    Returns:
        dict: Crop information or None if not found
    """
    from flask import current_app
    
    # Query with device_address if provided, otherwise get the first crop info found
    query = CropInfo.query.filter_by(filename=filename)
    if device_address:
        c = query.filter_by(device_address=device_address).first()
        if not c:
            current_app.logger.debug(f"No crop info found for {filename} with device {device_address}")
            # If we don't find a specific device crop, try to fall back to a default if it exists
            c = query.filter_by(device_address='default_device').first()
    else:
        # Get the first crop info for this filename (could be any device)
        c = query.first()
    
    if not c:
        current_app.logger.debug(f"No crop info found for {filename}")
        return None
    
    result = {
        "x": c.x,
        "y": c.y,
        "width": c.width,
        "height": c.height,
        "resolution": c.resolution,
        "device_address": c.device_address,
        "updated_at": c.updated_at if hasattr(c, 'updated_at') else None
    }
    current_app.logger.debug(f"Loaded crop info for {filename} with device {c.device_address}: {result}")
    
    return result

def save_crop_info_to_db(filename, crop_data):
    """
    Save crop information to the database with enhanced logging and timestamp.
    
    Args:
        filename: The filename to save crop info for
        crop_data: Dictionary containing crop information
                   Should include 'device_address' or 'device' key
    """
    from flask import current_app
    from datetime import datetime
    import json
    from models import Device
    
    # Get device address - either directly provided in crop_data or resolve from device ID
    device_address = None
    if "device_address" in crop_data:
        device_address = crop_data.get("device_address")
    elif "device" in crop_data:
        # Try to get device address from the device ID
        device_id = crop_data.get("device")
        device_obj = Device.query.filter_by(address=device_id).first()
        if device_obj:
            device_address = device_obj.address
            current_app.logger.info(f"Resolved device address {device_address} from device {device_id}")
        else:
            current_app.logger.warning(f"Could not find device with address: {device_id}, using as-is")
            device_address = device_id
    
    # If still no device_address, use a default value
    if not device_address:
        device_address = "default_device"
        current_app.logger.warning(f"No device address provided, using default_device")
    
    current_app.logger.info(f"Saving crop info for {filename} with device {device_address}: {json.dumps(crop_data)}")
    
    # Check if there's existing crop info for this device to detect changes
    existing = CropInfo.query.filter_by(filename=filename, device_address=device_address).first()
    is_update = existing is not None
    
    if not existing:
        current_app.logger.info(f"Creating new crop record for {filename} and device {device_address}")
        existing = CropInfo(filename=filename, device_address=device_address)
        db.session.add(existing)
    else:
        # Log the old values for comparison
        old_values = {
            "x": existing.x,
            "y": existing.y,
            "width": existing.width,
            "height": existing.height,
            "resolution": existing.resolution,
            "device_address": existing.device_address
        }
        current_app.logger.info(f"Updating existing crop record for {filename} and device {device_address}")
        current_app.logger.info(f"Old values: {json.dumps(old_values)}")
    
    # Update the values
    existing.x = crop_data.get("x", 0)
    existing.y = crop_data.get("y", 0)
    existing.width = crop_data.get("width", 0)
    existing.height = crop_data.get("height", 0)
    if "resolution" in crop_data:
        existing.resolution = crop_data.get("resolution")
    
    # Add updated_at field if it doesn't exist
    if hasattr(existing, 'updated_at'):
        existing.updated_at = datetime.utcnow()
    
    # Commit the changes
    try:
        db.session.commit()
        current_app.logger.info(f"Successfully saved crop info for {filename} and device {device_address}")
    except Exception as e:
        db.session.rollback()
        current_app.logger.error(f"Error saving crop info for {filename} with device {device_address}: {str(e)}")
        raise

def add_send_log_entry(filename):
    entry = SendLog(filename=filename)
    db.session.add(entry)
    db.session.commit()

def get_last_sent():
    latest = SendLog.query.order_by(SendLog.id.desc()).first()
    return latest.filename if latest else None