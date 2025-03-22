from models import CropInfo, SendLog, db

def load_crop_info_from_db(filename):
    """
    Load crop information from the database with enhanced logging.
    
    Args:
        filename: The filename to load crop info for
        
    Returns:
        dict: Crop information or None if not found
    """
    from flask import current_app
    
    c = CropInfo.query.filter_by(filename=filename).first()
    if not c:
        current_app.logger.debug(f"No crop info found for {filename}")
        return None
    
    result = {
        "x": c.x,
        "y": c.y,
        "width": c.width,
        "height": c.height,
        "resolution": c.resolution,
        "updated_at": c.updated_at if hasattr(c, 'updated_at') else None
    }
    current_app.logger.debug(f"Loaded crop info for {filename}: {result}")
    
    return result

def save_crop_info_to_db(filename, crop_data):
    """
    Save crop information to the database with enhanced logging and timestamp.
    
    Args:
        filename: The filename to save crop info for
        crop_data: Dictionary containing crop information
    """
    from flask import current_app
    from datetime import datetime
    import json
    
    current_app.logger.info(f"Saving new crop info for {filename}: {json.dumps(crop_data)}")
    
    # Check if there's existing crop info to detect changes
    existing = CropInfo.query.filter_by(filename=filename).first()
    is_update = existing is not None
    
    if not existing:
        current_app.logger.info(f"Creating new crop record for {filename}")
        existing = CropInfo(filename=filename)
        db.session.add(existing)
    else:
        # Log the old values for comparison
        old_values = {
            "x": existing.x,
            "y": existing.y,
            "width": existing.width,
            "height": existing.height,
            "resolution": existing.resolution
        }
        current_app.logger.info(f"Updating existing crop record for {filename}")
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
        current_app.logger.info(f"Successfully saved crop info for {filename}")
    except Exception as e:
        db.session.rollback()
        current_app.logger.error(f"Error saving crop info for {filename}: {str(e)}")
        raise

def add_send_log_entry(filename):
    entry = SendLog(filename=filename)
    db.session.add(entry)
    db.session.commit()

def get_last_sent():
    latest = SendLog.query.order_by(SendLog.id.desc()).first()
    return latest.filename if latest else None