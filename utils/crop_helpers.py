from models import CropInfo, SendLog, db

def load_crop_info_from_db(filename):
    c = CropInfo.query.filter_by(filename=filename).first()
    if not c:
        return None
    return {
        "x": c.x,
        "y": c.y,
        "width": c.width,
        "height": c.height,
        "resolution": c.resolution
    }

def save_crop_info_to_db(filename, crop_data):
    c = CropInfo.query.filter_by(filename=filename).first()
    if not c:
        c = CropInfo(filename=filename)
        db.session.add(c)
    c.x = crop_data.get("x", 0)
    c.y = crop_data.get("y", 0)
    c.width = crop_data.get("width", 0)
    c.height = crop_data.get("height", 0)
    if "resolution" in crop_data:
        c.resolution = crop_data.get("resolution")
    db.session.commit()

def add_send_log_entry(filename):
    entry = SendLog(filename=filename)
    db.session.add(entry)
    db.session.commit()

def get_last_sent():
    latest = SendLog.query.order_by(SendLog.id.desc()).first()
    return latest.filename if latest else None