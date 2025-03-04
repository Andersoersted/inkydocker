import os
from PIL import Image

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def convert_to_jpeg(file, base, image_folder):
    try:
        image = Image.open(file)
        new_filename = f"{base}.jpg"
        filepath = os.path.join(image_folder, new_filename)
        image.convert("RGB").save(filepath, "JPEG")
        return new_filename
    except Exception as e:
        return None

def add_send_log_entry(filename):
    # Log the image send by adding an entry to the SendLog table.
    from models import db, SendLog
    entry = SendLog(filename=filename)
    db.session.add(entry)
    db.session.commit()
