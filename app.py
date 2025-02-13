import os
import subprocess
import json
import threading, time, datetime
from flask import Flask, request, redirect, url_for, render_template, send_from_directory, flash, jsonify, send_file
from PIL import Image
import pillow_heif
pillow_heif.register_heif_opener()

app = Flask(__name__)
app.secret_key = "super-secret-key"

# --- Configuration ---
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'heic', 'nef', 'cr2', 'arw', 'dng'}

IMAGE_FOLDER = 'images'
THUMBNAIL_FOLDER = os.path.join(IMAGE_FOLDER, 'thumbnails')
app.config['IMAGE_FOLDER'] = IMAGE_FOLDER

DATA_FOLDER = os.path.join(app.root_path, "data")
if not os.path.exists(DATA_FOLDER):
    os.makedirs(DATA_FOLDER)
SCHEDULE_FILE = os.path.join(DATA_FOLDER, "schedule.json")
SETTINGS_FILE = os.path.join(DATA_FOLDER, "settings.json")
SEND_LOG_FILE = os.path.join(DATA_FOLDER, "send_log.json")

if not os.path.exists(IMAGE_FOLDER):
    os.makedirs(IMAGE_FOLDER)
if not os.path.exists(THUMBNAIL_FOLDER):
    os.makedirs(THUMBNAIL_FOLDER)

# --- Helper Functions ---
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_schedule():
    if os.path.exists(SCHEDULE_FILE):
        with open(SCHEDULE_FILE, "r") as f:
            try:
                return json.load(f)
            except Exception:
                return []
    return []

def save_schedule(schedule):
    with open(SCHEDULE_FILE, "w") as f:
        json.dump(schedule, f)

def load_settings():
    if os.path.exists(SETTINGS_FILE):
        with open(SETTINGS_FILE, "r") as f:
            try:
                return json.load(f)
            except Exception:
                return []
    return []

def save_settings(settings):
    with open(SETTINGS_FILE, "w") as f:
        json.dump(settings, f, indent=4)

def load_send_log():
    if os.path.exists(SEND_LOG_FILE):
        with open(SEND_LOG_FILE, "r") as f:
            try:
                return json.load(f)
            except Exception:
                return {}
    return {}

def save_send_log(log):
    with open(SEND_LOG_FILE, "w") as f:
        json.dump(log, f)

def convert_to_jpeg(file_storage, base_filename):
    try:
        file_storage.seek(0)
        image = Image.open(file_storage)
        if image.mode in ("RGBA", "P"):
            image = image.convert("RGB")
        new_filename = base_filename + ".jpg"
        output_path = os.path.join(app.config['IMAGE_FOLDER'], new_filename)
        image.save(output_path, "JPEG", quality=95)
        return new_filename
    except Exception as e:
        app.logger.error("Error converting image to JPEG: %s", e)
        return None

# --- Thumbnail Route ---
@app.route('/thumbnail/<filename>')
def thumbnail(filename):
    thumb_path = os.path.join(THUMBNAIL_FOLDER, filename)
    image_path = os.path.join(IMAGE_FOLDER, filename)
    if not os.path.exists(image_path):
        return "Not Found", 404
    if not os.path.exists(thumb_path):
        try:
            with Image.open(image_path) as img:
                img.thumbnail((200, 200))
                if img.mode in ("RGBA", "P"):
                    img = img.convert("RGB")
                img.save(thumb_path, "JPEG")
        except Exception as e:
            app.logger.error("Error generating thumbnail for %s: %s", filename, e)
            return "Error generating thumbnail", 500
    return send_from_directory(THUMBNAIL_FOLDER, filename)

# ----------------- Image Manager Routes -----------------
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        files = request.files.getlist('file')
        if not files or all(file.filename == '' for file in files):
            flash('No file selected')
            return redirect(request.url)
        for file in files:
            if file and allowed_file(file.filename):
                from werkzeug.utils import secure_filename
                original_filename = secure_filename(file.filename)
                ext = original_filename.rsplit('.', 1)[1].lower()
                if ext in ['heic', 'nef', 'cr2', 'arw', 'dng']:
                    base = os.path.splitext(original_filename)[0]
                    new_filename = convert_to_jpeg(file, base)
                    if new_filename is None:
                        flash(f"Error converting {original_filename} to JPEG.")
                else:
                    filepath = os.path.join(app.config['IMAGE_FOLDER'], original_filename)
                    file.save(filepath)
        return redirect(url_for('upload_file'))
    
    images = [f for f in os.listdir(app.config['IMAGE_FOLDER'])
              if os.path.isfile(os.path.join(app.config['IMAGE_FOLDER'], f)) and allowed_file(f)]
    devices = load_settings()
    send_log = load_send_log()
    last_sent = send_log.get("last_sent", None)
    return render_template('index.html', images=images, devices=devices, last_sent=last_sent)

@app.route('/images/<filename>')
def uploaded_file(filename):
    filepath = os.path.join(app.config['IMAGE_FOLDER'], filename)
    ext = filename.rsplit('.', 1)[1].lower()
    if ext == "heic":
        try:
            with Image.open(filepath) as img:
                from io import BytesIO
                buf = BytesIO()
                img.save(buf, format="JPEG")
                buf.seek(0)
                return send_file(buf, mimetype='image/jpeg')
        except Exception as e:
            app.logger.error("Error processing HEIC image %s: %s", filename, e)
            return "Error processing image", 500
    else:
        return send_from_directory(app.config['IMAGE_FOLDER'], filename)

@app.route('/send_image/<filename>', methods=['POST'])
def send_image(filename):
    filepath = os.path.join(app.config['IMAGE_FOLDER'], filename)
    if not os.path.exists(filepath):
        return "File not found", 404

    device = request.form.get("device")
    if not device:
        return "No device specified", 400

    command = (
        f'curl "{device}/update_now" '
        f'-X POST '
        f'-F "imageFile=@{filepath}" '
        f'-F "plugin_id=image_upload"'
    )
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        return f"Error sending image: {result.stderr}", 500

    log = load_send_log()
    log["last_sent"] = filename
    if "history" not in log:
        log["history"] = []
    log["history"].append({"filename": filename, "timestamp": datetime.datetime.now().isoformat()})
    save_send_log(log)

    return f"Image sent successfully: {result.stdout}", 200

@app.route('/delete_image/<filename>', methods=['POST'])
def delete_image(filename):
    filepath = os.path.join(app.config['IMAGE_FOLDER'], filename)
    thumb_path = os.path.join(THUMBNAIL_FOLDER, filename)
    if os.path.exists(filepath):
        try:
            os.remove(filepath)
            if os.path.exists(thumb_path):
                os.remove(thumb_path)
            return jsonify({"status": "success", "message": "Image deleted"}), 200
        except Exception as e:
            app.logger.error("Error removing file %s: %s", filepath, e)
            return jsonify({"status": "error", "message": str(e)}), 500
    else:
        return jsonify({"status": "error", "message": "File not found"}), 404

# ----------------- Scheduling Routes -----------------
@app.route('/add_schedule', methods=['POST'])
def add_schedule():
    filename = request.form.get("filename")
    datetime_str = request.form.get("datetime")
    device = request.form.get("device")
    if not (filename and datetime_str and device):
        return jsonify({"status": "error", "message": "Missing parameters"}), 400
    
    schedule = load_schedule()
    event = {
        "id": int(time.time()*1000),
        "filename": filename,
        "datetime": datetime_str,
        "device": device,
        "sent": False
    }
    schedule.append(event)
    save_schedule(schedule)
    return jsonify({"status": "success", "event": event})

@app.route('/schedule')
def schedule_page():
    devices = load_settings()
    images = [f for f in os.listdir(app.config['IMAGE_FOLDER'])
              if os.path.isfile(os.path.join(app.config['IMAGE_FOLDER'], f)) and allowed_file(f)]
    schedule_events = load_schedule()
    return render_template("schedule.html", devices=devices, images=images, schedule_events=schedule_events)

@app.route('/remove_schedule/<int:event_id>', methods=['POST'])
def remove_schedule(event_id):
    schedule = load_schedule()
    new_schedule = [e for e in schedule if e["id"] != event_id]
    save_schedule(new_schedule)
    return jsonify({"status": "success"})

@app.route('/update_schedule', methods=['POST'])
def update_schedule():
    event_id = request.args.get('event_id')
    new_datetime = request.form.get('datetime')
    if not event_id or not new_datetime:
        return jsonify({"status": "error", "message": "Missing parameters"}), 400
    schedule = load_schedule()
    updated = False
    for event in schedule:
        if str(event["id"]) == str(event_id):
            event["datetime"] = new_datetime
            updated = True
            break
    if updated:
        save_schedule(schedule)
        return jsonify({"status": "success"})
    else:
        return jsonify({"status": "error", "message": "Event not found"}), 404

# ----------------- Settings Routes -----------------
@app.route('/settings', methods=['GET', 'POST'])
def settings():
    if request.method == 'POST':
        # Fields in the order: Color, Friendly Name, Orientation, Address, Display Name, Resolution
        color = request.form.get("color")
        friendly_name = request.form.get("friendly_name")
        orientation = request.form.get("orientation")
        address = request.form.get("address")
        display_name = request.form.get("display_name")
        resolution = request.form.get("resolution")
        if color and friendly_name and orientation and address and display_name and resolution:
            settings_data = load_settings()
            settings_data.append({
                "color": color,
                "friendly_name": friendly_name,
                "orientation": orientation,
                "address": address,
                "display_name": display_name,
                "resolution": resolution,
                "online": False
            })
            save_settings(settings_data)
            flash("Device added successfully", "success")
        else:
            flash("Missing fields", "error")
        return redirect(url_for("settings"))
    else:
        devices = load_settings()
        return render_template("settings.html", devices=devices)

@app.route('/delete_device/<int:index>', methods=['POST'])
def delete_device(index):
    settings_data = load_settings()
    if 0 <= index < len(settings_data):
        del settings_data[index]
        save_settings(settings_data)
        flash("Device deleted", "success")
    else:
        flash("Device not found", "error")
    return redirect(url_for("settings"))

@app.route('/edit_device', methods=['POST'])
def edit_device():
    try:
        index = int(request.form.get("device_index"))
        # Fields in the order: Color, Friendly Name, Orientation, Address, Display Name, Resolution
        color = request.form.get("color")
        friendly_name = request.form.get("friendly_name")
        orientation = request.form.get("orientation")
        address = request.form.get("address")
        display_name = request.form.get("display_name")
        resolution = request.form.get("resolution")
        settings_data = load_settings()
        if 0 <= index < len(settings_data):
            settings_data[index]['color'] = color
            settings_data[index]['friendly_name'] = friendly_name
            settings_data[index]['orientation'] = orientation
            settings_data[index]['address'] = address
            settings_data[index]['display_name'] = display_name
            settings_data[index]['resolution'] = resolution
            save_settings(settings_data)
            flash("Device updated successfully", "success")
        else:
            flash("Device index not found", "error")
    except Exception as e:
        flash("Error editing device: " + str(e), "error")
    return redirect(url_for("settings"))

@app.route('/test_device/<int:index>', methods=['GET'])
def test_device(index):
    devices = load_settings()
    if index < 0 or index >= len(devices):
        return jsonify({"status": "error", "message": "Device not found"}), 404
    device = devices[index]
    address = device.get("address")
    command = f'curl -I --max-time 5 -s "{address}"'
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    app.logger.info("Curl output for %s: %s", address, result.stdout)
    # Convert output to uppercase and check for "200 OK"
    if result.returncode == 0 and "200 OK" in result.stdout.upper():
        device["online"] = True
        save_settings(devices)
        return jsonify({"status": "ok"}), 200
    else:
        device["online"] = False
        save_settings(devices)
        return jsonify({"status": "error"}), 500

# New endpoint to test connection by address (using query parameter)
@app.route('/test_connection_address', methods=['GET'])
def test_connection_address():
    address = request.args.get('address')
    if not address:
        return jsonify({"status": "error", "message": "No address provided"}), 400
    command = f'curl -I --max-time 5 -s "{address}"'
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    app.logger.info("Curl output for %s: %s", address, result.stdout)
    if result.returncode == 0 and "200 OK" in result.stdout.upper():
        return jsonify({"status": "ok"}), 200
    else:
        return jsonify({"status": "failed"}), 500

# ----------------- Background Scheduler -----------------
def schedule_checker():
    while True:
        try:
            events = load_schedule()
            now = datetime.datetime.now()
            updated = False
            for event in events:
                if not event.get("sent", False):
                    scheduled_time = datetime.datetime.strptime(event["datetime"], "%Y-%m-%d %H:%M")
                    if now >= scheduled_time:
                        filepath = os.path.join(app.config['IMAGE_FOLDER'], event["filename"])
                        if os.path.exists(filepath):
                            command = (
                                f'curl "{event["device"]}/update_now" '
                                f'-X POST '
                                f'-F "imageFile=@{filepath}" '
                                f'-F "plugin_id=image_upload"'
                            )
                            result = subprocess.run(command, shell=True, capture_output=True, text=True)
                            if result.returncode == 0:
                                event["sent"] = True
                                log = load_send_log()
                                log["last_sent"] = event["filename"]
                                if "history" not in log:
                                    log["history"] = []
                                log["history"].append({"filename": event["filename"], "timestamp": datetime.datetime.now().isoformat()})
                                save_send_log(log)
                                app.logger.info(f"Scheduled event {event['id']} sent.")
                                updated = True
                            else:
                                app.logger.error(f"Error sending scheduled event {event['id']}: {result.stderr}")
                        else:
                            event["sent"] = True
                            updated = True
            if updated:
                save_schedule(events)
        except Exception as e:
            app.logger.error("Error in schedule_checker: %s", e)
        time.sleep(60)

threading.Thread(target=schedule_checker, daemon=True).start()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)