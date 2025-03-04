# Project Structure

```
assets/
  js/
    main.js
data/
images/
migrations/
  versions/
  __init__.py
  alembic.ini
  env.py
  script.py.mako
routes/
  __init__.py
  additional_routes.py
  ai_tagging_routes.py
  device_info_routes.py
  device_routes.py
  image_routes.py
  schedule_routes.py
  settings_routes.py
static/
  icons/
  send-icon-old.png
  send-icon.png
  settings-wheel.png
  style.css
  trash-icon.png
templates/
  base.html
  index.html
  schedule.html
  settings.html
utils/
  __init__.py
  crop_helpers.py
  image_helpers.py
.DS_Store
.export-ignore
.gitattributes
app.py
config.py
Dockerfile
entrypoint.sh
export.md
exportconfig.json
LICENSE
models.py
package.json
README.md
requirements.txt
scheduler.py
supervisord.conf
tasks.py
webpack.config.js
```


## assets/js/main.js

```js
import { Calendar } from '@fullcalendar/core';
import timeGridPlugin from '@fullcalendar/timegrid';
import '@fullcalendar/core/main.css';
import '@fullcalendar/timegrid/main.css';

document.addEventListener('DOMContentLoaded', function() {
  var calendarEl = document.getElementById('calendar');
  var calendar = new Calendar(calendarEl, {
    plugins: [ timeGridPlugin ],
    initialView: 'timeGridWeek',
    firstDay: 1, 
    nowIndicator: true,
    headerToolbar: {
      left: 'prev,next today',
      center: 'title',
      right: 'timeGridWeek,timeGridDay'
    },
    events: '/schedule/events',
    dateClick: function(info) {
      
      var dtLocal = new Date(info.date);
      var isoStr = dtLocal.toISOString().substring(0,16);
      document.getElementById('eventDate').value = isoStr;
      openEventModal();
    }
  });
  calendar.render();
});
```


## migrations/__init__.py

```py

```


## migrations/alembic.ini

```ini
[alembic]
# Path to migration scripts
script_location = migrations
# Database URL - this must point to the same absolute path as used by your app.
sqlalchemy.url = sqlite:////app/data/mydb.sqlite

[loggers]
keys = root,sqlalchemy,alembic

[handlers]
keys = console

[formatters]
keys = generic

[logger_root]
level = WARN
handlers = console

[logger_sqlalchemy]
level = WARN
handlers =
qualname = sqlalchemy.engine
propagate = 0

[logger_alembic]
level = INFO
handlers =
qualname = alembic
propagate = 0

[handler_console]
class = StreamHandler
args = (sys.stderr,)
level = NOTSET
formatter = generic

[formatter_generic]
format = %(levelname)-5.5s [%(name)s] %(message)s
datefmt = %Y-%m-%d %H:%M:%S
```


## migrations/env.py

```py
from __future__ import with_statement
import os
from logging.config import fileConfig

from sqlalchemy import engine_from_config, pool
from alembic import context

# Get the Alembic config and set up logging
config = context.config
fileConfig(config.config_file_name)

# Determine the project root and ensure the data folder exists.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
data_dir = os.path.join(project_root, 'data')
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

# Ensure an empty database file exists so that autogenerate has something to compare.
db_path = os.path.join(data_dir, 'mydb.sqlite')
if not os.path.exists(db_path):
    open(db_path, 'a').close()

# Import all your models so that they are registered with SQLAlchemy's metadata.
from models import db, Device, ImageDB, CropInfo, SendLog, ScheduleEvent, UserConfig, DeviceMetrics
target_metadata = db.metadata

def run_migrations_offline():
    """Run migrations in 'offline' mode."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True
    )

    with context.begin_transaction():
        context.run_migrations()

def run_migrations_online():
    """Run migrations in 'online' mode."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)

        with context.begin_transaction():
            context.run_migrations()

if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
```


## migrations/script.py.mako

```mako
<% 
import re
import uuid
%>
"""${message}

Revision ID: ${up_revision}
Revises: ${down_revision | n}
Create Date: ${create_date}
"""

# revision identifiers, used by Alembic.
revision = '${up_revision}'
down_revision = ${repr(down_revision)}
branch_labels = None
depends_on = None

def upgrade():
    ${upgrades if upgrades else "pass"}

def downgrade():
    ${downgrades if downgrades else "pass"}
```


## routes/__init__.py

```py

```


## routes/additional_routes.py

```py
from flask import Blueprint, request, jsonify, current_app
from models import Device, db
import subprocess, json

additional_bp = Blueprint('additional', __name__)

@additional_bp.route('/fetch_display_info', methods=['GET'])
def fetch_display_info():
    address = request.args.get('address')
    if not address:
        return jsonify({"status": "error", "message": "No address provided"}), 400
    cmd = f'curl -s "{address}/display_info"'
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        return jsonify({"status": "error", "message": result.stderr}), 500
    try:
        raw_info = json.loads(result.stdout)
        colour_str = raw_info.get("colour", "").capitalize()
        model_str = raw_info.get("model", "")
        resolution_arr = raw_info.get("resolution", [])
        if colour_str:
            display_name = f"{colour_str} Colour - {model_str}"
        else:
            display_name = model_str or "Unknown"
        if len(resolution_arr) == 2:
            resolution_str = f"{resolution_arr[0]}x{resolution_arr[1]}"
        else:
            resolution_str = "N/A"
        return jsonify({
            "status": "ok",
            "info": {
                "display_name": display_name,
                "resolution": resolution_str
            }
        }), 200
    except Exception:
        return jsonify({"status": "error", "message": "Invalid JSON returned"}), 500
```


## routes/ai_tagging_routes.py

```py
from flask import Blueprint, request, jsonify, current_app
import os
from tasks import (
    get_image_embedding,
    generate_tags_and_description,
    reembed_image,
    bulk_tag_images,
    BULK_PROGRESS
)
from models import db, ImageDB
from PIL import Image

ai_bp = Blueprint("ai_tagging", __name__)

@ai_bp.route("/api/ai_tag_image", methods=["POST"])
def ai_tag_image():
    data = request.get_json()
    filename = data.get("filename")
    if not filename:
        return jsonify({"status": "error", "message": "Filename is required"}), 400

    image_folder = current_app.config.get("IMAGE_FOLDER", "./images")
    image_path = os.path.join(image_folder, filename)
    if not os.path.exists(image_path):
        return jsonify({"status": "error", "message": "Image file not found"}), 404

    image_embedding = get_image_embedding(image_path)
    if image_embedding is None:
        return jsonify({"status": "error", "message": "Failed to get embedding"}), 500

    tags, description = generate_tags_and_description(image_embedding)
    # Update the ImageDB record with generated tags
    image_record = ImageDB.query.filter_by(filename=filename).first()
    if image_record:
        image_record.tags = ", ".join(tags)
        db.session.commit()
    else:
        image_record = ImageDB(filename=filename, tags=", ".join(tags))
        db.session.add(image_record)
        db.session.commit()

    return jsonify({
        "status": "success",
        "filename": filename,
        "tags": tags
    }), 200

@ai_bp.route("/api/search_images", methods=["GET"])
def search_images():
    q = request.args.get("q", "").strip()
    if not q:
        return jsonify({"status": "error", "message": "Missing query parameter"}), 400
    images = ImageDB.query.filter(ImageDB.tags.ilike(f"%{q}%")).all()
    results = {
        "ids": [img.filename for img in images],
        "tags": [img.tags for img in images]
    }
    return jsonify({"status": "success", "results": results}), 200

@ai_bp.route("/api/get_image_metadata", methods=["GET"])
def get_image_metadata():
    filename = request.args.get("filename", "").strip()
    if not filename:
        return jsonify({"status": "error", "message": "Missing filename"}), 400

    image_folder = current_app.config.get("IMAGE_FOLDER", "./images")
    image_path = os.path.join(image_folder, filename)
    resolution_str = "N/A"
    filesize_str = "N/A"
    if os.path.exists(image_path):
        try:
            size_bytes = os.path.getsize(image_path)
            filesize_mb = size_bytes / (1024 * 1024)
            filesize_str = f"{filesize_mb:.2f} MB"
            with Image.open(image_path) as im:
                w, h = im.size
                resolution_str = f"{w}x{h}"
        except Exception as ex:
            current_app.logger.warning(f"Could not read file info for {filename}: {ex}")

    image_record = ImageDB.query.filter_by(filename=filename).first()
    if image_record:
        tags = [t.strip() for t in image_record.tags.split(",")] if image_record.tags else []
        favorite = image_record.favorite
    else:
        tags = []
        favorite = False

    return jsonify({
        "status": "success",
        "tags": tags,
        "favorite": favorite,
        "resolution": resolution_str,
        "filesize": filesize_str
    }), 200

@ai_bp.route("/api/update_image_metadata", methods=["POST"])
def update_image_metadata():
    data = request.get_json() or {}
    filename = data.get("filename", "").strip()
    new_tags = data.get("tags", [])
    if isinstance(new_tags, list):
        tags_str = ", ".join(new_tags)
    else:
        tags_str = new_tags
    favorite = data.get("favorite", None)
    if not filename:
        return jsonify({"status": "error", "message": "Missing filename"}), 400

    image_record = ImageDB.query.filter_by(filename=filename).first()
    if not image_record:
        return jsonify({"status": "error", "message": "Image not found"}), 404

    image_record.tags = tags_str
    if favorite is not None:
        image_record.favorite = bool(favorite)
    db.session.commit()
    return jsonify({"status": "success"}), 200

@ai_bp.route("/api/reembed_image", methods=["GET"])
def reembed_image_endpoint():
    filename = request.args.get("filename", "").strip()
    if not filename:
        return jsonify({"status": "error", "message": "Missing filename"}), 400
    result = reembed_image(filename)
    return jsonify(result)

@ai_bp.route("/api/reembed_all_images", methods=["GET"])
def reembed_all_images_endpoint():
    task_id = bulk_tag_images.delay()
    if not task_id:
        return jsonify({"status": "error", "message": "No images found"}), 404
    return jsonify({"status": "success", "message": f"Reembedding images in background. Task ID: {task_id}"}), 200
```


## routes/device_info_routes.py

```py
# routes/device_info_routes.py

from flask import Blueprint, request, jsonify, Response, stream_with_context
import httpx
import json
import time
import threading
from queue import Queue, Empty
from datetime import datetime
from models import db, Device

device_info_bp = Blueprint('device_info', __name__)

@device_info_bp.route('/device_info', methods=['GET'])
def get_device_info():
    address = request.args.get('address')
    if not address:
        return jsonify({"status": "error", "message": "No address provided"}), 400

    # Ensure the address has a scheme
    if not (address.startswith("http://") or address.startswith("https://")):
        address = "http://" + address

    try:
        # Use httpx with a 10-second timeout and a curl-like User-Agent
        # Reduce timeout to 5 seconds to fail faster
        response = httpx.get(f"{address}/display_info", timeout=5.0, headers={'User-Agent': 'curl/7.68.0'})
        response.raise_for_status()
        raw_info = response.json()
    except httpx.TimeoutException:
        # Handle timeout specifically to provide a clearer error message
        return jsonify({"status": "error", "message": "Connection timed out. Device may be offline."}), 500
    except httpx.HTTPStatusError as e:
        # Handle HTTP errors specifically
        return jsonify({"status": "error", "message": f"HTTP error: {e.response.status_code}"}), 500
    except Exception as e:
        # Handle other exceptions
        return jsonify({"status": "error", "message": f"Error fetching display info: {str(e)}"}), 500

    try:
        # Build display name as "model colour color"
        colour = raw_info.get("colour", "").strip()
        model = raw_info.get("model", "").strip()
        if model and colour:
            display_name = f"{model} {colour} color"
        elif model:
            display_name = model
        else:
            display_name = "Unknown"
        # Format resolution as "widthxheight"
        resolution_arr = raw_info.get("resolution", [])
        if isinstance(resolution_arr, list) and len(resolution_arr) == 2:
            resolution = f"{resolution_arr[0]}x{resolution_arr[1]}"
        else:
            resolution = "N/A"
    except Exception:
        return jsonify({"status": "error", "message": "Invalid JSON returned"}), 500

    return jsonify({
        "status": "ok",
        "info": {
            "display_name": display_name,
            "resolution": resolution
        }
    }), 200

# Global dictionary for active device stream clients (if needed in future)
active_device_streams = {}

@device_info_bp.route('/device/<int:device_index>/stream', methods=['GET'])
def device_stream(device_index):
    """
    Updated streaming endpoint that connects to the device's live stream,
    updates the device's online status, and pushes status updates into a
    thread-safe queue which is then served via Server-Sent Events.
    """
    devices = Device.query.order_by(Device.id).all()
    if not (0 <= device_index < len(devices)):
        return jsonify({"status": "error", "message": "Device not found"}), 404
    device = devices[device_index]
    address = device.address
    if not (address.startswith("http://") or address.startswith("https://")):
        address = "http://" + address

    # Create a thread-safe queue to pass status updates immediately
    status_queue = Queue()

    def stream_reader():
        try:
            # First, check if the device is reachable via its display_info endpoint
            # Use a try-except block to handle connection errors more gracefully
            try:
                info_response = httpx.get(f"{address}/display_info", timeout=5.0)
                if info_response.status_code != 200:
                    status_queue.put({"status": "error", "message": f"Device not responding: {info_response.status_code}"})
                    return
            except Exception as e:
                status_queue.put({"status": "error", "message": f"Cannot connect to device: {str(e)}"})
                return

            # Now, connect to the live stream (no timeout so it remains open)
            client = httpx.Client(timeout=None)
            response = client.get(f"{address}/stream", stream=True)
            if response.status_code != 200:
                status_queue.put({"status": "error", "message": f"Error connecting to stream: {response.status_code}"})
                return

            # Mark the device as online and update the database
            device.online = True
            db.session.commit()

            # Continuously read and process incoming lines from the stream
            for line in response.iter_lines():
                if line:
                    if isinstance(line, bytes):
                        line = line.decode("utf-8")
                    if line.startswith("data:"):
                        data_str = line[len("data:"):].strip()
                        try:
                            data = json.loads(data_str)
                        except json.JSONDecodeError:
                            continue
                        # Only update online status, ignore metrics
                        device.online = True
                        db.session.commit()
                        # Immediately push the status update to the queue for SSE output
                        status_queue.put({"status": "online", "device": device.friendly_name})
        except Exception as e:
            device.online = False
            db.session.commit()
            status_queue.put({"status": "error", "message": f"Stream reader error: {str(e)}"})

    # Start the stream_reader thread as a daemon
    threading.Thread(target=stream_reader, daemon=True).start()

    def generate():
        # Continuously yield each status update received from the queue as an SSE event.
        while True:
            try:
                data = status_queue.get(timeout=10)
                yield f"data: {json.dumps(data)}\n\n"
            except Empty:
                # If no new status update arrives within 10 seconds, send a heartbeat event.
                yield "data: {}\n\n"

    return Response(stream_with_context(generate()), mimetype='text/event-stream')

@device_info_bp.route('/devices/status', methods=['GET'])
def devices_status():
    """Get online status for all devices"""
    # Don't check device status here - just return the current status from the database
    # This prevents duplicate status checks that could flood the logs
    devices = Device.query.all()
    data = []
    for idx, device in enumerate(devices):
        data.append({
            "index": idx,
            "online": device.online
        })
    return jsonify({"status": "success", "devices": data})
```


## routes/device_routes.py

```py
from flask import Blueprint, request, jsonify, current_app, send_file
from models import db, Device
import httpx
import subprocess, os, datetime, json

device_bp = Blueprint('device', __name__)

@device_bp.route('/device/<int:index>/set_orientation', methods=['POST'])
def set_device_orientation(index):
    orientation = request.form.get('orientation')
    if not orientation:
        return jsonify({"status": "error", "message": "No orientation provided"}), 400
    all_devices = Device.query.order_by(Device.id).all()
    if index < 0 or index >= len(all_devices):
        return jsonify({"status": "error", "message": "Device not found"}), 404
    device_address = all_devices[index].address
    if not (device_address.startswith("http://") or device_address.startswith("https://")):
        device_address = "http://" + device_address
    try:
        response = httpx.post(f"{device_address}/set_orientation", data={"orientation": orientation}, timeout=5.0)
        response.raise_for_status()
        return jsonify({"status": "ok"}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@device_bp.route('/device/<int:index>/display_info', methods=['GET'])
def get_device_info(index):
    all_devices = Device.query.order_by(Device.id).all()
    if index < 0 or index >= len(all_devices):
        return jsonify({"status": "error", "message": "Device not found"}), 404
    device_address = all_devices[index].address
    if not (device_address.startswith("http://") or device_address.startswith("https://")):
        device_address = "http://" + device_address
    try:
        # Use a curl-like User-Agent to mimic curl behavior
        response = httpx.get(f"{device_address}/display_info", timeout=5.0, headers={"User-Agent": "curl/7.68.0"})
        response.raise_for_status()
        raw = response.json()
        return jsonify({"status": "ok", "info": raw})
    except Exception as e:
        return jsonify({"status": "error", "message": f"Error fetching display info: {str(e)}"}), 500

@device_bp.route('/device/<int:index>/fetch_metrics', methods=['GET'])
def fetch_metrics(index):
    """
    Fetch the first SSE metric line from the device's /stream endpoint using httpx streaming.
    """
    all_devices = Device.query.order_by(Device.id).all()
    if index < 0 or index >= len(all_devices):
        return jsonify({"status": "error", "message": "Device not found"}), 404
    device = all_devices[index]
    address = device.address
    if not (address.startswith("http://") or address.startswith("https://")):
        address = "http://" + address
    try:
        with httpx.stream("GET", f"{address}/stream", timeout=10.0, headers={"User-Agent": "curl/7.68.0"}) as response:
            for line in response.iter_lines():
                if line:
                    # httpx.iter_lines() returns bytes if no decoding is set
                    if isinstance(line, bytes):
                        line = line.decode("utf-8")
                    if line.startswith("data:"):
                        data_str = line[len("data:"):].strip()
                        data_json = json.loads(data_str)
                        cpu_usage = str(data_json.get("cpu", "N/A"))
                        mem_usage = str(data_json.get("memory", "N/A"))
                        disk_usage = str(data_json.get("disk", "N/A"))
                        device.cpu_usage = cpu_usage
                        device.mem_usage = mem_usage
                        device.disk_usage = disk_usage
                        device.online = True
                        db.session.commit()
                        return jsonify({
                            "status": "ok",
                            "cpu": cpu_usage + "%",
                            "mem": mem_usage + "%",
                            "disk": disk_usage + "%",
                            "online": device.online
                        })
            # If no valid line is found:
            device.online = False
            db.session.commit()
            return jsonify({"status": "error", "message": "No metrics data received"}), 500
    except Exception as e:
        device.online = False
        db.session.commit()
        return jsonify({"status": "error", "message": str(e)}), 500

@device_bp.route('/device/<int:index>/system_update', methods=['POST'])
def system_update(index):
    all_devices = Device.query.order_by(Device.id).all()
    if index < 0 or index >= len(all_devices):
        return jsonify({"status": "error", "message": "Device not found"}), 404
    device_address = all_devices[index].address
    if not (device_address.startswith("http://") or device_address.startswith("https://")):
        device_address = "http://" + device_address
    try:
        response = httpx.post(f"{device_address}/system_update", timeout=10.0)
        response.raise_for_status()
        return jsonify({"status": "ok"}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@device_bp.route('/device/<int:index>/backup', methods=['GET'])
def create_disk_backup(index):
    from flask import send_file
    all_devices = Device.query.order_by(Device.id).all()
    if index < 0 or index >= len(all_devices):
        return jsonify({"status": "error", "message": "Device not found"}), 404
    device_address = all_devices[index].address
    if not (device_address.startswith("http://") or device_address.startswith("https://")):
        device_address = "http://" + device_address
    data_folder = current_app.config['DATA_FOLDER']
    backup_dir = os.path.join(data_folder, "display_backup")
    if not os.path.exists(backup_dir):
        os.makedirs(backup_dir)
    backup_filename = f"backup_{index}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.img.gz"
    backup_path = os.path.join(backup_dir, backup_filename)
    try:
        response = httpx.post(f"{device_address}/backup", timeout=30.0)
        response.raise_for_status()
        with open(backup_path, "wb") as f:
            f.write(response.content)
        if os.path.exists(backup_path):
            return send_file(backup_path, mimetype='application/gzip',
                             as_attachment=True, download_name=backup_filename)
        else:
            return jsonify({"status": "error", "message": "Backup file not created"}), 500
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@device_bp.route('/device/<int:index>/update', methods=['POST'])
def update_application(index):
    all_devices = Device.query.order_by(Device.id).all()
    if index < 0 or index >= len(all_devices):
        return jsonify({"status": "error", "message": "Device not found"}), 404
    device_address = all_devices[index].address
    if not (device_address.startswith("http://") or device_address.startswith("https://")):
        device_address = "http://" + device_address
    try:
        response = httpx.post(f"{device_address}/update", timeout=10.0)
        response.raise_for_status()
        return jsonify({"status": "ok"}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@device_bp.route('/device/<int:index>/stream')
def metrics_stream(index):
    all_devices = Device.query.order_by(Device.id).all()
    if index < 0 or index >= len(all_devices):
        return "Device not found", 404
    device_address = all_devices[index].address
    def generate():
        command = f'curl -N -s "{device_address}/stream"'
        process = subprocess.Popen(command, shell=True,
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, text=True)
        try:
            while True:
                line = process.stdout.readline()
                if not line:
                    break
                yield line
        except Exception:
            process.kill()
        finally:
            process.kill()
    return current_app.response_class(generate(), mimetype='text/event-stream')

@device_bp.route('/test_device/<int:index>', methods=['GET'])
def test_device(index):
    all_devices = Device.query.order_by(Device.id).all()
    if index < 0 or index >= len(all_devices):
        return jsonify({"status": "error", "message": "Device not found"}), 404
    device = all_devices[index]
    address = device.address
    cmd = f'curl -I --max-time 5 -s "{address}"'
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    current_app.logger.info("Curl output for %s: %s", address, result.stdout)
    if result.returncode == 0 and "200 OK" in result.stdout.upper():
        device.online = True
        db.session.commit()
        return jsonify({"status": "ok"}), 200
    else:
        device.online = False
        db.session.commit()
        return jsonify({"status": "error"}), 500

@device_bp.route('/test_connection_address', methods=['GET'])
def test_connection_address():
    address = request.args.get('address')
    if not address:
        return jsonify({"status": "error", "message": "No address provided"}), 400
    cmd = f'curl -I --max-time 5 -s "{address}"'
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    current_app.logger.info("Curl output for %s: %s", address, result.stdout)
    if result.returncode == 0 and "200 OK" in result.stdout.upper():
        return jsonify({"status": "ok"}), 200
    else:
        return jsonify({"status": "failed"}), 500
```


## routes/image_routes.py

```py
from flask import Blueprint, request, redirect, url_for, render_template, flash, send_from_directory, send_file, jsonify, current_app
from models import db, ImageDB, CropInfo, SendLog, Device
import os
from PIL import Image
import subprocess
from utils.image_helpers import allowed_file, convert_to_jpeg
from utils.crop_helpers import load_crop_info_from_db, save_crop_info_to_db, add_send_log_entry, get_last_sent

image_bp = Blueprint('image', __name__)

@image_bp.route('/thumbnail/<filename>')
def thumbnail(filename):
    image_folder = current_app.config['IMAGE_FOLDER']
    thumbnail_folder = current_app.config['THUMBNAIL_FOLDER']
    thumb_path = os.path.join(thumbnail_folder, filename)
    image_path = os.path.join(image_folder, filename)
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
            current_app.logger.error("Error generating thumbnail for %s: %s", filename, e)
            return "Error generating thumbnail", 500
    return send_from_directory(thumbnail_folder, filename)

@image_bp.route('/', methods=['GET', 'POST'])
def upload_file():
    image_folder = current_app.config['IMAGE_FOLDER']
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
                    new_filename = convert_to_jpeg(file, base, image_folder)
                    if new_filename is None:
                        flash(f"Error converting {original_filename} to JPEG.")
                    else:
                        if not ImageDB.query.filter_by(filename=new_filename).first():
                            db.session.add(ImageDB(filename=new_filename))
                            db.session.commit()
                            # Trigger automatic image tagging
                            from tasks import process_image_tagging
                            process_image_tagging.delay(new_filename)
                else:
                    filepath = os.path.join(image_folder, original_filename)
                    file.save(filepath)
                    if not ImageDB.query.filter_by(filename=original_filename).first():
                        db.session.add(ImageDB(filename=original_filename))
                        db.session.commit()
                        # Trigger automatic image tagging
                        from tasks import process_image_tagging
                        process_image_tagging.delay(original_filename)
        return redirect(url_for('image.upload_file'))
    
    images_db = ImageDB.query.all()
    images = [img.filename for img in images_db]
    devices_db = Device.query.all()
    devices = []
    for d in devices_db:
        devices.append({
            "color": d.color,
            "friendly_name": d.friendly_name,
            "orientation": d.orientation,
            "address": d.address,
            "display_name": d.display_name,
            "resolution": d.resolution,
            "online": d.online,
            "last_sent": d.last_sent
        })
    last_sent = get_last_sent()
    return render_template('index.html', images=images, devices=devices, last_sent=last_sent)

@image_bp.route('/images/<filename>')
def uploaded_file(filename):
    image_folder = current_app.config['IMAGE_FOLDER']
    filepath = os.path.join(image_folder, filename)
    if not os.path.exists(filepath):
        return "File not found", 404
    if request.args.get("size") == "info":
        try:
            with Image.open(filepath) as img:
                max_width = 300
                w, h = img.size
                if w > max_width:
                    ratio = max_width / float(w)
                    new_size = (max_width, int(h * ratio))
                    img = img.resize(new_size, Image.LANCZOS)
                from io import BytesIO
                buf = BytesIO()
                img.save(buf, format="JPEG")
                buf.seek(0)
                return send_file(buf, mimetype='image/jpeg')
        except Exception as e:
            current_app.logger.error("Error processing image %s for info: %s", filename, e)
            return "Error processing image", 500
    ext = filename.rsplit('.', 1)[1].lower()
    if ext == "heic":
        try:
            from io import BytesIO
            with Image.open(filepath) as img:
                buf = BytesIO()
                img.save(buf, format="JPEG")
                buf.seek(0)
                return send_file(buf, mimetype='image/jpeg')
        except Exception as e:
            current_app.logger.error("Error processing HEIC image %s: %s", filename, e)
            return "Error processing image", 500
    else:
        return send_from_directory(image_folder, filename)

@image_bp.route('/save_crop_info/<filename>', methods=['POST'])
def save_crop_info_endpoint(filename):
    crop_data = request.get_json()
    if not crop_data:
        return jsonify({"status": "error", "message": "No crop data provided"}), 400
    
    # Validate crop data
    required_fields = ["x", "y", "width", "height"]
    for field in required_fields:
        if field not in crop_data or not isinstance(crop_data[field], (int, float)) or crop_data[field] < 0:
            return jsonify({"status": "error", "message": f"Invalid crop data: {field} is missing or invalid"}), 400
    
    # Get the selected device resolution if provided
    if "device" in crop_data:
        device_addr = crop_data.get("device")
        device_obj = Device.query.filter_by(address=device_addr).first()
        if device_obj and device_obj.resolution:
            crop_data["resolution"] = device_obj.resolution
            current_app.logger.info(f"Saving crop with resolution: {device_obj.resolution}")
        else:
            current_app.logger.warning(f"Device not found or missing resolution: {device_addr}")
    else:
        current_app.logger.warning("No device provided for crop data")
    
    save_crop_info_to_db(filename, crop_data)
    return jsonify({"status": "success"}), 200

@image_bp.route('/send_image/<filename>', methods=['POST'])
def send_image(filename):
    image_folder = current_app.config['IMAGE_FOLDER']
    data_folder = current_app.config['DATA_FOLDER']
    filepath = os.path.join(image_folder, filename)
    if not os.path.exists(filepath):
        return "File not found", 404
    device_addr = request.form.get("device")
    if not device_addr:
        return "No device specified", 400

    from models import Device
    device_obj = Device.query.filter_by(address=device_addr).first()
    if not device_obj:
        return "Device not found in DB", 500
    dev_width = None
    dev_height = None
    if device_obj.resolution:
        parts = device_obj.resolution.split("x")
        if len(parts) == 2:
            try:
                dev_width = int(parts[0])
                dev_height = int(parts[1])
            except:
                pass
    if not (dev_width and dev_height):
        return "Target resolution not found", 500

    try:
        with Image.open(filepath) as orig_img:
            orig_w, orig_h = orig_img.size
            
            # Check if device is in portrait orientation
            is_portrait = device_obj.orientation.lower() == 'portrait'
            
            # If portrait, swap width and height for target ratio calculation
            if is_portrait:
                target_ratio = dev_height / dev_width
            else:
                target_ratio = dev_width / dev_height
                
            # Log the original image dimensions and target ratio
            current_app.logger.info(f"Original image dimensions: {orig_w}x{orig_h}, target ratio: {target_ratio}")
            
            # Step 1: Apply crop if available
            cdata = load_crop_info_from_db(filename)
            if cdata and all(key in cdata for key in ["x", "y", "width", "height"]):
                x = cdata.get("x", 0)
                y = cdata.get("y", 0)
                w = cdata.get("width", orig_w)
                h = cdata.get("height", orig_h)
                
                # Validate crop coordinates
                if x < 0 or y < 0 or w <= 0 or h <= 0 or x + w > orig_w or y + h > orig_h:
                    current_app.logger.warning(f"Invalid crop coordinates: ({x}, {y}, {w}, {h}) for image {orig_w}x{orig_h}")
                    # Fall back to auto-centered crop
                    cdata = None
                else:
                    # If we have stored resolution and it matches the current device,
                    # use the stored crop data directly
                    stored_resolution = cdata.get("resolution")
                    current_app.logger.info(f"Stored resolution: {stored_resolution}, device resolution: {device_obj.resolution}")
                    
                    if stored_resolution and stored_resolution == device_obj.resolution:
                        current_app.logger.info(f"Using stored crop data: ({x}, {y}, {w}, {h})")
                        cropped = orig_img.crop((x, y, x+w, y+h))
                    else:
                        # If resolutions don't match, we need to recalculate the crop
                        # to maintain the correct aspect ratio
                        current_app.logger.info(f"Recalculating crop to match target ratio")
                        cropped = orig_img.crop((x, y, x+w, y+h))
                        crop_w, crop_h = cropped.size
                        crop_ratio = crop_w / crop_h
                        
                        # Adjust the crop to match the target ratio
                        if crop_ratio > target_ratio:
                            new_width = int(crop_h * target_ratio)
                            left = (crop_w - new_width) // 2
                            crop_box = (left, 0, left + new_width, crop_h)
                        else:
                            new_height = int(crop_w / target_ratio)
                            top = (crop_h - new_height) // 2
                            crop_box = (0, top, crop_w, top + new_height)
                        cropped = cropped.crop(crop_box)
            
            # If no valid crop data, create an auto-centered crop with the correct aspect ratio
            if not cdata or "x" not in cdata:
                current_app.logger.info(f"No crop data found, using auto-centered crop")
                orig_ratio = orig_w / orig_h
                
                if orig_ratio > target_ratio:
                    # Image is wider than target ratio, use full height
                    new_width = int(orig_h * target_ratio)
                    left = (orig_w - new_width) // 2
                    crop_box = (left, 0, left + new_width, orig_h)
                else:
                    # Image is taller than target ratio, use full width
                    new_height = int(orig_w / target_ratio)
                    top = (orig_h - new_height) // 2
                    crop_box = (0, top, orig_w, top + new_height)
                
                current_app.logger.info(f"Auto crop box: {crop_box}")
                cropped = orig_img.crop(crop_box)

            # Step 2: Resize the cropped image to match the target resolution
            current_app.logger.info(f"Cropped image size: {cropped.size}")
            
            # If portrait, rotate the image 90 degrees clockwise and swap dimensions
            if is_portrait:
                current_app.logger.info("Rotating image for portrait orientation")
                cropped = cropped.rotate(-90, expand=True)  # -90 for clockwise rotation
                current_app.logger.info(f"After rotation size: {cropped.size}")
                final_img = cropped.resize((dev_height, dev_width), Image.LANCZOS)  # Note swapped dimensions
            else:
                final_img = cropped.resize((dev_width, dev_height), Image.LANCZOS)
            
            current_app.logger.info(f"Final image size: {final_img.size}")
            
            # Save the processed image as a temporary file
            temp_dir = os.path.join(data_folder, "temp")
            if not os.path.exists(temp_dir):
                os.makedirs(temp_dir)
            temp_filename = os.path.join(temp_dir, f"temp_{filename}")
            final_img.save(temp_filename, format="JPEG", quality=95)
            current_app.logger.info(f"Saved temporary file: {temp_filename}")

        cmd = f'curl "{device_addr}/send_image" -X POST -F "file=@{temp_filename}"'
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        os.remove(temp_filename)
        if result.returncode != 0:
            return f"Error sending image: {result.stderr}", 500

        device_obj.last_sent = filename
        db.session.commit()
        add_send_log_entry(filename)
        return f"Image sent successfully: {result.stdout}", 200
    except Exception as e:
        current_app.logger.error("Error resizing/cropping image: %s", e)
        return f"Error processing image: {e}", 500

@image_bp.route('/delete_image/<filename>', methods=['POST'])
def delete_image_endpoint(filename):
    image_folder = current_app.config['IMAGE_FOLDER']
    thumbnail_folder = current_app.config['THUMBNAIL_FOLDER']
    filepath = os.path.join(image_folder, filename)
    thumb_path = os.path.join(thumbnail_folder, filename)
    if os.path.exists(filepath):
        try:
            os.remove(filepath)
            if os.path.exists(thumb_path):
                os.remove(thumb_path)
            img_obj = ImageDB.query.filter_by(filename=filename).first()
            if img_obj:
                db.session.delete(img_obj)
                db.session.commit()
            return jsonify({"status": "success", "message": "Image deleted"}), 200
        except Exception as e:
            current_app.logger.error("Error removing file %s: %s", filepath, e)
            return jsonify({"status": "error", "message": str(e)}), 500
    else:
        return jsonify({"status": "error", "message": "File not found"}), 404
```


## routes/schedule_routes.py

```py
from flask import Blueprint, request, jsonify, render_template
from models import db, ScheduleEvent, Device, ImageDB
import datetime
from tasks import send_scheduled_image
# Import the scheduler from the dedicated scheduler module instead of tasks
# This ensures we're using the scheduler that runs in a dedicated process
from scheduler import scheduler

schedule_bp = Blueprint('schedule', __name__)

@schedule_bp.route('/schedule')
def schedule_page():
    devs = Device.query.all()
    devices = []
    for d in devs:
        devices.append({
            "color": d.color,
            "friendly_name": d.friendly_name,
            "orientation": d.orientation,
            "address": d.address,
            "display_name": d.display_name,
            "resolution": d.resolution,
            "online": d.online,
            "last_sent": d.last_sent
        })
    
    # Get all images and their tags
    imgs_db = ImageDB.query.all()
    images = [i.filename for i in imgs_db]
    
    # Create a dictionary of image tags for the template
    image_tags = {}
    for img in imgs_db:
        if img.tags:
            image_tags[img.filename] = img.tags
    
    return render_template("schedule.html", devices=devices, images=images, image_tags=image_tags)

@schedule_bp.route('/schedule/events')
def get_events():
    # Ensure we're getting the latest data from the database
    db.session.expire_all()
    events = ScheduleEvent.query.all()
    event_list = []
    
    # FIXED: Use timezone-aware datetime objects to avoid "can't compare offset-naive and offset-aware datetimes" error
    # Create now and horizon as UTC timezone-aware datetimes
    now = datetime.datetime.now(datetime.timezone.utc)
    horizon = now + datetime.timedelta(days=90)
    print(f"Loading {len(events)} events from database")
    
    # Create a device lookup dictionary for colors and friendly names
    device_lookup = {}
    devices = Device.query.all()
    for device in devices:
        device_lookup[device.address] = {
            "color": device.color,
            "friendly_name": device.friendly_name
        }
    
    # Create an image lookup for thumbnails
    image_lookup = {}
    images = ImageDB.query.all()
    for img in images:
        image_lookup[img.filename] = {
            "tags": img.tags or ""
        }
    
    for ev in events:
        # Get device color and friendly name
        device_info = device_lookup.get(ev.device, {"color": "#cccccc", "friendly_name": ev.device})
        device_color = device_info["color"]
        device_name = device_info["friendly_name"]
        
        if ev.recurrence.lower() == "none":
            event_list.append({
                "id": ev.id,
                "title": f"{ev.filename}",
                "start": ev.datetime_str,
                "device": ev.device,
                "deviceName": device_name,
                "filename": ev.filename,
                "recurrence": ev.recurrence,
                "series": False,
                "backgroundColor": device_color,
                "borderColor": device_color,
                "textColor": "#ffffff",
                "extendedProps": {
                    "thumbnail": f"/thumbnail/{ev.filename}"
                }
            })
        else:
            # Generate recurring occurrences from the next occurrence up to the horizon
            try:
                # Parse the datetime string and ensure it's timezone-aware
                start_dt = datetime.datetime.fromisoformat(ev.datetime_str)
                # FIXED: Ensure start_dt is timezone-aware for proper comparison with now
                if start_dt.tzinfo is None:
                    # If the datetime is naive, assume it's in UTC
                    start_dt = start_dt.replace(tzinfo=datetime.timezone.utc)
            except Exception as e:
                print(f"Error parsing datetime for event {ev.id}: {str(e)}")
                continue
                
            # Advance to the first occurrence that is >= now
            rec = ev.recurrence.lower()
            occurrence = start_dt
            # FIXED: Now both occurrence and now are timezone-aware, so comparison works correctly
            while occurrence < now:
                if rec == "daily":
                    occurrence += datetime.timedelta(days=1)
                elif rec == "weekly":
                    occurrence += datetime.timedelta(weeks=1)
                elif rec == "monthly":
                    occurrence += datetime.timedelta(days=30)
                else:
                    break
            # Generate occurrences until the horizon
            while occurrence <= horizon:
                event_list.append({
                    "id": ev.id,  # same series id
                    "title": f"{ev.filename} (Recurring)",  # Mark as recurring in title
                    "start": occurrence.isoformat(),  # FIXED: Use standard ISO format without sep parameter
                    "device": ev.device,
                    "deviceName": device_name,
                    "filename": ev.filename,
                    "recurrence": ev.recurrence,
                    "series": True,
                    "backgroundColor": device_color,
                    "borderColor": device_color,
                    "textColor": "#ffffff",
                    "classNames": ["recurring-event"],  # Add a class for styling
                    "extendedProps": {
                        "thumbnail": f"/thumbnail/{ev.filename}",
                        "isRecurring": True  # Flag for frontend to identify recurring events
                    }
                })
                if rec == "daily":
                    occurrence += datetime.timedelta(days=1)
                elif rec == "weekly":
                    occurrence += datetime.timedelta(weeks=1)
                elif rec == "monthly":
                    occurrence += datetime.timedelta(days=30)
                else:
                    break
    return jsonify(event_list)

@schedule_bp.route('/schedule/add', methods=['POST'])
def add_event():
    data = request.get_json()
    datetime_str = data.get("datetime")
    device = data.get("device")
    filename = data.get("filename")
    recurrence = data.get("recurrence", "none")
    timezone_offset = data.get("timezone_offset", 0)  # Minutes offset from UTC
    
    print(f"Adding event with datetime: {datetime_str}, timezone offset: {timezone_offset}")
    if not (datetime_str and device and filename):
        return jsonify({"status": "error", "message": "Missing parameters"}), 400
    try:
        # Check if the datetime string already contains timezone information
        has_timezone = 'Z' in datetime_str or '+' in datetime_str or '-' in datetime_str and 'T' in datetime_str
        
        if 'Z' in datetime_str:
            # Handle UTC timezone indicator ('Z')
            dt = datetime.datetime.fromisoformat(datetime_str.replace('Z', '+00:00'))
            print(f"Parsed UTC datetime with Z: {dt}")
        else:
            # Parse the datetime string
            dt = datetime.datetime.fromisoformat(datetime_str)
            
            # Only apply timezone offset if the datetime doesn't already have timezone info
            if timezone_offset and not has_timezone:
                # timezone_offset is in minutes, positive for behind UTC, negative for ahead
                # We need to SUBTRACT it to convert from local to UTC
                dt = dt - datetime.timedelta(minutes=timezone_offset)
                print(f"Applied timezone offset: -{timezone_offset} minutes, new datetime: {dt}")
            elif has_timezone:
                print(f"Datetime already has timezone info, not applying offset: {dt}")
        
        # Ensure the datetime is timezone-aware and in UTC
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=datetime.timezone.utc)
        else:
            dt = dt.astimezone(datetime.timezone.utc)
            
        # Convert to proper UTC ISO8601 string for FullCalendar
        formatted_dt_str = dt.isoformat()
        print(f"Parsed datetime: {datetime_str} -> {formatted_dt_str}")
    except Exception as e:
        return jsonify({"status": "error", "message": f"Invalid datetime format: {str(e)}"}), 400
    new_event = ScheduleEvent(
        filename=filename,
        device=device,
        datetime_str=formatted_dt_str,
        sent=False,
        recurrence=recurrence
    )
    db.session.add(new_event)
    db.session.commit()
    
    # Schedule the event using the scheduler from scheduler.py
    # This will be picked up by the dedicated scheduler process
    # Use a unique job ID to avoid conflicts and make scheduling faster
    job_id = f"event_{new_event.id}"
    try:
        scheduler.add_job(
            send_scheduled_image,
            'date',
            run_date=dt,
            args=[new_event.id],
            id=job_id,
            replace_existing=True,
            misfire_grace_time=3600  # Allow misfires up to 1 hour
        )
    except Exception as e:
        print(f"Warning: Could not schedule job: {str(e)}")
        # Continue anyway - the scheduler process will pick up the event later
    
    return jsonify({"status": "success", "event": {
        "id": new_event.id,
        "filename": new_event.filename,
        "datetime": new_event.datetime_str,
        "recurrence": recurrence
    }})

@schedule_bp.route('/schedule/remove/<int:event_id>', methods=['POST'])
def remove_event(event_id):
    ev = ScheduleEvent.query.get(event_id)
    if ev:
        db.session.delete(ev)
        db.session.commit()
    return jsonify({"status": "success"})

@schedule_bp.route('/schedule/update', methods=['POST'])
def update_event():
    data = request.get_json()
    event_id = data.get("event_id")
    new_datetime = data.get("datetime")
    timezone_offset = data.get("timezone_offset", 0)  # Minutes offset from UTC
    
    # Optional parameters for full event update
    device = data.get("device")
    filename = data.get("filename")
    recurrence = data.get("recurrence")
    
    print(f"Updating event {event_id} to {new_datetime} with timezone offset {timezone_offset}")
    
    if not (event_id and new_datetime):
        return jsonify({"status": "error", "message": "Missing parameters"}), 400
    
    try:
        # Convert event_id to integer if it's a string
        if isinstance(event_id, str):
            event_id = int(event_id)
            
        ev = ScheduleEvent.query.get(event_id)
        if not ev:
            return jsonify({"status": "error", "message": "Event not found"}), 404
            
        # Format the datetime string properly
        try:
            # Check if the datetime string already contains timezone information
            has_timezone = 'Z' in new_datetime or '+' in new_datetime or '-' in new_datetime and 'T' in new_datetime
            
            if 'Z' in new_datetime:
                # Handle UTC timezone indicator ('Z')
                dt = datetime.datetime.fromisoformat(new_datetime.replace('Z', '+00:00'))
                print(f"Parsed UTC datetime with Z: {dt}")
            else:
                # Parse the datetime string
                dt = datetime.datetime.fromisoformat(new_datetime)
                
                # Only apply timezone offset if the datetime doesn't already have timezone info
                if timezone_offset and not has_timezone:
                    # timezone_offset is in minutes, positive for behind UTC, negative for ahead
                    # We need to SUBTRACT it to convert from local to UTC
                    dt = dt - datetime.timedelta(minutes=timezone_offset)
                    print(f"Applied timezone offset: -{timezone_offset} minutes, new datetime: {dt}")
                elif has_timezone:
                    print(f"Datetime already has timezone info, not applying offset: {dt}")
            
            # Ensure the datetime is timezone-aware and in UTC
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=datetime.timezone.utc)
            else:
                dt = dt.astimezone(datetime.timezone.utc)
                
            # Convert to proper UTC ISO8601 string for FullCalendar
            formatted_dt_str = dt.isoformat()
            print(f"Parsed datetime: {new_datetime} -> {formatted_dt_str}")
        except Exception as e:
            return jsonify({"status": "error", "message": f"Invalid datetime format: {str(e)}"}), 400
            
        # Update the event in the database
        ev.datetime_str = formatted_dt_str
        
        # Update other fields if provided
        if device:
            ev.device = device
        if filename:
            ev.filename = filename
        if recurrence:
            ev.recurrence = recurrence
        
        # Make sure changes are committed to the database
        try:
            db.session.commit()
            print(f"Successfully updated event {event_id} in database to {formatted_dt_str}")
        except Exception as e:
            db.session.rollback()
            print(f"Error committing changes to database: {str(e)}")
            raise
        
        # If this is a recurring event, we need to update the base date
        if ev.recurrence.lower() != "none":
            # Also update the scheduler if needed
            try:
                scheduler.reschedule_job(
                    job_id=str(event_id),
                    trigger='date',
                    run_date=dt
                )
            except Exception as e:
                print(f"Warning: Could not reschedule job: {str(e)}")
        
        # Log the update for debugging
        print(f"Updated event {event_id} to {formatted_dt_str}")
        
        return jsonify({"status": "success"})
    except Exception as e:
        db.session.rollback()
        return jsonify({"status": "error", "message": f"Error updating event: {str(e)}"}), 500

@schedule_bp.route('/schedule/skip/<int:event_id>', methods=['POST'])
def skip_event_occurrence(event_id):
    ev = ScheduleEvent.query.get(event_id)
    if not ev:
        return jsonify({"status": "error", "message": "Event not found"}), 404
    if ev.recurrence.lower() == "none":
        return jsonify({"status": "error", "message": "Not a recurring event"}), 400
    try:
        dt = datetime.datetime.fromisoformat(ev.datetime_str)
        if ev.recurrence.lower() == "daily":
            next_dt = dt + datetime.timedelta(days=1)
        elif ev.recurrence.lower() == "weekly":
            next_dt = dt + datetime.timedelta(weeks=1)
        elif ev.recurrence.lower() == "monthly":
            next_dt = dt + datetime.timedelta(days=30)
        else:
            return jsonify({"status": "error", "message": "Unknown recurrence type"}), 400
        
        # Update the event in the database with the new date
        # FIXED: Use standard ISO format for consistent datetime handling
        ev.datetime_str = next_dt.isoformat()
        ev.sent = False
        db.session.commit()
        
        # Schedule the event using the scheduler from scheduler.py with improved configuration
        job_id = f"event_{ev.id}"
        try:
            scheduler.add_job(
                send_scheduled_image,
                'date',
                run_date=next_dt,
                args=[ev.id],
                id=job_id,
                replace_existing=True,
                misfire_grace_time=3600  # Allow misfires up to 1 hour
            )
        except Exception as e:
            print(f"Warning: Could not schedule job: {str(e)}")
            # Continue anyway - the scheduler process will pick up the event later
        
        return jsonify({"status": "success", "message": "Occurrence skipped"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500
```


## routes/settings_routes.py

```py
from flask import Blueprint, request, render_template, flash, redirect, url_for, jsonify
from models import db, Device, UserConfig
import logging
import httpx  # for querying the Ollama API
from datetime import datetime

settings_bp = Blueprint('settings', __name__)
logger = logging.getLogger(__name__)

@settings_bp.route('/settings', methods=['GET', 'POST'])
def settings():
    if request.method == 'POST':
        color = request.form.get("color")
        friendly_name = request.form.get("friendly_name")
        orientation = request.form.get("orientation")
        address = request.form.get("address")
        display_name = request.form.get("display_name") or "Unknown"
        resolution = request.form.get("resolution") or "N/A"
        if color and friendly_name and orientation and address:
            new_dev = Device(
                color=color,
                friendly_name=friendly_name,
                orientation=orientation,
                address=address,
                display_name=display_name,
                resolution=resolution,
                online=False
            )
            db.session.add(new_dev)
            db.session.commit()
            flash("Device added successfully", "success")
        else:
            flash("Missing mandatory fields (color, friendly name, orientation, address).", "error")
        return redirect(url_for("settings.settings"))
    else:
        devs = Device.query.all()
        devices = []
        for d in devs:
            devices.append({
                "color": d.color,
                "friendly_name": d.friendly_name,
                "orientation": d.orientation,
                "address": d.address,
                "display_name": d.display_name,
                "resolution": d.resolution,
                "online": d.online
            })
        config = UserConfig.query.first()
        return render_template("settings.html", devices=devices, config=config)

@settings_bp.route('/delete_device/<int:device_index>', methods=['POST'])
def delete_device(device_index):
    all_devices = Device.query.order_by(Device.id).all()
    if 0 <= device_index < len(all_devices):
        db.session.delete(all_devices[device_index])
        db.session.commit()
        flash("Device deleted", "success")
    else:
        flash("Device not found", "error")
    return redirect(url_for("settings.settings"))

@settings_bp.route('/edit_device', methods=['POST'])
def edit_device():
    try:
        index = int(request.form.get("device_index"))
        color = request.form.get("color")
        friendly_name = request.form.get("friendly_name")
        orientation = request.form.get("orientation")
        address = request.form.get("address")
        all_devices = Device.query.order_by(Device.id).all()
        if 0 <= index < len(all_devices):
            d = all_devices[index]
            d.color = color or d.color
            d.friendly_name = friendly_name
            d.orientation = orientation
            d.address = address
            db.session.commit()
            flash("Device updated successfully", "success")
        else:
            flash("Device index not found", "error")
    except Exception as e:
        flash("Error editing device: " + str(e), "error")
    return redirect(url_for("settings.settings"))

@settings_bp.route('/settings/update_clip_model', methods=['POST'])
def update_clip_model():
    data = request.get_json()
    config = UserConfig.query.first()
    if not config:
        config = UserConfig(location="London")
        db.session.add(config)
    
    if "clip_model" in data:
        config.clip_model = data.get("clip_model")
        db.session.commit()
        return jsonify({"status": "success", "message": "CLIP model updated."})
    else:
        return jsonify({"status": "error", "message": "No CLIP model provided."})

@settings_bp.route('/settings/rerun_all_tagging', methods=['POST'])
def rerun_all_tagging():
    try:
        # Import the task for rerunning tagging
        from tasks import reembed_all_images
        
        # Start the task
        task = reembed_all_images.delay()
        
        return jsonify({
            "status": "success",
            "message": "Tagging process started.",
            "task_id": str(task.id)
        })
    except Exception as e:
        logger.error(f"Error starting retagging: {str(e)}")
        return jsonify({"status": "error", "message": f"Error: {str(e)}"})

@settings_bp.route('/settings/ollama_models', methods=['GET'])
def ollama_models():
    config = UserConfig.query.first()
    if not config or not config.ollama_address:
        return jsonify({"status": "error", "message": "Ollama address not configured."}), 400
    try:
        url = config.ollama_address.rstrip('/') + '/api/tags'
        response = httpx.get(url, timeout=5)
        response.raise_for_status()
        json_data = response.json()
        models = json_data.get("models", [])
        model_names = []
        for model in models:
            if model.get("name"):
                model_names.append(model["name"])
            else:
                model_names.append(str(model))
        return jsonify({"status": "success", "models": model_names})
    except Exception as e:
        return jsonify({"status": "error", "message": f"Failed to fetch models from Ollama: {str(e)}"}), 500

@settings_bp.route('/device/<int:device_index>/update_status', methods=['POST'])
def update_status(device_index):
    all_devices = Device.query.order_by(Device.id).all()
    if 0 <= device_index < len(all_devices):
        device = all_devices[device_index]
        
        # Update device status
        device.online = True
        db.session.commit()
        
        return jsonify({"status": "ok"})
    return jsonify({"status": "error", "message": "Device not found"}), 404

@settings_bp.route('/devices/status', methods=['GET'])
def devices_status():
    devices = Device.query.all()
    data = []
    for idx, device in enumerate(devices):
        data.append({
            "index": idx,
            "online": device.online
        })
    
    return jsonify({"status": "success", "devices": data})
```


## static/send-icon-old.png

```png
�PNG

   IHDR   `   `   �w8   	pHYs     ��  qIDATx��][�E�E�(�h����jLLDQ���aWF��h�;�n�_��HD�S�b�7��a٭�@�$1�c��2�v���������[����[��c
(P�@�
(P�@0����׹�]B���PX���lpD(�
^���Ol�pY~�K�KZ)"yRv������
�	�W�^�eH��%�!�D����]����%E�|>ֱ�)�q*գ7�7�6;�W����ժ|:d��|.�7.a)�ִ�=�YNѹ�g��ʏ��B�&���~O���M|2c�����P��}���U/v���]�Ϫ���*���4#�P]��נ �������^�f7s�%�����4��9�P�,23�%m4��ky��܆�T��'
�4~�	Z�> S�r�	=�����s���T�R�����GɉV�N@��:e���׳��}Kƙ
�p�6�Q腚M>������d]��*k�
-��G/����A��ñ��K�����I�E�,�X�(bp		ۂ�"��Au̒�=�ĥ5�e�=\�G�����:.;BXw��`N��[���\�Ű�~W3C�P�Af�,	�f׵sG�x������ӹ;��XZ\�1�N������XJ@%D�℄_�o(�y��5X��kܫD�8n*x)�z���9���ˎ?M�ɯ��ϓ(u�X|�!�w��C��b�������;^�%:�f�nk4_*�����������41��w�]��iVź�ټ�" ��s��4#�M�JF��v�}�;��=���eSM@�
���P��\����c �R;4���G�TJ � �u������k��_טH*Wf�c޾y7�z��w�d_��a��mV���1�b=!�$�#�%�����=��;-���(Ck�ZL32O���o�)�����d�(wب.#*��/�7hn����,���l� �d� 9��咀፰��%�k��s�3����g*x֎�l�VB;p�p�6N�_EJ�L(Xe(|�YG`-�{C(��K���S�S(XK��6>�c^Pem�H�\���č\�Kn����z���H@PԊG߅=؇x<h*\����,,�@����#��P��F���V'@����AI��M����U���2��+�� ��Zf�$U�&~����FB	
V%����VK *�)P�3(o%����IU�K�+t�5Kp�����/�=Y(���_�\�_Ի{�'���r	��e����&���tF��	�@·��"
0M����'�S�<��nEB?�$�[�d]l^����[ 4��^�U�&Kf�?QG�$H<(7D��Lx"��F4�'�
>�[@���Ƀm������-Ix��4Cu�T�tCD�A	\§���]�G��f��-s~/�K<���+'��tM��
�A�t#�K�����ٻ��Bp�>իE�p2��V�F}[F��^�8PK[��=
/�\`��f�Ac��%��J��
�)��ݚ0/�z��t}�LP=�Zഘ���tI�7�F}�e��d��pXCDt���������am��i*�w��,o1�ub\��,o0bHE8���T>}!�%~�gC�;Y���5ax/fw�3��Ҩ�Fp�Ӽ�kc�+�,��^�	�	��=nS[��p �KI"��pQZ��*�g� �30�Eҵ�0�V��T�&�$ueGq�H�a*�V��鉠]=�M��m�K\�ZUjO���)�OZ��p��W"V���N�{�G�ʷ�6�r��fN�.��ۯv�\f��QV���L������q�05�7���N��Q��äH�X�j�}������8�\M[��B��ܦ�(P� ��1���`�+    IEND�B`�
```


## static/send-icon.png

```png
�PNG

   IHDR   `   `   �w8   	pHYs     ��  �IDATx��?hTAƟh�XX�Z�hc)(7s$�� o���mJ[K[K۔i��0;���Iaac}��"D�pg�E����|?x���ۛ��{w]          �!d�~b��A(����+�'6}�>�/x�#,�/J�GI^Pzp�{m!X���tklr�{�!8R�黱���;�ސ�6�>����g�����'Bf������E���,`q�W6٤t��}-`>����%Y�����q`C�<�ؕR Į�x^�ε >zN@�<��x��y��OD`�������[����b�0/{����=X;�xj덝w�]켃��b�G;��8��y�����?$�-v��pt������{\�F;�����/S�o�i��k�<�qx�锒~�$��/B��HEQ�b�RQT�صX�$vm%�]���d��T oˍ�uF� 2�BI^N��W��lS�pM���fwb�����-�%�X�`��G�9��Z7$�7���y����7n�~x�%��^���,Nm�v)_�~V�85|*H���������w���8妾�?O-N�)~�[a�d�M�cf�Xq�MY�]�����qʍ�٭N�r��^q��p�]���0f���_onV~�┛����)7��Lq������┛>�8�愣┛%V�Ӫ��n�8����3�t���n�՟�^           ]�|gi�:�.�    IEND�B`�
```


## static/settings-wheel.png

```png
�PNG

   IHDR         �x��   sBIT|d�   	pHYs  �  ��+   tEXtSoftware www.inkscape.org��<    IDATx���w�U����I/�$��{����ذ �"�rE/W�^�]���k�],�*��$������}~�'�9{f�g����<�'q�63{�5����������������������������������������������������������������������������������������������������������������YZ=��%0x0����e�jh2q,��cV>w ��& { ;�� ���K������ s�Ȩek
�_'�ll���.�n%���� *ɩ�Yˍ^�x
X^ � ��$�������u_�z
8x5qm��Y�v�I�b/��^ك�G�(�5�����_�)����7�]**��YcM��%�{�X��x�`Ͱ�k�m������^A����8�9�~�%�b�w�V���P�UQ��G�,헸�ff�5	x7p;���`q�E�r[[W��~����׺�Y�m
|�t��ˈ���*�J�"b٧��YQ�� l����rv 1{��,�b)�Z���n�sM-!�&읤&��228��÷�xo�bey?��۸x%0��Z13� � �@��-#<��w��.ʈ��{��,��jk5����2c)pd��d�A����2c�qbKb3�ژ�H쟮~���'��Mk;b���zH�⎀���Z��j6�g1�VB�Yw��BT�X���33+���'�MO�ʪ�br�Ukp)���:>�w43��Ą�Y���8�����Fg�owe<�w>ͬb#�%Km~J��S/�V���K��_{fV�7���_'��,��DHu;�Wv_�ff+�p1�]��}��K
#��o�����]ֱ��3L#�\�<J���|�`�F��o׺����:�T���hb�_·�����eX8}{�1� ���U3���{�2�v`�aֽ=mS���X��p�0���ZfSbI���դx8t8�` <��;u�5%z�9S��f�|c�� �?��ˀ��ZC18	�9I�o�ע�{������FCk�V��;�vjC\�:�f1��Y�*q���aԦ��i��&=�[h��>9�R�0a�MdfMq8p�P��<A`+�\����xΪ���m
�bB������)�hL�L N�[���L�����xVu�q?��{6�����������̬VV~����Xu�K37n��}=;Vˀ/�mI3����/��1�x�_��Z�~&' ��׫cxq>`ȬvF��3��s��\kw��>��w,%�e�Wf5�-p5���ܸ�ob��\m��}}9ʍ�0�,� �O�X�������%E��عq��F��^��^R���ה:/�xx1f������? ��������E�-e�%΀� p��3�5�=�?�I�	a����<���׀��M���ձX���3`����B�����u�p��~,�S�u�O'f��щ���Y�XJ|y/�����j�Q�a1S����'6+ڢ���U ˉU��M �&�n�Se�2rpq}���[h�>�����>�A���14;��>|��$�Y�& ?A� �:"�̇{�i����G|��[�X���Ͷ�K�(jf�m\����2n#֒٘��xP����;~L��r��7gP�*�2`��ff�p͟x�?.".*���H���ˑg��b_��� ^B,�S���x��|��Jv1iK}�W��n��H�����+N#ݒ���33(c����df%�*���8�j"�JP>G��s�9�� �o��)⛴�p+�����fNg{�UaC4�F	yw�;�Fy��CՖ��E����̬uރ�&N�C���׃CE� �?G�~`��*̬-������x9�l��.�[׎�����#�{B]/��	b�J3�uh�z����r<����=u9��lb;�܌�I3���Ė�/,����,�7l�1�8�fB����6����ˑ&n�&o�Gy7�P��s˫*��9
��Zf�ذ�Jk
p�zs������m��8|G]we��Ĺf6�x�>�7iqq_�>��4!�[􎤞�'�%u=�7ˬ$�&8��Y4fwʑ�{��:���N��C4c��(➚��Nˊo�ZCf57��$��1��^�G�ZeW��T�����1�8��]� �A_�eD/pH��cV_Ǡ�)����C˯�l��ƚ���1x+�,1M�p�1Y�f�[�Pϝ��_$���e�8}�;�?���`�Y`	�z/�,�b��f�w��u��)*�^<���Ǯ�ŚkO�}����O��z�q����[�^)��%�d�g����n,1P�k���W�Y}���&J�욨�j�Cu۴-��4��i�gעo���%)*ì.B�,z�/�mpG����s�[���M�hp2��}�F�$6A�,�NU��	܁�ݚ���뺡O�$N�S��P��T�Y�G�(~����5
x#�<R݆u���7Ќ����,��m8��,Q�e�x�7����'�a$�j�b�mZ���82ڿ���!��^��MW�i�rb���$�7_���>i��iw��ăN�ƹ�"� �ݺ�c[���{[��+��$+�Y�����?VK[��[xp=���%�#���Z�^m�& �G��+
o
d���7�b�T�֮�׀��_U�C�)�.�kц�M��J�+)m��o�����Mg$1��%�=q�Nb�����Ԝuk�;z���%6��W��tK����҆��Ĳ�s�	��s�1��$�D`�Rk�ʰ6p��/����f� �ￒ�Һ5���b�{�����EtZ>J��o�v�u0
�*�kg9pu�� M>2�.��@��+���y��ۘ�?�+��zK`sbO�*,�&6<����U�����w�.Ż�x5aq@o?��{K�����(�-�M�u���"�z�&u��Ľ�����6���ģ�/���t��Alm�r�>��-L߬rk�z[|$uA�,[��:�3���Ki����zs��ԙ �Tg��d^�~D�aq���@.Pg�x�<M�	3�8Z�b	�U��<��� ���:fV�͈ j��3�6� ��l�	u&��3`f�{1���k�0S�)�����cW���O�=�&/��� ��� S��ՙ0�ʌTg�(�@>�F��΀�Uf_�8�w���; ��%��U�G�3��a� s��ՙh#w ��ku��Y��0��H�	�܎~B��Kifj�yzhe<7uAmp�ϟ��r�����xqf ���Z� ��� ۫3`f��p����R2� ��r�U����Yz9t �Rg�����"�XL%� �5�.�����)w �)��;�5�`Kq��g�K���+��%�7�f�����鷞; y�N�`#u�,�m� .Qg�����L�q��of�l-N!p�8��@����̀̚kq�7���r� ��.q�0k.u�Fq��; 9�.N_��0�t���8}���M�?E������8�[��� ��q����7�4F ��y�W���@��?I�����,�$@����� �l�8}w ̚I}o?<)΃�@�� �0k����?*N�:�ȗ�D�1���,�����ӷw �<'{�0m3KG�
`�8}�p o˄i�`�L�C��ӷw lE�0k�q�ʑM���|�A�T� �fR�ӷw 򥞩�T�����/`�șz��g�5��0N��u���5��`�L����=F��@�6��:f�4O��zt�:�ȗ��.� �5�z��5��[�; ���������C��p g[�ӟ!N���Pw �k��`�������/N���x
X$�����w r5�R��{��Y:��W?�w r�0J�����Y:��[=�i���}��# fM�����}��?<*΃���� �$N�p W������,-�$߭��v�z� �gS`3q�0k�;��� {���z� ��� 7�3`fI����a�S�����3 ܠ΀�%�00K��C��ee�I�ra,&�.������f)� ����/��5�|q�,��������˫� .Ug��*�Ou�#�0��:��rHn91
afͷ��͓�����y oA��/�e��Y%�D&�j���y0�E�̥�	�Y�~���sF�Rڠ<��#�ՙ �Rg��*�Ü��3�F� ���t�Y�3����ޤ΄��!�����щ�jfy�f��<��?����G��8�X�`f��8G�	`=��L�U)�_����g�V�D��Y��(��	\���[�#��Y��A{�,�J\V�,����?O\V3��_�?��K�s��,�Ձ��l}����5�̽	�s�/ޚ��fR�F���L��6k�5���Gˉ>��� ��M�_JZZ3����?���3��j�P��D�:���[{�[��%v�ܠ��F�ހx�4�8jz1�s40��u=X���9��~�(��NL�GF7ѫ�Sՙ�X삷'O��j�x�:���⨛�����rg`'`����ppc'�'�*·�g,p?0U���K���Z;����8"i��}��}�x >���YQ,�|�Xεn�������:�'�-�Yz[s��L����F�F�'���_+E�.�;������-�e诛�X l���f	�܊�F�NYh�q�ˁߐ�/��188��`y:���?�#^O���(򻙖{�OXn[�Q��?&�ѡ*b!��5x�Wn�D}�o'-�Y�@��IYh[�g�f��r�����]ԫ�gy�Z�9e����q�7�`�0!]�m#�!���o���jb'8��z�ka`,�JXf�R��fYQ�/a��V'f1߃���� '��Z�F��u;�*a��
y'y���?��f5����o��S�+���V�\��A>{��'��9VoHVr�D\m�ԗ:�	="P���uhY��
�$�L�{W�o���?��TV>�F��M���/S��2V�[ѷ���
�	0�q�/����SU@���<��}����|�eZ#�-������wLdm�R�7���T�b��*�m{�����b^���W��m��b{��,ف1�4Q��d�`)��u<���R:}�,�!N65K���J�E?���y1p�6u�oç����?�f�� ���s�_�C���Ŀ2L���=C���j����mWK�	�V��-�ky�B`���2��'���1��3H{Z�V�>#`� 6�2+d��/���GRTF��"�s��hT������#ǃ�Vw{��k�)������`q�|Ǻ��'�����x�X�~��M�K�O��m�V#vs{���M�6.�V��(�I���(7�/��0���M�w��������֗�]1-�|�|�qp�^b�N���"�w��?�/�:����N3~�}��i���Oз���8�8�������&�'��Zk"�:br_S&y]��P��Z�����Qm\l�1�}[v������j�+�Y�O���ʌx�s7v ��o?�&�㥲E�D��V�e�x��v�2�د������/�� أ�:k�]��|G񘍗����XF,N��w���qDO�X���_hޯ�ű%�_��N���m��#��[�u
�v,3�>	lSf%�N���p���Ľ-�M:�֡��CW6t�g�]�왞"VМ��HM�&�T=G��n&�g۫/�f�=��8��S�
s�o�׶:����t�����nM�Dߎ)c.p���s���V��w1��7x�q*>�g����@��|��+�*[Ҿ�5��'���^���8�}���t[�-�����QM��k�؛x��nGe<�xކx�6&��Ս�{�o^1\�7��������"^J�+W�c1��$�Y�M �a��_�_l�r.ћ\��H��\JLm��������}���w���B�u�d� ����i�k���̨�t`�i�w�#����2�W����ڬ��}��G<�4\��ѷ]�x8�����2*o&��> �N���&�$�%�Λh�n�e�Bb�ak���נo�:�o�%66t#�?�o���8����N��_�=�.��Kh���Wlo�~U7���&�=�+�q
>���B�ve�2b��[��K����	���4���멍ށG��}�
K&ԕ�{,N貎��4�As?���F��P56'6��}=����eVP��v�(�H���LnB_ѹ�|b���4c�ߋ�ӌџ��+�w��z-sh߄Ҳ=/�]Q����jA�������^b�z����K�<���+&#�����MxR`Q{Ҿ͂�ߥ�#'�����bٚu��۰��;�:�n_bOuu�w�*�JZgb��-s�����<�8fQ]���2�=o#�~*r4�v�&�^��>ꠇx5P׽�_R~�����Vu[��Ğ/�7�z��S���]׮A@R���y�ҹq	�n� &�F��e8��=��L#��P��B)�q`������Wd�q:y/骋_�o��ĥ��=�v�o㦳��D;EL�T�iNq^�MlC����b.�1�w$��j,$��4af*#���tj�k��D;mDl(�nӜ�U�jT��+/�8�z�������t(q�c�jh���;з�P�<y�L���Ҝ�<�����B5*r���%�'&<Yy~��]�g{`���}�%~�������T�xw������+-�X|�8����\�4����ָ5���O=�88Q��Db;����O�΃��JSǙ�<�F��AR����z9�����He_�5����Q�e�uy�".�/^����,�여���7�/�|G���X��n#\�'���,E\���6�}[�(f;%+�mG��?���j�����XLM��}eU /��z����{Eq^�_�́��{E��tE�~� F[��]eSJ�%�5��:��#��jL%ߍB�NWt`K�A��>X<����6���R���J�������RŃĶ���V[6T�v,���
�k���?X����6�#��o��qci��У�+��X
�	x)0��z��[��e���<b��i�A�����{MB��j��e���l����ʈ�Ļ�w��|�51���I�zG��_OYh[���6hڪ��ʬ��=}u��Ky��]1ֵ	���C)m��1���������� W��&ʈו[=�z?�
j���v�GM�[n�w��V��x�GNz�S�_�#)mö-p2y�"YU|��Z)�O�WЊ�b3�����-o#�;�_;��_�d=nˬ܆����9A���p;��d8��Q�k�W��o�w��/6NYhK�U诧���ꗳ=�S��I�xK�[�'v<�<'���;�Aa����rޓ��V����p��Ĵŵ�Y�Ui�k%@�+�;��k�,%�Q���W��Ҫ�5��\�{����诗��s�[J#�gѫ�/�Ǐ�,��c�W���5ǧ�_O}1o�['���_7}�ŵ
MC=���(냈�Ni�S����Z��ib2��íį�\��X�j��0q�Ri�̗�ش��������<�3�q�u&l�>ܥ�D�db�5ÿ��g�ؤ���; ��zu�y?1���eyM
~�:Vu`sq��R�����hX�|���[�Zz�L\���h��^K���(k`4�	xh���|P�+�c�t�ï�B=��zTV`C�K��h��3�q>�����y�Lt�΀�B� �H���F;$27}�#����.L\V���诧�ī�1��j�B�̴�Je� l\��t�>q�V����$�ˈw��W ��3A���_�	+l)p�8�� ��ؤ���ֽ���G�3�q�:V���3Б�5nŨt��L� �I��g Vg�J�;�m՞�΀��~q��G�V�Z���L ?��3a�[J���=��:V��;'��zF�z8Ɗ{��$�D���'D+� ���`Ź��Ԓ>�[~P9��?_KMv?q���; ��� �Ot&<��    IDAT)�0�X���v���� �[u,�S���&Pw F��Β�]��б�Clo������DKk*�������d�5�5�جh!�X���(b&��Uw[�[*]F�i�6�X��1�d@����.��[F�Y%|F��ӷ��Rg 8]����� ;�3`�=(N?��Z%|F��ӷ�r� h�C[�΀���E@=�8}+n;q���[�y��\�������+ �# �ي�T��%�ׇ[u��*y@���{�� �G ��0V�H�I]"Nߪ�q�� �y�b��)E?�	 �0��!��U)���Z׉��A������b@�P��b6Wg �I��ܵ��?CŊQ�d�_�g�n+��f�Dߓ����A=�ŊQ?7V/�et Ɣ�E�Po��ӿU����(Nq�V���gl�(�P8-�oŬ#N����K}��z��8��?��0�X��Sx����雎�@�j+f�8���0�@��� �������� x w �N}��#��MG��+�zS������+�Y��MG���$Nߊ� ~`�x�T���~vZ1� ������7u��H�K�P?;��w�; ���%N�8}�Qw�<Po(#�Po�/`u���p����,6�)�3�X&NߊQ��7������
_?et ���´��6�/p�++���O�%%|F���?���b���>_���~�]��G LMف�N��3Q��G �M���0m+���z��>O� w �M��b@���m�8}w �k�8}��7u � ��"�;�M�o:��x<Po�@# �u�� ԟ���8}��X��z�ˊQw ���� �F���>�wq����8}�i�V�8q�Y� (wC�A��ǊQ� l.N�t��������,� <Q�g��Y1��_��v��=Po�%�O��2: sJ��"��7�+�M���V�5��yp�ަ��/�㩌�z"�; �v�8�`7q�z;��?�@��8<Z��00^��s��$��o��K�~/��8V�z���� S[�~`q�V�}��?��2�;w �w <P�����p�Uk?q�׋ӷ���Ǌ~@^���Xq��:���<Xu�G�����8�$@� �o+N� 8B��Lm�@��#N?��z@}��w�:������ ~��
ӞG&[�7rq��nD2���Zj�)��<�������5R)߻et ,�3��+��[\)��H�e�<Xz/C�˵� ���F<3T�� �B����f�X����XrG�3 \�΀��8��e|H�^�>�[�m�K� �Ǉ5�:���L Ug�
[O���2>�� �̒>���m����{�N�K����J��c�ˊQN ���)����R�V�y�u�L����,��M�L]��W+�# �('�Oy�0��� 6^�΄���q����X)6�?��i�+���Z���<Tg������Ҧ� 4�´��	��7o$2��]��*0�Xޢ���⠴E�
��zZ�F%.�U�!t�Qi?��P�k�^�a�X���D�����|T���s�Ot��&��hzYTV�Β>�[� 4��t��Xh��y���+u���(qɲ: ��J��n���+r�y|LpݝLm88W�	+�z���: ���%}V7^|x-�������*��>�Q�LX�^E� ���G��G=���i���'�􏇉�>K���Q��x��/�D�w��$`��/I[\����^K;�/��}�M��x �=p<�U�j��F;�v`|8mq-�/��n�b&�\��F;�I�NG�Ri)0.i����7�pc:�]����׈q2��/ۥ-��hG�2�u�_M[\+Ѧ�v���'^ۨ���q[��s��)��ݫ��J��-��%��/.ǿ��`����z�^�y��z�[�S�b�����}�j(n#��SV<�x)޸C�诅��޴ŵ|�u�?<�?/#�=�I��$�C�O�_%�A|q�+��x���˫*�cз�X 운�V���i{���0i�m��'&�ߋ��(ǔ]1e�}��e�Y�19�+n���ju���;��)m]Y�ؑT}}�;�AeJӀ��E-�;�ZC%�:�
�"n �ï��ې�r��"w��C~ː�'�,���n�$���H/&�%�G���*�N���/UKcu�q�m=0|V@>>��zO���*�%��}ۧ��J��D&��;�*�~�mx�x*'�oぱxy�Bې��h��00�������!�����I?(�Β�m�v�qpx�*��$~Q��w`, ���ܶr{4�����$�v�rL$V�<�������Rs��}E��|b3+��з�`1�)a�mp�_���,>��ح7��"�i�Ъb��/�	��W��R�bb7�	Ū�:��ѷ�`q���dK�۱�,���Sy>�\�Ɗ�����̧�WX.q�tЊ���\Q� �NWt�ؒ��[��+��IV��ڎ�PIݶ��X�Z��T�Lwu���&�*[�&��F�W?)�N~�B��_���Srۏ��XF�[+E_q�Ńx�`QǢoǕ�#��J�^���;����$��؏8�^ݦ9�
֥�x�n���[�_�ԏ_p1�v\Y, ^��Z��D���ueq%^\�	�7�si�*�.T�B���\Q܌����'�_W��(�w�^��O�m��%Q��.����3�8�P�f�+�+1ט����m��o��ĩ��Du�dkF�~C��&����އ�����6�5c��Wf��K�����Y�|���;�Q��P���f�U[w��Cߎ9�g�kV�w��М�|�J`��D�nC�%Čfw�V��8@�Nۉ��&�a7bu��Ӱ3f6�����آ��m���o��ą��Ij��v.B�>ÉZ���ı4﴾��I2?��[[������2��L�:���S���G�L��k�g��n�����r�e�K����F,�QWt�18��
n�É��vn��8A}�������ˈ��lx��&���9���ڭ���w�Wvαx}��B�B�f�ƥ�A��H���ާ�~��*i�5�Kз]����k������:5�o�v�ep-�6+�ƛ~���Y��H\�.�bnCb�u��K���vkn
�m���!rw�nc`�6+W�ь/��P�L��Z4�$#��ۊ�,�T�������
�%�ˀ7v_��r(ѣV�Y1���n�#�4���w�Rk�����s<��+�M��݆ڀ8�u:��)���Z[���۫춿x'��F���sh^G�}%�Sl���(���{��T����x��rb��O\�b^u{��%���E����&vs��R���J��v،�-ͭ"����j�U��JF�%���M:�1���-��=I��F���M$Nl�!-�� �!g%Jk2�a���đ����������3R��Z6�#'�{ܩ��3թ�2#���g��݁Q�L%��/p�:#��Ft�6q��F���?�K��}�I�O|��%�O�D�+�gub1�`�NlKt���.bUƣ�����N�;�3���߉��ӉN@v���jDG�P�eİT��FtWg$s�돳v�Zx8��:#51�X�y�:#��_�7�p� .��{��N��췣TV��(��j+b�eu{9����D���ۭ�����h�<bt����!F~H�OW_PE��VOc�<���y�|�p}}�������SV�YV'����^�eWLC
,D�f�<�	��c
�ќ}7��>b���4�5�u��@}O*�Cs۪DL~S��#��O��c(���$uە��?$�U�K�����<���n\�ɿ�ڞ��nu�9�9Ą?�	Ą7u��;�wK\̀��?I���?��2j[�S�#:�^�6|?F�vE��U�7Ҳ�mI̴W_�C�e�Ij��6&�S��͡�[����:}�u���͆�Xbi���J�CLn���F�P���ڸ�8���g}�1��7ܘ	���ng�m@r����?LTM5�X�n7G5q:��:�L=ę��N,�	���>�ez�S�r?��h��ۈ-r���Hw_����v�з�p�`�$5a���_)p06U4؁x��&�,|�fQ���z����͒X��ž��@��7ۆ�߶��Ǖ�t��������MTf�0���~E��;�*�o�&����.z�/�X�^��-���z�b#�����Wg�+z+���c�^2H{��F=�J�2��oB�C�(�KX�6�L�ST��ch��Y�������se�x{�қ�I�o���┅n�����6ǃ��+l=���׈�v]Q�^���f]�,�c�xa�B����>�з�#��6� �l9�1�']�ͺ�|�20n F$,w��\��]�W�r�\;�o'w>��26
8��20�IY��^	LG߶m�����ܩM#�]O���}�&G��o��qM���D�S�;Iu7=f�Ǜ\��?�v,�5�V#;��o��qX������'зs�b>��˳��;}{�^�P6�Z9�X������/i�k�d���i��,b)ں�j�ֳ�s���,�YJG��]����L�x}��-�����W���~`�,i��\��Fꋟ�-�0��,��Y�K\�
�ا�9y�V.'�-��f�����4���)i�k+��z�^��A.18�Xzf:_F-�9��IKlV����T}q|��ʍ&Y9�vN|��h�`��/��oΊ'�Y�� �W�'.��Xb������3�]_I2c�8���?~���f���{���Նo�����b��I����q���XOV�K�_/}1�������o�.�6�&�p��#,�>AҲf���������N[\����Tb��$q>&N�Z&·�(``[`;`��?oI�6��"b�㻁[��:qWE�[�� �O�����3�� �|T�	�@�"u&��Q�f�Ds�΀�����y�Ļ���/Otbn�?� �_�}�@��[�� ����-&:�w�3b��$��&���5�l���_K\V��|�M7/�2k���-'F��%.�YV��C��lm��A� P?�N\V�,���H�R�Yn�@��YN�5Y#qYm���[�t�8}3����t|��lj�JC�_��3�ͬ.F��>��Z��oD��5k���c�I�L*�W y8�ء��ͬ:��?�}W��6s �?��8}3�΁��H^��D�����������̪������ m��q�υ�>y)�Lm<1�W���%��[�# �XH�P���fͷ+��"W����z� �����w�of���� �;u��ܨG v�of������:f��!�^U��{ }�L�����O_D
� �e9p�0�u�fM6�Z�����[�; ��F�v��0}3Kk`�8�ӷw ��Oq�[��7�t�Sg�X`p ?7����Ysm)N.p�8��@~���_G������8�k�M�,� �g)�	P�*L���Z_������w �t�0���i�YZ����F��Ӄ´=`�\�W ��ӷ~��ӣ´=`�\�9>w�ӷ~�ȓ���0m3Kgq��C���w ���0��´�,u��)`�8֏; yZ$L� �f�$N�q�6�; yZ&L{0R������ <!N�p Oʍ� Ɖ�7��_,�o������=`�<���?%N�p O�#y�s�,������ � �I��� �&R�,�G l w �� ,�mf�G ƈӷ��Ӛ´�0k&w ��ȓ����5���^]���@��Gv*�!6�t��_K���@��# ޭˬ���5�wNV��l&L�aa�f�Γhw�G��@~6E��O�2k���\q6�o����m����of�(O�B����@~v�?C����s�8}w 2�@~���8}3K�>q��N������� �!N���Qw v�o������Β���ofi�+N`�8��@^����fSw z�]�y�w �r�8����YZ��3 �΀w �1x�8��ofi�D��W=�i� �� `5q��of��$N/�G�[�; �x�8����<�Yz7���G��@����z�ǅ�Yz� ���0w r�"�ge{�߬�Tg x	q�	���7�3 \�΀�U�&�8X=��Lns����XLM]P3��Yh�9ˁ�'/���G �G�ע_dfչD�b�Ӛ�L������& oPg8O�3�������Ug�L�-����.��ee��g�թj�������Y�������:uAmp~��y�y&^�o�Fg�3�q�:m��N.a�F�3��Ы�~i-t��7���ە�C��j��;1i�vWg���`�N� S:�~ʀ�7��e���$>J,���<g {��0��FZ9w 4v"���X��������z�:�4`����$q���������~�U������N�P9w 4r��z?p�:ֵI�(����;�v^ϤNl�;��A<į���7���tbS���8�Vr@csu�[ [=L!&KF|�n�����mЉ#���������A~~����8}��|�������KiE� ��O.'�L=Y�h��_ ����r�&1Oym����f��9ڛ���E�.�_�_@���:f �D~�m�1�Cy-�'�Yr�G{�����6������RV�L���1?�M^AL�T���#Yk�������`�� �$b�E� �)����u�ڪ�?�G���o��,?A�-'��|g�6!~��B����l�x�X�&� �۷|?i��2�y�7\_�왶�F����}��-?�v��@[��oӁ��e6��	�o��� xS����ĩk���9z���;���1�q�`q\�r�e��o���{�Cal���M�ˈ�49�Q��'u��,vKVz�̬���[Q�L�*g�Y�6�}[6=��Թn�z�m��X �NUf9�����������x�=fK��s���;�>����xa�
0����x����Cdl�'��U�W��z`�U�UL�ϗ�x�j0����o����Ĝ�w�������t�?�ߏ8n�}���0��*IM�el"������<[x�8��q G����g�gB�Ό�,Ee�����߀É�����D}L%����±���Ka�d;�,�뻛xs��0����߀�<T����>rw8� �6p/�&Yj���v������q�׊Y�����M�<?A}�h,p
޳�α�X)�;�D��H�w�bV3�S�/���	W�Q�����5�~%0���R�n����ߠ�!��Lb�q����71n%�`�������2⨒�Ƭ�6$63QߔE�"��O�ۉCh�u�Hs�C��5�u�u)��+�z����o�2b	�e�;Ip�5���H����D�F_We��ZeV�YS��ZV<��z��}�9��O��u�/p��)3���f�\��F-3. �;�n"���rh�'Ĳ:����2�����K�'�FZ��L����ؤ�j*�z���ɡ�SѝJ7�00oy�k�WyUe�l���iˎ���IM�X����H���Vep0��2���{ʪ,��X���"�G�sWZ�}}8򊳨f��K��*�-���Zfp&�9UL'�W��C�Wׁ#�8�t�����1e<�'��68��M
�wSmG`pK�29�?���;��r��K��53+����o�1�د=嫁	���Q��:��B����N|_|�j�PX��}��ۚ����:#�M>M,�ZR��"��}a����G�+Kn'�m~x��ӻ������|,1Ys*�knm`�NlELTQU!�>|����A��o]v,b�n�{ꌘ5�$�4�=�*�>b���Rj.~��˔2��ۼ���Ne��?��8KT�*����j�� .� �U��}�Yez�K���U�\�K����C�#E�ک��=]m�+q>��2�uSf<��7�K,�H7�+  �IDAT6��V+�̬b�E��*� �b�;��f���,��a�C���.�9u?�z������V���2���Faf2��-l/ ^ɪWlE� ��[4���B���l	|�����Ǣq�k���	}3�S�1�v�k0���į�%����g�M��������;��)_݌!���}����A!�њYf����e���KyzX�Χ+>|�8��iF GW��g��b	�ڳ�ì���#mg����&�"F4Vذ���c�=�%�'6��x�,L��p-�NL�h��l�Q�;�3"���xf���Qaf5�p���c�q���a�L~FsV�9!Fg̬�F��8�K�/���%6�R�O[�t<�ϬQv�B�pqD�O�
g��L�wn�-��ͬ�F���wh�b�}[�c�y�۬�q����5�N�?�?t�ˈ�r붑��N���ۯ�1����kҬEFƣUœ��w��b�u;6)N�������E�FM��o]�~��=�w'�����C�NM�{�-���j'⥂���`��W��5���ўc�SǍx�5�p'`8q���l�v.G�Ъs��5܊�aq'`���v��������,������zïr���o�c�E�y���UdM����n�A`�jں�Y��S�lY�F���ي8a�C�+���=��`�Z�#����[��~3Kh?bB��a�[�+)Lc4�7�ׁ"&��,\�ff������/��x�
�RL�]��%���K�;3�aMl#:��Pǿ�r�3�*F}M��E�AI>����&G�p�:�)^�V�7��.R}��01�M&�x�ò���_J�Yپ���(+�?6-�����|���!�����7
�#�k�H,~lWrݘ�%7�ح�A�Ӳ���Z�� F�7?%�ޚ���D�4�#p#�y]�ߡ�f�s���3k����k�?l����ˮKj�����a��x�^3k���w���?����~��H�z�����63�Ֆ�Y��?�WĿК`=�讣e��È���Zo
p"��䶔X���u�<�'�]G���mVE����h$�"b�������ׄz�������-sB5E23k�u�_�wR��\�=�Zrk���P��3��{UZ3��!&�XH�/���N����n[ o �\<Ċ��Y�?����V�]3�vX��\�2�YA���,���2E���$Nܬ��a�Ֆ'6Y�M�&�[�g�K��$��^\���wff���5ͺ���yk��� ��G����e�33333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333k���jb����a    IEND�B`�
```


## static/style.css

```css

* {
    box-sizing: border-box;
  }
  body {
    margin: 0;
    font-family: 'Helvetica Neue', Arial, sans-serif;
    background: #f0f2f5;
    color: #333;
    line-height: 1.6;
  }
  .container {
    max-width: 1200px;
    margin: 20px auto;
    padding: 0 20px;
  }
  .card {
    background: #fff;
    border-radius: 8px;
    padding: 20px;
    margin-bottom: 20px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
  }
  
  
  .navbar {
    background: #2c3e50;
    color: #ecf0f1;
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 10px 20px;
    position: relative;
  }
  .navbar .logo {
    font-size: 1.8em;
    text-decoration: none;
    color: #ecf0f1;
  }
  .navbar .nav-links {
    display: flex;
    gap: 15px;
  }
  .navbar .nav-links a {
    text-decoration: none;
    color: #bdc3c7;
    padding: 8px 12px;
    transition: background 0.3s, color 0.3s;
    border-radius: 4px;
  }
  .navbar .nav-links a:hover,
  .navbar .nav-links a.active {
    background: #34495e;
    color: #ecf0f1;
  }
  
  
  .nav-toggle {
    display: none;
  }
  .nav-toggle-label {
    display: none;
    cursor: pointer;
  }
  .nav-toggle-label span,
  .nav-toggle-label span::before,
  .nav-toggle-label span::after {
    display: block;
    background: #ecf0f1;
    height: 3px;
    width: 25px;
    border-radius: 3px;
    position: relative;
    transition: all 0.3s ease;
  }
  .nav-toggle-label span::before,
  .nav-toggle-label span::after {
    content: '';
    position: absolute;
  }
  .nav-toggle-label span::before {
    top: -8px;
  }
  .nav-toggle-label span::after {
    top: 8px;
  }
  @media (max-width: 768px) {
    .nav-links {
      position: absolute;
      top: 100%;
      right: 0;
      background: #2c3e50;
      flex-direction: column;
      width: 200px;
      transform: translateY(-200%);
      transition: transform 0.3s ease;
    }
    .nav-links a {
      padding: 15px;
      border-bottom: 1px solid #34495e;
    }
    .nav-toggle:checked ~ .nav-links {
      transform: translateY(0);
    }
    .nav-toggle {
      display: block;
    }
    .nav-toggle-label {
      display: block;
    }
  }
  
  
  .gallery {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
    gap: 15px;
  }
  .gallery-item {
    position: relative;
    overflow: hidden;
    border-radius: 8px;
    transition: transform 0.3s ease, box-shadow 0.3s ease;
    height: 150px; 
  }
  .gallery-item:hover {
    transform: scale(1.05);
    box-shadow: 0 4px 10px rgba(0,0,0,0.2);
  }
  .img-container {
    height: 100%;
    overflow: hidden;
  }
  .img-container img {
    height: 100%;
    width: auto;
    display: block;
    margin: 0 auto;
    object-fit: cover;
    cursor: pointer;
  }
  
  
  .current-image-container img,
  .last-sent-img {
    max-width: 300px;
    max-height: 300px;
    width: auto;
    height: auto;
    margin: 0 auto;
    display: block;
  }
  
  
  .overlay {
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: rgba(0,0,0,0.5);
    opacity: 0;
    transition: opacity 0.3s ease;
    border-radius: 8px;
  }
  .img-container:hover .overlay {
    opacity: 1;
  }
  
  .crop-icon {
    position: absolute;
    top: 5px;
    left: 5px;
    width: 30px;
    height: 30px;
    display: flex;
    align-items: center;
    justify-content: center;
    color: #fff;
    background: rgba(0,0,0,0.5);
    border-radius: 50%;
    cursor: pointer;
    z-index: 10;
  }
  .crop-icon:hover {
    background: rgba(0,0,0,0.7);
  }
  
  .delete-icon {
    position: absolute;
    top: 5px;
    right: 5px;
    width: 24px;
    height: 24px;
    cursor: pointer;
    z-index: 10;
    color: #fff;
    font-size: 20px;
  }
  
  
  .favorite-icon {
    position: absolute;
    top: 5px;
    left: 50%;
    transform: translateX(-50%);
    width: 24px;
    height: 24px;
    cursor: pointer;
    z-index: 10;
    color: #fff;
    font-size: 20px;
  }
  
  .send-button {
    position: absolute;
    left: 50%;
    top: 50%;
    transform: translate(-50%, -50%);
    background: #28a745;
    color: #fff;
    border: none;
    padding: 8px 12px;
    border-radius: 4px;
    cursor: pointer;
    transition: background 0.3s ease;
  }
  .send-button:hover {
    background: #218838;
  }
  
  
  .lightbox-modal {
    display: none;
    position: fixed;
    z-index: 4000;
    left: 0;
    top: 0;
    width: 100%;
    height: 100%;
    overflow: auto;
    background-color: rgba(0,0,0,0.9);
  }
  .lightbox-content {
    margin: auto;
    display: block;
    max-width: 90%;
    max-height: 80%;
    animation: zoomIn 0.3s;
  }
  @keyframes zoomIn {
    from { transform: scale(0.8); opacity: 0; }
    to { transform: scale(1); opacity: 1; }
  }
  .lightbox-close {
    position: absolute;
    top: 20px;
    right: 35px;
    color: #f1f1f1;
    font-size: 40px;
    font-weight: bold;
    transition: 0.3s;
    cursor: pointer;
  }
  .lightbox-close:hover,
  .lightbox-close:focus {
    color: #bbb;
  }
  #lightboxCaption {
    text-align: center;
    color: #ccc;
    padding: 10px 0;
  }
  
  
  .popup {
    position: fixed;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    background: rgba(255,255,255,0.95);
    border: 2px solid #ccc;
    padding: 30px;
    box-shadow: 0 4px 10px rgba(0,0,0,0.3);
    z-index: 10000;
    text-align: center;
    font-size: 1.5em;
    display: none;
    border-radius: 8px;
    animation: popupFade 0.5s ease;
  }
  @keyframes popupFade {
    from { opacity: 0; transform: translate(-50%, -50%) scale(0.8); }
    to { opacity: 1; transform: translate(-50%, -50%) scale(1); }
  }
  
  
  .progress-container {
    width: 60%;
    margin: 20px auto;
    background: #ddd;
    border-radius: 5px;
    display: none;
  }
  .progress-bar {
    width: 0%;
    height: 30px;
    background: #28a745;
    border-radius: 5px;
    transition: width 0.4s ease;
    color: #fff;
    line-height: 30px;
    font-size: 1em;
    text-align: center;
  }
  
  
  .modal {
    display: none;
    position: fixed;
    z-index: 3000;
    left: 0;
    top: 0;
    width: 100%;
    height: 100%;
    background-color: rgba(0,0,0,0.7);
    padding: 20px;
  }
  .modal-content {
    background: #fff;
    margin: 5% auto;
    padding: 20px;
    border-radius: 8px;
    max-width: 500px;
    position: relative;
  }
  .close {
    position: absolute;
    top: 10px;
    right: 15px;
    font-size: 1.5em;
    color: #333;
    cursor: pointer;
  }
  
  
  input[type="submit"],
  button,
  .primary-btn {
    background: linear-gradient(to right, #28a745, #218838);
    border: none;
    color: #fff;
    padding: 10px 15px;
    border-radius: 4px;
    cursor: pointer;
    font-size: 1em;
    transition: background 0.3s ease;
  }
  input[type="submit"]:hover,
  button:hover,
  .primary-btn:hover {
    background: linear-gradient(to right, #218838, #1e7e34);
  }
  
  
  input[type="text"],
  input[type="password"],
  select {
    width: 100%;
    padding: 10px;
    margin: 10px 0;
    border: 1px solid #ccc;
    border-radius: 4px;
  }
  label {
    font-weight: bold;
  }
  
  
  .calendar {
    width: 100%;
    border-collapse: collapse;
    margin-top: 20px;
  }
  .calendar th,
  .calendar td {
    border: 1px solid #ddd;
    padding: 8px;
    text-align: center;
  }
  .calendar th {
    background: #f8f8f8;
  }
  .calendar .droppable.over {
    background: #dff0d8;
  }
  
  
  .footer {
    text-align: center;
    padding: 15px;
    background: #2c3e50;
    color: #bdc3c7;
    position: fixed;
    bottom: 0;
    width: 100%;
  }
  
  
  a:focus,
  button:focus,
  input:focus {
    outline: 2px solid #2980b9;
    outline-offset: 2px;
  }
```


## static/trash-icon.png

```png
�PNG

   IHDR   0   0   W��   	pHYs     ��  �IDATx��KNA��^A]�]|$�H�"}
ءG���[�`���	P��63�!H��46Z_R!���ꯙE�R� T�պf��x�Q� -�S)��h�{Cd�p��Q)b��]�����m�J�v�$}�����<�V(�R�����rg�o�.��*5	Z����W��&�d`��LG5&L�-��Uj���o��
u�V�e�AU$L�3��50��!�goઌ/�8�q_2��'���H$
�!#D2Ba���P2B$#ƿ!η/�櫘Ѳ5�*_2$}�����c�����N������sq@W?#:�R x�:�{o<�܁U(a��e����)�SU3&:b�e���X�C���E00̚��'���; �i������ݰE�7�H    IEND�B`�
```


## templates/base.html

```html
<!doctype html>
<html>
  <head>
    <title>{% block title %}InkyDocker{% endblock %}</title>
    <link rel="stylesheet" href="{{ url_for('static', filename='style.css') }}">
    
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    
    <link rel="stylesheet"
          href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css"
          crossorigin="anonymous"
          referrerpolicy="no-referrer" />
    
    <link rel="stylesheet"
          href="https://cdnjs.cloudflare.com/ajax/libs/cropperjs/1.6.2/cropper.min.css"
          crossorigin="anonymous"
          referrerpolicy="no-referrer" />
    <style>
      /* Make the site scroll fully behind the footer with a sticky footer */
      html, body {
        height: 100%;
        margin: 0;
        padding: 0;
      }
      /* FIXED: Use min-height: 100vh instead of 100% to ensure the wrapper takes up the full viewport height
       * This ensures the footer stays at the bottom even when content is minimal
       */
      .wrapper {
        min-height: 100vh; /* Use viewport height instead of percentage */
        display: flex;
        flex-direction: column;
      }
      /* FIXED: Use flex: 1 instead of flex: 1 0 auto to ensure the content area expands to fill available space
       * This pushes the footer to the bottom of the viewport
       */
      .main-content-wrapper {
        flex: 1;
      }
      .footer {
        flex-shrink: 0;
      }
    </style>
    {% block head %}{% endblock %}
  </head>
  <body>
    <div class="wrapper">
      <nav class="navbar navbar-expand-lg navbar-dark bg-dark" role="navigation" aria-label="Main Navigation">
        <div class="container-fluid">
          <a href="{{ url_for('image.upload_file') }}" class="navbar-brand">InkyDocker</a>
          <button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarNav"
            aria-controls="navbarNav" aria-expanded="false" aria-label="Toggle navigation">
            <span class="navbar-toggler-icon"></span>
          </button>
          <div class="collapse navbar-collapse" id="navbarNav">
            <ul class="navbar-nav ms-auto">
              <li class="nav-item">
                <a class="nav-link {% if request.endpoint.startswith('image.') %}active{% endif %}" href="{{ url_for('image.upload_file') }}">Gallery</a>
              </li>
              <li class="nav-item">
                <a class="nav-link {% if request.endpoint.startswith('schedule.') %}active{% endif %}" href="{{ url_for('schedule.schedule_page') }}">Schedule</a>
              </li>
              <li class="nav-item">
                <a class="nav-link {% if request.endpoint.startswith('settings.') %}active{% endif %}" href="{{ url_for('settings.settings') }}">Settings</a>
              </li>
            </ul>
          </div>
        </div>
      </nav>
      <div class="main-content-wrapper">
        {% block content %}{% endblock %}
      </div>
      <footer class="footer bg-dark text-light text-center py-3">
        <p class="mb-0">© 2025 InkyDocker | Built with AI by Me</p>
      </footer>
    </div>
    
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
    
    <script src="https://cdnjs.cloudflare.com/ajax/libs/cropperjs/1.6.2/cropper.min.js"
            crossorigin="anonymous"
            referrerpolicy="no-referrer"></script>
    {% block scripts %}{% endblock %}
  </body>
</html>
```


## templates/index.html

```html
{% extends "base.html" %}
{% block title %}Gallery - InkyDocker{% endblock %}

{% block content %}
<div class="container">
  
  <div class="card current-image-section">
    <h2 id="currentImageTitle">Current image on {{ devices[0].friendly_name if devices else 'N/A' }}</h2>
    <div class="current-image-container">
      {% if devices and devices[0].last_sent %}
        <img
          id="currentImage"
          src="{{ url_for('image.uploaded_file', filename=devices[0].last_sent) }}"
          alt="Current Image"
          class="last-sent-img small-current"
          loading="lazy"
        >
      {% else %}
        <p id="currentImagePlaceholder">No image available.</p>
      {% endif %}
    </div>
    {% if devices|length > 1 %}
      <div class="arrow-controls">
        <button id="prevDevice">&larr;</button>
        <button id="nextDevice">&rarr;</button>
      </div>
    {% endif %}
  </div>

  
  
  
  <div id="uploadPopup" style="position: fixed; top: 50%; left: 50%; transform: translate(-50%, -50%); background: rgba(0,0,0,0.8); color: #fff; padding: 15px 20px; border-radius: 5px; z-index: 1000; display: none;">
    <div class="spinner"></div> Processing...
  </div>

  
  <div class="main-content">
    
    <div class="left-panel">
      <div class="card device-section">
        <h2>Select eInk Display</h2>
        {% if devices %}
          <div class="device-options">
            {% for device in devices %}
              <label class="device-option">
                <input
                  type="radio"
                  name="device"
                  value="{{ device.address }}"
                  data-index="{{ loop.index0 }}"
                  data-friendly="{{ device.friendly_name }}"
                  data-resolution="{{ device.resolution }}"
                  {% if loop.first %}checked{% endif %}
                >
                {{ device.friendly_name }}
              </label>
            {% endfor %}
          </div>
        {% else %}
          <p>No devices configured. Go to <a href="{{ url_for('settings.settings') }}">Settings</a>.</p>
        {% endif %}
      </div>

      <div class="card upload-section">
        <h2>Upload Images</h2>
        <form id="uploadForm" class="upload-form" method="post" enctype="multipart/form-data" action="{{ url_for('image.upload_file') }}">
          <input type="file" name="file" multiple id="fileInput" required>
          <br>
          <input type="submit" value="Upload">
          <div class="progress-container" id="progressContainer" style="display: none;">
            <div class="progress-bar" id="progressBar">0%</div>
          </div>
          <div id="uploadStatus"></div>
        </form>
      </div>
      
      
    </div>

    
    <div class="gallery-section">
      <h2>Gallery</h2>
      <input type="text" id="gallerySearch" placeholder="Search images by tags..." style="width:100%; padding:10px; margin-bottom:20px;">
      <div id="searchSpinner" style="display:none;">Loading...</div>
      <div class="gallery" id="gallery">
        {% for image in images %}
          <div class="gallery-item">
            <div class="img-container">
              <img src="{{ url_for('image.uploaded_file', filename=image) }}" alt="{{ image }}" data-filename="{{ image }}" loading="lazy">
              <div class="overlay">
                
                <div class="favorite-icon" title="Favorite" data-image="{{ image }}">
                  <i class="fa fa-heart"></i>
                </div>
                
                <button class="send-button" data-image="{{ image }}">Send</button>
                
                <button class="info-button" data-image="{{ image }}">Info</button>
                
                <div class="delete-icon" title="Delete" data-image="{{ image }}">
                  <i class="fa fa-trash"></i>
                </div>
              </div>
            </div>
          </div>
        {% endfor %}
      </div>
    </div>
  </div>
  
  <div style="height: 100px;"></div>
</div>


<div id="infoModal" class="modal" style="display:none;">
  <div class="modal-content" style="max-width:800px; margin:auto; position:relative; padding:20px;">
    <span class="close" onclick="closeInfoModal()" style="position:absolute; top:10px; right:15px; cursor:pointer; font-size:1.5em;">&times;</span>
    <h2>Image Info</h2>
    <div style="text-align:center; margin-bottom:20px;">
      <img id="infoImagePreview" src="" alt="Info Preview" style="max-width:300px;">
      <div style="margin-top:10px;">
        <button type="button" onclick="openCropModal()">Crop Image</button>
      </div>
    </div>
    <div style="display:flex; gap:20px;">
      
      <div style="flex:1;" id="infoLeftColumn">
        <p><strong>Filename:</strong> <span id="infoFilename">N/A</span></p>
        <p><strong>Resolution:</strong> <span id="infoResolution">N/A</span></p>
        <p><strong>Filesize:</strong> <span id="infoFilesize">N/A</span></p>
      </div>
      
      <div style="flex:1;">
        <div style="margin-bottom:10px;">
          <label><strong>Tags:</strong></label>
          <div id="tagContainer" style="margin-top:5px; margin-bottom:10px;"></div>
          <div style="display:flex;">
            <input type="text" id="newTagInput" style="flex-grow:1;" placeholder="Add a new tag...">
            <button type="button" onclick="addTag()" style="margin-left:5px;">Add</button>
          </div>
          <input type="hidden" id="infoTags">
        </div>
        <div style="margin-bottom:10px;">
          <label><strong>Favorite:</strong></label>
          <input type="checkbox" id="infoFavorite">
        </div>
        <div id="infoStatus" style="color: green; margin-bottom:10px;"></div>
        <button onclick="saveInfoEdits()">Save</button>
        <button onclick="runOpenClip()">Re-run Tagging</button>
      </div>
    </div>
  </div>
</div>


<div id="lightboxModal" class="modal lightbox-modal" style="display:none;">
  <span class="lightbox-close" onclick="closeLightbox()">&times;</span>
  <img class="lightbox-content" id="lightboxImage" alt="Enlarged Image">
  <div id="lightboxCaption"></div>
</div>


<div id="cropModal" class="modal" style="display:none;">
  <div class="modal-content">
    <span class="close" onclick="closeCropModal()" style="cursor:pointer; font-size:1.5em;">&times;</span>
    <h2>Crop Image</h2>
    <div id="cropContainer" style="max-width:100%; max-height:80vh;">
      <img id="cropImage" src="" alt="Crop Image" style="width:100%;">
    </div>
    <div style="margin-top:10px;">
      <button type="button" onclick="saveCropData()">Save Crop</button>
      <button type="button" onclick="closeCropModal()">Cancel</button>
    </div>
  </div>
</div>
{% endblock %}

{% block scripts %}
<script>
document.addEventListener("DOMContentLoaded", function() {
  // Inject dynamic CSS for info button, favorite icon, and tag boxes
  const styleTag = document.createElement('style');
  styleTag.innerHTML = `
    .info-button {
      position: absolute;
      left: 50%;
      bottom: 10px;
      transform: translateX(-50%);
      background: #17a2b8;
      color: #fff;
      border: none;
      padding: 8px 12px;
      border-radius: 4px;
      cursor: pointer;
      transition: background 0.3s ease;
      font-size: 0.9em;
    }
    .info-button:hover {
      background: #138496;
    }
    .favorite-icon i {
      font-size: 1.5em;
      color: #ccc;
      transition: color 0.3s;
    }
    .favorite-icon.favorited i {
      color: red;
    }
    .tag-box {
      display: inline-block;
      background-color: #e9ecef;
      border-radius: 4px;
      padding: 5px 10px;
      margin: 3px;
      font-size: 0.9em;
    }
    .tag-remove {
      margin-left: 5px;
      cursor: pointer;
      font-weight: bold;
      color: #dc3545;
    }
    .tag-remove:hover {
      color: #bd2130;
    }
  `;
  document.head.appendChild(styleTag);
});

/* Lightbox functions */
function openLightbox(src, alt) {
  const lightboxModal = document.getElementById('lightboxModal');
  const lightboxImage = document.getElementById('lightboxImage');
  const lightboxCaption = document.getElementById('lightboxCaption');
  lightboxModal.style.display = 'block';
  lightboxImage.src = src;
  lightboxCaption.innerText = alt;
}
function closeLightbox() {
  document.getElementById('lightboxModal').style.display = 'none';
}

/* Debounce helper */
function debounce(func, wait) {
  let timeout;
  return function(...args) {
    clearTimeout(timeout);
    timeout = setTimeout(() => func.apply(this, args), wait);
  };
}

/* Gallery search */
const searchInput = document.getElementById('gallerySearch');
const searchSpinner = document.getElementById('searchSpinner');
const gallery = document.getElementById('gallery');

const performSearch = debounce(function() {
  const query = searchInput.value.trim();
  if (!query) {
    searchSpinner.style.display = 'none';
    location.reload();
    return;
  }
  searchSpinner.style.display = 'block';
  fetch(`/api/search_images?q=${encodeURIComponent(query)}`)
    .then(response => response.json())
    .then(data => {
      searchSpinner.style.display = 'none';
      gallery.innerHTML = "";
      if (data.status === "success" && data.results.ids) {
        if (data.results.ids.length === 0) {
          gallery.innerHTML = "<p>No matching images found.</p>";
        } else {
          data.results.ids.forEach((id) => {
            const imageUrl = `/images/${encodeURIComponent(id)}`;
            const item = document.createElement('div');
            item.className = 'gallery-item';
            item.innerHTML = `
              <div class="img-container">
                <img src="${imageUrl}" alt="${id}" data-filename="${id}" loading="lazy">
                <div class="overlay">
                  <div class="favorite-icon" title="Favorite" data-image="${id}">
                    <i class="fa fa-heart"></i>
                  </div>
                  <button class="send-button" data-image="${id}">Send</button>
                  <button class="info-button" data-image="${id}">Info</button>
                  <div class="delete-icon" title="Delete" data-image="${id}">
                    <i class="fa fa-trash"></i>
                  </div>
                </div>
              </div>
            `;
            gallery.appendChild(item);
          });
        }
      } else {
        gallery.innerHTML = "<p>No matching images found.</p>";
      }
    })
    .catch(err => {
      searchSpinner.style.display = 'none';
      console.error("Search error:", err);
    });
}, 500);

if (searchInput) {
  searchInput.addEventListener('input', performSearch);
}

/* Upload form */
const uploadForm = document.getElementById('uploadForm');
uploadForm.addEventListener('submit', function(e) {
  e.preventDefault();
  const fileInput = document.getElementById('fileInput');
  if (!fileInput.files.length) return;
  
  const formData = new FormData();
  for (let i = 0; i < fileInput.files.length; i++) {
    formData.append('file', fileInput.files[i]);
  }
  
  const selectedDevice = document.querySelector('input[name="device"]:checked');
  const deviceFriendly = selectedDevice ? selectedDevice.getAttribute('data-friendly') : "unknown display";
  
  const xhr = new XMLHttpRequest();
  xhr.open('POST', uploadForm.action, true);

  xhr.upload.addEventListener("progress", function(e) {
    if (e.lengthComputable) {
      const percentComplete = (e.loaded / e.total) * 100;
      const progressBar = document.getElementById('progressBar');
      progressBar.style.width = percentComplete + '%';
      progressBar.textContent = Math.round(percentComplete) + '%';
      document.getElementById('progressContainer').style.display = 'block';
      
      const popup = document.getElementById('uploadPopup');
      popup.style.display = 'block';
      popup.innerHTML = `<div class="spinner"></div> Uploading image to ${deviceFriendly}... ${Math.round(percentComplete)}%`;
    }
  });

  xhr.onload = function() {
    const popup = document.getElementById('uploadPopup');
    if (xhr.status === 200) {
      popup.innerHTML = `<div class="spinner"></div> Image uploaded successfully!`;
    } else {
      popup.innerHTML = `<div class="spinner"></div> Error uploading image.`;
    }
    setTimeout(() => {
      popup.style.display = 'none';
      location.reload();
    }, 1500);
  };

  xhr.onerror = function() {
    const popup = document.getElementById('uploadPopup');
    popup.innerHTML = `<div class="spinner"></div> Error uploading image.`;
    setTimeout(() => {
      popup.style.display = 'none';
    }, 1500);
  };

  xhr.send(formData);
});

/* Send image */
document.addEventListener('click', function(e) {
  if (e.target && e.target.classList.contains('send-button')) {
    e.stopPropagation();
    const imageFilename = e.target.getAttribute('data-image');
    const selectedDevice = document.querySelector('input[name="device"]:checked');
    if (!selectedDevice) return;
    
    const deviceFriendly = selectedDevice.getAttribute('data-friendly');
    const formData = new FormData();
    formData.append("device", selectedDevice.value);

    const baseUrl = "{{ url_for('image.send_image', filename='') }}";
    const finalUrl = baseUrl + encodeURIComponent(imageFilename);

    const xhr = new XMLHttpRequest();
    xhr.open('POST', finalUrl, true);

    xhr.upload.addEventListener("progress", function(ev) {
      if (ev.lengthComputable) {
        const percentComplete = (ev.loaded / ev.total) * 100;
        const popup = document.getElementById('uploadPopup');
        popup.style.display = 'block';
        popup.innerHTML = `<div class="spinner"></div> Sending image to ${deviceFriendly}... ${Math.round(percentComplete)}%`;
      }
    });

    xhr.onload = function() {
      const popup = document.getElementById('uploadPopup');
      if (xhr.status === 200) {
        popup.innerHTML = `<div class="spinner"></div> Image sent successfully!`;
      } else {
        popup.innerHTML = `<div class="spinner"></div> Error sending image.`;
      }
      setTimeout(() => {
        popup.style.display = 'none';
        location.reload();
      }, 1500);
    };

    xhr.onerror = function() {
      const popup = document.getElementById('uploadPopup');
      popup.innerHTML = `<div class="spinner"></div> Error sending image.`;
      setTimeout(() => {
        popup.style.display = 'none';
      }, 1500);
    };

    xhr.send(formData);
  }
});

/* Delete image */
document.addEventListener('click', function(e) {
  if (e.target && e.target.closest('.delete-icon')) {
    e.stopPropagation();
    const imageFilename = e.target.closest('.delete-icon').getAttribute('data-image');
    
    const deleteBaseUrl = "/delete_image/";
    const deleteUrl = deleteBaseUrl + encodeURIComponent(imageFilename);

    fetch(deleteUrl, { method: 'POST' })
      .then(response => response.json())
      .then(data => {
        if (data.status === "success") {
          location.reload();
        } else {
          console.error("Error deleting image:", data.message);
        }
      })
      .catch(error => {
        console.error("Error deleting image:", error);
      });
  }
});

/* Favorite toggle */
document.addEventListener('click', function(e) {
  if (e.target && e.target.closest('.favorite-icon')) {
    e.stopPropagation();
    const favIcon = e.target.closest('.favorite-icon');
    const imageFilename = favIcon.getAttribute('data-image');
    favIcon.classList.toggle('favorited');
    const isFavorited = favIcon.classList.contains('favorited');
    fetch("/api/update_image_metadata", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        filename: imageFilename,
        tags: [],  // do not modify tags in favorite toggle
        favorite: isFavorited
      })
    })
      .then(resp => resp.json())
      .then(data => {
        if (data.status !== "success") {
          console.error("Error updating favorite:", data.message);
        }
      })
      .catch(err => {
        console.error("Error updating favorite:", err);
      });
  }
});

/* Info Modal Logic */
let currentInfoFilename = null;

document.addEventListener('click', function(e) {
  if (e.target && e.target.classList.contains('info-button')) {
    e.stopPropagation();
    const filename = e.target.getAttribute('data-image');
    currentInfoFilename = filename;
    openInfoModal(filename);
  }
});

// Tag management functions
let currentTags = [];

function renderTags() {
  const tagContainer = document.getElementById('tagContainer');
  tagContainer.innerHTML = '';
  
  currentTags.forEach((tag, index) => {
    const tagElement = document.createElement('span');
    tagElement.className = 'tag-box';
    tagElement.innerHTML = `${tag} <span class="tag-remove" onclick="removeTag(${index})">×</span>`;
    tagContainer.appendChild(tagElement);
  });
  
  // Update the hidden input with comma-separated tags
  document.getElementById('infoTags').value = currentTags.join(', ');
}

function addTag() {
  const newTagInput = document.getElementById('newTagInput');
  const tag = newTagInput.value.trim();
  
  if (tag && !currentTags.includes(tag)) {
    currentTags.push(tag);
    renderTags();
    newTagInput.value = '';
  }
}

function removeTag(index) {
  currentTags.splice(index, 1);
  renderTags();
}

// Add event listener for Enter key on the new tag input
document.addEventListener('DOMContentLoaded', function() {
  const newTagInput = document.getElementById('newTagInput');
  if (newTagInput) {
    newTagInput.addEventListener('keypress', function(e) {
      if (e.key === 'Enter') {
        e.preventDefault();
        addTag();
      }
    });
  }
});

function openInfoModal(filename) {
  const imgUrl = `/images/${encodeURIComponent(filename)}?size=info`;
  fetch(`/api/get_image_metadata?filename=${encodeURIComponent(filename)}`)
    .then(resp => resp.json())
    .then(data => {
      if (data.status === "success") {
        document.getElementById('infoImagePreview').src = imgUrl;
        document.getElementById('infoFilename').textContent = filename;
        document.getElementById('infoResolution').textContent = data.resolution || "N/A";
        document.getElementById('infoFilesize').textContent = data.filesize || "N/A";
        
        // Set up tags
        currentTags = data.tags || [];
        renderTags();
        
        document.getElementById('infoFavorite').checked = data.favorite || false;
        document.getElementById('infoStatus').textContent = "";
        document.getElementById('infoModal').style.display = 'block';
      } else {
        document.getElementById('infoStatus').textContent = "Error: " + data.message;
        document.getElementById('infoModal').style.display = 'block';
      }
    })
    .catch(err => {
      console.error("Error fetching metadata:", err);
      document.getElementById('infoStatus').textContent = "Failed to fetch metadata. Check console.";
      document.getElementById('infoModal').style.display = 'block';
    });
}

function closeInfoModal() {
  document.getElementById('infoModal').style.display = 'none';
  currentInfoFilename = null;
}

function saveInfoEdits() {
  if (!currentInfoFilename) return;
  fetch("/api/update_image_metadata", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      filename: currentInfoFilename,
      tags: currentTags,
      favorite: document.getElementById('infoFavorite').checked
    })
  })
    .then(resp => resp.json())
    .then(data => {
      if (data.status === "success") {
        document.getElementById('infoStatus').textContent = "Metadata updated successfully!";
        setTimeout(() => { closeInfoModal(); }, 1500);
      } else {
        document.getElementById('infoStatus').textContent = "Error updating metadata: " + data.message;
      }
    })
    .catch(err => {
      console.error("Error updating metadata:", err);
      document.getElementById('infoStatus').textContent = "Failed to update metadata. Check console.";
    });
}

function runOpenClip() {
  if (!currentInfoFilename) return;
  fetch(`/api/reembed_image?filename=${encodeURIComponent(currentInfoFilename)}`)
    .then(resp => resp.json())
    .then(data => {
      if (data.status === "success") {
        currentTags = data.tags || [];
        renderTags();
        document.getElementById('infoStatus').textContent = "Re-ran tagging successfully!";
      } else {
        document.getElementById('infoStatus').textContent = "Error re-running tagging: " + data.message;
      }
    })
    .catch(err => {
      console.error("Error re-running tagging:", err);
      document.getElementById('infoStatus').textContent = "Failed to re-run tagging. Check console.";
    });
}

// Crop Modal Functions
let cropperInstance = null;

function openCropModal() {
  if (!currentInfoFilename) return;
  
  const cropModal = document.getElementById('cropModal');
  const cropImage = document.getElementById('cropImage');
  
  // Set the image source to the current image
  cropImage.src = `/images/${encodeURIComponent(currentInfoFilename)}`;
  
  // Show the modal
  cropModal.style.display = 'block';
  
  // Initialize Cropper.js after the image is loaded
  cropImage.onload = function() {
    if (cropperInstance) {
      cropperInstance.destroy();
    }
    
    // Import Cropper.js dynamically if needed
    if (typeof Cropper === 'undefined') {
      const script = document.createElement('script');
      script.src = 'https://cdnjs.cloudflare.com/ajax/libs/cropperjs/1.5.12/cropper.min.js';
      document.head.appendChild(script);
      
      const link = document.createElement('link');
      link.rel = 'stylesheet';
      link.href = 'https://cdnjs.cloudflare.com/ajax/libs/cropperjs/1.5.12/cropper.min.css';
      document.head.appendChild(link);
      
      script.onload = initCropper;
    } else {
      initCropper();
    }
  };
}

function initCropper() {
  const cropImage = document.getElementById('cropImage');
  
  // Get the selected device's resolution
  const selectedDevice = document.querySelector('input[name="device"]:checked');
  let aspectRatio = NaN; // Default to free aspect ratio
  let targetWidth = 0;
  let targetHeight = 0;
  let isPortrait = false;
  
  if (selectedDevice) {
    const resolution = selectedDevice.getAttribute('data-resolution');
    if (resolution) {
      const parts = resolution.split('x');
      if (parts.length === 2) {
        targetWidth = parseInt(parts[0], 10);
        targetHeight = parseInt(parts[1], 10);
        
        // Check if the device is in portrait orientation
        isPortrait = selectedDevice.parentNode.textContent.trim().toLowerCase().includes('portrait');
        
        if (isPortrait) {
          // For portrait orientation, swap width and height for aspect ratio
          aspectRatio = targetHeight / targetWidth;
        } else {
          aspectRatio = targetWidth / targetHeight;
        }
      }
    }
  }
  
  // Initialize cropper with the calculated aspect ratio
  cropperInstance = new Cropper(cropImage, {
    aspectRatio: aspectRatio, // Use the calculated aspect ratio
    viewMode: 1,
    autoCropArea: 1, // Start with maximum possible area
    responsive: true,
    restore: true,
    guides: true,
    center: true,
    highlight: true,
    cropBoxMovable: true,
    cropBoxResizable: true,
    toggleDragModeOnDblclick: true,
    ready: function() {
      // This function runs when the cropper is fully initialized
      if (aspectRatio && cropperInstance) {
        // Get the image dimensions
        const imageData = cropperInstance.getImageData();
        const imageWidth = imageData.naturalWidth;
        const imageHeight = imageData.naturalHeight;
        
        // Calculate the optimal crop box dimensions to cover as much of the image as possible
        // while maintaining the target aspect ratio
        let cropBoxWidth, cropBoxHeight;
        
        const imageRatio = imageWidth / imageHeight;
        
        if (aspectRatio > imageRatio) {
          // If target aspect ratio is wider than the image, use full width
          cropBoxWidth = imageWidth;
          cropBoxHeight = cropBoxWidth / aspectRatio;
        } else {
          // If target aspect ratio is taller than the image, use full height
          cropBoxHeight = imageHeight;
          cropBoxWidth = cropBoxHeight * aspectRatio;
        }
        
        // Calculate the position to center the crop box
        const left = (imageWidth - cropBoxWidth) / 2;
        const top = (imageHeight - cropBoxHeight) / 2;
        
        // Set the crop box
        cropperInstance.setCropBoxData({
          left: left,
          top: top,
          width: cropBoxWidth,
          height: cropBoxHeight
        });
      }
    }
  });
}

function closeCropModal() {
  const cropModal = document.getElementById('cropModal');
  cropModal.style.display = 'none';
  
  if (cropperInstance) {
    cropperInstance.destroy();
    cropperInstance = null;
  }
}

function saveCropData() {
  if (!cropperInstance || !currentInfoFilename) return;
  
  const cropData = cropperInstance.getData();
  const selectedDevice = document.querySelector('input[name="device"]:checked');
  const deviceAddress = selectedDevice ? selectedDevice.value : null;
  
  fetch(`/save_crop_info/${encodeURIComponent(currentInfoFilename)}`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({
      x: cropData.x,
      y: cropData.y,
      width: cropData.width,
      height: cropData.height,
      device: deviceAddress
    })
  })
  .then(response => response.json())
  .then(data => {
    if (data.status === 'success') {
      document.getElementById('infoStatus').textContent = 'Crop data saved successfully!';
      closeCropModal();
    } else {
      document.getElementById('infoStatus').textContent = 'Error saving crop data: ' + data.message;
    }
  })
  .catch(error => {
    console.error('Error saving crop data:', error);
    document.getElementById('infoStatus').textContent = 'Error saving crop data. Check console.';
  });
}

const prevButton = document.getElementById('prevDevice');
const nextButton = document.getElementById('nextDevice');
// Define devices array using template data
const devices = [
  {% for device in devices %}
    {
      "friendly_name": "{{ device.friendly_name|e }}",
      "address": "{{ device.address|e }}",
      "last_sent": "{{ device.last_sent|e if device.last_sent is defined else '' }}"
    }{% if not loop.last %},{% endif %}
  {% endfor %}
];
let currentDeviceIndex = 0;

function updateCurrentImageDisplay() {
  const device = devices[currentDeviceIndex];
  const titleEl = document.getElementById('currentImageTitle');
  const imageEl = document.getElementById('currentImage');
  const placeholderEl = document.getElementById('currentImagePlaceholder');
  
  titleEl.textContent = "Current image on " + device.friendly_name;
  if (device.last_sent) {
    if (placeholderEl) {
      placeholderEl.style.display = 'none';
    }
    if (imageEl) {
      imageEl.src = "{{ url_for('image.uploaded_file', filename='') }}" + device.last_sent;
      imageEl.style.display = 'block';
    }
  } else {
    if (imageEl) {
      imageEl.style.display = 'none';
    }
    if (placeholderEl) {
      placeholderEl.style.display = 'block';
    }
  }
}

if (prevButton && nextButton) {
  prevButton.addEventListener('click', function() {
    currentDeviceIndex = (currentDeviceIndex - 1 + devices.length) % devices.length;
    updateCurrentImageDisplay();
  });
  nextButton.addEventListener('click', function() {
    currentDeviceIndex = (currentDeviceIndex + 1) % devices.length;
    updateCurrentImageDisplay();
  });
}

if (devices.length > 0) {
  updateCurrentImageDisplay();
}

// Bulk tagging moved to settings page
</script>
{% endblock %}
```


## templates/schedule.html

```html
{% extends "base.html" %}
{% block title %}Schedule - InkyDocker{% endblock %}
{% block head %}
  
  <link href="https://cdn.jsdelivr.net/npm/fullcalendar@6.1.8/main.min.css" rel="stylesheet">
  
  <script src="https://cdn.jsdelivr.net/npm/fullcalendar@6.1.8/index.global.min.js" defer></script>
  <style>
    #calendar {
      max-width: 1000px;
      margin: 40px auto;
      margin-bottom: 100px; /* Add more space at the bottom */
    }
    
    /* Style for the event content to show thumbnails */
    .event-thumbnail {
      width: 100%;
      height: 40px;
      object-fit: cover;
      border-radius: 3px;
      margin-bottom: 2px;
    }
    
    /* Search bar styles */
    .search-container {
      margin-bottom: 15px;
    }
    
    #imageSearch {
      width: 100%;
      padding: 8px;
      border: 1px solid #ddd;
      border-radius: 4px;
      margin-bottom: 10px;
    }
    
    /* Improve gallery layout */
    .gallery {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
      gap: 10px;
      max-height: 500px;
      overflow-y: auto;
    }
    
    .gallery-item {
      height: 150px !important;
      position: relative;
      border: 1px solid #ddd;
      border-radius: 4px;
      overflow: hidden;
    }
    
    .gallery-item img {
      width: 100%;
      height: 100%;
      object-fit: cover;
      cursor: pointer;
      transition: transform 0.2s;
    }
    
    .gallery-item img:hover {
      transform: scale(1.05);
    }
    
    .img-container {
      height: 100%;
    }
    
    /* Tags display */
    .image-tags {
      position: absolute;
      bottom: 0;
      left: 0;
      right: 0;
      background: rgba(0,0,0,0.6);
      color: white;
      padding: 3px;
      font-size: 10px;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    /* Modal styles for event creation and image gallery (existing) */
    .modal {
      display: none;
      position: fixed;
      z-index: 10000;
      left: 0;
      top: 0;
      width: 100%;
      height: 100%;
      overflow: auto;
      background-color: rgba(0,0,0,0.7);
      padding: 20px;
    }
    .modal-content {
      background: #fff;
      margin: 10% auto;
      padding: 20px;
      border-radius: 8px;
      max-width: 500px;
      position: relative;
    }
    .close {
      position: absolute;
      top: 10px;
      right: 15px;
      font-size: 1.5em;
      cursor: pointer;
    }
    /* Deletion modal for recurring events */
    #deleteModal .modal-content {
      max-width: 400px;
      text-align: center;
    }
    #deleteModal button {
      margin: 5px;
    }
    
    /* Styling for recurring events */
    .recurring-event {
      border-left: 4px dashed #fff !important;  /* Dashed border to indicate recurring */
      border-right: 4px dashed #fff !important;
    }
    
    .recurring-event:before {
      content: "↻";  /* Recurrence symbol */
      position: absolute;
      top: 2px;
      left: 2px;
      font-weight: bold;
      color: white;
      background-color: rgba(0, 0, 0, 0.5);
      border-radius: 50%;
      width: 16px;
      height: 16px;
      line-height: 16px;
      text-align: center;
      z-index: 100;
    }
  </style>
{% endblock %}
{% block content %}
<div class="container">
  <header class="page-header">
    <h1>Schedule Images</h1>
    <p>Manage your scheduled image updates with our interactive calendar.</p>
  </header>
  <div id="calendar"></div>
  
</div>


<div id="eventModal" class="modal">
  <div class="modal-content">
    <span class="close" onclick="closeEventModal()">&times;</span>
    <h2 id="eventModalTitle">Add Scheduled Event</h2>
    <form id="eventForm">
      <input type="hidden" id="editingEventId" value="">
      <div>
        <label for="eventDate">Date &amp; Time:</label>
        <input type="datetime-local" id="eventDate" name="eventDate" required>
      </div>
      <div>
        <label>Select eInk Display:</label>
        {% if devices %}
          {% for device in devices %}
            <label>
              <input type="radio" name="device" value="{{ device.address }}" {% if loop.first %}checked{% endif %}>
              {{ device.friendly_name }}
            </label>
          {% endfor %}
        {% else %}
          <p>No devices configured. Please add devices in <a href="{{ url_for('settings.settings') }}">Settings</a>.</p>
        {% endif %}
      </div>
      <div>
        <label for="recurrence">Recurrence:</label>
        <select id="recurrence" name="recurrence">
          <option value="none">None</option>
          <option value="daily">Daily</option>
          <option value="weekly">Weekly</option>
          <option value="monthly">Same date next month</option>
        </select>
      </div>
      <div>
        <label>Choose Image:</label>
        <button type="button" onclick="openImageGallery()">Select Image</button>
        <input type="hidden" id="selectedImage" name="selectedImage">
        <span id="selectedImageName"></span>
      </div>
      <div style="margin-top:10px;">
        <input type="submit" id="eventSubmitButton" value="Save Event">
      </div>
    </form>
  </div>
</div>


<div id="imageGalleryModal" class="modal">
  <div class="modal-content" style="max-width:800px;">
    <span class="close" onclick="closeImageGallery()">&times;</span>
    <h2>Select an Image</h2>
    
    
    <div class="search-container">
      <input type="text" id="imageSearch" placeholder="Search by tags...">
    </div>
    
    <div class="gallery" id="galleryModal">
      {% for image in images %}
        <div class="gallery-item" data-tags="{{ image_tags.get(image, '') }}">
          <div class="img-container">
            <img src="{{ url_for('image.thumbnail', filename=image) }}" alt="{{ image }}" data-filename="{{ image }}" onclick="selectImage('{{ image }}', this.src)">
            {% if image_tags.get(image) %}
              <div class="image-tags">{{ image_tags.get(image) }}</div>
            {% endif %}
          </div>
        </div>
      {% endfor %}
    </div>
  </div>
</div>


<div id="deleteModal" class="modal">
  <div class="modal-content">
    <span class="close" onclick="closeDeleteModal()">&times;</span>
    <h3>Delete Recurring Event</h3>
    <p>Delete this occurrence or the entire series?</p>
    <button id="deleteOccurrenceBtn" class="btn btn-danger">Delete this occurrence</button>
    <button id="deleteSeriesBtn" class="btn btn-danger">Delete entire series</button>
    <button onclick="closeDeleteModal()" class="btn btn-secondary">Cancel</button>
  </div>
</div>
{% endblock %}
{% block scripts %}
  <script>
    var currentDeleteEventId = null; // store event id for deletion modal

    // FIXED: Store the calendar instance globally for easy access
    // This allows us to call calendar.refetchEvents() from anywhere in the code
    var calendar;
    
    document.addEventListener('DOMContentLoaded', function() {
      var calendarEl = document.getElementById('calendar');
      if (!calendarEl) return;
      if (typeof FullCalendar === 'undefined') {
        console.error("FullCalendar is not defined");
        return;
      }
      // Initialize the global calendar variable
      calendar = new FullCalendar.Calendar(calendarEl, {
        initialView: 'timeGridWeek',
        firstDay: 1,
        nowIndicator: true,
        editable: true,
        headerToolbar: {
          left: 'prev,next today refresh',
          center: 'title',
          right: 'timeGridWeek,timeGridDay'
        },
        events: '/schedule/events',
        customButtons: {
          refresh: {
            text: '↻',
            click: function() {
              // Force a full refresh of events from the server
              calendar.refetchEvents();
            }
          }
        },
        eventDrop: function(info) {
          var newDate = info.event.start;
          
          // FIXED: Convert to ISO string for consistent timezone handling
          // This ensures the datetime is properly formatted when sent to the server
          var isoString = newDate.toISOString();
          console.log("ISO date string:", isoString);
          
          // Get timezone offset in minutes
          // getTimezoneOffset() returns positive minutes for times behind UTC
          // and negative minutes for times ahead of UTC
          var timezoneOffset = newDate.getTimezoneOffset();
          console.log("Timezone offset:", timezoneOffset, "minutes");
          
          fetch("/schedule/update", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              event_id: info.event.id,
              datetime: isoString,
              timezone_offset: timezoneOffset
            })
          })
          .then(response => response.json())
          .then(data => {
            if(data.status !== "success"){
              alert("Error updating event: " + data.message);
              // Revert the drag if there was an error
              info.revert();
            } else {
              console.log("Event successfully updated:", info.event.id);
              // FIXED: Use the global calendar instance to refresh events
              // This ensures the calendar is immediately updated after an event is dragged
              if (calendar) {
                calendar.refetchEvents();
              }
            }
          })
          .catch(err => {
            console.error("Error updating event:", err);
            // Revert the drag if there was an error
            info.revert();
          });
        },
        eventDidMount: function(info) {
          // Add special styling for recurring events
          if (info.event.extendedProps.isRecurring) {
            info.el.classList.add('recurring-event');
          }
          
          // Add delete button with improved visibility
          var deleteEl = document.createElement('span');
          deleteEl.innerHTML = '&times;';
          deleteEl.style.position = 'absolute';
          deleteEl.style.top = '2px';
          deleteEl.style.right = '2px';
          deleteEl.style.color = 'white';  // White text for better contrast
          deleteEl.style.backgroundColor = 'rgba(0, 0, 0, 0.6)';  // Semi-transparent black background
          deleteEl.style.borderRadius = '50%';  // Circular background
          deleteEl.style.width = '16px';
          deleteEl.style.height = '16px';
          deleteEl.style.lineHeight = '14px';
          deleteEl.style.textAlign = 'center';
          deleteEl.style.cursor = 'pointer';
          deleteEl.style.fontWeight = 'bold';
          deleteEl.style.zIndex = '100';
          deleteEl.style.border = '1px solid white';  // White border for additional contrast
          info.el.appendChild(deleteEl);
          
          // Add thumbnail to event
          if (info.event.extendedProps.thumbnail) {
            var thumbnailEl = document.createElement('img');
            thumbnailEl.src = info.event.extendedProps.thumbnail;
            thumbnailEl.className = 'event-thumbnail';
            thumbnailEl.alt = info.event.title;
            thumbnailEl.style.cursor = 'pointer';
            
            // Add click event to open the event for editing
            thumbnailEl.addEventListener('click', function(e) {
              e.stopPropagation();
              // Get event data
              var eventData = {
                id: info.event.id,
                title: info.event.title,
                start: info.event.start,
                device: info.event.extendedProps.device,
                filename: info.event.extendedProps.filename,
                recurrence: info.event.extendedProps.recurrence
              };
              
              // Populate the event form with this data - use local timezone
              var startDate = eventData.start;
              var year = startDate.getFullYear();
              var month = String(startDate.getMonth() + 1).padStart(2, '0');
              var day = String(startDate.getDate()).padStart(2, '0');
              var hours = String(startDate.getHours()).padStart(2, '0');
              var minutes = String(startDate.getMinutes()).padStart(2, '0');
              
              var localDateStr = `${year}-${month}-${day}T${hours}:${minutes}`;
              document.getElementById('eventDate').value = localDateStr;
              console.log("Event edit local time:", localDateStr);
              
              // Select the correct device radio button
              var deviceRadios = document.querySelectorAll('input[name="device"]');
              for (var i = 0; i < deviceRadios.length; i++) {
                if (deviceRadios[i].value === eventData.device) {
                  deviceRadios[i].checked = true;
                  break;
                }
              }
              
              // Set the recurrence dropdown
              document.getElementById('recurrence').value = eventData.recurrence || 'none';
              
              // Set the selected image
              document.getElementById('selectedImage').value = eventData.filename;
              document.getElementById('selectedImageName').textContent = eventData.filename;
              
              // Set the editing event ID
              document.getElementById('editingEventId').value = eventData.id;
              
              // Update modal title and button text
              document.getElementById('eventModalTitle').textContent = 'Edit Scheduled Event';
              document.getElementById('eventSubmitButton').value = 'Update Event';
              
              // Open the event modal
              openEventModal();
            });
            
            // Find the event title element and insert the thumbnail before it
            var titleEl = info.el.querySelector('.fc-event-title');
            if (titleEl && titleEl.parentNode) {
              titleEl.parentNode.insertBefore(thumbnailEl, titleEl);
              
              // Add device name to title
              if (info.event.extendedProps.deviceName) {
                titleEl.textContent = info.event.title + ' on ' + info.event.extendedProps.deviceName;
              }
            }
          }
          
          // Add click handler for delete button
          deleteEl.addEventListener('click', function(e) {
            e.stopPropagation();
            // If recurring, show custom deletion modal; otherwise, delete directly.
            if(info.event.extendedProps.recurrence && info.event.extendedProps.recurrence.toLowerCase() !== "none"){
              currentDeleteEventId = info.event.id;
              openDeleteModal();
            } else {
              // Directly delete non-recurring event without popup.
              fetch("/schedule/remove/" + info.event.id, { method: "POST" })
              .then(response => response.json())
              .then(data => {
                if(data.status === "success"){
                  info.event.remove();
                }
              })
              .catch(err => {
                console.error("Error deleting event:", err);
              });
            }
          });
        },
        dateClick: function(info) {
          // Create a date object from the clicked date
          var dtLocal = new Date(info.date);
          
          // Format the date in local timezone format (YYYY-MM-DDTHH:MM)
          var year = dtLocal.getFullYear();
          var month = String(dtLocal.getMonth() + 1).padStart(2, '0');
          var day = String(dtLocal.getDate()).padStart(2, '0');
          var hours = String(dtLocal.getHours()).padStart(2, '0');
          var minutes = String(dtLocal.getMinutes()).padStart(2, '0');
          
          var localDateStr = `${year}-${month}-${day}T${hours}:${minutes}`;
          console.log("Clicked date local time:", localDateStr);
          
          openNewEventModal(localDateStr);
        }
      });
      calendar.render();
      
      // Add event listener for page refresh
      window.addEventListener('beforeunload', function() {
        // Store a flag indicating that we're refreshing the page
        localStorage.setItem('calendarRefreshing', 'true');
      });
      
      // Check if we're coming back from a refresh
      if (localStorage.getItem('calendarRefreshing') === 'true') {
        // Clear the flag
        localStorage.removeItem('calendarRefreshing');
        // Force a refresh of the events
        setTimeout(function() {
          calendar.refetchEvents();
          console.log("Refreshed events after page reload");
        }, 500);
      }
    });

    function openEventModal() { document.getElementById('eventModal').style.display = 'block'; }
    function closeEventModal() { document.getElementById('eventModal').style.display = 'none'; }
    
    function openImageGallery() {
      document.getElementById('imageGalleryModal').style.display = 'block';
      // Clear search field when opening
      var searchField = document.getElementById('imageSearch');
      if (searchField) {
        searchField.value = '';
        searchField.focus();
        // Trigger search to show all images
        filterImages('');
      }
    }
    
    function closeImageGallery() { document.getElementById('imageGalleryModal').style.display = 'none'; }
    
    function selectImage(filename, src) {
      document.getElementById('selectedImage').value = filename;
      document.getElementById('selectedImageName').textContent = filename;
      // Also show the thumbnail
      var nameSpan = document.getElementById('selectedImageName');
      if (nameSpan) {
        nameSpan.innerHTML = `<img src="${src}" style="height:40px;margin-right:5px;vertical-align:middle;"> ${filename}`;
      }
      closeImageGallery();
    }
    
    function openDeleteModal() { document.getElementById('deleteModal').style.display = 'block'; }
    function closeDeleteModal() { document.getElementById('deleteModal').style.display = 'none'; }
    
    // Function to filter images based on search input
    function filterImages(searchText) {
      searchText = searchText.toLowerCase();
      var items = document.querySelectorAll('#galleryModal .gallery-item');
      
      items.forEach(function(item) {
        var tags = item.getAttribute('data-tags') || '';
        var filename = item.querySelector('img').getAttribute('data-filename') || '';
        
        if (tags.toLowerCase().includes(searchText) || filename.toLowerCase().includes(searchText) || searchText === '') {
          item.style.display = '';
        } else {
          item.style.display = 'none';
        }
      });
    }
    
    // Add event listener for search input
    document.addEventListener('DOMContentLoaded', function() {
      var searchInput = document.getElementById('imageSearch');
      if (searchInput) {
        searchInput.addEventListener('input', function() {
          filterImages(this.value);
        });
      }
    });

    document.getElementById('deleteOccurrenceBtn').addEventListener('click', function() {
      // Skip this occurrence for recurring event.
      fetch("/schedule/skip/" + currentDeleteEventId, { method: "POST" })
      .then(response => response.json())
      .then(data => {
        if(data.status === "success"){
          // Remove the occurrence from the calendar.
          // Use the global calendar instance to refresh events
          if (calendar) {
            calendar.refetchEvents();
          } else {
            location.reload(); // Fallback if calendar instance is not available
          }
        }
      })
      .catch(err => {
        console.error("Error skipping occurrence:", err);
      });
      closeDeleteModal();
    });

    document.getElementById('deleteSeriesBtn').addEventListener('click', function() {
      // Delete the entire series.
      fetch("/schedule/remove/" + currentDeleteEventId, { method: "POST" })
      .then(response => response.json())
      .then(data => {
        if(data.status === "success"){
          // Use the global calendar instance to refresh events
          if (calendar) {
            calendar.refetchEvents();
          } else {
            location.reload(); // Fallback if calendar instance is not available
          }
        }
      })
      .catch(err => {
        console.error("Error deleting series:", err);
      });
      closeDeleteModal();
    });

    // Reset the event form when opening for a new event
    function resetEventForm() {
      document.getElementById('eventForm').reset();
      document.getElementById('selectedImage').value = '';
      document.getElementById('selectedImageName').textContent = '';
      document.getElementById('editingEventId').value = '';
      document.getElementById('eventModalTitle').textContent = 'Add Scheduled Event';
      document.getElementById('eventSubmitButton').value = 'Save Event';
    }
    
    // When clicking on a date, reset the form for a new event
    function openNewEventModal(date) {
      resetEventForm();
      if (date) {
        document.getElementById('eventDate').value = date;
      }
      openEventModal();
    }
    
    // Update the dateClick handler to use the new function
    document.addEventListener('DOMContentLoaded', function() {
      // Existing code will still run, this just adds additional functionality
      var calendarEl = document.getElementById('calendar');
      if (calendarEl && typeof FullCalendar !== 'undefined') {
        var existingCalendar = calendarEl._fullCalendar;
        if (existingCalendar) {
          existingCalendar.setOption('dateClick', function(info) {
            var dtLocal = new Date(info.date);
            var isoStr = dtLocal.toISOString().substring(0,16);
            openNewEventModal(isoStr);
          });
        }
      }
    });
    
    // Handle form submission for both adding and updating events
    document.getElementById('eventForm').addEventListener('submit', function(e) {
      e.preventDefault();
      var datetime = document.getElementById('eventDate').value;
      var device = document.querySelector('input[name="device"]:checked').value;
      var recurrence = document.getElementById('recurrence').value;
      var filename = document.getElementById('selectedImage').value;
      var eventId = document.getElementById('editingEventId').value;
      
      if (!datetime || !device || !filename) {
        alert("Please fill in all fields and select an image.");
        return;
      }
      
      // Determine if we're adding a new event or updating an existing one
      var isUpdate = eventId !== '';
      var url = isUpdate ? "/schedule/update" : "/schedule/add";
      var requestData = {
        datetime: datetime,
        device: device,
        recurrence: recurrence,
        filename: filename,
        timezone_offset: new Date().getTimezoneOffset()  // Add timezone offset
      };
      
      // If updating, include the event ID
      if (isUpdate) {
        requestData.event_id = eventId;
      }
      
      // Show immediate feedback to the user
      closeEventModal();
      const feedbackEl = document.createElement('div');
      feedbackEl.style.position = 'fixed';
      feedbackEl.style.top = '20px';
      feedbackEl.style.left = '50%';
      feedbackEl.style.transform = 'translateX(-50%)';
      feedbackEl.style.padding = '10px 20px';
      feedbackEl.style.backgroundColor = '#4CAF50';
      feedbackEl.style.color = 'white';
      feedbackEl.style.borderRadius = '5px';
      feedbackEl.style.zIndex = '10000';
      feedbackEl.textContent = 'Saving event...';
      document.body.appendChild(feedbackEl);
      
      fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(requestData)
      })
      .then(response => response.json())
      .then(data => {
        if(data.status === "success"){
          // Update feedback message
          feedbackEl.textContent = 'Event saved successfully!';
          
          // Use the global calendar instance to refresh events
          if (calendar) {
            calendar.refetchEvents();
            console.log("Refreshed calendar events after saving");
          } else {
            console.warn("Global calendar instance not available for refresh");
          }
          
          // Remove feedback after 2 seconds
          setTimeout(() => {
            document.body.removeChild(feedbackEl);
          }, 2000);
        } else {
          feedbackEl.style.backgroundColor = '#F44336';
          feedbackEl.textContent = "Error: " + data.message;
          setTimeout(() => {
            document.body.removeChild(feedbackEl);
          }, 3000);
        }
      })
      .catch(err => {
        console.error("Error " + (isUpdate ? "updating" : "adding") + " event:", err);
        feedbackEl.style.backgroundColor = '#F44336';
        feedbackEl.textContent = "Error saving event. Please try again.";
        setTimeout(() => {
          document.body.removeChild(feedbackEl);
        }, 3000);
      });
    });
  </script>
{% endblock %}
```


## templates/settings.html

```html
{% extends "base.html" %}
{% block title %}Settings - InkyDocker{% endblock %}

{% block content %}
<div class="container">
  
  <header class="page-header">
    <h1>Settings</h1>
    <p>Manage your eInk displays and AI settings.</p>
  </header>

  
  <div class="card text-center">
    <button id="clipSettingsBtn" class="primary-btn">CLIP Model Settings</button>
  </div>

  
  <div id="clipSettingsModal" class="modal">
    <div class="modal-content">
      <span class="close" id="closeClipSettingsModal">&times;</span>
      <h2>CLIP Model Settings</h2>
      <form id="clipSettingsForm">
        
        <div class="form-group">
          <label for="clip_model">Select CLIP Model for Image Tagging:</label>
          <select id="clip_model" name="clip_model" class="form-select" data-current="{{ config.clip_model if config and config.clip_model }}">
            <option value="">-- Select a model --</option>
            <option value="ViT-B-32" {% if config and config.clip_model == 'ViT-B-32' %}selected{% endif %}>ViT-B-32 (Faster, less accurate)</option>
            <option value="ViT-B-16" {% if config and config.clip_model == 'ViT-B-16' %}selected{% endif %}>ViT-B-16 (Balanced)</option>
            <option value="ViT-L-14" {% if config and config.clip_model == 'ViT-L-14' %}selected{% endif %}>ViT-L-14 (Slower, more accurate)</option>
          </select>
          <button type="button" class="field-save-btn" onclick="saveClipModel()">Save Model</button>
        </div>
        
        
        <div id="modelDownloadContainer" style="margin-top: 15px; display: none;">
          <p>Downloading model: <span id="modelDownloadName"></span></p>
          <div class="progress-container" style="width: 100%; background: #ddd; border-radius: 5px;">
            <div id="modelDownloadProgress" class="progress-bar" style="width: 0%; height: 20px; background: #28a745; border-radius: 5px; color: #fff; text-align: center; line-height: 20px;">0%</div>
          </div>
        </div>
        
        <div style="margin-top: 20px;">
          <p>All models are pre-installed in the system.</p>
          <p>Larger models provide more accurate tagging but require more processing power and memory.</p>
        </div>
        
        
        <div style="margin-top: 20px; text-align: center;">
          <button type="button" class="primary-btn" onclick="rerunAllTagging()">Rerun Tagging on All Images</button>
        </div>
      </form>
    </div>
  </div>

  
  <div class="card text-center">
    <button id="addNewDisplayBtn" class="primary-btn">Add New Display</button>
  </div>

  
  <div id="addDisplayModal" class="modal">
    <div class="modal-content">
      <span class="close" id="closeAddDisplayModal">&times;</span>
      <h2>Add New Display</h2>
      <form id="addDisplayForm" method="POST" action="{{ url_for('settings.settings') }}">
        <input type="text" name="address" id="newAddress" placeholder="Device Address (e.g., 192.168.1.100)" required>
        <select name="orientation" id="newOrientation" required>
          <option value="portrait">Portrait</option>
          <option value="landscape">Landscape</option>
        </select>
        <input type="text" name="friendly_name" id="newFriendlyName" placeholder="Friendly Name" required>
        
        <input type="hidden" name="display_name" id="newDisplayName">
        <input type="hidden" name="resolution" id="newResolution">
        <input type="hidden" name="color" id="newColor">
        <div style="margin-top: 10px;">
          <button type="button" class="primary-btn" onclick="fetchDisplayInfo('new')">Fetch Display Info</button>
        </div>
        <div style="margin-top: 10px;">
          <button type="submit" class="primary-btn">Save</button>
          <button type="button" class="primary-btn" onclick="closeAddDisplayModal()">Cancel</button>
        </div>
      </form>
    </div>
  </div>

  
  <div id="editDisplayModal" class="modal">
    <div class="modal-content">
      <span class="close" id="closeEditDisplayModal">&times;</span>
      <h2>Edit Display</h2>
      <form id="editDisplayForm" method="POST" action="{{ url_for('settings.edit_device') }}">
        <input type="hidden" name="device_index" id="editDeviceIndex">
        <label for="editFriendlyName">Friendly Name:</label>
        <input type="text" name="friendly_name" id="editFriendlyName" placeholder="Friendly Name" required>
        <label for="editOrientation">Orientation:</label>
        <select name="orientation" id="editOrientation" required>
          <option value="portrait">Portrait</option>
          <option value="landscape">Landscape</option>
        </select>
        <label for="editAddress">Device Address:</label>
        <input type="text" name="address" id="editAddress" placeholder="Device Address" required>
        <div style="margin-top: 10px;">
          <button type="submit" class="primary-btn">Save Changes</button>
          <button type="button" class="primary-btn" onclick="closeEditDisplayModal()">Cancel</button>
        </div>
      </form>
    </div>
  </div>

  
  <div id="advancedActionsModal" class="modal">
    <div class="modal-content">
      <span class="close" id="closeAdvancedActionsModal">&times;</span>
      <h2>Advanced Actions</h2>
      <p id="advancedDeviceTitle" style="font-weight:bold;"></p>
      <div style="margin-top: 10px;">
        <button type="button" class="primary-btn" onclick="triggerSystemUpdate()">System Update & Reboot</button>
        <button type="button" class="primary-btn" onclick="triggerBackup()">Create Backup</button>
        <button type="button" class="primary-btn" onclick="triggerAppUpdate()">Update Application</button>
      </div>
      <div style="margin-top: 10px;">
        <button type="button" class="primary-btn" onclick="closeAdvancedActionsModal()">Close</button>
      </div>
    </div>
  </div>

  
  <div class="card">
    <h2>Existing Devices</h2>
    <table class="device-table">
      <thead>
        <tr>
          <th>#</th>
          <th>Color</th>
          <th>Friendly Name</th>
          <th>Orientation</th>
          <th>Address</th>
          <th>Display Name</th>
          <th>Resolution</th>
          <th>Status</th>
          <th>Action</th>
        </tr>
      </thead>
      <tbody>
        {% for device in devices %}
        <tr data-index="{{ loop.index0 }}" data-address="{{ device.address }}">
          <td>{{ loop.index }}</td>
          <td>
            <div style="width:20px; height:20px; border-radius:50%; background:{{ device.color }};"></div>
          </td>
          <td>{{ device.friendly_name }}</td>
          <td>{{ device.orientation }}</td>
          <td>{{ device.address }}</td>
          <td>{{ device.display_name }}</td>
          <td>{{ device.resolution }}</td>
          <td>
            {% if device.online %}
              <span style="color:green;">&#9679;</span>
            {% else %}
              <span style="color:red;">&#9679;</span>
            {% endif %}
          </td>
          <td>
            <form method="POST" action="{{ url_for('settings.delete_device', device_index=loop.index0) }}" style="display:inline;">
              <input type="submit" value="Delete">
            </form>
            <button type="button" class="edit-button" onclick="openEditModal('{{ loop.index0 }}', '{{ device.friendly_name }}', '{{ device.orientation }}', '{{ device.address }}')">
              Edit
            </button>
            <button type="button" class="advanced-button" onclick="openAdvancedModal('{{ loop.index0 }}', '{{ device.friendly_name }}')">
              Advanced
            </button>
          </td>
        </tr>
        {% endfor %}
      </tbody>
    </table>
  </div>
</div>
{% endblock %}

{% block scripts %}
<style>
  .modal {
    display: none;
    position: fixed;
    z-index: 3000;
    left: 0;
    top: 0;
    width: 100%;
    height: 100%;
    overflow: auto;
    background-color: rgba(0, 0, 0, 0.5);
  }
  .modal-content {
    background-color: #fff;
    margin: 10% auto;
    padding: 20px;
    border-radius: 5px;
    max-width: 500px;
    position: relative;
  }
  .close {
    color: #aaa;
    float: right;
    font-size: 28px;
    font-weight: bold;
    cursor: pointer;
  }
  .close:hover,
  .close:focus {
    color: #000;
  }
  .primary-btn {
    background: linear-gradient(to right, #28a745, #218838);
    border: none;
    color: #fff;
    padding: 10px 20px;
    border-radius: 5px;
    cursor: pointer;
    font-size: 1em;
  }
  .primary-btn:hover {
    background: linear-gradient(to right, #218838, #1e7e34);
  }
  .field-save-btn {
    margin-top: 5px;
    font-size: 0.9em;
    padding: 5px 10px;
    background-color: #007bff;
    color: #fff;
    border: none;
    border-radius: 3px;
    cursor: pointer;
  }
  .field-save-btn:hover {
    background-color: #0056b3;
  }
  .overlay-popup {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background-color: rgba(0, 0, 0, 0.7);
    display: flex;
    justify-content: center;
    align-items: center;
    z-index: 5000;
  }
  .overlay-content {
    background-color: #fff;
    padding: 20px;
    border-radius: 5px;
    max-width: 400px;
    text-align: center;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
  }
  .overlay-buttons {
    margin-top: 15px;
    display: flex;
    justify-content: center;
    gap: 10px;
  }
  .cancel-btn {
    background: #6c757d;
    border: none;
    color: #fff;
    padding: 8px 15px;
    border-radius: 5px;
    cursor: pointer;
  }
  .cancel-btn:hover {
    background: #5a6268;
  }
  .progress-container {
    margin: 10px 0;
    background-color: #f1f1f1;
    border-radius: 5px;
    overflow: hidden;
  }
  .progress-bar {
    height: 20px;
    background-color: #4CAF50;
    text-align: center;
    line-height: 20px;
    color: white;
    transition: width 0.3s ease;
  }
</style>

<script>
  // Global modal closing functions
  window.closeAddDisplayModal = function() {
    document.getElementById('addDisplayModal').style.display = 'none';
  };
  window.closeEditDisplayModal = function() {
    document.getElementById('editDisplayModal').style.display = 'none';
  };
  window.closeAdvancedActionsModal = function() {
    document.getElementById('advancedActionsModal').style.display = 'none';
  };
  
  // Overlay message functions
  function showOverlayMessage(message, duration) {
    var overlay = document.createElement('div');
    overlay.className = 'overlay-popup';
    overlay.innerHTML = `
      <div class="overlay-content">
        <p>${message}</p>
        <button class="primary-btn" onclick="this.parentNode.parentNode.remove()">OK</button>
      </div>
    `;
    document.body.appendChild(overlay);
    
    if (duration) {
      setTimeout(function() {
        if (overlay.parentNode) {
          overlay.parentNode.removeChild(overlay);
        }
      }, duration);
    }
  }
  
  function showConfirmOverlay(message, confirmCallback) {
    var overlay = document.createElement('div');
    overlay.className = 'overlay-popup';
    overlay.innerHTML = `
      <div class="overlay-content">
        <p>${message}</p>
        <div class="overlay-buttons">
          <button class="primary-btn" id="confirmYes">Yes</button>
          <button class="cancel-btn" id="confirmNo">No</button>
        </div>
      </div>
    `;
    document.body.appendChild(overlay);
    
    document.getElementById('confirmYes').addEventListener('click', function() {
      overlay.parentNode.removeChild(overlay);
      if (typeof confirmCallback === 'function') {
        confirmCallback();
      }
    });
    
    document.getElementById('confirmNo').addEventListener('click', function() {
      overlay.parentNode.removeChild(overlay);
    });
  }

  // Device status checking
  
  function checkDeviceStatus() {
    fetch("/devices/status")
      .then(response => response.json())
      .then(data => {
        if(data.status === "success") {
          data.devices.forEach(function(device) {
            var row = document.querySelector('tr[data-index="' + device.index + '"]');
            if (row) {
              var statusCell = row.querySelector('td:nth-child(8)');
              if(device.online) {
                statusCell.innerHTML = '<span style="color:green;">&#9679;</span>';
              } else {
                statusCell.innerHTML = '<span style="color:red;">&#9679;</span>';
              }
            }
          });
        }
      })
      .catch(error => {
        console.error("Error checking device status:", error);
      });
  }
  setInterval(checkDeviceStatus, 5000);
  checkDeviceStatus();

  // Modal functions for editing and advanced actions
  function openEditModal(index, friendlyName, orientation, address) {
    document.getElementById('editDisplayModal').style.display = 'block';
    document.getElementById('editDeviceIndex').value = index;
    document.getElementById('editFriendlyName').value = friendlyName;
    document.getElementById('editOrientation').value = orientation;
    document.getElementById('editAddress').value = address;
  }

  function openAdvancedModal(index, friendlyName) {
    document.getElementById('advancedActionsModal').style.display = 'block';
    document.getElementById('advancedDeviceTitle').textContent = "Advanced Actions for " + friendlyName;
    document.getElementById('advancedActionsModal').setAttribute('data-device-index', index);
  }
  
  // Device API functions: triggerSystemUpdate, triggerBackup, triggerAppUpdate remain unchanged
  function triggerSystemUpdate() {
    var deviceIndex = document.getElementById('advancedActionsModal').getAttribute('data-device-index');
    var deviceRow = document.querySelector('tr[data-index="' + deviceIndex + '"]');
    var deviceAddress = deviceRow.getAttribute('data-address');
    if (!deviceAddress) {
      showOverlayMessage("Error: Device address not found");
      return;
    }
    showConfirmOverlay(
      "This will trigger a system update and reboot the device. Continue?",
      function() {
        var progressOverlay = document.createElement('div');
        progressOverlay.className = 'overlay-popup';
        progressOverlay.innerHTML = `
          <div class="overlay-content">
            <p>Triggering system update...</p>
            <div class="progress-container">
              <div id="updateProgressBar" class="progress-bar" style="width: 0%">0%</div>
            </div>
          </div>
        `;
        document.body.appendChild(progressOverlay);
        if (!deviceAddress.startsWith("http://") && !deviceAddress.startsWith("https://")) {
          deviceAddress = "http://" + deviceAddress;
        }
        fetch(deviceAddress + "/system_update", { method: 'POST' })
          .then(response => {
            if (!response.ok) {
              throw new Error("Network response was not ok");
            }
            return response.json();
          })
          .then(data => {
            var progress = 0;
            var interval = setInterval(function() {
              progress += 5;
              if (progress >= 100) {
                progress = 100;
                clearInterval(interval);
                document.body.removeChild(progressOverlay);
                showOverlayMessage("System update triggered successfully. Device will reboot.");
                closeAdvancedActionsModal();
              }
              var progressBar = document.getElementById('updateProgressBar');
              if (progressBar) {
                progressBar.style.width = progress + '%';
                progressBar.textContent = progress + '%';
              }
            }, 500);
          })
          .catch(error => {
            document.body.removeChild(progressOverlay);
            showOverlayMessage("Error triggering system update: " + error.message);
          });
      }
    );
  }
  
  function triggerBackup() {
    var deviceIndex = document.getElementById('advancedActionsModal').getAttribute('data-device-index');
    var deviceRow = document.querySelector('tr[data-index="' + deviceIndex + '"]');
    var deviceAddress = deviceRow.getAttribute('data-address');
    if (!deviceAddress) {
      showOverlayMessage("Error: Device address not found");
      return;
    }
    showConfirmOverlay(
      "This will create a backup of the device. This may take several minutes. Continue?",
      function() {
        var progressOverlay = document.createElement('div');
        progressOverlay.className = 'overlay-popup';
        progressOverlay.innerHTML = `
          <div class="overlay-content">
            <p>Creating backup...</p>
            <div class="progress-container">
              <div id="backupProgressBar" class="progress-bar" style="width: 0%">0%</div>
            </div>
          </div>
        `;
        document.body.appendChild(progressOverlay);
        if (!deviceAddress.startsWith("http://") && !deviceAddress.startsWith("https://")) {
          deviceAddress = "http://" + deviceAddress;
        }
        var progress = 0;
        var interval = setInterval(function() {
          progress += 2;
          if (progress >= 100) {
            progress = 100;
            clearInterval(interval);
            var a = document.createElement('a');
            a.href = deviceAddress + "/backup";
            a.download = "backup_" + new Date().toISOString().replace(/:/g, '-') + ".img.gz";
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            document.body.removeChild(progressOverlay);
            showOverlayMessage("Backup created successfully. Download started.");
            closeAdvancedActionsModal();
          }
          var progressBar = document.getElementById('backupProgressBar');
          if (progressBar) {
            progressBar.style.width = progress + '%';
            progressBar.textContent = progress + '%';
          }
        }, 500);
      }
    );
  }
  
  function triggerAppUpdate() {
    var deviceIndex = document.getElementById('advancedActionsModal').getAttribute('data-device-index');
    var deviceRow = document.querySelector('tr[data-index="' + deviceIndex + '"]');
    var deviceAddress = deviceRow.getAttribute('data-address');
    if (!deviceAddress) {
      showOverlayMessage("Error: Device address not found");
      return;
    }
    showConfirmOverlay(
      "This will update the application on the device and reboot it. Continue?",
      function() {
        var progressOverlay = document.createElement('div');
        progressOverlay.className = 'overlay-popup';
        progressOverlay.innerHTML = `
          <div class="overlay-content">
            <p>Updating application...</p>
            <div class="progress-container">
              <div id="appUpdateProgressBar" class="progress-bar" style="width: 0%">0%</div>
            </div>
          </div>
        `;
        document.body.appendChild(progressOverlay);
        if (!deviceAddress.startsWith("http://") && !deviceAddress.startsWith("https://")) {
          deviceAddress = "http://" + deviceAddress;
        }
        fetch(deviceAddress + "/update", { method: 'POST' })
          .then(response => {
            if (!response.ok) {
              throw new Error("Network response was not ok");
            }
            return response.json();
          })
          .then(data => {
            var progress = 0;
            var interval = setInterval(function() {
              progress += 10;
              if (progress >= 100) {
                progress = 100;
                clearInterval(interval);
                document.body.removeChild(progressOverlay);
                showOverlayMessage("Application updated successfully. Device will reboot.");
                closeAdvancedActionsModal();
              }
              var progressBar = document.getElementById('appUpdateProgressBar');
              if (progressBar) {
                progressBar.style.width = progress + '%';
                progressBar.textContent = progress + '%';
              }
            }, 300);
          })
          .catch(function(error) {
            document.body.removeChild(progressOverlay);
            showOverlayMessage("Error updating application: " + error.message);
          });
      }
    );
  }

  document.addEventListener('DOMContentLoaded', function() {
    var addNewDisplayBtn = document.getElementById('addNewDisplayBtn');
    if (addNewDisplayBtn) {
      addNewDisplayBtn.addEventListener('click', function() {
        document.getElementById('addDisplayModal').style.display = 'block';
      });
    }
    var addDisplayForm = document.getElementById('addDisplayForm');
    addDisplayForm.addEventListener('submit', function(e) {
      e.preventDefault();
      fetchDisplayInfo('new').then(function() {
        addDisplayForm.submit();
      }).catch(function() {
        addDisplayForm.submit();
      });
    });
    document.getElementById('closeAddDisplayModal').addEventListener('click', function() {
      closeAddDisplayModal();
    });
    document.getElementById('closeEditDisplayModal').addEventListener('click', function() {
      closeEditDisplayModal();
    });
    document.getElementById('closeAdvancedActionsModal').addEventListener('click', function() {
      closeAdvancedActionsModal();
    });
    var clipSettingsBtn = document.getElementById('clipSettingsBtn');
    var clipSettingsModal = document.getElementById('clipSettingsModal');
    var closeClipSettingsModal = document.getElementById('closeClipSettingsModal');
    if (clipSettingsBtn) {
      clipSettingsBtn.addEventListener('click', function() {
        clipSettingsModal.style.display = 'block';
      });
    }
    if (closeClipSettingsModal) {
      closeClipSettingsModal.addEventListener('click', function() {
        clipSettingsModal.style.display = 'none';
      });
    }
    window.addEventListener('click', function(e) {
      if (e.target == clipSettingsModal) {
        clipSettingsModal.style.display = 'none';
      }
    });
  });

  function saveClipModel() {
    var clipModel = document.getElementById('clip_model').value;
    if (!clipModel) {
      showOverlayMessage("Please select a CLIP model");
      return;
    }
    var payload = {
      clip_model: clipModel
    };
    showOverlayMessage("Switching to model: " + clipModel + "...", 1500);
    fetch("{{ url_for('settings.update_clip_model') }}", {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    })
    .then(function(response) { return response.json(); })
    .then(function(data) {
      if (data.status === "success") {
        showOverlayMessage("CLIP model updated successfully to " + clipModel);
      } else {
        showOverlayMessage("Error updating CLIP model: " + data.message);
      }
    })
    .catch(function(error) {
      console.error("Error:", error);
      showOverlayMessage("An error occurred while updating CLIP model.");
    });
  }

  function rerunAllTagging() {
    showConfirmOverlay(
      "This will rerun tagging on all images using the selected CLIP model. This may take some time depending on the number of images. Continue?",
      function() {
        fetch("{{ url_for('settings.rerun_all_tagging') }}", {
          method: 'POST'
        })
        .then(function(response) { return response.json(); })
        .then(function(data) {
          if (data.status === "success") {
            showOverlayMessage("Tagging process started! This will run in the background.");
            document.getElementById('clipSettingsModal').style.display = 'none';
          } else {
            showOverlayMessage("Error starting tagging process: " + data.message);
          }
        })
        .catch(function(error) {
          console.error("Error:", error);
          showOverlayMessage("An error occurred while starting the tagging process.");
        });
      }
    );
  }

  function fetchDisplayInfo(mode) {
    return new Promise(function(resolve, reject) {
      if (mode === 'new') {
        var addressInput = document.getElementById('newAddress');
        var address = addressInput.value.trim();
        if (!address) {
          alert("Please enter the device address.");
          reject("No address provided");
          return;
        }
        fetch("/device_info?address=" + encodeURIComponent(address), { timeout: 5000 })
          .then(function(response) {
            if (!response.ok) {
              throw new Error("HTTP error " + response.status);
            }
            return response.json();
          })
          .then(function(data) {
            if (data.status === "ok") {
              document.getElementById('newDisplayName').value = data.info.display_name;
              document.getElementById('newResolution').value = data.info.resolution;
              var availableColors = ['#FF5733', '#33FF57', '#3357FF', '#F39C12', '#8E44AD', '#2ECC71', '#E74C3C'];
              var randomColor = availableColors[Math.floor(Math.random() * availableColors.length)];
              document.getElementById('newColor').value = randomColor;
              resolve();
            } else {
              alert("Error fetching display info: " + data.message + "; using default values.");
              document.getElementById('newDisplayName').value = "DefaultDisplay color";
              document.getElementById('newResolution').value = "800x600";
              document.getElementById('newColor').value = "#FF5733";
              resolve();
            }
          })
          .catch(function(error) {
            console.error("Error fetching display info:", error);
            alert("Error fetching display info; using default values.");
            document.getElementById('newDisplayName').value = "DefaultDisplay color";
            document.getElementById('newResolution').value = "800x600";
            document.getElementById('newColor').value = "#FF5733";
            resolve();
          });
      } else if (mode === 'edit') {
        alert("Fetch Display Info for edit is not implemented yet.");
        resolve();
      }
    });
  }
</script>
{% endblock %}
```


## utils/__init__.py

```py

```


## utils/crop_helpers.py

```py
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
```


## utils/image_helpers.py

```py
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
```


## .DS_Store

```
   Bud1           	                                                           i cbwspblob                                                                                                                                                                                                                                                                                                                                                                                                                                           s t a t i cbwspblob   �bplist00�]ShowStatusBar[ShowToolbar[ShowTabView_ContainerShowSidebar\WindowBounds[ShowSidebar		_{{188, 391}, {920, 436}}	#/;R_klmno�                            �    s t a t i cfdscbool     s t a t i cvSrnlong                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            @      �                                        @      �                                          @      �                                          @                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   E  	                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       DSDB                                 `          �                                         @      �                                          @      �                                          @                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              
```


## .export-ignore

```
node_modules/
*.log
dist/
.vscode/
```


## .gitattributes

```
# Auto detect text files and perform LF normalization
* text=auto
```


## app.py

```py
from flask import Flask
import os
from config import Config
from models import db
import pillow_heif
from tasks import celery, start_scheduler

# Register HEIF opener for Pillow
pillow_heif.register_heif_opener()

def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)

    # Ensure required folders exist
    for folder in [app.config['IMAGE_FOLDER'], app.config['THUMBNAIL_FOLDER'], app.config['DATA_FOLDER']]:
        if not os.path.exists(folder):
            os.makedirs(folder)

    # Initialize database without migrations
    db.init_app(app)

    # Register blueprints
    from routes.image_routes import image_bp
    from routes.device_routes import device_bp
    from routes.schedule_routes import schedule_bp
    from routes.settings_routes import settings_bp
    from routes.device_info_routes import device_info_bp
    from routes.ai_tagging_routes import ai_bp

    app.register_blueprint(image_bp)
    app.register_blueprint(device_bp)
    app.register_blueprint(schedule_bp)
    app.register_blueprint(settings_bp)
    app.register_blueprint(device_info_bp)
    app.register_blueprint(ai_bp)

    # Create database tables if they don't exist.
    with app.app_context():
        db.create_all()

    # Configure Celery
    celery.conf.update(
        broker_url='redis://localhost:6379/0',
        result_backend='redis://localhost:6379/0'
    )

    # Note: Scheduler is now started in a dedicated process (scheduler.py)
    # We still call the function for backward compatibility, but it doesn't start the scheduler
    start_scheduler(app)
    
    # We no longer run fetch_device_metrics immediately here
    # The dedicated scheduler process will handle this

    return app

app = create_app()

# Make the app available to Celery tasks
celery.conf.update(app=app)

if __name__ == '__main__':
    # When running via 'python app.py' this block will execute.
    app.run(host='0.0.0.0', port=5001, debug=True)
```


## config.py

```py
import os

basedir = os.path.abspath(os.path.dirname(__file__))

class Config:
    SECRET_KEY = "super-secret-key"
    # Database: using an absolute path in a “data” folder in the project directory.
    # In the container, basedir will be /app so the DB will be at /app/data/mydb.sqlite.
    SQLALCHEMY_DATABASE_URI = 'sqlite:///' + os.path.join(basedir, 'data', 'mydb.sqlite')
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # Folders for images, thumbnails, and data storage
    IMAGE_FOLDER = os.path.join(basedir, 'images')
    THUMBNAIL_FOLDER = os.path.join(basedir, 'images', 'thumbnails')
    DATA_FOLDER = os.path.join(basedir, 'data')
```


## Dockerfile

```
# Use an official Python image
FROM python:3.13.2-slim

# Set timezone and cache directory for models (persisted in /data/model_cache)
ENV TZ=Europe/Copenhagen
ENV XDG_CACHE_HOME=/app/data/model_cache

# Install system dependencies and redis-server
RUN apt-get update && apt-get install -y \
    curl \
    sqlite3 \
    supervisor \
    tzdata \
    build-essential \
    gcc \
    git \
    libffi-dev \
    libssl-dev \
    libjpeg-dev \
    zlib1g-dev \
    libpng-dev \
    libfreetype6-dev \
    redis-server \
    && ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone \
    && rm -rf /var/lib/apt/lists/*

# Create necessary directories for the database and model cache
RUN mkdir -p /data /app/data/model_cache

# Set working directory
WORKDIR /app

# Copy only the requirements file first for caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install -r requirements.txt

# Pre-download all CLIP models (this layer will be cached if requirements.txt hasn't changed)
RUN python -c "import open_clip; \
    open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai', jit=False, force_quick_gelu=True); \
    open_clip.create_model_and_transforms('ViT-B-16', pretrained='openai', jit=False, force_quick_gelu=True); \
    open_clip.create_model_and_transforms('ViT-L-14', pretrained='openai', jit=False, force_quick_gelu=True)"

# Copy the rest of the project files
COPY . .

# Make scheduler.py executable
RUN chmod +x /app/scheduler.py

# Set environment variables for Celery
ENV CELERY_WORKER_MAX_MEMORY_PER_CHILD=500000
ENV CELERY_WORKERS=2

# Expose port 5001
EXPOSE 5001

# Copy entrypoint script and make it executable
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Copy Supervisor configuration file
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Tell Flask which file is our app
ENV FLASK_APP=app.py

# Run entrypoint.sh (which handles migrations and launches Supervisor)
CMD ["/entrypoint.sh"]
```


## entrypoint.sh

```sh
#!/bin/sh
# entrypoint.sh - Auto-create the database tables then launch the app via Supervisor.

echo "Ensuring /app/data folder exists..."
mkdir -p /app/data

# Start Redis first and ensure it's running
echo "Starting Redis server..."
redis-server --daemonize yes
sleep 2
echo "Checking Redis connection..."
redis-cli ping
if [ $? -ne 0 ]; then
  echo "Redis is not responding. Waiting a bit longer..."
  sleep 5
  redis-cli ping
  if [ $? -ne 0 ]; then
    echo "Redis still not responding. Please check Redis configuration."
  else
    echo "Redis is now running."
  fi
else
  echo "Redis is running."
fi

echo "Creating database tables..."
python -c "from app import app; from models import db; app.app_context().push(); db.create_all()"
echo "Database tables created successfully."

echo "Starting Supervisor..."
exec /usr/bin/supervisord -c /etc/supervisor/conf.d/supervisord.conf
```


## export.md

```md
# Project Structure

```
assets/
  js/
    main.js
data/
images/
migrations/
  versions/
  __init__.py
  alembic.ini
  env.py
  script.py.mako
routes/
  __init__.py
  additional_routes.py
  ai_tagging_routes.py
  device_info_routes.py
  device_routes.py
  image_routes.py
  schedule_routes.py
  settings_routes.py
static/
  icons/
  send-icon-old.png
  send-icon.png
  settings-wheel.png
  style.css
  trash-icon.png
templates/
  base.html
  index.html
  schedule.html
  settings.html
utils/
  __init__.py
  crop_helpers.py
  image_helpers.py
.DS_Store
.export-ignore
.gitattributes
app.py
config.py
Dockerfile
entrypoint.sh
export.md
exportconfig.json
LICENSE
models.py
package.json
README.md
requirements.txt
scheduler.py
supervisord.conf
tasks.py
webpack.config.js
```


## assets/js/main.js

```js
import { Calendar } from '@fullcalendar/core';
import timeGridPlugin from '@fullcalendar/timegrid';
import '@fullcalendar/core/main.css';
import '@fullcalendar/timegrid/main.css';

document.addEventListener('DOMContentLoaded', function() {
  var calendarEl = document.getElementById('calendar');
  var calendar = new Calendar(calendarEl, {
    plugins: [ timeGridPlugin ],
    initialView: 'timeGridWeek',
    firstDay: 1, 
    nowIndicator: true,
    headerToolbar: {
      left: 'prev,next today',
      center: 'title',
      right: 'timeGridWeek,timeGridDay'
    },
    events: '/schedule/events',
    dateClick: function(info) {
      
      var dtLocal = new Date(info.date);
      var isoStr = dtLocal.toISOString().substring(0,16);
      document.getElementById('eventDate').value = isoStr;
      openEventModal();
    }
  });
  calendar.render();
});
```


## migrations/__init__.py

```py

```


## migrations/alembic.ini

```ini
[alembic]
# Path to migration scripts
script_location = migrations
# Database URL - this must point to the same absolute path as used by your app.
sqlalchemy.url = sqlite:////app/data/mydb.sqlite

[loggers]
keys = root,sqlalchemy,alembic

[handlers]
keys = console

[formatters]
keys = generic

[logger_root]
level = WARN
handlers = console

[logger_sqlalchemy]
level = WARN
handlers =
qualname = sqlalchemy.engine
propagate = 0

[logger_alembic]
level = INFO
handlers =
qualname = alembic
propagate = 0

[handler_console]
class = StreamHandler
args = (sys.stderr,)
level = NOTSET
formatter = generic

[formatter_generic]
format = %(levelname)-5.5s [%(name)s] %(message)s
datefmt = %Y-%m-%d %H:%M:%S
```


## migrations/env.py

```py
from __future__ import with_statement
import os
from logging.config import fileConfig

from sqlalchemy import engine_from_config, pool
from alembic import context

# Get the Alembic config and set up logging
config = context.config
fileConfig(config.config_file_name)

# Determine the project root and ensure the data folder exists.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
data_dir = os.path.join(project_root, 'data')
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

# Ensure an empty database file exists so that autogenerate has something to compare.
db_path = os.path.join(data_dir, 'mydb.sqlite')
if not os.path.exists(db_path):
    open(db_path, 'a').close()

# Import all your models so that they are registered with SQLAlchemy's metadata.
from models import db, Device, ImageDB, CropInfo, SendLog, ScheduleEvent, UserConfig, DeviceMetrics
target_metadata = db.metadata

def run_migrations_offline():
    """Run migrations in 'offline' mode."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True
    )

    with context.begin_transaction():
        context.run_migrations()

def run_migrations_online():
    """Run migrations in 'online' mode."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)

        with context.begin_transaction():
            context.run_migrations()

if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
```


## migrations/script.py.mako

```mako
<% 
import re
import uuid
%>
"""${message}

Revision ID: ${up_revision}
Revises: ${down_revision | n}
Create Date: ${create_date}
"""

# revision identifiers, used by Alembic.
revision = '${up_revision}'
down_revision = ${repr(down_revision)}
branch_labels = None
depends_on = None

def upgrade():
    ${upgrades if upgrades else "pass"}

def downgrade():
    ${downgrades if downgrades else "pass"}
```


## routes/__init__.py

```py

```


## routes/additional_routes.py

```py
from flask import Blueprint, request, jsonify, current_app
from models import Device, db
import subprocess, json

additional_bp = Blueprint('additional', __name__)

@additional_bp.route('/fetch_display_info', methods=['GET'])
def fetch_display_info():
    address = request.args.get('address')
    if not address:
        return jsonify({"status": "error", "message": "No address provided"}), 400
    cmd = f'curl -s "{address}/display_info"'
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        return jsonify({"status": "error", "message": result.stderr}), 500
    try:
        raw_info = json.loads(result.stdout)
        colour_str = raw_info.get("colour", "").capitalize()
        model_str = raw_info.get("model", "")
        resolution_arr = raw_info.get("resolution", [])
        if colour_str:
            display_name = f"{colour_str} Colour - {model_str}"
        else:
            display_name = model_str or "Unknown"
        if len(resolution_arr) == 2:
            resolution_str = f"{resolution_arr[0]}x{resolution_arr[1]}"
        else:
            resolution_str = "N/A"
        return jsonify({
            "status": "ok",
            "info": {
                "display_name": display_name,
                "resolution": resolution_str
            }
        }), 200
    except Exception:
        return jsonify({"status": "error", "message": "Invalid JSON returned"}), 500
```


## routes/ai_tagging_routes.py

```py
from flask import Blueprint, request, jsonify, current_app
import os
from tasks import (
    get_image_embedding,
    generate_tags_and_description,
    reembed_image,
    bulk_tag_images,
    BULK_PROGRESS
)
from models import db, ImageDB
from PIL import Image

ai_bp = Blueprint("ai_tagging", __name__)

@ai_bp.route("/api/ai_tag_image", methods=["POST"])
def ai_tag_image():
    data = request.get_json()
    filename = data.get("filename")
    if not filename:
        return jsonify({"status": "error", "message": "Filename is required"}), 400

    image_folder = current_app.config.get("IMAGE_FOLDER", "./images")
    image_path = os.path.join(image_folder, filename)
    if not os.path.exists(image_path):
        return jsonify({"status": "error", "message": "Image file not found"}), 404

    image_embedding = get_image_embedding(image_path)
    if image_embedding is None:
        return jsonify({"status": "error", "message": "Failed to get embedding"}), 500

    tags, description = generate_tags_and_description(image_embedding)
    # Update the ImageDB record with generated tags
    image_record = ImageDB.query.filter_by(filename=filename).first()
    if image_record:
        image_record.tags = ", ".join(tags)
        db.session.commit()
    else:
        image_record = ImageDB(filename=filename, tags=", ".join(tags))
        db.session.add(image_record)
        db.session.commit()

    return jsonify({
        "status": "success",
        "filename": filename,
        "tags": tags
    }), 200

@ai_bp.route("/api/search_images", methods=["GET"])
def search_images():
    q = request.args.get("q", "").strip()
    if not q:
        return jsonify({"status": "error", "message": "Missing query parameter"}), 400
    images = ImageDB.query.filter(ImageDB.tags.ilike(f"%{q}%")).all()
    results = {
        "ids": [img.filename for img in images],
        "tags": [img.tags for img in images]
    }
    return jsonify({"status": "success", "results": results}), 200

@ai_bp.route("/api/get_image_metadata", methods=["GET"])
def get_image_metadata():
    filename = request.args.get("filename", "").strip()
    if not filename:
        return jsonify({"status": "error", "message": "Missing filename"}), 400

    image_folder = current_app.config.get("IMAGE_FOLDER", "./images")
    image_path = os.path.join(image_folder, filename)
    resolution_str = "N/A"
    filesize_str = "N/A"
    if os.path.exists(image_path):
        try:
            size_bytes = os.path.getsize(image_path)
            filesize_mb = size_bytes / (1024 * 1024)
            filesize_str = f"{filesize_mb:.2f} MB"
            with Image.open(image_path) as im:
                w, h = im.size
                resolution_str = f"{w}x{h}"
        except Exception as ex:
            current_app.logger.warning(f"Could not read file info for {filename}: {ex}")

    image_record = ImageDB.query.filter_by(filename=filename).first()
    if image_record:
        tags = [t.strip() for t in image_record.tags.split(",")] if image_record.tags else []
        favorite = image_record.favorite
    else:
        tags = []
        favorite = False

    return jsonify({
        "status": "success",
        "tags": tags,
        "favorite": favorite,
        "resolution": resolution_str,
        "filesize": filesize_str
    }), 200

@ai_bp.route("/api/update_image_metadata", methods=["POST"])
def update_image_metadata():
    data = request.get_json() or {}
    filename = data.get("filename", "").strip()
    new_tags = data.get("tags", [])
    if isinstance(new_tags, list):
        tags_str = ", ".join(new_tags)
    else:
        tags_str = new_tags
    favorite = data.get("favorite", None)
    if not filename:
        return jsonify({"status": "error", "message": "Missing filename"}), 400

    image_record = ImageDB.query.filter_by(filename=filename).first()
    if not image_record:
        return jsonify({"status": "error", "message": "Image not found"}), 404

    image_record.tags = tags_str
    if favorite is not None:
        image_record.favorite = bool(favorite)
    db.session.commit()
    return jsonify({"status": "success"}), 200

@ai_bp.route("/api/reembed_image", methods=["GET"])
def reembed_image_endpoint():
    filename = request.args.get("filename", "").strip()
    if not filename:
        return jsonify({"status": "error", "message": "Missing filename"}), 400
    result = reembed_image(filename)
    return jsonify(result)

@ai_bp.route("/api/reembed_all_images", methods=["GET"])
def reembed_all_images_endpoint():
    task_id = bulk_tag_images.delay()
    if not task_id:
        return jsonify({"status": "error", "message": "No images found"}), 404
    return jsonify({"status": "success", "message": f"Reembedding images in background. Task ID: {task_id}"}), 200
```


## routes/device_info_routes.py

```py
# routes/device_info_routes.py

from flask import Blueprint, request, jsonify, Response, stream_with_context
import httpx
import json
import time
import threading
from queue import Queue, Empty
from datetime import datetime
from models import db, Device

device_info_bp = Blueprint('device_info', __name__)

@device_info_bp.route('/device_info', methods=['GET'])
def get_device_info():
    address = request.args.get('address')
    if not address:
        return jsonify({"status": "error", "message": "No address provided"}), 400

    # Ensure the address has a scheme
    if not (address.startswith("http://") or address.startswith("https://")):
        address = "http://" + address

    try:
        # Use httpx with a 10-second timeout and a curl-like User-Agent
        # Reduce timeout to 5 seconds to fail faster
        response = httpx.get(f"{address}/display_info", timeout=5.0, headers={'User-Agent': 'curl/7.68.0'})
        response.raise_for_status()
        raw_info = response.json()
    except httpx.TimeoutException:
        # Handle timeout specifically to provide a clearer error message
        return jsonify({"status": "error", "message": "Connection timed out. Device may be offline."}), 500
    except httpx.HTTPStatusError as e:
        # Handle HTTP errors specifically
        return jsonify({"status": "error", "message": f"HTTP error: {e.response.status_code}"}), 500
    except Exception as e:
        # Handle other exceptions
        return jsonify({"status": "error", "message": f"Error fetching display info: {str(e)}"}), 500

    try:
        # Build display name as "model colour color"
        colour = raw_info.get("colour", "").strip()
        model = raw_info.get("model", "").strip()
        if model and colour:
            display_name = f"{model} {colour} color"
        elif model:
            display_name = model
        else:
            display_name = "Unknown"
        # Format resolution as "widthxheight"
        resolution_arr = raw_info.get("resolution", [])
        if isinstance(resolution_arr, list) and len(resolution_arr) == 2:
            resolution = f"{resolution_arr[0]}x{resolution_arr[1]}"
        else:
            resolution = "N/A"
    except Exception:
        return jsonify({"status": "error", "message": "Invalid JSON returned"}), 500

    return jsonify({
        "status": "ok",
        "info": {
            "display_name": display_name,
            "resolution": resolution
        }
    }), 200

# Global dictionary for active device stream clients (if needed in future)
active_device_streams = {}

@device_info_bp.route('/device/<int:device_index>/stream', methods=['GET'])
def device_stream(device_index):
    """
    Updated streaming endpoint that connects to the device's live stream,
    updates the device's online status, and pushes status updates into a
    thread-safe queue which is then served via Server-Sent Events.
    """
    devices = Device.query.order_by(Device.id).all()
    if not (0 <= device_index < len(devices)):
        return jsonify({"status": "error", "message": "Device not found"}), 404
    device = devices[device_index]
    address = device.address
    if not (address.startswith("http://") or address.startswith("https://")):
        address = "http://" + address

    # Create a thread-safe queue to pass status updates immediately
    status_queue = Queue()

    def stream_reader():
        try:
            # First, check if the device is reachable via its display_info endpoint
            # Use a try-except block to handle connection errors more gracefully
            try:
                info_response = httpx.get(f"{address}/display_info", timeout=5.0)
                if info_response.status_code != 200:
                    status_queue.put({"status": "error", "message": f"Device not responding: {info_response.status_code}"})
                    return
            except Exception as e:
                status_queue.put({"status": "error", "message": f"Cannot connect to device: {str(e)}"})
                return

            # Now, connect to the live stream (no timeout so it remains open)
            client = httpx.Client(timeout=None)
            response = client.get(f"{address}/stream", stream=True)
            if response.status_code != 200:
                status_queue.put({"status": "error", "message": f"Error connecting to stream: {response.status_code}"})
                return

            # Mark the device as online and update the database
            device.online = True
            db.session.commit()

            # Continuously read and process incoming lines from the stream
            for line in response.iter_lines():
                if line:
                    if isinstance(line, bytes):
                        line = line.decode("utf-8")
                    if line.startswith("data:"):
                        data_str = line[len("data:"):].strip()
                        try:
                            data = json.loads(data_str)
                        except json.JSONDecodeError:
                            continue
                        # Only update online status, ignore metrics
                        device.online = True
                        db.session.commit()
                        # Immediately push the status update to the queue for SSE output
                        status_queue.put({"status": "online", "device": device.friendly_name})
        except Exception as e:
            device.online = False
            db.session.commit()
            status_queue.put({"status": "error", "message": f"Stream reader error: {str(e)}"})

    # Start the stream_reader thread as a daemon
    threading.Thread(target=stream_reader, daemon=True).start()

    def generate():
        # Continuously yield each status update received from the queue as an SSE event.
        while True:
            try:
                data = status_queue.get(timeout=10)
                yield f"data: {json.dumps(data)}\n\n"
            except Empty:
                # If no new status update arrives within 10 seconds, send a heartbeat event.
                yield "data: {}\n\n"

    return Response(stream_with_context(generate()), mimetype='text/event-stream')

@device_info_bp.route('/devices/status', methods=['GET'])
def devices_status():
    """Get online status for all devices"""
    # Don't check device status here - just return the current status from the database
    # This prevents duplicate status checks that could flood the logs
    devices = Device.query.all()
    data = []
    for idx, device in enumerate(devices):
        data.append({
            "index": idx,
            "online": device.online
        })
    return jsonify({"status": "success", "devices": data})
```


## routes/device_routes.py

```py
from flask import Blueprint, request, jsonify, current_app, send_file
from models import db, Device
import httpx
import subprocess, os, datetime, json

device_bp = Blueprint('device', __name__)

@device_bp.route('/device/<int:index>/set_orientation', methods=['POST'])
def set_device_orientation(index):
    orientation = request.form.get('orientation')
    if not orientation:
        return jsonify({"status": "error", "message": "No orientation provided"}), 400
    all_devices = Device.query.order_by(Device.id).all()
    if index < 0 or index >= len(all_devices):
        return jsonify({"status": "error", "message": "Device not found"}), 404
    device_address = all_devices[index].address
    if not (device_address.startswith("http://") or device_address.startswith("https://")):
        device_address = "http://" + device_address
    try:
        response = httpx.post(f"{device_address}/set_orientation", data={"orientation": orientation}, timeout=5.0)
        response.raise_for_status()
        return jsonify({"status": "ok"}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@device_bp.route('/device/<int:index>/display_info', methods=['GET'])
def get_device_info(index):
    all_devices = Device.query.order_by(Device.id).all()
    if index < 0 or index >= len(all_devices):
        return jsonify({"status": "error", "message": "Device not found"}), 404
    device_address = all_devices[index].address
    if not (device_address.startswith("http://") or device_address.startswith("https://")):
        device_address = "http://" + device_address
    try:
        # Use a curl-like User-Agent to mimic curl behavior
        response = httpx.get(f"{device_address}/display_info", timeout=5.0, headers={"User-Agent": "curl/7.68.0"})
        response.raise_for_status()
        raw = response.json()
        return jsonify({"status": "ok", "info": raw})
    except Exception as e:
        return jsonify({"status": "error", "message": f"Error fetching display info: {str(e)}"}), 500

@device_bp.route('/device/<int:index>/fetch_metrics', methods=['GET'])
def fetch_metrics(index):
    """
    Fetch the first SSE metric line from the device's /stream endpoint using httpx streaming.
    """
    all_devices = Device.query.order_by(Device.id).all()
    if index < 0 or index >= len(all_devices):
        return jsonify({"status": "error", "message": "Device not found"}), 404
    device = all_devices[index]
    address = device.address
    if not (address.startswith("http://") or address.startswith("https://")):
        address = "http://" + address
    try:
        with httpx.stream("GET", f"{address}/stream", timeout=10.0, headers={"User-Agent": "curl/7.68.0"}) as response:
            for line in response.iter_lines():
                if line:
                    # httpx.iter_lines() returns bytes if no decoding is set
                    if isinstance(line, bytes):
                        line = line.decode("utf-8")
                    if line.startswith("data:"):
                        data_str = line[len("data:"):].strip()
                        data_json = json.loads(data_str)
                        cpu_usage = str(data_json.get("cpu", "N/A"))
                        mem_usage = str(data_json.get("memory", "N/A"))
                        disk_usage = str(data_json.get("disk", "N/A"))
                        device.cpu_usage = cpu_usage
                        device.mem_usage = mem_usage
                        device.disk_usage = disk_usage
                        device.online = True
                        db.session.commit()
                        return jsonify({
                            "status": "ok",
                            "cpu": cpu_usage + "%",
                            "mem": mem_usage + "%",
                            "disk": disk_usage + "%",
                            "online": device.online
                        })
            # If no valid line is found:
            device.online = False
            db.session.commit()
            return jsonify({"status": "error", "message": "No metrics data received"}), 500
    except Exception as e:
        device.online = False
        db.session.commit()
        return jsonify({"status": "error", "message": str(e)}), 500

@device_bp.route('/device/<int:index>/system_update', methods=['POST'])
def system_update(index):
    all_devices = Device.query.order_by(Device.id).all()
    if index < 0 or index >= len(all_devices):
        return jsonify({"status": "error", "message": "Device not found"}), 404
    device_address = all_devices[index].address
    if not (device_address.startswith("http://") or device_address.startswith("https://")):
        device_address = "http://" + device_address
    try:
        response = httpx.post(f"{device_address}/system_update", timeout=10.0)
        response.raise_for_status()
        return jsonify({"status": "ok"}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@device_bp.route('/device/<int:index>/backup', methods=['GET'])
def create_disk_backup(index):
    from flask import send_file
    all_devices = Device.query.order_by(Device.id).all()
    if index < 0 or index >= len(all_devices):
        return jsonify({"status": "error", "message": "Device not found"}), 404
    device_address = all_devices[index].address
    if not (device_address.startswith("http://") or device_address.startswith("https://")):
        device_address = "http://" + device_address
    data_folder = current_app.config['DATA_FOLDER']
    backup_dir = os.path.join(data_folder, "display_backup")
    if not os.path.exists(backup_dir):
        os.makedirs(backup_dir)
    backup_filename = f"backup_{index}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.img.gz"
    backup_path = os.path.join(backup_dir, backup_filename)
    try:
        response = httpx.post(f"{device_address}/backup", timeout=30.0)
        response.raise_for_status()
        with open(backup_path, "wb") as f:
            f.write(response.content)
        if os.path.exists(backup_path):
            return send_file(backup_path, mimetype='application/gzip',
                             as_attachment=True, download_name=backup_filename)
        else:
            return jsonify({"status": "error", "message": "Backup file not created"}), 500
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@device_bp.route('/device/<int:index>/update', methods=['POST'])
def update_application(index):
    all_devices = Device.query.order_by(Device.id).all()
    if index < 0 or index >= len(all_devices):
        return jsonify({"status": "error", "message": "Device not found"}), 404
    device_address = all_devices[index].address
    if not (device_address.startswith("http://") or device_address.startswith("https://")):
        device_address = "http://" + device_address
    try:
        response = httpx.post(f"{device_address}/update", timeout=10.0)
        response.raise_for_status()
        return jsonify({"status": "ok"}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@device_bp.route('/device/<int:index>/stream')
def metrics_stream(index):
    all_devices = Device.query.order_by(Device.id).all()
    if index < 0 or index >= len(all_devices):
        return "Device not found", 404
    device_address = all_devices[index].address
    def generate():
        command = f'curl -N -s "{device_address}/stream"'
        process = subprocess.Popen(command, shell=True,
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, text=True)
        try:
            while True:
                line = process.stdout.readline()
                if not line:
                    break
                yield line
        except Exception:
            process.kill()
        finally:
            process.kill()
    return current_app.response_class(generate(), mimetype='text/event-stream')

@device_bp.route('/test_device/<int:index>', methods=['GET'])
def test_device(index):
    all_devices = Device.query.order_by(Device.id).all()
    if index < 0 or index >= len(all_devices):
        return jsonify({"status": "error", "message": "Device not found"}), 404
    device = all_devices[index]
    address = device.address
    cmd = f'curl -I --max-time 5 -s "{address}"'
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    current_app.logger.info("Curl output for %s: %s", address, result.stdout)
    if result.returncode == 0 and "200 OK" in result.stdout.upper():
        device.online = True
        db.session.commit()
        return jsonify({"status": "ok"}), 200
    else:
        device.online = False
        db.session.commit()
        return jsonify({"status": "error"}), 500

@device_bp.route('/test_connection_address', methods=['GET'])
def test_connection_address():
    address = request.args.get('address')
    if not address:
        return jsonify({"status": "error", "message": "No address provided"}), 400
    cmd = f'curl -I --max-time 5 -s "{address}"'
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    current_app.logger.info("Curl output for %s: %s", address, result.stdout)
    if result.returncode == 0 and "200 OK" in result.stdout.upper():
        return jsonify({"status": "ok"}), 200
    else:
        return jsonify({"status": "failed"}), 500
```


## routes/image_routes.py

```py
from flask import Blueprint, request, redirect, url_for, render_template, flash, send_from_directory, send_file, jsonify, current_app
from models import db, ImageDB, CropInfo, SendLog, Device
import os
from PIL import Image
import subprocess
from utils.image_helpers import allowed_file, convert_to_jpeg
from utils.crop_helpers import load_crop_info_from_db, save_crop_info_to_db, add_send_log_entry, get_last_sent

image_bp = Blueprint('image', __name__)

@image_bp.route('/thumbnail/<filename>')
def thumbnail(filename):
    image_folder = current_app.config['IMAGE_FOLDER']
    thumbnail_folder = current_app.config['THUMBNAIL_FOLDER']
    thumb_path = os.path.join(thumbnail_folder, filename)
    image_path = os.path.join(image_folder, filename)
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
            current_app.logger.error("Error generating thumbnail for %s: %s", filename, e)
            return "Error generating thumbnail", 500
    return send_from_directory(thumbnail_folder, filename)

@image_bp.route('/', methods=['GET', 'POST'])
def upload_file():
    image_folder = current_app.config['IMAGE_FOLDER']
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
                    new_filename = convert_to_jpeg(file, base, image_folder)
                    if new_filename is None:
                        flash(f"Error converting {original_filename} to JPEG.")
                    else:
                        if not ImageDB.query.filter_by(filename=new_filename).first():
                            db.session.add(ImageDB(filename=new_filename))
                            db.session.commit()
                            # Trigger automatic image tagging
                            from tasks import process_image_tagging
                            process_image_tagging.delay(new_filename)
                else:
                    filepath = os.path.join(image_folder, original_filename)
                    file.save(filepath)
                    if not ImageDB.query.filter_by(filename=original_filename).first():
                        db.session.add(ImageDB(filename=original_filename))
                        db.session.commit()
                        # Trigger automatic image tagging
                        from tasks import process_image_tagging
                        process_image_tagging.delay(original_filename)
        return redirect(url_for('image.upload_file'))
    
    images_db = ImageDB.query.all()
    images = [img.filename for img in images_db]
    devices_db = Device.query.all()
    devices = []
    for d in devices_db:
        devices.append({
            "color": d.color,
            "friendly_name": d.friendly_name,
            "orientation": d.orientation,
            "address": d.address,
            "display_name": d.display_name,
            "resolution": d.resolution,
            "online": d.online,
            "last_sent": d.last_sent
        })
    last_sent = get_last_sent()
    return render_template('index.html', images=images, devices=devices, last_sent=last_sent)

@image_bp.route('/images/<filename>')
def uploaded_file(filename):
    image_folder = current_app.config['IMAGE_FOLDER']
    filepath = os.path.join(image_folder, filename)
    if not os.path.exists(filepath):
        return "File not found", 404
    if request.args.get("size") == "info":
        try:
            with Image.open(filepath) as img:
                max_width = 300
                w, h = img.size
                if w > max_width:
                    ratio = max_width / float(w)
                    new_size = (max_width, int(h * ratio))
                    img = img.resize(new_size, Image.LANCZOS)
                from io import BytesIO
                buf = BytesIO()
                img.save(buf, format="JPEG")
                buf.seek(0)
                return send_file(buf, mimetype='image/jpeg')
        except Exception as e:
            current_app.logger.error("Error processing image %s for info: %s", filename, e)
            return "Error processing image", 500
    ext = filename.rsplit('.', 1)[1].lower()
    if ext == "heic":
        try:
            from io import BytesIO
            with Image.open(filepath) as img:
                buf = BytesIO()
                img.save(buf, format="JPEG")
                buf.seek(0)
                return send_file(buf, mimetype='image/jpeg')
        except Exception as e:
            current_app.logger.error("Error processing HEIC image %s: %s", filename, e)
            return "Error processing image", 500
    else:
        return send_from_directory(image_folder, filename)

@image_bp.route('/save_crop_info/<filename>', methods=['POST'])
def save_crop_info_endpoint(filename):
    crop_data = request.get_json()
    if not crop_data:
        return jsonify({"status": "error", "message": "No crop data provided"}), 400
    
    # Validate crop data
    required_fields = ["x", "y", "width", "height"]
    for field in required_fields:
        if field not in crop_data or not isinstance(crop_data[field], (int, float)) or crop_data[field] < 0:
            return jsonify({"status": "error", "message": f"Invalid crop data: {field} is missing or invalid"}), 400
    
    # Get the selected device resolution if provided
    if "device" in crop_data:
        device_addr = crop_data.get("device")
        device_obj = Device.query.filter_by(address=device_addr).first()
        if device_obj and device_obj.resolution:
            crop_data["resolution"] = device_obj.resolution
            current_app.logger.info(f"Saving crop with resolution: {device_obj.resolution}")
        else:
            current_app.logger.warning(f"Device not found or missing resolution: {device_addr}")
    else:
        current_app.logger.warning("No device provided for crop data")
    
    save_crop_info_to_db(filename, crop_data)
    return jsonify({"status": "success"}), 200

@image_bp.route('/send_image/<filename>', methods=['POST'])
def send_image(filename):
    image_folder = current_app.config['IMAGE_FOLDER']
    data_folder = current_app.config['DATA_FOLDER']
    filepath = os.path.join(image_folder, filename)
    if not os.path.exists(filepath):
        return "File not found", 404
    device_addr = request.form.get("device")
    if not device_addr:
        return "No device specified", 400

    from models import Device
    device_obj = Device.query.filter_by(address=device_addr).first()
    if not device_obj:
        return "Device not found in DB", 500
    dev_width = None
    dev_height = None
    if device_obj.resolution:
        parts = device_obj.resolution.split("x")
        if len(parts) == 2:
            try:
                dev_width = int(parts[0])
                dev_height = int(parts[1])
            except:
                pass
    if not (dev_width and dev_height):
        return "Target resolution not found", 500

    try:
        with Image.open(filepath) as orig_img:
            orig_w, orig_h = orig_img.size
            
            # Check if device is in portrait orientation
            is_portrait = device_obj.orientation.lower() == 'portrait'
            
            # If portrait, swap width and height for target ratio calculation
            if is_portrait:
                target_ratio = dev_height / dev_width
            else:
                target_ratio = dev_width / dev_height
                
            # Log the original image dimensions and target ratio
            current_app.logger.info(f"Original image dimensions: {orig_w}x{orig_h}, target ratio: {target_ratio}")
            
            # Step 1: Apply crop if available
            cdata = load_crop_info_from_db(filename)
            if cdata and all(key in cdata for key in ["x", "y", "width", "height"]):
                x = cdata.get("x", 0)
                y = cdata.get("y", 0)
                w = cdata.get("width", orig_w)
                h = cdata.get("height", orig_h)
                
                # Validate crop coordinates
                if x < 0 or y < 0 or w <= 0 or h <= 0 or x + w > orig_w or y + h > orig_h:
                    current_app.logger.warning(f"Invalid crop coordinates: ({x}, {y}, {w}, {h}) for image {orig_w}x{orig_h}")
                    # Fall back to auto-centered crop
                    cdata = None
                else:
                    # If we have stored resolution and it matches the current device,
                    # use the stored crop data directly
                    stored_resolution = cdata.get("resolution")
                    current_app.logger.info(f"Stored resolution: {stored_resolution}, device resolution: {device_obj.resolution}")
                    
                    if stored_resolution and stored_resolution == device_obj.resolution:
                        current_app.logger.info(f"Using stored crop data: ({x}, {y}, {w}, {h})")
                        cropped = orig_img.crop((x, y, x+w, y+h))
                    else:
                        # If resolutions don't match, we need to recalculate the crop
                        # to maintain the correct aspect ratio
                        current_app.logger.info(f"Recalculating crop to match target ratio")
                        cropped = orig_img.crop((x, y, x+w, y+h))
                        crop_w, crop_h = cropped.size
                        crop_ratio = crop_w / crop_h
                        
                        # Adjust the crop to match the target ratio
                        if crop_ratio > target_ratio:
                            new_width = int(crop_h * target_ratio)
                            left = (crop_w - new_width) // 2
                            crop_box = (left, 0, left + new_width, crop_h)
                        else:
                            new_height = int(crop_w / target_ratio)
                            top = (crop_h - new_height) // 2
                            crop_box = (0, top, crop_w, top + new_height)
                        cropped = cropped.crop(crop_box)
            
            # If no valid crop data, create an auto-centered crop with the correct aspect ratio
            if not cdata or "x" not in cdata:
                current_app.logger.info(f"No crop data found, using auto-centered crop")
                orig_ratio = orig_w / orig_h
                
                if orig_ratio > target_ratio:
                    # Image is wider than target ratio, use full height
                    new_width = int(orig_h * target_ratio)
                    left = (orig_w - new_width) // 2
                    crop_box = (left, 0, left + new_width, orig_h)
                else:
                    # Image is taller than target ratio, use full width
                    new_height = int(orig_w / target_ratio)
                    top = (orig_h - new_height) // 2
                    crop_box = (0, top, orig_w, top + new_height)
                
                current_app.logger.info(f"Auto crop box: {crop_box}")
                cropped = orig_img.crop(crop_box)

            # Step 2: Resize the cropped image to match the target resolution
            current_app.logger.info(f"Cropped image size: {cropped.size}")
            
            # If portrait, rotate the image 90 degrees clockwise and swap dimensions
            if is_portrait:
                current_app.logger.info("Rotating image for portrait orientation")
                cropped = cropped.rotate(-90, expand=True)  # -90 for clockwise rotation
                current_app.logger.info(f"After rotation size: {cropped.size}")
                final_img = cropped.resize((dev_height, dev_width), Image.LANCZOS)  # Note swapped dimensions
            else:
                final_img = cropped.resize((dev_width, dev_height), Image.LANCZOS)
            
            current_app.logger.info(f"Final image size: {final_img.size}")
            
            # Save the processed image as a temporary file
            temp_dir = os.path.join(data_folder, "temp")
            if not os.path.exists(temp_dir):
                os.makedirs(temp_dir)
            temp_filename = os.path.join(temp_dir, f"temp_{filename}")
            final_img.save(temp_filename, format="JPEG", quality=95)
            current_app.logger.info(f"Saved temporary file: {temp_filename}")

        cmd = f'curl "{device_addr}/send_image" -X POST -F "file=@{temp_filename}"'
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        os.remove(temp_filename)
        if result.returncode != 0:
            return f"Error sending image: {result.stderr}", 500

        device_obj.last_sent = filename
        db.session.commit()
        add_send_log_entry(filename)
        return f"Image sent successfully: {result.stdout}", 200
    except Exception as e:
        current_app.logger.error("Error resizing/cropping image: %s", e)
        return f"Error processing image: {e}", 500

@image_bp.route('/delete_image/<filename>', methods=['POST'])
def delete_image_endpoint(filename):
    image_folder = current_app.config['IMAGE_FOLDER']
    thumbnail_folder = current_app.config['THUMBNAIL_FOLDER']
    filepath = os.path.join(image_folder, filename)
    thumb_path = os.path.join(thumbnail_folder, filename)
    if os.path.exists(filepath):
        try:
            os.remove(filepath)
            if os.path.exists(thumb_path):
                os.remove(thumb_path)
            img_obj = ImageDB.query.filter_by(filename=filename).first()
            if img_obj:
                db.session.delete(img_obj)
                db.session.commit()
            return jsonify({"status": "success", "message": "Image deleted"}), 200
        except Exception as e:
            current_app.logger.error("Error removing file %s: %s", filepath, e)
            return jsonify({"status": "error", "message": str(e)}), 500
    else:
        return jsonify({"status": "error", "message": "File not found"}), 404
```


## routes/schedule_routes.py

```py
from flask import Blueprint, request, jsonify, render_template
from models import db, ScheduleEvent, Device, ImageDB
import datetime
from tasks import send_scheduled_image
# Import the scheduler from the dedicated scheduler module instead of tasks
# This ensures we're using the scheduler that runs in a dedicated process
from scheduler import scheduler

schedule_bp = Blueprint('schedule', __name__)

@schedule_bp.route('/schedule')
def schedule_page():
    devs = Device.query.all()
    devices = []
    for d in devs:
        devices.append({
            "color": d.color,
            "friendly_name": d.friendly_name,
            "orientation": d.orientation,
            "address": d.address,
            "display_name": d.display_name,
            "resolution": d.resolution,
            "online": d.online,
            "last_sent": d.last_sent
        })
    
    # Get all images and their tags
    imgs_db = ImageDB.query.all()
    images = [i.filename for i in imgs_db]
    
    # Create a dictionary of image tags for the template
    image_tags = {}
    for img in imgs_db:
        if img.tags:
            image_tags[img.filename] = img.tags
    
    return render_template("schedule.html", devices=devices, images=images, image_tags=image_tags)

@schedule_bp.route('/schedule/events')
def get_events():
    # Ensure we're getting the latest data from the database
    db.session.expire_all()
    events = ScheduleEvent.query.all()
    event_list = []
    
    # FIXED: Use timezone-aware datetime objects to avoid "can't compare offset-naive and offset-aware datetimes" error
    # Create now and horizon as UTC timezone-aware datetimes
    now = datetime.datetime.now(datetime.timezone.utc)
    horizon = now + datetime.timedelta(days=90)
    print(f"Loading {len(events)} events from database")
    
    # Create a device lookup dictionary for colors and friendly names
    device_lookup = {}
    devices = Device.query.all()
    for device in devices:
        device_lookup[device.address] = {
            "color": device.color,
            "friendly_name": device.friendly_name
        }
    
    # Create an image lookup for thumbnails
    image_lookup = {}
    images = ImageDB.query.all()
    for img in images:
        image_lookup[img.filename] = {
            "tags": img.tags or ""
        }
    
    for ev in events:
        # Get device color and friendly name
        device_info = device_lookup.get(ev.device, {"color": "#cccccc", "friendly_name": ev.device})
        device_color = device_info["color"]
        device_name = device_info["friendly_name"]
        
        if ev.recurrence.lower() == "none":
            event_list.append({
                "id": ev.id,
                "title": f"{ev.filename}",
                "start": ev.datetime_str,
                "device": ev.device,
                "deviceName": device_name,
                "filename": ev.filename,
                "recurrence": ev.recurrence,
                "series": False,
                "backgroundColor": device_color,
                "borderColor": device_color,
                "textColor": "#ffffff",
                "extendedProps": {
                    "thumbnail": f"/thumbnail/{ev.filename}"
                }
            })
        else:
            # Generate recurring occurrences from the next occurrence up to the horizon
            try:
                # Parse the datetime string and ensure it's timezone-aware
                start_dt = datetime.datetime.fromisoformat(ev.datetime_str)
                # FIXED: Ensure start_dt is timezone-aware for proper comparison with now
                if start_dt.tzinfo is None:
                    # If the datetime is naive, assume it's in UTC
                    start_dt = start_dt.replace(tzinfo=datetime.timezone.utc)
            except Exception as e:
                print(f"Error parsing datetime for event {ev.id}: {str(e)}")
                continue
                
            # Advance to the first occurrence that is >= now
            rec = ev.recurrence.lower()
            occurrence = start_dt
            # FIXED: Now both occurrence and now are timezone-aware, so comparison works correctly
            while occurrence < now:
                if rec == "daily":
                    occurrence += datetime.timedelta(days=1)
                elif rec == "weekly":
                    occurrence += datetime.timedelta(weeks=1)
                elif rec == "monthly":
                    occurrence += datetime.timedelta(days=30)
                else:
                    break
            # Generate occurrences until the horizon
            while occurrence <= horizon:
                event_list.append({
                    "id": ev.id,  # same series id
                    "title": f"{ev.filename} (Recurring)",  # Mark as recurring in title
                    "start": occurrence.isoformat(),  # FIXED: Use standard ISO format without sep parameter
                    "device": ev.device,
                    "deviceName": device_name,
                    "filename": ev.filename,
                    "recurrence": ev.recurrence,
                    "series": True,
                    "backgroundColor": device_color,
                    "borderColor": device_color,
                    "textColor": "#ffffff",
                    "classNames": ["recurring-event"],  # Add a class for styling
                    "extendedProps": {
                        "thumbnail": f"/thumbnail/{ev.filename}",
                        "isRecurring": True  # Flag for frontend to identify recurring events
                    }
                })
                if rec == "daily":
                    occurrence += datetime.timedelta(days=1)
                elif rec == "weekly":
                    occurrence += datetime.timedelta(weeks=1)
                elif rec == "monthly":
                    occurrence += datetime.timedelta(days=30)
                else:
                    break
    return jsonify(event_list)

@schedule_bp.route('/schedule/add', methods=['POST'])
def add_event():
    data = request.get_json()
    datetime_str = data.get("datetime")
    device = data.get("device")
    filename = data.get("filename")
    recurrence = data.get("recurrence", "none")
    timezone_offset = data.get("timezone_offset", 0)  # Minutes offset from UTC
    
    print(f"Adding event with datetime: {datetime_str}, timezone offset: {timezone_offset}")
    if not (datetime_str and device and filename):
        return jsonify({"status": "error", "message": "Missing parameters"}), 400
    try:
        dt = datetime.datetime.fromisoformat(datetime_str)
        
        # Apply timezone offset if provided (convert from local to UTC)
        if timezone_offset:
            # timezone_offset is in minutes, positive for behind UTC, negative for ahead
            # FIXED: We need to SUBTRACT it to convert from local to UTC (not add)
            # This is because getTimezoneOffset() returns positive minutes for times behind UTC
            # and negative minutes for times ahead of UTC
            dt = dt - datetime.timedelta(minutes=timezone_offset)
            print(f"Applied timezone offset: -{timezone_offset} minutes, new datetime: {dt}")
            
        # FIXED: Convert to proper UTC ISO8601 string for FullCalendar
        # This ensures FullCalendar can automatically convert it to the client's local time
        formatted_dt_str = dt.astimezone(datetime.timezone.utc).isoformat()
        print(f"Parsed datetime: {datetime_str} -> {formatted_dt_str}")
    except Exception as e:
        return jsonify({"status": "error", "message": f"Invalid datetime format: {str(e)}"}), 400
    new_event = ScheduleEvent(
        filename=filename,
        device=device,
        datetime_str=formatted_dt_str,
        sent=False,
        recurrence=recurrence
    )
    db.session.add(new_event)
    db.session.commit()
    
    # Schedule the event using the scheduler from scheduler.py
    # This will be picked up by the dedicated scheduler process
    # Use a unique job ID to avoid conflicts and make scheduling faster
    job_id = f"event_{new_event.id}"
    try:
        scheduler.add_job(
            send_scheduled_image,
            'date',
            run_date=dt,
            args=[new_event.id],
            id=job_id,
            replace_existing=True,
            misfire_grace_time=3600  # Allow misfires up to 1 hour
        )
    except Exception as e:
        print(f"Warning: Could not schedule job: {str(e)}")
        # Continue anyway - the scheduler process will pick up the event later
    
    return jsonify({"status": "success", "event": {
        "id": new_event.id,
        "filename": new_event.filename,
        "datetime": new_event.datetime_str,
        "recurrence": recurrence
    }})

@schedule_bp.route('/schedule/remove/<int:event_id>', methods=['POST'])
def remove_event(event_id):
    ev = ScheduleEvent.query.get(event_id)
    if ev:
        db.session.delete(ev)
        db.session.commit()
    return jsonify({"status": "success"})

@schedule_bp.route('/schedule/update', methods=['POST'])
def update_event():
    data = request.get_json()
    event_id = data.get("event_id")
    new_datetime = data.get("datetime")
    timezone_offset = data.get("timezone_offset", 0)  # Minutes offset from UTC
    
    # Optional parameters for full event update
    device = data.get("device")
    filename = data.get("filename")
    recurrence = data.get("recurrence")
    
    print(f"Updating event {event_id} to {new_datetime} with timezone offset {timezone_offset}")
    
    if not (event_id and new_datetime):
        return jsonify({"status": "error", "message": "Missing parameters"}), 400
    
    try:
        # Convert event_id to integer if it's a string
        if isinstance(event_id, str):
            event_id = int(event_id)
            
        ev = ScheduleEvent.query.get(event_id)
        if not ev:
            return jsonify({"status": "error", "message": "Event not found"}), 404
            
        # Format the datetime string properly
        try:
            # Parse the datetime string
            if 'Z' in new_datetime:
                dt = datetime.datetime.fromisoformat(new_datetime.replace('Z', '+00:00'))
            elif 'T' in new_datetime:
                # Handle ISO format without timezone
                dt = datetime.datetime.fromisoformat(new_datetime)
            else:
                # Handle other formats
                dt = datetime.datetime.fromisoformat(new_datetime)
            
            # Apply timezone offset if provided (convert from local to UTC)
            if timezone_offset:
                # timezone_offset is in minutes, positive for behind UTC, negative for ahead
                # FIXED: We need to SUBTRACT it to convert from local to UTC (not add)
                # This is because getTimezoneOffset() returns positive minutes for times behind UTC
                # and negative minutes for times ahead of UTC
                dt = dt - datetime.timedelta(minutes=timezone_offset)
                print(f"Applied timezone offset: -{timezone_offset} minutes, new datetime: {dt}")
                
            # FIXED: Convert to proper UTC ISO8601 string for FullCalendar
            # This ensures FullCalendar can automatically convert it to the client's local time
            formatted_dt_str = dt.astimezone(datetime.timezone.utc).isoformat()
            print(f"Parsed datetime: {new_datetime} -> {formatted_dt_str}")
        except Exception as e:
            return jsonify({"status": "error", "message": f"Invalid datetime format: {str(e)}"}), 400
            
        # Update the event in the database
        ev.datetime_str = formatted_dt_str
        
        # Update other fields if provided
        if device:
            ev.device = device
        if filename:
            ev.filename = filename
        if recurrence:
            ev.recurrence = recurrence
        
        # Make sure changes are committed to the database
        try:
            db.session.commit()
            print(f"Successfully updated event {event_id} in database to {formatted_dt_str}")
        except Exception as e:
            db.session.rollback()
            print(f"Error committing changes to database: {str(e)}")
            raise
        
        # If this is a recurring event, we need to update the base date
        if ev.recurrence.lower() != "none":
            # Also update the scheduler if needed
            try:
                scheduler.reschedule_job(
                    job_id=str(event_id),
                    trigger='date',
                    run_date=dt
                )
            except Exception as e:
                print(f"Warning: Could not reschedule job: {str(e)}")
        
        # Log the update for debugging
        print(f"Updated event {event_id} to {formatted_dt_str}")
        
        return jsonify({"status": "success"})
    except Exception as e:
        db.session.rollback()
        return jsonify({"status": "error", "message": f"Error updating event: {str(e)}"}), 500

@schedule_bp.route('/schedule/skip/<int:event_id>', methods=['POST'])
def skip_event_occurrence(event_id):
    ev = ScheduleEvent.query.get(event_id)
    if not ev:
        return jsonify({"status": "error", "message": "Event not found"}), 404
    if ev.recurrence.lower() == "none":
        return jsonify({"status": "error", "message": "Not a recurring event"}), 400
    try:
        dt = datetime.datetime.fromisoformat(ev.datetime_str)
        if ev.recurrence.lower() == "daily":
            next_dt = dt + datetime.timedelta(days=1)
        elif ev.recurrence.lower() == "weekly":
            next_dt = dt + datetime.timedelta(weeks=1)
        elif ev.recurrence.lower() == "monthly":
            next_dt = dt + datetime.timedelta(days=30)
        else:
            return jsonify({"status": "error", "message": "Unknown recurrence type"}), 400
        
        # Update the event in the database with the new date
        # FIXED: Use standard ISO format for consistent datetime handling
        ev.datetime_str = next_dt.isoformat()
        ev.sent = False
        db.session.commit()
        
        # Schedule the event using the scheduler from scheduler.py with improved configuration
        job_id = f"event_{ev.id}"
        try:
            scheduler.add_job(
                send_scheduled_image,
                'date',
                run_date=next_dt,
                args=[ev.id],
                id=job_id,
                replace_existing=True,
                misfire_grace_time=3600  # Allow misfires up to 1 hour
            )
        except Exception as e:
            print(f"Warning: Could not schedule job: {str(e)}")
            # Continue anyway - the scheduler process will pick up the event later
        
        return jsonify({"status": "success", "message": "Occurrence skipped"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500
```


## routes/settings_routes.py

```py
from flask import Blueprint, request, render_template, flash, redirect, url_for, jsonify
from models import db, Device, UserConfig
import logging
import httpx  # for querying the Ollama API
from datetime import datetime

settings_bp = Blueprint('settings', __name__)
logger = logging.getLogger(__name__)

@settings_bp.route('/settings', methods=['GET', 'POST'])
def settings():
    if request.method == 'POST':
        color = request.form.get("color")
        friendly_name = request.form.get("friendly_name")
        orientation = request.form.get("orientation")
        address = request.form.get("address")
        display_name = request.form.get("display_name") or "Unknown"
        resolution = request.form.get("resolution") or "N/A"
        if color and friendly_name and orientation and address:
            new_dev = Device(
                color=color,
                friendly_name=friendly_name,
                orientation=orientation,
                address=address,
                display_name=display_name,
                resolution=resolution,
                online=False
            )
            db.session.add(new_dev)
            db.session.commit()
            flash("Device added successfully", "success")
        else:
            flash("Missing mandatory fields (color, friendly name, orientation, address).", "error")
        return redirect(url_for("settings.settings"))
    else:
        devs = Device.query.all()
        devices = []
        for d in devs:
            devices.append({
                "color": d.color,
                "friendly_name": d.friendly_name,
                "orientation": d.orientation,
                "address": d.address,
                "display_name": d.display_name,
                "resolution": d.resolution,
                "online": d.online
            })
        config = UserConfig.query.first()
        return render_template("settings.html", devices=devices, config=config)

@settings_bp.route('/delete_device/<int:device_index>', methods=['POST'])
def delete_device(device_index):
    all_devices = Device.query.order_by(Device.id).all()
    if 0 <= device_index < len(all_devices):
        db.session.delete(all_devices[device_index])
        db.session.commit()
        flash("Device deleted", "success")
    else:
        flash("Device not found", "error")
    return redirect(url_for("settings.settings"))

@settings_bp.route('/edit_device', methods=['POST'])
def edit_device():
    try:
        index = int(request.form.get("device_index"))
        color = request.form.get("color")
        friendly_name = request.form.get("friendly_name")
        orientation = request.form.get("orientation")
        address = request.form.get("address")
        all_devices = Device.query.order_by(Device.id).all()
        if 0 <= index < len(all_devices):
            d = all_devices[index]
            d.color = color or d.color
            d.friendly_name = friendly_name
            d.orientation = orientation
            d.address = address
            db.session.commit()
            flash("Device updated successfully", "success")
        else:
            flash("Device index not found", "error")
    except Exception as e:
        flash("Error editing device: " + str(e), "error")
    return redirect(url_for("settings.settings"))

@settings_bp.route('/settings/update_clip_model', methods=['POST'])
def update_clip_model():
    data = request.get_json()
    config = UserConfig.query.first()
    if not config:
        config = UserConfig(location="London")
        db.session.add(config)
    
    if "clip_model" in data:
        config.clip_model = data.get("clip_model")
        db.session.commit()
        return jsonify({"status": "success", "message": "CLIP model updated."})
    else:
        return jsonify({"status": "error", "message": "No CLIP model provided."})

@settings_bp.route('/settings/rerun_all_tagging', methods=['POST'])
def rerun_all_tagging():
    try:
        # Import the task for rerunning tagging
        from tasks import reembed_all_images
        
        # Start the task
        task = reembed_all_images.delay()
        
        return jsonify({
            "status": "success",
            "message": "Tagging process started.",
            "task_id": str(task.id)
        })
    except Exception as e:
        logger.error(f"Error starting retagging: {str(e)}")
        return jsonify({"status": "error", "message": f"Error: {str(e)}"})

@settings_bp.route('/settings/ollama_models', methods=['GET'])
def ollama_models():
    config = UserConfig.query.first()
    if not config or not config.ollama_address:
        return jsonify({"status": "error", "message": "Ollama address not configured."}), 400
    try:
        url = config.ollama_address.rstrip('/') + '/api/tags'
        response = httpx.get(url, timeout=5)
        response.raise_for_status()
        json_data = response.json()
        models = json_data.get("models", [])
        model_names = []
        for model in models:
            if model.get("name"):
                model_names.append(model["name"])
            else:
                model_names.append(str(model))
        return jsonify({"status": "success", "models": model_names})
    except Exception as e:
        return jsonify({"status": "error", "message": f"Failed to fetch models from Ollama: {str(e)}"}), 500

@settings_bp.route('/device/<int:device_index>/update_status', methods=['POST'])
def update_status(device_index):
    all_devices = Device.query.order_by(Device.id).all()
    if 0 <= device_index < len(all_devices):
        device = all_devices[device_index]
        
        # Update device status
        device.online = True
        db.session.commit()
        
        return jsonify({"status": "ok"})
    return jsonify({"status": "error", "message": "Device not found"}), 404

@settings_bp.route('/devices/status', methods=['GET'])
def devices_status():
    devices = Device.query.all()
    data = []
    for idx, device in enumerate(devices):
        data.append({
            "index": idx,
            "online": device.online
        })
    
    return jsonify({"status": "success", "devices": data})
```


## static/send-icon-old.png

```png
�PNG

   IHDR   `   `   �w8   	pHYs     ��  qIDATx��][�E�E�(�h����jLLDQ���aWF��h�;�n�_��HD�S�b�7��a٭�@�$1�c��2�v���������[����[��c
(P�@�
(P�@0����׹�]B���PX���lpD(�
^���Ol�pY~�K�KZ)"yRv������
�	�W�^�eH��%�!�D����]����%E�|>ֱ�)�q*գ7�7�6;�W����ժ|:d��|.�7.a)�ִ�=�YNѹ�g��ʏ��B�&���~O���M|2c�����P��}���U/v���]�Ϫ���*���4#�P]��נ �������^�f7s�%�����4��9�P�,23�%m4��ky��܆�T��'
�4~�	Z�> S�r�	=�����s���T�R�����GɉV�N@��:e���׳��}Kƙ
�p�6�Q腚M>������d]��*k�
-��G/����A��ñ��K�����I�E�,�X�(bp		ۂ�"��Au̒�=�ĥ5�e�=\�G�����:.;BXw��`N��[���\�Ű�~W3C�P�Af�,	�f׵sG�x������ӹ;��XZ\�1�N������XJ@%D�℄_�o(�y��5X��kܫD�8n*x)�z���9���ˎ?M�ɯ��ϓ(u�X|�!�w��C��b�������;^�%:�f�nk4_*�����������41��w�]��iVź�ټ�" ��s��4#�M�JF��v�}�;��=���eSM@�
���P��\����c �R;4���G�TJ � �u������k��_טH*Wf�c޾y7�z��w�d_��a��mV���1�b=!�$�#�%�����=��;-���(Ck�ZL32O���o�)�����d�(wب.#*��/�7hn����,���l� �d� 9��咀፰��%�k��s�3����g*x֎�l�VB;p�p�6N�_EJ�L(Xe(|�YG`-�{C(��K���S�S(XK��6>�c^Pem�H�\���č\�Kn����z���H@PԊG߅=؇x<h*\����,,�@����#��P��F���V'@����AI��M����U���2��+�� ��Zf�$U�&~����FB	
V%����VK *�)P�3(o%����IU�K�+t�5Kp�����/�=Y(���_�\�_Ի{�'���r	��e����&���tF��	�@·��"
0M����'�S�<��nEB?�$�[�d]l^����[ 4��^�U�&Kf�?QG�$H<(7D��Lx"��F4�'�
>�[@���Ƀm������-Ix��4Cu�T�tCD�A	\§���]�G��f��-s~/�K<���+'��tM��
�A�t#�K�����ٻ��Bp�>իE�p2��V�F}[F��^�8PK[��=
/�\`��f�Ac��%��J��
�)��ݚ0/�z��t}�LP=�Zഘ���tI�7�F}�e��d��pXCDt���������am��i*�w��,o1�ub\��,o0bHE8���T>}!�%~�gC�;Y���5ax/fw�3��Ҩ�Fp�Ӽ�kc�+�,��^�	�	��=nS[��p �KI"��pQZ��*�g� �30�Eҵ�0�V��T�&�$ueGq�H�a*�V��鉠]=�M��m�K\�ZUjO���)�OZ��p��W"V���N�{�G�ʷ�6�r��fN�.��ۯv�\f��QV���L������q�05�7���N��Q��äH�X�j�}������8�\M[��B��ܦ�(P� ��1���`�+    IEND�B`�
```


## static/send-icon.png

```png
�PNG

   IHDR   `   `   �w8   	pHYs     ��  �IDATx��?hTAƟh�XX�Z�hc)(7s$�� o���mJ[K[K۔i��0;���Iaac}��"D�pg�E����|?x���ۛ��{w]          �!d�~b��A(����+�'6}�>�/x�#,�/J�GI^Pzp�{m!X���tklr�{�!8R�黱���;�ސ�6�>����g�����'Bf������E���,`q�W6٤t��}-`>����%Y�����q`C�<�ؕR Į�x^�ε >zN@�<��x��y��OD`�������[����b�0/{����=X;�xj덝w�]켃��b�G;��8��y�����?$�-v��pt������{\�F;�����/S�o�i��k�<�qx�锒~�$��/B��HEQ�b�RQT�صX�$vm%�]���d��T oˍ�uF� 2�BI^N��W��lS�pM���fwb�����-�%�X�`��G�9��Z7$�7���y����7n�~x�%��^���,Nm�v)_�~V�85|*H���������w���8妾�?O-N�)~�[a�d�M�cf�Xq�MY�]�����qʍ�٭N�r��^q��p�]���0f���_onV~�┛����)7��Lq������┛>�8�愣┛%V�Ӫ��n�8����3�t���n�՟�^           ]�|gi�:�.�    IEND�B`�
```


## static/settings-wheel.png

```png
�PNG

   IHDR         �x��   sBIT|d�   	pHYs  �  ��+   tEXtSoftware www.inkscape.org��<    IDATx���w�U����I/�$��{����ذ �"�rE/W�^�]���k�],�*��$������}~�'�9{f�g����<�'q�63{�5����������������������������������������������������������������������������������������������������������������YZ=��%0x0����e�jh2q,��cV>w ��& { ;�� ���K������ s�Ȩek
�_'�ll���.�n%���� *ɩ�Yˍ^�x
X^ � ��$�������u_�z
8x5qm��Y�v�I�b/��^ك�G�(�5�����_�)����7�]**��YcM��%�{�X��x�`Ͱ�k�m������^A����8�9�~�%�b�w�V���P�UQ��G�,헸�ff�5	x7p;���`q�E�r[[W��~����׺�Y�m
|�t��ˈ���*�J�"b٧��YQ�� l����rv 1{��,�b)�Z���n�sM-!�&읤&��228��÷�xo�bey?��۸x%0��Z13� � �@��-#<��w��.ʈ��{��,��jk5����2c)pd��d�A����2c�qbKb3�ژ�H쟮~���'��Mk;b���zH�⎀���Z��j6�g1�VB�Yw��BT�X���33+���'�MO�ʪ�br�Ukp)���:>�w43��Ą�Y���8�����Fg�owe<�w>ͬb#�%Km~J��S/�V���K��_{fV�7���_'��,��DHu;�Wv_�ff+�p1�]��}��K
#��o�����]ֱ��3L#�\�<J���|�`�F��o׺����:�T���hb�_·�����eX8}{�1� ���U3���{�2�v`�aֽ=mS���X��p�0���ZfSbI���դx8t8�` <��;u�5%z�9S��f�|c�� �?��ˀ��ZC18	�9I�o�ע�{������FCk�V��;�vjC\�:�f1��Y�*q���aԦ��i��&=�[h��>9�R�0a�MdfMq8p�P��<A`+�\����xΪ���m
�bB������)�hL�L N�[���L�����xVu�q?��{6�����������̬VV~����Xu�K37n��}=;Vˀ/�mI3����/��1�x�_��Z�~&' ��׫cxq>`ȬvF��3��s��\kw��>��w,%�e�Wf5�-p5���ܸ�ob��\m��}}9ʍ�0�,� �O�X�������%E��عq��F��^��^R���ה:/�xx1f������? ��������E�-e�%΀� p��3�5�=�?�I�	a����<���׀��M���ձX���3`����B�����u�p��~,�S�u�O'f��щ���Y�XJ|y/�����j�Q�a1S����'6+ڢ���U ˉU��M �&�n�Se�2rpq}���[h�>�����>�A���14;��>|��$�Y�& ?A� �:"�̇{�i����G|��[�X���Ͷ�K�(jf�m\����2n#֒٘��xP����;~L��r��7gP�*�2`��ff�p͟x�?.".*���H���ˑg��b_��� ^B,�S���x��|��Jv1iK}�W��n��H�����+N#ݒ���33(c����df%�*���8�j"�JP>G��s�9�� �o��)⛴�p+�����fNg{�UaC4�F	yw�;�Fy��CՖ��E����̬uރ�&N�C���׃CE� �?G�~`��*̬-������x9�l��.�[׎�����#�{B]/��	b�J3�uh�z����r<����=u9��lb;�܌�I3���Ė�/,����,�7l�1�8�fB����6����ˑ&n�&o�Gy7�P��s˫*��9
��Zf�ذ�Jk
p�zs������m��8|G]we��Ĺf6�x�>�7iqq_�>��4!�[􎤞�'�%u=�7ˬ$�&8��Y4fwʑ�{��:���N��C4c��(➚��Nˊo�ZCf57��$��1��^�G�ZeW��T�����1�8��]� �A_�eD/pH��cV_Ǡ�)����C˯�l��ƚ���1x+�,1M�p�1Y�f�[�Pϝ��_$���e�8}�;�?���`�Y`	�z/�,�b��f�w��u��)*�^<���Ǯ�ŚkO�}����O��z�q����[�^)��%�d�g����n,1P�k���W�Y}���&J�욨�j�Cu۴-��4��i�gעo���%)*ì.B�,z�/�mpG����s�[���M�hp2��}�F�$6A�,�NU��	܁�ݚ���뺡O�$N�S��P��T�Y�G�(~����5
x#�<R݆u���7Ќ����,��m8��,Q�e�x�7����'�a$�j�b�mZ���82ڿ���!��^��MW�i�rb���$�7_���>i��iw��ăN�ƹ�"� �ݺ�c[���{[��+��$+�Y�����?VK[��[xp=���%�#���Z�^m�& �G��+
o
d���7�b�T�֮�׀��_U�C�)�.�kц�M��J�+)m��o�����Mg$1��%�=q�Nb�����Ԝuk�;z���%6��W��tK����҆��Ĳ�s�	��s�1��$�D`�Rk�ʰ6p��/����f� �ￒ�Һ5���b�{�����EtZ>J��o�v�u0
�*�kg9pu�� M>2�.��@��+���y��ۘ�?�+��zK`sbO�*,�&6<����U�����w�.Ż�x5aq@o?��{K�����(�-�M�u���"�z�&u��Ľ�����6���ģ�/���t��Alm�r�>��-L߬rk�z[|$uA�,[��:�3���Ki����zs��ԙ �Tg��d^�~D�aq���@.Pg�x�<M�	3�8Z�b	�U��<��� ���:fV�͈ j��3�6� ��l�	u&��3`f�{1���k�0S�)�����cW���O�=�&/��� ��� S��ՙ0�ʌTg�(�@>�F��΀�Uf_�8�w���; ��%��U�G�3��a� s��ՙh#w ��ku��Y��0��H�	�܎~B��Kifj�yzhe<7uAmp�ϟ��r�����xqf ���Z� ��� ۫3`f��p����R2� ��r�U����Yz9t �Rg�����"�XL%� �5�.�����)w �)��;�5�`Kq��g�K���+��%�7�f�����鷞; y�N�`#u�,�m� .Qg�����L�q��of�l-N!p�8��@����̀̚kq�7���r� ��.q�0k.u�Fq��; 9�.N_��0�t���8}���M�?E������8�[��� ��q����7�4F ��y�W���@��?I�����,�$@����� �l�8}w ̚I}o?<)΃�@�� �0k����?*N�:�ȗ�D�1���,�����ӷw �<'{�0m3KG�
`�8}�p o˄i�`�L�C��ӷw lE�0k�q�ʑM���|�A�T� �fR�ӷw 򥞩�T�����/`�șz��g�5��0N��u���5��`�L����=F��@�6��:f�4O��zt�:�ȗ��.� �5�z��5��[�; ���������C��p g[�ӟ!N���Pw �k��`�������/N���x
X$�����w r5�R��{��Y:��W?�w r�0J�����Y:��[=�i���}��# fM�����}��?<*΃���� �$N�p W������,-�$߭��v�z� �gS`3q�0k�;��� {���z� ��� 7�3`fI����a�S�����3 ܠ΀�%�00K��C��ee�I�ra,&�.������f)� ����/��5�|q�,��������˫� .Ug��*�Ou�#�0��:��rHn91
afͷ��͓�����y oA��/�e��Y%�D&�j���y0�E�̥�	�Y�~���sF�Rڠ<��#�ՙ �Rg��*�Ü��3�F� ���t�Y�3����ޤ΄��!�����щ�jfy�f��<��?����G��8�X�`f��8G�	`=��L�U)�_����g�V�D��Y��(��	\���[�#��Y��A{�,�J\V�,����?O\V3��_�?��K�s��,�Ձ��l}����5�̽	�s�/ޚ��fR�F���L��6k�5���Gˉ>��� ��M�_JZZ3����?���3��j�P��D�:���[{�[��%v�ܠ��F�ހx�4�8jz1�s40��u=X���9��~�(��NL�GF7ѫ�Sՙ�X삷'O��j�x�:���⨛�����rg`'`����ppc'�'�*·�g,p?0U���K���Z;����8"i��}��}�x >���YQ,�|�Xεn�������:�'�-�Yz[s��L����F�F�'���_+E�.�;������-�e诛�X l���f	�܊�F�NYh�q�ˁߐ�/��188��`y:���?�#^O���(򻙖{�OXn[�Q��?&�ѡ*b!��5x�Wn�D}�o'-�Y�@��IYh[�g�f��r�����]ԫ�gy�Z�9e����q�7�`�0!]�m#�!���o���jb'8��z�ka`,�JXf�R��fYQ�/a��V'f1߃���� '��Z�F��u;�*a��
y'y���?��f5����o��S�+���V�\��A>{��'��9VoHVr�D\m�ԗ:�	="P���uhY��
�$�L�{W�o���?��TV>�F��M���/S��2V�[ѷ���
�	0�q�/����SU@���<��}����|�eZ#�-������wLdm�R�7���T�b��*�m{�����b^���W��m��b{��,ف1�4Q��d�`)��u<���R:}�,�!N65K���J�E?���y1p�6u�oç����?�f�� ���s�_�C���Ŀ2L���=C���j����mWK�	�V��-�ky�B`���2��'���1��3H{Z�V�>#`� 6�2+d��/���GRTF��"�s��hT������#ǃ�Vw{��k�)������`q�|Ǻ��'�����x�X�~��M�K�O��m�V#vs{���M�6.�V��(�I���(7�/��0���M�w��������֗�]1-�|�|�qp�^b�N���"�w��?�/�:����N3~�}��i���Oз���8�8�������&�'��Zk"�:br_S&y]��P��Z�����Qm\l�1�}[v������j�+�Y�O���ʌx�s7v ��o?�&�㥲E�D��V�e�x��v�2�د������/�� أ�:k�]��|G񘍗����XF,N��w���qDO�X���_hޯ�ű%�_��N���m��#��[�u
�v,3�>	lSf%�N���p���Ľ-�M:�֡��CW6t�g�]�왞"VМ��HM�&�T=G��n&�g۫/�f�=��8��S�
s�o�׶:����t�����nM�Dߎ)c.p���s���V��w1��7x�q*>�g����@��|��+�*[Ҿ�5��'���^���8�}���t[�-�����QM��k�؛x��nGe<�xކx�6&��Ս�{�o^1\�7��������"^J�+W�c1��$�Y�M �a��_�_l�r.ћ\��H��\JLm��������}���w���B�u�d� ����i�k���̨�t`�i�w�#����2�W����ڬ��}��G<�4\��ѷ]�x8�����2*o&��> �N���&�$�%�Λh�n�e�Bb�ak���נo�:�o�%66t#�?�o���8����N��_�=�.��Kh���Wlo�~U7���&�=�+�q
>���B�ve�2b��[��K����	���4���멍ށG��}�
K&ԕ�{,N貎��4�As?���F��P56'6��}=����eVP��v�(�H���LnB_ѹ�|b���4c�ߋ�ӌџ��+�w��z-sh߄Ҳ=/�]Q����jA�������^b�z����K�<���+&#�����MxR`Q{Ҿ͂�ߥ�#'�����bٚu��۰��;�:�n_bOuu�w�*�JZgb��-s�����<�8fQ]���2�=o#�~*r4�v�&�^��>ꠇx5P׽�_R~�����Vu[��Ğ/�7�z��S���]׮A@R���y�ҹq	�n� &�F��e8��=��L#��P��B)�q`������Wd�q:y/骋_�o��ĥ��=�v�o㦳��D;EL�T�iNq^�MlC����b.�1�w$��j,$��4af*#���tj�k��D;mDl(�nӜ�U�jT��+/�8�z�������t(q�c�jh���;з�P�<y�L���Ҝ�<�����B5*r���%�'&<Yy~��]�g{`���}�%~�������T�xw������+-�X|�8����\�4����ָ5���O=�88Q��Db;����O�΃��JSǙ�<�F��AR����z9�����He_�5����Q�e�uy�".�/^����,�여���7�/�|G���X��n#\�'���,E\���6�}[�(f;%+�mG��?���j�����XLM��}eU /��z����{Eq^�_�́��{E��tE�~� F[��]eSJ�%�5��:��#��jL%ߍB�NWt`K�A��>X<����6���R���J�������RŃĶ���V[6T�v,���
�k���?X����6�#��o��qci��У�+��X
�	x)0��z��[��e���<b��i�A�����{MB��j��e���l����ʈ�Ļ�w��|�51���I�zG��_OYh[���6hڪ��ʬ��=}u��Ky��]1ֵ	���C)m��1���������� W��&ʈו[=�z?�
j���v�GM�[n�w��V��x�GNz�S�_�#)mö-p2y�"YU|��Z)�O�WЊ�b3�����-o#�;�_;��_�d=nˬ܆����9A���p;��d8��Q�k�W��o�w��/6NYhK�U诧���ꗳ=�S��I�xK�[�'v<�<'���;�Aa����rޓ��V����p��Ĵŵ�Y�Ui�k%@�+�;��k�,%�Q���W��Ҫ�5��\�{����诗��s�[J#�gѫ�/�Ǐ�,��c�W���5ǧ�_O}1o�['���_7}�ŵ
MC=���(냈�Ni�S����Z��ib2��íį�\��X�j��0q�Ri�̗�ش��������<�3�q�u&l�>ܥ�D�db�5ÿ��g�ؤ���; ��zu�y?1���eyM
~�:Vu`sq��R�����hX�|���[�Zz�L\���h��^K���(k`4�	xh���|P�+�c�t�ï�B=��zTV`C�K��h��3�q>�����y�Lt�΀�B� �H���F;$27}�#����.L\V���诧�ī�1��j�B�̴�Je� l\��t�>q�V����$�ˈw��W ��3A���_�	+l)p�8�� ��ؤ���ֽ���G�3�q�:V���3Б�5nŨt��L� �I��g Vg�J�;�m՞�΀��~q��G�V�Z���L ?��3a�[J���=��:V��;'��zF�z8Ɗ{��$�D���'D+� ���`Ź��Ԓ>�[~P9��?_KMv?q���; ��� �Ot&<��    IDAT)�0�X���v���� �[u,�S���&Pw F��Β�]��б�Clo������DKk*�������d�5�5�جh!�X���(b&��Uw[�[*]F�i�6�X��1�d@����.��[F�Y%|F��ӷ��Rg 8]����� ;�3`�=(N?��Z%|F��ӷ�r� h�C[�΀���E@=�8}+n;q���[�y��\�������+ �# �ي�T��%�ׇ[u��*y@���{�� �G ��0V�H�I]"Nߪ�q�� �y�b��)E?�	 �0��!��U)���Z׉��A������b@�P��b6Wg �I��ܵ��?CŊQ�d�_�g�n+��f�Dߓ����A=�ŊQ?7V/�et Ɣ�E�Po��ӿU����(Nq�V���gl�(�P8-�oŬ#N����K}��z��8��?��0�X��Sx����雎�@�j+f�8���0�@��� �������� x w �N}��#��MG��+�zS������+�Y��MG���$Nߊ� ~`�x�T���~vZ1� ������7u��H�K�P?;��w�; ���%N�8}�Qw�<Po(#�Po�/`u���p����,6�)�3�X&NߊQ��7������
_?et ���´��6�/p�++���O�%%|F���?���b���>_���~�]��G LMف�N��3Q��G �M���0m+���z��>O� w �M��b@���m�8}w �k�8}��7u � ��"�;�M�o:��x<Po�@# �u�� ԟ���8}��X��z�ˊQw ���� �F���>�wq����8}�i�V�8q�Y� (wC�A��ǊQ� l.N�t��������,� <Q�g��Y1��_��v��=Po�%�O��2: sJ��"��7�+�M���V�5��yp�ަ��/�㩌�z"�; �v�8�`7q�z;��?�@��8<Z��00^��s��$��o��K�~/��8V�z���� S[�~`q�V�}��?��2�;w �w <P�����p�Uk?q�׋ӷ���Ǌ~@^���Xq��:���<Xu�G�����8�$@� �o+N� 8B��Lm�@��#N?��z@}��w�:������ ~��
ӞG&[�7rq��nD2���Zj�)��<�������5R)߻et ,�3��+��[\)��H�e�<Xz/C�˵� ���F<3T�� �B����f�X����XrG�3 \�΀��8��e|H�^�>�[�m�K� �Ǉ5�:���L Ug�
[O���2>�� �̒>���m����{�N�K����J��c�ˊQN ���)����R�V�y�u�L����,��M�L]��W+�# �('�Oy�0��� 6^�΄���q����X)6�?��i�+���Z���<Tg������Ҧ� 4�´��	��7o$2��]��*0�Xޢ���⠴E�
��zZ�F%.�U�!t�Qi?��P�k�^�a�X���D�����|T���s�Ot��&��hzYTV�Β>�[� 4��t��Xh��y���+u���(qɲ: ��J��n���+r�y|LpݝLm88W�	+�z���: ���%}V7^|x-�������*��>�Q�LX�^E� ���G��G=���i���'�􏇉�>K���Q��x��/�D�w��$`��/I[\����^K;�/��}�M��x �=p<�U�j��F;�v`|8mq-�/��n�b&�\��F;�I�NG�Ri)0.i����7�pc:�]����׈q2��/ۥ-��hG�2�u�_M[\+Ѧ�v���'^ۨ���q[��s��)��ݫ��J��-��%��/.ǿ��`����z�^�y��z�[�S�b�����}�j(n#��SV<�x)޸C�诅��޴ŵ|�u�?<�?/#�=�I��$�C�O�_%�A|q�+��x���˫*�cз�X 운�V���i{���0i�m��'&�ߋ��(ǔ]1e�}��e�Y�19�+n���ju���;��)m]Y�ؑT}}�;�AeJӀ��E-�;�ZC%�:�
�"n �ï��ې�r��"w��C~ː�'�,���n�$���H/&�%�G���*�N���/UKcu�q�m=0|V@>>��zO���*�%��}ۧ��J��D&��;�*�~�mx�x*'�oぱxy�Bې��h��00�������!�����I?(�Β�m�v�qpx�*��$~Q��w`, ���ܶr{4�����$�v�rL$V�<�������Rs��}E��|b3+��з�`1�)a�mp�_���,>��ح7��"�i�Ъb��/�	��W��R�bb7�	Ū�:��ѷ�`q���dK�۱�,���Sy>�\�Ɗ�����̧�WX.q�tЊ���\Q� �NWt�ؒ��[��+��IV��ڎ�PIݶ��X�Z��T�Lwu���&�*[�&��F�W?)�N~�B��_���Srۏ��XF�[+E_q�Ńx�`QǢoǕ�#��J�^���;����$��؏8�^ݦ9�
֥�x�n���[�_�ԏ_p1�v\Y, ^��Z��D���ueq%^\�	�7�si�*�.T�B���\Q܌����'�_W��(�w�^��O�m��%Q��.����3�8�P�f�+�+1ט����m��o��ĩ��Du�dkF�~C��&����އ�����6�5c��Wf��K�����Y�|���;�Q��P���f�U[w��Cߎ9�g�kV�w��М�|�J`��D�nC�%Čfw�V��8@�Nۉ��&�a7bu��Ӱ3f6�����آ��m���o��ą��Ij��v.B�>ÉZ���ı4﴾��I2?��[[������2��L�:���S���G�L��k�g��n�����r�e�K����F,�QWt�18��
n�É��vn��8A}�������ˈ��lx��&���9���ڭ���w�Wvαx}��B�B�f�ƥ�A��H���ާ�~��*i�5�Kз]����k������:5�o�v�ep-�6+�ƛ~���Y��H\�.�bnCb�u��K���vkn
�m���!rw�nc`�6+W�ь/��P�L��Z4�$#��ۊ�,�T�������
�%�ˀ7v_��r(ѣV�Y1���n�#�4���w�Rk�����s<��+�M��݆ڀ8�u:��)���Z[���۫춿x'��F���sh^G�}%�Sl���(���{��T����x��rb��O\�b^u{��%���E����&vs��R���J��v،�-ͭ"����j�U��JF�%���M:�1���-��=I��F���M$Nl�!-�� �!g%Jk2�a���đ����������3R��Z6�#'�{ܩ��3թ�2#���g��݁Q�L%��/p�:#��Ft�6q��F���?�K��}�I�O|��%�O�D�+�gub1�`�NlKt���.bUƣ�����N�;�3���߉��ӉN@v���jDG�P�eİT��FtWg$s�돳v�Zx8��:#51�X�y�:#��_�7�p� .��{��N��췣TV��(��j+b�eu{9����D���ۭ�����h�<bt����!F~H�OW_PE��VOc�<���y�|�p}}�������SV�YV'����^�eWLC
,D�f�<�	��c
�ќ}7��>b���4�5�u��@}O*�Cs۪DL~S��#��O��c(���$uە��?$�U�K�����<���n\�ɿ�ڞ��nu�9�9Ą?�	Ą7u��;�wK\̀��?I���?��2j[�S�#:�^�6|?F�vE��U�7Ҳ�mI̴W_�C�e�Ij��6&�S��͡�[����:}�u���͆�Xbi���J�CLn���F�P���ڸ�8���g}�1��7ܘ	���ng�m@r����?LTM5�X�n7G5q:��:�L=ę��N,�	���>�ez�S�r?��h��ۈ-r���Hw_����v�з�p�`�$5a���_)p06U4؁x��&�,|�fQ���z����͒X��ž��@��7ۆ�߶��Ǖ�t��������MTf�0���~E��;�*�o�&����.z�/�X�^��-���z�b#�����Wg�+z+���c�^2H{��F=�J�2��oB�C�(�KX�6�L�ST��ch��Y�������se�x{�қ�I�o���┅n�����6ǃ��+l=���׈�v]Q�^���f]�,�c�xa�B����>�з�#��6� �l9�1�']�ͺ�|�20n F$,w��\��]�W�r�\;�o'w>��26
8��20�IY��^	LG߶m�����ܩM#�]O���}�&G��o��qM���D�S�;Iu7=f�Ǜ\��?�v,�5�V#;��o��qX������'зs�b>��˳��;}{�^�P6�Z9�X������/i�k�d���i��,b)ں�j�ֳ�s���,�YJG��]����L�x}��-�����W���~`�,i��\��Fꋟ�-�0��,��Y�K\�
�ا�9y�V.'�-��f�����4���)i�k+��z�^��A.18�Xzf:_F-�9��IKlV����T}q|��ʍ&Y9�vN|��h�`��/��oΊ'�Y�� �W�'.��Xb������3�]_I2c�8���?~���f���{���Նo�����b��I����q���XOV�K�_/}1�������o�.�6�&�p��#,�>AҲf���������N[\����Tb��$q>&N�Z&·�(``[`;`��?oI�6��"b�㻁[��:qWE�[�� �O�����3�� �|T�	�@�"u&��Q�f�Ds�΀�����y�Ļ���/Otbn�?� �_�}�@��[�� ����-&:�w�3b��$��&���5�l���_K\V��|�M7/�2k���-'F��%.�YV��C��lm��A� P?�N\V�,���H�R�Yn�@��YN�5Y#qYm���[�t�8}3����t|��lj�JC�_��3�ͬ.F��>��Z��oD��5k���c�I�L*�W y8�ء��ͬ:��?�}W��6s �?��8}3�΁��H^��D�����������̪������ m��q�υ�>y)�Lm<1�W���%��[�# �XH�P���fͷ+��"W����z� �����w�of���� �;u��ܨG v�of������:f��!�^U��{ }�L�����O_D
� �e9p�0�u�fM6�Z�����[�; ��F�v��0}3Kk`�8�ӷw ��Oq�[��7�t�Sg�X`p ?7����Ysm)N.p�8��@~���_G������8�k�M�,� �g)�	P�*L���Z_������w �t�0���i�YZ����F��Ӄ´=`�\�W ��ӷ~��ӣ´=`�\�9>w�ӷ~�ȓ���0m3Kgq��C���w ���0��´�,u��)`�8֏; yZ$L� �f�$N�q�6�; yZ&L{0R������ <!N�p Oʍ� Ɖ�7��_,�o������=`�<���?%N�p O�#y�s�,������ � �I��� �&R�,�G l w �� ,�mf�G ƈӷ��Ӛ´�0k&w ��ȓ����5���^]���@��Gv*�!6�t��_K���@��# ޭˬ���5�wNV��l&L�aa�f�Γhw�G��@~6E��O�2k���\q6�o����m����of�(O�B����@~v�?C����s�8}w 2�@~���8}3K�>q��N������� �!N���Qw v�o������Β���ofi�+N`�8��@^����fSw z�]�y�w �r�8����YZ��3 �΀w �1x�8��ofi�D��W=�i� �� `5q��of��$N/�G�[�; �x�8����<�Yz7���G��@����z�ǅ�Yz� ���0w r�"�ge{�߬�Tg x	q�	���7�3 \�΀�U�&�8X=��Lns����XLM]P3��Yh�9ˁ�'/���G �G�ע_dfչD�b�Ӛ�L������& oPg8O�3�������Ug�L�-����.��ee��g�թj�������Y�������:uAmp~��y�y&^�o�Fg�3�q�:m��N.a�F�3��Ы�~i-t��7���ە�C��j��;1i�vWg���`�N� S:�~ʀ�7��e���$>J,���<g {��0��FZ9w 4v"���X��������z�:�4`����$q���������~�U������N�P9w 4r��z?p�:ֵI�(����;�v^ϤNl�;��A<į���7���tbS���8�Vr@csu�[ [=L!&KF|�n�����mЉ#���������A~~����8}��|�������KiE� ��O.'�L=Y�h��_ ����r�&1Oym����f��9ڛ���E�.�_�_@���:f �D~�m�1�Cy-�'�Yr�G{�����6������RV�L���1?�M^AL�T���#Yk�������`�� �$b�E� �)����u�ڪ�?�G���o��,?A�-'��|g�6!~��B����l�x�X�&� �۷|?i��2�y�7\_�왶�F����}��-?�v��@[��oӁ��e6��	�o��� xS����ĩk���9z���;���1�q�`q\�r�e��o���{�Cal���M�ˈ�49�Q��'u��,vKVz�̬���[Q�L�*g�Y�6�}[6=��Թn�z�m��X �NUf9�����������x�=fK��s���;�>����xa�
0����x����Cdl�'��U�W��z`�U�UL�ϗ�x�j0����o����Ĝ�w�������t�?�ߏ8n�}���0��*IM�el"������<[x�8��q G����g�gB�Ό�,Ee�����߀É�����D}L%����±���Ka�d;�,�뻛xs��0����߀�<T����>rw8� �6p/�&Yj���v������q�׊Y�����M�<?A}�h,p
޳�α�X)�;�D��H�w�bV3�S�/���	W�Q�����5�~%0���R�n����ߠ�!��Lb�q����71n%�`�������2⨒�Ƭ�6$63QߔE�"��O�ۉCh�u�Hs�C��5�u�u)��+�z����o�2b	�e�;Ip�5���H����D�F_We��ZeV�YS��ZV<��z��}�9��O��u�/p��)3���f�\��F-3. �;�n"���rh�'Ĳ:����2�����K�'�FZ��L����ؤ�j*�z���ɡ�SѝJ7�00oy�k�WyUe�l���iˎ���IM�X����H���Vep0��2���{ʪ,��X���"�G�sWZ�}}8򊳨f��K��*�-���Zfp&�9UL'�W��C�Wׁ#�8�t�����1e<�'��68��M
�wSmG`pK�29�?���;��r��K��53+����o�1�د=嫁	���Q��:��B����N|_|�j�PX��}��ۚ����:#�M>M,�ZR��"��}a����G�+Kn'�m~x��ӻ������|,1Ys*�knm`�NlELTQU!�>|����A��o]v,b�n�{ꌘ5�$�4�=�*�>b���Rj.~��˔2��ۼ���Ne��?��8KT�*����j�� .� �U��}�Yez�K���U�\�K����C�#E�ک��=]m�+q>��2�uSf<��7�K,�H7�+  �IDAT6��V+�̬b�E��*� �b�;��f���,��a�C���.�9u?�z������V���2���Faf2��-l/ ^ɪWlE� ��[4���B���l	|�����Ǣq�k���	}3�S�1�v�k0���į�%����g�M��������;��)_݌!���}����A!�њYf����e���KyzX�Χ+>|�8��iF GW��g��b	�ڳ�ì���#mg����&�"F4Vذ���c�=�%�'6��x�,L��p-�NL�h��l�Q�;�3"���xf���Qaf5�p���c�q���a�L~FsV�9!Fg̬�F��8�K�/���%6�R�O[�t<�ϬQv�B�pqD�O�
g��L�wn�-��ͬ�F���wh�b�}[�c�y�۬�q����5�N�?�?t�ˈ�r붑��N���ۯ�1����kҬEFƣUœ��w��b�u;6)N�������E�FM��o]�~��=�w'�����C�NM�{�-���j'⥂���`��W��5���ўc�SǍx�5�p'`8q���l�v.G�Ъs��5܊�aq'`���v��������,������zïr���o�c�E�y���UdM����n�A`�jں�Y��S�lY�F���ي8a�C�+���=��`�Z�#����[��~3Kh?bB��a�[�+)Lc4�7�ׁ"&��,\�ff������/��x�
�RL�]��%���K�;3�aMl#:��Pǿ�r�3�*F}M��E�AI>����&G�p�:�)^�V�7��.R}��01�M&�x�ò���_J�Yپ���(+�?6-�����|���!�����7
�#�k�H,~lWrݘ�%7�ح�A�Ӳ���Z�� F�7?%�ޚ���D�4�#p#�y]�ߡ�f�s���3k����k�?l����ˮKj�����a��x�^3k���w���?����~��H�z�����63�Ֆ�Y��?�WĿК`=�讣e��È���Zo
p"��䶔X���u�<�'�]G���mVE����h$�"b�������ׄz�������-sB5E23k�u�_�wR��\�=�Zrk���P��3��{UZ3��!&�XH�/���N����n[ o �\<Ċ��Y�?����V�]3�vX��\�2�YA���,���2E���$Nܬ��a�Ֆ'6Y�M�&�[�g�K��$��^\���wff���5ͺ���yk��� ��G����e�33333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333k���jb����a    IEND�B`�
```


## static/style.css

```css

* {
    box-sizing: border-box;
  }
  body {
    margin: 0;
    font-family: 'Helvetica Neue', Arial, sans-serif;
    background: #f0f2f5;
    color: #333;
    line-height: 1.6;
  }
  .container {
    max-width: 1200px;
    margin: 20px auto;
    padding: 0 20px;
  }
  .card {
    background: #fff;
    border-radius: 8px;
    padding: 20px;
    margin-bottom: 20px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
  }
  
  
  .navbar {
    background: #2c3e50;
    color: #ecf0f1;
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 10px 20px;
    position: relative;
  }
  .navbar .logo {
    font-size: 1.8em;
    text-decoration: none;
    color: #ecf0f1;
  }
  .navbar .nav-links {
    display: flex;
    gap: 15px;
  }
  .navbar .nav-links a {
    text-decoration: none;
    color: #bdc3c7;
    padding: 8px 12px;
    transition: background 0.3s, color 0.3s;
    border-radius: 4px;
  }
  .navbar .nav-links a:hover,
  .navbar .nav-links a.active {
    background: #34495e;
    color: #ecf0f1;
  }
  
  
  .nav-toggle {
    display: none;
  }
  .nav-toggle-label {
    display: none;
    cursor: pointer;
  }
  .nav-toggle-label span,
  .nav-toggle-label span::before,
  .nav-toggle-label span::after {
    display: block;
    background: #ecf0f1;
    height: 3px;
    width: 25px;
    border-radius: 3px;
    position: relative;
    transition: all 0.3s ease;
  }
  .nav-toggle-label span::before,
  .nav-toggle-label span::after {
    content: '';
    position: absolute;
  }
  .nav-toggle-label span::before {
    top: -8px;
  }
  .nav-toggle-label span::after {
    top: 8px;
  }
  @media (max-width: 768px) {
    .nav-links {
      position: absolute;
      top: 100%;
      right: 0;
      background: #2c3e50;
      flex-direction: column;
      width: 200px;
      transform: translateY(-200%);
      transition: transform 0.3s ease;
    }
    .nav-links a {
      padding: 15px;
      border-bottom: 1px solid #34495e;
    }
    .nav-toggle:checked ~ .nav-links {
      transform: translateY(0);
    }
    .nav-toggle {
      display: block;
    }
    .nav-toggle-label {
      display: block;
    }
  }
  
  
  .gallery {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
    gap: 15px;
  }
  .gallery-item {
    position: relative;
    overflow: hidden;
    border-radius: 8px;
    transition: transform 0.3s ease, box-shadow 0.3s ease;
    height: 150px; 
  }
  .gallery-item:hover {
    transform: scale(1.05);
    box-shadow: 0 4px 10px rgba(0,0,0,0.2);
  }
  .img-container {
    height: 100%;
    overflow: hidden;
  }
  .img-container img {
    height: 100%;
    width: auto;
    display: block;
    margin: 0 auto;
    object-fit: cover;
    cursor: pointer;
  }
  
  
  .current-image-container img,
  .last-sent-img {
    max-width: 300px;
    max-height: 300px;
    width: auto;
    height: auto;
    margin: 0 auto;
    display: block;
  }
  
  
  .overlay {
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: rgba(0,0,0,0.5);
    opacity: 0;
    transition: opacity 0.3s ease;
    border-radius: 8px;
  }
  .img-container:hover .overlay {
    opacity: 1;
  }
  
  .crop-icon {
    position: absolute;
    top: 5px;
    left: 5px;
    width: 30px;
    height: 30px;
    display: flex;
    align-items: center;
    justify-content: center;
    color: #fff;
    background: rgba(0,0,0,0.5);
    border-radius: 50%;
    cursor: pointer;
    z-index: 10;
  }
  .crop-icon:hover {
    background: rgba(0,0,0,0.7);
  }
  
  .delete-icon {
    position: absolute;
    top: 5px;
    right: 5px;
    width: 24px;
    height: 24px;
    cursor: pointer;
    z-index: 10;
    color: #fff;
    font-size: 20px;
  }
  
  
  .favorite-icon {
    position: absolute;
    top: 5px;
    left: 50%;
    transform: translateX(-50%);
    width: 24px;
    height: 24px;
    cursor: pointer;
    z-index: 10;
    color: #fff;
    font-size: 20px;
  }
  
  .send-button {
    position: absolute;
    left: 50%;
    top: 50%;
    transform: translate(-50%, -50%);
    background: #28a745;
    color: #fff;
    border: none;
    padding: 8px 12px;
    border-radius: 4px;
    cursor: pointer;
    transition: background 0.3s ease;
  }
  .send-button:hover {
    background: #218838;
  }
  
  
  .lightbox-modal {
    display: none;
    position: fixed;
    z-index: 4000;
    left: 0;
    top: 0;
    width: 100%;
    height: 100%;
    overflow: auto;
    background-color: rgba(0,0,0,0.9);
  }
  .lightbox-content {
    margin: auto;
    display: block;
    max-width: 90%;
    max-height: 80%;
    animation: zoomIn 0.3s;
  }
  @keyframes zoomIn {
    from { transform: scale(0.8); opacity: 0; }
    to { transform: scale(1); opacity: 1; }
  }
  .lightbox-close {
    position: absolute;
    top: 20px;
    right: 35px;
    color: #f1f1f1;
    font-size: 40px;
    font-weight: bold;
    transition: 0.3s;
    cursor: pointer;
  }
  .lightbox-close:hover,
  .lightbox-close:focus {
    color: #bbb;
  }
  #lightboxCaption {
    text-align: center;
    color: #ccc;
    padding: 10px 0;
  }
  
  
  .popup {
    position: fixed;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    background: rgba(255,255,255,0.95);
    border: 2px solid #ccc;
    padding: 30px;
    box-shadow: 0 4px 10px rgba(0,0,0,0.3);
    z-index: 10000;
    text-align: center;
    font-size: 1.5em;
    display: none;
    border-radius: 8px;
    animation: popupFade 0.5s ease;
  }
  @keyframes popupFade {
    from { opacity: 0; transform: translate(-50%, -50%) scale(0.8); }
    to { opacity: 1; transform: translate(-50%, -50%) scale(1); }
  }
  
  
  .progress-container {
    width: 60%;
    margin: 20px auto;
    background: #ddd;
    border-radius: 5px;
    display: none;
  }
  .progress-bar {
    width: 0%;
    height: 30px;
    background: #28a745;
    border-radius: 5px;
    transition: width 0.4s ease;
    color: #fff;
    line-height: 30px;
    font-size: 1em;
    text-align: center;
  }
  
  
  .modal {
    display: none;
    position: fixed;
    z-index: 3000;
    left: 0;
    top: 0;
    width: 100%;
    height: 100%;
    background-color: rgba(0,0,0,0.7);
    padding: 20px;
  }
  .modal-content {
    background: #fff;
    margin: 5% auto;
    padding: 20px;
    border-radius: 8px;
    max-width: 500px;
    position: relative;
  }
  .close {
    position: absolute;
    top: 10px;
    right: 15px;
    font-size: 1.5em;
    color: #333;
    cursor: pointer;
  }
  
  
  input[type="submit"],
  button,
  .primary-btn {
    background: linear-gradient(to right, #28a745, #218838);
    border: none;
    color: #fff;
    padding: 10px 15px;
    border-radius: 4px;
    cursor: pointer;
    font-size: 1em;
    transition: background 0.3s ease;
  }
  input[type="submit"]:hover,
  button:hover,
  .primary-btn:hover {
    background: linear-gradient(to right, #218838, #1e7e34);
  }
  
  
  input[type="text"],
  input[type="password"],
  select {
    width: 100%;
    padding: 10px;
    margin: 10px 0;
    border: 1px solid #ccc;
    border-radius: 4px;
  }
  label {
    font-weight: bold;
  }
  
  
  .calendar {
    width: 100%;
    border-collapse: collapse;
    margin-top: 20px;
  }
  .calendar th,
  .calendar td {
    border: 1px solid #ddd;
    padding: 8px;
    text-align: center;
  }
  .calendar th {
    background: #f8f8f8;
  }
  .calendar .droppable.over {
    background: #dff0d8;
  }
  
  
  .footer {
    text-align: center;
    padding: 15px;
    background: #2c3e50;
    color: #bdc3c7;
    position: fixed;
    bottom: 0;
    width: 100%;
  }
  
  
  a:focus,
  button:focus,
  input:focus {
    outline: 2px solid #2980b9;
    outline-offset: 2px;
  }
```


## static/trash-icon.png

```png
�PNG

   IHDR   0   0   W��   	pHYs     ��  �IDATx��KNA��^A]�]|$�H�"}
ءG���[�`���	P��63�!H��46Z_R!���ꯙE�R� T�պf��x�Q� -�S)��h�{Cd�p��Q)b��]�����m�J�v�$}�����<�V(�R�����rg�o�.��*5	Z����W��&�d`��LG5&L�-��Uj���o��
u�V�e�AU$L�3��50��!�goઌ/�8�q_2��'���H$
�!#D2Ba���P2B$#ƿ!η/�櫘Ѳ5�*_2$}�����c�����N������sq@W?#:�R x�:�{o<�܁U(a��e����)�SU3&:b�e���X�C���E00̚��'���; �i������ݰE�7�H    IEND�B`�
```


## templates/base.html

```html
<!doctype html>
<html>
  <head>
    <title>{% block title %}InkyDocker{% endblock %}</title>
    <link rel="stylesheet" href="{{ url_for('static', filename='style.css') }}">
    
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    
    <link rel="stylesheet"
          href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css"
          crossorigin="anonymous"
          referrerpolicy="no-referrer" />
    
    <link rel="stylesheet"
          href="https://cdnjs.cloudflare.com/ajax/libs/cropperjs/1.6.2/cropper.min.css"
          crossorigin="anonymous"
          referrerpolicy="no-referrer" />
    <style>
      /* Make the site scroll fully behind the footer with a sticky footer */
      html, body {
        height: 100%;
        margin: 0;
        padding: 0;
      }
      /* FIXED: Use min-height: 100vh instead of 100% to ensure the wrapper takes up the full viewport height
       * This ensures the footer stays at the bottom even when content is minimal
       */
      .wrapper {
        min-height: 100vh; /* Use viewport height instead of percentage */
        display: flex;
        flex-direction: column;
      }
      /* FIXED: Use flex: 1 instead of flex: 1 0 auto to ensure the content area expands to fill available space
       * This pushes the footer to the bottom of the viewport
       */
      .main-content-wrapper {
        flex: 1;
      }
      .footer {
        flex-shrink: 0;
      }
    </style>
    {% block head %}{% endblock %}
  </head>
  <body>
    <div class="wrapper">
      <nav class="navbar navbar-expand-lg navbar-dark bg-dark" role="navigation" aria-label="Main Navigation">
        <div class="container-fluid">
          <a href="{{ url_for('image.upload_file') }}" class="navbar-brand">InkyDocker</a>
          <button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarNav"
            aria-controls="navbarNav" aria-expanded="false" aria-label="Toggle navigation">
            <span class="navbar-toggler-icon"></span>
          </button>
          <div class="collapse navbar-collapse" id="navbarNav">
            <ul class="navbar-nav ms-auto">
              <li class="nav-item">
                <a class="nav-link {% if request.endpoint.startswith('image.') %}active{% endif %}" href="{{ url_for('image.upload_file') }}">Gallery</a>
              </li>
              <li class="nav-item">
                <a class="nav-link {% if request.endpoint.startswith('schedule.') %}active{% endif %}" href="{{ url_for('schedule.schedule_page') }}">Schedule</a>
              </li>
              <li class="nav-item">
                <a class="nav-link {% if request.endpoint.startswith('settings.') %}active{% endif %}" href="{{ url_for('settings.settings') }}">Settings</a>
              </li>
            </ul>
          </div>
        </div>
      </nav>
      <div class="main-content-wrapper">
        {% block content %}{% endblock %}
      </div>
      <footer class="footer bg-dark text-light text-center py-3">
        <p class="mb-0">© 2025 InkyDocker | Built with AI by Me</p>
      </footer>
    </div>
    
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
    
    <script src="https://cdnjs.cloudflare.com/ajax/libs/cropperjs/1.6.2/cropper.min.js"
            crossorigin="anonymous"
            referrerpolicy="no-referrer"></script>
    {% block scripts %}{% endblock %}
  </body>
</html>
```


## templates/index.html

```html
{% extends "base.html" %}
{% block title %}Gallery - InkyDocker{% endblock %}

{% block content %}
<div class="container">
  
  <div class="card current-image-section">
    <h2 id="currentImageTitle">Current image on {{ devices[0].friendly_name if devices else 'N/A' }}</h2>
    <div class="current-image-container">
      {% if devices and devices[0].last_sent %}
        <img
          id="currentImage"
          src="{{ url_for('image.uploaded_file', filename=devices[0].last_sent) }}"
          alt="Current Image"
          class="last-sent-img small-current"
          loading="lazy"
        >
      {% else %}
        <p id="currentImagePlaceholder">No image available.</p>
      {% endif %}
    </div>
    {% if devices|length > 1 %}
      <div class="arrow-controls">
        <button id="prevDevice">&larr;</button>
        <button id="nextDevice">&rarr;</button>
      </div>
    {% endif %}
  </div>

  
  
  
  <div id="uploadPopup" style="position: fixed; top: 50%; left: 50%; transform: translate(-50%, -50%); background: rgba(0,0,0,0.8); color: #fff; padding: 15px 20px; border-radius: 5px; z-index: 1000; display: none;">
    <div class="spinner"></div> Processing...
  </div>

  
  <div class="main-content">
    
    <div class="left-panel">
      <div class="card device-section">
        <h2>Select eInk Display</h2>
        {% if devices %}
          <div class="device-options">
            {% for device in devices %}
              <label class="device-option">
                <input
                  type="radio"
                  name="device"
                  value="{{ device.address }}"
                  data-index="{{ loop.index0 }}"
                  data-friendly="{{ device.friendly_name }}"
                  data-resolution="{{ device.resolution }}"
                  {% if loop.first %}checked{% endif %}
                >
                {{ device.friendly_name }}
              </label>
            {% endfor %}
          </div>
        {% else %}
          <p>No devices configured. Go to <a href="{{ url_for('settings.settings') }}">Settings</a>.</p>
        {% endif %}
      </div>

      <div class="card upload-section">
        <h2>Upload Images</h2>
        <form id="uploadForm" class="upload-form" method="post" enctype="multipart/form-data" action="{{ url_for('image.upload_file') }}">
          <input type="file" name="file" multiple id="fileInput" required>
          <br>
          <input type="submit" value="Upload">
          <div class="progress-container" id="progressContainer" style="display: none;">
            <div class="progress-bar" id="progressBar">0%</div>
          </div>
          <div id="uploadStatus"></div>
        </form>
      </div>
      
      
    </div>

    
    <div class="gallery-section">
      <h2>Gallery</h2>
      <input type="text" id="gallerySearch" placeholder="Search images by tags..." style="width:100%; padding:10px; margin-bottom:20px;">
      <div id="searchSpinner" style="display:none;">Loading...</div>
      <div class="gallery" id="gallery">
        {% for image in images %}
          <div class="gallery-item">
            <div class="img-container">
              <img src="{{ url_for('image.uploaded_file', filename=image) }}" alt="{{ image }}" data-filename="{{ image }}" loading="lazy">
              <div class="overlay">
                
                <div class="favorite-icon" title="Favorite" data-image="{{ image }}">
                  <i class="fa fa-heart"></i>
                </div>
                
                <button class="send-button" data-image="{{ image }}">Send</button>
                
                <button class="info-button" data-image="{{ image }}">Info</button>
                
                <div class="delete-icon" title="Delete" data-image="{{ image }}">
                  <i class="fa fa-trash"></i>
                </div>
              </div>
            </div>
          </div>
        {% endfor %}
      </div>
    </div>
  </div>
  
  <div style="height: 100px;"></div>
</div>


<div id="infoModal" class="modal" style="display:none;">
  <div class="modal-content" style="max-width:800px; margin:auto; position:relative; padding:20px;">
    <span class="close" onclick="closeInfoModal()" style="position:absolute; top:10px; right:15px; cursor:pointer; font-size:1.5em;">&times;</span>
    <h2>Image Info</h2>
    <div style="text-align:center; margin-bottom:20px;">
      <img id="infoImagePreview" src="" alt="Info Preview" style="max-width:300px;">
      <div style="margin-top:10px;">
        <button type="button" onclick="openCropModal()">Crop Image</button>
      </div>
    </div>
    <div style="display:flex; gap:20px;">
      
      <div style="flex:1;" id="infoLeftColumn">
        <p><strong>Filename:</strong> <span id="infoFilename">N/A</span></p>
        <p><strong>Resolution:</strong> <span id="infoResolution">N/A</span></p>
        <p><strong>Filesize:</strong> <span id="infoFilesize">N/A</span></p>
      </div>
      
      <div style="flex:1;">
        <div style="margin-bottom:10px;">
          <label><strong>Tags:</strong></label>
          <div id="tagContainer" style="margin-top:5px; margin-bottom:10px;"></div>
          <div style="display:flex;">
            <input type="text" id="newTagInput" style="flex-grow:1;" placeholder="Add a new tag...">
            <button type="button" onclick="addTag()" style="margin-left:5px;">Add</button>
          </div>
          <input type="hidden" id="infoTags">
        </div>
        <div style="margin-bottom:10px;">
          <label><strong>Favorite:</strong></label>
          <input type="checkbox" id="infoFavorite">
        </div>
        <div id="infoStatus" style="color: green; margin-bottom:10px;"></div>
        <button onclick="saveInfoEdits()">Save</button>
        <button onclick="runOpenClip()">Re-run Tagging</button>
      </div>
    </div>
  </div>
</div>


<div id="lightboxModal" class="modal lightbox-modal" style="display:none;">
  <span class="lightbox-close" onclick="closeLightbox()">&times;</span>
  <img class="lightbox-content" id="lightboxImage" alt="Enlarged Image">
  <div id="lightboxCaption"></div>
</div>


<div id="cropModal" class="modal" style="display:none;">
  <div class="modal-content">
    <span class="close" onclick="closeCropModal()" style="cursor:pointer; font-size:1.5em;">&times;</span>
    <h2>Crop Image</h2>
    <div id="cropContainer" style="max-width:100%; max-height:80vh;">
      <img id="cropImage" src="" alt="Crop Image" style="width:100%;">
    </div>
    <div style="margin-top:10px;">
      <button type="button" onclick="saveCropData()">Save Crop</button>
      <button type="button" onclick="closeCropModal()">Cancel</button>
    </div>
  </div>
</div>
{% endblock %}

{% block scripts %}
<script>
document.addEventListener("DOMContentLoaded", function() {
  // Inject dynamic CSS for info button, favorite icon, and tag boxes
  const styleTag = document.createElement('style');
  styleTag.innerHTML = `
    .info-button {
      position: absolute;
      left: 50%;
      bottom: 10px;
      transform: translateX(-50%);
      background: #17a2b8;
      color: #fff;
      border: none;
      padding: 8px 12px;
      border-radius: 4px;
      cursor: pointer;
      transition: background 0.3s ease;
      font-size: 0.9em;
    }
    .info-button:hover {
      background: #138496;
    }
    .favorite-icon i {
      font-size: 1.5em;
      color: #ccc;
      transition: color 0.3s;
    }
    .favorite-icon.favorited i {
      color: red;
    }
    .tag-box {
      display: inline-block;
      background-color: #e9ecef;
      border-radius: 4px;
      padding: 5px 10px;
      margin: 3px;
      font-size: 0.9em;
    }
    .tag-remove {
      margin-left: 5px;
      cursor: pointer;
      font-weight: bold;
      color: #dc3545;
    }
    .tag-remove:hover {
      color: #bd2130;
    }
  `;
  document.head.appendChild(styleTag);
});

/* Lightbox functions */
function openLightbox(src, alt) {
  const lightboxModal = document.getElementById('lightboxModal');
  const lightboxImage = document.getElementById('lightboxImage');
  const lightboxCaption = document.getElementById('lightboxCaption');
  lightboxModal.style.display = 'block';
  lightboxImage.src = src;
  lightboxCaption.innerText = alt;
}
function closeLightbox() {
  document.getElementById('lightboxModal').style.display = 'none';
}

/* Debounce helper */
function debounce(func, wait) {
  let timeout;
  return function(...args) {
    clearTimeout(timeout);
    timeout = setTimeout(() => func.apply(this, args), wait);
  };
}

/* Gallery search */
const searchInput = document.getElementById('gallerySearch');
const searchSpinner = document.getElementById('searchSpinner');
const gallery = document.getElementById('gallery');

const performSearch = debounce(function() {
  const query = searchInput.value.trim();
  if (!query) {
    searchSpinner.style.display = 'none';
    location.reload();
    return;
  }
  searchSpinner.style.display = 'block';
  fetch(`/api/search_images?q=${encodeURIComponent(query)}`)
    .then(response => response.json())
    .then(data => {
      searchSpinner.style.display = 'none';
      gallery.innerHTML = "";
      if (data.status === "success" && data.results.ids) {
        if (data.results.ids.length === 0) {
          gallery.innerHTML = "<p>No matching images found.</p>";
        } else {
          data.results.ids.forEach((id) => {
            const imageUrl = `/images/${encodeURIComponent(id)}`;
            const item = document.createElement('div');
            item.className = 'gallery-item';
            item.innerHTML = `
              <div class="img-container">
                <img src="${imageUrl}" alt="${id}" data-filename="${id}" loading="lazy">
                <div class="overlay">
                  <div class="favorite-icon" title="Favorite" data-image="${id}">
                    <i class="fa fa-heart"></i>
                  </div>
                  <button class="send-button" data-image="${id}">Send</button>
                  <button class="info-button" data-image="${id}">Info</button>
                  <div class="delete-icon" title="Delete" data-image="${id}">
                    <i class="fa fa-trash"></i>
                  </div>
                </div>
              </div>
            `;
            gallery.appendChild(item);
          });
        }
      } else {
        gallery.innerHTML = "<p>No matching images found.</p>";
      }
    })
    .catch(err => {
      searchSpinner.style.display = 'none';
      console.error("Search error:", err);
    });
}, 500);

if (searchInput) {
  searchInput.addEventListener('input', performSearch);
}

/* Upload form */
const uploadForm = document.getElementById('uploadForm');
uploadForm.addEventListener('submit', function(e) {
  e.preventDefault();
  const fileInput = document.getElementById('fileInput');
  if (!fileInput.files.length) return;
  
  const formData = new FormData();
  for (let i = 0; i < fileInput.files.length; i++) {
    formData.append('file', fileInput.files[i]);
  }
  
  const selectedDevice = document.querySelector('input[name="device"]:checked');
  const deviceFriendly = selectedDevice ? selectedDevice.getAttribute('data-friendly') : "unknown display";
  
  const xhr = new XMLHttpRequest();
  xhr.open('POST', uploadForm.action, true);

  xhr.upload.addEventListener("progress", function(e) {
    if (e.lengthComputable) {
      const percentComplete = (e.loaded / e.total) * 100;
      const progressBar = document.getElementById('progressBar');
      progressBar.style.width = percentComplete + '%';
      progressBar.textContent = Math.round(percentComplete) + '%';
      document.getElementById('progressContainer').style.display = 'block';
      
      const popup = document.getElementById('uploadPopup');
      popup.style.display = 'block';
      popup.innerHTML = `<div class="spinner"></div> Uploading image to ${deviceFriendly}... ${Math.round(percentComplete)}%`;
    }
  });

  xhr.onload = function() {
    const popup = document.getElementById('uploadPopup');
    if (xhr.status === 200) {
      popup.innerHTML = `<div class="spinner"></div> Image uploaded successfully!`;
    } else {
      popup.innerHTML = `<div class="spinner"></div> Error uploading image.`;
    }
    setTimeout(() => {
      popup.style.display = 'none';
      location.reload();
    }, 1500);
  };

  xhr.onerror = function() {
    const popup = document.getElementById('uploadPopup');
    popup.innerHTML = `<div class="spinner"></div> Error uploading image.`;
    setTimeout(() => {
      popup.style.display = 'none';
    }, 1500);
  };

  xhr.send(formData);
});

/* Send image */
document.addEventListener('click', function(e) {
  if (e.target && e.target.classList.contains('send-button')) {
    e.stopPropagation();
    const imageFilename = e.target.getAttribute('data-image');
    const selectedDevice = document.querySelector('input[name="device"]:checked');
    if (!selectedDevice) return;
    
    const deviceFriendly = selectedDevice.getAttribute('data-friendly');
    const formData = new FormData();
    formData.append("device", selectedDevice.value);

    const baseUrl = "{{ url_for('image.send_image', filename='') }}";
    const finalUrl = baseUrl + encodeURIComponent(imageFilename);

    const xhr = new XMLHttpRequest();
    xhr.open('POST', finalUrl, true);

    xhr.upload.addEventListener("progress", function(ev) {
      if (ev.lengthComputable) {
        const percentComplete = (ev.loaded / ev.total) * 100;
        const popup = document.getElementById('uploadPopup');
        popup.style.display = 'block';
        popup.innerHTML = `<div class="spinner"></div> Sending image to ${deviceFriendly}... ${Math.round(percentComplete)}%`;
      }
    });

    xhr.onload = function() {
      const popup = document.getElementById('uploadPopup');
      if (xhr.status === 200) {
        popup.innerHTML = `<div class="spinner"></div> Image sent successfully!`;
      } else {
        popup.innerHTML = `<div class="spinner"></div> Error sending image.`;
      }
      setTimeout(() => {
        popup.style.display = 'none';
        location.reload();
      }, 1500);
    };

    xhr.onerror = function() {
      const popup = document.getElementById('uploadPopup');
      popup.innerHTML = `<div class="spinner"></div> Error sending image.`;
      setTimeout(() => {
        popup.style.display = 'none';
      }, 1500);
    };

    xhr.send(formData);
  }
});

/* Delete image */
document.addEventListener('click', function(e) {
  if (e.target && e.target.closest('.delete-icon')) {
    e.stopPropagation();
    const imageFilename = e.target.closest('.delete-icon').getAttribute('data-image');
    
    const deleteBaseUrl = "/delete_image/";
    const deleteUrl = deleteBaseUrl + encodeURIComponent(imageFilename);

    fetch(deleteUrl, { method: 'POST' })
      .then(response => response.json())
      .then(data => {
        if (data.status === "success") {
          location.reload();
        } else {
          console.error("Error deleting image:", data.message);
        }
      })
      .catch(error => {
        console.error("Error deleting image:", error);
      });
  }
});

/* Favorite toggle */
document.addEventListener('click', function(e) {
  if (e.target && e.target.closest('.favorite-icon')) {
    e.stopPropagation();
    const favIcon = e.target.closest('.favorite-icon');
    const imageFilename = favIcon.getAttribute('data-image');
    favIcon.classList.toggle('favorited');
    const isFavorited = favIcon.classList.contains('favorited');
    fetch("/api/update_image_metadata", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        filename: imageFilename,
        tags: [],  // do not modify tags in favorite toggle
        favorite: isFavorited
      })
    })
      .then(resp => resp.json())
      .then(data => {
        if (data.status !== "success") {
          console.error("Error updating favorite:", data.message);
        }
      })
      .catch(err => {
        console.error("Error updating favorite:", err);
      });
  }
});

/* Info Modal Logic */
let currentInfoFilename = null;

document.addEventListener('click', function(e) {
  if (e.target && e.target.classList.contains('info-button')) {
    e.stopPropagation();
    const filename = e.target.getAttribute('data-image');
    currentInfoFilename = filename;
    openInfoModal(filename);
  }
});

// Tag management functions
let currentTags = [];

function renderTags() {
  const tagContainer = document.getElementById('tagContainer');
  tagContainer.innerHTML = '';
  
  currentTags.forEach((tag, index) => {
    const tagElement = document.createElement('span');
    tagElement.className = 'tag-box';
    tagElement.innerHTML = `${tag} <span class="tag-remove" onclick="removeTag(${index})">×</span>`;
    tagContainer.appendChild(tagElement);
  });
  
  // Update the hidden input with comma-separated tags
  document.getElementById('infoTags').value = currentTags.join(', ');
}

function addTag() {
  const newTagInput = document.getElementById('newTagInput');
  const tag = newTagInput.value.trim();
  
  if (tag && !currentTags.includes(tag)) {
    currentTags.push(tag);
    renderTags();
    newTagInput.value = '';
  }
}

function removeTag(index) {
  currentTags.splice(index, 1);
  renderTags();
}

// Add event listener for Enter key on the new tag input
document.addEventListener('DOMContentLoaded', function() {
  const newTagInput = document.getElementById('newTagInput');
  if (newTagInput) {
    newTagInput.addEventListener('keypress', function(e) {
      if (e.key === 'Enter') {
        e.preventDefault();
        addTag();
      }
    });
  }
});

function openInfoModal(filename) {
  const imgUrl = `/images/${encodeURIComponent(filename)}?size=info`;
  fetch(`/api/get_image_metadata?filename=${encodeURIComponent(filename)}`)
    .then(resp => resp.json())
    .then(data => {
      if (data.status === "success") {
        document.getElementById('infoImagePreview').src = imgUrl;
        document.getElementById('infoFilename').textContent = filename;
        document.getElementById('infoResolution').textContent = data.resolution || "N/A";
        document.getElementById('infoFilesize').textContent = data.filesize || "N/A";
        
        // Set up tags
        currentTags = data.tags || [];
        renderTags();
        
        document.getElementById('infoFavorite').checked = data.favorite || false;
        document.getElementById('infoStatus').textContent = "";
        document.getElementById('infoModal').style.display = 'block';
      } else {
        document.getElementById('infoStatus').textContent = "Error: " + data.message;
        document.getElementById('infoModal').style.display = 'block';
      }
    })
    .catch(err => {
      console.error("Error fetching metadata:", err);
      document.getElementById('infoStatus').textContent = "Failed to fetch metadata. Check console.";
      document.getElementById('infoModal').style.display = 'block';
    });
}

function closeInfoModal() {
  document.getElementById('infoModal').style.display = 'none';
  currentInfoFilename = null;
}

function saveInfoEdits() {
  if (!currentInfoFilename) return;
  fetch("/api/update_image_metadata", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      filename: currentInfoFilename,
      tags: currentTags,
      favorite: document.getElementById('infoFavorite').checked
    })
  })
    .then(resp => resp.json())
    .then(data => {
      if (data.status === "success") {
        document.getElementById('infoStatus').textContent = "Metadata updated successfully!";
        setTimeout(() => { closeInfoModal(); }, 1500);
      } else {
        document.getElementById('infoStatus').textContent = "Error updating metadata: " + data.message;
      }
    })
    .catch(err => {
      console.error("Error updating metadata:", err);
      document.getElementById('infoStatus').textContent = "Failed to update metadata. Check console.";
    });
}

function runOpenClip() {
  if (!currentInfoFilename) return;
  fetch(`/api/reembed_image?filename=${encodeURIComponent(currentInfoFilename)}`)
    .then(resp => resp.json())
    .then(data => {
      if (data.status === "success") {
        currentTags = data.tags || [];
        renderTags();
        document.getElementById('infoStatus').textContent = "Re-ran tagging successfully!";
      } else {
        document.getElementById('infoStatus').textContent = "Error re-running tagging: " + data.message;
      }
    })
    .catch(err => {
      console.error("Error re-running tagging:", err);
      document.getElementById('infoStatus').textContent = "Failed to re-run tagging. Check console.";
    });
}

// Crop Modal Functions
let cropperInstance = null;

function openCropModal() {
  if (!currentInfoFilename) return;
  
  const cropModal = document.getElementById('cropModal');
  const cropImage = document.getElementById('cropImage');
  
  // Set the image source to the current image
  cropImage.src = `/images/${encodeURIComponent(currentInfoFilename)}`;
  
  // Show the modal
  cropModal.style.display = 'block';
  
  // Initialize Cropper.js after the image is loaded
  cropImage.onload = function() {
    if (cropperInstance) {
      cropperInstance.destroy();
    }
    
    // Import Cropper.js dynamically if needed
    if (typeof Cropper === 'undefined') {
      const script = document.createElement('script');
      script.src = 'https://cdnjs.cloudflare.com/ajax/libs/cropperjs/1.5.12/cropper.min.js';
      document.head.appendChild(script);
      
      const link = document.createElement('link');
      link.rel = 'stylesheet';
      link.href = 'https://cdnjs.cloudflare.com/ajax/libs/cropperjs/1.5.12/cropper.min.css';
      document.head.appendChild(link);
      
      script.onload = initCropper;
    } else {
      initCropper();
    }
  };
}

function initCropper() {
  const cropImage = document.getElementById('cropImage');
  
  // Get the selected device's resolution
  const selectedDevice = document.querySelector('input[name="device"]:checked');
  let aspectRatio = NaN; // Default to free aspect ratio
  let targetWidth = 0;
  let targetHeight = 0;
  let isPortrait = false;
  
  if (selectedDevice) {
    const resolution = selectedDevice.getAttribute('data-resolution');
    if (resolution) {
      const parts = resolution.split('x');
      if (parts.length === 2) {
        targetWidth = parseInt(parts[0], 10);
        targetHeight = parseInt(parts[1], 10);
        
        // Check if the device is in portrait orientation
        isPortrait = selectedDevice.parentNode.textContent.trim().toLowerCase().includes('portrait');
        
        if (isPortrait) {
          // For portrait orientation, swap width and height for aspect ratio
          aspectRatio = targetHeight / targetWidth;
        } else {
          aspectRatio = targetWidth / targetHeight;
        }
      }
    }
  }
  
  // Initialize cropper with the calculated aspect ratio
  cropperInstance = new Cropper(cropImage, {
    aspectRatio: aspectRatio, // Use the calculated aspect ratio
    viewMode: 1,
    autoCropArea: 1, // Start with maximum possible area
    responsive: true,
    restore: true,
    guides: true,
    center: true,
    highlight: true,
    cropBoxMovable: true,
    cropBoxResizable: true,
    toggleDragModeOnDblclick: true,
    ready: function() {
      // This function runs when the cropper is fully initialized
      if (aspectRatio && cropperInstance) {
        // Get the image dimensions
        const imageData = cropperInstance.getImageData();
        const imageWidth = imageData.naturalWidth;
        const imageHeight = imageData.naturalHeight;
        
        // Calculate the optimal crop box dimensions to cover as much of the image as possible
        // while maintaining the target aspect ratio
        let cropBoxWidth, cropBoxHeight;
        
        const imageRatio = imageWidth / imageHeight;
        
        if (aspectRatio > imageRatio) {
          // If target aspect ratio is wider than the image, use full width
          cropBoxWidth = imageWidth;
          cropBoxHeight = cropBoxWidth / aspectRatio;
        } else {
          // If target aspect ratio is taller than the image, use full height
          cropBoxHeight = imageHeight;
          cropBoxWidth = cropBoxHeight * aspectRatio;
        }
        
        // Calculate the position to center the crop box
        const left = (imageWidth - cropBoxWidth) / 2;
        const top = (imageHeight - cropBoxHeight) / 2;
        
        // Set the crop box
        cropperInstance.setCropBoxData({
          left: left,
          top: top,
          width: cropBoxWidth,
          height: cropBoxHeight
        });
      }
    }
  });
}

function closeCropModal() {
  const cropModal = document.getElementById('cropModal');
  cropModal.style.display = 'none';
  
  if (cropperInstance) {
    cropperInstance.destroy();
    cropperInstance = null;
  }
}

function saveCropData() {
  if (!cropperInstance || !currentInfoFilename) return;
  
  const cropData = cropperInstance.getData();
  const selectedDevice = document.querySelector('input[name="device"]:checked');
  const deviceAddress = selectedDevice ? selectedDevice.value : null;
  
  fetch(`/save_crop_info/${encodeURIComponent(currentInfoFilename)}`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({
      x: cropData.x,
      y: cropData.y,
      width: cropData.width,
      height: cropData.height,
      device: deviceAddress
    })
  })
  .then(response => response.json())
  .then(data => {
    if (data.status === 'success') {
      document.getElementById('infoStatus').textContent = 'Crop data saved successfully!';
      closeCropModal();
    } else {
      document.getElementById('infoStatus').textContent = 'Error saving crop data: ' + data.message;
    }
  })
  .catch(error => {
    console.error('Error saving crop data:', error);
    document.getElementById('infoStatus').textContent = 'Error saving crop data. Check console.';
  });
}

const prevButton = document.getElementById('prevDevice');
const nextButton = document.getElementById('nextDevice');
// Define devices array using template data
const devices = [
  {% for device in devices %}
    {
      "friendly_name": "{{ device.friendly_name|e }}",
      "address": "{{ device.address|e }}",
      "last_sent": "{{ device.last_sent|e if device.last_sent is defined else '' }}"
    }{% if not loop.last %},{% endif %}
  {% endfor %}
];
let currentDeviceIndex = 0;

function updateCurrentImageDisplay() {
  const device = devices[currentDeviceIndex];
  const titleEl = document.getElementById('currentImageTitle');
  const imageEl = document.getElementById('currentImage');
  const placeholderEl = document.getElementById('currentImagePlaceholder');
  
  titleEl.textContent = "Current image on " + device.friendly_name;
  if (device.last_sent) {
    if (placeholderEl) {
      placeholderEl.style.display = 'none';
    }
    if (imageEl) {
      imageEl.src = "{{ url_for('image.uploaded_file', filename='') }}" + device.last_sent;
      imageEl.style.display = 'block';
    }
  } else {
    if (imageEl) {
      imageEl.style.display = 'none';
    }
    if (placeholderEl) {
      placeholderEl.style.display = 'block';
    }
  }
}

if (prevButton && nextButton) {
  prevButton.addEventListener('click', function() {
    currentDeviceIndex = (currentDeviceIndex - 1 + devices.length) % devices.length;
    updateCurrentImageDisplay();
  });
  nextButton.addEventListener('click', function() {
    currentDeviceIndex = (currentDeviceIndex + 1) % devices.length;
    updateCurrentImageDisplay();
  });
}

if (devices.length > 0) {
  updateCurrentImageDisplay();
}

// Bulk tagging moved to settings page
</script>
{% endblock %}
```


## templates/schedule.html

```html
{% extends "base.html" %}
{% block title %}Schedule - InkyDocker{% endblock %}
{% block head %}
  
  <link href="https://cdn.jsdelivr.net/npm/fullcalendar@6.1.8/main.min.css" rel="stylesheet">
  
  <script src="https://cdn.jsdelivr.net/npm/fullcalendar@6.1.8/index.global.min.js" defer></script>
  <style>
    #calendar {
      max-width: 1000px;
      margin: 40px auto;
      margin-bottom: 100px; /* Add more space at the bottom */
    }
    
    /* Style for the event content to show thumbnails */
    .event-thumbnail {
      width: 100%;
      height: 40px;
      object-fit: cover;
      border-radius: 3px;
      margin-bottom: 2px;
    }
    
    /* Search bar styles */
    .search-container {
      margin-bottom: 15px;
    }
    
    #imageSearch {
      width: 100%;
      padding: 8px;
      border: 1px solid #ddd;
      border-radius: 4px;
      margin-bottom: 10px;
    }
    
    /* Improve gallery layout */
    .gallery {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
      gap: 10px;
      max-height: 500px;
      overflow-y: auto;
    }
    
    .gallery-item {
      height: 150px !important;
      position: relative;
      border: 1px solid #ddd;
      border-radius: 4px;
      overflow: hidden;
    }
    
    .gallery-item img {
      width: 100%;
      height: 100%;
      object-fit: cover;
      cursor: pointer;
      transition: transform 0.2s;
    }
    
    .gallery-item img:hover {
      transform: scale(1.05);
    }
    
    .img-container {
      height: 100%;
    }
    
    /* Tags display */
    .image-tags {
      position: absolute;
      bottom: 0;
      left: 0;
      right: 0;
      background: rgba(0,0,0,0.6);
      color: white;
      padding: 3px;
      font-size: 10px;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    /* Modal styles for event creation and image gallery (existing) */
    .modal {
      display: none;
      position: fixed;
      z-index: 10000;
      left: 0;
      top: 0;
      width: 100%;
      height: 100%;
      overflow: auto;
      background-color: rgba(0,0,0,0.7);
      padding: 20px;
    }
    .modal-content {
      background: #fff;
      margin: 10% auto;
      padding: 20px;
      border-radius: 8px;
      max-width: 500px;
      position: relative;
    }
    .close {
      position: absolute;
      top: 10px;
      right: 15px;
      font-size: 1.5em;
      cursor: pointer;
    }
    /* Deletion modal for recurring events */
    #deleteModal .modal-content {
      max-width: 400px;
      text-align: center;
    }
    #deleteModal button {
      margin: 5px;
    }
    
    /* Styling for recurring events */
    .recurring-event {
      border-left: 4px dashed #fff !important;  /* Dashed border to indicate recurring */
      border-right: 4px dashed #fff !important;
    }
    
    .recurring-event:before {
      content: "↻";  /* Recurrence symbol */
      position: absolute;
      top: 2px;
      left: 2px;
      font-weight: bold;
      color: white;
      background-color: rgba(0, 0, 0, 0.5);
      border-radius: 50%;
      width: 16px;
      height: 16px;
      line-height: 16px;
      text-align: center;
      z-index: 100;
    }
  </style>
{% endblock %}
{% block content %}
<div class="container">
  <header class="page-header">
    <h1>Schedule Images</h1>
    <p>Manage your scheduled image updates with our interactive calendar.</p>
  </header>
  <div id="calendar"></div>
  
</div>


<div id="eventModal" class="modal">
  <div class="modal-content">
    <span class="close" onclick="closeEventModal()">&times;</span>
    <h2 id="eventModalTitle">Add Scheduled Event</h2>
    <form id="eventForm">
      <input type="hidden" id="editingEventId" value="">
      <div>
        <label for="eventDate">Date &amp; Time:</label>
        <input type="datetime-local" id="eventDate" name="eventDate" required>
      </div>
      <div>
        <label>Select eInk Display:</label>
        {% if devices %}
          {% for device in devices %}
            <label>
              <input type="radio" name="device" value="{{ device.address }}" {% if loop.first %}checked{% endif %}>
              {{ device.friendly_name }}
            </label>
          {% endfor %}
        {% else %}
          <p>No devices configured. Please add devices in <a href="{{ url_for('settings.settings') }}">Settings</a>.</p>
        {% endif %}
      </div>
      <div>
        <label for="recurrence">Recurrence:</label>
        <select id="recurrence" name="recurrence">
          <option value="none">None</option>
          <option value="daily">Daily</option>
          <option value="weekly">Weekly</option>
          <option value="monthly">Same date next month</option>
        </select>
      </div>
      <div>
        <label>Choose Image:</label>
        <button type="button" onclick="openImageGallery()">Select Image</button>
        <input type="hidden" id="selectedImage" name="selectedImage">
        <span id="selectedImageName"></span>
      </div>
      <div style="margin-top:10px;">
        <input type="submit" id="eventSubmitButton" value="Save Event">
      </div>
    </form>
  </div>
</div>


<div id="imageGalleryModal" class="modal">
  <div class="modal-content" style="max-width:800px;">
    <span class="close" onclick="closeImageGallery()">&times;</span>
    <h2>Select an Image</h2>
    
    
    <div class="search-container">
      <input type="text" id="imageSearch" placeholder="Search by tags...">
    </div>
    
    <div class="gallery" id="galleryModal">
      {% for image in images %}
        <div class="gallery-item" data-tags="{{ image_tags.get(image, '') }}">
          <div class="img-container">
            <img src="{{ url_for('image.thumbnail', filename=image) }}" alt="{{ image }}" data-filename="{{ image }}" onclick="selectImage('{{ image }}', this.src)">
            {% if image_tags.get(image) %}
              <div class="image-tags">{{ image_tags.get(image) }}</div>
            {% endif %}
          </div>
        </div>
      {% endfor %}
    </div>
  </div>
</div>


<div id="deleteModal" class="modal">
  <div class="modal-content">
    <span class="close" onclick="closeDeleteModal()">&times;</span>
    <h3>Delete Recurring Event</h3>
    <p>Delete this occurrence or the entire series?</p>
    <button id="deleteOccurrenceBtn" class="btn btn-danger">Delete this occurrence</button>
    <button id="deleteSeriesBtn" class="btn btn-danger">Delete entire series</button>
    <button onclick="closeDeleteModal()" class="btn btn-secondary">Cancel</button>
  </div>
</div>
{% endblock %}
{% block scripts %}
  <script>
    var currentDeleteEventId = null; // store event id for deletion modal

    // FIXED: Store the calendar instance globally for easy access
    // This allows us to call calendar.refetchEvents() from anywhere in the code
    var calendar;
    
    document.addEventListener('DOMContentLoaded', function() {
      var calendarEl = document.getElementById('calendar');
      if (!calendarEl) return;
      if (typeof FullCalendar === 'undefined') {
        console.error("FullCalendar is not defined");
        return;
      }
      // Initialize the global calendar variable
      calendar = new FullCalendar.Calendar(calendarEl, {
        initialView: 'timeGridWeek',
        firstDay: 1,
        nowIndicator: true,
        editable: true,
        headerToolbar: {
          left: 'prev,next today refresh',
          center: 'title',
          right: 'timeGridWeek,timeGridDay'
        },
        events: '/schedule/events',
        customButtons: {
          refresh: {
            text: '↻',
            click: function() {
              // Force a full refresh of events from the server
              calendar.refetchEvents();
            }
          }
        },
        eventDrop: function(info) {
          var newDate = info.event.start;
          
          // FIXED: Convert to ISO string for consistent timezone handling
          // This ensures the datetime is properly formatted when sent to the server
          var isoString = newDate.toISOString();
          console.log("ISO date string:", isoString);
          
          // Get timezone offset in minutes
          // getTimezoneOffset() returns positive minutes for times behind UTC
          // and negative minutes for times ahead of UTC
          var timezoneOffset = newDate.getTimezoneOffset();
          console.log("Timezone offset:", timezoneOffset, "minutes");
          
          fetch("/schedule/update", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              event_id: info.event.id,
              datetime: isoString,
              timezone_offset: timezoneOffset
            })
          })
          .then(response => response.json())
          .then(data => {
            if(data.status !== "success"){
              alert("Error updating event: " + data.message);
              // Revert the drag if there was an error
              info.revert();
            } else {
              console.log("Event successfully updated:", info.event.id);
              // FIXED: Use the global calendar instance to refresh events
              // This ensures the calendar is immediately updated after an event is dragged
              if (calendar) {
                calendar.refetchEvents();
              }
            }
          })
          .catch(err => {
            console.error("Error updating event:", err);
            // Revert the drag if there was an error
            info.revert();
          });
        },
        eventDidMount: function(info) {
          // Add special styling for recurring events
          if (info.event.extendedProps.isRecurring) {
            info.el.classList.add('recurring-event');
          }
          
          // Add delete button with improved visibility
          var deleteEl = document.createElement('span');
          deleteEl.innerHTML = '&times;';
          deleteEl.style.position = 'absolute';
          deleteEl.style.top = '2px';
          deleteEl.style.right = '2px';
          deleteEl.style.color = 'white';  // White text for better contrast
          deleteEl.style.backgroundColor = 'rgba(0, 0, 0, 0.6)';  // Semi-transparent black background
          deleteEl.style.borderRadius = '50%';  // Circular background
          deleteEl.style.width = '16px';
          deleteEl.style.height = '16px';
          deleteEl.style.lineHeight = '14px';
          deleteEl.style.textAlign = 'center';
          deleteEl.style.cursor = 'pointer';
          deleteEl.style.fontWeight = 'bold';
          deleteEl.style.zIndex = '100';
          deleteEl.style.border = '1px solid white';  // White border for additional contrast
          info.el.appendChild(deleteEl);
          
          // Add thumbnail to event
          if (info.event.extendedProps.thumbnail) {
            var thumbnailEl = document.createElement('img');
            thumbnailEl.src = info.event.extendedProps.thumbnail;
            thumbnailEl.className = 'event-thumbnail';
            thumbnailEl.alt = info.event.title;
            thumbnailEl.style.cursor = 'pointer';
            
            // Add click event to open the event for editing
            thumbnailEl.addEventListener('click', function(e) {
              e.stopPropagation();
              // Get event data
              var eventData = {
                id: info.event.id,
                title: info.event.title,
                start: info.event.start,
                device: info.event.extendedProps.device,
                filename: info.event.extendedProps.filename,
                recurrence: info.event.extendedProps.recurrence
              };
              
              // Populate the event form with this data - use local timezone
              var startDate = eventData.start;
              var year = startDate.getFullYear();
              var month = String(startDate.getMonth() + 1).padStart(2, '0');
              var day = String(startDate.getDate()).padStart(2, '0');
              var hours = String(startDate.getHours()).padStart(2, '0');
              var minutes = String(startDate.getMinutes()).padStart(2, '0');
              
              var localDateStr = `${year}-${month}-${day}T${hours}:${minutes}`;
              document.getElementById('eventDate').value = localDateStr;
              console.log("Event edit local time:", localDateStr);
              
              // Select the correct device radio button
              var deviceRadios = document.querySelectorAll('input[name="device"]');
              for (var i = 0; i < deviceRadios.length; i++) {
                if (deviceRadios[i].value === eventData.device) {
                  deviceRadios[i].checked = true;
                  break;
                }
              }
              
              // Set the recurrence dropdown
              document.getElementById('recurrence').value = eventData.recurrence || 'none';
              
              // Set the selected image
              document.getElementById('selectedImage').value = eventData.filename;
              document.getElementById('selectedImageName').textContent = eventData.filename;
              
              // Set the editing event ID
              document.getElementById('editingEventId').value = eventData.id;
              
              // Update modal title and button text
              document.getElementById('eventModalTitle').textContent = 'Edit Scheduled Event';
              document.getElementById('eventSubmitButton').value = 'Update Event';
              
              // Open the event modal
              openEventModal();
            });
            
            // Find the event title element and insert the thumbnail before it
            var titleEl = info.el.querySelector('.fc-event-title');
            if (titleEl && titleEl.parentNode) {
              titleEl.parentNode.insertBefore(thumbnailEl, titleEl);
              
              // Add device name to title
              if (info.event.extendedProps.deviceName) {
                titleEl.textContent = info.event.title + ' on ' + info.event.extendedProps.deviceName;
              }
            }
          }
          
          // Add click handler for delete button
          deleteEl.addEventListener('click', function(e) {
            e.stopPropagation();
            // If recurring, show custom deletion modal; otherwise, delete directly.
            if(info.event.extendedProps.recurrence && info.event.extendedProps.recurrence.toLowerCase() !== "none"){
              currentDeleteEventId = info.event.id;
              openDeleteModal();
            } else {
              // Directly delete non-recurring event without popup.
              fetch("/schedule/remove/" + info.event.id, { method: "POST" })
              .then(response => response.json())
              .then(data => {
                if(data.status === "success"){
                  info.event.remove();
                }
              })
              .catch(err => {
                console.error("Error deleting event:", err);
              });
            }
          });
        },
        dateClick: function(info) {
          // Create a date object from the clicked date
          var dtLocal = new Date(info.date);
          
          // Format the date in local timezone format (YYYY-MM-DDTHH:MM)
          var year = dtLocal.getFullYear();
          var month = String(dtLocal.getMonth() + 1).padStart(2, '0');
          var day = String(dtLocal.getDate()).padStart(2, '0');
          var hours = String(dtLocal.getHours()).padStart(2, '0');
          var minutes = String(dtLocal.getMinutes()).padStart(2, '0');
          
          var localDateStr = `${year}-${month}-${day}T${hours}:${minutes}`;
          console.log("Clicked date local time:", localDateStr);
          
          openNewEventModal(localDateStr);
        }
      });
      calendar.render();
      
      // Add event listener for page refresh
      window.addEventListener('beforeunload', function() {
        // Store a flag indicating that we're refreshing the page
        localStorage.setItem('calendarRefreshing', 'true');
      });
      
      // Check if we're coming back from a refresh
      if (localStorage.getItem('calendarRefreshing') === 'true') {
        // Clear the flag
        localStorage.removeItem('calendarRefreshing');
        // Force a refresh of the events
        setTimeout(function() {
          calendar.refetchEvents();
          console.log("Refreshed events after page reload");
        }, 500);
      }
    });

    function openEventModal() { document.getElementById('eventModal').style.display = 'block'; }
    function closeEventModal() { document.getElementById('eventModal').style.display = 'none'; }
    
    function openImageGallery() {
      document.getElementById('imageGalleryModal').style.display = 'block';
      // Clear search field when opening
      var searchField = document.getElementById('imageSearch');
      if (searchField) {
        searchField.value = '';
        searchField.focus();
        // Trigger search to show all images
        filterImages('');
      }
    }
    
    function closeImageGallery() { document.getElementById('imageGalleryModal').style.display = 'none'; }
    
    function selectImage(filename, src) {
      document.getElementById('selectedImage').value = filename;
      document.getElementById('selectedImageName').textContent = filename;
      // Also show the thumbnail
      var nameSpan = document.getElementById('selectedImageName');
      if (nameSpan) {
        nameSpan.innerHTML = `<img src="${src}" style="height:40px;margin-right:5px;vertical-align:middle;"> ${filename}`;
      }
      closeImageGallery();
    }
    
    function openDeleteModal() { document.getElementById('deleteModal').style.display = 'block'; }
    function closeDeleteModal() { document.getElementById('deleteModal').style.display = 'none'; }
    
    // Function to filter images based on search input
    function filterImages(searchText) {
      searchText = searchText.toLowerCase();
      var items = document.querySelectorAll('#galleryModal .gallery-item');
      
      items.forEach(function(item) {
        var tags = item.getAttribute('data-tags') || '';
        var filename = item.querySelector('img').getAttribute('data-filename') || '';
        
        if (tags.toLowerCase().includes(searchText) || filename.toLowerCase().includes(searchText) || searchText === '') {
          item.style.display = '';
        } else {
          item.style.display = 'none';
        }
      });
    }
    
    // Add event listener for search input
    document.addEventListener('DOMContentLoaded', function() {
      var searchInput = document.getElementById('imageSearch');
      if (searchInput) {
        searchInput.addEventListener('input', function() {
          filterImages(this.value);
        });
      }
    });

    document.getElementById('deleteOccurrenceBtn').addEventListener('click', function() {
      // Skip this occurrence for recurring event.
      fetch("/schedule/skip/" + currentDeleteEventId, { method: "POST" })
      .then(response => response.json())
      .then(data => {
        if(data.status === "success"){
          // Remove the occurrence from the calendar.
          // Use the global calendar instance to refresh events
          if (calendar) {
            calendar.refetchEvents();
          } else {
            location.reload(); // Fallback if calendar instance is not available
          }
        }
      })
      .catch(err => {
        console.error("Error skipping occurrence:", err);
      });
      closeDeleteModal();
    });

    document.getElementById('deleteSeriesBtn').addEventListener('click', function() {
      // Delete the entire series.
      fetch("/schedule/remove/" + currentDeleteEventId, { method: "POST" })
      .then(response => response.json())
      .then(data => {
        if(data.status === "success"){
          // Use the global calendar instance to refresh events
          if (calendar) {
            calendar.refetchEvents();
          } else {
            location.reload(); // Fallback if calendar instance is not available
          }
        }
      })
      .catch(err => {
        console.error("Error deleting series:", err);
      });
      closeDeleteModal();
    });

    // Reset the event form when opening for a new event
    function resetEventForm() {
      document.getElementById('eventForm').reset();
      document.getElementById('selectedImage').value = '';
      document.getElementById('selectedImageName').textContent = '';
      document.getElementById('editingEventId').value = '';
      document.getElementById('eventModalTitle').textContent = 'Add Scheduled Event';
      document.getElementById('eventSubmitButton').value = 'Save Event';
    }
    
    // When clicking on a date, reset the form for a new event
    function openNewEventModal(date) {
      resetEventForm();
      if (date) {
        document.getElementById('eventDate').value = date;
      }
      openEventModal();
    }
    
    // Update the dateClick handler to use the new function
    document.addEventListener('DOMContentLoaded', function() {
      // Existing code will still run, this just adds additional functionality
      var calendarEl = document.getElementById('calendar');
      if (calendarEl && typeof FullCalendar !== 'undefined') {
        var existingCalendar = calendarEl._fullCalendar;
        if (existingCalendar) {
          existingCalendar.setOption('dateClick', function(info) {
            var dtLocal = new Date(info.date);
            var isoStr = dtLocal.toISOString().substring(0,16);
            openNewEventModal(isoStr);
          });
        }
      }
    });
    
    // Handle form submission for both adding and updating events
    document.getElementById('eventForm').addEventListener('submit', function(e) {
      e.preventDefault();
      var datetime = document.getElementById('eventDate').value;
      var device = document.querySelector('input[name="device"]:checked').value;
      var recurrence = document.getElementById('recurrence').value;
      var filename = document.getElementById('selectedImage').value;
      var eventId = document.getElementById('editingEventId').value;
      
      if (!datetime || !device || !filename) {
        alert("Please fill in all fields and select an image.");
        return;
      }
      
      // Determine if we're adding a new event or updating an existing one
      var isUpdate = eventId !== '';
      var url = isUpdate ? "/schedule/update" : "/schedule/add";
      var requestData = {
        datetime: datetime,
        device: device,
        recurrence: recurrence,
        filename: filename,
        timezone_offset: new Date().getTimezoneOffset()  // Add timezone offset
      };
      
      // If updating, include the event ID
      if (isUpdate) {
        requestData.event_id = eventId;
      }
      
      // Show immediate feedback to the user
      closeEventModal();
      const feedbackEl = document.createElement('div');
      feedbackEl.style.position = 'fixed';
      feedbackEl.style.top = '20px';
      feedbackEl.style.left = '50%';
      feedbackEl.style.transform = 'translateX(-50%)';
      feedbackEl.style.padding = '10px 20px';
      feedbackEl.style.backgroundColor = '#4CAF50';
      feedbackEl.style.color = 'white';
      feedbackEl.style.borderRadius = '5px';
      feedbackEl.style.zIndex = '10000';
      feedbackEl.textContent = 'Saving event...';
      document.body.appendChild(feedbackEl);
      
      fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(requestData)
      })
      .then(response => response.json())
      .then(data => {
        if(data.status === "success"){
          // Update feedback message
          feedbackEl.textContent = 'Event saved successfully!';
          
          // Use the global calendar instance to refresh events
          if (calendar) {
            calendar.refetchEvents();
            console.log("Refreshed calendar events after saving");
          } else {
            console.warn("Global calendar instance not available for refresh");
          }
          
          // Remove feedback after 2 seconds
          setTimeout(() => {
            document.body.removeChild(feedbackEl);
          }, 2000);
        } else {
          feedbackEl.style.backgroundColor = '#F44336';
          feedbackEl.textContent = "Error: " + data.message;
          setTimeout(() => {
            document.body.removeChild(feedbackEl);
          }, 3000);
        }
      })
      .catch(err => {
        console.error("Error " + (isUpdate ? "updating" : "adding") + " event:", err);
        feedbackEl.style.backgroundColor = '#F44336';
        feedbackEl.textContent = "Error saving event. Please try again.";
        setTimeout(() => {
          document.body.removeChild(feedbackEl);
        }, 3000);
      });
    });
  </script>
{% endblock %}
```


## templates/settings.html

```html
{% extends "base.html" %}
{% block title %}Settings - InkyDocker{% endblock %}

{% block content %}
<div class="container">
  
  <header class="page-header">
    <h1>Settings</h1>
    <p>Manage your eInk displays and AI settings.</p>
  </header>

  
  <div class="card text-center">
    <button id="clipSettingsBtn" class="primary-btn">CLIP Model Settings</button>
  </div>

  
  <div id="clipSettingsModal" class="modal">
    <div class="modal-content">
      <span class="close" id="closeClipSettingsModal">&times;</span>
      <h2>CLIP Model Settings</h2>
      <form id="clipSettingsForm">
        
        <div class="form-group">
          <label for="clip_model">Select CLIP Model for Image Tagging:</label>
          <select id="clip_model" name="clip_model" class="form-select" data-current="{{ config.clip_model if config and config.clip_model }}">
            <option value="">-- Select a model --</option>
            <option value="ViT-B-32" {% if config and config.clip_model == 'ViT-B-32' %}selected{% endif %}>ViT-B-32 (Faster, less accurate)</option>
            <option value="ViT-B-16" {% if config and config.clip_model == 'ViT-B-16' %}selected{% endif %}>ViT-B-16 (Balanced)</option>
            <option value="ViT-L-14" {% if config and config.clip_model == 'ViT-L-14' %}selected{% endif %}>ViT-L-14 (Slower, more accurate)</option>
          </select>
          <button type="button" class="field-save-btn" onclick="saveClipModel()">Save Model</button>
        </div>
        
        
        <div id="modelDownloadContainer" style="margin-top: 15px; display: none;">
          <p>Downloading model: <span id="modelDownloadName"></span></p>
          <div class="progress-container" style="width: 100%; background: #ddd; border-radius: 5px;">
            <div id="modelDownloadProgress" class="progress-bar" style="width: 0%; height: 20px; background: #28a745; border-radius: 5px; color: #fff; text-align: center; line-height: 20px;">0%</div>
          </div>
        </div>
        
        <div style="margin-top: 20px;">
          <p>All models are pre-installed in the system.</p>
          <p>Larger models provide more accurate tagging but require more processing power and memory.</p>
        </div>
        
        
        <div style="margin-top: 20px; text-align: center;">
          <button type="button" class="primary-btn" onclick="rerunAllTagging()">Rerun Tagging on All Images</button>
        </div>
      </form>
    </div>
  </div>

  
  <div class="card text-center">
    <button id="addNewDisplayBtn" class="primary-btn">Add New Display</button>
  </div>

  
  <div id="addDisplayModal" class="modal">
    <div class="modal-content">
      <span class="close" id="closeAddDisplayModal">&times;</span>
      <h2>Add New Display</h2>
      <form id="addDisplayForm" method="POST" action="{{ url_for('settings.settings') }}">
        <input type="text" name="address" id="newAddress" placeholder="Device Address (e.g., 192.168.1.100)" required>
        <select name="orientation" id="newOrientation" required>
          <option value="portrait">Portrait</option>
          <option value="landscape">Landscape</option>
        </select>
        <input type="text" name="friendly_name" id="newFriendlyName" placeholder="Friendly Name" required>
        
        <input type="hidden" name="display_name" id="newDisplayName">
        <input type="hidden" name="resolution" id="newResolution">
        <input type="hidden" name="color" id="newColor">
        <div style="margin-top: 10px;">
          <button type="button" class="primary-btn" onclick="fetchDisplayInfo('new')">Fetch Display Info</button>
        </div>
        <div style="margin-top: 10px;">
          <button type="submit" class="primary-btn">Save</button>
          <button type="button" class="primary-btn" onclick="closeAddDisplayModal()">Cancel</button>
        </div>
      </form>
    </div>
  </div>

  
  <div id="editDisplayModal" class="modal">
    <div class="modal-content">
      <span class="close" id="closeEditDisplayModal">&times;</span>
      <h2>Edit Display</h2>
      <form id="editDisplayForm" method="POST" action="{{ url_for('settings.edit_device') }}">
        <input type="hidden" name="device_index" id="editDeviceIndex">
        <label for="editFriendlyName">Friendly Name:</label>
        <input type="text" name="friendly_name" id="editFriendlyName" placeholder="Friendly Name" required>
        <label for="editOrientation">Orientation:</label>
        <select name="orientation" id="editOrientation" required>
          <option value="portrait">Portrait</option>
          <option value="landscape">Landscape</option>
        </select>
        <label for="editAddress">Device Address:</label>
        <input type="text" name="address" id="editAddress" placeholder="Device Address" required>
        <div style="margin-top: 10px;">
          <button type="submit" class="primary-btn">Save Changes</button>
          <button type="button" class="primary-btn" onclick="closeEditDisplayModal()">Cancel</button>
        </div>
      </form>
    </div>
  </div>

  
  <div id="advancedActionsModal" class="modal">
    <div class="modal-content">
      <span class="close" id="closeAdvancedActionsModal">&times;</span>
      <h2>Advanced Actions</h2>
      <p id="advancedDeviceTitle" style="font-weight:bold;"></p>
      <div style="margin-top: 10px;">
        <button type="button" class="primary-btn" onclick="triggerSystemUpdate()">System Update & Reboot</button>
        <button type="button" class="primary-btn" onclick="triggerBackup()">Create Backup</button>
        <button type="button" class="primary-btn" onclick="triggerAppUpdate()">Update Application</button>
      </div>
      <div style="margin-top: 10px;">
        <button type="button" class="primary-btn" onclick="closeAdvancedActionsModal()">Close</button>
      </div>
    </div>
  </div>

  
  <div class="card">
    <h2>Existing Devices</h2>
    <table class="device-table">
      <thead>
        <tr>
          <th>#</th>
          <th>Color</th>
          <th>Friendly Name</th>
          <th>Orientation</th>
          <th>Address</th>
          <th>Display Name</th>
          <th>Resolution</th>
          <th>Status</th>
          <th>Action</th>
        </tr>
      </thead>
      <tbody>
        {% for device in devices %}
        <tr data-index="{{ loop.index0 }}" data-address="{{ device.address }}">
          <td>{{ loop.index }}</td>
          <td>
            <div style="width:20px; height:20px; border-radius:50%; background:{{ device.color }};"></div>
          </td>
          <td>{{ device.friendly_name }}</td>
          <td>{{ device.orientation }}</td>
          <td>{{ device.address }}</td>
          <td>{{ device.display_name }}</td>
          <td>{{ device.resolution }}</td>
          <td>
            {% if device.online %}
              <span style="color:green;">&#9679;</span>
            {% else %}
              <span style="color:red;">&#9679;</span>
            {% endif %}
          </td>
          <td>
            <form method="POST" action="{{ url_for('settings.delete_device', device_index=loop.index0) }}" style="display:inline;">
              <input type="submit" value="Delete">
            </form>
            <button type="button" class="edit-button" onclick="openEditModal('{{ loop.index0 }}', '{{ device.friendly_name }}', '{{ device.orientation }}', '{{ device.address }}')">
              Edit
            </button>
            <button type="button" class="advanced-button" onclick="openAdvancedModal('{{ loop.index0 }}', '{{ device.friendly_name }}')">
              Advanced
            </button>
          </td>
        </tr>
        {% endfor %}
      </tbody>
    </table>
  </div>
</div>
{% endblock %}

{% block scripts %}
<style>
  .modal {
    display: none;
    position: fixed;
    z-index: 3000;
    left: 0;
    top: 0;
    width: 100%;
    height: 100%;
    overflow: auto;
    background-color: rgba(0, 0, 0, 0.5);
  }
  .modal-content {
    background-color: #fff;
    margin: 10% auto;
    padding: 20px;
    border-radius: 5px;
    max-width: 500px;
    position: relative;
  }
  .close {
    color: #aaa;
    float: right;
    font-size: 28px;
    font-weight: bold;
    cursor: pointer;
  }
  .close:hover,
  .close:focus {
    color: #000;
  }
  .primary-btn {
    background: linear-gradient(to right, #28a745, #218838);
    border: none;
    color: #fff;
    padding: 10px 20px;
    border-radius: 5px;
    cursor: pointer;
    font-size: 1em;
  }
  .primary-btn:hover {
    background: linear-gradient(to right, #218838, #1e7e34);
  }
  .field-save-btn {
    margin-top: 5px;
    font-size: 0.9em;
    padding: 5px 10px;
    background-color: #007bff;
    color: #fff;
    border: none;
    border-radius: 3px;
    cursor: pointer;
  }
  .field-save-btn:hover {
    background-color: #0056b3;
  }
  .overlay-popup {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background-color: rgba(0, 0, 0, 0.7);
    display: flex;
    justify-content: center;
    align-items: center;
    z-index: 5000;
  }
  .overlay-content {
    background-color: #fff;
    padding: 20px;
    border-radius: 5px;
    max-width: 400px;
    text-align: center;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
  }
  .overlay-buttons {
    margin-top: 15px;
    display: flex;
    justify-content: center;
    gap: 10px;
  }
  .cancel-btn {
    background: #6c757d;
    border: none;
    color: #fff;
    padding: 8px 15px;
    border-radius: 5px;
    cursor: pointer;
  }
  .cancel-btn:hover {
    background: #5a6268;
  }
  .progress-container {
    margin: 10px 0;
    background-color: #f1f1f1;
    border-radius: 5px;
    overflow: hidden;
  }
  .progress-bar {
    height: 20px;
    background-color: #4CAF50;
    text-align: center;
    line-height: 20px;
    color: white;
    transition: width 0.3s ease;
  }
</style>

<script>
  // Global modal closing functions
  window.closeAddDisplayModal = function() {
    document.getElementById('addDisplayModal').style.display = 'none';
  };
  window.closeEditDisplayModal = function() {
    document.getElementById('editDisplayModal').style.display = 'none';
  };
  window.closeAdvancedActionsModal = function() {
    document.getElementById('advancedActionsModal').style.display = 'none';
  };
  
  // Overlay message functions
  function showOverlayMessage(message, duration) {
    var overlay = document.createElement('div');
    overlay.className = 'overlay-popup';
    overlay.innerHTML = `
      <div class="overlay-content">
        <p>${message}</p>
        <button class="primary-btn" onclick="this.parentNode.parentNode.remove()">OK</button>
      </div>
    `;
    document.body.appendChild(overlay);
    
    if (duration) {
      setTimeout(function() {
        if (overlay.parentNode) {
          overlay.parentNode.removeChild(overlay);
        }
      }, duration);
    }
  }
  
  function showConfirmOverlay(message, confirmCallback) {
    var overlay = document.createElement('div');
    overlay.className = 'overlay-popup';
    overlay.innerHTML = `
      <div class="overlay-content">
        <p>${message}</p>
        <div class="overlay-buttons">
          <button class="primary-btn" id="confirmYes">Yes</button>
          <button class="cancel-btn" id="confirmNo">No</button>
        </div>
      </div>
    `;
    document.body.appendChild(overlay);
    
    document.getElementById('confirmYes').addEventListener('click', function() {
      overlay.parentNode.removeChild(overlay);
      if (typeof confirmCallback === 'function') {
        confirmCallback();
      }
    });
    
    document.getElementById('confirmNo').addEventListener('click', function() {
      overlay.parentNode.removeChild(overlay);
    });
  }

  // Device status checking
  
  function checkDeviceStatus() {
    fetch("/devices/status")
      .then(response => response.json())
      .then(data => {
        if(data.status === "success") {
          data.devices.forEach(function(device) {
            var row = document.querySelector('tr[data-index="' + device.index + '"]');
            if (row) {
              var statusCell = row.querySelector('td:nth-child(8)');
              if(device.online) {
                statusCell.innerHTML = '<span style="color:green;">&#9679;</span>';
              } else {
                statusCell.innerHTML = '<span style="color:red;">&#9679;</span>';
              }
            }
          });
        }
      })
      .catch(error => {
        console.error("Error checking device status:", error);
      });
  }
  setInterval(checkDeviceStatus, 5000);
  checkDeviceStatus();

  // Modal functions for editing and advanced actions
  function openEditModal(index, friendlyName, orientation, address) {
    document.getElementById('editDisplayModal').style.display = 'block';
    document.getElementById('editDeviceIndex').value = index;
    document.getElementById('editFriendlyName').value = friendlyName;
    document.getElementById('editOrientation').value = orientation;
    document.getElementById('editAddress').value = address;
  }

  function openAdvancedModal(index, friendlyName) {
    document.getElementById('advancedActionsModal').style.display = 'block';
    document.getElementById('advancedDeviceTitle').textContent = "Advanced Actions for " + friendlyName;
    document.getElementById('advancedActionsModal').setAttribute('data-device-index', index);
  }
  
  // Device API functions: triggerSystemUpdate, triggerBackup, triggerAppUpdate remain unchanged
  function triggerSystemUpdate() {
    var deviceIndex = document.getElementById('advancedActionsModal').getAttribute('data-device-index');
    var deviceRow = document.querySelector('tr[data-index="' + deviceIndex + '"]');
    var deviceAddress = deviceRow.getAttribute('data-address');
    if (!deviceAddress) {
      showOverlayMessage("Error: Device address not found");
      return;
    }
    showConfirmOverlay(
      "This will trigger a system update and reboot the device. Continue?",
      function() {
        var progressOverlay = document.createElement('div');
        progressOverlay.className = 'overlay-popup';
        progressOverlay.innerHTML = `
          <div class="overlay-content">
            <p>Triggering system update...</p>
            <div class="progress-container">
              <div id="updateProgressBar" class="progress-bar" style="width: 0%">0%</div>
            </div>
          </div>
        `;
        document.body.appendChild(progressOverlay);
        if (!deviceAddress.startsWith("http://") && !deviceAddress.startsWith("https://")) {
          deviceAddress = "http://" + deviceAddress;
        }
        fetch(deviceAddress + "/system_update", { method: 'POST' })
          .then(response => {
            if (!response.ok) {
              throw new Error("Network response was not ok");
            }
            return response.json();
          })
          .then(data => {
            var progress = 0;
            var interval = setInterval(function() {
              progress += 5;
              if (progress >= 100) {
                progress = 100;
                clearInterval(interval);
                document.body.removeChild(progressOverlay);
                showOverlayMessage("System update triggered successfully. Device will reboot.");
                closeAdvancedActionsModal();
              }
              var progressBar = document.getElementById('updateProgressBar');
              if (progressBar) {
                progressBar.style.width = progress + '%';
                progressBar.textContent = progress + '%';
              }
            }, 500);
          })
          .catch(error => {
            document.body.removeChild(progressOverlay);
            showOverlayMessage("Error triggering system update: " + error.message);
          });
      }
    );
  }
  
  function triggerBackup() {
    var deviceIndex = document.getElementById('advancedActionsModal').getAttribute('data-device-index');
    var deviceRow = document.querySelector('tr[data-index="' + deviceIndex + '"]');
    var deviceAddress = deviceRow.getAttribute('data-address');
    if (!deviceAddress) {
      showOverlayMessage("Error: Device address not found");
      return;
    }
    showConfirmOverlay(
      "This will create a backup of the device. This may take several minutes. Continue?",
      function() {
        var progressOverlay = document.createElement('div');
        progressOverlay.className = 'overlay-popup';
        progressOverlay.innerHTML = `
          <div class="overlay-content">
            <p>Creating backup...</p>
            <div class="progress-container">
              <div id="backupProgressBar" class="progress-bar" style="width: 0%">0%</div>
            </div>
          </div>
        `;
        document.body.appendChild(progressOverlay);
        if (!deviceAddress.startsWith("http://") && !deviceAddress.startsWith("https://")) {
          deviceAddress = "http://" + deviceAddress;
        }
        var progress = 0;
        var interval = setInterval(function() {
          progress += 2;
          if (progress >= 100) {
            progress = 100;
            clearInterval(interval);
            var a = document.createElement('a');
            a.href = deviceAddress + "/backup";
            a.download = "backup_" + new Date().toISOString().replace(/:/g, '-') + ".img.gz";
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            document.body.removeChild(progressOverlay);
            showOverlayMessage("Backup created successfully. Download started.");
            closeAdvancedActionsModal();
          }
          var progressBar = document.getElementById('backupProgressBar');
          if (progressBar) {
            progressBar.style.width = progress + '%';
            progressBar.textContent = progress + '%';
          }
        }, 500);
      }
    );
  }
  
  function triggerAppUpdate() {
    var deviceIndex = document.getElementById('advancedActionsModal').getAttribute('data-device-index');
    var deviceRow = document.querySelector('tr[data-index="' + deviceIndex + '"]');
    var deviceAddress = deviceRow.getAttribute('data-address');
    if (!deviceAddress) {
      showOverlayMessage("Error: Device address not found");
      return;
    }
    showConfirmOverlay(
      "This will update the application on the device and reboot it. Continue?",
      function() {
        var progressOverlay = document.createElement('div');
        progressOverlay.className = 'overlay-popup';
        progressOverlay.innerHTML = `
          <div class="overlay-content">
            <p>Updating application...</p>
            <div class="progress-container">
              <div id="appUpdateProgressBar" class="progress-bar" style="width: 0%">0%</div>
            </div>
          </div>
        `;
        document.body.appendChild(progressOverlay);
        if (!deviceAddress.startsWith("http://") && !deviceAddress.startsWith("https://")) {
          deviceAddress = "http://" + deviceAddress;
        }
        fetch(deviceAddress + "/update", { method: 'POST' })
          .then(response => {
            if (!response.ok) {
              throw new Error("Network response was not ok");
            }
            return response.json();
          })
          .then(data => {
            var progress = 0;
            var interval = setInterval(function() {
              progress += 10;
              if (progress >= 100) {
                progress = 100;
                clearInterval(interval);
                document.body.removeChild(progressOverlay);
                showOverlayMessage("Application updated successfully. Device will reboot.");
                closeAdvancedActionsModal();
              }
              var progressBar = document.getElementById('appUpdateProgressBar');
              if (progressBar) {
                progressBar.style.width = progress + '%';
                progressBar.textContent = progress + '%';
              }
            }, 300);
          })
          .catch(function(error) {
            document.body.removeChild(progressOverlay);
            showOverlayMessage("Error updating application: " + error.message);
          });
      }
    );
  }

  document.addEventListener('DOMContentLoaded', function() {
    var addNewDisplayBtn = document.getElementById('addNewDisplayBtn');
    if (addNewDisplayBtn) {
      addNewDisplayBtn.addEventListener('click', function() {
        document.getElementById('addDisplayModal').style.display = 'block';
      });
    }
    var addDisplayForm = document.getElementById('addDisplayForm');
    addDisplayForm.addEventListener('submit', function(e) {
      e.preventDefault();
      fetchDisplayInfo('new').then(function() {
        addDisplayForm.submit();
      }).catch(function() {
        addDisplayForm.submit();
      });
    });
    document.getElementById('closeAddDisplayModal').addEventListener('click', function() {
      closeAddDisplayModal();
    });
    document.getElementById('closeEditDisplayModal').addEventListener('click', function() {
      closeEditDisplayModal();
    });
    document.getElementById('closeAdvancedActionsModal').addEventListener('click', function() {
      closeAdvancedActionsModal();
    });
    var clipSettingsBtn = document.getElementById('clipSettingsBtn');
    var clipSettingsModal = document.getElementById('clipSettingsModal');
    var closeClipSettingsModal = document.getElementById('closeClipSettingsModal');
    if (clipSettingsBtn) {
      clipSettingsBtn.addEventListener('click', function() {
        clipSettingsModal.style.display = 'block';
      });
    }
    if (closeClipSettingsModal) {
      closeClipSettingsModal.addEventListener('click', function() {
        clipSettingsModal.style.display = 'none';
      });
    }
    window.addEventListener('click', function(e) {
      if (e.target == clipSettingsModal) {
        clipSettingsModal.style.display = 'none';
      }
    });
  });

  function saveClipModel() {
    var clipModel = document.getElementById('clip_model').value;
    if (!clipModel) {
      showOverlayMessage("Please select a CLIP model");
      return;
    }
    var payload = {
      clip_model: clipModel
    };
    showOverlayMessage("Switching to model: " + clipModel + "...", 1500);
    fetch("{{ url_for('settings.update_clip_model') }}", {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    })
    .then(function(response) { return response.json(); })
    .then(function(data) {
      if (data.status === "success") {
        showOverlayMessage("CLIP model updated successfully to " + clipModel);
      } else {
        showOverlayMessage("Error updating CLIP model: " + data.message);
      }
    })
    .catch(function(error) {
      console.error("Error:", error);
      showOverlayMessage("An error occurred while updating CLIP model.");
    });
  }

  function rerunAllTagging() {
    showConfirmOverlay(
      "This will rerun tagging on all images using the selected CLIP model. This may take some time depending on the number of images. Continue?",
      function() {
        fetch("{{ url_for('settings.rerun_all_tagging') }}", {
          method: 'POST'
        })
        .then(function(response) { return response.json(); })
        .then(function(data) {
          if (data.status === "success") {
            showOverlayMessage("Tagging process started! This will run in the background.");
            document.getElementById('clipSettingsModal').style.display = 'none';
          } else {
            showOverlayMessage("Error starting tagging process: " + data.message);
          }
        })
        .catch(function(error) {
          console.error("Error:", error);
          showOverlayMessage("An error occurred while starting the tagging process.");
        });
      }
    );
  }

  function fetchDisplayInfo(mode) {
    return new Promise(function(resolve, reject) {
      if (mode === 'new') {
        var addressInput = document.getElementById('newAddress');
        var address = addressInput.value.trim();
        if (!address) {
          alert("Please enter the device address.");
          reject("No address provided");
          return;
        }
        fetch("/device_info?address=" + encodeURIComponent(address), { timeout: 5000 })
          .then(function(response) {
            if (!response.ok) {
              throw new Error("HTTP error " + response.status);
            }
            return response.json();
          })
          .then(function(data) {
            if (data.status === "ok") {
              document.getElementById('newDisplayName').value = data.info.display_name;
              document.getElementById('newResolution').value = data.info.resolution;
              var availableColors = ['#FF5733', '#33FF57', '#3357FF', '#F39C12', '#8E44AD', '#2ECC71', '#E74C3C'];
              var randomColor = availableColors[Math.floor(Math.random() * availableColors.length)];
              document.getElementById('newColor').value = randomColor;
              resolve();
            } else {
              alert("Error fetching display info: " + data.message + "; using default values.");
              document.getElementById('newDisplayName').value = "DefaultDisplay color";
              document.getElementById('newResolution').value = "800x600";
              document.getElementById('newColor').value = "#FF5733";
              resolve();
            }
          })
          .catch(function(error) {
            console.error("Error fetching display info:", error);
            alert("Error fetching display info; using default values.");
            document.getElementById('newDisplayName').value = "DefaultDisplay color";
            document.getElementById('newResolution').value = "800x600";
            document.getElementById('newColor').value = "#FF5733";
            resolve();
          });
      } else if (mode === 'edit') {
        alert("Fetch Display Info for edit is not implemented yet.");
        resolve();
      }
    });
  }
</script>
{% endblock %}
```


## utils/__init__.py

```py

```


## utils/crop_helpers.py

```py
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
```


## utils/image_helpers.py

```py
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
```


## .DS_Store

```
   Bud1           	                                                           i cbwspblob                                                                                                                                                                                                                                                                                                                                                                                                                                           s t a t i cbwspblob   �bplist00�]ShowStatusBar[ShowToolbar[ShowTabView_ContainerShowSidebar\WindowBounds[ShowSidebar		_{{188, 391}, {920, 436}}	#/;R_klmno�                            �    s t a t i cfdscbool     s t a t i cvSrnlong                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            @      �                                        @      �                                          @      �                                          @                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   E  	                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       DSDB                                 `          �                                         @      �                                          @      �                                          @                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              
```


## .export-ignore

```
node_modules/
*.log
dist/
.vscode/
```


## .gitattributes

```
# Auto detect text files and perform LF normalization
* text=auto
```


## app.py

```py
from flask import Flask
import os
from config import Config
from models import db
import pillow_heif
from tasks import celery, start_scheduler

# Register HEIF opener for Pillow
pillow_heif.register_heif_opener()

def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)

    # Ensure required folders exist
    for folder in [app.config['IMAGE_FOLDER'], app.config['THUMBNAIL_FOLDER'], app.config['DATA_FOLDER']]:
        if not os.path.exists(folder):
            os.makedirs(folder)

    # Initialize database without migrations
    db.init_app(app)

    # Register blueprints
    from routes.image_routes import image_bp
    from routes.device_routes import device_bp
    from routes.schedule_routes import schedule_bp
    from routes.settings_routes import settings_bp
    from routes.device_info_routes import device_info_bp
    from routes.ai_tagging_routes import ai_bp

    app.register_blueprint(image_bp)
    app.register_blueprint(device_bp)
    app.register_blueprint(schedule_bp)
    app.register_blueprint(settings_bp)
    app.register_blueprint(device_info_bp)
    app.register_blueprint(ai_bp)

    # Create database tables if they don't exist.
    with app.app_context():
        db.create_all()

    # Configure Celery
    celery.conf.update(
        broker_url='redis://localhost:6379/0',
        result_backend='redis://localhost:6379/0'
    )

    # Note: Scheduler is now started in a dedicated process (scheduler.py)
    # We still call the function for backward compatibility, but it doesn't start the scheduler
    start_scheduler(app)
    
    # We no longer run fetch_device_metrics immediately here
    # The dedicated scheduler process will handle this

    return app

app = create_app()

# Make the app available to Celery tasks
celery.conf.update(app=app)

if __name__ == '__main__':
    # When running via 'python app.py' this block will execute.
    app.run(host='0.0.0.0', port=5001, debug=True)
```


## config.py

```py
import os

basedir = os.path.abspath(os.path.dirname(__file__))

class Config:
    SECRET_KEY = "super-secret-key"
    # Database: using an absolute path in a “data” folder in the project directory.
    # In the container, basedir will be /app so the DB will be at /app/data/mydb.sqlite.
    SQLALCHEMY_DATABASE_URI = 'sqlite:///' + os.path.join(basedir, 'data', 'mydb.sqlite')
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # Folders for images, thumbnails, and data storage
    IMAGE_FOLDER = os.path.join(basedir, 'images')
    THUMBNAIL_FOLDER = os.path.join(basedir, 'images', 'thumbnails')
    DATA_FOLDER = os.path.join(basedir, 'data')
```


## Dockerfile

```
# Use an official Python image
FROM python:3.13.2-slim

# Set timezone and cache directory for models (persisted in /data/model_cache)
ENV TZ=Europe/Copenhagen
ENV XDG_CACHE_HOME=/app/data/model_cache

# Install system dependencies and redis-server
RUN apt-get update && apt-get install -y \
    curl \
    sqlite3 \
    supervisor \
    tzdata \
    build-essential \
    gcc \
    git \
    libffi-dev \
    libssl-dev \
    libjpeg-dev \
    zlib1g-dev \
    libpng-dev \
    libfreetype6-dev \
    redis-server \
    && ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone \
    && rm -rf /var/lib/apt/lists/*

# Create necessary directories for the database and model cache
RUN mkdir -p /data /app/data/model_cache

# Set working directory
WORKDIR /app

# Copy only the requirements file first for caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install -r requirements.txt

# Pre-download all CLIP models (this layer will be cached if requirements.txt hasn't changed)
RUN python -c "import open_clip; \
    open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai', jit=False, force_quick_gelu=True); \
    open_clip.create_model_and_transforms('ViT-B-16', pretrained='openai', jit=False, force_quick_gelu=True); \
    open_clip.create_model_and_transforms('ViT-L-14', pretrained='openai', jit=False, force_quick_gelu=True)"

# Copy the rest of the project files
COPY . .

# Make scheduler.py executable
RUN chmod +x /app/scheduler.py

# Set environment variables for Celery
ENV CELERY_WORKER_MAX_MEMORY_PER_CHILD=500000
ENV CELERY_WORKERS=2

# Expose port 5001
EXPOSE 5001

# Copy entrypoint script and make it executable
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Copy Supervisor configuration file
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Tell Flask which file is our app
ENV FLASK_APP=app.py

# Run entrypoint.sh (which handles migrations and launches Supervisor)
CMD ["/entrypoint.sh"]
```


## entrypoint.sh

```sh
#!/bin/sh
# entrypoint.sh - Auto-create the database tables then launch the app via Supervisor.

echo "Ensuring /app/data folder exists..."
mkdir -p /app/data

# Start Redis first and ensure it's running
echo "Starting Redis server..."
redis-server --daemonize yes
sleep 2
echo "Checking Redis connection..."
redis-cli ping
if [ $? -ne 0 ]; then
  echo "Redis is not responding. Waiting a bit longer..."
  sleep 5
  redis-cli ping
  if [ $? -ne 0 ]; then
    echo "Redis still not responding. Please check Redis configuration."
  else
    echo "Redis is now running."
  fi
else
  echo "Redis is running."
fi

echo "Creating database tables..."
python -c "from app import app; from models import db; app.app_context().push(); db.create_all()"
echo "Database tables created successfully."

echo "Starting Supervisor..."
exec /usr/bin/supervisord -c /etc/supervisor/conf.d/supervisord.conf
```


## export.md

```md
# Project Structure

```
assets/
  js/
    main.js
data/
images/
migrations/
  versions/
  __init__.py
  alembic.ini
  env.py
  script.py.mako
routes/
  __init__.py
  additional_routes.py
  ai_tagging_routes.py
  device_info_routes.py
  device_routes.py
  image_routes.py
  schedule_routes.py
  settings_routes.py
static/
  icons/
  send-icon-old.png
  send-icon.png
  settings-wheel.png
  style.css
  trash-icon.png
templates/
  base.html
  index.html
  schedule.html
  settings.html
utils/
  __init__.py
  crop_helpers.py
  image_helpers.py
.DS_Store
.export-ignore
.gitattributes
app.py
config.py
Dockerfile
entrypoint.sh
export.md
exportconfig.json
LICENSE
models.py
package.json
README.md
requirements.txt
scheduler.py
supervisord.conf
tasks.py
webpack.config.js
```


## assets/js/main.js

```js
import { Calendar } from '@fullcalendar/core';
import timeGridPlugin from '@fullcalendar/timegrid';
import '@fullcalendar/core/main.css';
import '@fullcalendar/timegrid/main.css';

document.addEventListener('DOMContentLoaded', function() {
  var calendarEl = document.getElementById('calendar');
  var calendar = new Calendar(calendarEl, {
    plugins: [ timeGridPlugin ],
    initialView: 'timeGridWeek',
    firstDay: 1, 
    nowIndicator: true,
    headerToolbar: {
      left: 'prev,next today',
      center: 'title',
      right: 'timeGridWeek,timeGridDay'
    },
    events: '/schedule/events',
    dateClick: function(info) {
      
      var dtLocal = new Date(info.date);
      var isoStr = dtLocal.toISOString().substring(0,16);
      document.getElementById('eventDate').value = isoStr;
      openEventModal();
    }
  });
  calendar.render();
});
```


## migrations/__init__.py

```py

```


## migrations/alembic.ini

```ini
[alembic]
# Path to migration scripts
script_location = migrations
# Database URL - this must point to the same absolute path as used by your app.
sqlalchemy.url = sqlite:////app/data/mydb.sqlite

[loggers]
keys = root,sqlalchemy,alembic

[handlers]
keys = console

[formatters]
keys = generic

[logger_root]
level = WARN
handlers = console

[logger_sqlalchemy]
level = WARN
handlers =
qualname = sqlalchemy.engine
propagate = 0

[logger_alembic]
level = INFO
handlers =
qualname = alembic
propagate = 0

[handler_console]
class = StreamHandler
args = (sys.stderr,)
level = NOTSET
formatter = generic

[formatter_generic]
format = %(levelname)-5.5s [%(name)s] %(message)s
datefmt = %Y-%m-%d %H:%M:%S
```


## migrations/env.py

```py
from __future__ import with_statement
import os
from logging.config import fileConfig

from sqlalchemy import engine_from_config, pool
from alembic import context

# Get the Alembic config and set up logging
config = context.config
fileConfig(config.config_file_name)

# Determine the project root and ensure the data folder exists.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
data_dir = os.path.join(project_root, 'data')
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

# Ensure an empty database file exists so that autogenerate has something to compare.
db_path = os.path.join(data_dir, 'mydb.sqlite')
if not os.path.exists(db_path):
    open(db_path, 'a').close()

# Import all your models so that they are registered with SQLAlchemy's metadata.
from models import db, Device, ImageDB, CropInfo, SendLog, ScheduleEvent, UserConfig, DeviceMetrics
target_metadata = db.metadata

def run_migrations_offline():
    """Run migrations in 'offline' mode."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True
    )

    with context.begin_transaction():
        context.run_migrations()

def run_migrations_online():
    """Run migrations in 'online' mode."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)

        with context.begin_transaction():
            context.run_migrations()

if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
```


## migrations/script.py.mako

```mako
<% 
import re
import uuid
%>
"""${message}

Revision ID: ${up_revision}
Revises: ${down_revision | n}
Create Date: ${create_date}
"""

# revision identifiers, used by Alembic.
revision = '${up_revision}'
down_revision = ${repr(down_revision)}
branch_labels = None
depends_on = None

def upgrade():
    ${upgrades if upgrades else "pass"}

def downgrade():
    ${downgrades if downgrades else "pass"}
```


## routes/__init__.py

```py

```


## routes/additional_routes.py

```py
from flask import Blueprint, request, jsonify, current_app
from models import Device, db
import subprocess, json

additional_bp = Blueprint('additional', __name__)

@additional_bp.route('/fetch_display_info', methods=['GET'])
def fetch_display_info():
    address = request.args.get('address')
    if not address:
        return jsonify({"status": "error", "message": "No address provided"}), 400
    cmd = f'curl -s "{address}/display_info"'
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        return jsonify({"status": "error", "message": result.stderr}), 500
    try:
        raw_info = json.loads(result.stdout)
        colour_str = raw_info.get("colour", "").capitalize()
        model_str = raw_info.get("model", "")
        resolution_arr = raw_info.get("resolution", [])
        if colour_str:
            display_name = f"{colour_str} Colour - {model_str}"
        else:
            display_name = model_str or "Unknown"
        if len(resolution_arr) == 2:
            resolution_str = f"{resolution_arr[0]}x{resolution_arr[1]}"
        else:
            resolution_str = "N/A"
        return jsonify({
            "status": "ok",
            "info": {
                "display_name": display_name,
                "resolution": resolution_str
            }
        }), 200
    except Exception:
        return jsonify({"status": "error", "message": "Invalid JSON returned"}), 500
```


## routes/ai_tagging_routes.py

```py
from flask import Blueprint, request, jsonify, current_app
import os
from tasks import (
    get_image_embedding,
    generate_tags_and_description,
    reembed_image,
    bulk_tag_images,
    BULK_PROGRESS
)
from models import db, ImageDB
from PIL import Image

ai_bp = Blueprint("ai_tagging", __name__)

@ai_bp.route("/api/ai_tag_image", methods=["POST"])
def ai_tag_image():
    data = request.get_json()
    filename = data.get("filename")
    if not filename:
        return jsonify({"status": "error", "message": "Filename is required"}), 400

    image_folder = current_app.config.get("IMAGE_FOLDER", "./images")
    image_path = os.path.join(image_folder, filename)
    if not os.path.exists(image_path):
        return jsonify({"status": "error", "message": "Image file not found"}), 404

    image_embedding = get_image_embedding(image_path)
    if image_embedding is None:
        return jsonify({"status": "error", "message": "Failed to get embedding"}), 500

    tags, description = generate_tags_and_description(image_embedding)
    # Update the ImageDB record with generated tags
    image_record = ImageDB.query.filter_by(filename=filename).first()
    if image_record:
        image_record.tags = ", ".join(tags)
        db.session.commit()
    else:
        image_record = ImageDB(filename=filename, tags=", ".join(tags))
        db.session.add(image_record)
        db.session.commit()

    return jsonify({
        "status": "success",
        "filename": filename,
        "tags": tags
    }), 200

@ai_bp.route("/api/search_images", methods=["GET"])
def search_images():
    q = request.args.get("q", "").strip()
    if not q:
        return jsonify({"status": "error", "message": "Missing query parameter"}), 400
    images = ImageDB.query.filter(ImageDB.tags.ilike(f"%{q}%")).all()
    results = {
        "ids": [img.filename for img in images],
        "tags": [img.tags for img in images]
    }
    return jsonify({"status": "success", "results": results}), 200

@ai_bp.route("/api/get_image_metadata", methods=["GET"])
def get_image_metadata():
    filename = request.args.get("filename", "").strip()
    if not filename:
        return jsonify({"status": "error", "message": "Missing filename"}), 400

    image_folder = current_app.config.get("IMAGE_FOLDER", "./images")
    image_path = os.path.join(image_folder, filename)
    resolution_str = "N/A"
    filesize_str = "N/A"
    if os.path.exists(image_path):
        try:
            size_bytes = os.path.getsize(image_path)
            filesize_mb = size_bytes / (1024 * 1024)
            filesize_str = f"{filesize_mb:.2f} MB"
            with Image.open(image_path) as im:
                w, h = im.size
                resolution_str = f"{w}x{h}"
        except Exception as ex:
            current_app.logger.warning(f"Could not read file info for {filename}: {ex}")

    image_record = ImageDB.query.filter_by(filename=filename).first()
    if image_record:
        tags = [t.strip() for t in image_record.tags.split(",")] if image_record.tags else []
        favorite = image_record.favorite
    else:
        tags = []
        favorite = False

    return jsonify({
        "status": "success",
        "tags": tags,
        "favorite": favorite,
        "resolution": resolution_str,
        "filesize": filesize_str
    }), 200

@ai_bp.route("/api/update_image_metadata", methods=["POST"])
def update_image_metadata():
    data = request.get_json() or {}
    filename = data.get("filename", "").strip()
    new_tags = data.get("tags", [])
    if isinstance(new_tags, list):
        tags_str = ", ".join(new_tags)
    else:
        tags_str = new_tags
    favorite = data.get("favorite", None)
    if not filename:
        return jsonify({"status": "error", "message": "Missing filename"}), 400

    image_record = ImageDB.query.filter_by(filename=filename).first()
    if not image_record:
        return jsonify({"status": "error", "message": "Image not found"}), 404

    image_record.tags = tags_str
    if favorite is not None:
        image_record.favorite = bool(favorite)
    db.session.commit()
    return jsonify({"status": "success"}), 200

@ai_bp.route("/api/reembed_image", methods=["GET"])
def reembed_image_endpoint():
    filename = request.args.get("filename", "").strip()
    if not filename:
        return jsonify({"status": "error", "message": "Missing filename"}), 400
    result = reembed_image(filename)
    return jsonify(result)

@ai_bp.route("/api/reembed_all_images", methods=["GET"])
def reembed_all_images_endpoint():
    task_id = bulk_tag_images.delay()
    if not task_id:
        return jsonify({"status": "error", "message": "No images found"}), 404
    return jsonify({"status": "success", "message": f"Reembedding images in background. Task ID: {task_id}"}), 200
```


## routes/device_info_routes.py

```py
# routes/device_info_routes.py

from flask import Blueprint, request, jsonify, Response, stream_with_context
import httpx
import json
import time
import threading
from queue import Queue, Empty
from datetime import datetime
from models import db, Device

device_info_bp = Blueprint('device_info', __name__)

@device_info_bp.route('/device_info', methods=['GET'])
def get_device_info():
    address = request.args.get('address')
    if not address:
        return jsonify({"status": "error", "message": "No address provided"}), 400

    # Ensure the address has a scheme
    if not (address.startswith("http://") or address.startswith("https://")):
        address = "http://" + address

    try:
        # Use httpx with a 10-second timeout and a curl-like User-Agent
        # Reduce timeout to 5 seconds to fail faster
        response = httpx.get(f"{address}/display_info", timeout=5.0, headers={'User-Agent': 'curl/7.68.0'})
        response.raise_for_status()
        raw_info = response.json()
    except httpx.TimeoutException:
        # Handle timeout specifically to provide a clearer error message
        return jsonify({"status": "error", "message": "Connection timed out. Device may be offline."}), 500
    except httpx.HTTPStatusError as e:
        # Handle HTTP errors specifically
        return jsonify({"status": "error", "message": f"HTTP error: {e.response.status_code}"}), 500
    except Exception as e:
        # Handle other exceptions
        return jsonify({"status": "error", "message": f"Error fetching display info: {str(e)}"}), 500

    try:
        # Build display name as "model colour color"
        colour = raw_info.get("colour", "").strip()
        model = raw_info.get("model", "").strip()
        if model and colour:
            display_name = f"{model} {colour} color"
        elif model:
            display_name = model
        else:
            display_name = "Unknown"
        # Format resolution as "widthxheight"
        resolution_arr = raw_info.get("resolution", [])
        if isinstance(resolution_arr, list) and len(resolution_arr) == 2:
            resolution = f"{resolution_arr[0]}x{resolution_arr[1]}"
        else:
            resolution = "N/A"
    except Exception:
        return jsonify({"status": "error", "message": "Invalid JSON returned"}), 500

    return jsonify({
        "status": "ok",
        "info": {
            "display_name": display_name,
            "resolution": resolution
        }
    }), 200

# Global dictionary for active device stream clients (if needed in future)
active_device_streams = {}

@device_info_bp.route('/device/<int:device_index>/stream', methods=['GET'])
def device_stream(device_index):
    """
    Updated streaming endpoint that connects to the device's live stream,
    updates the device's online status, and pushes status updates into a
    thread-safe queue which is then served via Server-Sent Events.
    """
    devices = Device.query.order_by(Device.id).all()
    if not (0 <= device_index < len(devices)):
        return jsonify({"status": "error", "message": "Device not found"}), 404
    device = devices[device_index]
    address = device.address
    if not (address.startswith("http://") or address.startswith("https://")):
        address = "http://" + address

    # Create a thread-safe queue to pass status updates immediately
    status_queue = Queue()

    def stream_reader():
        try:
            # First, check if the device is reachable via its display_info endpoint
            # Use a try-except block to handle connection errors more gracefully
            try:
                info_response = httpx.get(f"{address}/display_info", timeout=5.0)
                if info_response.status_code != 200:
                    status_queue.put({"status": "error", "message": f"Device not responding: {info_response.status_code}"})
                    return
            except Exception as e:
                status_queue.put({"status": "error", "message": f"Cannot connect to device: {str(e)}"})
                return

            # Now, connect to the live stream (no timeout so it remains open)
            client = httpx.Client(timeout=None)
            response = client.get(f"{address}/stream", stream=True)
            if response.status_code != 200:
                status_queue.put({"status": "error", "message": f"Error connecting to stream: {response.status_code}"})
                return

            # Mark the device as online and update the database
            device.online = True
            db.session.commit()

            # Continuously read and process incoming lines from the stream
            for line in response.iter_lines():
                if line:
                    if isinstance(line, bytes):
                        line = line.decode("utf-8")
                    if line.startswith("data:"):
                        data_str = line[len("data:"):].strip()
                        try:
                            data = json.loads(data_str)
                        except json.JSONDecodeError:
                            continue
                        # Only update online status, ignore metrics
                        device.online = True
                        db.session.commit()
                        # Immediately push the status update to the queue for SSE output
                        status_queue.put({"status": "online", "device": device.friendly_name})
        except Exception as e:
            device.online = False
            db.session.commit()
            status_queue.put({"status": "error", "message": f"Stream reader error: {str(e)}"})

    # Start the stream_reader thread as a daemon
    threading.Thread(target=stream_reader, daemon=True).start()

    def generate():
        # Continuously yield each status update received from the queue as an SSE event.
        while True:
            try:
                data = status_queue.get(timeout=10)
                yield f"data: {json.dumps(data)}\n\n"
            except Empty:
                # If no new status update arrives within 10 seconds, send a heartbeat event.
                yield "data: {}\n\n"

    return Response(stream_with_context(generate()), mimetype='text/event-stream')

@device_info_bp.route('/devices/status', methods=['GET'])
def devices_status():
    """Get online status for all devices"""
    # Don't check device status here - just return the current status from the database
    # This prevents duplicate status checks that could flood the logs
    devices = Device.query.all()
    data = []
    for idx, device in enumerate(devices):
        data.append({
            "index": idx,
            "online": device.online
        })
    return jsonify({"status": "success", "devices": data})
```


## routes/device_routes.py

```py
from flask import Blueprint, request, jsonify, current_app, send_file
from models import db, Device
import httpx
import subprocess, os, datetime, json

device_bp = Blueprint('device', __name__)

@device_bp.route('/device/<int:index>/set_orientation', methods=['POST'])
def set_device_orientation(index):
    orientation = request.form.get('orientation')
    if not orientation:
        return jsonify({"status": "error", "message": "No orientation provided"}), 400
    all_devices = Device.query.order_by(Device.id).all()
    if index < 0 or index >= len(all_devices):
        return jsonify({"status": "error", "message": "Device not found"}), 404
    device_address = all_devices[index].address
    if not (device_address.startswith("http://") or device_address.startswith("https://")):
        device_address = "http://" + device_address
    try:
        response = httpx.post(f"{device_address}/set_orientation", data={"orientation": orientation}, timeout=5.0)
        response.raise_for_status()
        return jsonify({"status": "ok"}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@device_bp.route('/device/<int:index>/display_info', methods=['GET'])
def get_device_info(index):
    all_devices = Device.query.order_by(Device.id).all()
    if index < 0 or index >= len(all_devices):
        return jsonify({"status": "error", "message": "Device not found"}), 404
    device_address = all_devices[index].address
    if not (device_address.startswith("http://") or device_address.startswith("https://")):
        device_address = "http://" + device_address
    try:
        # Use a curl-like User-Agent to mimic curl behavior
        response = httpx.get(f"{device_address}/display_info", timeout=5.0, headers={"User-Agent": "curl/7.68.0"})
        response.raise_for_status()
        raw = response.json()
        return jsonify({"status": "ok", "info": raw})
    except Exception as e:
        return jsonify({"status": "error", "message": f"Error fetching display info: {str(e)}"}), 500

@device_bp.route('/device/<int:index>/fetch_metrics', methods=['GET'])
def fetch_metrics(index):
    """
    Fetch the first SSE metric line from the device's /stream endpoint using httpx streaming.
    """
    all_devices = Device.query.order_by(Device.id).all()
    if index < 0 or index >= len(all_devices):
        return jsonify({"status": "error", "message": "Device not found"}), 404
    device = all_devices[index]
    address = device.address
    if not (address.startswith("http://") or address.startswith("https://")):
        address = "http://" + address
    try:
        with httpx.stream("GET", f"{address}/stream", timeout=10.0, headers={"User-Agent": "curl/7.68.0"}) as response:
            for line in response.iter_lines():
                if line:
                    # httpx.iter_lines() returns bytes if no decoding is set
                    if isinstance(line, bytes):
                        line = line.decode("utf-8")
                    if line.startswith("data:"):
                        data_str = line[len("data:"):].strip()
                        data_json = json.loads(data_str)
                        cpu_usage = str(data_json.get("cpu", "N/A"))
                        mem_usage = str(data_json.get("memory", "N/A"))
                        disk_usage = str(data_json.get("disk", "N/A"))
                        device.cpu_usage = cpu_usage
                        device.mem_usage = mem_usage
                        device.disk_usage = disk_usage
                        device.online = True
                        db.session.commit()
                        return jsonify({
                            "status": "ok",
                            "cpu": cpu_usage + "%",
                            "mem": mem_usage + "%",
                            "disk": disk_usage + "%",
                            "online": device.online
                        })
            # If no valid line is found:
            device.online = False
            db.session.commit()
            return jsonify({"status": "error", "message": "No metrics data received"}), 500
    except Exception as e:
        device.online = False
        db.session.commit()
        return jsonify({"status": "error", "message": str(e)}), 500

@device_bp.route('/device/<int:index>/system_update', methods=['POST'])
def system_update(index):
    all_devices = Device.query.order_by(Device.id).all()
    if index < 0 or index >= len(all_devices):
        return jsonify({"status": "error", "message": "Device not found"}), 404
    device_address = all_devices[index].address
    if not (device_address.startswith("http://") or device_address.startswith("https://")):
        device_address = "http://" + device_address
    try:
        response = httpx.post(f"{device_address}/system_update", timeout=10.0)
        response.raise_for_status()
        return jsonify({"status": "ok"}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@device_bp.route('/device/<int:index>/backup', methods=['GET'])
def create_disk_backup(index):
    from flask import send_file
    all_devices = Device.query.order_by(Device.id).all()
    if index < 0 or index >= len(all_devices):
        return jsonify({"status": "error", "message": "Device not found"}), 404
    device_address = all_devices[index].address
    if not (device_address.startswith("http://") or device_address.startswith("https://")):
        device_address = "http://" + device_address
    data_folder = current_app.config['DATA_FOLDER']
    backup_dir = os.path.join(data_folder, "display_backup")
    if not os.path.exists(backup_dir):
        os.makedirs(backup_dir)
    backup_filename = f"backup_{index}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.img.gz"
    backup_path = os.path.join(backup_dir, backup_filename)
    try:
        response = httpx.post(f"{device_address}/backup", timeout=30.0)
        response.raise_for_status()
        with open(backup_path, "wb") as f:
            f.write(response.content)
        if os.path.exists(backup_path):
            return send_file(backup_path, mimetype='application/gzip',
                             as_attachment=True, download_name=backup_filename)
        else:
            return jsonify({"status": "error", "message": "Backup file not created"}), 500
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@device_bp.route('/device/<int:index>/update', methods=['POST'])
def update_application(index):
    all_devices = Device.query.order_by(Device.id).all()
    if index < 0 or index >= len(all_devices):
        return jsonify({"status": "error", "message": "Device not found"}), 404
    device_address = all_devices[index].address
    if not (device_address.startswith("http://") or device_address.startswith("https://")):
        device_address = "http://" + device_address
    try:
        response = httpx.post(f"{device_address}/update", timeout=10.0)
        response.raise_for_status()
        return jsonify({"status": "ok"}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@device_bp.route('/device/<int:index>/stream')
def metrics_stream(index):
    all_devices = Device.query.order_by(Device.id).all()
    if index < 0 or index >= len(all_devices):
        return "Device not found", 404
    device_address = all_devices[index].address
    def generate():
        command = f'curl -N -s "{device_address}/stream"'
        process = subprocess.Popen(command, shell=True,
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, text=True)
        try:
            while True:
                line = process.stdout.readline()
                if not line:
                    break
                yield line
        except Exception:
            process.kill()
        finally:
            process.kill()
    return current_app.response_class(generate(), mimetype='text/event-stream')

@device_bp.route('/test_device/<int:index>', methods=['GET'])
def test_device(index):
    all_devices = Device.query.order_by(Device.id).all()
    if index < 0 or index >= len(all_devices):
        return jsonify({"status": "error", "message": "Device not found"}), 404
    device = all_devices[index]
    address = device.address
    cmd = f'curl -I --max-time 5 -s "{address}"'
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    current_app.logger.info("Curl output for %s: %s", address, result.stdout)
    if result.returncode == 0 and "200 OK" in result.stdout.upper():
        device.online = True
        db.session.commit()
        return jsonify({"status": "ok"}), 200
    else:
        device.online = False
        db.session.commit()
        return jsonify({"status": "error"}), 500

@device_bp.route('/test_connection_address', methods=['GET'])
def test_connection_address():
    address = request.args.get('address')
    if not address:
        return jsonify({"status": "error", "message": "No address provided"}), 400
    cmd = f'curl -I --max-time 5 -s "{address}"'
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    current_app.logger.info("Curl output for %s: %s", address, result.stdout)
    if result.returncode == 0 and "200 OK" in result.stdout.upper():
        return jsonify({"status": "ok"}), 200
    else:
        return jsonify({"status": "failed"}), 500
```


## routes/image_routes.py

```py
from flask import Blueprint, request, redirect, url_for, render_template, flash, send_from_directory, send_file, jsonify, current_app
from models import db, ImageDB, CropInfo, SendLog, Device
import os
from PIL import Image
import subprocess
from utils.image_helpers import allowed_file, convert_to_jpeg
from utils.crop_helpers import load_crop_info_from_db, save_crop_info_to_db, add_send_log_entry, get_last_sent

image_bp = Blueprint('image', __name__)

@image_bp.route('/thumbnail/<filename>')
def thumbnail(filename):
    image_folder = current_app.config['IMAGE_FOLDER']
    thumbnail_folder = current_app.config['THUMBNAIL_FOLDER']
    thumb_path = os.path.join(thumbnail_folder, filename)
    image_path = os.path.join(image_folder, filename)
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
            current_app.logger.error("Error generating thumbnail for %s: %s", filename, e)
            return "Error generating thumbnail", 500
    return send_from_directory(thumbnail_folder, filename)

@image_bp.route('/', methods=['GET', 'POST'])
def upload_file():
    image_folder = current_app.config['IMAGE_FOLDER']
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
                    new_filename = convert_to_jpeg(file, base, image_folder)
                    if new_filename is None:
                        flash(f"Error converting {original_filename} to JPEG.")
                    else:
                        if not ImageDB.query.filter_by(filename=new_filename).first():
                            db.session.add(ImageDB(filename=new_filename))
                            db.session.commit()
                            # Trigger automatic image tagging
                            from tasks import process_image_tagging
                            process_image_tagging.delay(new_filename)
                else:
                    filepath = os.path.join(image_folder, original_filename)
                    file.save(filepath)
                    if not ImageDB.query.filter_by(filename=original_filename).first():
                        db.session.add(ImageDB(filename=original_filename))
                        db.session.commit()
                        # Trigger automatic image tagging
                        from tasks import process_image_tagging
                        process_image_tagging.delay(original_filename)
        return redirect(url_for('image.upload_file'))
    
    images_db = ImageDB.query.all()
    images = [img.filename for img in images_db]
    devices_db = Device.query.all()
    devices = []
    for d in devices_db:
        devices.append({
            "color": d.color,
            "friendly_name": d.friendly_name,
            "orientation": d.orientation,
            "address": d.address,
            "display_name": d.display_name,
            "resolution": d.resolution,
            "online": d.online,
            "last_sent": d.last_sent
        })
    last_sent = get_last_sent()
    return render_template('index.html', images=images, devices=devices, last_sent=last_sent)

@image_bp.route('/images/<filename>')
def uploaded_file(filename):
    image_folder = current_app.config['IMAGE_FOLDER']
    filepath = os.path.join(image_folder, filename)
    if not os.path.exists(filepath):
        return "File not found", 404
    if request.args.get("size") == "info":
        try:
            with Image.open(filepath) as img:
                max_width = 300
                w, h = img.size
                if w > max_width:
                    ratio = max_width / float(w)
                    new_size = (max_width, int(h * ratio))
                    img = img.resize(new_size, Image.LANCZOS)
                from io import BytesIO
                buf = BytesIO()
                img.save(buf, format="JPEG")
                buf.seek(0)
                return send_file(buf, mimetype='image/jpeg')
        except Exception as e:
            current_app.logger.error("Error processing image %s for info: %s", filename, e)
            return "Error processing image", 500
    ext = filename.rsplit('.', 1)[1].lower()
    if ext == "heic":
        try:
            from io import BytesIO
            with Image.open(filepath) as img:
                buf = BytesIO()
                img.save(buf, format="JPEG")
                buf.seek(0)
                return send_file(buf, mimetype='image/jpeg')
        except Exception as e:
            current_app.logger.error("Error processing HEIC image %s: %s", filename, e)
            return "Error processing image", 500
    else:
        return send_from_directory(image_folder, filename)

@image_bp.route('/save_crop_info/<filename>', methods=['POST'])
def save_crop_info_endpoint(filename):
    crop_data = request.get_json()
    if not crop_data:
        return jsonify({"status": "error", "message": "No crop data provided"}), 400
    save_crop_info_to_db(filename, crop_data)
    return jsonify({"status": "success"}), 200

@image_bp.route('/send_image/<filename>', methods=['POST'])
def send_image(filename):
    image_folder = current_app.config['IMAGE_FOLDER']
    data_folder = current_app.config['DATA_FOLDER']
    filepath = os.path.join(image_folder, filename)
    if not os.path.exists(filepath):
        return "File not found", 404
    device_addr = request.form.get("device")
    if not device_addr:
        return "No device specified", 400

    from models import Device
    device_obj = Device.query.filter_by(address=device_addr).first()
    if not device_obj:
        return "Device not found in DB", 500
    dev_width = None
    dev_height = None
    if device_obj.resolution:
        parts = device_obj.resolution.split("x")
        if len(parts) == 2:
            try:
                dev_width = int(parts[0])
                dev_height = int(parts[1])
            except:
                pass
    if not (dev_width and dev_height):
        return "Target resolution not found", 500

    try:
        with Image.open(filepath) as orig_img:
            orig_w, orig_h = orig_img.size
            
            # Check if device is in portrait orientation
            is_portrait = device_obj.orientation.lower() == 'portrait'
            
            # If portrait, swap width and height for target ratio calculation
            if is_portrait:
                target_ratio = dev_height / dev_width
            else:
                target_ratio = dev_width / dev_height
                
            cdata = load_crop_info_from_db(filename)
            if cdata:
                x = cdata.get("x", 0)
                y = cdata.get("y", 0)
                w = cdata.get("width", orig_w)
                h = cdata.get("height", orig_h)
                cropped = orig_img.crop((x, y, x+w, y+h))
            else:
                orig_ratio = orig_w / orig_h
                if orig_ratio > target_ratio:
                    new_width = int(orig_h * target_ratio)
                    left = (orig_w - new_width) // 2
                    crop_box = (left, 0, left + new_width, orig_h)
                else:
                    new_height = int(orig_w / target_ratio)
                    top = (orig_h - new_height) // 2
                    crop_box = (0, top, orig_w, top + new_height)
                cropped = orig_img.crop(crop_box)

            # If portrait, rotate the image 90 degrees clockwise and swap dimensions
            if is_portrait:
                cropped = cropped.rotate(-90, expand=True)  # -90 for clockwise rotation
                final_img = cropped.resize((dev_height, dev_width), Image.LANCZOS)  # Note swapped dimensions
            else:
                final_img = cropped.resize((dev_width, dev_height), Image.LANCZOS)
            temp_dir = os.path.join(data_folder, "temp")
            if not os.path.exists(temp_dir):
                os.makedirs(temp_dir)
            temp_filename = os.path.join(temp_dir, f"temp_{filename}")
            final_img.save(temp_filename, format="JPEG", quality=95)

        cmd = f'curl "{device_addr}/send_image" -X POST -F "file=@{temp_filename}"'
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        os.remove(temp_filename)
        if result.returncode != 0:
            return f"Error sending image: {result.stderr}", 500

        device_obj.last_sent = filename
        db.session.commit()
        add_send_log_entry(filename)
        return f"Image sent successfully: {result.stdout}", 200
    except Exception as e:
        current_app.logger.error("Error resizing/cropping image: %s", e)
        return f"Error processing image: {e}", 500

@image_bp.route('/delete_image/<filename>', methods=['POST'])
def delete_image_endpoint(filename):
    image_folder = current_app.config['IMAGE_FOLDER']
    thumbnail_folder = current_app.config['THUMBNAIL_FOLDER']
    filepath = os.path.join(image_folder, filename)
    thumb_path = os.path.join(thumbnail_folder, filename)
    if os.path.exists(filepath):
        try:
            os.remove(filepath)
            if os.path.exists(thumb_path):
                os.remove(thumb_path)
            img_obj = ImageDB.query.filter_by(filename=filename).first()
            if img_obj:
                db.session.delete(img_obj)
                db.session.commit()
            return jsonify({"status": "success", "message": "Image deleted"}), 200
        except Exception as e:
            current_app.logger.error("Error removing file %s: %s", filepath, e)
            return jsonify({"status": "error", "message": str(e)}), 500
    else:
        return jsonify({"status": "error", "message": "File not found"}), 404
```


## routes/schedule_routes.py

```py
from flask import Blueprint, request, jsonify, render_template
from models import db, ScheduleEvent, Device, ImageDB
import datetime
from tasks import send_scheduled_image
# Import the scheduler from the dedicated scheduler module instead of tasks
# This ensures we're using the scheduler that runs in a dedicated process
from scheduler import scheduler

schedule_bp = Blueprint('schedule', __name__)

@schedule_bp.route('/schedule')
def schedule_page():
    devs = Device.query.all()
    devices = []
    for d in devs:
        devices.append({
            "color": d.color,
            "friendly_name": d.friendly_name,
            "orientation": d.orientation,
            "address": d.address,
            "display_name": d.display_name,
            "resolution": d.resolution,
            "online": d.online,
            "last_sent": d.last_sent
        })
    
    # Get all images and their tags
    imgs_db = ImageDB.query.all()
    images = [i.filename for i in imgs_db]
    
    # Create a dictionary of image tags for the template
    image_tags = {}
    for img in imgs_db:
        if img.tags:
            image_tags[img.filename] = img.tags
    
    return render_template("schedule.html", devices=devices, images=images, image_tags=image_tags)

@schedule_bp.route('/schedule/events')
def get_events():
    # Ensure we're getting the latest data from the database
    db.session.expire_all()
    events = ScheduleEvent.query.all()
    event_list = []
    now = datetime.datetime.now()
    horizon = now + datetime.timedelta(days=90)
    print(f"Loading {len(events)} events from database")
    
    # Create a device lookup dictionary for colors and friendly names
    device_lookup = {}
    devices = Device.query.all()
    for device in devices:
        device_lookup[device.address] = {
            "color": device.color,
            "friendly_name": device.friendly_name
        }
    
    # Create an image lookup for thumbnails
    image_lookup = {}
    images = ImageDB.query.all()
    for img in images:
        image_lookup[img.filename] = {
            "tags": img.tags or ""
        }
    
    for ev in events:
        # Get device color and friendly name
        device_info = device_lookup.get(ev.device, {"color": "#cccccc", "friendly_name": ev.device})
        device_color = device_info["color"]
        device_name = device_info["friendly_name"]
        
        if ev.recurrence.lower() == "none":
            event_list.append({
                "id": ev.id,
                "title": f"{ev.filename}",
                "start": ev.datetime_str,
                "device": ev.device,
                "deviceName": device_name,
                "filename": ev.filename,
                "recurrence": ev.recurrence,
                "series": False,
                "backgroundColor": device_color,
                "borderColor": device_color,
                "textColor": "#ffffff",
                "extendedProps": {
                    "thumbnail": f"/thumbnail/{ev.filename}"
                }
            })
        else:
            # Generate recurring occurrences from the next occurrence up to the horizon
            try:
                start_dt = datetime.datetime.fromisoformat(ev.datetime_str)
            except Exception:
                continue
            # Advance to the first occurrence that is >= now
            rec = ev.recurrence.lower()
            occurrence = start_dt
            while occurrence < now:
                if rec == "daily":
                    occurrence += datetime.timedelta(days=1)
                elif rec == "weekly":
                    occurrence += datetime.timedelta(weeks=1)
                elif rec == "monthly":
                    occurrence += datetime.timedelta(days=30)
                else:
                    break
            # Generate occurrences until the horizon
            while occurrence <= horizon:
                event_list.append({
                    "id": ev.id,  # same series id
                    "title": f"{ev.filename} (Recurring)",  # Mark as recurring in title
                    "start": occurrence.isoformat(sep=' '),
                    "device": ev.device,
                    "deviceName": device_name,
                    "filename": ev.filename,
                    "recurrence": ev.recurrence,
                    "series": True,
                    "backgroundColor": device_color,
                    "borderColor": device_color,
                    "textColor": "#ffffff",
                    "classNames": ["recurring-event"],  # Add a class for styling
                    "extendedProps": {
                        "thumbnail": f"/thumbnail/{ev.filename}",
                        "isRecurring": True  # Flag for frontend to identify recurring events
                    }
                })
                if rec == "daily":
                    occurrence += datetime.timedelta(days=1)
                elif rec == "weekly":
                    occurrence += datetime.timedelta(weeks=1)
                elif rec == "monthly":
                    occurrence += datetime.timedelta(days=30)
                else:
                    break
    return jsonify(event_list)

@schedule_bp.route('/schedule/add', methods=['POST'])
def add_event():
    data = request.get_json()
    datetime_str = data.get("datetime")
    device = data.get("device")
    filename = data.get("filename")
    recurrence = data.get("recurrence", "none")
    timezone_offset = data.get("timezone_offset", 0)  # Minutes offset from UTC
    
    print(f"Adding event with datetime: {datetime_str}, timezone offset: {timezone_offset}")
    if not (datetime_str and device and filename):
        return jsonify({"status": "error", "message": "Missing parameters"}), 400
    try:
        dt = datetime.datetime.fromisoformat(datetime_str)
        
        # Apply timezone offset if provided (convert from local to UTC)
        if timezone_offset:
            # timezone_offset is in minutes, positive for behind UTC, negative for ahead
            # We need to subtract it to convert from local to UTC
            dt = dt - datetime.timedelta(minutes=timezone_offset)
            print(f"Applied timezone offset: {timezone_offset} minutes, new datetime: {dt}")
            
        formatted_dt_str = dt.isoformat(sep=' ')
        print(f"Parsed datetime: {datetime_str} -> {formatted_dt_str}")
    except Exception as e:
        return jsonify({"status": "error", "message": f"Invalid datetime format: {str(e)}"}), 400
    new_event = ScheduleEvent(
        filename=filename,
        device=device,
        datetime_str=formatted_dt_str,
        sent=False,
        recurrence=recurrence
    )
    db.session.add(new_event)
    db.session.commit()
    
    # Schedule the event using the scheduler from scheduler.py
    # This will be picked up by the dedicated scheduler process
    scheduler.add_job(send_scheduled_image, 'date', run_date=dt, args=[new_event.id])
    
    return jsonify({"status": "success", "event": {
        "id": new_event.id,
        "filename": new_event.filename,
        "datetime": new_event.datetime_str,
        "recurrence": recurrence
    }})

@schedule_bp.route('/schedule/remove/<int:event_id>', methods=['POST'])
def remove_event(event_id):
    ev = ScheduleEvent.query.get(event_id)
    if ev:
        db.session.delete(ev)
        db.session.commit()
    return jsonify({"status": "success"})

@schedule_bp.route('/schedule/update', methods=['POST'])
def update_event():
    data = request.get_json()
    event_id = data.get("event_id")
    new_datetime = data.get("datetime")
    timezone_offset = data.get("timezone_offset", 0)  # Minutes offset from UTC
    
    # Optional parameters for full event update
    device = data.get("device")
    filename = data.get("filename")
    recurrence = data.get("recurrence")
    
    print(f"Updating event {event_id} to {new_datetime} with timezone offset {timezone_offset}")
    
    if not (event_id and new_datetime):
        return jsonify({"status": "error", "message": "Missing parameters"}), 400
    
    try:
        # Convert event_id to integer if it's a string
        if isinstance(event_id, str):
            event_id = int(event_id)
            
        ev = ScheduleEvent.query.get(event_id)
        if not ev:
            return jsonify({"status": "error", "message": "Event not found"}), 404
            
        # Format the datetime string properly
        try:
            # Parse the datetime string
            if 'Z' in new_datetime:
                dt = datetime.datetime.fromisoformat(new_datetime.replace('Z', '+00:00'))
            elif 'T' in new_datetime:
                # Handle ISO format without timezone
                dt = datetime.datetime.fromisoformat(new_datetime)
            else:
                # Handle other formats
                dt = datetime.datetime.fromisoformat(new_datetime)
            
            # Apply timezone offset if provided (convert from local to UTC)
            if timezone_offset:
                # timezone_offset is in minutes, positive for behind UTC, negative for ahead
                # We need to subtract it to convert from local to UTC
                dt = dt - datetime.timedelta(minutes=timezone_offset)
                print(f"Applied timezone offset: {timezone_offset} minutes, new datetime: {dt}")
                
            formatted_dt_str = dt.isoformat(sep=' ')
            print(f"Parsed datetime: {new_datetime} -> {formatted_dt_str}")
        except Exception as e:
            return jsonify({"status": "error", "message": f"Invalid datetime format: {str(e)}"}), 400
            
        # Update the event in the database
        ev.datetime_str = formatted_dt_str
        
        # Update other fields if provided
        if device:
            ev.device = device
        if filename:
            ev.filename = filename
        if recurrence:
            ev.recurrence = recurrence
        
        # Make sure changes are committed to the database
        try:
            db.session.commit()
            print(f"Successfully updated event {event_id} in database to {formatted_dt_str}")
        except Exception as e:
            db.session.rollback()
            print(f"Error committing changes to database: {str(e)}")
            raise
        
        # If this is a recurring event, we need to update the base date
        if ev.recurrence.lower() != "none":
            # Also update the scheduler if needed
            try:
                scheduler.reschedule_job(
                    job_id=str(event_id),
                    trigger='date',
                    run_date=dt
                )
            except Exception as e:
                print(f"Warning: Could not reschedule job: {str(e)}")
        
        # Log the update for debugging
        print(f"Updated event {event_id} to {formatted_dt_str}")
        
        return jsonify({"status": "success"})
    except Exception as e:
        db.session.rollback()
        return jsonify({"status": "error", "message": f"Error updating event: {str(e)}"}), 500

@schedule_bp.route('/schedule/skip/<int:event_id>', methods=['POST'])
def skip_event_occurrence(event_id):
    ev = ScheduleEvent.query.get(event_id)
    if not ev:
        return jsonify({"status": "error", "message": "Event not found"}), 404
    if ev.recurrence.lower() == "none":
        return jsonify({"status": "error", "message": "Not a recurring event"}), 400
    try:
        dt = datetime.datetime.fromisoformat(ev.datetime_str)
        if ev.recurrence.lower() == "daily":
            next_dt = dt + datetime.timedelta(days=1)
        elif ev.recurrence.lower() == "weekly":
            next_dt = dt + datetime.timedelta(weeks=1)
        elif ev.recurrence.lower() == "monthly":
            next_dt = dt + datetime.timedelta(days=30)
        else:
            return jsonify({"status": "error", "message": "Unknown recurrence type"}), 400
        
        # Update the event in the database with the new date
        ev.datetime_str = next_dt.isoformat(sep=' ')
        ev.sent = False
        db.session.commit()
        
        # Schedule the event using the scheduler from scheduler.py
        # This will be picked up by the dedicated scheduler process
        scheduler.add_job(send_scheduled_image, 'date', run_date=next_dt, args=[ev.id])
        
        return jsonify({"status": "success", "message": "Occurrence skipped"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500
```


## routes/settings_routes.py

```py
from flask import Blueprint, request, render_template, flash, redirect, url_for, jsonify
from models import db, Device, UserConfig
import logging
import httpx  # for querying the Ollama API
from datetime import datetime

settings_bp = Blueprint('settings', __name__)
logger = logging.getLogger(__name__)

@settings_bp.route('/settings', methods=['GET', 'POST'])
def settings():
    if request.method == 'POST':
        color = request.form.get("color")
        friendly_name = request.form.get("friendly_name")
        orientation = request.form.get("orientation")
        address = request.form.get("address")
        display_name = request.form.get("display_name") or "Unknown"
        resolution = request.form.get("resolution") or "N/A"
        if color and friendly_name and orientation and address:
            new_dev = Device(
                color=color,
                friendly_name=friendly_name,
                orientation=orientation,
                address=address,
                display_name=display_name,
                resolution=resolution,
                online=False
            )
            db.session.add(new_dev)
            db.session.commit()
            flash("Device added successfully", "success")
        else:
            flash("Missing mandatory fields (color, friendly name, orientation, address).", "error")
        return redirect(url_for("settings.settings"))
    else:
        devs = Device.query.all()
        devices = []
        for d in devs:
            devices.append({
                "color": d.color,
                "friendly_name": d.friendly_name,
                "orientation": d.orientation,
                "address": d.address,
                "display_name": d.display_name,
                "resolution": d.resolution,
                "online": d.online
            })
        config = UserConfig.query.first()
        return render_template("settings.html", devices=devices, config=config)

@settings_bp.route('/delete_device/<int:device_index>', methods=['POST'])
def delete_device(device_index):
    all_devices = Device.query.order_by(Device.id).all()
    if 0 <= device_index < len(all_devices):
        db.session.delete(all_devices[device_index])
        db.session.commit()
        flash("Device deleted", "success")
    else:
        flash("Device not found", "error")
    return redirect(url_for("settings.settings"))

@settings_bp.route('/edit_device', methods=['POST'])
def edit_device():
    try:
        index = int(request.form.get("device_index"))
        color = request.form.get("color")
        friendly_name = request.form.get("friendly_name")
        orientation = request.form.get("orientation")
        address = request.form.get("address")
        all_devices = Device.query.order_by(Device.id).all()
        if 0 <= index < len(all_devices):
            d = all_devices[index]
            d.color = color or d.color
            d.friendly_name = friendly_name
            d.orientation = orientation
            d.address = address
            db.session.commit()
            flash("Device updated successfully", "success")
        else:
            flash("Device index not found", "error")
    except Exception as e:
        flash("Error editing device: " + str(e), "error")
    return redirect(url_for("settings.settings"))

@settings_bp.route('/settings/update_clip_model', methods=['POST'])
def update_clip_model():
    data = request.get_json()
    config = UserConfig.query.first()
    if not config:
        config = UserConfig(location="London")
        db.session.add(config)
    
    if "clip_model" in data:
        config.clip_model = data.get("clip_model")
        db.session.commit()
        return jsonify({"status": "success", "message": "CLIP model updated."})
    else:
        return jsonify({"status": "error", "message": "No CLIP model provided."})

@settings_bp.route('/settings/rerun_all_tagging', methods=['POST'])
def rerun_all_tagging():
    try:
        # Import the task for rerunning tagging
        from tasks import reembed_all_images
        
        # Start the task
        task = reembed_all_images.delay()
        
        return jsonify({
            "status": "success",
            "message": "Tagging process started.",
            "task_id": str(task.id)
        })
    except Exception as e:
        logger.error(f"Error starting retagging: {str(e)}")
        return jsonify({"status": "error", "message": f"Error: {str(e)}"})

@settings_bp.route('/settings/ollama_models', methods=['GET'])
def ollama_models():
    config = UserConfig.query.first()
    if not config or not config.ollama_address:
        return jsonify({"status": "error", "message": "Ollama address not configured."}), 400
    try:
        url = config.ollama_address.rstrip('/') + '/api/tags'
        response = httpx.get(url, timeout=5)
        response.raise_for_status()
        json_data = response.json()
        models = json_data.get("models", [])
        model_names = []
        for model in models:
            if model.get("name"):
                model_names.append(model["name"])
            else:
                model_names.append(str(model))
        return jsonify({"status": "success", "models": model_names})
    except Exception as e:
        return jsonify({"status": "error", "message": f"Failed to fetch models from Ollama: {str(e)}"}), 500

@settings_bp.route('/device/<int:device_index>/update_status', methods=['POST'])
def update_status(device_index):
    all_devices = Device.query.order_by(Device.id).all()
    if 0 <= device_index < len(all_devices):
        device = all_devices[device_index]
        
        # Update device status
        device.online = True
        db.session.commit()
        
        return jsonify({"status": "ok"})
    return jsonify({"status": "error", "message": "Device not found"}), 404

@settings_bp.route('/devices/status', methods=['GET'])
def devices_status():
    devices = Device.query.all()
    data = []
    for idx, device in enumerate(devices):
        data.append({
            "index": idx,
            "online": device.online
        })
    
    return jsonify({"status": "success", "devices": data})
```


## static/send-icon-old.png

```png
�PNG

   IHDR   `   `   �w8   	pHYs     ��  qIDATx��][�E�E�(�h����jLLDQ���aWF��h�;�n�_��HD�S�b�7��a٭�@�$1�c��2�v���������[����[��c
(P�@�
(P�@0����׹�]B���PX���lpD(�
^���Ol�pY~�K�KZ)"yRv������
�	�W�^�eH��%�!�D����]����%E�|>ֱ�)�q*գ7�7�6;�W����ժ|:d��|.�7.a)�ִ�=�YNѹ�g��ʏ��B�&���~O���M|2c�����P��}���U/v���]�Ϫ���*���4#�P]��נ �������^�f7s�%�����4��9�P�,23�%m4��ky��܆�T��'
�4~�	Z�> S�r�	=�����s���T�R�����GɉV�N@��:e���׳��}Kƙ
�p�6�Q腚M>������d]��*k�
-��G/����A��ñ��K�����I�E�,�X�(bp		ۂ�"��Au̒�=�ĥ5�e�=\�G�����:.;BXw��`N��[���\�Ű�~W3C�P�Af�,	�f׵sG�x������ӹ;��XZ\�1�N������XJ@%D�℄_�o(�y��5X��kܫD�8n*x)�z���9���ˎ?M�ɯ��ϓ(u�X|�!�w��C��b�������;^�%:�f�nk4_*�����������41��w�]��iVź�ټ�" ��s��4#�M�JF��v�}�;��=���eSM@�
���P��\����c �R;4���G�TJ � �u������k��_טH*Wf�c޾y7�z��w�d_��a��mV���1�b=!�$�#�%�����=��;-���(Ck�ZL32O���o�)�����d�(wب.#*��/�7hn����,���l� �d� 9��咀፰��%�k��s�3����g*x֎�l�VB;p�p�6N�_EJ�L(Xe(|�YG`-�{C(��K���S�S(XK��6>�c^Pem�H�\���č\�Kn����z���H@PԊG߅=؇x<h*\����,,�@����#��P��F���V'@����AI��M����U���2��+�� ��Zf�$U�&~����FB	
V%����VK *�)P�3(o%����IU�K�+t�5Kp�����/�=Y(���_�\�_Ի{�'���r	��e����&���tF��	�@·��"
0M����'�S�<��nEB?�$�[�d]l^����[ 4��^�U�&Kf�?QG�$H<(7D��Lx"��F4�'�
>�[@���Ƀm������-Ix��4Cu�T�tCD�A	\§���]�G��f��-s~/�K<���+'��tM��
�A�t#�K�����ٻ��Bp�>իE�p2��V�F}[F��^�8PK[��=
/�\`��f�Ac��%��J��
�)��ݚ0/�z��t}�LP=�Zഘ���tI�7�F}�e��d��pXCDt���������am��i*�w��,o1�ub\��,o0bHE8���T>}!�%~�gC�;Y���5ax/fw�3��Ҩ�Fp�Ӽ�kc�+�,��^�	�	��=nS[��p �KI"��pQZ��*�g� �30�Eҵ�0�V��T�&�$ueGq�H�a*�V��鉠]=�M��m�K\�ZUjO���)�OZ��p��W"V���N�{�G�ʷ�6�r��fN�.��ۯv�\f��QV���L������q�05�7���N��Q��äH�X�j�}������8�\M[��B��ܦ�(P� ��1���`�+    IEND�B`�
```


## static/send-icon.png

```png
�PNG

   IHDR   `   `   �w8   	pHYs     ��  �IDATx��?hTAƟh�XX�Z�hc)(7s$�� o���mJ[K[K۔i��0;���Iaac}��"D�pg�E����|?x���ۛ��{w]          �!d�~b��A(����+�'6}�>�/x�#,�/J�GI^Pzp�{m!X���tklr�{�!8R�黱���;�ސ�6�>����g�����'Bf������E���,`q�W6٤t��}-`>����%Y�����q`C�<�ؕR Į�x^�ε >zN@�<��x��y��OD`�������[����b�0/{����=X;�xj덝w�]켃��b�G;��8��y�����?$�-v��pt������{\�F;�����/S�o�i��k�<�qx�锒~�$��/B��HEQ�b�RQT�صX�$vm%�]���d��T oˍ�uF� 2�BI^N��W��lS�pM���fwb�����-�%�X�`��G�9��Z7$�7���y����7n�~x�%��^���,Nm�v)_�~V�85|*H���������w���8妾�?O-N�)~�[a�d�M�cf�Xq�MY�]�����qʍ�٭N�r��^q��p�]���0f���_onV~�┛����)7��Lq������┛>�8�愣┛%V�Ӫ��n�8����3�t���n�՟�^           ]�|gi�:�.�    IEND�B`�
```


## static/settings-wheel.png

```png
�PNG

   IHDR         �x��   sBIT|d�   	pHYs  �  ��+   tEXtSoftware www.inkscape.org��<    IDATx���w�U����I/�$��{����ذ �"�rE/W�^�]���k�],�*��$������}~�'�9{f�g����<�'q�63{�5����������������������������������������������������������������������������������������������������������������YZ=��%0x0����e�jh2q,��cV>w ��& { ;�� ���K������ s�Ȩek
�_'�ll���.�n%���� *ɩ�Yˍ^�x
X^ � ��$�������u_�z
8x5qm��Y�v�I�b/��^ك�G�(�5�����_�)����7�]**��YcM��%�{�X��x�`Ͱ�k�m������^A����8�9�~�%�b�w�V���P�UQ��G�,헸�ff�5	x7p;���`q�E�r[[W��~����׺�Y�m
|�t��ˈ���*�J�"b٧��YQ�� l����rv 1{��,�b)�Z���n�sM-!�&읤&��228��÷�xo�bey?��۸x%0��Z13� � �@��-#<��w��.ʈ��{��,��jk5����2c)pd��d�A����2c�qbKb3�ژ�H쟮~���'��Mk;b���zH�⎀���Z��j6�g1�VB�Yw��BT�X���33+���'�MO�ʪ�br�Ukp)���:>�w43��Ą�Y���8�����Fg�owe<�w>ͬb#�%Km~J��S/�V���K��_{fV�7���_'��,��DHu;�Wv_�ff+�p1�]��}��K
#��o�����]ֱ��3L#�\�<J���|�`�F��o׺����:�T���hb�_·�����eX8}{�1� ���U3���{�2�v`�aֽ=mS���X��p�0���ZfSbI���դx8t8�` <��;u�5%z�9S��f�|c�� �?��ˀ��ZC18	�9I�o�ע�{������FCk�V��;�vjC\�:�f1��Y�*q���aԦ��i��&=�[h��>9�R�0a�MdfMq8p�P��<A`+�\����xΪ���m
�bB������)�hL�L N�[���L�����xVu�q?��{6�����������̬VV~����Xu�K37n��}=;Vˀ/�mI3����/��1�x�_��Z�~&' ��׫cxq>`ȬvF��3��s��\kw��>��w,%�e�Wf5�-p5���ܸ�ob��\m��}}9ʍ�0�,� �O�X�������%E��عq��F��^��^R���ה:/�xx1f������? ��������E�-e�%΀� p��3�5�=�?�I�	a����<���׀��M���ձX���3`����B�����u�p��~,�S�u�O'f��щ���Y�XJ|y/�����j�Q�a1S����'6+ڢ���U ˉU��M �&�n�Se�2rpq}���[h�>�����>�A���14;��>|��$�Y�& ?A� �:"�̇{�i����G|��[�X���Ͷ�K�(jf�m\����2n#֒٘��xP����;~L��r��7gP�*�2`��ff�p͟x�?.".*���H���ˑg��b_��� ^B,�S���x��|��Jv1iK}�W��n��H�����+N#ݒ���33(c����df%�*���8�j"�JP>G��s�9�� �o��)⛴�p+�����fNg{�UaC4�F	yw�;�Fy��CՖ��E����̬uރ�&N�C���׃CE� �?G�~`��*̬-������x9�l��.�[׎�����#�{B]/��	b�J3�uh�z����r<����=u9��lb;�܌�I3���Ė�/,����,�7l�1�8�fB����6����ˑ&n�&o�Gy7�P��s˫*��9
��Zf�ذ�Jk
p�zs������m��8|G]we��Ĺf6�x�>�7iqq_�>��4!�[􎤞�'�%u=�7ˬ$�&8��Y4fwʑ�{��:���N��C4c��(➚��Nˊo�ZCf57��$��1��^�G�ZeW��T�����1�8��]� �A_�eD/pH��cV_Ǡ�)����C˯�l��ƚ���1x+�,1M�p�1Y�f�[�Pϝ��_$���e�8}�;�?���`�Y`	�z/�,�b��f�w��u��)*�^<���Ǯ�ŚkO�}����O��z�q����[�^)��%�d�g����n,1P�k���W�Y}���&J�욨�j�Cu۴-��4��i�gעo���%)*ì.B�,z�/�mpG����s�[���M�hp2��}�F�$6A�,�NU��	܁�ݚ���뺡O�$N�S��P��T�Y�G�(~����5
x#�<R݆u���7Ќ����,��m8��,Q�e�x�7����'�a$�j�b�mZ���82ڿ���!��^��MW�i�rb���$�7_���>i��iw��ăN�ƹ�"� �ݺ�c[���{[��+��$+�Y�����?VK[��[xp=���%�#���Z�^m�& �G��+
o
d���7�b�T�֮�׀��_U�C�)�.�kц�M��J�+)m��o�����Mg$1��%�=q�Nb�����Ԝuk�;z���%6��W��tK����҆��Ĳ�s�	��s�1��$�D`�Rk�ʰ6p��/����f� �ￒ�Һ5���b�{�����EtZ>J��o�v�u0
�*�kg9pu�� M>2�.��@��+���y��ۘ�?�+��zK`sbO�*,�&6<����U�����w�.Ż�x5aq@o?��{K�����(�-�M�u���"�z�&u��Ľ�����6���ģ�/���t��Alm�r�>��-L߬rk�z[|$uA�,[��:�3���Ki����zs��ԙ �Tg��d^�~D�aq���@.Pg�x�<M�	3�8Z�b	�U��<��� ���:fV�͈ j��3�6� ��l�	u&��3`f�{1���k�0S�)�����cW���O�=�&/��� ��� S��ՙ0�ʌTg�(�@>�F��΀�Uf_�8�w���; ��%��U�G�3��a� s��ՙh#w ��ku��Y��0��H�	�܎~B��Kifj�yzhe<7uAmp�ϟ��r�����xqf ���Z� ��� ۫3`f��p����R2� ��r�U����Yz9t �Rg�����"�XL%� �5�.�����)w �)��;�5�`Kq��g�K���+��%�7�f�����鷞; y�N�`#u�,�m� .Qg�����L�q��of�l-N!p�8��@����̀̚kq�7���r� ��.q�0k.u�Fq��; 9�.N_��0�t���8}���M�?E������8�[��� ��q����7�4F ��y�W���@��?I�����,�$@����� �l�8}w ̚I}o?<)΃�@�� �0k����?*N�:�ȗ�D�1���,�����ӷw �<'{�0m3KG�
`�8}�p o˄i�`�L�C��ӷw lE�0k�q�ʑM���|�A�T� �fR�ӷw 򥞩�T�����/`�șz��g�5��0N��u���5��`�L����=F��@�6��:f�4O��zt�:�ȗ��.� �5�z��5��[�; ���������C��p g[�ӟ!N���Pw �k��`�������/N���x
X$�����w r5�R��{��Y:��W?�w r�0J�����Y:��[=�i���}��# fM�����}��?<*΃���� �$N�p W������,-�$߭��v�z� �gS`3q�0k�;��� {���z� ��� 7�3`fI����a�S�����3 ܠ΀�%�00K��C��ee�I�ra,&�.������f)� ����/��5�|q�,��������˫� .Ug��*�Ou�#�0��:��rHn91
afͷ��͓�����y oA��/�e��Y%�D&�j���y0�E�̥�	�Y�~���sF�Rڠ<��#�ՙ �Rg��*�Ü��3�F� ���t�Y�3����ޤ΄��!�����щ�jfy�f��<��?����G��8�X�`f��8G�	`=��L�U)�_����g�V�D��Y��(��	\���[�#��Y��A{�,�J\V�,����?O\V3��_�?��K�s��,�Ձ��l}����5�̽	�s�/ޚ��fR�F���L��6k�5���Gˉ>��� ��M�_JZZ3����?���3��j�P��D�:���[{�[��%v�ܠ��F�ހx�4�8jz1�s40��u=X���9��~�(��NL�GF7ѫ�Sՙ�X삷'O��j�x�:���⨛�����rg`'`����ppc'�'�*·�g,p?0U���K���Z;����8"i��}��}�x >���YQ,�|�Xεn�������:�'�-�Yz[s��L����F�F�'���_+E�.�;������-�e诛�X l���f	�܊�F�NYh�q�ˁߐ�/��188��`y:���?�#^O���(򻙖{�OXn[�Q��?&�ѡ*b!��5x�Wn�D}�o'-�Y�@��IYh[�g�f��r�����]ԫ�gy�Z�9e����q�7�`�0!]�m#�!���o���jb'8��z�ka`,�JXf�R��fYQ�/a��V'f1߃���� '��Z�F��u;�*a��
y'y���?��f5����o��S�+���V�\��A>{��'��9VoHVr�D\m�ԗ:�	="P���uhY��
�$�L�{W�o���?��TV>�F��M���/S��2V�[ѷ���
�	0�q�/����SU@���<��}����|�eZ#�-������wLdm�R�7���T�b��*�m{�����b^���W��m��b{��,ف1�4Q��d�`)��u<���R:}�,�!N65K���J�E?���y1p�6u�oç����?�f�� ���s�_�C���Ŀ2L���=C���j����mWK�	�V��-�ky�B`���2��'���1��3H{Z�V�>#`� 6�2+d��/���GRTF��"�s��hT������#ǃ�Vw{��k�)������`q�|Ǻ��'�����x�X�~��M�K�O��m�V#vs{���M�6.�V��(�I���(7�/��0���M�w��������֗�]1-�|�|�qp�^b�N���"�w��?�/�:����N3~�}��i���Oз���8�8�������&�'��Zk"�:br_S&y]��P��Z�����Qm\l�1�}[v������j�+�Y�O���ʌx�s7v ��o?�&�㥲E�D��V�e�x��v�2�د������/�� أ�:k�]��|G񘍗����XF,N��w���qDO�X���_hޯ�ű%�_��N���m��#��[�u
�v,3�>	lSf%�N���p���Ľ-�M:�֡��CW6t�g�]�왞"VМ��HM�&�T=G��n&�g۫/�f�=��8��S�
s�o�׶:����t�����nM�Dߎ)c.p���s���V��w1��7x�q*>�g����@��|��+�*[Ҿ�5��'���^���8�}���t[�-�����QM��k�؛x��nGe<�xކx�6&��Ս�{�o^1\�7��������"^J�+W�c1��$�Y�M �a��_�_l�r.ћ\��H��\JLm��������}���w���B�u�d� ����i�k���̨�t`�i�w�#����2�W����ڬ��}��G<�4\��ѷ]�x8�����2*o&��> �N���&�$�%�Λh�n�e�Bb�ak���נo�:�o�%66t#�?�o���8����N��_�=�.��Kh���Wlo�~U7���&�=�+�q
>���B�ve�2b��[��K����	���4���멍ށG��}�
K&ԕ�{,N貎��4�As?���F��P56'6��}=����eVP��v�(�H���LnB_ѹ�|b���4c�ߋ�ӌџ��+�w��z-sh߄Ҳ=/�]Q����jA�������^b�z����K�<���+&#�����MxR`Q{Ҿ͂�ߥ�#'�����bٚu��۰��;�:�n_bOuu�w�*�JZgb��-s�����<�8fQ]���2�=o#�~*r4�v�&�^��>ꠇx5P׽�_R~�����Vu[��Ğ/�7�z��S���]׮A@R���y�ҹq	�n� &�F��e8��=��L#��P��B)�q`������Wd�q:y/骋_�o��ĥ��=�v�o㦳��D;EL�T�iNq^�MlC����b.�1�w$��j,$��4af*#���tj�k��D;mDl(�nӜ�U�jT��+/�8�z�������t(q�c�jh���;з�P�<y�L���Ҝ�<�����B5*r���%�'&<Yy~��]�g{`���}�%~�������T�xw������+-�X|�8����\�4����ָ5���O=�88Q��Db;����O�΃��JSǙ�<�F��AR����z9�����He_�5����Q�e�uy�".�/^����,�여���7�/�|G���X��n#\�'���,E\���6�}[�(f;%+�mG��?���j�����XLM��}eU /��z����{Eq^�_�́��{E��tE�~� F[��]eSJ�%�5��:��#��jL%ߍB�NWt`K�A��>X<����6���R���J�������RŃĶ���V[6T�v,���
�k���?X����6�#��o��qci��У�+��X
�	x)0��z��[��e���<b��i�A�����{MB��j��e���l����ʈ�Ļ�w��|�51���I�zG��_OYh[���6hڪ��ʬ��=}u��Ky��]1ֵ	���C)m��1���������� W��&ʈו[=�z?�
j���v�GM�[n�w��V��x�GNz�S�_�#)mö-p2y�"YU|��Z)�O�WЊ�b3�����-o#�;�_;��_�d=nˬ܆����9A���p;��d8��Q�k�W��o�w��/6NYhK�U诧���ꗳ=�S��I�xK�[�'v<�<'���;�Aa����rޓ��V����p��Ĵŵ�Y�Ui�k%@�+�;��k�,%�Q���W��Ҫ�5��\�{����诗��s�[J#�gѫ�/�Ǐ�,��c�W���5ǧ�_O}1o�['���_7}�ŵ
MC=���(냈�Ni�S����Z��ib2��íį�\��X�j��0q�Ri�̗�ش��������<�3�q�u&l�>ܥ�D�db�5ÿ��g�ؤ���; ��zu�y?1���eyM
~�:Vu`sq��R�����hX�|���[�Zz�L\���h��^K���(k`4�	xh���|P�+�c�t�ï�B=��zTV`C�K��h��3�q>�����y�Lt�΀�B� �H���F;$27}�#����.L\V���诧�ī�1��j�B�̴�Je� l\��t�>q�V����$�ˈw��W ��3A���_�	+l)p�8�� ��ؤ���ֽ���G�3�q�:V���3Б�5nŨt��L� �I��g Vg�J�;�m՞�΀��~q��G�V�Z���L ?��3a�[J���=��:V��;'��zF�z8Ɗ{��$�D���'D+� ���`Ź��Ԓ>�[~P9��?_KMv?q���; ��� �Ot&<��    IDAT)�0�X���v���� �[u,�S���&Pw F��Β�]��б�Clo������DKk*�������d�5�5�جh!�X���(b&��Uw[�[*]F�i�6�X��1�d@����.��[F�Y%|F��ӷ��Rg 8]����� ;�3`�=(N?��Z%|F��ӷ�r� h�C[�΀���E@=�8}+n;q���[�y��\�������+ �# �ي�T��%�ׇ[u��*y@���{�� �G ��0V�H�I]"Nߪ�q�� �y�b��)E?�	 �0��!��U)���Z׉��A������b@�P��b6Wg �I��ܵ��?CŊQ�d�_�g�n+��f�Dߓ����A=�ŊQ?7V/�et Ɣ�E�Po��ӿU����(Nq�V���gl�(�P8-�oŬ#N����K}��z��8��?��0�X��Sx����雎�@�j+f�8���0�@��� �������� x w �N}��#��MG��+�zS������+�Y��MG���$Nߊ� ~`�x�T���~vZ1� ������7u��H�K�P?;��w�; ���%N�8}�Qw�<Po(#�Po�/`u���p����,6�)�3�X&NߊQ��7������
_?et ���´��6�/p�++���O�%%|F���?���b���>_���~�]��G LMف�N��3Q��G �M���0m+���z��>O� w �M��b@���m�8}w �k�8}��7u � ��"�;�M�o:��x<Po�@# �u�� ԟ���8}��X��z�ˊQw ���� �F���>�wq����8}�i�V�8q�Y� (wC�A��ǊQ� l.N�t��������,� <Q�g��Y1��_��v��=Po�%�O��2: sJ��"��7�+�M���V�5��yp�ަ��/�㩌�z"�; �v�8�`7q�z;��?�@��8<Z��00^��s��$��o��K�~/��8V�z���� S[�~`q�V�}��?��2�;w �w <P�����p�Uk?q�׋ӷ���Ǌ~@^���Xq��:���<Xu�G�����8�$@� �o+N� 8B��Lm�@��#N?��z@}��w�:������ ~��
ӞG&[�7rq��nD2���Zj�)��<�������5R)߻et ,�3��+��[\)��H�e�<Xz/C�˵� ���F<3T�� �B����f�X����XrG�3 \�΀��8��e|H�^�>�[�m�K� �Ǉ5�:���L Ug�
[O���2>�� �̒>���m����{�N�K����J��c�ˊQN ���)����R�V�y�u�L����,��M�L]��W+�# �('�Oy�0��� 6^�΄���q����X)6�?��i�+���Z���<Tg������Ҧ� 4�´��	��7o$2��]��*0�Xޢ���⠴E�
��zZ�F%.�U�!t�Qi?��P�k�^�a�X���D�����|T���s�Ot��&��hzYTV�Β>�[� 4��t��Xh��y���+u���(qɲ: ��J��n���+r�y|LpݝLm88W�	+�z���: ���%}V7^|x-�������*��>�Q�LX�^E� ���G��G=���i���'�􏇉�>K���Q��x��/�D�w��$`��/I[\����^K;�/��}�M��x �=p<�U�j��F;�v`|8mq-�/��n�b&�\��F;�I�NG�Ri)0.i����7�pc:�]����׈q2��/ۥ-��hG�2�u�_M[\+Ѧ�v���'^ۨ���q[��s��)��ݫ��J��-��%��/.ǿ��`����z�^�y��z�[�S�b�����}�j(n#��SV<�x)޸C�诅��޴ŵ|�u�?<�?/#�=�I��$�C�O�_%�A|q�+��x���˫*�cз�X 운�V���i{���0i�m��'&�ߋ��(ǔ]1e�}��e�Y�19�+n���ju���;��)m]Y�ؑT}}�;�AeJӀ��E-�;�ZC%�:�
�"n �ï��ې�r��"w��C~ː�'�,���n�$���H/&�%�G���*�N���/UKcu�q�m=0|V@>>��zO���*�%��}ۧ��J��D&��;�*�~�mx�x*'�oぱxy�Bې��h��00�������!�����I?(�Β�m�v�qpx�*��$~Q��w`, ���ܶr{4�����$�v�rL$V�<�������Rs��}E��|b3+��з�`1�)a�mp�_���,>��ح7��"�i�Ъb��/�	��W��R�bb7�	Ū�:��ѷ�`q���dK�۱�,���Sy>�\�Ɗ�����̧�WX.q�tЊ���\Q� �NWt�ؒ��[��+��IV��ڎ�PIݶ��X�Z��T�Lwu���&�*[�&��F�W?)�N~�B��_���Srۏ��XF�[+E_q�Ńx�`QǢoǕ�#��J�^���;����$��؏8�^ݦ9�
֥�x�n���[�_�ԏ_p1�v\Y, ^��Z��D���ueq%^\�	�7�si�*�.T�B���\Q܌����'�_W��(�w�^��O�m��%Q��.����3�8�P�f�+�+1ט����m��o��ĩ��Du�dkF�~C��&����އ�����6�5c��Wf��K�����Y�|���;�Q��P���f�U[w��Cߎ9�g�kV�w��М�|�J`��D�nC�%Čfw�V��8@�Nۉ��&�a7bu��Ӱ3f6�����آ��m���o��ą��Ij��v.B�>ÉZ���ı4﴾��I2?��[[������2��L�:���S���G�L��k�g��n�����r�e�K����F,�QWt�18��
n�É��vn��8A}�������ˈ��lx��&���9���ڭ���w�Wvαx}��B�B�f�ƥ�A��H���ާ�~��*i�5�Kз]����k������:5�o�v�ep-�6+�ƛ~���Y��H\�.�bnCb�u��K���vkn
�m���!rw�nc`�6+W�ь/��P�L��Z4�$#��ۊ�,�T�������
�%�ˀ7v_��r(ѣV�Y1���n�#�4���w�Rk�����s<��+�M��݆ڀ8�u:��)���Z[���۫춿x'��F���sh^G�}%�Sl���(���{��T����x��rb��O\�b^u{��%���E����&vs��R���J��v،�-ͭ"����j�U��JF�%���M:�1���-��=I��F���M$Nl�!-�� �!g%Jk2�a���đ����������3R��Z6�#'�{ܩ��3թ�2#���g��݁Q�L%��/p�:#��Ft�6q��F���?�K��}�I�O|��%�O�D�+�gub1�`�NlKt���.bUƣ�����N�;�3���߉��ӉN@v���jDG�P�eİT��FtWg$s�돳v�Zx8��:#51�X�y�:#��_�7�p� .��{��N��췣TV��(��j+b�eu{9����D���ۭ�����h�<bt����!F~H�OW_PE��VOc�<���y�|�p}}�������SV�YV'����^�eWLC
,D�f�<�	��c
�ќ}7��>b���4�5�u��@}O*�Cs۪DL~S��#��O��c(���$uە��?$�U�K�����<���n\�ɿ�ڞ��nu�9�9Ą?�	Ą7u��;�wK\̀��?I���?��2j[�S�#:�^�6|?F�vE��U�7Ҳ�mI̴W_�C�e�Ij��6&�S��͡�[����:}�u���͆�Xbi���J�CLn���F�P���ڸ�8���g}�1��7ܘ	���ng�m@r����?LTM5�X�n7G5q:��:�L=ę��N,�	���>�ez�S�r?��h��ۈ-r���Hw_����v�з�p�`�$5a���_)p06U4؁x��&�,|�fQ���z����͒X��ž��@��7ۆ�߶��Ǖ�t��������MTf�0���~E��;�*�o�&����.z�/�X�^��-���z�b#�����Wg�+z+���c�^2H{��F=�J�2��oB�C�(�KX�6�L�ST��ch��Y�������se�x{�қ�I�o���┅n�����6ǃ��+l=���׈�v]Q�^���f]�,�c�xa�B����>�з�#��6� �l9�1�']�ͺ�|�20n F$,w��\��]�W�r�\;�o'w>��26
8��20�IY��^	LG߶m�����ܩM#�]O���}�&G��o��qM���D�S�;Iu7=f�Ǜ\��?�v,�5�V#;��o��qX������'зs�b>��˳��;}{�^�P6�Z9�X������/i�k�d���i��,b)ں�j�ֳ�s���,�YJG��]����L�x}��-�����W���~`�,i��\��Fꋟ�-�0��,��Y�K\�
�ا�9y�V.'�-��f�����4���)i�k+��z�^��A.18�Xzf:_F-�9��IKlV����T}q|��ʍ&Y9�vN|��h�`��/��oΊ'�Y�� �W�'.��Xb������3�]_I2c�8���?~���f���{���Նo�����b��I����q���XOV�K�_/}1�������o�.�6�&�p��#,�>AҲf���������N[\����Tb��$q>&N�Z&·�(``[`;`��?oI�6��"b�㻁[��:qWE�[�� �O�����3�� �|T�	�@�"u&��Q�f�Ds�΀�����y�Ļ���/Otbn�?� �_�}�@��[�� ����-&:�w�3b��$��&���5�l���_K\V��|�M7/�2k���-'F��%.�YV��C��lm��A� P?�N\V�,���H�R�Yn�@��YN�5Y#qYm���[�t�8}3����t|��lj�JC�_��3�ͬ.F��>��Z��oD��5k���c�I�L*�W y8�ء��ͬ:��?�}W��6s �?��8}3�΁��H^��D�����������̪������ m��q�υ�>y)�Lm<1�W���%��[�# �XH�P���fͷ+��"W����z� �����w�of���� �;u��ܨG v�of������:f��!�^U��{ }�L�����O_D
� �e9p�0�u�fM6�Z�����[�; ��F�v��0}3Kk`�8�ӷw ��Oq�[��7�t�Sg�X`p ?7����Ysm)N.p�8��@~���_G������8�k�M�,� �g)�	P�*L���Z_������w �t�0���i�YZ����F��Ӄ´=`�\�W ��ӷ~��ӣ´=`�\�9>w�ӷ~�ȓ���0m3Kgq��C���w ���0��´�,u��)`�8֏; yZ$L� �f�$N�q�6�; yZ&L{0R������ <!N�p Oʍ� Ɖ�7��_,�o������=`�<���?%N�p O�#y�s�,������ � �I��� �&R�,�G l w �� ,�mf�G ƈӷ��Ӛ´�0k&w ��ȓ����5���^]���@��Gv*�!6�t��_K���@��# ޭˬ���5�wNV��l&L�aa�f�Γhw�G��@~6E��O�2k���\q6�o����m����of�(O�B����@~v�?C����s�8}w 2�@~���8}3K�>q��N������� �!N���Qw v�o������Β���ofi�+N`�8��@^����fSw z�]�y�w �r�8����YZ��3 �΀w �1x�8��ofi�D��W=�i� �� `5q��of��$N/�G�[�; �x�8����<�Yz7���G��@����z�ǅ�Yz� ���0w r�"�ge{�߬�Tg x	q�	���7�3 \�΀�U�&�8X=��Lns����XLM]P3��Yh�9ˁ�'/���G �G�ע_dfչD�b�Ӛ�L������& oPg8O�3�������Ug�L�-����.��ee��g�թj�������Y�������:uAmp~��y�y&^�o�Fg�3�q�:m��N.a�F�3��Ы�~i-t��7���ە�C��j��;1i�vWg���`�N� S:�~ʀ�7��e���$>J,���<g {��0��FZ9w 4v"���X��������z�:�4`����$q���������~�U������N�P9w 4r��z?p�:ֵI�(����;�v^ϤNl�;��A<į���7���tbS���8�Vr@csu�[ [=L!&KF|�n�����mЉ#���������A~~����8}��|�������KiE� ��O.'�L=Y�h��_ ����r�&1Oym����f��9ڛ���E�.�_�_@���:f �D~�m�1�Cy-�'�Yr�G{�����6������RV�L���1?�M^AL�T���#Yk�������`�� �$b�E� �)����u�ڪ�?�G���o��,?A�-'��|g�6!~��B����l�x�X�&� �۷|?i��2�y�7\_�왶�F����}��-?�v��@[��oӁ��e6��	�o��� xS����ĩk���9z���;���1�q�`q\�r�e��o���{�Cal���M�ˈ�49�Q��'u��,vKVz�̬���[Q�L�*g�Y�6�}[6=��Թn�z�m��X �NUf9�����������x�=fK��s���;�>����xa�
0����x����Cdl�'��U�W��z`�U�UL�ϗ�x�j0����o����Ĝ�w�������t�?�ߏ8n�}���0��*IM�el"������<[x�8��q G����g�gB�Ό�,Ee�����߀É�����D}L%����±���Ka�d;�,�뻛xs��0����߀�<T����>rw8� �6p/�&Yj���v������q�׊Y�����M�<?A}�h,p
޳�α�X)�;�D��H�w�bV3�S�/���	W�Q�����5�~%0���R�n����ߠ�!��Lb�q����71n%�`�������2⨒�Ƭ�6$63QߔE�"��O�ۉCh�u�Hs�C��5�u�u)��+�z����o�2b	�e�;Ip�5���H����D�F_We��ZeV�YS��ZV<��z��}�9��O��u�/p��)3���f�\��F-3. �;�n"���rh�'Ĳ:����2�����K�'�FZ��L����ؤ�j*�z���ɡ�SѝJ7�00oy�k�WyUe�l���iˎ���IM�X����H���Vep0��2���{ʪ,��X���"�G�sWZ�}}8򊳨f��K��*�-���Zfp&�9UL'�W��C�Wׁ#�8�t�����1e<�'��68��M
�wSmG`pK�29�?���;��r��K��53+����o�1�د=嫁	���Q��:��B����N|_|�j�PX��}��ۚ����:#�M>M,�ZR��"��}a����G�+Kn'�m~x��ӻ������|,1Ys*�knm`�NlELTQU!�>|����A��o]v,b�n�{ꌘ5�$�4�=�*�>b���Rj.~��˔2��ۼ���Ne��?��8KT�*����j�� .� �U��}�Yez�K���U�\�K����C�#E�ک��=]m�+q>��2�uSf<��7�K,�H7�+  �IDAT6��V+�̬b�E��*� �b�;��f���,��a�C���.�9u?�z������V���2���Faf2��-l/ ^ɪWlE� ��[4���B���l	|�����Ǣq�k���	}3�S�1�v�k0���į�%����g�M��������;��)_݌!���}����A!�њYf����e���KyzX�Χ+>|�8��iF GW��g��b	�ڳ�ì���#mg����&�"F4Vذ���c�=�%�'6��x�,L��p-�NL�h��l�Q�;�3"���xf���Qaf5�p���c�q���a�L~FsV�9!Fg̬�F��8�K�/���%6�R�O[�t<�ϬQv�B�pqD�O�
g��L�wn�-��ͬ�F���wh�b�}[�c�y�۬�q����5�N�?�?t�ˈ�r붑��N���ۯ�1����kҬEFƣUœ��w��b�u;6)N�������E�FM��o]�~��=�w'�����C�NM�{�-���j'⥂���`��W��5���ўc�SǍx�5�p'`8q���l�v.G�Ъs��5܊�aq'`���v��������,������zïr���o�c�E�y���UdM����n�A`�jں�Y��S�lY�F���ي8a�C�+���=��`�Z�#����[��~3Kh?bB��a�[�+)Lc4�7�ׁ"&��,\�ff������/��x�
�RL�]��%���K�;3�aMl#:��Pǿ�r�3�*F}M��E�AI>����&G�p�:�)^�V�7��.R}��01�M&�x�ò���_J�Yپ���(+�?6-�����|���!�����7
�#�k�H,~lWrݘ�%7�ح�A�Ӳ���Z�� F�7?%�ޚ���D�4�#p#�y]�ߡ�f�s���3k����k�?l����ˮKj�����a��x�^3k���w���?����~��H�z�����63�Ֆ�Y��?�WĿК`=�讣e��È���Zo
p"��䶔X���u�<�'�]G���mVE����h$�"b�������ׄz�������-sB5E23k�u�_�wR��\�=�Zrk���P��3��{UZ3��!&�XH�/���N����n[ o �\<Ċ��Y�?����V�]3�vX��\�2�YA���,���2E���$Nܬ��a�Ֆ'6Y�M�&�[�g�K��$��^\���wff���5ͺ���yk��� ��G����e�33333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333k���jb����a    IEND�B`�
```


## static/style.css

```css

* {
    box-sizing: border-box;
  }
  body {
    margin: 0;
    font-family: 'Helvetica Neue', Arial, sans-serif;
    background: #f0f2f5;
    color: #333;
    line-height: 1.6;
  }
  .container {
    max-width: 1200px;
    margin: 20px auto;
    padding: 0 20px;
  }
  .card {
    background: #fff;
    border-radius: 8px;
    padding: 20px;
    margin-bottom: 20px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
  }
  
  
  .navbar {
    background: #2c3e50;
    color: #ecf0f1;
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 10px 20px;
    position: relative;
  }
  .navbar .logo {
    font-size: 1.8em;
    text-decoration: none;
    color: #ecf0f1;
  }
  .navbar .nav-links {
    display: flex;
    gap: 15px;
  }
  .navbar .nav-links a {
    text-decoration: none;
    color: #bdc3c7;
    padding: 8px 12px;
    transition: background 0.3s, color 0.3s;
    border-radius: 4px;
  }
  .navbar .nav-links a:hover,
  .navbar .nav-links a.active {
    background: #34495e;
    color: #ecf0f1;
  }
  
  
  .nav-toggle {
    display: none;
  }
  .nav-toggle-label {
    display: none;
    cursor: pointer;
  }
  .nav-toggle-label span,
  .nav-toggle-label span::before,
  .nav-toggle-label span::after {
    display: block;
    background: #ecf0f1;
    height: 3px;
    width: 25px;
    border-radius: 3px;
    position: relative;
    transition: all 0.3s ease;
  }
  .nav-toggle-label span::before,
  .nav-toggle-label span::after {
    content: '';
    position: absolute;
  }
  .nav-toggle-label span::before {
    top: -8px;
  }
  .nav-toggle-label span::after {
    top: 8px;
  }
  @media (max-width: 768px) {
    .nav-links {
      position: absolute;
      top: 100%;
      right: 0;
      background: #2c3e50;
      flex-direction: column;
      width: 200px;
      transform: translateY(-200%);
      transition: transform 0.3s ease;
    }
    .nav-links a {
      padding: 15px;
      border-bottom: 1px solid #34495e;
    }
    .nav-toggle:checked ~ .nav-links {
      transform: translateY(0);
    }
    .nav-toggle {
      display: block;
    }
    .nav-toggle-label {
      display: block;
    }
  }
  
  
  .gallery {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
    gap: 15px;
  }
  .gallery-item {
    position: relative;
    overflow: hidden;
    border-radius: 8px;
    transition: transform 0.3s ease, box-shadow 0.3s ease;
    height: 150px; 
  }
  .gallery-item:hover {
    transform: scale(1.05);
    box-shadow: 0 4px 10px rgba(0,0,0,0.2);
  }
  .img-container {
    height: 100%;
    overflow: hidden;
  }
  .img-container img {
    height: 100%;
    width: auto;
    display: block;
    margin: 0 auto;
    object-fit: cover;
    cursor: pointer;
  }
  
  
  .current-image-container img,
  .last-sent-img {
    max-width: 300px;
    max-height: 300px;
    width: auto;
    height: auto;
    margin: 0 auto;
    display: block;
  }
  
  
  .overlay {
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: rgba(0,0,0,0.5);
    opacity: 0;
    transition: opacity 0.3s ease;
    border-radius: 8px;
  }
  .img-container:hover .overlay {
    opacity: 1;
  }
  
  .crop-icon {
    position: absolute;
    top: 5px;
    left: 5px;
    width: 30px;
    height: 30px;
    display: flex;
    align-items: center;
    justify-content: center;
    color: #fff;
    background: rgba(0,0,0,0.5);
    border-radius: 50%;
    cursor: pointer;
    z-index: 10;
  }
  .crop-icon:hover {
    background: rgba(0,0,0,0.7);
  }
  
  .delete-icon {
    position: absolute;
    top: 5px;
    right: 5px;
    width: 24px;
    height: 24px;
    cursor: pointer;
    z-index: 10;
    color: #fff;
    font-size: 20px;
  }
  
  
  .favorite-icon {
    position: absolute;
    top: 5px;
    left: 50%;
    transform: translateX(-50%);
    width: 24px;
    height: 24px;
    cursor: pointer;
    z-index: 10;
    color: #fff;
    font-size: 20px;
  }
  
  .send-button {
    position: absolute;
    left: 50%;
    top: 50%;
    transform: translate(-50%, -50%);
    background: #28a745;
    color: #fff;
    border: none;
    padding: 8px 12px;
    border-radius: 4px;
    cursor: pointer;
    transition: background 0.3s ease;
  }
  .send-button:hover {
    background: #218838;
  }
  
  
  .lightbox-modal {
    display: none;
    position: fixed;
    z-index: 4000;
    left: 0;
    top: 0;
    width: 100%;
    height: 100%;
    overflow: auto;
    background-color: rgba(0,0,0,0.9);
  }
  .lightbox-content {
    margin: auto;
    display: block;
    max-width: 90%;
    max-height: 80%;
    animation: zoomIn 0.3s;
  }
  @keyframes zoomIn {
    from { transform: scale(0.8); opacity: 0; }
    to { transform: scale(1); opacity: 1; }
  }
  .lightbox-close {
    position: absolute;
    top: 20px;
    right: 35px;
    color: #f1f1f1;
    font-size: 40px;
    font-weight: bold;
    transition: 0.3s;
    cursor: pointer;
  }
  .lightbox-close:hover,
  .lightbox-close:focus {
    color: #bbb;
  }
  #lightboxCaption {
    text-align: center;
    color: #ccc;
    padding: 10px 0;
  }
  
  
  .popup {
    position: fixed;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    background: rgba(255,255,255,0.95);
    border: 2px solid #ccc;
    padding: 30px;
    box-shadow: 0 4px 10px rgba(0,0,0,0.3);
    z-index: 10000;
    text-align: center;
    font-size: 1.5em;
    display: none;
    border-radius: 8px;
    animation: popupFade 0.5s ease;
  }
  @keyframes popupFade {
    from { opacity: 0; transform: translate(-50%, -50%) scale(0.8); }
    to { opacity: 1; transform: translate(-50%, -50%) scale(1); }
  }
  
  
  .progress-container {
    width: 60%;
    margin: 20px auto;
    background: #ddd;
    border-radius: 5px;
    display: none;
  }
  .progress-bar {
    width: 0%;
    height: 30px;
    background: #28a745;
    border-radius: 5px;
    transition: width 0.4s ease;
    color: #fff;
    line-height: 30px;
    font-size: 1em;
    text-align: center;
  }
  
  
  .modal {
    display: none;
    position: fixed;
    z-index: 3000;
    left: 0;
    top: 0;
    width: 100%;
    height: 100%;
    background-color: rgba(0,0,0,0.7);
    padding: 20px;
  }
  .modal-content {
    background: #fff;
    margin: 5% auto;
    padding: 20px;
    border-radius: 8px;
    max-width: 500px;
    position: relative;
  }
  .close {
    position: absolute;
    top: 10px;
    right: 15px;
    font-size: 1.5em;
    color: #333;
    cursor: pointer;
  }
  
  
  input[type="submit"],
  button,
  .primary-btn {
    background: linear-gradient(to right, #28a745, #218838);
    border: none;
    color: #fff;
    padding: 10px 15px;
    border-radius: 4px;
    cursor: pointer;
    font-size: 1em;
    transition: background 0.3s ease;
  }
  input[type="submit"]:hover,
  button:hover,
  .primary-btn:hover {
    background: linear-gradient(to right, #218838, #1e7e34);
  }
  
  
  input[type="text"],
  input[type="password"],
  select {
    width: 100%;
    padding: 10px;
    margin: 10px 0;
    border: 1px solid #ccc;
    border-radius: 4px;
  }
  label {
    font-weight: bold;
  }
  
  
  .calendar {
    width: 100%;
    border-collapse: collapse;
    margin-top: 20px;
  }
  .calendar th,
  .calendar td {
    border: 1px solid #ddd;
    padding: 8px;
    text-align: center;
  }
  .calendar th {
    background: #f8f8f8;
  }
  .calendar .droppable.over {
    background: #dff0d8;
  }
  
  
  .footer {
    text-align: center;
    padding: 15px;
    background: #2c3e50;
    color: #bdc3c7;
    position: fixed;
    bottom: 0;
    width: 100%;
  }
  
  
  a:focus,
  button:focus,
  input:focus {
    outline: 2px solid #2980b9;
    outline-offset: 2px;
  }
```


## static/trash-icon.png

```png
�PNG

   IHDR   0   0   W��   	pHYs     ��  �IDATx��KNA��^A]�]|$�H�"}
ءG���[�`���	P��63�!H��46Z_R!���ꯙE�R� T�պf��x�Q� -�S)��h�{Cd�p��Q)b��]�����m�J�v�$}�����<�V(�R�����rg�o�.��*5	Z����W��&�d`��LG5&L�-��Uj���o��
u�V�e�AU$L�3��50��!�goઌ/�8�q_2��'���H$
�!#D2Ba���P2B$#ƿ!η/�櫘Ѳ5�*_2$}�����c�����N������sq@W?#:�R x�:�{o<�܁U(a��e����)�SU3&:b�e���X�C���E00̚��'���; �i������ݰE�7�H    IEND�B`�
```


## templates/base.html

```html
<!doctype html>
<html>
  <head>
    <title>{% block title %}InkyDocker{% endblock %}</title>
    <link rel="stylesheet" href="{{ url_for('static', filename='style.css') }}">
    
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    
    <link rel="stylesheet"
          href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css"
          crossorigin="anonymous"
          referrerpolicy="no-referrer" />
    
    <link rel="stylesheet"
          href="https://cdnjs.cloudflare.com/ajax/libs/cropperjs/1.6.2/cropper.min.css"
          crossorigin="anonymous"
          referrerpolicy="no-referrer" />
    <style>
      /* Make the site scroll fully behind the footer */
      html, body {
        height: 100%;
        margin: 0;
        padding: 0;
      }
      .wrapper {
        min-height: 100%;
        display: flex;
        flex-direction: column;
      }
      .main-content-wrapper {
        flex: 1 0 auto;
      }
      .footer {
        flex-shrink: 0;
      }
    </style>
    {% block head %}{% endblock %}
  </head>
  <body>
    <div class="wrapper">
      <nav class="navbar navbar-expand-lg navbar-dark bg-dark" role="navigation" aria-label="Main Navigation">
        <div class="container-fluid">
          <a href="{{ url_for('image.upload_file') }}" class="navbar-brand">InkyDocker</a>
          <button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarNav"
            aria-controls="navbarNav" aria-expanded="false" aria-label="Toggle navigation">
            <span class="navbar-toggler-icon"></span>
          </button>
          <div class="collapse navbar-collapse" id="navbarNav">
            <ul class="navbar-nav ms-auto">
              <li class="nav-item">
                <a class="nav-link {% if request.endpoint.startswith('image.') %}active{% endif %}" href="{{ url_for('image.upload_file') }}">Gallery</a>
              </li>
              <li class="nav-item">
                <a class="nav-link {% if request.endpoint.startswith('schedule.') %}active{% endif %}" href="{{ url_for('schedule.schedule_page') }}">Schedule</a>
              </li>
              <li class="nav-item">
                <a class="nav-link {% if request.endpoint.startswith('settings.') %}active{% endif %}" href="{{ url_for('settings.settings') }}">Settings</a>
              </li>
            </ul>
          </div>
        </div>
      </nav>
      <div class="main-content-wrapper">
        {% block content %}{% endblock %}
      </div>
      <footer class="footer bg-dark text-light text-center py-3">
        <p class="mb-0">© 2025 InkyDocker | Built with AI by Me</p>
      </footer>
    </div>
    
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
    
    <script src="https://cdnjs.cloudflare.com/ajax/libs/cropperjs/1.6.2/cropper.min.js"
            crossorigin="anonymous"
            referrerpolicy="no-referrer"></script>
    {% block scripts %}{% endblock %}
  </body>
</html>
```


## templates/index.html

```html
{% extends "base.html" %}
{% block title %}Gallery - InkyDocker{% endblock %}

{% block content %}
<div class="container">
  
  <div class="card current-image-section">
    <h2 id="currentImageTitle">Current image on {{ devices[0].friendly_name if devices else 'N/A' }}</h2>
    <div class="current-image-container">
      {% if devices and devices[0].last_sent %}
        <img
          id="currentImage"
          src="{{ url_for('image.uploaded_file', filename=devices[0].last_sent) }}"
          alt="Current Image"
          class="last-sent-img small-current"
          loading="lazy"
        >
      {% else %}
        <p id="currentImagePlaceholder">No image available.</p>
      {% endif %}
    </div>
    {% if devices|length > 1 %}
      <div class="arrow-controls">
        <button id="prevDevice">&larr;</button>
        <button id="nextDevice">&rarr;</button>
      </div>
    {% endif %}
  </div>

  
  
  
  <div id="uploadPopup" style="position: fixed; top: 50%; left: 50%; transform: translate(-50%, -50%); background: rgba(0,0,0,0.8); color: #fff; padding: 15px 20px; border-radius: 5px; z-index: 1000; display: none;">
    <div class="spinner"></div> Processing...
  </div>

  
  <div class="main-content">
    
    <div class="left-panel">
      <div class="card device-section">
        <h2>Select eInk Display</h2>
        {% if devices %}
          <div class="device-options">
            {% for device in devices %}
              <label class="device-option">
                <input
                  type="radio"
                  name="device"
                  value="{{ device.address }}"
                  data-index="{{ loop.index0 }}"
                  data-friendly="{{ device.friendly_name }}"
                  data-resolution="{{ device.resolution }}"
                  {% if loop.first %}checked{% endif %}
                >
                {{ device.friendly_name }}
              </label>
            {% endfor %}
          </div>
        {% else %}
          <p>No devices configured. Go to <a href="{{ url_for('settings.settings') }}">Settings</a>.</p>
        {% endif %}
      </div>

      <div class="card upload-section">
        <h2>Upload Images</h2>
        <form id="uploadForm" class="upload-form" method="post" enctype="multipart/form-data" action="{{ url_for('image.upload_file') }}">
          <input type="file" name="file" multiple id="fileInput" required>
          <br>
          <input type="submit" value="Upload">
          <div class="progress-container" id="progressContainer" style="display: none;">
            <div class="progress-bar" id="progressBar">0%</div>
          </div>
          <div id="uploadStatus"></div>
        </form>
      </div>
      
      
    </div>

    
    <div class="gallery-section">
      <h2>Gallery</h2>
      <input type="text" id="gallerySearch" placeholder="Search images by tags..." style="width:100%; padding:10px; margin-bottom:20px;">
      <div id="searchSpinner" style="display:none;">Loading...</div>
      <div class="gallery" id="gallery">
        {% for image in images %}
          <div class="gallery-item">
            <div class="img-container">
              <img src="{{ url_for('image.uploaded_file', filename=image) }}" alt="{{ image }}" data-filename="{{ image }}" loading="lazy">
              <div class="overlay">
                
                <div class="favorite-icon" title="Favorite" data-image="{{ image }}">
                  <i class="fa fa-heart"></i>
                </div>
                
                <button class="send-button" data-image="{{ image }}">Send</button>
                
                <button class="info-button" data-image="{{ image }}">Info</button>
                
                <div class="delete-icon" title="Delete" data-image="{{ image }}">
                  <i class="fa fa-trash"></i>
                </div>
              </div>
            </div>
          </div>
        {% endfor %}
      </div>
    </div>
  </div>
  
  <div style="height: 100px;"></div>
</div>


<div id="infoModal" class="modal" style="display:none;">
  <div class="modal-content" style="max-width:800px; margin:auto; position:relative; padding:20px;">
    <span class="close" onclick="closeInfoModal()" style="position:absolute; top:10px; right:15px; cursor:pointer; font-size:1.5em;">&times;</span>
    <h2>Image Info</h2>
    <div style="text-align:center; margin-bottom:20px;">
      <img id="infoImagePreview" src="" alt="Info Preview" style="max-width:300px;">
      <div style="margin-top:10px;">
        <button type="button" onclick="openCropModal()">Crop Image</button>
      </div>
    </div>
    <div style="display:flex; gap:20px;">
      
      <div style="flex:1;" id="infoLeftColumn">
        <p><strong>Filename:</strong> <span id="infoFilename">N/A</span></p>
        <p><strong>Resolution:</strong> <span id="infoResolution">N/A</span></p>
        <p><strong>Filesize:</strong> <span id="infoFilesize">N/A</span></p>
      </div>
      
      <div style="flex:1;">
        <div style="margin-bottom:10px;">
          <label><strong>Tags:</strong></label>
          <div id="tagContainer" style="margin-top:5px; margin-bottom:10px;"></div>
          <div style="display:flex;">
            <input type="text" id="newTagInput" style="flex-grow:1;" placeholder="Add a new tag...">
            <button type="button" onclick="addTag()" style="margin-left:5px;">Add</button>
          </div>
          <input type="hidden" id="infoTags">
        </div>
        <div style="margin-bottom:10px;">
          <label><strong>Favorite:</strong></label>
          <input type="checkbox" id="infoFavorite">
        </div>
        <div id="infoStatus" style="color: green; margin-bottom:10px;"></div>
        <button onclick="saveInfoEdits()">Save</button>
        <button onclick="runOpenClip()">Re-run Tagging</button>
      </div>
    </div>
  </div>
</div>


<div id="lightboxModal" class="modal lightbox-modal" style="display:none;">
  <span class="lightbox-close" onclick="closeLightbox()">&times;</span>
  <img class="lightbox-content" id="lightboxImage" alt="Enlarged Image">
  <div id="lightboxCaption"></div>
</div>


<div id="cropModal" class="modal" style="display:none;">
  <div class="modal-content">
    <span class="close" onclick="closeCropModal()" style="cursor:pointer; font-size:1.5em;">&times;</span>
    <h2>Crop Image</h2>
    <div id="cropContainer" style="max-width:100%; max-height:80vh;">
      <img id="cropImage" src="" alt="Crop Image" style="width:100%;">
    </div>
    <div style="margin-top:10px;">
      <button type="button" onclick="saveCropData()">Save Crop</button>
      <button type="button" onclick="closeCropModal()">Cancel</button>
    </div>
  </div>
</div>
{% endblock %}

{% block scripts %}
<script>
document.addEventListener("DOMContentLoaded", function() {
  // Inject dynamic CSS for info button, favorite icon, and tag boxes
  const styleTag = document.createElement('style');
  styleTag.innerHTML = `
    .info-button {
      position: absolute;
      left: 50%;
      bottom: 10px;
      transform: translateX(-50%);
      background: #17a2b8;
      color: #fff;
      border: none;
      padding: 8px 12px;
      border-radius: 4px;
      cursor: pointer;
      transition: background 0.3s ease;
      font-size: 0.9em;
    }
    .info-button:hover {
      background: #138496;
    }
    .favorite-icon i {
      font-size: 1.5em;
      color: #ccc;
      transition: color 0.3s;
    }
    .favorite-icon.favorited i {
      color: red;
    }
    .tag-box {
      display: inline-block;
      background-color: #e9ecef;
      border-radius: 4px;
      padding: 5px 10px;
      margin: 3px;
      font-size: 0.9em;
    }
    .tag-remove {
      margin-left: 5px;
      cursor: pointer;
      font-weight: bold;
      color: #dc3545;
    }
    .tag-remove:hover {
      color: #bd2130;
    }
  `;
  document.head.appendChild(styleTag);
});

/* Lightbox functions */
function openLightbox(src, alt) {
  const lightboxModal = document.getElementById('lightboxModal');
  const lightboxImage = document.getElementById('lightboxImage');
  const lightboxCaption = document.getElementById('lightboxCaption');
  lightboxModal.style.display = 'block';
  lightboxImage.src = src;
  lightboxCaption.innerText = alt;
}
function closeLightbox() {
  document.getElementById('lightboxModal').style.display = 'none';
}

/* Debounce helper */
function debounce(func, wait) {
  let timeout;
  return function(...args) {
    clearTimeout(timeout);
    timeout = setTimeout(() => func.apply(this, args), wait);
  };
}

/* Gallery search */
const searchInput = document.getElementById('gallerySearch');
const searchSpinner = document.getElementById('searchSpinner');
const gallery = document.getElementById('gallery');

const performSearch = debounce(function() {
  const query = searchInput.value.trim();
  if (!query) {
    searchSpinner.style.display = 'none';
    location.reload();
    return;
  }
  searchSpinner.style.display = 'block';
  fetch(`/api/search_images?q=${encodeURIComponent(query)}`)
    .then(response => response.json())
    .then(data => {
      searchSpinner.style.display = 'none';
      gallery.innerHTML = "";
      if (data.status === "success" && data.results.ids) {
        if (data.results.ids.length === 0) {
          gallery.innerHTML = "<p>No matching images found.</p>";
        } else {
          data.results.ids.forEach((id) => {
            const imageUrl = `/images/${encodeURIComponent(id)}`;
            const item = document.createElement('div');
            item.className = 'gallery-item';
            item.innerHTML = `
              <div class="img-container">
                <img src="${imageUrl}" alt="${id}" data-filename="${id}" loading="lazy">
                <div class="overlay">
                  <div class="favorite-icon" title="Favorite" data-image="${id}">
                    <i class="fa fa-heart"></i>
                  </div>
                  <button class="send-button" data-image="${id}">Send</button>
                  <button class="info-button" data-image="${id}">Info</button>
                  <div class="delete-icon" title="Delete" data-image="${id}">
                    <i class="fa fa-trash"></i>
                  </div>
                </div>
              </div>
            `;
            gallery.appendChild(item);
          });
        }
      } else {
        gallery.innerHTML = "<p>No matching images found.</p>";
      }
    })
    .catch(err => {
      searchSpinner.style.display = 'none';
      console.error("Search error:", err);
    });
}, 500);

if (searchInput) {
  searchInput.addEventListener('input', performSearch);
}

/* Upload form */
const uploadForm = document.getElementById('uploadForm');
uploadForm.addEventListener('submit', function(e) {
  e.preventDefault();
  const fileInput = document.getElementById('fileInput');
  if (!fileInput.files.length) return;
  
  const formData = new FormData();
  for (let i = 0; i < fileInput.files.length; i++) {
    formData.append('file', fileInput.files[i]);
  }
  
  const selectedDevice = document.querySelector('input[name="device"]:checked');
  const deviceFriendly = selectedDevice ? selectedDevice.getAttribute('data-friendly') : "unknown display";
  
  const xhr = new XMLHttpRequest();
  xhr.open('POST', uploadForm.action, true);

  xhr.upload.addEventListener("progress", function(e) {
    if (e.lengthComputable) {
      const percentComplete = (e.loaded / e.total) * 100;
      const progressBar = document.getElementById('progressBar');
      progressBar.style.width = percentComplete + '%';
      progressBar.textContent = Math.round(percentComplete) + '%';
      document.getElementById('progressContainer').style.display = 'block';
      
      const popup = document.getElementById('uploadPopup');
      popup.style.display = 'block';
      popup.innerHTML = `<div class="spinner"></div> Uploading image to ${deviceFriendly}... ${Math.round(percentComplete)}%`;
    }
  });

  xhr.onload = function() {
    const popup = document.getElementById('uploadPopup');
    if (xhr.status === 200) {
      popup.innerHTML = `<div class="spinner"></div> Image uploaded successfully!`;
    } else {
      popup.innerHTML = `<div class="spinner"></div> Error uploading image.`;
    }
    setTimeout(() => {
      popup.style.display = 'none';
      location.reload();
    }, 1500);
  };

  xhr.onerror = function() {
    const popup = document.getElementById('uploadPopup');
    popup.innerHTML = `<div class="spinner"></div> Error uploading image.`;
    setTimeout(() => {
      popup.style.display = 'none';
    }, 1500);
  };

  xhr.send(formData);
});

/* Send image */
document.addEventListener('click', function(e) {
  if (e.target && e.target.classList.contains('send-button')) {
    e.stopPropagation();
    const imageFilename = e.target.getAttribute('data-image');
    const selectedDevice = document.querySelector('input[name="device"]:checked');
    if (!selectedDevice) return;
    
    const deviceFriendly = selectedDevice.getAttribute('data-friendly');
    const formData = new FormData();
    formData.append("device", selectedDevice.value);

    const baseUrl = "{{ url_for('image.send_image', filename='') }}";
    const finalUrl = baseUrl + encodeURIComponent(imageFilename);

    const xhr = new XMLHttpRequest();
    xhr.open('POST', finalUrl, true);

    xhr.upload.addEventListener("progress", function(ev) {
      if (ev.lengthComputable) {
        const percentComplete = (ev.loaded / ev.total) * 100;
        const popup = document.getElementById('uploadPopup');
        popup.style.display = 'block';
        popup.innerHTML = `<div class="spinner"></div> Sending image to ${deviceFriendly}... ${Math.round(percentComplete)}%`;
      }
    });

    xhr.onload = function() {
      const popup = document.getElementById('uploadPopup');
      if (xhr.status === 200) {
        popup.innerHTML = `<div class="spinner"></div> Image sent successfully!`;
      } else {
        popup.innerHTML = `<div class="spinner"></div> Error sending image.`;
      }
      setTimeout(() => {
        popup.style.display = 'none';
        location.reload();
      }, 1500);
    };

    xhr.onerror = function() {
      const popup = document.getElementById('uploadPopup');
      popup.innerHTML = `<div class="spinner"></div> Error sending image.`;
      setTimeout(() => {
        popup.style.display = 'none';
      }, 1500);
    };

    xhr.send(formData);
  }
});

/* Delete image */
document.addEventListener('click', function(e) {
  if (e.target && e.target.closest('.delete-icon')) {
    e.stopPropagation();
    const imageFilename = e.target.closest('.delete-icon').getAttribute('data-image');
    
    const deleteBaseUrl = "/delete_image/";
    const deleteUrl = deleteBaseUrl + encodeURIComponent(imageFilename);

    fetch(deleteUrl, { method: 'POST' })
      .then(response => response.json())
      .then(data => {
        if (data.status === "success") {
          location.reload();
        } else {
          console.error("Error deleting image:", data.message);
        }
      })
      .catch(error => {
        console.error("Error deleting image:", error);
      });
  }
});

/* Favorite toggle */
document.addEventListener('click', function(e) {
  if (e.target && e.target.closest('.favorite-icon')) {
    e.stopPropagation();
    const favIcon = e.target.closest('.favorite-icon');
    const imageFilename = favIcon.getAttribute('data-image');
    favIcon.classList.toggle('favorited');
    const isFavorited = favIcon.classList.contains('favorited');
    fetch("/api/update_image_metadata", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        filename: imageFilename,
        tags: [],  // do not modify tags in favorite toggle
        favorite: isFavorited
      })
    })
      .then(resp => resp.json())
      .then(data => {
        if (data.status !== "success") {
          console.error("Error updating favorite:", data.message);
        }
      })
      .catch(err => {
        console.error("Error updating favorite:", err);
      });
  }
});

/* Info Modal Logic */
let currentInfoFilename = null;

document.addEventListener('click', function(e) {
  if (e.target && e.target.classList.contains('info-button')) {
    e.stopPropagation();
    const filename = e.target.getAttribute('data-image');
    currentInfoFilename = filename;
    openInfoModal(filename);
  }
});

// Tag management functions
let currentTags = [];

function renderTags() {
  const tagContainer = document.getElementById('tagContainer');
  tagContainer.innerHTML = '';
  
  currentTags.forEach((tag, index) => {
    const tagElement = document.createElement('span');
    tagElement.className = 'tag-box';
    tagElement.innerHTML = `${tag} <span class="tag-remove" onclick="removeTag(${index})">×</span>`;
    tagContainer.appendChild(tagElement);
  });
  
  // Update the hidden input with comma-separated tags
  document.getElementById('infoTags').value = currentTags.join(', ');
}

function addTag() {
  const newTagInput = document.getElementById('newTagInput');
  const tag = newTagInput.value.trim();
  
  if (tag && !currentTags.includes(tag)) {
    currentTags.push(tag);
    renderTags();
    newTagInput.value = '';
  }
}

function removeTag(index) {
  currentTags.splice(index, 1);
  renderTags();
}

// Add event listener for Enter key on the new tag input
document.addEventListener('DOMContentLoaded', function() {
  const newTagInput = document.getElementById('newTagInput');
  if (newTagInput) {
    newTagInput.addEventListener('keypress', function(e) {
      if (e.key === 'Enter') {
        e.preventDefault();
        addTag();
      }
    });
  }
});

function openInfoModal(filename) {
  const imgUrl = `/images/${encodeURIComponent(filename)}?size=info`;
  fetch(`/api/get_image_metadata?filename=${encodeURIComponent(filename)}`)
    .then(resp => resp.json())
    .then(data => {
      if (data.status === "success") {
        document.getElementById('infoImagePreview').src = imgUrl;
        document.getElementById('infoFilename').textContent = filename;
        document.getElementById('infoResolution').textContent = data.resolution || "N/A";
        document.getElementById('infoFilesize').textContent = data.filesize || "N/A";
        
        // Set up tags
        currentTags = data.tags || [];
        renderTags();
        
        document.getElementById('infoFavorite').checked = data.favorite || false;
        document.getElementById('infoStatus').textContent = "";
        document.getElementById('infoModal').style.display = 'block';
      } else {
        document.getElementById('infoStatus').textContent = "Error: " + data.message;
        document.getElementById('infoModal').style.display = 'block';
      }
    })
    .catch(err => {
      console.error("Error fetching metadata:", err);
      document.getElementById('infoStatus').textContent = "Failed to fetch metadata. Check console.";
      document.getElementById('infoModal').style.display = 'block';
    });
}

function closeInfoModal() {
  document.getElementById('infoModal').style.display = 'none';
  currentInfoFilename = null;
}

function saveInfoEdits() {
  if (!currentInfoFilename) return;
  fetch("/api/update_image_metadata", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      filename: currentInfoFilename,
      tags: currentTags,
      favorite: document.getElementById('infoFavorite').checked
    })
  })
    .then(resp => resp.json())
    .then(data => {
      if (data.status === "success") {
        document.getElementById('infoStatus').textContent = "Metadata updated successfully!";
        setTimeout(() => { closeInfoModal(); }, 1500);
      } else {
        document.getElementById('infoStatus').textContent = "Error updating metadata: " + data.message;
      }
    })
    .catch(err => {
      console.error("Error updating metadata:", err);
      document.getElementById('infoStatus').textContent = "Failed to update metadata. Check console.";
    });
}

function runOpenClip() {
  if (!currentInfoFilename) return;
  fetch(`/api/reembed_image?filename=${encodeURIComponent(currentInfoFilename)}`)
    .then(resp => resp.json())
    .then(data => {
      if (data.status === "success") {
        currentTags = data.tags || [];
        renderTags();
        document.getElementById('infoStatus').textContent = "Re-ran tagging successfully!";
      } else {
        document.getElementById('infoStatus').textContent = "Error re-running tagging: " + data.message;
      }
    })
    .catch(err => {
      console.error("Error re-running tagging:", err);
      document.getElementById('infoStatus').textContent = "Failed to re-run tagging. Check console.";
    });
}

// Crop Modal Functions
let cropperInstance = null;

function openCropModal() {
  if (!currentInfoFilename) return;
  
  const cropModal = document.getElementById('cropModal');
  const cropImage = document.getElementById('cropImage');
  
  // Set the image source to the current image
  cropImage.src = `/images/${encodeURIComponent(currentInfoFilename)}`;
  
  // Show the modal
  cropModal.style.display = 'block';
  
  // Initialize Cropper.js after the image is loaded
  cropImage.onload = function() {
    if (cropperInstance) {
      cropperInstance.destroy();
    }
    
    // Import Cropper.js dynamically if needed
    if (typeof Cropper === 'undefined') {
      const script = document.createElement('script');
      script.src = 'https://cdnjs.cloudflare.com/ajax/libs/cropperjs/1.5.12/cropper.min.js';
      document.head.appendChild(script);
      
      const link = document.createElement('link');
      link.rel = 'stylesheet';
      link.href = 'https://cdnjs.cloudflare.com/ajax/libs/cropperjs/1.5.12/cropper.min.css';
      document.head.appendChild(link);
      
      script.onload = initCropper;
    } else {
      initCropper();
    }
  };
}

function initCropper() {
  const cropImage = document.getElementById('cropImage');
  cropperInstance = new Cropper(cropImage, {
    aspectRatio: NaN, // Free aspect ratio
    viewMode: 1,
    autoCropArea: 1,
    responsive: true,
    restore: true,
    guides: true,
    center: true,
    highlight: true,
    cropBoxMovable: true,
    cropBoxResizable: true,
    toggleDragModeOnDblclick: true,
  });
}

function closeCropModal() {
  const cropModal = document.getElementById('cropModal');
  cropModal.style.display = 'none';
  
  if (cropperInstance) {
    cropperInstance.destroy();
    cropperInstance = null;
  }
}

function saveCropData() {
  if (!cropperInstance || !currentInfoFilename) return;
  
  const cropData = cropperInstance.getData();
  
  fetch(`/save_crop_info/${encodeURIComponent(currentInfoFilename)}`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({
      x: cropData.x,
      y: cropData.y,
      width: cropData.width,
      height: cropData.height
    })
  })
  .then(response => response.json())
  .then(data => {
    if (data.status === 'success') {
      document.getElementById('infoStatus').textContent = 'Crop data saved successfully!';
      closeCropModal();
    } else {
      document.getElementById('infoStatus').textContent = 'Error saving crop data: ' + data.message;
    }
  })
  .catch(error => {
    console.error('Error saving crop data:', error);
    document.getElementById('infoStatus').textContent = 'Error saving crop data. Check console.';
  });
}

const prevButton = document.getElementById('prevDevice');
const nextButton = document.getElementById('nextDevice');
// Define devices array using template data
const devices = [
  {% for device in devices %}
    {
      friendly_name: "{{ device.friendly_name|e }}",
      address: "{{ device.address|e }}",
      last_sent: "{{ device.last_sent|e if device.last_sent is defined else '' }}"
    }{% if not loop.last %},{% endif %}
  {% endfor %}
];
let currentDeviceIndex = 0;

function updateCurrentImageDisplay() {
  const device = devices[currentDeviceIndex];
  const titleEl = document.getElementById('currentImageTitle');
  const imageEl = document.getElementById('currentImage');
  const placeholderEl = document.getElementById('currentImagePlaceholder');
  
  titleEl.textContent = "Current image on " + device.friendly_name;
  if (device.last_sent) {
    if (placeholderEl) {
      placeholderEl.style.display = 'none';
    }
    if (imageEl) {
      imageEl.src = "{{ url_for('image.uploaded_file', filename='') }}" + device.last_sent;
      imageEl.style.display = 'block';
    }
  } else {
    if (imageEl) {
      imageEl.style.display = 'none';
    }
    if (placeholderEl) {
      placeholderEl.style.display = 'block';
    }
  }
}

if (prevButton && nextButton) {
  prevButton.addEventListener('click', function() {
    currentDeviceIndex = (currentDeviceIndex - 1 + devices.length) % devices.length;
    updateCurrentImageDisplay();
  });
  nextButton.addEventListener('click', function() {
    currentDeviceIndex = (currentDeviceIndex + 1) % devices.length;
    updateCurrentImageDisplay();
  });
}

if (devices.length > 0) {
  updateCurrentImageDisplay();
}

// Bulk tagging moved to settings page
</script>
{% endblock %}
```


## templates/schedule.html

```html
{% extends "base.html" %}
{% block title %}Schedule - InkyDocker{% endblock %}
{% block head %}
  
  <link href="https://cdn.jsdelivr.net/npm/fullcalendar@6.1.8/main.min.css" rel="stylesheet">
  
  <script src="https://cdn.jsdelivr.net/npm/fullcalendar@6.1.8/index.global.min.js" defer></script>
  <style>
    #calendar {
      max-width: 1000px;
      margin: 40px auto;
      margin-bottom: 100px; /* Add more space at the bottom */
    }
    
    /* Style for the event content to show thumbnails */
    .event-thumbnail {
      width: 100%;
      height: 40px;
      object-fit: cover;
      border-radius: 3px;
      margin-bottom: 2px;
    }
    
    /* Search bar styles */
    .search-container {
      margin-bottom: 15px;
    }
    
    #imageSearch {
      width: 100%;
      padding: 8px;
      border: 1px solid #ddd;
      border-radius: 4px;
      margin-bottom: 10px;
    }
    
    /* Improve gallery layout */
    .gallery {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
      gap: 10px;
      max-height: 500px;
      overflow-y: auto;
    }
    
    .gallery-item {
      height: 150px !important;
      position: relative;
      border: 1px solid #ddd;
      border-radius: 4px;
      overflow: hidden;
    }
    
    .gallery-item img {
      width: 100%;
      height: 100%;
      object-fit: cover;
      cursor: pointer;
      transition: transform 0.2s;
    }
    
    .gallery-item img:hover {
      transform: scale(1.05);
    }
    
    .img-container {
      height: 100%;
    }
    
    /* Tags display */
    .image-tags {
      position: absolute;
      bottom: 0;
      left: 0;
      right: 0;
      background: rgba(0,0,0,0.6);
      color: white;
      padding: 3px;
      font-size: 10px;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    /* Modal styles for event creation and image gallery (existing) */
    .modal {
      display: none;
      position: fixed;
      z-index: 10000;
      left: 0;
      top: 0;
      width: 100%;
      height: 100%;
      overflow: auto;
      background-color: rgba(0,0,0,0.7);
      padding: 20px;
    }
    .modal-content {
      background: #fff;
      margin: 10% auto;
      padding: 20px;
      border-radius: 8px;
      max-width: 500px;
      position: relative;
    }
    .close {
      position: absolute;
      top: 10px;
      right: 15px;
      font-size: 1.5em;
      cursor: pointer;
    }
    /* Deletion modal for recurring events */
    #deleteModal .modal-content {
      max-width: 400px;
      text-align: center;
    }
    #deleteModal button {
      margin: 5px;
    }
    
    /* Styling for recurring events */
    .recurring-event {
      border-left: 4px dashed #fff !important;  /* Dashed border to indicate recurring */
      border-right: 4px dashed #fff !important;
    }
    
    .recurring-event:before {
      content: "↻";  /* Recurrence symbol */
      position: absolute;
      top: 2px;
      left: 2px;
      font-weight: bold;
      color: white;
      background-color: rgba(0, 0, 0, 0.5);
      border-radius: 50%;
      width: 16px;
      height: 16px;
      line-height: 16px;
      text-align: center;
      z-index: 100;
    }
  </style>
{% endblock %}
{% block content %}
<div class="container">
  <header class="page-header">
    <h1>Schedule Images</h1>
    <p>Manage your scheduled image updates with our interactive calendar.</p>
  </header>
  <div id="calendar"></div>
  
  <div style="height: 200px;"></div>
</div>


<div id="eventModal" class="modal">
  <div class="modal-content">
    <span class="close" onclick="closeEventModal()">&times;</span>
    <h2 id="eventModalTitle">Add Scheduled Event</h2>
    <form id="eventForm">
      <input type="hidden" id="editingEventId" value="">
      <div>
        <label for="eventDate">Date &amp; Time:</label>
        <input type="datetime-local" id="eventDate" name="eventDate" required>
      </div>
      <div>
        <label>Select eInk Display:</label>
        {% if devices %}
          {% for device in devices %}
            <label>
              <input type="radio" name="device" value="{{ device.address }}" {% if loop.first %}checked{% endif %}>
              {{ device.friendly_name }}
            </label>
          {% endfor %}
        {% else %}
          <p>No devices configured. Please add devices in <a href="{{ url_for('settings.settings') }}">Settings</a>.</p>
        {% endif %}
      </div>
      <div>
        <label for="recurrence">Recurrence:</label>
        <select id="recurrence" name="recurrence">
          <option value="none">None</option>
          <option value="daily">Daily</option>
          <option value="weekly">Weekly</option>
          <option value="monthly">Same date next month</option>
        </select>
      </div>
      <div>
        <label>Choose Image:</label>
        <button type="button" onclick="openImageGallery()">Select Image</button>
        <input type="hidden" id="selectedImage" name="selectedImage">
        <span id="selectedImageName"></span>
      </div>
      <div style="margin-top:10px;">
        <input type="submit" id="eventSubmitButton" value="Save Event">
      </div>
    </form>
  </div>
</div>


<div id="imageGalleryModal" class="modal">
  <div class="modal-content" style="max-width:800px;">
    <span class="close" onclick="closeImageGallery()">&times;</span>
    <h2>Select an Image</h2>
    
    
    <div class="search-container">
      <input type="text" id="imageSearch" placeholder="Search by tags...">
    </div>
    
    <div class="gallery" id="galleryModal">
      {% for image in images %}
        <div class="gallery-item" data-tags="{{ image_tags.get(image, '') }}">
          <div class="img-container">
            <img src="{{ url_for('image.thumbnail', filename=image) }}" alt="{{ image }}" data-filename="{{ image }}" onclick="selectImage('{{ image }}', this.src)">
            {% if image_tags.get(image) %}
              <div class="image-tags">{{ image_tags.get(image) }}</div>
            {% endif %}
          </div>
        </div>
      {% endfor %}
    </div>
  </div>
</div>


<div id="deleteModal" class="modal">
  <div class="modal-content">
    <span class="close" onclick="closeDeleteModal()">&times;</span>
    <h3>Delete Recurring Event</h3>
    <p>Delete this occurrence or the entire series?</p>
    <button id="deleteOccurrenceBtn" class="btn btn-danger">Delete this occurrence</button>
    <button id="deleteSeriesBtn" class="btn btn-danger">Delete entire series</button>
    <button onclick="closeDeleteModal()" class="btn btn-secondary">Cancel</button>
  </div>
</div>
{% endblock %}
{% block scripts %}
  <script>
    var currentDeleteEventId = null; // store event id for deletion modal

    document.addEventListener('DOMContentLoaded', function() {
      var calendarEl = document.getElementById('calendar');
      if (!calendarEl) return;
      if (typeof FullCalendar === 'undefined') {
        console.error("FullCalendar is not defined");
        return;
      }
      var calendar = new FullCalendar.Calendar(calendarEl, {
        initialView: 'timeGridWeek',
        firstDay: 1,
        nowIndicator: true,
        editable: true,
        headerToolbar: {
          left: 'prev,next today refresh',
          center: 'title',
          right: 'timeGridWeek,timeGridDay'
        },
        events: '/schedule/events',
        customButtons: {
          refresh: {
            text: '↻',
            click: function() {
              // Force a full refresh of events from the server
              calendar.refetchEvents();
            }
          }
        },
        eventDrop: function(info) {
          var newDate = info.event.start;
          
          // Fix timezone issue - convert to local timezone string
          // This ensures we preserve the exact time shown in the UI
          var year = newDate.getFullYear();
          var month = String(newDate.getMonth() + 1).padStart(2, '0');
          var day = String(newDate.getDate()).padStart(2, '0');
          var hours = String(newDate.getHours()).padStart(2, '0');
          var minutes = String(newDate.getMinutes()).padStart(2, '0');
          
          var localDateStr = `${year}-${month}-${day}T${hours}:${minutes}`;
          console.log("Local date string:", localDateStr);
          
          fetch("/schedule/update", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              event_id: info.event.id,
              datetime: localDateStr,
              timezone_offset: newDate.getTimezoneOffset()
            })
          })
          .then(response => response.json())
          .then(data => {
            if(data.status !== "success"){
              alert("Error updating event: " + data.message);
              // Revert the drag if there was an error
              info.revert();
            } else {
              console.log("Event successfully updated:", info.event.id);
              // Ensure the event stays at the new position after refresh
              calendar.refetchEvents();
            }
          })
          .catch(err => {
            console.error("Error updating event:", err);
            // Revert the drag if there was an error
            info.revert();
          });
        },
        eventDidMount: function(info) {
          // Add special styling for recurring events
          if (info.event.extendedProps.isRecurring) {
            info.el.classList.add('recurring-event');
          }
          
          // Add delete button with improved visibility
          var deleteEl = document.createElement('span');
          deleteEl.innerHTML = '&times;';
          deleteEl.style.position = 'absolute';
          deleteEl.style.top = '2px';
          deleteEl.style.right = '2px';
          deleteEl.style.color = 'white';  // White text for better contrast
          deleteEl.style.backgroundColor = 'rgba(0, 0, 0, 0.6)';  // Semi-transparent black background
          deleteEl.style.borderRadius = '50%';  // Circular background
          deleteEl.style.width = '16px';
          deleteEl.style.height = '16px';
          deleteEl.style.lineHeight = '14px';
          deleteEl.style.textAlign = 'center';
          deleteEl.style.cursor = 'pointer';
          deleteEl.style.fontWeight = 'bold';
          deleteEl.style.zIndex = '100';
          deleteEl.style.border = '1px solid white';  // White border for additional contrast
          info.el.appendChild(deleteEl);
          
          // Add thumbnail to event
          if (info.event.extendedProps.thumbnail) {
            var thumbnailEl = document.createElement('img');
            thumbnailEl.src = info.event.extendedProps.thumbnail;
            thumbnailEl.className = 'event-thumbnail';
            thumbnailEl.alt = info.event.title;
            thumbnailEl.style.cursor = 'pointer';
            
            // Add click event to open the event for editing
            thumbnailEl.addEventListener('click', function(e) {
              e.stopPropagation();
              // Get event data
              var eventData = {
                id: info.event.id,
                title: info.event.title,
                start: info.event.start,
                device: info.event.extendedProps.device,
                filename: info.event.extendedProps.filename,
                recurrence: info.event.extendedProps.recurrence
              };
              
              // Populate the event form with this data - use local timezone
              var startDate = eventData.start;
              var year = startDate.getFullYear();
              var month = String(startDate.getMonth() + 1).padStart(2, '0');
              var day = String(startDate.getDate()).padStart(2, '0');
              var hours = String(startDate.getHours()).padStart(2, '0');
              var minutes = String(startDate.getMinutes()).padStart(2, '0');
              
              var localDateStr = `${year}-${month}-${day}T${hours}:${minutes}`;
              document.getElementById('eventDate').value = localDateStr;
              console.log("Event edit local time:", localDateStr);
              
              // Select the correct device radio button
              var deviceRadios = document.querySelectorAll('input[name="device"]');
              for (var i = 0; i < deviceRadios.length; i++) {
                if (deviceRadios[i].value === eventData.device) {
                  deviceRadios[i].checked = true;
                  break;
                }
              }
              
              // Set the recurrence dropdown
              document.getElementById('recurrence').value = eventData.recurrence || 'none';
              
              // Set the selected image
              document.getElementById('selectedImage').value = eventData.filename;
              document.getElementById('selectedImageName').textContent = eventData.filename;
              
              // Set the editing event ID
              document.getElementById('editingEventId').value = eventData.id;
              
              // Update modal title and button text
              document.getElementById('eventModalTitle').textContent = 'Edit Scheduled Event';
              document.getElementById('eventSubmitButton').value = 'Update Event';
              
              // Open the event modal
              openEventModal();
            });
            
            // Find the event title element and insert the thumbnail before it
            var titleEl = info.el.querySelector('.fc-event-title');
            if (titleEl && titleEl.parentNode) {
              titleEl.parentNode.insertBefore(thumbnailEl, titleEl);
              
              // Add device name to title
              if (info.event.extendedProps.deviceName) {
                titleEl.textContent = info.event.title + ' on ' + info.event.extendedProps.deviceName;
              }
            }
          }
          
          // Add click handler for delete button
          deleteEl.addEventListener('click', function(e) {
            e.stopPropagation();
            // If recurring, show custom deletion modal; otherwise, delete directly.
            if(info.event.extendedProps.recurrence && info.event.extendedProps.recurrence.toLowerCase() !== "none"){
              currentDeleteEventId = info.event.id;
              openDeleteModal();
            } else {
              // Directly delete non-recurring event without popup.
              fetch("/schedule/remove/" + info.event.id, { method: "POST" })
              .then(response => response.json())
              .then(data => {
                if(data.status === "success"){
                  info.event.remove();
                }
              })
              .catch(err => {
                console.error("Error deleting event:", err);
              });
            }
          });
        },
        dateClick: function(info) {
          // Create a date object from the clicked date
          var dtLocal = new Date(info.date);
          
          // Format the date in local timezone format (YYYY-MM-DDTHH:MM)
          var year = dtLocal.getFullYear();
          var month = String(dtLocal.getMonth() + 1).padStart(2, '0');
          var day = String(dtLocal.getDate()).padStart(2, '0');
          var hours = String(dtLocal.getHours()).padStart(2, '0');
          var minutes = String(dtLocal.getMinutes()).padStart(2, '0');
          
          var localDateStr = `${year}-${month}-${day}T${hours}:${minutes}`;
          console.log("Clicked date local time:", localDateStr);
          
          openNewEventModal(localDateStr);
        }
      });
      calendar.render();
      
      // Add event listener for page refresh
      window.addEventListener('beforeunload', function() {
        // Store a flag indicating that we're refreshing the page
        localStorage.setItem('calendarRefreshing', 'true');
      });
      
      // Check if we're coming back from a refresh
      if (localStorage.getItem('calendarRefreshing') === 'true') {
        // Clear the flag
        localStorage.removeItem('calendarRefreshing');
        // Force a refresh of the events
        setTimeout(function() {
          calendar.refetchEvents();
          console.log("Refreshed events after page reload");
        }, 500);
      }
    });

    function openEventModal() { document.getElementById('eventModal').style.display = 'block'; }
    function closeEventModal() { document.getElementById('eventModal').style.display = 'none'; }
    
    function openImageGallery() {
      document.getElementById('imageGalleryModal').style.display = 'block';
      // Clear search field when opening
      var searchField = document.getElementById('imageSearch');
      if (searchField) {
        searchField.value = '';
        searchField.focus();
        // Trigger search to show all images
        filterImages('');
      }
    }
    
    function closeImageGallery() { document.getElementById('imageGalleryModal').style.display = 'none'; }
    
    function selectImage(filename, src) {
      document.getElementById('selectedImage').value = filename;
      document.getElementById('selectedImageName').textContent = filename;
      // Also show the thumbnail
      var nameSpan = document.getElementById('selectedImageName');
      if (nameSpan) {
        nameSpan.innerHTML = `<img src="${src}" style="height:40px;margin-right:5px;vertical-align:middle;"> ${filename}`;
      }
      closeImageGallery();
    }
    
    function openDeleteModal() { document.getElementById('deleteModal').style.display = 'block'; }
    function closeDeleteModal() { document.getElementById('deleteModal').style.display = 'none'; }
    
    // Function to filter images based on search input
    function filterImages(searchText) {
      searchText = searchText.toLowerCase();
      var items = document.querySelectorAll('#galleryModal .gallery-item');
      
      items.forEach(function(item) {
        var tags = item.getAttribute('data-tags') || '';
        var filename = item.querySelector('img').getAttribute('data-filename') || '';
        
        if (tags.toLowerCase().includes(searchText) || filename.toLowerCase().includes(searchText) || searchText === '') {
          item.style.display = '';
        } else {
          item.style.display = 'none';
        }
      });
    }
    
    // Add event listener for search input
    document.addEventListener('DOMContentLoaded', function() {
      var searchInput = document.getElementById('imageSearch');
      if (searchInput) {
        searchInput.addEventListener('input', function() {
          filterImages(this.value);
        });
      }
    });

    document.getElementById('deleteOccurrenceBtn').addEventListener('click', function() {
      // Skip this occurrence for recurring event.
      fetch("/schedule/skip/" + currentDeleteEventId, { method: "POST" })
      .then(response => response.json())
      .then(data => {
        if(data.status === "success"){
          // Remove the occurrence from the calendar.
          location.reload();
        }
      })
      .catch(err => {
        console.error("Error skipping occurrence:", err);
      });
      closeDeleteModal();
    });

    document.getElementById('deleteSeriesBtn').addEventListener('click', function() {
      // Delete the entire series.
      fetch("/schedule/remove/" + currentDeleteEventId, { method: "POST" })
      .then(response => response.json())
      .then(data => {
        if(data.status === "success"){
          location.reload();
        }
      })
      .catch(err => {
        console.error("Error deleting series:", err);
      });
      closeDeleteModal();
    });

    // Reset the event form when opening for a new event
    function resetEventForm() {
      document.getElementById('eventForm').reset();
      document.getElementById('selectedImage').value = '';
      document.getElementById('selectedImageName').textContent = '';
      document.getElementById('editingEventId').value = '';
      document.getElementById('eventModalTitle').textContent = 'Add Scheduled Event';
      document.getElementById('eventSubmitButton').value = 'Save Event';
    }
    
    // When clicking on a date, reset the form for a new event
    function openNewEventModal(date) {
      resetEventForm();
      if (date) {
        document.getElementById('eventDate').value = date;
      }
      openEventModal();
    }
    
    // Update the dateClick handler to use the new function
    document.addEventListener('DOMContentLoaded', function() {
      // Existing code will still run, this just adds additional functionality
      var calendarEl = document.getElementById('calendar');
      if (calendarEl && typeof FullCalendar !== 'undefined') {
        var existingCalendar = calendarEl._fullCalendar;
        if (existingCalendar) {
          existingCalendar.setOption('dateClick', function(info) {
            var dtLocal = new Date(info.date);
            var isoStr = dtLocal.toISOString().substring(0,16);
            openNewEventModal(isoStr);
          });
        }
      }
    });
    
    // Handle form submission for both adding and updating events
    document.getElementById('eventForm').addEventListener('submit', function(e) {
      e.preventDefault();
      var datetime = document.getElementById('eventDate').value;
      var device = document.querySelector('input[name="device"]:checked').value;
      var recurrence = document.getElementById('recurrence').value;
      var filename = document.getElementById('selectedImage').value;
      var eventId = document.getElementById('editingEventId').value;
      
      if (!datetime || !device || !filename) {
        alert("Please fill in all fields and select an image.");
        return;
      }
      
      // Determine if we're adding a new event or updating an existing one
      var isUpdate = eventId !== '';
      var url = isUpdate ? "/schedule/update" : "/schedule/add";
      var requestData = {
        datetime: datetime,
        device: device,
        recurrence: recurrence,
        filename: filename,
        timezone_offset: new Date().getTimezoneOffset()  // Add timezone offset
      };
      
      // If updating, include the event ID
      if (isUpdate) {
        requestData.event_id = eventId;
      }
      
      fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(requestData)
      })
      .then(response => response.json())
      .then(data => {
        if(data.status === "success"){
          location.reload();
        } else {
          alert("Error: " + data.message);
        }
      })
      .catch(err => {
        console.error("Error " + (isUpdate ? "updating" : "adding") + " event:", err);
      });
    });
  </script>
{% endblock %}
```


## templates/settings.html

```html
{% extends "base.html" %}
{% block title %}Settings - InkyDocker{% endblock %}

{% block content %}
<div class="container">
  
  <header class="page-header">
    <h1>Settings</h1>
    <p>Manage your eInk displays and AI settings.</p>
  </header>

  
  <div class="card text-center">
    <button id="clipSettingsBtn" class="primary-btn">CLIP Model Settings</button>
  </div>

  
  <div id="clipSettingsModal" class="modal">
    <div class="modal-content">
      <span class="close" id="closeClipSettingsModal">&times;</span>
      <h2>CLIP Model Settings</h2>
      <form id="clipSettingsForm">
        
        <div class="form-group">
          <label for="clip_model">Select CLIP Model for Image Tagging:</label>
          <select id="clip_model" name="clip_model" class="form-select" data-current="{{ config.clip_model if config and config.clip_model }}">
            <option value="">-- Select a model --</option>
            <option value="ViT-B-32" {% if config and config.clip_model == 'ViT-B-32' %}selected{% endif %}>ViT-B-32 (Faster, less accurate)</option>
            <option value="ViT-B-16" {% if config and config.clip_model == 'ViT-B-16' %}selected{% endif %}>ViT-B-16 (Balanced)</option>
            <option value="ViT-L-14" {% if config and config.clip_model == 'ViT-L-14' %}selected{% endif %}>ViT-L-14 (Slower, more accurate)</option>
          </select>
          <button type="button" class="field-save-btn" onclick="saveClipModel()">Save Model</button>
        </div>
        
        
        <div id="modelDownloadContainer" style="margin-top: 15px; display: none;">
          <p>Downloading model: <span id="modelDownloadName"></span></p>
          <div class="progress-container" style="width: 100%; background: #ddd; border-radius: 5px;">
            <div id="modelDownloadProgress" class="progress-bar" style="width: 0%; height: 20px; background: #28a745; border-radius: 5px; color: #fff; text-align: center; line-height: 20px;">0%</div>
          </div>
        </div>
        
        <div style="margin-top: 20px;">
          <p>All models are pre-installed in the system.</p>
          <p>Larger models provide more accurate tagging but require more processing power and memory.</p>
        </div>
        
        
        <div style="margin-top: 20px; text-align: center;">
          <button type="button" class="primary-btn" onclick="rerunAllTagging()">Rerun Tagging on All Images</button>
        </div>
      </form>
    </div>
  </div>

  
  <div class="card text-center">
    <button id="addNewDisplayBtn" class="primary-btn">Add New Display</button>
  </div>

  
  <div id="addDisplayModal" class="modal">
    <div class="modal-content">
      <span class="close" id="closeAddDisplayModal">&times;</span>
      <h2>Add New Display</h2>
      <form id="addDisplayForm" method="POST" action="{{ url_for('settings.settings') }}">
        <input type="text" name="address" id="newAddress" placeholder="Device Address (e.g., 192.168.1.100)" required>
        <select name="orientation" id="newOrientation" required>
          <option value="portrait">Portrait</option>
          <option value="landscape">Landscape</option>
        </select>
        <input type="text" name="friendly_name" id="newFriendlyName" placeholder="Friendly Name" required>
        
        <input type="hidden" name="display_name" id="newDisplayName">
        <input type="hidden" name="resolution" id="newResolution">
        <input type="hidden" name="color" id="newColor">
        <div style="margin-top: 10px;">
          <button type="button" class="primary-btn" onclick="fetchDisplayInfo('new')">Fetch Display Info</button>
        </div>
        <div style="margin-top: 10px;">
          <button type="submit" class="primary-btn">Save</button>
          <button type="button" class="primary-btn" onclick="closeAddDisplayModal()">Cancel</button>
        </div>
      </form>
    </div>
  </div>

  
  <div id="editDisplayModal" class="modal">
    <div class="modal-content">
      <span class="close" id="closeEditDisplayModal">&times;</span>
      <h2>Edit Display</h2>
      <form id="editDisplayForm" method="POST" action="{{ url_for('settings.edit_device') }}">
        <input type="hidden" name="device_index" id="editDeviceIndex">
        <label for="editFriendlyName">Friendly Name:</label>
        <input type="text" name="friendly_name" id="editFriendlyName" placeholder="Friendly Name" required>
        <label for="editOrientation">Orientation:</label>
        <select name="orientation" id="editOrientation" required>
          <option value="portrait">Portrait</option>
          <option value="landscape">Landscape</option>
        </select>
        <label for="editAddress">Device Address:</label>
        <input type="text" name="address" id="editAddress" placeholder="Device Address" required>
        <div style="margin-top: 10px;">
          <button type="submit" class="primary-btn">Save Changes</button>
          <button type="button" class="primary-btn" onclick="closeEditDisplayModal()">Cancel</button>
        </div>
      </form>
    </div>
  </div>

  
  <div id="advancedActionsModal" class="modal">
    <div class="modal-content">
      <span class="close" id="closeAdvancedActionsModal">&times;</span>
      <h2>Advanced Actions</h2>
      <p id="advancedDeviceTitle" style="font-weight:bold;"></p>
      <div style="margin-top: 10px;">
        <button type="button" class="primary-btn" onclick="triggerSystemUpdate()">System Update & Reboot</button>
        <button type="button" class="primary-btn" onclick="triggerBackup()">Create Backup</button>
        <button type="button" class="primary-btn" onclick="triggerAppUpdate()">Update Application</button>
      </div>
      <div style="margin-top: 10px;">
        <button type="button" class="primary-btn" onclick="closeAdvancedActionsModal()">Close</button>
      </div>
    </div>
  </div>

  
  <div class="card">
    <h2>Existing Devices</h2>
    <table class="device-table">
      <thead>
        <tr>
          <th>#</th>
          <th>Color</th>
          <th>Friendly Name</th>
          <th>Orientation</th>
          <th>Address</th>
          <th>Display Name</th>
          <th>Resolution</th>
          <th>Status</th>
          <th>Action</th>
        </tr>
      </thead>
      <tbody>
        {% for device in devices %}
        <tr data-index="{{ loop.index0 }}" data-address="{{ device.address }}">
          <td>{{ loop.index }}</td>
          <td>
            <div style="width:20px; height:20px; border-radius:50%; background:{{ device.color }};"></div>
          </td>
          <td>{{ device.friendly_name }}</td>
          <td>{{ device.orientation }}</td>
          <td>{{ device.address }}</td>
          <td>{{ device.display_name }}</td>
          <td>{{ device.resolution }}</td>
          <td>
            {% if device.online %}
              <span style="color:green;">&#9679;</span>
            {% else %}
              <span style="color:red;">&#9679;</span>
            {% endif %}
          </td>
          <td>
            <form method="POST" action="{{ url_for('settings.delete_device', device_index=loop.index0) }}" style="display:inline;">
              <input type="submit" value="Delete">
            </form>
            <button type="button" class="edit-button" onclick="openEditModal('{{ loop.index0 }}', '{{ device.friendly_name }}', '{{ device.orientation }}', '{{ device.address }}')">
              Edit
            </button>
            <button type="button" class="advanced-button" onclick="openAdvancedModal('{{ loop.index0 }}', '{{ device.friendly_name }}')">
              Advanced
            </button>
          </td>
        </tr>
        {% endfor %}
      </tbody>
    </table>
  </div>
</div>
{% endblock %}

{% block scripts %}
<style>
  .modal {
    display: none;
    position: fixed;
    z-index: 3000;
    left: 0;
    top: 0;
    width: 100%;
    height: 100%;
    overflow: auto;
    background-color: rgba(0, 0, 0, 0.5);
  }
  .modal-content {
    background-color: #fff;
    margin: 10% auto;
    padding: 20px;
    border-radius: 5px;
    max-width: 500px;
    position: relative;
  }
  .close {
    color: #aaa;
    float: right;
    font-size: 28px;
    font-weight: bold;
    cursor: pointer;
  }
  .close:hover,
  .close:focus {
    color: #000;
  }
  .primary-btn {
    background: linear-gradient(to right, #28a745, #218838);
    border: none;
    color: #fff;
    padding: 10px 20px;
    border-radius: 5px;
    cursor: pointer;
    font-size: 1em;
  }
  .primary-btn:hover {
    background: linear-gradient(to right, #218838, #1e7e34);
  }
  .field-save-btn {
    margin-top: 5px;
    font-size: 0.9em;
    padding: 5px 10px;
    background-color: #007bff;
    color: #fff;
    border: none;
    border-radius: 3px;
    cursor: pointer;
  }
  .field-save-btn:hover {
    background-color: #0056b3;
  }
  .overlay-popup {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background-color: rgba(0, 0, 0, 0.7);
    display: flex;
    justify-content: center;
    align-items: center;
    z-index: 5000;
  }
  .overlay-content {
    background-color: #fff;
    padding: 20px;
    border-radius: 5px;
    max-width: 400px;
    text-align: center;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
  }
  .overlay-buttons {
    margin-top: 15px;
    display: flex;
    justify-content: center;
    gap: 10px;
  }
  .cancel-btn {
    background: #6c757d;
    border: none;
    color: #fff;
    padding: 8px 15px;
    border-radius: 5px;
    cursor: pointer;
  }
  .cancel-btn:hover {
    background: #5a6268;
  }
  .progress-container {
    margin: 10px 0;
    background-color: #f1f1f1;
    border-radius: 5px;
    overflow: hidden;
  }
  .progress-bar {
    height: 20px;
    background-color: #4CAF50;
    text-align: center;
    line-height: 20px;
    color: white;
    transition: width 0.3s ease;
  }
</style>

<script>
  // Global modal closing functions
  window.closeAddDisplayModal = function() {
    document.getElementById('addDisplayModal').style.display = 'none';
  };
  window.closeEditDisplayModal = function() {
    document.getElementById('editDisplayModal').style.display = 'none';
  };
  window.closeAdvancedActionsModal = function() {
    document.getElementById('advancedActionsModal').style.display = 'none';
  };
  
  // Overlay message functions
  function showOverlayMessage(message, duration) {
    var overlay = document.createElement('div');
    overlay.className = 'overlay-popup';
    overlay.innerHTML = `
      <div class="overlay-content">
        <p>${message}</p>
        <button class="primary-btn" onclick="this.parentNode.parentNode.remove()">OK</button>
      </div>
    `;
    document.body.appendChild(overlay);
    
    if (duration) {
      setTimeout(function() {
        if (overlay.parentNode) {
          overlay.parentNode.removeChild(overlay);
        }
      }, duration);
    }
  }
  
  function showConfirmOverlay(message, confirmCallback) {
    var overlay = document.createElement('div');
    overlay.className = 'overlay-popup';
    overlay.innerHTML = `
      <div class="overlay-content">
        <p>${message}</p>
        <div class="overlay-buttons">
          <button class="primary-btn" id="confirmYes">Yes</button>
          <button class="cancel-btn" id="confirmNo">No</button>
        </div>
      </div>
    `;
    document.body.appendChild(overlay);
    
    document.getElementById('confirmYes').addEventListener('click', function() {
      overlay.parentNode.removeChild(overlay);
      if (typeof confirmCallback === 'function') {
        confirmCallback();
      }
    });
    
    document.getElementById('confirmNo').addEventListener('click', function() {
      overlay.parentNode.removeChild(overlay);
    });
  }

  // Device status checking
  
  function checkDeviceStatus() {
    fetch("/devices/status")
      .then(response => response.json())
      .then(data => {
        if(data.status === "success") {
          data.devices.forEach(function(device) {
            var row = document.querySelector('tr[data-index="' + device.index + '"]');
            if (row) {
              var statusCell = row.querySelector('td:nth-child(8)');
              if(device.online) {
                statusCell.innerHTML = '<span style="color:green;">&#9679;</span>';
              } else {
                statusCell.innerHTML = '<span style="color:red;">&#9679;</span>';
              }
            }
          });
        }
      })
      .catch(error => {
        console.error("Error checking device status:", error);
      });
  }
  setInterval(checkDeviceStatus, 5000);
  checkDeviceStatus();

  // Modal functions for editing and advanced actions
  function openEditModal(index, friendlyName, orientation, address) {
    document.getElementById('editDisplayModal').style.display = 'block';
    document.getElementById('editDeviceIndex').value = index;
    document.getElementById('editFriendlyName').value = friendlyName;
    document.getElementById('editOrientation').value = orientation;
    document.getElementById('editAddress').value = address;
  }

  function openAdvancedModal(index, friendlyName) {
    document.getElementById('advancedActionsModal').style.display = 'block';
    document.getElementById('advancedDeviceTitle').textContent = "Advanced Actions for " + friendlyName;
    document.getElementById('advancedActionsModal').setAttribute('data-device-index', index);
  }
  
  // Device API functions: triggerSystemUpdate, triggerBackup, triggerAppUpdate remain unchanged
  function triggerSystemUpdate() {
    var deviceIndex = document.getElementById('advancedActionsModal').getAttribute('data-device-index');
    var deviceRow = document.querySelector('tr[data-index="' + deviceIndex + '"]');
    var deviceAddress = deviceRow.getAttribute('data-address');
    if (!deviceAddress) {
      showOverlayMessage("Error: Device address not found");
      return;
    }
    showConfirmOverlay(
      "This will trigger a system update and reboot the device. Continue?",
      function() {
        var progressOverlay = document.createElement('div');
        progressOverlay.className = 'overlay-popup';
        progressOverlay.innerHTML = `
          <div class="overlay-content">
            <p>Triggering system update...</p>
            <div class="progress-container">
              <div id="updateProgressBar" class="progress-bar" style="width: 0%">0%</div>
            </div>
          </div>
        `;
        document.body.appendChild(progressOverlay);
        if (!deviceAddress.startsWith("http://") && !deviceAddress.startsWith("https://")) {
          deviceAddress = "http://" + deviceAddress;
        }
        fetch(deviceAddress + "/system_update", { method: 'POST' })
          .then(response => {
            if (!response.ok) {
              throw new Error("Network response was not ok");
            }
            return response.json();
          })
          .then(data => {
            var progress = 0;
            var interval = setInterval(function() {
              progress += 5;
              if (progress >= 100) {
                progress = 100;
                clearInterval(interval);
                document.body.removeChild(progressOverlay);
                showOverlayMessage("System update triggered successfully. Device will reboot.");
                closeAdvancedActionsModal();
              }
              var progressBar = document.getElementById('updateProgressBar');
              if (progressBar) {
                progressBar.style.width = progress + '%';
                progressBar.textContent = progress + '%';
              }
            }, 500);
          })
          .catch(error => {
            document.body.removeChild(progressOverlay);
            showOverlayMessage("Error triggering system update: " + error.message);
          });
      }
    );
  }
  
  function triggerBackup() {
    var deviceIndex = document.getElementById('advancedActionsModal').getAttribute('data-device-index');
    var deviceRow = document.querySelector('tr[data-index="' + deviceIndex + '"]');
    var deviceAddress = deviceRow.getAttribute('data-address');
    if (!deviceAddress) {
      showOverlayMessage("Error: Device address not found");
      return;
    }
    showConfirmOverlay(
      "This will create a backup of the device. This may take several minutes. Continue?",
      function() {
        var progressOverlay = document.createElement('div');
        progressOverlay.className = 'overlay-popup';
        progressOverlay.innerHTML = `
          <div class="overlay-content">
            <p>Creating backup...</p>
            <div class="progress-container">
              <div id="backupProgressBar" class="progress-bar" style="width: 0%">0%</div>
            </div>
          </div>
        `;
        document.body.appendChild(progressOverlay);
        if (!deviceAddress.startsWith("http://") && !deviceAddress.startsWith("https://")) {
          deviceAddress = "http://" + deviceAddress;
        }
        var progress = 0;
        var interval = setInterval(function() {
          progress += 2;
          if (progress >= 100) {
            progress = 100;
            clearInterval(interval);
            var a = document.createElement('a');
            a.href = deviceAddress + "/backup";
            a.download = "backup_" + new Date().toISOString().replace(/:/g, '-') + ".img.gz";
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            document.body.removeChild(progressOverlay);
            showOverlayMessage("Backup created successfully. Download started.");
            closeAdvancedActionsModal();
          }
          var progressBar = document.getElementById('backupProgressBar');
          if (progressBar) {
            progressBar.style.width = progress + '%';
            progressBar.textContent = progress + '%';
          }
        }, 500);
      }
    );
  }
  
  function triggerAppUpdate() {
    var deviceIndex = document.getElementById('advancedActionsModal').getAttribute('data-device-index');
    var deviceRow = document.querySelector('tr[data-index="' + deviceIndex + '"]');
    var deviceAddress = deviceRow.getAttribute('data-address');
    if (!deviceAddress) {
      showOverlayMessage("Error: Device address not found");
      return;
    }
    showConfirmOverlay(
      "This will update the application on the device and reboot it. Continue?",
      function() {
        var progressOverlay = document.createElement('div');
        progressOverlay.className = 'overlay-popup';
        progressOverlay.innerHTML = `
          <div class="overlay-content">
            <p>Updating application...</p>
            <div class="progress-container">
              <div id="appUpdateProgressBar" class="progress-bar" style="width: 0%">0%</div>
            </div>
          </div>
        `;
        document.body.appendChild(progressOverlay);
        if (!deviceAddress.startsWith("http://") && !deviceAddress.startsWith("https://")) {
          deviceAddress = "http://" + deviceAddress;
        }
        fetch(deviceAddress + "/update", { method: 'POST' })
          .then(response => {
            if (!response.ok) {
              throw new Error("Network response was not ok");
            }
            return response.json();
          })
          .then(data => {
            var progress = 0;
            var interval = setInterval(function() {
              progress += 10;
              if (progress >= 100) {
                progress = 100;
                clearInterval(interval);
                document.body.removeChild(progressOverlay);
                showOverlayMessage("Application updated successfully. Device will reboot.");
                closeAdvancedActionsModal();
              }
              var progressBar = document.getElementById('appUpdateProgressBar');
              if (progressBar) {
                progressBar.style.width = progress + '%';
                progressBar.textContent = progress + '%';
              }
            }, 300);
          })
          .catch(function(error) {
            document.body.removeChild(progressOverlay);
            showOverlayMessage("Error updating application: " + error.message);
          });
      }
    );
  }

  document.addEventListener('DOMContentLoaded', function() {
    var addNewDisplayBtn = document.getElementById('addNewDisplayBtn');
    if (addNewDisplayBtn) {
      addNewDisplayBtn.addEventListener('click', function() {
        document.getElementById('addDisplayModal').style.display = 'block';
      });
    }
    var addDisplayForm = document.getElementById('addDisplayForm');
    addDisplayForm.addEventListener('submit', function(e) {
      e.preventDefault();
      fetchDisplayInfo('new').then(function() {
        addDisplayForm.submit();
      }).catch(function() {
        addDisplayForm.submit();
      });
    });
    document.getElementById('closeAddDisplayModal').addEventListener('click', function() {
      closeAddDisplayModal();
    });
    document.getElementById('closeEditDisplayModal').addEventListener('click', function() {
      closeEditDisplayModal();
    });
    document.getElementById('closeAdvancedActionsModal').addEventListener('click', function() {
      closeAdvancedActionsModal();
    });
    var clipSettingsBtn = document.getElementById('clipSettingsBtn');
    var clipSettingsModal = document.getElementById('clipSettingsModal');
    var closeClipSettingsModal = document.getElementById('closeClipSettingsModal');
    if (clipSettingsBtn) {
      clipSettingsBtn.addEventListener('click', function() {
        clipSettingsModal.style.display = 'block';
      });
    }
    if (closeClipSettingsModal) {
      closeClipSettingsModal.addEventListener('click', function() {
        clipSettingsModal.style.display = 'none';
      });
    }
    window.addEventListener('click', function(e) {
      if (e.target == clipSettingsModal) {
        clipSettingsModal.style.display = 'none';
      }
    });
  });

  function saveClipModel() {
    var clipModel = document.getElementById('clip_model').value;
    if (!clipModel) {
      showOverlayMessage("Please select a CLIP model");
      return;
    }
    var payload = {
      clip_model: clipModel
    };
    showOverlayMessage("Switching to model: " + clipModel + "...", 1500);
    fetch("{{ url_for('settings.update_clip_model') }}", {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    })
    .then(function(response) { return response.json(); })
    .then(function(data) {
      if (data.status === "success") {
        showOverlayMessage("CLIP model updated successfully to " + clipModel);
      } else {
        showOverlayMessage("Error updating CLIP model: " + data.message);
      }
    })
    .catch(function(error) {
      console.error("Error:", error);
      showOverlayMessage("An error occurred while updating CLIP model.");
    });
  }

  function rerunAllTagging() {
    showConfirmOverlay(
      "This will rerun tagging on all images using the selected CLIP model. This may take some time depending on the number of images. Continue?",
      function() {
        fetch("{{ url_for('settings.rerun_all_tagging') }}", {
          method: 'POST'
        })
        .then(function(response) { return response.json(); })
        .then(function(data) {
          if (data.status === "success") {
            showOverlayMessage("Tagging process started! This will run in the background.");
            document.getElementById('clipSettingsModal').style.display = 'none';
          } else {
            showOverlayMessage("Error starting tagging process: " + data.message);
          }
        })
        .catch(function(error) {
          console.error("Error:", error);
          showOverlayMessage("An error occurred while starting the tagging process.");
        });
      }
    );
  }

  function fetchDisplayInfo(mode) {
    return new Promise(function(resolve, reject) {
      if (mode === 'new') {
        var addressInput = document.getElementById('newAddress');
        var address = addressInput.value.trim();
        if (!address) {
          alert("Please enter the device address.");
          reject("No address provided");
          return;
        }
        fetch("/device_info?address=" + encodeURIComponent(address), { timeout: 5000 })
          .then(function(response) {
            if (!response.ok) {
              throw new Error("HTTP error " + response.status);
            }
            return response.json();
          })
          .then(function(data) {
            if (data.status === "ok") {
              document.getElementById('newDisplayName').value = data.info.display_name;
              document.getElementById('newResolution').value = data.info.resolution;
              var availableColors = ['#FF5733', '#33FF57', '#3357FF', '#F39C12', '#8E44AD', '#2ECC71', '#E74C3C'];
              var randomColor = availableColors[Math.floor(Math.random() * availableColors.length)];
              document.getElementById('newColor').value = randomColor;
              resolve();
            } else {
              alert("Error fetching display info: " + data.message + "; using default values.");
              document.getElementById('newDisplayName').value = "DefaultDisplay color";
              document.getElementById('newResolution').value = "800x600";
              document.getElementById('newColor').value = "#FF5733";
              resolve();
            }
          })
          .catch(function(error) {
            console.error("Error fetching display info:", error);
            alert("Error fetching display info; using default values.");
            document.getElementById('newDisplayName').value = "DefaultDisplay color";
            document.getElementById('newResolution').value = "800x600";
            document.getElementById('newColor').value = "#FF5733";
            resolve();
          });
      } else if (mode === 'edit') {
        alert("Fetch Display Info for edit is not implemented yet.");
        resolve();
      }
    });
  }
</script>
{% endblock %}
```


## utils/__init__.py

```py

```


## utils/crop_helpers.py

```py
from models import CropInfo, SendLog, db

def load_crop_info_from_db(filename):
    c = CropInfo.query.filter_by(filename=filename).first()
    if not c:
        return None
    return {"x": c.x, "y": c.y, "width": c.width, "height": c.height}

def save_crop_info_to_db(filename, crop_data):
    c = CropInfo.query.filter_by(filename=filename).first()
    if not c:
        c = CropInfo(filename=filename)
        db.session.add(c)
    c.x = crop_data.get("x", 0)
    c.y = crop_data.get("y", 0)
    c.width = crop_data.get("width", 0)
    c.height = crop_data.get("height", 0)
    db.session.commit()

def add_send_log_entry(filename):
    entry = SendLog(filename=filename)
    db.session.add(entry)
    db.session.commit()

def get_last_sent():
    latest = SendLog.query.order_by(SendLog.id.desc()).first()
    return latest.filename if latest else None
```


## utils/image_helpers.py

```py
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
```


## .DS_Store

```
   Bud1           	                                                           i cbwspblob                                                                                                                                                                                                                                                                                                                                                                                                                                           s t a t i cbwspblob   �bplist00�]ShowStatusBar[ShowToolbar[ShowTabView_ContainerShowSidebar\WindowBounds[ShowSidebar		_{{188, 391}, {920, 436}}	#/;R_klmno�                            �    s t a t i cfdscbool     s t a t i cvSrnlong                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            @      �                                        @      �                                          @      �                                          @                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   E  	                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       DSDB                                 `          �                                         @      �                                          @      �                                          @                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              
```


## .export-ignore

```
node_modules/
*.log
dist/
.vscode/
```


## .gitattributes

```
# Auto detect text files and perform LF normalization
* text=auto
```


## app.py

```py
from flask import Flask
import os
from config import Config
from models import db
import pillow_heif
from tasks import celery, start_scheduler

# Register HEIF opener for Pillow
pillow_heif.register_heif_opener()

def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)

    # Ensure required folders exist
    for folder in [app.config['IMAGE_FOLDER'], app.config['THUMBNAIL_FOLDER'], app.config['DATA_FOLDER']]:
        if not os.path.exists(folder):
            os.makedirs(folder)

    # Initialize database without migrations
    db.init_app(app)

    # Register blueprints
    from routes.image_routes import image_bp
    from routes.device_routes import device_bp
    from routes.schedule_routes import schedule_bp
    from routes.settings_routes import settings_bp
    from routes.device_info_routes import device_info_bp
    from routes.ai_tagging_routes import ai_bp

    app.register_blueprint(image_bp)
    app.register_blueprint(device_bp)
    app.register_blueprint(schedule_bp)
    app.register_blueprint(settings_bp)
    app.register_blueprint(device_info_bp)
    app.register_blueprint(ai_bp)

    # Create database tables if they don't exist.
    with app.app_context():
        db.create_all()

    # Configure Celery
    celery.conf.update(
        broker_url='redis://localhost:6379/0',
        result_backend='redis://localhost:6379/0'
    )

    # Note: Scheduler is now started in a dedicated process (scheduler.py)
    # We still call the function for backward compatibility, but it doesn't start the scheduler
    start_scheduler(app)
    
    # We no longer run fetch_device_metrics immediately here
    # The dedicated scheduler process will handle this

    return app

app = create_app()

# Make the app available to Celery tasks
celery.conf.update(app=app)

if __name__ == '__main__':
    # When running via 'python app.py' this block will execute.
    app.run(host='0.0.0.0', port=5001, debug=True)
```


## config.py

```py
import os

basedir = os.path.abspath(os.path.dirname(__file__))

class Config:
    SECRET_KEY = "super-secret-key"
    # Database: using an absolute path in a “data” folder in the project directory.
    # In the container, basedir will be /app so the DB will be at /app/data/mydb.sqlite.
    SQLALCHEMY_DATABASE_URI = 'sqlite:///' + os.path.join(basedir, 'data', 'mydb.sqlite')
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # Folders for images, thumbnails, and data storage
    IMAGE_FOLDER = os.path.join(basedir, 'images')
    THUMBNAIL_FOLDER = os.path.join(basedir, 'images', 'thumbnails')
    DATA_FOLDER = os.path.join(basedir, 'data')
```


## Dockerfile

```
# Use an official Python image
FROM python:3.13.2-slim

# Set timezone and cache directory for models (persisted in /data/model_cache)
ENV TZ=Europe/Copenhagen
ENV XDG_CACHE_HOME=/app/data/model_cache

# Install system dependencies and redis-server
RUN apt-get update && apt-get install -y \
    curl \
    sqlite3 \
    supervisor \
    tzdata \
    build-essential \
    gcc \
    git \
    libffi-dev \
    libssl-dev \
    libjpeg-dev \
    zlib1g-dev \
    libpng-dev \
    libfreetype6-dev \
    redis-server \
    && ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone \
    && rm -rf /var/lib/apt/lists/*

# Create necessary directories for the database and model cache
RUN mkdir -p /data /app/data/model_cache

# Set working directory
WORKDIR /app

# Copy only the requirements file first for caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install -r requirements.txt

# Pre-download all CLIP models (this layer will be cached if requirements.txt hasn't changed)
RUN python -c "import open_clip; \
    open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai', jit=False, force_quick_gelu=True); \
    open_clip.create_model_and_transforms('ViT-B-16', pretrained='openai', jit=False, force_quick_gelu=True); \
    open_clip.create_model_and_transforms('ViT-L-14', pretrained='openai', jit=False, force_quick_gelu=True)"

# Copy the rest of the project files
COPY . .

# Make scheduler.py executable
RUN chmod +x /app/scheduler.py

# Set environment variables for Celery
ENV CELERY_WORKER_MAX_MEMORY_PER_CHILD=500000
ENV CELERY_WORKERS=2

# Expose port 5001
EXPOSE 5001

# Copy entrypoint script and make it executable
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Copy Supervisor configuration file
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Tell Flask which file is our app
ENV FLASK_APP=app.py

# Run entrypoint.sh (which handles migrations and launches Supervisor)
CMD ["/entrypoint.sh"]
```


## entrypoint.sh

```sh
#!/bin/sh
# entrypoint.sh - Auto-create the database tables then launch the app via Supervisor.

echo "Ensuring /app/data folder exists..."
mkdir -p /app/data

# Start Redis first and ensure it's running
echo "Starting Redis server..."
redis-server --daemonize yes
sleep 2
echo "Checking Redis connection..."
redis-cli ping
if [ $? -ne 0 ]; then
  echo "Redis is not responding. Waiting a bit longer..."
  sleep 5
  redis-cli ping
  if [ $? -ne 0 ]; then
    echo "Redis still not responding. Please check Redis configuration."
  else
    echo "Redis is now running."
  fi
else
  echo "Redis is running."
fi

echo "Creating database tables..."
python -c "from app import app; from models import db; app.app_context().push(); db.create_all()"
echo "Database tables created successfully."

echo "Starting Supervisor..."
exec /usr/bin/supervisord -c /etc/supervisor/conf.d/supervisord.conf
```


## export.md

```md
# Project Structure

```
assets/
  js/
    main.js
data/
images/
migrations/
  versions/
  __init__.py
  alembic.ini
  env.py
  script.py.mako
routes/
  __init__.py
  additional_routes.py
  ai_tagging_routes.py
  device_info_routes.py
  device_routes.py
  image_routes.py
  schedule_routes.py
  settings_routes.py
static/
  icons/
  send-icon-old.png
  send-icon.png
  settings-wheel.png
  style.css
  trash-icon.png
templates/
  base.html
  index.html
  schedule.html
  settings.html
utils/
  __init__.py
  crop_helpers.py
  image_helpers.py
.DS_Store
.export-ignore
.gitattributes
app.py
config.py
Dockerfile
entrypoint.sh
exportconfig.json
LICENSE
models.py
package.json
README.md
requirements.txt
supervisord.conf
tasks.py
webpack.config.js
```


## assets/js/main.js

```js
import { Calendar } from '@fullcalendar/core';
import timeGridPlugin from '@fullcalendar/timegrid';
import '@fullcalendar/core/main.css';
import '@fullcalendar/timegrid/main.css';

document.addEventListener('DOMContentLoaded', function() {
  var calendarEl = document.getElementById('calendar');
  var calendar = new Calendar(calendarEl, {
    plugins: [ timeGridPlugin ],
    initialView: 'timeGridWeek',
    firstDay: 1, 
    nowIndicator: true,
    headerToolbar: {
      left: 'prev,next today',
      center: 'title',
      right: 'timeGridWeek,timeGridDay'
    },
    events: '/schedule/events',
    dateClick: function(info) {
      
      var dtLocal = new Date(info.date);
      var isoStr = dtLocal.toISOString().substring(0,16);
      document.getElementById('eventDate').value = isoStr;
      openEventModal();
    }
  });
  calendar.render();
});
```


## migrations/__init__.py

```py

```


## migrations/alembic.ini

```ini
[alembic]
# Path to migration scripts
script_location = migrations
# Database URL - this must point to the same absolute path as used by your app.
sqlalchemy.url = sqlite:////app/data/mydb.sqlite

[loggers]
keys = root,sqlalchemy,alembic

[handlers]
keys = console

[formatters]
keys = generic

[logger_root]
level = WARN
handlers = console

[logger_sqlalchemy]
level = WARN
handlers =
qualname = sqlalchemy.engine
propagate = 0

[logger_alembic]
level = INFO
handlers =
qualname = alembic
propagate = 0

[handler_console]
class = StreamHandler
args = (sys.stderr,)
level = NOTSET
formatter = generic

[formatter_generic]
format = %(levelname)-5.5s [%(name)s] %(message)s
datefmt = %Y-%m-%d %H:%M:%S
```


## migrations/env.py

```py
from __future__ import with_statement
import os
from logging.config import fileConfig

from sqlalchemy import engine_from_config, pool
from alembic import context

# Get the Alembic config and set up logging
config = context.config
fileConfig(config.config_file_name)

# Determine the project root and ensure the data folder exists.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
data_dir = os.path.join(project_root, 'data')
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

# Ensure an empty database file exists so that autogenerate has something to compare.
db_path = os.path.join(data_dir, 'mydb.sqlite')
if not os.path.exists(db_path):
    open(db_path, 'a').close()

# Import all your models so that they are registered with SQLAlchemy's metadata.
from models import db, Device, ImageDB, CropInfo, SendLog, ScheduleEvent, UserConfig, DeviceMetrics
target_metadata = db.metadata

def run_migrations_offline():
    """Run migrations in 'offline' mode."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True
    )

    with context.begin_transaction():
        context.run_migrations()

def run_migrations_online():
    """Run migrations in 'online' mode."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)

        with context.begin_transaction():
            context.run_migrations()

if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
```


## migrations/script.py.mako

```mako
<% 
import re
import uuid
%>
"""${message}

Revision ID: ${up_revision}
Revises: ${down_revision | n}
Create Date: ${create_date}
"""

# revision identifiers, used by Alembic.
revision = '${up_revision}'
down_revision = ${repr(down_revision)}
branch_labels = None
depends_on = None

def upgrade():
    ${upgrades if upgrades else "pass"}

def downgrade():
    ${downgrades if downgrades else "pass"}
```


## routes/__init__.py

```py

```


## routes/additional_routes.py

```py
from flask import Blueprint, request, jsonify, current_app
from models import Device, db
import subprocess, json

additional_bp = Blueprint('additional', __name__)

@additional_bp.route('/fetch_display_info', methods=['GET'])
def fetch_display_info():
    address = request.args.get('address')
    if not address:
        return jsonify({"status": "error", "message": "No address provided"}), 400
    cmd = f'curl -s "{address}/display_info"'
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        return jsonify({"status": "error", "message": result.stderr}), 500
    try:
        raw_info = json.loads(result.stdout)
        colour_str = raw_info.get("colour", "").capitalize()
        model_str = raw_info.get("model", "")
        resolution_arr = raw_info.get("resolution", [])
        if colour_str:
            display_name = f"{colour_str} Colour - {model_str}"
        else:
            display_name = model_str or "Unknown"
        if len(resolution_arr) == 2:
            resolution_str = f"{resolution_arr[0]}x{resolution_arr[1]}"
        else:
            resolution_str = "N/A"
        return jsonify({
            "status": "ok",
            "info": {
                "display_name": display_name,
                "resolution": resolution_str
            }
        }), 200
    except Exception:
        return jsonify({"status": "error", "message": "Invalid JSON returned"}), 500
```


## routes/ai_tagging_routes.py

```py
from flask import Blueprint, request, jsonify, current_app
import os
from tasks import (
    get_image_embedding,
    generate_tags_and_description,
    reembed_image,
    bulk_tag_images,
    BULK_PROGRESS
)
from models import db, ImageDB
from PIL import Image

ai_bp = Blueprint("ai_tagging", __name__)

@ai_bp.route("/api/ai_tag_image", methods=["POST"])
def ai_tag_image():
    data = request.get_json()
    filename = data.get("filename")
    if not filename:
        return jsonify({"status": "error", "message": "Filename is required"}), 400

    image_folder = current_app.config.get("IMAGE_FOLDER", "./images")
    image_path = os.path.join(image_folder, filename)
    if not os.path.exists(image_path):
        return jsonify({"status": "error", "message": "Image file not found"}), 404

    image_embedding = get_image_embedding(image_path)
    if image_embedding is None:
        return jsonify({"status": "error", "message": "Failed to get embedding"}), 500

    tags, description = generate_tags_and_description(image_embedding)
    # Update the ImageDB record with generated tags
    image_record = ImageDB.query.filter_by(filename=filename).first()
    if image_record:
        image_record.tags = ", ".join(tags)
        db.session.commit()
    else:
        image_record = ImageDB(filename=filename, tags=", ".join(tags))
        db.session.add(image_record)
        db.session.commit()

    return jsonify({
        "status": "success",
        "filename": filename,
        "tags": tags
    }), 200

@ai_bp.route("/api/search_images", methods=["GET"])
def search_images():
    q = request.args.get("q", "").strip()
    if not q:
        return jsonify({"status": "error", "message": "Missing query parameter"}), 400
    images = ImageDB.query.filter(ImageDB.tags.ilike(f"%{q}%")).all()
    results = {
        "ids": [img.filename for img in images],
        "tags": [img.tags for img in images]
    }
    return jsonify({"status": "success", "results": results}), 200

@ai_bp.route("/api/get_image_metadata", methods=["GET"])
def get_image_metadata():
    filename = request.args.get("filename", "").strip()
    if not filename:
        return jsonify({"status": "error", "message": "Missing filename"}), 400

    image_folder = current_app.config.get("IMAGE_FOLDER", "./images")
    image_path = os.path.join(image_folder, filename)
    resolution_str = "N/A"
    filesize_str = "N/A"
    if os.path.exists(image_path):
        try:
            size_bytes = os.path.getsize(image_path)
            filesize_mb = size_bytes / (1024 * 1024)
            filesize_str = f"{filesize_mb:.2f} MB"
            with Image.open(image_path) as im:
                w, h = im.size
                resolution_str = f"{w}x{h}"
        except Exception as ex:
            current_app.logger.warning(f"Could not read file info for {filename}: {ex}")

    image_record = ImageDB.query.filter_by(filename=filename).first()
    if image_record:
        tags = [t.strip() for t in image_record.tags.split(",")] if image_record.tags else []
        favorite = image_record.favorite
    else:
        tags = []
        favorite = False

    return jsonify({
        "status": "success",
        "tags": tags,
        "favorite": favorite,
        "resolution": resolution_str,
        "filesize": filesize_str
    }), 200

@ai_bp.route("/api/update_image_metadata", methods=["POST"])
def update_image_metadata():
    data = request.get_json() or {}
    filename = data.get("filename", "").strip()
    new_tags = data.get("tags", [])
    if isinstance(new_tags, list):
        tags_str = ", ".join(new_tags)
    else:
        tags_str = new_tags
    favorite = data.get("favorite", None)
    if not filename:
        return jsonify({"status": "error", "message": "Missing filename"}), 400

    image_record = ImageDB.query.filter_by(filename=filename).first()
    if not image_record:
        return jsonify({"status": "error", "message": "Image not found"}), 404

    image_record.tags = tags_str
    if favorite is not None:
        image_record.favorite = bool(favorite)
    db.session.commit()
    return jsonify({"status": "success"}), 200

@ai_bp.route("/api/reembed_image", methods=["GET"])
def reembed_image_endpoint():
    filename = request.args.get("filename", "").strip()
    if not filename:
        return jsonify({"status": "error", "message": "Missing filename"}), 400
    result = reembed_image(filename)
    return jsonify(result)

@ai_bp.route("/api/reembed_all_images", methods=["GET"])
def reembed_all_images_endpoint():
    task_id = bulk_tag_images.delay()
    if not task_id:
        return jsonify({"status": "error", "message": "No images found"}), 404
    return jsonify({"status": "success", "message": f"Reembedding images in background. Task ID: {task_id}"}), 200
```


## routes/device_info_routes.py

```py
from flask import Blueprint, request, jsonify, Response, stream_with_context
import httpx
import json
import time
from datetime import datetime
from models import db, Device, DeviceMetrics

device_info_bp = Blueprint('device_info', __name__)

@device_info_bp.route('/device_info', methods=['GET'])
def get_device_info():
    address = request.args.get('address')
    if not address:
        return jsonify({"status": "error", "message": "No address provided"}), 400

    # Ensure the address has a scheme
    if not (address.startswith("http://") or address.startswith("https://")):
        address = "http://" + address

    try:
        # Use httpx with a 10-second timeout and a curl-like User-Agent
        response = httpx.get(f"{address}/display_info", timeout=10.0, headers={'User-Agent': 'curl/7.68.0'})
        response.raise_for_status()
        raw_info = response.json()
    except Exception as e:
        return jsonify({"status": "error", "message": f"Error fetching display info: {str(e)}"}), 500

    try:
        # Build display name as "model colour color"
        colour = raw_info.get("colour", "").strip()
        model = raw_info.get("model", "").strip()
        if model and colour:
            display_name = f"{model} {colour} color"
        elif model:
            display_name = model
        else:
            display_name = "Unknown"
        # Format resolution as "widthxheight"
        resolution_arr = raw_info.get("resolution", [])
        if isinstance(resolution_arr, list) and len(resolution_arr) == 2:
            resolution = f"{resolution_arr[0]}x{resolution_arr[1]}"
        else:
            resolution = "N/A"
    except Exception:
        return jsonify({"status": "error", "message": "Invalid JSON returned"}), 500

    return jsonify({
        "status": "ok",
        "info": {
            "display_name": display_name,
            "resolution": resolution
        }
    }), 200

# Global dictionary to store active device streams
active_device_streams = {}

@device_info_bp.route('/device/<int:device_index>/stream', methods=['GET'])
def device_stream(device_index):
    """
    Server-Sent Events (SSE) endpoint for streaming device metrics.
    This connects to the device's /stream endpoint and forwards the data.
    """
    devices = Device.query.order_by(Device.id).all()
    if not (0 <= device_index < len(devices)):
        return jsonify({"status": "error", "message": "Device not found"}), 404
    
    device = devices[device_index]
    address = device.address
    
    # Ensure the address has a scheme
    if not (address.startswith("http://") or address.startswith("https://")):
        address = "http://" + address
    
    def generate():
        """Generate SSE data from the device's stream endpoint"""
        try:
            # Initial connection message
            yield f"data: {json.dumps({'status': 'connecting'})}\n\n"
            
            # Check if we already have an active stream for this device
            if device_index in active_device_streams:
                # Use the existing stream
                client = active_device_streams[device_index]
            else:
                # Try to connect to the device's stream endpoint with a longer timeout
                try:
                    # First, check if the device is reachable
                    info_response = httpx.get(f"{address}/display_info", timeout=5.0)
                    if info_response.status_code != 200:
                        yield f"data: {json.dumps({'status': 'error', 'message': f'Device not responding: {info_response.status_code}'})}\n\n"
                        return
                    
                    # Device is reachable, now connect to the stream
                    client = httpx.Client(timeout=None)
                    response = client.get(f"{address}/stream", stream=True)
                    if response.status_code != 200:
                        yield f"data: {json.dumps({'status': 'error', 'message': f'Error connecting to stream: {response.status_code}'})}\n\n"
                        return
                    
                    # Store the client for future use
                    active_device_streams[device_index] = client
                    
                    # Mark device as online
                    device.online = True
                    db.session.commit()
                    
                    # Start a background thread to continuously read from the stream and update the database
                    import threading
                    
                    def stream_reader():
                        try:
                            for line in response.iter_lines():
                                if line:
                                    try:
                                        # Parse the SSE data
                                        if line.startswith(b'data: '):
                                            data_str = line[6:].decode('utf-8')
                                        else:
                                            data_str = line.decode('utf-8')
                                        
                                        # Try to parse as JSON
                                        try:
                                            data = json.loads(data_str)
                                            
                                            # Update device metrics in the database
                                            if 'cpu' in data and ('memory' in data or 'mem' in data) and 'disk' in data:
                                                memory_value = data.get('memory', data.get('mem', 0))
                                                
                                                device.cpu_usage = str(data['cpu'])
                                                device.mem_usage = str(memory_value)
                                                device.disk_usage = str(data['disk'])
                                                device.online = True
                                                db.session.commit()
                                                
                                                # Also save to DeviceMetrics table
                                                new_metric = DeviceMetrics(
                                                    device_id=device.id,
                                                    cpu=data['cpu'],
                                                    memory=memory_value,
                                                    disk=data['disk']
                                                )
                                                db.session.add(new_metric)
                                                db.session.commit()
                                        except json.JSONDecodeError:
                                            # If not valid JSON, try to extract metrics from the text
                                            if "cpu:" in data_str.lower() or "memory:" in data_str.lower():
                                                # Simple parsing for non-JSON format
                                                metrics = {}
                                                for part in data_str.split(','):
                                                    if ':' in part:
                                                        key, value = part.split(':', 1)
                                                        key = key.strip().lower()
                                                        value = value.strip().replace('%', '')
                                                        try:
                                                            metrics[key] = float(value)
                                                        except ValueError:
                                                            pass
                                                
                                                if metrics:
                                                    if 'cpu' in metrics:
                                                        device.cpu_usage = str(metrics['cpu'])
                                                    if 'memory' in metrics or 'mem' in metrics:
                                                        device.mem_usage = str(metrics.get('memory', metrics.get('mem', 0)))
                                                    if 'disk' in metrics:
                                                        device.disk_usage = str(metrics['disk'])
                                                    device.online = True
                                                    db.session.commit()
                                    except Exception as e:
                                        print(f"Error processing stream data: {e}")
                        except Exception as e:
                            print(f"Stream reader thread error: {e}")
                            # Mark device as offline
                            device.online = False
                            db.session.commit()
                            # Remove from active streams
                            if device_index in active_device_streams:
                                del active_device_streams[device_index]
                    
                    # Start the stream reader thread
                    thread = threading.Thread(target=stream_reader, daemon=True)
                    thread.start()
                except httpx.RequestError as e:
                    yield f"data: {json.dumps({'status': 'error', 'message': f'Connection error: {str(e)}'})}\n\n"
                    # Mark device as offline
                    device.online = False
                    db.session.commit()
                    
                    # Simulate some metrics for testing
                    # This is just for development and should be removed in production
                    import random
                    for _ in range(10):
                        time.sleep(1)
                        fake_metrics = {
                            'cpu': round(random.uniform(10, 90), 1),
                            'memory': round(random.uniform(20, 80), 1),
                            'disk': round(random.uniform(30, 70), 1)
                        }
                        yield f"data: {json.dumps(fake_metrics)}\n\n"
                        
                        # Update device metrics with simulated data
                        device.cpu_usage = str(fake_metrics['cpu'])
                        device.mem_usage = str(fake_metrics['memory'])
                        device.disk_usage = str(fake_metrics['disk'])
                        db.session.commit()
                        
                        # Also save to DeviceMetrics table
                        new_metric = DeviceMetrics(
                            device_id=device.id,
                            cpu=fake_metrics['cpu'],
                            memory=fake_metrics['memory'],
                            disk=fake_metrics['disk']
                        )
                        db.session.add(new_metric)
                        db.session.commit()
            
            # Now that we have the stream set up, periodically send the latest metrics to the client
            while True:
                # Get the latest metrics from the database
                latest_metric = DeviceMetrics.query.filter_by(device_id=device.id).order_by(DeviceMetrics.timestamp.desc()).first()
                
                if latest_metric:
                    data = {
                        'cpu': latest_metric.cpu,
                        'memory': latest_metric.memory,
                        'disk': latest_metric.disk,
                        'timestamp': latest_metric.timestamp.isoformat()
                    }
                    yield f"data: {json.dumps(data)}\n\n"
                else:
                    # If no metrics are available, send the current device status
                    data = {
                        'cpu': device.cpu_usage,
                        'memory': device.mem_usage,
                        'disk': device.disk_usage,
                        'online': device.online
                    }
                    yield f"data: {json.dumps(data)}\n\n"
                
                # Wait 5 seconds before sending the next update
                time.sleep(5)
                
        except Exception as e:
            yield f"data: {json.dumps({'status': 'error', 'message': f'Stream error: {str(e)}'})}\n\n"
            # Mark device as offline
            device.online = False
            db.session.commit()
    
    return Response(stream_with_context(generate()), mimetype='text/event-stream')

@device_info_bp.route('/devices/metrics', methods=['GET'])
def devices_metrics():
    """Get metrics for all devices from the database"""
    devices = Device.query.all()
    data = []
    for idx, device in enumerate(devices):
        # Get the latest metrics from DeviceMetrics table
        latest_metric = DeviceMetrics.query.filter_by(device_id=device.id).order_by(DeviceMetrics.timestamp.desc()).first()
        
        if latest_metric and (datetime.utcnow() - latest_metric.timestamp).total_seconds() < 30:
            # If we have recent metrics (less than 30 seconds old), use them and mark device as online
            device.online = True
            device.cpu_usage = str(latest_metric.cpu)
            device.mem_usage = str(latest_metric.memory)
            device.disk_usage = str(latest_metric.disk)
            db.session.commit()
            
            data.append({
                "index": idx,
                "cpu": device.cpu_usage,
                "mem": device.mem_usage,
                "disk": device.disk_usage,
                "online": True
            })
        else:
            # If no recent metrics, use stored values but mark as offline if too old
            if latest_metric and (datetime.utcnow() - latest_metric.timestamp).total_seconds() > 300:
                device.online = False
                db.session.commit()
                
            data.append({
                "index": idx,
                "cpu": device.cpu_usage,
                "mem": device.mem_usage,
                "disk": device.disk_usage,
                "online": device.online
            })
    
    return jsonify({"status": "success", "devices": data})
```


## routes/device_routes.py

```py
from flask import Blueprint, request, jsonify, current_app, send_file
from models import db, Device
import httpx
import subprocess, os, datetime, json

device_bp = Blueprint('device', __name__)

@device_bp.route('/device/<int:index>/set_orientation', methods=['POST'])
def set_device_orientation(index):
    orientation = request.form.get('orientation')
    if not orientation:
        return jsonify({"status": "error", "message": "No orientation provided"}), 400
    all_devices = Device.query.order_by(Device.id).all()
    if index < 0 or index >= len(all_devices):
        return jsonify({"status": "error", "message": "Device not found"}), 404
    device_address = all_devices[index].address
    if not (device_address.startswith("http://") or device_address.startswith("https://")):
        device_address = "http://" + device_address
    try:
        response = httpx.post(f"{device_address}/set_orientation", data={"orientation": orientation}, timeout=5.0)
        response.raise_for_status()
        return jsonify({"status": "ok"}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@device_bp.route('/device/<int:index>/display_info', methods=['GET'])
def get_device_info(index):
    all_devices = Device.query.order_by(Device.id).all()
    if index < 0 or index >= len(all_devices):
        return jsonify({"status": "error", "message": "Device not found"}), 404
    device_address = all_devices[index].address
    if not (device_address.startswith("http://") or device_address.startswith("https://")):
        device_address = "http://" + device_address
    try:
        # Use a curl-like User-Agent to mimic curl behavior
        response = httpx.get(f"{device_address}/display_info", timeout=5.0, headers={"User-Agent": "curl/7.68.0"})
        response.raise_for_status()
        raw = response.json()
        return jsonify({"status": "ok", "info": raw})
    except Exception as e:
        return jsonify({"status": "error", "message": f"Error fetching display info: {str(e)}"}), 500

@device_bp.route('/device/<int:index>/fetch_metrics', methods=['GET'])
def fetch_metrics(index):
    """
    Fetch the first SSE metric line from the device's /stream endpoint using httpx streaming.
    """
    all_devices = Device.query.order_by(Device.id).all()
    if index < 0 or index >= len(all_devices):
        return jsonify({"status": "error", "message": "Device not found"}), 404
    device = all_devices[index]
    address = device.address
    if not (address.startswith("http://") or address.startswith("https://")):
        address = "http://" + address
    try:
        with httpx.stream("GET", f"{address}/stream", timeout=10.0, headers={"User-Agent": "curl/7.68.0"}) as response:
            for line in response.iter_lines():
                if line:
                    # httpx.iter_lines() returns bytes if no decoding is set
                    if isinstance(line, bytes):
                        line = line.decode("utf-8")
                    if line.startswith("data:"):
                        data_str = line[len("data:"):].strip()
                        data_json = json.loads(data_str)
                        cpu_usage = str(data_json.get("cpu", "N/A"))
                        mem_usage = str(data_json.get("memory", "N/A"))
                        disk_usage = str(data_json.get("disk", "N/A"))
                        device.cpu_usage = cpu_usage
                        device.mem_usage = mem_usage
                        device.disk_usage = disk_usage
                        device.online = True
                        db.session.commit()
                        return jsonify({
                            "status": "ok",
                            "cpu": cpu_usage + "%",
                            "mem": mem_usage + "%",
                            "disk": disk_usage + "%",
                            "online": device.online
                        })
            # If no valid line is found:
            device.online = False
            db.session.commit()
            return jsonify({"status": "error", "message": "No metrics data received"}), 500
    except Exception as e:
        device.online = False
        db.session.commit()
        return jsonify({"status": "error", "message": str(e)}), 500

@device_bp.route('/device/<int:index>/system_update', methods=['POST'])
def system_update(index):
    all_devices = Device.query.order_by(Device.id).all()
    if index < 0 or index >= len(all_devices):
        return jsonify({"status": "error", "message": "Device not found"}), 404
    device_address = all_devices[index].address
    if not (device_address.startswith("http://") or device_address.startswith("https://")):
        device_address = "http://" + device_address
    try:
        response = httpx.post(f"{device_address}/system_update", timeout=10.0)
        response.raise_for_status()
        return jsonify({"status": "ok"}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@device_bp.route('/device/<int:index>/backup', methods=['GET'])
def create_disk_backup(index):
    from flask import send_file
    all_devices = Device.query.order_by(Device.id).all()
    if index < 0 or index >= len(all_devices):
        return jsonify({"status": "error", "message": "Device not found"}), 404
    device_address = all_devices[index].address
    if not (device_address.startswith("http://") or device_address.startswith("https://")):
        device_address = "http://" + device_address
    data_folder = current_app.config['DATA_FOLDER']
    backup_dir = os.path.join(data_folder, "display_backup")
    if not os.path.exists(backup_dir):
        os.makedirs(backup_dir)
    backup_filename = f"backup_{index}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.img.gz"
    backup_path = os.path.join(backup_dir, backup_filename)
    try:
        response = httpx.post(f"{device_address}/backup", timeout=30.0)
        response.raise_for_status()
        with open(backup_path, "wb") as f:
            f.write(response.content)
        if os.path.exists(backup_path):
            return send_file(backup_path, mimetype='application/gzip',
                             as_attachment=True, download_name=backup_filename)
        else:
            return jsonify({"status": "error", "message": "Backup file not created"}), 500
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@device_bp.route('/device/<int:index>/update', methods=['POST'])
def update_application(index):
    all_devices = Device.query.order_by(Device.id).all()
    if index < 0 or index >= len(all_devices):
        return jsonify({"status": "error", "message": "Device not found"}), 404
    device_address = all_devices[index].address
    if not (device_address.startswith("http://") or device_address.startswith("https://")):
        device_address = "http://" + device_address
    try:
        response = httpx.post(f"{device_address}/update", timeout=10.0)
        response.raise_for_status()
        return jsonify({"status": "ok"}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@device_bp.route('/device/<int:index>/stream')
def metrics_stream(index):
    all_devices = Device.query.order_by(Device.id).all()
    if index < 0 or index >= len(all_devices):
        return "Device not found", 404
    device_address = all_devices[index].address
    def generate():
        command = f'curl -N -s "{device_address}/stream"'
        process = subprocess.Popen(command, shell=True,
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, text=True)
        try:
            while True:
                line = process.stdout.readline()
                if not line:
                    break
                yield line
        except Exception:
            process.kill()
        finally:
            process.kill()
    return current_app.response_class(generate(), mimetype='text/event-stream')

@device_bp.route('/test_device/<int:index>', methods=['GET'])
def test_device(index):
    all_devices = Device.query.order_by(Device.id).all()
    if index < 0 or index >= len(all_devices):
        return jsonify({"status": "error", "message": "Device not found"}), 404
    device = all_devices[index]
    address = device.address
    cmd = f'curl -I --max-time 5 -s "{address}"'
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    current_app.logger.info("Curl output for %s: %s", address, result.stdout)
    if result.returncode == 0 and "200 OK" in result.stdout.upper():
        device.online = True
        db.session.commit()
        return jsonify({"status": "ok"}), 200
    else:
        device.online = False
        db.session.commit()
        return jsonify({"status": "error"}), 500

@device_bp.route('/test_connection_address', methods=['GET'])
def test_connection_address():
    address = request.args.get('address')
    if not address:
        return jsonify({"status": "error", "message": "No address provided"}), 400
    cmd = f'curl -I --max-time 5 -s "{address}"'
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    current_app.logger.info("Curl output for %s: %s", address, result.stdout)
    if result.returncode == 0 and "200 OK" in result.stdout.upper():
        return jsonify({"status": "ok"}), 200
    else:
        return jsonify({"status": "failed"}), 500
```


## routes/image_routes.py

```py
from flask import Blueprint, request, redirect, url_for, render_template, flash, send_from_directory, send_file, jsonify, current_app
from models import db, ImageDB, CropInfo, SendLog, Device
import os
from PIL import Image
import subprocess
from utils.image_helpers import allowed_file, convert_to_jpeg
from utils.crop_helpers import load_crop_info_from_db, save_crop_info_to_db, add_send_log_entry, get_last_sent

image_bp = Blueprint('image', __name__)

@image_bp.route('/thumbnail/<filename>')
def thumbnail(filename):
    image_folder = current_app.config['IMAGE_FOLDER']
    thumbnail_folder = current_app.config['THUMBNAIL_FOLDER']
    thumb_path = os.path.join(thumbnail_folder, filename)
    image_path = os.path.join(image_folder, filename)
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
            current_app.logger.error("Error generating thumbnail for %s: %s", filename, e)
            return "Error generating thumbnail", 500
    return send_from_directory(thumbnail_folder, filename)

@image_bp.route('/', methods=['GET', 'POST'])
def upload_file():
    image_folder = current_app.config['IMAGE_FOLDER']
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
                    new_filename = convert_to_jpeg(file, base, image_folder)
                    if new_filename is None:
                        flash(f"Error converting {original_filename} to JPEG.")
                    else:
                        if not ImageDB.query.filter_by(filename=new_filename).first():
                            db.session.add(ImageDB(filename=new_filename))
                            db.session.commit()
                            # Trigger automatic image tagging
                            from tasks import process_image_tagging
                            process_image_tagging.delay(new_filename)
                else:
                    filepath = os.path.join(image_folder, original_filename)
                    file.save(filepath)
                    if not ImageDB.query.filter_by(filename=original_filename).first():
                        db.session.add(ImageDB(filename=original_filename))
                        db.session.commit()
                        # Trigger automatic image tagging
                        from tasks import process_image_tagging
                        process_image_tagging.delay(original_filename)
        return redirect(url_for('image.upload_file'))
    
    images_db = ImageDB.query.all()
    images = [img.filename for img in images_db]
    devices_db = Device.query.all()
    devices = []
    for d in devices_db:
        devices.append({
            "color": d.color,
            "friendly_name": d.friendly_name,
            "orientation": d.orientation,
            "address": d.address,
            "display_name": d.display_name,
            "resolution": d.resolution,
            "online": d.online,
            "cpu_usage": d.cpu_usage,
            "mem_usage": d.mem_usage,
            "disk_usage": d.disk_usage,
            "last_sent": d.last_sent
        })
    last_sent = get_last_sent()
    return render_template('index.html', images=images, devices=devices, last_sent=last_sent)

@image_bp.route('/images/<filename>')
def uploaded_file(filename):
    image_folder = current_app.config['IMAGE_FOLDER']
    filepath = os.path.join(image_folder, filename)
    if not os.path.exists(filepath):
        return "File not found", 404
    if request.args.get("size") == "info":
        try:
            with Image.open(filepath) as img:
                max_width = 300
                w, h = img.size
                if w > max_width:
                    ratio = max_width / float(w)
                    new_size = (max_width, int(h * ratio))
                    img = img.resize(new_size, Image.LANCZOS)
                from io import BytesIO
                buf = BytesIO()
                img.save(buf, format="JPEG")
                buf.seek(0)
                return send_file(buf, mimetype='image/jpeg')
        except Exception as e:
            current_app.logger.error("Error processing image %s for info: %s", filename, e)
            return "Error processing image", 500
    ext = filename.rsplit('.', 1)[1].lower()
    if ext == "heic":
        try:
            from io import BytesIO
            with Image.open(filepath) as img:
                buf = BytesIO()
                img.save(buf, format="JPEG")
                buf.seek(0)
                return send_file(buf, mimetype='image/jpeg')
        except Exception as e:
            current_app.logger.error("Error processing HEIC image %s: %s", filename, e)
            return "Error processing image", 500
    else:
        return send_from_directory(image_folder, filename)

@image_bp.route('/save_crop_info/<filename>', methods=['POST'])
def save_crop_info_endpoint(filename):
    crop_data = request.get_json()
    if not crop_data:
        return jsonify({"status": "error", "message": "No crop data provided"}), 400
    save_crop_info_to_db(filename, crop_data)
    return jsonify({"status": "success"}), 200

@image_bp.route('/send_image/<filename>', methods=['POST'])
def send_image(filename):
    image_folder = current_app.config['IMAGE_FOLDER']
    data_folder = current_app.config['DATA_FOLDER']
    filepath = os.path.join(image_folder, filename)
    if not os.path.exists(filepath):
        return "File not found", 404
    device_addr = request.form.get("device")
    if not device_addr:
        return "No device specified", 400

    from models import Device
    device_obj = Device.query.filter_by(address=device_addr).first()
    if not device_obj:
        return "Device not found in DB", 500
    dev_width = None
    dev_height = None
    if device_obj.resolution:
        parts = device_obj.resolution.split("x")
        if len(parts) == 2:
            try:
                dev_width = int(parts[0])
                dev_height = int(parts[1])
            except:
                pass
    if not (dev_width and dev_height):
        return "Target resolution not found", 500

    try:
        with Image.open(filepath) as orig_img:
            orig_w, orig_h = orig_img.size
            
            # Check if device is in portrait orientation
            is_portrait = device_obj.orientation.lower() == 'portrait'
            
            # If portrait, swap width and height for target ratio calculation
            if is_portrait:
                target_ratio = dev_height / dev_width
            else:
                target_ratio = dev_width / dev_height
                
            cdata = load_crop_info_from_db(filename)
            if cdata:
                x = cdata.get("x", 0)
                y = cdata.get("y", 0)
                w = cdata.get("width", orig_w)
                h = cdata.get("height", orig_h)
                cropped = orig_img.crop((x, y, x+w, y+h))
            else:
                orig_ratio = orig_w / orig_h
                if orig_ratio > target_ratio:
                    new_width = int(orig_h * target_ratio)
                    left = (orig_w - new_width) // 2
                    crop_box = (left, 0, left + new_width, orig_h)
                else:
                    new_height = int(orig_w / target_ratio)
                    top = (orig_h - new_height) // 2
                    crop_box = (0, top, orig_w, top + new_height)
                cropped = orig_img.crop(crop_box)

            # If portrait, rotate the image 90 degrees clockwise and swap dimensions
            if is_portrait:
                cropped = cropped.rotate(-90, expand=True)  # -90 for clockwise rotation
                final_img = cropped.resize((dev_height, dev_width), Image.LANCZOS)  # Note swapped dimensions
            else:
                final_img = cropped.resize((dev_width, dev_height), Image.LANCZOS)
            temp_dir = os.path.join(data_folder, "temp")
            if not os.path.exists(temp_dir):
                os.makedirs(temp_dir)
            temp_filename = os.path.join(temp_dir, f"temp_{filename}")
            final_img.save(temp_filename, format="JPEG", quality=95)

        cmd = f'curl "{device_addr}/send_image" -X POST -F "file=@{temp_filename}"'
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        os.remove(temp_filename)
        if result.returncode != 0:
            return f"Error sending image: {result.stderr}", 500

        device_obj.last_sent = filename
        db.session.commit()
        add_send_log_entry(filename)
        return f"Image sent successfully: {result.stdout}", 200
    except Exception as e:
        current_app.logger.error("Error resizing/cropping image: %s", e)
        return f"Error processing image: {e}", 500

@image_bp.route('/delete_image/<filename>', methods=['POST'])
def delete_image_endpoint(filename):
    image_folder = current_app.config['IMAGE_FOLDER']
    thumbnail_folder = current_app.config['THUMBNAIL_FOLDER']
    filepath = os.path.join(image_folder, filename)
    thumb_path = os.path.join(thumbnail_folder, filename)
    if os.path.exists(filepath):
        try:
            os.remove(filepath)
            if os.path.exists(thumb_path):
                os.remove(thumb_path)
            img_obj = ImageDB.query.filter_by(filename=filename).first()
            if img_obj:
                db.session.delete(img_obj)
                db.session.commit()
            return jsonify({"status": "success", "message": "Image deleted"}), 200
        except Exception as e:
            current_app.logger.error("Error removing file %s: %s", filepath, e)
            return jsonify({"status": "error", "message": str(e)}), 500
    else:
        return jsonify({"status": "error", "message": "File not found"}), 404
```


## routes/schedule_routes.py

```py
from flask import Blueprint, request, jsonify, render_template
from models import db, ScheduleEvent, Device, ImageDB
import datetime
from tasks import scheduler, send_scheduled_image

schedule_bp = Blueprint('schedule', __name__)

@schedule_bp.route('/schedule')
def schedule_page():
    devs = Device.query.all()
    devices = []
    for d in devs:
        devices.append({
            "color": d.color,
            "friendly_name": d.friendly_name,
            "orientation": d.orientation,
            "address": d.address,
            "display_name": d.display_name,
            "resolution": d.resolution,
            "online": d.online,
            "cpu_usage": d.cpu_usage,
            "mem_usage": d.mem_usage,
            "disk_usage": d.disk_usage,
            "last_sent": d.last_sent
        })
    
    # Get all images and their tags
    imgs_db = ImageDB.query.all()
    images = [i.filename for i in imgs_db]
    
    # Create a dictionary of image tags for the template
    image_tags = {}
    for img in imgs_db:
        if img.tags:
            image_tags[img.filename] = img.tags
    
    return render_template("schedule.html", devices=devices, images=images, image_tags=image_tags)

@schedule_bp.route('/schedule/events')
def get_events():
    events = ScheduleEvent.query.all()
    event_list = []
    now = datetime.datetime.now()
    horizon = now + datetime.timedelta(days=90)
    
    # Create a device lookup dictionary for colors and friendly names
    device_lookup = {}
    devices = Device.query.all()
    for device in devices:
        device_lookup[device.address] = {
            "color": device.color,
            "friendly_name": device.friendly_name
        }
    
    # Create an image lookup for thumbnails
    image_lookup = {}
    images = ImageDB.query.all()
    for img in images:
        image_lookup[img.filename] = {
            "tags": img.tags or ""
        }
    
    for ev in events:
        # Get device color and friendly name
        device_info = device_lookup.get(ev.device, {"color": "#cccccc", "friendly_name": ev.device})
        device_color = device_info["color"]
        device_name = device_info["friendly_name"]
        
        if ev.recurrence.lower() == "none":
            event_list.append({
                "id": ev.id,
                "title": f"{ev.filename}",
                "start": ev.datetime_str,
                "device": ev.device,
                "deviceName": device_name,
                "filename": ev.filename,
                "recurrence": ev.recurrence,
                "series": False,
                "backgroundColor": device_color,
                "borderColor": device_color,
                "textColor": "#ffffff",
                "extendedProps": {
                    "thumbnail": f"/image/thumbnail/{ev.filename}"
                }
            })
        else:
            # Generate recurring occurrences from the next occurrence up to the horizon
            try:
                start_dt = datetime.datetime.fromisoformat(ev.datetime_str)
            except Exception:
                continue
            # Advance to the first occurrence that is >= now
            rec = ev.recurrence.lower()
            occurrence = start_dt
            while occurrence < now:
                if rec == "daily":
                    occurrence += datetime.timedelta(days=1)
                elif rec == "weekly":
                    occurrence += datetime.timedelta(weeks=1)
                elif rec == "monthly":
                    occurrence += datetime.timedelta(days=30)
                else:
                    break
            # Generate occurrences until the horizon
            while occurrence <= horizon:
                event_list.append({
                    "id": ev.id,  # same series id
                    "title": f"{ev.filename}",
                    "start": occurrence.isoformat(sep=' '),
                    "device": ev.device,
                    "deviceName": device_name,
                    "filename": ev.filename,
                    "recurrence": ev.recurrence,
                    "series": True,
                    "backgroundColor": device_color,
                    "borderColor": device_color,
                    "textColor": "#ffffff",
                    "extendedProps": {
                        "thumbnail": f"/image/thumbnail/{ev.filename}"
                    }
                })
                if rec == "daily":
                    occurrence += datetime.timedelta(days=1)
                elif rec == "weekly":
                    occurrence += datetime.timedelta(weeks=1)
                elif rec == "monthly":
                    occurrence += datetime.timedelta(days=30)
                else:
                    break
    return jsonify(event_list)

@schedule_bp.route('/schedule/add', methods=['POST'])
def add_event():
    data = request.get_json()
    datetime_str = data.get("datetime")
    device = data.get("device")
    filename = data.get("filename")
    recurrence = data.get("recurrence", "none")
    if not (datetime_str and device and filename):
        return jsonify({"status": "error", "message": "Missing parameters"}), 400
    try:
        dt = datetime.datetime.fromisoformat(datetime_str)
        formatted_dt_str = dt.isoformat(sep=' ')
    except Exception:
        return jsonify({"status": "error", "message": "Invalid datetime format"}), 400
    new_event = ScheduleEvent(
        filename=filename,
        device=device,
        datetime_str=formatted_dt_str,
        sent=False,
        recurrence=recurrence
    )
    db.session.add(new_event)
    db.session.commit()
    scheduler.add_job(send_scheduled_image, 'date', run_date=dt, args=[new_event.id])
    return jsonify({"status": "success", "event": {
        "id": new_event.id,
        "filename": new_event.filename,
        "datetime": new_event.datetime_str,
        "recurrence": recurrence
    }})

@schedule_bp.route('/schedule/remove/<int:event_id>', methods=['POST'])
def remove_event(event_id):
    ev = ScheduleEvent.query.get(event_id)
    if ev:
        db.session.delete(ev)
        db.session.commit()
    return jsonify({"status": "success"})

@schedule_bp.route('/schedule/update', methods=['POST'])
def update_event():
    data = request.get_json()
    event_id = data.get("event_id")
    new_datetime = data.get("datetime")
    if not (event_id and new_datetime):
        return jsonify({"status": "error", "message": "Missing parameters"}), 400
    ev = ScheduleEvent.query.get(event_id)
    if not ev:
        return jsonify({"status": "error", "message": "Event not found"}), 404
    ev.datetime_str = new_datetime
    db.session.commit()
    return jsonify({"status": "success"})

@schedule_bp.route('/schedule/skip/<int:event_id>', methods=['POST'])
def skip_event_occurrence(event_id):
    ev = ScheduleEvent.query.get(event_id)
    if not ev:
        return jsonify({"status": "error", "message": "Event not found"}), 404
    if ev.recurrence.lower() == "none":
        return jsonify({"status": "error", "message": "Not a recurring event"}), 400
    try:
        dt = datetime.datetime.fromisoformat(ev.datetime_str)
        if ev.recurrence.lower() == "daily":
            next_dt = dt + datetime.timedelta(days=1)
        elif ev.recurrence.lower() == "weekly":
            next_dt = dt + datetime.timedelta(weeks=1)
        elif ev.recurrence.lower() == "monthly":
            next_dt = dt + datetime.timedelta(days=30)
        else:
            return jsonify({"status": "error", "message": "Unknown recurrence type"}), 400
        ev.datetime_str = next_dt.isoformat(sep=' ')
        ev.sent = False
        db.session.commit()
        scheduler.add_job(send_scheduled_image, 'date', run_date=next_dt, args=[ev.id])
        return jsonify({"status": "success", "message": "Occurrence skipped"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500
```


## routes/settings_routes.py

```py
from flask import Blueprint, request, render_template, flash, redirect, url_for, jsonify
from models import db, Device, UserConfig, DeviceMetrics
import logging
import httpx  # for querying the Ollama API
from datetime import datetime

settings_bp = Blueprint('settings', __name__)
logger = logging.getLogger(__name__)

@settings_bp.route('/settings', methods=['GET', 'POST'])
def settings():
    if request.method == 'POST':
        color = request.form.get("color")
        friendly_name = request.form.get("friendly_name")
        orientation = request.form.get("orientation")
        address = request.form.get("address")
        display_name = request.form.get("display_name") or "Unknown"
        resolution = request.form.get("resolution") or "N/A"
        if color and friendly_name and orientation and address:
            new_dev = Device(
                color=color,
                friendly_name=friendly_name,
                orientation=orientation,
                address=address,
                display_name=display_name,
                resolution=resolution,
                online=False,
                cpu_usage="N/A",
                mem_usage="N/A",
                disk_usage="N/A"
            )
            db.session.add(new_dev)
            db.session.commit()
            flash("Device added successfully", "success")
        else:
            flash("Missing mandatory fields (color, friendly name, orientation, address).", "error")
        return redirect(url_for("settings.settings"))
    else:
        devs = Device.query.all()
        devices = []
        for d in devs:
            devices.append({
                "color": d.color,
                "friendly_name": d.friendly_name,
                "orientation": d.orientation,
                "address": d.address,
                "display_name": d.display_name,
                "resolution": d.resolution,
                "online": d.online,
                "cpu_usage": d.cpu_usage,
                "mem_usage": d.mem_usage,
                "disk_usage": d.disk_usage
            })
        config = UserConfig.query.first()
        return render_template("settings.html", devices=devices, config=config)

@settings_bp.route('/delete_device/<int:device_index>', methods=['POST'])
def delete_device(device_index):
    all_devices = Device.query.order_by(Device.id).all()
    if 0 <= device_index < len(all_devices):
        db.session.delete(all_devices[device_index])
        db.session.commit()
        flash("Device deleted", "success")
    else:
        flash("Device not found", "error")
    return redirect(url_for("settings.settings"))

@settings_bp.route('/edit_device', methods=['POST'])
def edit_device():
    try:
        index = int(request.form.get("device_index"))
        color = request.form.get("color")
        friendly_name = request.form.get("friendly_name")
        orientation = request.form.get("orientation")
        address = request.form.get("address")
        all_devices = Device.query.order_by(Device.id).all()
        if 0 <= index < len(all_devices):
            d = all_devices[index]
            d.color = color or d.color
            d.friendly_name = friendly_name
            d.orientation = orientation
            d.address = address
            db.session.commit()
            flash("Device updated successfully", "success")
        else:
            flash("Device index not found", "error")
    except Exception as e:
        flash("Error editing device: " + str(e), "error")
    return redirect(url_for("settings.settings"))

@settings_bp.route('/settings/update_clip_model', methods=['POST'])
def update_clip_model():
    data = request.get_json()
    config = UserConfig.query.first()
    if not config:
        config = UserConfig(location="London")
        db.session.add(config)
    
    if "clip_model" in data:
        config.clip_model = data.get("clip_model")
        db.session.commit()
        return jsonify({"status": "success", "message": "CLIP model updated."})
    else:
        return jsonify({"status": "error", "message": "No CLIP model provided."})

@settings_bp.route('/settings/rerun_all_tagging', methods=['POST'])
def rerun_all_tagging():
    try:
        # Import the task for rerunning tagging
        from tasks import reembed_all_images
        
        # Start the task
        task = reembed_all_images.delay()
        
        return jsonify({
            "status": "success",
            "message": "Tagging process started.",
            "task_id": str(task.id)
        })
    except Exception as e:
        logger.error(f"Error starting retagging: {str(e)}")
        return jsonify({"status": "error", "message": f"Error: {str(e)}"})

@settings_bp.route('/settings/ollama_models', methods=['GET'])
def ollama_models():
    config = UserConfig.query.first()
    if not config or not config.ollama_address:
        return jsonify({"status": "error", "message": "Ollama address not configured."}), 400
    try:
        url = config.ollama_address.rstrip('/') + '/api/tags'
        response = httpx.get(url, timeout=5)
        response.raise_for_status()
        json_data = response.json()
        models = json_data.get("models", [])
        model_names = []
        for model in models:
            if model.get("name"):
                model_names.append(model["name"])
            else:
                model_names.append(str(model))
        return jsonify({"status": "success", "models": model_names})
    except Exception as e:
        return jsonify({"status": "error", "message": f"Failed to fetch models from Ollama: {str(e)}"}), 500

@settings_bp.route('/device/<int:device_index>/update_metrics', methods=['POST'])
def update_metrics(device_index):
    data = request.get_json()
    all_devices = Device.query.order_by(Device.id).all()
    if 0 <= device_index < len(all_devices):
        device = all_devices[device_index]
        
        # Save metrics to DeviceMetrics table
        new_metric = DeviceMetrics(
            device_id=device.id,
            cpu=data.get("cpu"),
            memory=data.get("memory"),
            disk=data.get("disk")
        )
        db.session.add(new_metric)
        db.session.commit()
        
        # Update device status
        device.cpu_usage = str(data.get("cpu", device.cpu_usage))
        device.mem_usage = str(data.get("memory", device.mem_usage))
        device.disk_usage = str(data.get("disk", device.disk_usage))
        device.online = True
        db.session.commit()
        
        return jsonify({"status": "ok"})
    return jsonify({"status": "error", "message": "Device not found"}), 404

@settings_bp.route('/devices/metrics', methods=['GET'])
def devices_metrics():
    devices = Device.query.all()
    data = []
    for idx, device in enumerate(devices):
        # Get the latest metrics from DeviceMetrics table
        latest_metric = DeviceMetrics.query.filter_by(device_id=device.id).order_by(DeviceMetrics.timestamp.desc()).first()
        
        if latest_metric and (datetime.utcnow() - latest_metric.timestamp).total_seconds() < 30:
            # If we have recent metrics (less than 30 seconds old), use them and mark device as online
            device.online = True
            device.cpu_usage = str(latest_metric.cpu)
            device.mem_usage = str(latest_metric.memory)
            device.disk_usage = str(latest_metric.disk)
            db.session.commit()
            
            data.append({
                "index": idx,
                "cpu": device.cpu_usage,
                "mem": device.mem_usage,
                "disk": device.disk_usage,
                "online": True
            })
        else:
            # If no recent metrics, use stored values but mark as offline if too old
            if latest_metric and (datetime.utcnow() - latest_metric.timestamp).total_seconds() > 300:
                device.online = False
                db.session.commit()
                
            data.append({
                "index": idx,
                "cpu": device.cpu_usage,
                "mem": device.mem_usage,
                "disk": device.disk_usage,
                "online": device.online
            })
    
    return jsonify({"status": "success", "devices": data})
```


## static/send-icon-old.png

```png
�PNG

   IHDR   `   `   �w8   	pHYs     ��  qIDATx��][�E�E�(�h����jLLDQ���aWF��h�;�n�_��HD�S�b�7��a٭�@�$1�c��2�v���������[����[��c
(P�@�
(P�@0����׹�]B���PX���lpD(�
^���Ol�pY~�K�KZ)"yRv������
�	�W�^�eH��%�!�D����]����%E�|>ֱ�)�q*գ7�7�6;�W����ժ|:d��|.�7.a)�ִ�=�YNѹ�g��ʏ��B�&���~O���M|2c�����P��}���U/v���]�Ϫ���*���4#�P]��נ �������^�f7s�%�����4��9�P�,23�%m4��ky��܆�T��'
�4~�	Z�> S�r�	=�����s���T�R�����GɉV�N@��:e���׳��}Kƙ
�p�6�Q腚M>������d]��*k�
-��G/����A��ñ��K�����I�E�,�X�(bp		ۂ�"��Au̒�=�ĥ5�e�=\�G�����:.;BXw��`N��[���\�Ű�~W3C�P�Af�,	�f׵sG�x������ӹ;��XZ\�1�N������XJ@%D�℄_�o(�y��5X��kܫD�8n*x)�z���9���ˎ?M�ɯ��ϓ(u�X|�!�w��C��b�������;^�%:�f�nk4_*�����������41��w�]��iVź�ټ�" ��s��4#�M�JF��v�}�;��=���eSM@�
���P��\����c �R;4���G�TJ � �u������k��_טH*Wf�c޾y7�z��w�d_��a��mV���1�b=!�$�#�%�����=��;-���(Ck�ZL32O���o�)�����d�(wب.#*��/�7hn����,���l� �d� 9��咀፰��%�k��s�3����g*x֎�l�VB;p�p�6N�_EJ�L(Xe(|�YG`-�{C(��K���S�S(XK��6>�c^Pem�H�\���č\�Kn����z���H@PԊG߅=؇x<h*\����,,�@����#��P��F���V'@����AI��M����U���2��+�� ��Zf�$U�&~����FB	
V%����VK *�)P�3(o%����IU�K�+t�5Kp�����/�=Y(���_�\�_Ի{�'���r	��e����&���tF��	�@·��"
0M����'�S�<��nEB?�$�[�d]l^����[ 4��^�U�&Kf�?QG�$H<(7D��Lx"��F4�'�
>�[@���Ƀm������-Ix��4Cu�T�tCD�A	\§���]�G��f��-s~/�K<���+'��tM��
�A�t#�K�����ٻ��Bp�>իE�p2��V�F}[F��^�8PK[��=
/�\`��f�Ac��%��J��
�)��ݚ0/�z��t}�LP=�Zഘ���tI�7�F}�e��d��pXCDt���������am��i*�w��,o1�ub\��,o0bHE8���T>}!�%~�gC�;Y���5ax/fw�3��Ҩ�Fp�Ӽ�kc�+�,��^�	�	��=nS[��p �KI"��pQZ��*�g� �30�Eҵ�0�V��T�&�$ueGq�H�a*�V��鉠]=�M��m�K\�ZUjO���)�OZ��p��W"V���N�{�G�ʷ�6�r��fN�.��ۯv�\f��QV���L������q�05�7���N��Q��äH�X�j�}������8�\M[��B��ܦ�(P� ��1���`�+    IEND�B`�
```


## static/send-icon.png

```png
�PNG

   IHDR   `   `   �w8   	pHYs     ��  �IDATx��?hTAƟh�XX�Z�hc)(7s$�� o���mJ[K[K۔i��0;���Iaac}��"D�pg�E����|?x���ۛ��{w]          �!d�~b��A(����+�'6}�>�/x�#,�/J�GI^Pzp�{m!X���tklr�{�!8R�黱���;�ސ�6�>����g�����'Bf������E���,`q�W6٤t��}-`>����%Y�����q`C�<�ؕR Į�x^�ε >zN@�<��x��y��OD`�������[����b�0/{����=X;�xj덝w�]켃��b�G;��8��y�����?$�-v��pt������{\�F;�����/S�o�i��k�<�qx�锒~�$��/B��HEQ�b�RQT�صX�$vm%�]���d��T oˍ�uF� 2�BI^N��W��lS�pM���fwb�����-�%�X�`��G�9��Z7$�7���y����7n�~x�%��^���,Nm�v)_�~V�85|*H���������w���8妾�?O-N�)~�[a�d�M�cf�Xq�MY�]�����qʍ�٭N�r��^q��p�]���0f���_onV~�┛����)7��Lq������┛>�8�愣┛%V�Ӫ��n�8����3�t���n�՟�^           ]�|gi�:�.�    IEND�B`�
```


## static/settings-wheel.png

```png
�PNG

   IHDR         �x��   sBIT|d�   	pHYs  �  ��+   tEXtSoftware www.inkscape.org��<    IDATx���w�U����I/�$��{����ذ �"�rE/W�^�]���k�],�*��$������}~�'�9{f�g����<�'q�63{�5����������������������������������������������������������������������������������������������������������������YZ=��%0x0����e�jh2q,��cV>w ��& { ;�� ���K������ s�Ȩek
�_'�ll���.�n%���� *ɩ�Yˍ^�x
X^ � ��$�������u_�z
8x5qm��Y�v�I�b/��^ك�G�(�5�����_�)����7�]**��YcM��%�{�X��x�`Ͱ�k�m������^A����8�9�~�%�b�w�V���P�UQ��G�,헸�ff�5	x7p;���`q�E�r[[W��~����׺�Y�m
|�t��ˈ���*�J�"b٧��YQ�� l����rv 1{��,�b)�Z���n�sM-!�&읤&��228��÷�xo�bey?��۸x%0��Z13� � �@��-#<��w��.ʈ��{��,��jk5����2c)pd��d�A����2c�qbKb3�ژ�H쟮~���'��Mk;b���zH�⎀���Z��j6�g1�VB�Yw��BT�X���33+���'�MO�ʪ�br�Ukp)���:>�w43��Ą�Y���8�����Fg�owe<�w>ͬb#�%Km~J��S/�V���K��_{fV�7���_'��,��DHu;�Wv_�ff+�p1�]��}��K
#��o�����]ֱ��3L#�\�<J���|�`�F��o׺����:�T���hb�_·�����eX8}{�1� ���U3���{�2�v`�aֽ=mS���X��p�0���ZfSbI���դx8t8�` <��;u�5%z�9S��f�|c�� �?��ˀ��ZC18	�9I�o�ע�{������FCk�V��;�vjC\�:�f1��Y�*q���aԦ��i��&=�[h��>9�R�0a�MdfMq8p�P��<A`+�\����xΪ���m
�bB������)�hL�L N�[���L�����xVu�q?��{6�����������̬VV~����Xu�K37n��}=;Vˀ/�mI3����/��1�x�_��Z�~&' ��׫cxq>`ȬvF��3��s��\kw��>��w,%�e�Wf5�-p5���ܸ�ob��\m��}}9ʍ�0�,� �O�X�������%E��عq��F��^��^R���ה:/�xx1f������? ��������E�-e�%΀� p��3�5�=�?�I�	a����<���׀��M���ձX���3`����B�����u�p��~,�S�u�O'f��щ���Y�XJ|y/�����j�Q�a1S����'6+ڢ���U ˉU��M �&�n�Se�2rpq}���[h�>�����>�A���14;��>|��$�Y�& ?A� �:"�̇{�i����G|��[�X���Ͷ�K�(jf�m\����2n#֒٘��xP����;~L��r��7gP�*�2`��ff�p͟x�?.".*���H���ˑg��b_��� ^B,�S���x��|��Jv1iK}�W��n��H�����+N#ݒ���33(c����df%�*���8�j"�JP>G��s�9�� �o��)⛴�p+�����fNg{�UaC4�F	yw�;�Fy��CՖ��E����̬uރ�&N�C���׃CE� �?G�~`��*̬-������x9�l��.�[׎�����#�{B]/��	b�J3�uh�z����r<����=u9��lb;�܌�I3���Ė�/,����,�7l�1�8�fB����6����ˑ&n�&o�Gy7�P��s˫*��9
��Zf�ذ�Jk
p�zs������m��8|G]we��Ĺf6�x�>�7iqq_�>��4!�[􎤞�'�%u=�7ˬ$�&8��Y4fwʑ�{��:���N��C4c��(➚��Nˊo�ZCf57��$��1��^�G�ZeW��T�����1�8��]� �A_�eD/pH��cV_Ǡ�)����C˯�l��ƚ���1x+�,1M�p�1Y�f�[�Pϝ��_$���e�8}�;�?���`�Y`	�z/�,�b��f�w��u��)*�^<���Ǯ�ŚkO�}����O��z�q����[�^)��%�d�g����n,1P�k���W�Y}���&J�욨�j�Cu۴-��4��i�gעo���%)*ì.B�,z�/�mpG����s�[���M�hp2��}�F�$6A�,�NU��	܁�ݚ���뺡O�$N�S��P��T�Y�G�(~����5
x#�<R݆u���7Ќ����,��m8��,Q�e�x�7����'�a$�j�b�mZ���82ڿ���!��^��MW�i�rb���$�7_���>i��iw��ăN�ƹ�"� �ݺ�c[���{[��+��$+�Y�����?VK[��[xp=���%�#���Z�^m�& �G��+
o
d���7�b�T�֮�׀��_U�C�)�.�kц�M��J�+)m��o�����Mg$1��%�=q�Nb�����Ԝuk�;z���%6��W��tK����҆��Ĳ�s�	��s�1��$�D`�Rk�ʰ6p��/����f� �ￒ�Һ5���b�{�����EtZ>J��o�v�u0
�*�kg9pu�� M>2�.��@��+���y��ۘ�?�+��zK`sbO�*,�&6<����U�����w�.Ż�x5aq@o?��{K�����(�-�M�u���"�z�&u��Ľ�����6���ģ�/���t��Alm�r�>��-L߬rk�z[|$uA�,[��:�3���Ki����zs��ԙ �Tg��d^�~D�aq���@.Pg�x�<M�	3�8Z�b	�U��<��� ���:fV�͈ j��3�6� ��l�	u&��3`f�{1���k�0S�)�����cW���O�=�&/��� ��� S��ՙ0�ʌTg�(�@>�F��΀�Uf_�8�w���; ��%��U�G�3��a� s��ՙh#w ��ku��Y��0��H�	�܎~B��Kifj�yzhe<7uAmp�ϟ��r�����xqf ���Z� ��� ۫3`f��p����R2� ��r�U����Yz9t �Rg�����"�XL%� �5�.�����)w �)��;�5�`Kq��g�K���+��%�7�f�����鷞; y�N�`#u�,�m� .Qg�����L�q��of�l-N!p�8��@����̀̚kq�7���r� ��.q�0k.u�Fq��; 9�.N_��0�t���8}���M�?E������8�[��� ��q����7�4F ��y�W���@��?I�����,�$@����� �l�8}w ̚I}o?<)΃�@�� �0k����?*N�:�ȗ�D�1���,�����ӷw �<'{�0m3KG�
`�8}�p o˄i�`�L�C��ӷw lE�0k�q�ʑM���|�A�T� �fR�ӷw 򥞩�T�����/`�șz��g�5��0N��u���5��`�L����=F��@�6��:f�4O��zt�:�ȗ��.� �5�z��5��[�; ���������C��p g[�ӟ!N���Pw �k��`�������/N���x
X$�����w r5�R��{��Y:��W?�w r�0J�����Y:��[=�i���}��# fM�����}��?<*΃���� �$N�p W������,-�$߭��v�z� �gS`3q�0k�;��� {���z� ��� 7�3`fI����a�S�����3 ܠ΀�%�00K��C��ee�I�ra,&�.������f)� ����/��5�|q�,��������˫� .Ug��*�Ou�#�0��:��rHn91
afͷ��͓�����y oA��/�e��Y%�D&�j���y0�E�̥�	�Y�~���sF�Rڠ<��#�ՙ �Rg��*�Ü��3�F� ���t�Y�3����ޤ΄��!�����щ�jfy�f��<��?����G��8�X�`f��8G�	`=��L�U)�_����g�V�D��Y��(��	\���[�#��Y��A{�,�J\V�,����?O\V3��_�?��K�s��,�Ձ��l}����5�̽	�s�/ޚ��fR�F���L��6k�5���Gˉ>��� ��M�_JZZ3����?���3��j�P��D�:���[{�[��%v�ܠ��F�ހx�4�8jz1�s40��u=X���9��~�(��NL�GF7ѫ�Sՙ�X삷'O��j�x�:���⨛�����rg`'`����ppc'�'�*·�g,p?0U���K���Z;����8"i��}��}�x >���YQ,�|�Xεn�������:�'�-�Yz[s��L����F�F�'���_+E�.�;������-�e诛�X l���f	�܊�F�NYh�q�ˁߐ�/��188��`y:���?�#^O���(򻙖{�OXn[�Q��?&�ѡ*b!��5x�Wn�D}�o'-�Y�@��IYh[�g�f��r�����]ԫ�gy�Z�9e����q�7�`�0!]�m#�!���o���jb'8��z�ka`,�JXf�R��fYQ�/a��V'f1߃���� '��Z�F��u;�*a��
y'y���?��f5����o��S�+���V�\��A>{��'��9VoHVr�D\m�ԗ:�	="P���uhY��
�$�L�{W�o���?��TV>�F��M���/S��2V�[ѷ���
�	0�q�/����SU@���<��}����|�eZ#�-������wLdm�R�7���T�b��*�m{�����b^���W��m��b{��,ف1�4Q��d�`)��u<���R:}�,�!N65K���J�E?���y1p�6u�oç����?�f�� ���s�_�C���Ŀ2L���=C���j����mWK�	�V��-�ky�B`���2��'���1��3H{Z�V�>#`� 6�2+d��/���GRTF��"�s��hT������#ǃ�Vw{��k�)������`q�|Ǻ��'�����x�X�~��M�K�O��m�V#vs{���M�6.�V��(�I���(7�/��0���M�w��������֗�]1-�|�|�qp�^b�N���"�w��?�/�:����N3~�}��i���Oз���8�8�������&�'��Zk"�:br_S&y]��P��Z�����Qm\l�1�}[v������j�+�Y�O���ʌx�s7v ��o?�&�㥲E�D��V�e�x��v�2�د������/�� أ�:k�]��|G񘍗����XF,N��w���qDO�X���_hޯ�ű%�_��N���m��#��[�u
�v,3�>	lSf%�N���p���Ľ-�M:�֡��CW6t�g�]�왞"VМ��HM�&�T=G��n&�g۫/�f�=��8��S�
s�o�׶:����t�����nM�Dߎ)c.p���s���V��w1��7x�q*>�g����@��|��+�*[Ҿ�5��'���^���8�}���t[�-�����QM��k�؛x��nGe<�xކx�6&��Ս�{�o^1\�7��������"^J�+W�c1��$�Y�M �a��_�_l�r.ћ\��H��\JLm��������}���w���B�u�d� ����i�k���̨�t`�i�w�#����2�W����ڬ��}��G<�4\��ѷ]�x8�����2*o&��> �N���&�$�%�Λh�n�e�Bb�ak���נo�:�o�%66t#�?�o���8����N��_�=�.��Kh���Wlo�~U7���&�=�+�q
>���B�ve�2b��[��K����	���4���멍ށG��}�
K&ԕ�{,N貎��4�As?���F��P56'6��}=����eVP��v�(�H���LnB_ѹ�|b���4c�ߋ�ӌџ��+�w��z-sh߄Ҳ=/�]Q����jA�������^b�z����K�<���+&#�����MxR`Q{Ҿ͂�ߥ�#'�����bٚu��۰��;�:�n_bOuu�w�*�JZgb��-s�����<�8fQ]���2�=o#�~*r4�v�&�^��>ꠇx5P׽�_R~�����Vu[��Ğ/�7�z��S���]׮A@R���y�ҹq	�n� &�F��e8��=��L#��P��B)�q`������Wd�q:y/骋_�o��ĥ��=�v�o㦳��D;EL�T�iNq^�MlC����b.�1�w$��j,$��4af*#���tj�k��D;mDl(�nӜ�U�jT��+/�8�z�������t(q�c�jh���;з�P�<y�L���Ҝ�<�����B5*r���%�'&<Yy~��]�g{`���}�%~�������T�xw������+-�X|�8����\�4����ָ5���O=�88Q��Db;����O�΃��JSǙ�<�F��AR����z9�����He_�5����Q�e�uy�".�/^����,�여���7�/�|G���X��n#\�'���,E\���6�}[�(f;%+�mG��?���j�����XLM��}eU /��z����{Eq^�_�́��{E��tE�~� F[��]eSJ�%�5��:��#��jL%ߍB�NWt`K�A��>X<����6���R���J�������RŃĶ���V[6T�v,���
�k���?X����6�#��o��qci��У�+��X
�	x)0��z��[��e���<b��i�A�����{MB��j��e���l����ʈ�Ļ�w��|�51���I�zG��_OYh[���6hڪ��ʬ��=}u��Ky��]1ֵ	���C)m��1���������� W��&ʈו[=�z?�
j���v�GM�[n�w��V��x�GNz�S�_�#)mö-p2y�"YU|��Z)�O�WЊ�b3�����-o#�;�_;��_�d=nˬ܆����9A���p;��d8��Q�k�W��o�w��/6NYhK�U诧���ꗳ=�S��I�xK�[�'v<�<'���;�Aa����rޓ��V����p��Ĵŵ�Y�Ui�k%@�+�;��k�,%�Q���W��Ҫ�5��\�{����诗��s�[J#�gѫ�/�Ǐ�,��c�W���5ǧ�_O}1o�['���_7}�ŵ
MC=���(냈�Ni�S����Z��ib2��íį�\��X�j��0q�Ri�̗�ش��������<�3�q�u&l�>ܥ�D�db�5ÿ��g�ؤ���; ��zu�y?1���eyM
~�:Vu`sq��R�����hX�|���[�Zz�L\���h��^K���(k`4�	xh���|P�+�c�t�ï�B=��zTV`C�K��h��3�q>�����y�Lt�΀�B� �H���F;$27}�#����.L\V���诧�ī�1��j�B�̴�Je� l\��t�>q�V����$�ˈw��W ��3A���_�	+l)p�8�� ��ؤ���ֽ���G�3�q�:V���3Б�5nŨt��L� �I��g Vg�J�;�m՞�΀��~q��G�V�Z���L ?��3a�[J���=��:V��;'��zF�z8Ɗ{��$�D���'D+� ���`Ź��Ԓ>�[~P9��?_KMv?q���; ��� �Ot&<��    IDAT)�0�X���v���� �[u,�S���&Pw F��Β�]��б�Clo������DKk*�������d�5�5�جh!�X���(b&��Uw[�[*]F�i�6�X��1�d@����.��[F�Y%|F��ӷ��Rg 8]����� ;�3`�=(N?��Z%|F��ӷ�r� h�C[�΀���E@=�8}+n;q���[�y��\�������+ �# �ي�T��%�ׇ[u��*y@���{�� �G ��0V�H�I]"Nߪ�q�� �y�b��)E?�	 �0��!��U)���Z׉��A������b@�P��b6Wg �I��ܵ��?CŊQ�d�_�g�n+��f�Dߓ����A=�ŊQ?7V/�et Ɣ�E�Po��ӿU����(Nq�V���gl�(�P8-�oŬ#N����K}��z��8��?��0�X��Sx����雎�@�j+f�8���0�@��� �������� x w �N}��#��MG��+�zS������+�Y��MG���$Nߊ� ~`�x�T���~vZ1� ������7u��H�K�P?;��w�; ���%N�8}�Qw�<Po(#�Po�/`u���p����,6�)�3�X&NߊQ��7������
_?et ���´��6�/p�++���O�%%|F���?���b���>_���~�]��G LMف�N��3Q��G �M���0m+���z��>O� w �M��b@���m�8}w �k�8}��7u � ��"�;�M�o:��x<Po�@# �u�� ԟ���8}��X��z�ˊQw ���� �F���>�wq����8}�i�V�8q�Y� (wC�A��ǊQ� l.N�t��������,� <Q�g��Y1��_��v��=Po�%�O��2: sJ��"��7�+�M���V�5��yp�ަ��/�㩌�z"�; �v�8�`7q�z;��?�@��8<Z��00^��s��$��o��K�~/��8V�z���� S[�~`q�V�}��?��2�;w �w <P�����p�Uk?q�׋ӷ���Ǌ~@^���Xq��:���<Xu�G�����8�$@� �o+N� 8B��Lm�@��#N?��z@}��w�:������ ~��
ӞG&[�7rq��nD2���Zj�)��<�������5R)߻et ,�3��+��[\)��H�e�<Xz/C�˵� ���F<3T�� �B����f�X����XrG�3 \�΀��8��e|H�^�>�[�m�K� �Ǉ5�:���L Ug�
[O���2>�� �̒>���m����{�N�K����J��c�ˊQN ���)����R�V�y�u�L����,��M�L]��W+�# �('�Oy�0��� 6^�΄���q����X)6�?��i�+���Z���<Tg������Ҧ� 4�´��	��7o$2��]��*0�Xޢ���⠴E�
��zZ�F%.�U�!t�Qi?��P�k�^�a�X���D�����|T���s�Ot��&��hzYTV�Β>�[� 4��t��Xh��y���+u���(qɲ: ��J��n���+r�y|LpݝLm88W�	+�z���: ���%}V7^|x-�������*��>�Q�LX�^E� ���G��G=���i���'�􏇉�>K���Q��x��/�D�w��$`��/I[\����^K;�/��}�M��x �=p<�U�j��F;�v`|8mq-�/��n�b&�\��F;�I�NG�Ri)0.i����7�pc:�]����׈q2��/ۥ-��hG�2�u�_M[\+Ѧ�v���'^ۨ���q[��s��)��ݫ��J��-��%��/.ǿ��`����z�^�y��z�[�S�b�����}�j(n#��SV<�x)޸C�诅��޴ŵ|�u�?<�?/#�=�I��$�C�O�_%�A|q�+��x���˫*�cз�X 운�V���i{���0i�m��'&�ߋ��(ǔ]1e�}��e�Y�19�+n���ju���;��)m]Y�ؑT}}�;�AeJӀ��E-�;�ZC%�:�
�"n �ï��ې�r��"w��C~ː�'�,���n�$���H/&�%�G���*�N���/UKcu�q�m=0|V@>>��zO���*�%��}ۧ��J��D&��;�*�~�mx�x*'�oぱxy�Bې��h��00�������!�����I?(�Β�m�v�qpx�*��$~Q��w`, ���ܶr{4�����$�v�rL$V�<�������Rs��}E��|b3+��з�`1�)a�mp�_���,>��ح7��"�i�Ъb��/�	��W��R�bb7�	Ū�:��ѷ�`q���dK�۱�,���Sy>�\�Ɗ�����̧�WX.q�tЊ���\Q� �NWt�ؒ��[��+��IV��ڎ�PIݶ��X�Z��T�Lwu���&�*[�&��F�W?)�N~�B��_���Srۏ��XF�[+E_q�Ńx�`QǢoǕ�#��J�^���;����$��؏8�^ݦ9�
֥�x�n���[�_�ԏ_p1�v\Y, ^��Z��D���ueq%^\�	�7�si�*�.T�B���\Q܌����'�_W��(�w�^��O�m��%Q��.����3�8�P�f�+�+1ט����m��o��ĩ��Du�dkF�~C��&����އ�����6�5c��Wf��K�����Y�|���;�Q��P���f�U[w��Cߎ9�g�kV�w��М�|�J`��D�nC�%Čfw�V��8@�Nۉ��&�a7bu��Ӱ3f6�����آ��m���o��ą��Ij��v.B�>ÉZ���ı4﴾��I2?��[[������2��L�:���S���G�L��k�g��n�����r�e�K����F,�QWt�18��
n�É��vn��8A}�������ˈ��lx��&���9���ڭ���w�Wvαx}��B�B�f�ƥ�A��H���ާ�~��*i�5�Kз]����k������:5�o�v�ep-�6+�ƛ~���Y��H\�.�bnCb�u��K���vkn
�m���!rw�nc`�6+W�ь/��P�L��Z4�$#��ۊ�,�T�������
�%�ˀ7v_��r(ѣV�Y1���n�#�4���w�Rk�����s<��+�M��݆ڀ8�u:��)���Z[���۫춿x'��F���sh^G�}%�Sl���(���{��T����x��rb��O\�b^u{��%���E����&vs��R���J��v،�-ͭ"����j�U��JF�%���M:�1���-��=I��F���M$Nl�!-�� �!g%Jk2�a���đ����������3R��Z6�#'�{ܩ��3թ�2#���g��݁Q�L%��/p�:#��Ft�6q��F���?�K��}�I�O|��%�O�D�+�gub1�`�NlKt���.bUƣ�����N�;�3���߉��ӉN@v���jDG�P�eİT��FtWg$s�돳v�Zx8��:#51�X�y�:#��_�7�p� .��{��N��췣TV��(��j+b�eu{9����D���ۭ�����h�<bt����!F~H�OW_PE��VOc�<���y�|�p}}�������SV�YV'����^�eWLC
,D�f�<�	��c
�ќ}7��>b���4�5�u��@}O*�Cs۪DL~S��#��O��c(���$uە��?$�U�K�����<���n\�ɿ�ڞ��nu�9�9Ą?�	Ą7u��;�wK\̀��?I���?��2j[�S�#:�^�6|?F�vE��U�7Ҳ�mI̴W_�C�e�Ij��6&�S��͡�[����:}�u���͆�Xbi���J�CLn���F�P���ڸ�8���g}�1��7ܘ	���ng�m@r����?LTM5�X�n7G5q:��:�L=ę��N,�	���>�ez�S�r?��h��ۈ-r���Hw_����v�з�p�`�$5a���_)p06U4؁x��&�,|�fQ���z����͒X��ž��@��7ۆ�߶��Ǖ�t��������MTf�0���~E��;�*�o�&����.z�/�X�^��-���z�b#�����Wg�+z+���c�^2H{��F=�J�2��oB�C�(�KX�6�L�ST��ch��Y�������se�x{�қ�I�o���┅n�����6ǃ��+l=���׈�v]Q�^���f]�,�c�xa�B����>�з�#��6� �l9�1�']�ͺ�|�20n F$,w��\��]�W�r�\;�o'w>��26
8��20�IY��^	LG߶m�����ܩM#�]O���}�&G��o��qM���D�S�;Iu7=f�Ǜ\��?�v,�5�V#;��o��qX������'зs�b>��˳��;}{�^�P6�Z9�X������/i�k�d���i��,b)ں�j�ֳ�s���,�YJG��]����L�x}��-�����W���~`�,i��\��Fꋟ�-�0��,��Y�K\�
�ا�9y�V.'�-��f�����4���)i�k+��z�^��A.18�Xzf:_F-�9��IKlV����T}q|��ʍ&Y9�vN|��h�`��/��oΊ'�Y�� �W�'.��Xb������3�]_I2c�8���?~���f���{���Նo�����b��I����q���XOV�K�_/}1�������o�.�6�&�p��#,�>AҲf���������N[\����Tb��$q>&N�Z&·�(``[`;`��?oI�6��"b�㻁[��:qWE�[�� �O�����3�� �|T�	�@�"u&��Q�f�Ds�΀�����y�Ļ���/Otbn�?� �_�}�@��[�� ����-&:�w�3b��$��&���5�l���_K\V��|�M7/�2k���-'F��%.�YV��C��lm��A� P?�N\V�,���H�R�Yn�@��YN�5Y#qYm���[�t�8}3����t|��lj�JC�_��3�ͬ.F��>��Z��oD��5k���c�I�L*�W y8�ء��ͬ:��?�}W��6s �?��8}3�΁��H^��D�����������̪������ m��q�υ�>y)�Lm<1�W���%��[�# �XH�P���fͷ+��"W����z� �����w�of���� �;u��ܨG v�of������:f��!�^U��{ }�L�����O_D
� �e9p�0�u�fM6�Z�����[�; ��F�v��0}3Kk`�8�ӷw ��Oq�[��7�t�Sg�X`p ?7����Ysm)N.p�8��@~���_G������8�k�M�,� �g)�	P�*L���Z_������w �t�0���i�YZ����F��Ӄ´=`�\�W ��ӷ~��ӣ´=`�\�9>w�ӷ~�ȓ���0m3Kgq��C���w ���0��´�,u��)`�8֏; yZ$L� �f�$N�q�6�; yZ&L{0R������ <!N�p Oʍ� Ɖ�7��_,�o������=`�<���?%N�p O�#y�s�,������ � �I��� �&R�,�G l w �� ,�mf�G ƈӷ��Ӛ´�0k&w ��ȓ����5���^]���@��Gv*�!6�t��_K���@��# ޭˬ���5�wNV��l&L�aa�f�Γhw�G��@~6E��O�2k���\q6�o����m����of�(O�B����@~v�?C����s�8}w 2�@~���8}3K�>q��N������� �!N���Qw v�o������Β���ofi�+N`�8��@^����fSw z�]�y�w �r�8����YZ��3 �΀w �1x�8��ofi�D��W=�i� �� `5q��of��$N/�G�[�; �x�8����<�Yz7���G��@����z�ǅ�Yz� ���0w r�"�ge{�߬�Tg x	q�	���7�3 \�΀�U�&�8X=��Lns����XLM]P3��Yh�9ˁ�'/���G �G�ע_dfչD�b�Ӛ�L������& oPg8O�3�������Ug�L�-����.��ee��g�թj�������Y�������:uAmp~��y�y&^�o�Fg�3�q�:m��N.a�F�3��Ы�~i-t��7���ە�C��j��;1i�vWg���`�N� S:�~ʀ�7��e���$>J,���<g {��0��FZ9w 4v"���X��������z�:�4`����$q���������~�U������N�P9w 4r��z?p�:ֵI�(����;�v^ϤNl�;��A<į���7���tbS���8�Vr@csu�[ [=L!&KF|�n�����mЉ#���������A~~����8}��|�������KiE� ��O.'�L=Y�h��_ ����r�&1Oym����f��9ڛ���E�.�_�_@���:f �D~�m�1�Cy-�'�Yr�G{�����6������RV�L���1?�M^AL�T���#Yk�������`�� �$b�E� �)����u�ڪ�?�G���o��,?A�-'��|g�6!~��B����l�x�X�&� �۷|?i��2�y�7\_�왶�F����}��-?�v��@[��oӁ��e6��	�o��� xS����ĩk���9z���;���1�q�`q\�r�e��o���{�Cal���M�ˈ�49�Q��'u��,vKVz�̬���[Q�L�*g�Y�6�}[6=��Թn�z�m��X �NUf9�����������x�=fK��s���;�>����xa�
0����x����Cdl�'��U�W��z`�U�UL�ϗ�x�j0����o����Ĝ�w�������t�?�ߏ8n�}���0��*IM�el"������<[x�8��q G����g�gB�Ό�,Ee�����߀É�����D}L%����±���Ka�d;�,�뻛xs��0����߀�<T����>rw8� �6p/�&Yj���v������q�׊Y�����M�<?A}�h,p
޳�α�X)�;�D��H�w�bV3�S�/���	W�Q�����5�~%0���R�n����ߠ�!��Lb�q����71n%�`�������2⨒�Ƭ�6$63QߔE�"��O�ۉCh�u�Hs�C��5�u�u)��+�z����o�2b	�e�;Ip�5���H����D�F_We��ZeV�YS��ZV<��z��}�9��O��u�/p��)3���f�\��F-3. �;�n"���rh�'Ĳ:����2�����K�'�FZ��L����ؤ�j*�z���ɡ�SѝJ7�00oy�k�WyUe�l���iˎ���IM�X����H���Vep0��2���{ʪ,��X���"�G�sWZ�}}8򊳨f��K��*�-���Zfp&�9UL'�W��C�Wׁ#�8�t�����1e<�'��68��M
�wSmG`pK�29�?���;��r��K��53+����o�1�د=嫁	���Q��:��B����N|_|�j�PX��}��ۚ����:#�M>M,�ZR��"��}a����G�+Kn'�m~x��ӻ������|,1Ys*�knm`�NlELTQU!�>|����A��o]v,b�n�{ꌘ5�$�4�=�*�>b���Rj.~��˔2��ۼ���Ne��?��8KT�*����j�� .� �U��}�Yez�K���U�\�K����C�#E�ک��=]m�+q>��2�uSf<��7�K,�H7�+  �IDAT6��V+�̬b�E��*� �b�;��f���,��a�C���.�9u?�z������V���2���Faf2��-l/ ^ɪWlE� ��[4���B���l	|�����Ǣq�k���	}3�S�1�v�k0���į�%����g�M��������;��)_݌!���}����A!�њYf����e���KyzX�Χ+>|�8��iF GW��g��b	�ڳ�ì���#mg����&�"F4Vذ���c�=�%�'6��x�,L��p-�NL�h��l�Q�;�3"���xf���Qaf5�p���c�q���a�L~FsV�9!Fg̬�F��8�K�/���%6�R�O[�t<�ϬQv�B�pqD�O�
g��L�wn�-��ͬ�F���wh�b�}[�c�y�۬�q����5�N�?�?t�ˈ�r붑��N���ۯ�1����kҬEFƣUœ��w��b�u;6)N�������E�FM��o]�~��=�w'�����C�NM�{�-���j'⥂���`��W��5���ўc�SǍx�5�p'`8q���l�v.G�Ъs��5܊�aq'`���v��������,������zïr���o�c�E�y���UdM����n�A`�jں�Y��S�lY�F���ي8a�C�+���=��`�Z�#����[��~3Kh?bB��a�[�+)Lc4�7�ׁ"&��,\�ff������/��x�
�RL�]��%���K�;3�aMl#:��Pǿ�r�3�*F}M��E�AI>����&G�p�:�)^�V�7��.R}��01�M&�x�ò���_J�Yپ���(+�?6-�����|���!�����7
�#�k�H,~lWrݘ�%7�ح�A�Ӳ���Z�� F�7?%�ޚ���D�4�#p#�y]�ߡ�f�s���3k����k�?l����ˮKj�����a��x�^3k���w���?����~��H�z�����63�Ֆ�Y��?�WĿК`=�讣e��È���Zo
p"��䶔X���u�<�'�]G���mVE����h$�"b�������ׄz�������-sB5E23k�u�_�wR��\�=�Zrk���P��3��{UZ3��!&�XH�/���N����n[ o �\<Ċ��Y�?����V�]3�vX��\�2�YA���,���2E���$Nܬ��a�Ֆ'6Y�M�&�[�g�K��$��^\���wff���5ͺ���yk��� ��G����e�33333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333k���jb����a    IEND�B`�
```


## static/style.css

```css

* {
    box-sizing: border-box;
  }
  body {
    margin: 0;
    font-family: 'Helvetica Neue', Arial, sans-serif;
    background: #f0f2f5;
    color: #333;
    line-height: 1.6;
  }
  .container {
    max-width: 1200px;
    margin: 20px auto;
    padding: 0 20px;
  }
  .card {
    background: #fff;
    border-radius: 8px;
    padding: 20px;
    margin-bottom: 20px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
  }
  
  
  .navbar {
    background: #2c3e50;
    color: #ecf0f1;
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 10px 20px;
    position: relative;
  }
  .navbar .logo {
    font-size: 1.8em;
    text-decoration: none;
    color: #ecf0f1;
  }
  .navbar .nav-links {
    display: flex;
    gap: 15px;
  }
  .navbar .nav-links a {
    text-decoration: none;
    color: #bdc3c7;
    padding: 8px 12px;
    transition: background 0.3s, color 0.3s;
    border-radius: 4px;
  }
  .navbar .nav-links a:hover,
  .navbar .nav-links a.active {
    background: #34495e;
    color: #ecf0f1;
  }
  
  
  .nav-toggle {
    display: none;
  }
  .nav-toggle-label {
    display: none;
    cursor: pointer;
  }
  .nav-toggle-label span,
  .nav-toggle-label span::before,
  .nav-toggle-label span::after {
    display: block;
    background: #ecf0f1;
    height: 3px;
    width: 25px;
    border-radius: 3px;
    position: relative;
    transition: all 0.3s ease;
  }
  .nav-toggle-label span::before,
  .nav-toggle-label span::after {
    content: '';
    position: absolute;
  }
  .nav-toggle-label span::before {
    top: -8px;
  }
  .nav-toggle-label span::after {
    top: 8px;
  }
  @media (max-width: 768px) {
    .nav-links {
      position: absolute;
      top: 100%;
      right: 0;
      background: #2c3e50;
      flex-direction: column;
      width: 200px;
      transform: translateY(-200%);
      transition: transform 0.3s ease;
    }
    .nav-links a {
      padding: 15px;
      border-bottom: 1px solid #34495e;
    }
    .nav-toggle:checked ~ .nav-links {
      transform: translateY(0);
    }
    .nav-toggle {
      display: block;
    }
    .nav-toggle-label {
      display: block;
    }
  }
  
  
  .gallery {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
    gap: 15px;
  }
  .gallery-item {
    position: relative;
    overflow: hidden;
    border-radius: 8px;
    transition: transform 0.3s ease, box-shadow 0.3s ease;
    height: 150px; 
  }
  .gallery-item:hover {
    transform: scale(1.05);
    box-shadow: 0 4px 10px rgba(0,0,0,0.2);
  }
  .img-container {
    height: 100%;
    overflow: hidden;
  }
  .img-container img {
    height: 100%;
    width: auto;
    display: block;
    margin: 0 auto;
    object-fit: cover;
    cursor: pointer;
  }
  
  
  .current-image-container img,
  .last-sent-img {
    max-width: 300px;
    max-height: 300px;
    width: auto;
    height: auto;
    margin: 0 auto;
    display: block;
  }
  
  
  .overlay {
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: rgba(0,0,0,0.5);
    opacity: 0;
    transition: opacity 0.3s ease;
    border-radius: 8px;
  }
  .img-container:hover .overlay {
    opacity: 1;
  }
  
  .crop-icon {
    position: absolute;
    top: 5px;
    left: 5px;
    width: 30px;
    height: 30px;
    display: flex;
    align-items: center;
    justify-content: center;
    color: #fff;
    background: rgba(0,0,0,0.5);
    border-radius: 50%;
    cursor: pointer;
    z-index: 10;
  }
  .crop-icon:hover {
    background: rgba(0,0,0,0.7);
  }
  
  .delete-icon {
    position: absolute;
    top: 5px;
    right: 5px;
    width: 24px;
    height: 24px;
    cursor: pointer;
    z-index: 10;
    color: #fff;
    font-size: 20px;
  }
  
  
  .favorite-icon {
    position: absolute;
    top: 5px;
    left: 50%;
    transform: translateX(-50%);
    width: 24px;
    height: 24px;
    cursor: pointer;
    z-index: 10;
    color: #fff;
    font-size: 20px;
  }
  
  .send-button {
    position: absolute;
    left: 50%;
    top: 50%;
    transform: translate(-50%, -50%);
    background: #28a745;
    color: #fff;
    border: none;
    padding: 8px 12px;
    border-radius: 4px;
    cursor: pointer;
    transition: background 0.3s ease;
  }
  .send-button:hover {
    background: #218838;
  }
  
  
  .lightbox-modal {
    display: none;
    position: fixed;
    z-index: 4000;
    left: 0;
    top: 0;
    width: 100%;
    height: 100%;
    overflow: auto;
    background-color: rgba(0,0,0,0.9);
  }
  .lightbox-content {
    margin: auto;
    display: block;
    max-width: 90%;
    max-height: 80%;
    animation: zoomIn 0.3s;
  }
  @keyframes zoomIn {
    from { transform: scale(0.8); opacity: 0; }
    to { transform: scale(1); opacity: 1; }
  }
  .lightbox-close {
    position: absolute;
    top: 20px;
    right: 35px;
    color: #f1f1f1;
    font-size: 40px;
    font-weight: bold;
    transition: 0.3s;
    cursor: pointer;
  }
  .lightbox-close:hover,
  .lightbox-close:focus {
    color: #bbb;
  }
  #lightboxCaption {
    text-align: center;
    color: #ccc;
    padding: 10px 0;
  }
  
  
  .popup {
    position: fixed;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    background: rgba(255,255,255,0.95);
    border: 2px solid #ccc;
    padding: 30px;
    box-shadow: 0 4px 10px rgba(0,0,0,0.3);
    z-index: 10000;
    text-align: center;
    font-size: 1.5em;
    display: none;
    border-radius: 8px;
    animation: popupFade 0.5s ease;
  }
  @keyframes popupFade {
    from { opacity: 0; transform: translate(-50%, -50%) scale(0.8); }
    to { opacity: 1; transform: translate(-50%, -50%) scale(1); }
  }
  
  
  .progress-container {
    width: 60%;
    margin: 20px auto;
    background: #ddd;
    border-radius: 5px;
    display: none;
  }
  .progress-bar {
    width: 0%;
    height: 30px;
    background: #28a745;
    border-radius: 5px;
    transition: width 0.4s ease;
    color: #fff;
    line-height: 30px;
    font-size: 1em;
    text-align: center;
  }
  
  
  .modal {
    display: none;
    position: fixed;
    z-index: 3000;
    left: 0;
    top: 0;
    width: 100%;
    height: 100%;
    background-color: rgba(0,0,0,0.7);
    padding: 20px;
  }
  .modal-content {
    background: #fff;
    margin: 5% auto;
    padding: 20px;
    border-radius: 8px;
    max-width: 500px;
    position: relative;
  }
  .close {
    position: absolute;
    top: 10px;
    right: 15px;
    font-size: 1.5em;
    color: #333;
    cursor: pointer;
  }
  
  
  input[type="submit"],
  button,
  .primary-btn {
    background: linear-gradient(to right, #28a745, #218838);
    border: none;
    color: #fff;
    padding: 10px 15px;
    border-radius: 4px;
    cursor: pointer;
    font-size: 1em;
    transition: background 0.3s ease;
  }
  input[type="submit"]:hover,
  button:hover,
  .primary-btn:hover {
    background: linear-gradient(to right, #218838, #1e7e34);
  }
  
  
  input[type="text"],
  input[type="password"],
  select {
    width: 100%;
    padding: 10px;
    margin: 10px 0;
    border: 1px solid #ccc;
    border-radius: 4px;
  }
  label {
    font-weight: bold;
  }
  
  
  .calendar {
    width: 100%;
    border-collapse: collapse;
    margin-top: 20px;
  }
  .calendar th,
  .calendar td {
    border: 1px solid #ddd;
    padding: 8px;
    text-align: center;
  }
  .calendar th {
    background: #f8f8f8;
  }
  .calendar .droppable.over {
    background: #dff0d8;
  }
  
  
  .footer {
    text-align: center;
    padding: 15px;
    background: #2c3e50;
    color: #bdc3c7;
    position: fixed;
    bottom: 0;
    width: 100%;
  }
  
  
  a:focus,
  button:focus,
  input:focus {
    outline: 2px solid #2980b9;
    outline-offset: 2px;
  }
```


## static/trash-icon.png

```png
�PNG

   IHDR   0   0   W��   	pHYs     ��  �IDATx��KNA��^A]�]|$�H�"}
ءG���[�`���	P��63�!H��46Z_R!���ꯙE�R� T�պf��x�Q� -�S)��h�{Cd�p��Q)b��]�����m�J�v�$}�����<�V(�R�����rg�o�.��*5	Z����W��&�d`��LG5&L�-��Uj���o��
u�V�e�AU$L�3��50��!�goઌ/�8�q_2��'���H$
�!#D2Ba���P2B$#ƿ!η/�櫘Ѳ5�*_2$}�����c�����N������sq@W?#:�R x�:�{o<�܁U(a��e����)�SU3&:b�e���X�C���E00̚��'���; �i������ݰE�7�H    IEND�B`�
```


## templates/base.html

```html
<!doctype html>
<html>
  <head>
    <title>{% block title %}InkyDocker{% endblock %}</title>
    <link rel="stylesheet" href="{{ url_for('static', filename='style.css') }}">
    
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    
    <link rel="stylesheet"
          href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css"
          crossorigin="anonymous"
          referrerpolicy="no-referrer" />
    
    <link rel="stylesheet"
          href="https://cdnjs.cloudflare.com/ajax/libs/cropperjs/1.6.2/cropper.min.css"
          crossorigin="anonymous"
          referrerpolicy="no-referrer" />
    <style>
      /* Make the site scroll fully behind the footer */
      html, body {
        height: 100%;
        margin: 0;
        padding: 0;
      }
      .wrapper {
        min-height: 100%;
        display: flex;
        flex-direction: column;
      }
      .main-content-wrapper {
        flex: 1 0 auto;
      }
      .footer {
        flex-shrink: 0;
      }
    </style>
    {% block head %}{% endblock %}
  </head>
  <body>
    <div class="wrapper">
      <nav class="navbar navbar-expand-lg navbar-dark bg-dark" role="navigation" aria-label="Main Navigation">
        <div class="container-fluid">
          <a href="{{ url_for('image.upload_file') }}" class="navbar-brand">InkyDocker</a>
          <button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarNav"
            aria-controls="navbarNav" aria-expanded="false" aria-label="Toggle navigation">
            <span class="navbar-toggler-icon"></span>
          </button>
          <div class="collapse navbar-collapse" id="navbarNav">
            <ul class="navbar-nav ms-auto">
              <li class="nav-item">
                <a class="nav-link {% if request.endpoint.startswith('image.') %}active{% endif %}" href="{{ url_for('image.upload_file') }}">Gallery</a>
              </li>
              <li class="nav-item">
                <a class="nav-link {% if request.endpoint.startswith('schedule.') %}active{% endif %}" href="{{ url_for('schedule.schedule_page') }}">Schedule</a>
              </li>
              <li class="nav-item">
                <a class="nav-link {% if request.endpoint.startswith('settings.') %}active{% endif %}" href="{{ url_for('settings.settings') }}">Settings</a>
              </li>
            </ul>
          </div>
        </div>
      </nav>
      <div class="main-content-wrapper">
        {% block content %}{% endblock %}
      </div>
      <footer class="footer bg-dark text-light text-center py-3">
        <p class="mb-0">© 2025 InkyDocker | Built with AI by Me</p>
      </footer>
    </div>
    
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
    
    <script src="https://cdnjs.cloudflare.com/ajax/libs/cropperjs/1.6.2/cropper.min.js"
            crossorigin="anonymous"
            referrerpolicy="no-referrer"></script>
    {% block scripts %}{% endblock %}
  </body>
</html>
```


## templates/index.html

```html
{% extends "base.html" %}
{% block title %}Gallery - InkyDocker{% endblock %}

{% block content %}
<div class="container">
  
  <div class="card current-image-section">
    <h2 id="currentImageTitle">Current image on {{ devices[0].friendly_name if devices else 'N/A' }}</h2>
    <div class="current-image-container">
      {% if devices and devices[0].last_sent %}
        <img
          id="currentImage"
          src="{{ url_for('image.uploaded_file', filename=devices[0].last_sent) }}"
          alt="Current Image"
          class="last-sent-img small-current"
          loading="lazy"
        >
      {% else %}
        <p id="currentImagePlaceholder">No image available.</p>
      {% endif %}
    </div>
    {% if devices|length > 1 %}
      <div class="arrow-controls">
        <button id="prevDevice">&larr;</button>
        <button id="nextDevice">&rarr;</button>
      </div>
    {% endif %}
  </div>

  
  
  
  <div id="uploadPopup" style="position: fixed; top: 50%; left: 50%; transform: translate(-50%, -50%); background: rgba(0,0,0,0.8); color: #fff; padding: 15px 20px; border-radius: 5px; z-index: 1000; display: none;">
    <div class="spinner"></div> Processing...
  </div>

  
  <div class="main-content">
    
    <div class="left-panel">
      <div class="card device-section">
        <h2>Select eInk Display</h2>
        {% if devices %}
          <div class="device-options">
            {% for device in devices %}
              <label class="device-option">
                <input
                  type="radio"
                  name="device"
                  value="{{ device.address }}"
                  data-index="{{ loop.index0 }}"
                  data-friendly="{{ device.friendly_name }}"
                  data-resolution="{{ device.resolution }}"
                  {% if loop.first %}checked{% endif %}
                >
                {{ device.friendly_name }}
              </label>
            {% endfor %}
          </div>
        {% else %}
          <p>No devices configured. Go to <a href="{{ url_for('settings.settings') }}">Settings</a>.</p>
        {% endif %}
      </div>

      <div class="card upload-section">
        <h2>Upload Images</h2>
        <form id="uploadForm" class="upload-form" method="post" enctype="multipart/form-data" action="{{ url_for('image.upload_file') }}">
          <input type="file" name="file" multiple id="fileInput" required>
          <br>
          <input type="submit" value="Upload">
          <div class="progress-container" id="progressContainer" style="display: none;">
            <div class="progress-bar" id="progressBar">0%</div>
          </div>
          <div id="uploadStatus"></div>
        </form>
      </div>
      
      
    </div>

    
    <div class="gallery-section">
      <h2>Gallery</h2>
      <input type="text" id="gallerySearch" placeholder="Search images by tags..." style="width:100%; padding:10px; margin-bottom:20px;">
      <div id="searchSpinner" style="display:none;">Loading...</div>
      <div class="gallery" id="gallery">
        {% for image in images %}
          <div class="gallery-item">
            <div class="img-container">
              <img src="{{ url_for('image.uploaded_file', filename=image) }}" alt="{{ image }}" data-filename="{{ image }}" loading="lazy">
              <div class="overlay">
                
                <div class="favorite-icon" title="Favorite" data-image="{{ image }}">
                  <i class="fa fa-heart"></i>
                </div>
                
                <button class="send-button" data-image="{{ image }}">Send</button>
                
                <button class="info-button" data-image="{{ image }}">Info</button>
                
                <div class="delete-icon" title="Delete" data-image="{{ image }}">
                  <i class="fa fa-trash"></i>
                </div>
              </div>
            </div>
          </div>
        {% endfor %}
      </div>
    </div>
  </div>
</div>


<div id="infoModal" class="modal" style="display:none;">
  <div class="modal-content" style="max-width:800px; margin:auto; position:relative; padding:20px;">
    <span class="close" onclick="closeInfoModal()" style="position:absolute; top:10px; right:15px; cursor:pointer; font-size:1.5em;">&times;</span>
    <h2>Image Info</h2>
    <div style="text-align:center; margin-bottom:20px;">
      <img id="infoImagePreview" src="" alt="Info Preview" style="max-width:300px;">
      <div style="margin-top:10px;">
        <button type="button" onclick="openCropModal()">Crop Image</button>
      </div>
    </div>
    <div style="display:flex; gap:20px;">
      
      <div style="flex:1;" id="infoLeftColumn">
        <p><strong>Filename:</strong> <span id="infoFilename">N/A</span></p>
        <p><strong>Resolution:</strong> <span id="infoResolution">N/A</span></p>
        <p><strong>Filesize:</strong> <span id="infoFilesize">N/A</span></p>
      </div>
      
      <div style="flex:1;">
        <div style="margin-bottom:10px;">
          <label><strong>Tags:</strong></label>
          <div id="tagContainer" style="margin-top:5px; margin-bottom:10px;"></div>
          <div style="display:flex;">
            <input type="text" id="newTagInput" style="flex-grow:1;" placeholder="Add a new tag...">
            <button type="button" onclick="addTag()" style="margin-left:5px;">Add</button>
          </div>
          <input type="hidden" id="infoTags">
        </div>
        <div style="margin-bottom:10px;">
          <label><strong>Favorite:</strong></label>
          <input type="checkbox" id="infoFavorite">
        </div>
        <div id="infoStatus" style="color: green; margin-bottom:10px;"></div>
        <button onclick="saveInfoEdits()">Save</button>
        <button onclick="runOpenClip()">Re-run Tagging</button>
      </div>
    </div>
  </div>
</div>


<div id="lightboxModal" class="modal lightbox-modal" style="display:none;">
  <span class="lightbox-close" onclick="closeLightbox()">&times;</span>
  <img class="lightbox-content" id="lightboxImage" alt="Enlarged Image">
  <div id="lightboxCaption"></div>
</div>


<div id="cropModal" class="modal" style="display:none;">
  <div class="modal-content">
    <span class="close" onclick="closeCropModal()" style="cursor:pointer; font-size:1.5em;">&times;</span>
    <h2>Crop Image</h2>
    <div id="cropContainer" style="max-width:100%; max-height:80vh;">
      <img id="cropImage" src="" alt="Crop Image" style="width:100%;">
    </div>
    <div style="margin-top:10px;">
      <button type="button" onclick="saveCropData()">Save Crop</button>
      <button type="button" onclick="closeCropModal()">Cancel</button>
    </div>
  </div>
</div>
{% endblock %}

{% block scripts %}
<script>
document.addEventListener("DOMContentLoaded", function() {
  // Inject dynamic CSS for info button, favorite icon, and tag boxes
  const styleTag = document.createElement('style');
  styleTag.innerHTML = `
    .info-button {
      position: absolute;
      left: 50%;
      bottom: 10px;
      transform: translateX(-50%);
      background: #17a2b8;
      color: #fff;
      border: none;
      padding: 8px 12px;
      border-radius: 4px;
      cursor: pointer;
      transition: background 0.3s ease;
      font-size: 0.9em;
    }
    .info-button:hover {
      background: #138496;
    }
    .favorite-icon i {
      font-size: 1.5em;
      color: #ccc;
      transition: color 0.3s;
    }
    .favorite-icon.favorited i {
      color: red;
    }
    .tag-box {
      display: inline-block;
      background-color: #e9ecef;
      border-radius: 4px;
      padding: 5px 10px;
      margin: 3px;
      font-size: 0.9em;
    }
    .tag-remove {
      margin-left: 5px;
      cursor: pointer;
      font-weight: bold;
      color: #dc3545;
    }
    .tag-remove:hover {
      color: #bd2130;
    }
  `;
  document.head.appendChild(styleTag);
});

/* Lightbox functions */
function openLightbox(src, alt) {
  const lightboxModal = document.getElementById('lightboxModal');
  const lightboxImage = document.getElementById('lightboxImage');
  const lightboxCaption = document.getElementById('lightboxCaption');
  lightboxModal.style.display = 'block';
  lightboxImage.src = src;
  lightboxCaption.innerText = alt;
}
function closeLightbox() {
  document.getElementById('lightboxModal').style.display = 'none';
}

/* Debounce helper */
function debounce(func, wait) {
  let timeout;
  return function(...args) {
    clearTimeout(timeout);
    timeout = setTimeout(() => func.apply(this, args), wait);
  };
}

/* Gallery search */
const searchInput = document.getElementById('gallerySearch');
const searchSpinner = document.getElementById('searchSpinner');
const gallery = document.getElementById('gallery');

const performSearch = debounce(function() {
  const query = searchInput.value.trim();
  if (!query) {
    searchSpinner.style.display = 'none';
    location.reload();
    return;
  }
  searchSpinner.style.display = 'block';
  fetch(`/api/search_images?q=${encodeURIComponent(query)}`)
    .then(response => response.json())
    .then(data => {
      searchSpinner.style.display = 'none';
      gallery.innerHTML = "";
      if (data.status === "success" && data.results.ids) {
        if (data.results.ids.length === 0) {
          gallery.innerHTML = "<p>No matching images found.</p>";
        } else {
          data.results.ids.forEach((id) => {
            const imageUrl = `/images/${encodeURIComponent(id)}`;
            const item = document.createElement('div');
            item.className = 'gallery-item';
            item.innerHTML = `
              <div class="img-container">
                <img src="${imageUrl}" alt="${id}" data-filename="${id}" loading="lazy">
                <div class="overlay">
                  <div class="favorite-icon" title="Favorite" data-image="${id}">
                    <i class="fa fa-heart"></i>
                  </div>
                  <button class="send-button" data-image="${id}">Send</button>
                  <button class="info-button" data-image="${id}">Info</button>
                  <div class="delete-icon" title="Delete" data-image="${id}">
                    <i class="fa fa-trash"></i>
                  </div>
                </div>
              </div>
            `;
            gallery.appendChild(item);
          });
        }
      } else {
        gallery.innerHTML = "<p>No matching images found.</p>";
      }
    })
    .catch(err => {
      searchSpinner.style.display = 'none';
      console.error("Search error:", err);
    });
}, 500);

if (searchInput) {
  searchInput.addEventListener('input', performSearch);
}

/* Upload form */
const uploadForm = document.getElementById('uploadForm');
uploadForm.addEventListener('submit', function(e) {
  e.preventDefault();
  const fileInput = document.getElementById('fileInput');
  if (!fileInput.files.length) return;
  
  const formData = new FormData();
  for (let i = 0; i < fileInput.files.length; i++) {
    formData.append('file', fileInput.files[i]);
  }
  
  const selectedDevice = document.querySelector('input[name="device"]:checked');
  const deviceFriendly = selectedDevice ? selectedDevice.getAttribute('data-friendly') : "unknown display";
  
  const xhr = new XMLHttpRequest();
  xhr.open('POST', uploadForm.action, true);

  xhr.upload.addEventListener("progress", function(e) {
    if (e.lengthComputable) {
      const percentComplete = (e.loaded / e.total) * 100;
      const progressBar = document.getElementById('progressBar');
      progressBar.style.width = percentComplete + '%';
      progressBar.textContent = Math.round(percentComplete) + '%';
      document.getElementById('progressContainer').style.display = 'block';
      
      const popup = document.getElementById('uploadPopup');
      popup.style.display = 'block';
      popup.innerHTML = `<div class="spinner"></div> Uploading image to ${deviceFriendly}... ${Math.round(percentComplete)}%`;
    }
  });

  xhr.onload = function() {
    const popup = document.getElementById('uploadPopup');
    if (xhr.status === 200) {
      popup.innerHTML = `<div class="spinner"></div> Image uploaded successfully!`;
    } else {
      popup.innerHTML = `<div class="spinner"></div> Error uploading image.`;
    }
    setTimeout(() => {
      popup.style.display = 'none';
      location.reload();
    }, 1500);
  };

  xhr.onerror = function() {
    const popup = document.getElementById('uploadPopup');
    popup.innerHTML = `<div class="spinner"></div> Error uploading image.`;
    setTimeout(() => {
      popup.style.display = 'none';
    }, 1500);
  };

  xhr.send(formData);
});

/* Send image */
document.addEventListener('click', function(e) {
  if (e.target && e.target.classList.contains('send-button')) {
    e.stopPropagation();
    const imageFilename = e.target.getAttribute('data-image');
    const selectedDevice = document.querySelector('input[name="device"]:checked');
    if (!selectedDevice) return;
    
    const deviceFriendly = selectedDevice.getAttribute('data-friendly');
    const formData = new FormData();
    formData.append("device", selectedDevice.value);

    const baseUrl = "{{ url_for('image.send_image', filename='') }}";
    const finalUrl = baseUrl + encodeURIComponent(imageFilename);

    const xhr = new XMLHttpRequest();
    xhr.open('POST', finalUrl, true);

    xhr.upload.addEventListener("progress", function(ev) {
      if (ev.lengthComputable) {
        const percentComplete = (ev.loaded / ev.total) * 100;
        const popup = document.getElementById('uploadPopup');
        popup.style.display = 'block';
        popup.innerHTML = `<div class="spinner"></div> Sending image to ${deviceFriendly}... ${Math.round(percentComplete)}%`;
      }
    });

    xhr.onload = function() {
      const popup = document.getElementById('uploadPopup');
      if (xhr.status === 200) {
        popup.innerHTML = `<div class="spinner"></div> Image sent successfully!`;
      } else {
        popup.innerHTML = `<div class="spinner"></div> Error sending image.`;
      }
      setTimeout(() => {
        popup.style.display = 'none';
        location.reload();
      }, 1500);
    };

    xhr.onerror = function() {
      const popup = document.getElementById('uploadPopup');
      popup.innerHTML = `<div class="spinner"></div> Error sending image.`;
      setTimeout(() => {
        popup.style.display = 'none';
      }, 1500);
    };

    xhr.send(formData);
  }
});

/* Delete image */
document.addEventListener('click', function(e) {
  if (e.target && e.target.closest('.delete-icon')) {
    e.stopPropagation();
    const imageFilename = e.target.closest('.delete-icon').getAttribute('data-image');
    
    const deleteBaseUrl = "/delete_image/";
    const deleteUrl = deleteBaseUrl + encodeURIComponent(imageFilename);

    fetch(deleteUrl, { method: 'POST' })
      .then(response => response.json())
      .then(data => {
        if (data.status === "success") {
          location.reload();
        } else {
          console.error("Error deleting image:", data.message);
        }
      })
      .catch(error => {
        console.error("Error deleting image:", error);
      });
  }
});

/* Favorite toggle */
document.addEventListener('click', function(e) {
  if (e.target && e.target.closest('.favorite-icon')) {
    e.stopPropagation();
    const favIcon = e.target.closest('.favorite-icon');
    const imageFilename = favIcon.getAttribute('data-image');
    favIcon.classList.toggle('favorited');
    const isFavorited = favIcon.classList.contains('favorited');
    fetch("/api/update_image_metadata", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        filename: imageFilename,
        tags: [],  // do not modify tags in favorite toggle
        favorite: isFavorited
      })
    })
      .then(resp => resp.json())
      .then(data => {
        if (data.status !== "success") {
          console.error("Error updating favorite:", data.message);
        }
      })
      .catch(err => {
        console.error("Error updating favorite:", err);
      });
  }
});

/* Info Modal Logic */
let currentInfoFilename = null;

document.addEventListener('click', function(e) {
  if (e.target && e.target.classList.contains('info-button')) {
    e.stopPropagation();
    const filename = e.target.getAttribute('data-image');
    currentInfoFilename = filename;
    openInfoModal(filename);
  }
});

// Tag management functions
let currentTags = [];

function renderTags() {
  const tagContainer = document.getElementById('tagContainer');
  tagContainer.innerHTML = '';
  
  currentTags.forEach((tag, index) => {
    const tagElement = document.createElement('span');
    tagElement.className = 'tag-box';
    tagElement.innerHTML = `${tag} <span class="tag-remove" onclick="removeTag(${index})">×</span>`;
    tagContainer.appendChild(tagElement);
  });
  
  // Update the hidden input with comma-separated tags
  document.getElementById('infoTags').value = currentTags.join(', ');
}

function addTag() {
  const newTagInput = document.getElementById('newTagInput');
  const tag = newTagInput.value.trim();
  
  if (tag && !currentTags.includes(tag)) {
    currentTags.push(tag);
    renderTags();
    newTagInput.value = '';
  }
}

function removeTag(index) {
  currentTags.splice(index, 1);
  renderTags();
}

// Add event listener for Enter key on the new tag input
document.addEventListener('DOMContentLoaded', function() {
  const newTagInput = document.getElementById('newTagInput');
  if (newTagInput) {
    newTagInput.addEventListener('keypress', function(e) {
      if (e.key === 'Enter') {
        e.preventDefault();
        addTag();
      }
    });
  }
});

function openInfoModal(filename) {
  const imgUrl = `/images/${encodeURIComponent(filename)}?size=info`;
  fetch(`/api/get_image_metadata?filename=${encodeURIComponent(filename)}`)
    .then(resp => resp.json())
    .then(data => {
      if (data.status === "success") {
        document.getElementById('infoImagePreview').src = imgUrl;
        document.getElementById('infoFilename').textContent = filename;
        document.getElementById('infoResolution').textContent = data.resolution || "N/A";
        document.getElementById('infoFilesize').textContent = data.filesize || "N/A";
        
        // Set up tags
        currentTags = data.tags || [];
        renderTags();
        
        document.getElementById('infoFavorite').checked = data.favorite || false;
        document.getElementById('infoStatus').textContent = "";
        document.getElementById('infoModal').style.display = 'block';
      } else {
        document.getElementById('infoStatus').textContent = "Error: " + data.message;
        document.getElementById('infoModal').style.display = 'block';
      }
    })
    .catch(err => {
      console.error("Error fetching metadata:", err);
      document.getElementById('infoStatus').textContent = "Failed to fetch metadata. Check console.";
      document.getElementById('infoModal').style.display = 'block';
    });
}

function closeInfoModal() {
  document.getElementById('infoModal').style.display = 'none';
  currentInfoFilename = null;
}

function saveInfoEdits() {
  if (!currentInfoFilename) return;
  fetch("/api/update_image_metadata", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      filename: currentInfoFilename,
      tags: currentTags,
      favorite: document.getElementById('infoFavorite').checked
    })
  })
    .then(resp => resp.json())
    .then(data => {
      if (data.status === "success") {
        document.getElementById('infoStatus').textContent = "Metadata updated successfully!";
        setTimeout(() => { closeInfoModal(); }, 1500);
      } else {
        document.getElementById('infoStatus').textContent = "Error updating metadata: " + data.message;
      }
    })
    .catch(err => {
      console.error("Error updating metadata:", err);
      document.getElementById('infoStatus').textContent = "Failed to update metadata. Check console.";
    });
}

function runOpenClip() {
  if (!currentInfoFilename) return;
  fetch(`/api/reembed_image?filename=${encodeURIComponent(currentInfoFilename)}`)
    .then(resp => resp.json())
    .then(data => {
      if (data.status === "success") {
        currentTags = data.tags || [];
        renderTags();
        document.getElementById('infoStatus').textContent = "Re-ran tagging successfully!";
      } else {
        document.getElementById('infoStatus').textContent = "Error re-running tagging: " + data.message;
      }
    })
    .catch(err => {
      console.error("Error re-running tagging:", err);
      document.getElementById('infoStatus').textContent = "Failed to re-run tagging. Check console.";
    });
}

// Crop Modal Functions
let cropperInstance = null;

function openCropModal() {
  if (!currentInfoFilename) return;
  
  const cropModal = document.getElementById('cropModal');
  const cropImage = document.getElementById('cropImage');
  
  // Set the image source to the current image
  cropImage.src = `/images/${encodeURIComponent(currentInfoFilename)}`;
  
  // Show the modal
  cropModal.style.display = 'block';
  
  // Initialize Cropper.js after the image is loaded
  cropImage.onload = function() {
    if (cropperInstance) {
      cropperInstance.destroy();
    }
    
    // Import Cropper.js dynamically if needed
    if (typeof Cropper === 'undefined') {
      const script = document.createElement('script');
      script.src = 'https://cdnjs.cloudflare.com/ajax/libs/cropperjs/1.5.12/cropper.min.js';
      document.head.appendChild(script);
      
      const link = document.createElement('link');
      link.rel = 'stylesheet';
      link.href = 'https://cdnjs.cloudflare.com/ajax/libs/cropperjs/1.5.12/cropper.min.css';
      document.head.appendChild(link);
      
      script.onload = initCropper;
    } else {
      initCropper();
    }
  };
}

function initCropper() {
  const cropImage = document.getElementById('cropImage');
  cropperInstance = new Cropper(cropImage, {
    aspectRatio: NaN, // Free aspect ratio
    viewMode: 1,
    autoCropArea: 1,
    responsive: true,
    restore: true,
    guides: true,
    center: true,
    highlight: true,
    cropBoxMovable: true,
    cropBoxResizable: true,
    toggleDragModeOnDblclick: true,
  });
}

function closeCropModal() {
  const cropModal = document.getElementById('cropModal');
  cropModal.style.display = 'none';
  
  if (cropperInstance) {
    cropperInstance.destroy();
    cropperInstance = null;
  }
}

function saveCropData() {
  if (!cropperInstance || !currentInfoFilename) return;
  
  const cropData = cropperInstance.getData();
  
  fetch(`/save_crop_info/${encodeURIComponent(currentInfoFilename)}`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({
      x: cropData.x,
      y: cropData.y,
      width: cropData.width,
      height: cropData.height
    })
  })
  .then(response => response.json())
  .then(data => {
    if (data.status === 'success') {
      document.getElementById('infoStatus').textContent = 'Crop data saved successfully!';
      closeCropModal();
    } else {
      document.getElementById('infoStatus').textContent = 'Error saving crop data: ' + data.message;
    }
  })
  .catch(error => {
    console.error('Error saving crop data:', error);
    document.getElementById('infoStatus').textContent = 'Error saving crop data. Check console.';
  });
}

const prevButton = document.getElementById('prevDevice');
const nextButton = document.getElementById('nextDevice');
// Define devices array using template data
const devices = [
  {% for device in devices %}
    {
      friendly_name: "{{ device.friendly_name|e('js') }}",
      address: "{{ device.address|e('js') }}",
      last_sent: "{{ device.last_sent|e('js') if device.last_sent is defined else '' }}"
    }{% if not loop.last %},{% endif %}
  {% endfor %}
];
let currentDeviceIndex = 0;

function updateCurrentImageDisplay() {
  const device = devices[currentDeviceIndex];
  const titleEl = document.getElementById('currentImageTitle');
  const imageEl = document.getElementById('currentImage');
  const placeholderEl = document.getElementById('currentImagePlaceholder');
  
  titleEl.textContent = "Current image on " + device.friendly_name;
  if (device.last_sent) {
    if (placeholderEl) {
      placeholderEl.style.display = 'none';
    }
    if (imageEl) {
      imageEl.src = "{{ url_for('image.uploaded_file', filename='') }}" + device.last_sent;
      imageEl.style.display = 'block';
    }
  } else {
    if (imageEl) {
      imageEl.style.display = 'none';
    }
    if (placeholderEl) {
      placeholderEl.style.display = 'block';
    }
  }
}

if (prevButton && nextButton) {
  prevButton.addEventListener('click', function() {
    currentDeviceIndex = (currentDeviceIndex - 1 + devices.length) % devices.length;
    updateCurrentImageDisplay();
  });
  nextButton.addEventListener('click', function() {
    currentDeviceIndex = (currentDeviceIndex + 1) % devices.length;
    updateCurrentImageDisplay();
  });
}

if (devices.length > 0) {
  updateCurrentImageDisplay();
}

// Bulk tagging moved to settings page
</script>
{% endblock %}
```


## templates/schedule.html

```html
{% extends "base.html" %}
{% block title %}Schedule - InkyDocker{% endblock %}
{% block head %}
  
  <link href="https://cdn.jsdelivr.net/npm/fullcalendar@6.1.8/main.min.css" rel="stylesheet">
  
  <script src="https://cdn.jsdelivr.net/npm/fullcalendar@6.1.8/index.global.min.js" defer></script>
  <style>
    #calendar {
      max-width: 1000px;
      margin: 40px auto;
      margin-bottom: 100px; /* Add more space at the bottom */
    }
    
    /* Style for the event content to show thumbnails */
    .event-thumbnail {
      width: 100%;
      height: 40px;
      object-fit: cover;
      border-radius: 3px;
      margin-bottom: 2px;
    }
    
    /* Search bar styles */
    .search-container {
      margin-bottom: 15px;
    }
    
    #imageSearch {
      width: 100%;
      padding: 8px;
      border: 1px solid #ddd;
      border-radius: 4px;
      margin-bottom: 10px;
    }
    
    /* Improve gallery layout */
    .gallery {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
      gap: 10px;
      max-height: 500px;
      overflow-y: auto;
    }
    
    .gallery-item {
      height: 150px !important;
      position: relative;
      border: 1px solid #ddd;
      border-radius: 4px;
      overflow: hidden;
    }
    
    .gallery-item img {
      width: 100%;
      height: 100%;
      object-fit: cover;
      cursor: pointer;
      transition: transform 0.2s;
    }
    
    .gallery-item img:hover {
      transform: scale(1.05);
    }
    
    .img-container {
      height: 100%;
    }
    
    /* Tags display */
    .image-tags {
      position: absolute;
      bottom: 0;
      left: 0;
      right: 0;
      background: rgba(0,0,0,0.6);
      color: white;
      padding: 3px;
      font-size: 10px;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    /* Modal styles for event creation and image gallery (existing) */
    .modal {
      display: none;
      position: fixed;
      z-index: 10000;
      left: 0;
      top: 0;
      width: 100%;
      height: 100%;
      overflow: auto;
      background-color: rgba(0,0,0,0.7);
      padding: 20px;
    }
    .modal-content {
      background: #fff;
      margin: 10% auto;
      padding: 20px;
      border-radius: 8px;
      max-width: 500px;
      position: relative;
    }
    .close {
      position: absolute;
      top: 10px;
      right: 15px;
      font-size: 1.5em;
      cursor: pointer;
    }
    /* Deletion modal for recurring events */
    #deleteModal .modal-content {
      max-width: 400px;
      text-align: center;
    }
    #deleteModal button {
      margin: 5px;
    }
  </style>
{% endblock %}
{% block content %}
<div class="container">
  <header class="page-header">
    <h1>Schedule Images</h1>
    <p>Manage your scheduled image updates with our interactive calendar.</p>
  </header>
  <div id="calendar"></div>
</div>


<div id="eventModal" class="modal">
  <div class="modal-content">
    <span class="close" onclick="closeEventModal()">&times;</span>
    <h2>Add Scheduled Event</h2>
    <form id="eventForm">
      <div>
        <label for="eventDate">Date &amp; Time:</label>
        <input type="datetime-local" id="eventDate" name="eventDate" required>
      </div>
      <div>
        <label>Select eInk Display:</label>
        {% if devices %}
          {% for device in devices %}
            <label>
              <input type="radio" name="device" value="{{ device.address }}" {% if loop.first %}checked{% endif %}>
              {{ device.friendly_name }}
            </label>
          {% endfor %}
        {% else %}
          <p>No devices configured. Please add devices in <a href="{{ url_for('settings.settings') }}">Settings</a>.</p>
        {% endif %}
      </div>
      <div>
        <label for="recurrence">Recurrence:</label>
        <select id="recurrence" name="recurrence">
          <option value="none">None</option>
          <option value="daily">Daily</option>
          <option value="weekly">Weekly</option>
          <option value="monthly">Same date next month</option>
        </select>
      </div>
      <div>
        <label>Choose Image:</label>
        <button type="button" onclick="openImageGallery()">Select Image</button>
        <input type="hidden" id="selectedImage" name="selectedImage">
        <span id="selectedImageName"></span>
      </div>
      <div style="margin-top:10px;">
        <input type="submit" value="Save Event">
      </div>
    </form>
  </div>
</div>


<div id="imageGalleryModal" class="modal">
  <div class="modal-content" style="max-width:800px;">
    <span class="close" onclick="closeImageGallery()">&times;</span>
    <h2>Select an Image</h2>
    
    
    <div class="search-container">
      <input type="text" id="imageSearch" placeholder="Search by tags...">
    </div>
    
    <div class="gallery" id="galleryModal">
      {% for image in images %}
        <div class="gallery-item" data-tags="{{ image_tags.get(image, '') }}">
          <div class="img-container">
            <img src="{{ url_for('image.thumbnail', filename=image) }}" alt="{{ image }}" data-filename="{{ image }}" onclick="selectImage('{{ image }}', this.src)">
            {% if image_tags.get(image) %}
              <div class="image-tags">{{ image_tags.get(image) }}</div>
            {% endif %}
          </div>
        </div>
      {% endfor %}
    </div>
  </div>
</div>


<div id="deleteModal" class="modal">
  <div class="modal-content">
    <span class="close" onclick="closeDeleteModal()">&times;</span>
    <h3>Delete Recurring Event</h3>
    <p>Delete this occurrence or the entire series?</p>
    <button id="deleteOccurrenceBtn" class="btn btn-danger">Delete this occurrence</button>
    <button id="deleteSeriesBtn" class="btn btn-danger">Delete entire series</button>
    <button onclick="closeDeleteModal()" class="btn btn-secondary">Cancel</button>
  </div>
</div>
{% endblock %}
{% block scripts %}
  <script>
    var currentDeleteEventId = null; // store event id for deletion modal

    document.addEventListener('DOMContentLoaded', function() {
      var calendarEl = document.getElementById('calendar');
      if (!calendarEl) return;
      if (typeof FullCalendar === 'undefined') {
        console.error("FullCalendar is not defined");
        return;
      }
      var calendar = new FullCalendar.Calendar(calendarEl, {
        initialView: 'timeGridWeek',
        firstDay: 1,
        nowIndicator: true,
        editable: true,
        headerToolbar: {
          left: 'prev,next today',
          center: 'title',
          right: 'timeGridWeek,timeGridDay'
        },
        events: '/schedule/events',
        eventDrop: function(info) {
          var newDate = info.event.start;
          var isoStr = newDate.toISOString().substring(0,16);
          fetch("/schedule/update", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ event_id: info.event.id, datetime: isoStr })
          })
          .then(response => response.json())
          .then(data => {
            if(data.status !== "success"){
              alert("Error updating event: " + data.message);
            }
          })
          .catch(err => {
            console.error("Error updating event:", err);
          });
        },
        eventDidMount: function(info) {
          // Add delete button
          var deleteEl = document.createElement('span');
          deleteEl.innerHTML = '&times;';
          deleteEl.style.position = 'absolute';
          deleteEl.style.top = '2px';
          deleteEl.style.right = '2px';
          deleteEl.style.color = 'red';
          deleteEl.style.cursor = 'pointer';
          deleteEl.style.fontWeight = 'bold';
          deleteEl.style.zIndex = '100';
          info.el.appendChild(deleteEl);
          
          // Add thumbnail to event
          if (info.event.extendedProps.thumbnail) {
            var thumbnailEl = document.createElement('img');
            thumbnailEl.src = info.event.extendedProps.thumbnail;
            thumbnailEl.className = 'event-thumbnail';
            thumbnailEl.alt = info.event.title;
            
            // Find the event title element and insert the thumbnail before it
            var titleEl = info.el.querySelector('.fc-event-title');
            if (titleEl && titleEl.parentNode) {
              titleEl.parentNode.insertBefore(thumbnailEl, titleEl);
              
              // Add device name to title
              if (info.event.extendedProps.deviceName) {
                titleEl.textContent = info.event.title + ' on ' + info.event.extendedProps.deviceName;
              }
            }
          }
          
          // Add click handler for delete button
          deleteEl.addEventListener('click', function(e) {
            e.stopPropagation();
            // If recurring, show custom deletion modal; otherwise, delete directly.
            if(info.event.extendedProps.recurrence && info.event.extendedProps.recurrence.toLowerCase() !== "none"){
              currentDeleteEventId = info.event.id;
              openDeleteModal();
            } else {
              // Directly delete non-recurring event without popup.
              fetch("/schedule/remove/" + info.event.id, { method: "POST" })
              .then(response => response.json())
              .then(data => {
                if(data.status === "success"){
                  info.event.remove();
                }
              })
              .catch(err => {
                console.error("Error deleting event:", err);
              });
            }
          });
        },
        dateClick: function(info) {
          var dtLocal = new Date(info.date);
          var isoStr = dtLocal.toISOString().substring(0,16);
          document.getElementById('eventDate').value = isoStr;
          openEventModal();
        }
      });
      calendar.render();
    });

    function openEventModal() { document.getElementById('eventModal').style.display = 'block'; }
    function closeEventModal() { document.getElementById('eventModal').style.display = 'none'; }
    
    function openImageGallery() {
      document.getElementById('imageGalleryModal').style.display = 'block';
      // Clear search field when opening
      var searchField = document.getElementById('imageSearch');
      if (searchField) {
        searchField.value = '';
        searchField.focus();
        // Trigger search to show all images
        filterImages('');
      }
    }
    
    function closeImageGallery() { document.getElementById('imageGalleryModal').style.display = 'none'; }
    
    function selectImage(filename, src) {
      document.getElementById('selectedImage').value = filename;
      document.getElementById('selectedImageName').textContent = filename;
      // Also show the thumbnail
      var nameSpan = document.getElementById('selectedImageName');
      if (nameSpan) {
        nameSpan.innerHTML = `<img src="${src}" style="height:40px;margin-right:5px;vertical-align:middle;"> ${filename}`;
      }
      closeImageGallery();
    }
    
    function openDeleteModal() { document.getElementById('deleteModal').style.display = 'block'; }
    function closeDeleteModal() { document.getElementById('deleteModal').style.display = 'none'; }
    
    // Function to filter images based on search input
    function filterImages(searchText) {
      searchText = searchText.toLowerCase();
      var items = document.querySelectorAll('#galleryModal .gallery-item');
      
      items.forEach(function(item) {
        var tags = item.getAttribute('data-tags') || '';
        var filename = item.querySelector('img').getAttribute('data-filename') || '';
        
        if (tags.toLowerCase().includes(searchText) || filename.toLowerCase().includes(searchText) || searchText === '') {
          item.style.display = '';
        } else {
          item.style.display = 'none';
        }
      });
    }
    
    // Add event listener for search input
    document.addEventListener('DOMContentLoaded', function() {
      var searchInput = document.getElementById('imageSearch');
      if (searchInput) {
        searchInput.addEventListener('input', function() {
          filterImages(this.value);
        });
      }
    });

    document.getElementById('deleteOccurrenceBtn').addEventListener('click', function() {
      // Skip this occurrence for recurring event.
      fetch("/schedule/skip/" + currentDeleteEventId, { method: "POST" })
      .then(response => response.json())
      .then(data => {
        if(data.status === "success"){
          // Remove the occurrence from the calendar.
          location.reload();
        }
      })
      .catch(err => {
        console.error("Error skipping occurrence:", err);
      });
      closeDeleteModal();
    });

    document.getElementById('deleteSeriesBtn').addEventListener('click', function() {
      // Delete the entire series.
      fetch("/schedule/remove/" + currentDeleteEventId, { method: "POST" })
      .then(response => response.json())
      .then(data => {
        if(data.status === "success"){
          location.reload();
        }
      })
      .catch(err => {
        console.error("Error deleting series:", err);
      });
      closeDeleteModal();
    });

    document.getElementById('eventForm').addEventListener('submit', function(e) {
      e.preventDefault();
      var datetime = document.getElementById('eventDate').value;
      var device = document.querySelector('input[name="device"]:checked').value;
      var recurrence = document.getElementById('recurrence').value;
      var filename = document.getElementById('selectedImage').value;
      if (!datetime || !device || !filename) {
        alert("Please fill in all fields and select an image.");
        return;
      }
      fetch("/schedule/add", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          datetime: datetime,
          device: device,
          recurrence: recurrence,
          filename: filename
        })
      })
      .then(response => response.json())
      .then(data => {
        if(data.status === "success"){
          location.reload();
        } else {
          alert("Error: " + data.message);
        }
      })
      .catch(err => {
        console.error("Error adding event:", err);
      });
    });
  </script>
{% endblock %}
```


## templates/settings.html

```html
{% extends "base.html" %}
{% block title %}Settings - InkyDocker{% endblock %}

{% block content %}
<div class="container">
  
  <header class="page-header">
    <h1>Settings</h1>
    <p>Manage your eInk displays, AI settings, and device metrics.</p>
  </header>

  
  <div class="card text-center">
    <button id="clipSettingsBtn" class="primary-btn">CLIP Model Settings</button>
  </div>

  
  <div id="clipSettingsModal" class="modal">
    <div class="modal-content">
      <span class="close" id="closeClipSettingsModal">&times;</span>
      <h2>CLIP Model Settings</h2>
      <form id="clipSettingsForm">
        
        <div class="form-group">
          <label for="clip_model">Select CLIP Model for Image Tagging:</label>
          <select id="clip_model" name="clip_model" class="form-select" data-current="{{ config.clip_model if config and config.clip_model }}">
            <option value="">-- Select a model --</option>
            <option value="ViT-B-32" {% if config and config.clip_model == 'ViT-B-32' %}selected{% endif %}>ViT-B-32 (Faster, less accurate)</option>
            <option value="ViT-B-16" {% if config and config.clip_model == 'ViT-B-16' %}selected{% endif %}>ViT-B-16 (Balanced)</option>
            <option value="ViT-L-14" {% if config and config.clip_model == 'ViT-L-14' %}selected{% endif %}>ViT-L-14 (Slower, more accurate)</option>
          </select>
          <button type="button" class="field-save-btn" onclick="saveClipModel()">Save Model</button>
        </div>
        
        
        <div id="modelDownloadContainer" style="margin-top: 15px; display: none;">
          <p>Downloading model: <span id="modelDownloadName"></span></p>
          <div class="progress-container" style="width: 100%; background: #ddd; border-radius: 5px;">
            <div id="modelDownloadProgress" class="progress-bar" style="width: 0%; height: 20px; background: #28a745; border-radius: 5px; color: #fff; text-align: center; line-height: 20px;">0%</div>
          </div>
        </div>
        
        <div style="margin-top: 20px;">
          <p>All models are pre-installed in the system.</p>
          <p>Larger models provide more accurate tagging but require more processing power and memory.</p>
        </div>
        
        
        <div style="margin-top: 20px; text-align: center;">
          <button type="button" class="primary-btn" onclick="rerunAllTagging()">Rerun Tagging on All Images</button>
        </div>
      </form>
    </div>
  </div>

  
  <div class="card text-center">
    <button id="addNewDisplayBtn" class="primary-btn">Add New Display</button>
  </div>

  
  <div id="addDisplayModal" class="modal">
    <div class="modal-content">
      <span class="close" id="closeAddDisplayModal">&times;</span>
      <h2>Add New Display</h2>
      <form id="addDisplayForm" method="POST" action="{{ url_for('settings.settings') }}">
        <input type="text" name="address" id="newAddress" placeholder="Device Address (e.g., 192.168.1.100)" required>
        <select name="orientation" id="newOrientation" required>
          <option value="portrait">Portrait</option>
          <option value="landscape">Landscape</option>
        </select>
        <input type="text" name="friendly_name" id="newFriendlyName" placeholder="Friendly Name" required>
        
        <input type="hidden" name="display_name" id="newDisplayName">
        <input type="hidden" name="resolution" id="newResolution">
        <input type="hidden" name="color" id="newColor">
        <div style="margin-top: 10px;">
          <button type="button" class="primary-btn" onclick="fetchDisplayInfo('new')">Fetch Display Info</button>
        </div>
        <div style="margin-top: 10px;">
          <button type="submit" class="primary-btn">Save</button>
          <button type="button" class="primary-btn" onclick="closeAddDisplayModal()">Cancel</button>
        </div>
      </form>
    </div>
  </div>

  
  <div id="editDisplayModal" class="modal">
    <div class="modal-content">
      <span class="close" id="closeEditDisplayModal">&times;</span>
      <h2>Edit Display</h2>
      <form id="editDisplayForm" method="POST" action="{{ url_for('settings.edit_device') }}">
        <input type="hidden" name="device_index" id="editDeviceIndex">
        <label for="editFriendlyName">Friendly Name:</label>
        <input type="text" name="friendly_name" id="editFriendlyName" placeholder="Friendly Name" required>
        <label for="editOrientation">Orientation:</label>
        <select name="orientation" id="editOrientation" required>
          <option value="portrait">Portrait</option>
          <option value="landscape">Landscape</option>
        </select>
        <label for="editAddress">Device Address:</label>
        <input type="text" name="address" id="editAddress" placeholder="Device Address" required>
        <div style="margin-top: 10px;">
          <button type="submit" class="primary-btn">Save Changes</button>
          <button type="button" class="primary-btn" onclick="closeEditDisplayModal()">Cancel</button>
        </div>
      </form>
    </div>
  </div>

  
  <div id="advancedActionsModal" class="modal">
    <div class="modal-content">
      <span class="close" id="closeAdvancedActionsModal">&times;</span>
      <h2>Advanced Actions</h2>
      <p id="advancedDeviceTitle" style="font-weight:bold;"></p>
      <div style="margin-top: 10px;">
        <button type="button" class="primary-btn" onclick="triggerSystemUpdate()">System Update & Reboot</button>
        <button type="button" class="primary-btn" onclick="triggerBackup()">Create Backup</button>
        <button type="button" class="primary-btn" onclick="triggerAppUpdate()">Update Application</button>
      </div>
      <div style="margin-top: 10px;">
        <button type="button" class="primary-btn" onclick="closeAdvancedActionsModal()">Close</button>
      </div>
    </div>
  </div>

  
  <div class="card">
    <h2>Existing Devices</h2>
    <table class="device-table">
      <thead>
        <tr>
          <th>#</th>
          <th>Color</th>
          <th>Friendly Name</th>
          <th>Orientation</th>
          <th>Address</th>
          <th>Display Name</th>
          <th>Resolution</th>
          <th>Status</th>
          <th>CPU</th>
          <th>Memory</th>
          <th>Disk</th>
          <th>Action</th>
        </tr>
      </thead>
      <tbody>
        {% for device in devices %}
        <tr data-index="{{ loop.index0 }}" data-address="{{ device.address }}">
          <td>{{ loop.index }}</td>
          <td>
            <div style="width:20px; height:20px; border-radius:50%; background:{{ device.color }};"></div>
          </td>
          <td>{{ device.friendly_name }}</td>
          <td>{{ device.orientation }}</td>
          <td>{{ device.address }}</td>
          <td>{{ device.display_name }}</td>
          <td>{{ device.resolution }}</td>
          <td>
            {% if device.online %}
              <span style="color:green;">&#9679;</span>
            {% else %}
              <span style="color:red;">&#9679;</span>
            {% endif %}
          </td>
          <td class="cpu">{{ device.cpu_usage or 'N/A' }}%</td>
          <td class="mem">{{ device.mem_usage or 'N/A' }}%</td>
          <td class="disk">{{ device.disk_usage or 'N/A' }}%</td>
          <td>
            <form method="POST" action="{{ url_for('settings.delete_device', device_index=loop.index0) }}" style="display:inline;">
              <input type="submit" value="Delete">
            </form>
            <button type="button" class="edit-button" onclick="openEditModal('{{ loop.index0 }}', '{{ device.friendly_name }}', '{{ device.orientation }}', '{{ device.address }}')">
              Edit
            </button>
            <button type="button" class="advanced-button" onclick="openAdvancedModal('{{ loop.index0 }}', '{{ device.friendly_name }}')">
              Advanced
            </button>
          </td>
        </tr>
        {% endfor %}
      </tbody>
    </table>
  </div>
</div>
{% endblock %}

{% block scripts %}
<style>
  .modal {
    display: none;
    position: fixed;
    z-index: 3000;
    left: 0;
    top: 0;
    width: 100%;
    height: 100%;
    overflow: auto;
    background-color: rgba(0, 0, 0, 0.5);
  }
  .modal-content {
    background-color: #fff;
    margin: 10% auto;
    padding: 20px;
    border-radius: 5px;
    max-width: 500px;
    position: relative;
  }
  .close {
    color: #aaa;
    float: right;
    font-size: 28px;
    font-weight: bold;
    cursor: pointer;
  }
  .close:hover,
  .close:focus {
    color: #000;
  }
  .primary-btn {
    background: linear-gradient(to right, #28a745, #218838);
    border: none;
    color: #fff;
    padding: 10px 20px;
    border-radius: 5px;
    cursor: pointer;
    font-size: 1em;
  }
  .primary-btn:hover {
    background: linear-gradient(to right, #218838, #1e7e34);
  }
  .field-save-btn {
    margin-top: 5px;
    font-size: 0.9em;
    padding: 5px 10px;
    background-color: #007bff;
    color: #fff;
    border: none;
    border-radius: 3px;
    cursor: pointer;
  }
  .field-save-btn:hover {
    background-color: #0056b3;
  }
  .overlay-popup {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background-color: rgba(0, 0, 0, 0.7);
    display: flex;
    justify-content: center;
    align-items: center;
    z-index: 5000;
  }
  .overlay-content {
    background-color: #fff;
    padding: 20px;
    border-radius: 5px;
    max-width: 400px;
    text-align: center;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
  }
  .overlay-buttons {
    margin-top: 15px;
    display: flex;
    justify-content: center;
    gap: 10px;
  }
  .cancel-btn {
    background: #6c757d;
    border: none;
    color: #fff;
    padding: 8px 15px;
    border-radius: 5px;
    cursor: pointer;
  }
  .cancel-btn:hover {
    background: #5a6268;
  }
  .progress-container {
    margin: 10px 0;
    background-color: #f1f1f1;
    border-radius: 5px;
    overflow: hidden;
  }
  .progress-bar {
    height: 20px;
    background-color: #4CAF50;
    text-align: center;
    line-height: 20px;
    color: white;
    transition: width 0.3s ease;
  }
</style>

<script>
  // Attach global functions for modal closing
  window.closeAddDisplayModal = function() {
    document.getElementById('addDisplayModal').style.display = 'none';
  };
  window.closeEditDisplayModal = function() {
    document.getElementById('editDisplayModal').style.display = 'none';
  };
  window.closeAdvancedActionsModal = function() {
    document.getElementById('advancedActionsModal').style.display = 'none';
  };

  // Device metrics handling
  var activeStreams = {};
  
  // Function to connect to device metrics stream
  function connectToDeviceStream(index) {
    var row = document.querySelector('tr[data-index="' + index + '"]');
    if (!row) return;
    
    // If already connected, don't reconnect
    if (activeStreams[index]) return;
    
    console.log("Connecting to device stream for index:", index);
    
    // Connect to our backend SSE endpoint that proxies the device stream
    var source = new EventSource('/device/' + index + '/stream');
    activeStreams[index] = source;
    
    source.onopen = function() {
      console.log("SSE connection opened for device", index);
      // Update status indicator to connecting (yellow)
      var statusCell = row.querySelector('td:nth-child(8)');
      statusCell.innerHTML = '<span style="color:#FFA500;">&#9679;</span>';
    };
    
    source.onmessage = function(event) {
      try {
        console.log("Received SSE data for device", index, ":", event.data);
        var data = JSON.parse(event.data);
        
        // Check for status messages
        if (data.status === 'error') {
          console.error("Stream error:", data.message);
          return;
        }
        
        // Update the UI with the received metrics
        if (data.cpu !== undefined) {
          row.querySelector(".cpu").textContent = data.cpu + "%";
        }
        if (data.memory !== undefined) {
          row.querySelector(".mem").textContent = data.memory + "%";
        } else if (data.mem !== undefined) {
          row.querySelector(".mem").textContent = data.mem + "%";
        }
        if (data.disk !== undefined) {
          row.querySelector(".disk").textContent = data.disk + "%";
        }
        
        // Update status indicator
        var statusCell = row.querySelector('td:nth-child(8)');
        statusCell.innerHTML = '<span style="color:green;">&#9679;</span>';
      } catch (e) {
        console.error("Error parsing SSE data:", e, "Raw data:", event.data);
      }
    };
    
    source.onerror = function(event) {
      console.error("SSE connection error for device " + index, event);
      // Mark device as offline
      var statusCell = row.querySelector('td:nth-child(8)');
      statusCell.innerHTML = '<span style="color:red;">&#9679;</span>';
      
      // Close and remove the stream
      source.close();
      delete activeStreams[index];
      
      // Try to reconnect after a delay
      setTimeout(function() {
        connectToDeviceStream(index);
      }, 10000); // 10 second reconnection delay
    };
  }
  
  // Fallback polling for when SSE is not available
  function pollDeviceMetrics() {
    fetch("/devices/metrics")
      .then(function(response) { return response.json(); })
      .then(function(data) {
        if(data.status === "success") {
          data.devices.forEach(function(metric) {
            var row = document.querySelector('tr[data-index="' + metric.index + '"]');
            if (row) {
              // Only update if we don't have an active stream for this device
              if (!activeStreams[metric.index]) {
                row.querySelector(".cpu").textContent = metric.cpu + "%";
                row.querySelector(".mem").textContent = metric.mem + "%";
                row.querySelector(".disk").textContent = metric.disk + "%";
                var statusCell = row.querySelector('td:nth-child(8)');
                if(metric.online) {
                  statusCell.innerHTML = '<span style="color:green;">&#9679;</span>';
                  // Try to connect to stream if device is online
                  connectToDeviceStream(metric.index);
                } else {
                  statusCell.innerHTML = '<span style="color:red;">&#9679;</span>';
                }
              }
            }
          });
        }
      })
      .catch(function(error) {
        console.error("Error polling device metrics:", error);
      });
  }
  
  // Poll metrics every 5 seconds
  setInterval(pollDeviceMetrics, 5000);
  
  // Initial poll to get started
  pollDeviceMetrics();
  
  // Auto-refresh device metrics on the page every 5 seconds
  setInterval(function() {
    // Get the latest metrics for all devices
    fetch("/devices/metrics")
      .then(function(response) { return response.json(); })
      .then(function(data) {
        if(data.status === "success") {
          data.devices.forEach(function(metric) {
            var row = document.querySelector('tr[data-index="' + metric.index + '"]');
            if (row) {
              row.querySelector(".cpu").textContent = metric.cpu + "%";
              row.querySelector(".mem").textContent = metric.mem + "%";
              row.querySelector(".disk").textContent = metric.disk + "%";
              var statusCell = row.querySelector('td:nth-child(8)');
              if(metric.online) {
                statusCell.innerHTML = '<span style="color:green;">&#9679;</span>';
              } else {
                statusCell.innerHTML = '<span style="color:red;">&#9679;</span>';
              }
            }
          });
        }
      })
      .catch(function(error) {
        console.error("Error refreshing device metrics:", error);
      });
  }, 5000);

  // Open the edit modal and prefill values
  function openEditModal(index, friendlyName, orientation, address) {
    document.getElementById('editDisplayModal').style.display = 'block';
    document.getElementById('editDeviceIndex').value = index;
    document.getElementById('editFriendlyName').value = friendlyName;
    document.getElementById('editOrientation').value = orientation;
    document.getElementById('editAddress').value = address;
  }

  // Open the advanced modal
  function openAdvancedModal(index, friendlyName) {
    document.getElementById('advancedActionsModal').style.display = 'block';
    document.getElementById('advancedDeviceTitle').textContent = "Advanced Actions for " + friendlyName;
    // Store the device index for use in the action functions
    document.getElementById('advancedActionsModal').setAttribute('data-device-index', index);
  }
  
  // Device API functions
  function triggerSystemUpdate() {
    var deviceIndex = document.getElementById('advancedActionsModal').getAttribute('data-device-index');
    var deviceRow = document.querySelector('tr[data-index="' + deviceIndex + '"]');
    var deviceAddress = deviceRow.getAttribute('data-address');
    
    if (!deviceAddress) {
      showOverlayMessage("Error: Device address not found");
      return;
    }
    
    showConfirmOverlay(
      "This will trigger a system update and reboot the device. Continue?",
      function() {
        // Show progress overlay
        var progressOverlay = document.createElement('div');
        progressOverlay.className = 'overlay-popup';
        progressOverlay.innerHTML = `
          <div class="overlay-content">
            <p>Triggering system update...</p>
            <div class="progress-container">
              <div id="updateProgressBar" class="progress-bar" style="width: 0%">0%</div>
            </div>
          </div>
        `;
        document.body.appendChild(progressOverlay);
        
        // Format address properly
        if (!deviceAddress.startsWith("http://") && !deviceAddress.startsWith("https://")) {
          deviceAddress = "http://" + deviceAddress;
        }
        
        // Make the API call
        fetch(deviceAddress + "/system_update", {
          method: 'POST'
        })
        .then(function(response) {
          if (!response.ok) {
            throw new Error("Network response was not ok");
          }
          return response.json();
        })
        .then(function(data) {
          // Simulate progress
          var progress = 0;
          var interval = setInterval(function() {
            progress += 5;
            if (progress >= 100) {
              progress = 100;
              clearInterval(interval);
              
              // Remove progress overlay after completion
              document.body.removeChild(progressOverlay);
              showOverlayMessage("System update triggered successfully. Device will reboot.");
              closeAdvancedActionsModal();
            }
            
            var progressBar = document.getElementById('updateProgressBar');
            if (progressBar) {
              progressBar.style.width = progress + '%';
              progressBar.textContent = progress + '%';
            }
          }, 500);
        })
        .catch(function(error) {
          document.body.removeChild(progressOverlay);
          showOverlayMessage("Error triggering system update: " + error.message);
        });
      }
    );
  }
  
  function triggerBackup() {
    var deviceIndex = document.getElementById('advancedActionsModal').getAttribute('data-device-index');
    var deviceRow = document.querySelector('tr[data-index="' + deviceIndex + '"]');
    var deviceAddress = deviceRow.getAttribute('data-address');
    
    if (!deviceAddress) {
      showOverlayMessage("Error: Device address not found");
      return;
    }
    
    showConfirmOverlay(
      "This will create a backup of the device. This may take several minutes. Continue?",
      function() {
        // Show progress overlay
        var progressOverlay = document.createElement('div');
        progressOverlay.className = 'overlay-popup';
        progressOverlay.innerHTML = `
          <div class="overlay-content">
            <p>Creating backup...</p>
            <div class="progress-container">
              <div id="backupProgressBar" class="progress-bar" style="width: 0%">0%</div>
            </div>
          </div>
        `;
        document.body.appendChild(progressOverlay);
        
        // Format address properly
        if (!deviceAddress.startsWith("http://") && !deviceAddress.startsWith("https://")) {
          deviceAddress = "http://" + deviceAddress;
        }
        
        // Simulate backup progress (in a real app, you'd use a WebSocket or SSE for progress)
        var progress = 0;
        var interval = setInterval(function() {
          progress += 2;
          if (progress >= 100) {
            progress = 100;
            clearInterval(interval);
            
            // Download the backup
            var a = document.createElement('a');
            a.href = deviceAddress + "/backup";
            a.download = "backup_" + new Date().toISOString().replace(/:/g, '-') + ".img.gz";
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            
            // Remove progress overlay after completion
            document.body.removeChild(progressOverlay);
            showOverlayMessage("Backup created successfully. Download started.");
            closeAdvancedActionsModal();
          }
          
          var progressBar = document.getElementById('backupProgressBar');
          if (progressBar) {
            progressBar.style.width = progress + '%';
            progressBar.textContent = progress + '%';
          }
        }, 500);
      }
    );
  }
  
  function triggerAppUpdate() {
    var deviceIndex = document.getElementById('advancedActionsModal').getAttribute('data-device-index');
    var deviceRow = document.querySelector('tr[data-index="' + deviceIndex + '"]');
    var deviceAddress = deviceRow.getAttribute('data-address');
    
    if (!deviceAddress) {
      showOverlayMessage("Error: Device address not found");
      return;
    }
    
    showConfirmOverlay(
      "This will update the application on the device and reboot it. Continue?",
      function() {
        // Show progress overlay
        var progressOverlay = document.createElement('div');
        progressOverlay.className = 'overlay-popup';
        progressOverlay.innerHTML = `
          <div class="overlay-content">
            <p>Updating application...</p>
            <div class="progress-container">
              <div id="appUpdateProgressBar" class="progress-bar" style="width: 0%">0%</div>
            </div>
          </div>
        `;
        document.body.appendChild(progressOverlay);
        
        // Format address properly
        if (!deviceAddress.startsWith("http://") && !deviceAddress.startsWith("https://")) {
          deviceAddress = "http://" + deviceAddress;
        }
        
        // Make the API call
        fetch(deviceAddress + "/update", {
          method: 'POST'
        })
        .then(function(response) {
          if (!response.ok) {
            throw new Error("Network response was not ok");
          }
          return response.json();
        })
        .then(function(data) {
          // Simulate progress
          var progress = 0;
          var interval = setInterval(function() {
            progress += 10;
            if (progress >= 100) {
              progress = 100;
              clearInterval(interval);
              
              // Remove progress overlay after completion
              document.body.removeChild(progressOverlay);
              showOverlayMessage("Application updated successfully. Device will reboot.");
              closeAdvancedActionsModal();
            }
            
            var progressBar = document.getElementById('appUpdateProgressBar');
            if (progressBar) {
              progressBar.style.width = progress + '%';
              progressBar.textContent = progress + '%';
            }
          }, 300);
        })
        .catch(function(error) {
          document.body.removeChild(progressOverlay);
          showOverlayMessage("Error updating application: " + error.message);
        });
      }
    );
  }

  // Automatic submission of Add Display form after fetching info
  document.addEventListener('DOMContentLoaded', function() {
    var addNewDisplayBtn = document.getElementById('addNewDisplayBtn');
    if (addNewDisplayBtn) {
      addNewDisplayBtn.addEventListener('click', function() {
        document.getElementById('addDisplayModal').style.display = 'block';
      });
    }

    var addDisplayForm = document.getElementById('addDisplayForm');
    addDisplayForm.addEventListener('submit', function(e) {
      e.preventDefault();
      fetchDisplayInfo('new').then(function() {
        addDisplayForm.submit();
      }).catch(function() {
        addDisplayForm.submit();
      });
    });

    document.getElementById('closeAddDisplayModal').addEventListener('click', function() {
      closeAddDisplayModal();
    });
    document.getElementById('closeEditDisplayModal').addEventListener('click', function() {
      closeEditDisplayModal();
    });
    document.getElementById('closeAdvancedActionsModal').addEventListener('click', function() {
      closeAdvancedActionsModal();
    });

    // We rely on the backend polling for device metrics instead of direct SSE connections

    // Bind CLIP Settings modal open/close
    var clipSettingsBtn = document.getElementById('clipSettingsBtn');
    var clipSettingsModal = document.getElementById('clipSettingsModal');
    var closeClipSettingsModal = document.getElementById('closeClipSettingsModal');
    if (clipSettingsBtn) {
      clipSettingsBtn.addEventListener('click', function() {
        clipSettingsModal.style.display = 'block';
      });
    }
    if (closeClipSettingsModal) {
      closeClipSettingsModal.addEventListener('click', function() {
        clipSettingsModal.style.display = 'none';
      });
    }
    window.addEventListener('click', function(e) {
      if (e.target == clipSettingsModal) {
        clipSettingsModal.style.display = 'none';
      }
    });
  });

  // Function to save CLIP model setting
  function saveClipModel() {
    var clipModel = document.getElementById('clip_model').value;
    if (!clipModel) {
      showOverlayMessage("Please select a CLIP model");
      return;
    }
    
    var payload = {
      clip_model: clipModel
    };
    
    // Show a loading message
    showOverlayMessage("Switching to model: " + clipModel + "...", 1500);
    
    fetch("{{ url_for('settings.update_clip_model') }}", {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    })
    .then(function(response) { return response.json(); })
    .then(function(data) {
      if (data.status === "success") {
        showOverlayMessage("CLIP model updated successfully to " + clipModel);
      } else {
        showOverlayMessage("Error updating CLIP model: " + data.message);
      }
    })
    .catch(function(error) {
      console.error("Error:", error);
      showOverlayMessage("An error occurred while updating CLIP model.");
    });
  }

  // Function to rerun tagging on all images with the current CLIP model
  function rerunAllTagging() {
    showConfirmOverlay(
      "This will rerun tagging on all images using the selected CLIP model. This may take some time depending on the number of images. Continue?",
      function() {
        fetch("{{ url_for('settings.rerun_all_tagging') }}", {
          method: 'POST'
        })
        .then(function(response) { return response.json(); })
        .then(function(data) {
          if (data.status === "success") {
            showOverlayMessage("Tagging process started! This will run in the background.");
            document.getElementById('clipSettingsModal').style.display = 'none';
          } else {
            showOverlayMessage("Error starting tagging process: " + data.message);
          }
        })
        .catch(function(error) {
          console.error("Error:", error);
          showOverlayMessage("An error occurred while starting the tagging process.");
        });
      }
    );
  }
  
  // Overlay message functions
  function showOverlayMessage(message, duration) {
    var overlay = document.createElement('div');
    overlay.className = 'overlay-popup';
    overlay.innerHTML = `
      <div class="overlay-content">
        <p>${message}</p>
      </div>
    `;
    document.body.appendChild(overlay);
    
    // Remove after duration (default 3 seconds)
    setTimeout(function() {
      document.body.removeChild(overlay);
    }, duration || 3000);
  }
  
  function showConfirmOverlay(message, onConfirm) {
    var overlay = document.createElement('div');
    overlay.className = 'overlay-popup';
    overlay.innerHTML = `
      <div class="overlay-content">
        <p>${message}</p>
        <div class="overlay-buttons">
          <button class="primary-btn confirm-btn">Confirm</button>
          <button class="cancel-btn">Cancel</button>
        </div>
      </div>
    `;
    document.body.appendChild(overlay);
    
    // Add event listeners
    overlay.querySelector('.confirm-btn').addEventListener('click', function() {
      document.body.removeChild(overlay);
      if (onConfirm) onConfirm();
    });
    
    overlay.querySelector('.cancel-btn').addEventListener('click', function() {
      document.body.removeChild(overlay);
    });
  }

  // Function to fetch display info from an eInk device via our proxy endpoint.
  // Expected JSON from /device_info: {"status": "ok", "info": {"display_name": "...", "resolution": "..." }}
  function fetchDisplayInfo(mode) {
    return new Promise(function(resolve, reject) {
      if (mode === 'new') {
        var addressInput = document.getElementById('newAddress');
        var address = addressInput.value.trim();
        if (!address) {
          alert("Please enter the device address.");
          reject("No address provided");
          return;
        }
        // Call our proxy endpoint to avoid CORS issues
        fetch("/device_info?address=" + encodeURIComponent(address), { timeout: 5000 })
          .then(function(response) {
            if (!response.ok) {
              throw new Error("HTTP error " + response.status);
            }
            return response.json();
          })
          .then(function(data) {
            if (data.status === "ok") {
              document.getElementById('newDisplayName').value = data.info.display_name;
              document.getElementById('newResolution').value = data.info.resolution;
              var availableColors = ['#FF5733', '#33FF57', '#3357FF', '#F39C12', '#8E44AD', '#2ECC71', '#E74C3C'];
              var randomColor = availableColors[Math.floor(Math.random() * availableColors.length)];
              document.getElementById('newColor').value = randomColor;
              resolve();
            } else {
              alert("Error fetching display info: " + data.message + "; using default values.");
              document.getElementById('newDisplayName').value = "DefaultDisplay color";
              document.getElementById('newResolution').value = "800x600";
              document.getElementById('newColor').value = "#FF5733";
              resolve();
            }
          })
          .catch(function(error) {
            console.error("Error fetching display info:", error);
            alert("Error fetching display info; using default values.");
            document.getElementById('newDisplayName').value = "DefaultDisplay color";
            document.getElementById('newResolution').value = "800x600";
            document.getElementById('newColor').value = "#FF5733";
            resolve();
          });
      } else if (mode === 'edit') {
        alert("Fetch Display Info for edit is not implemented yet.");
        resolve();
      }
    });
  }
</script>
{% endblock %}
```


## utils/__init__.py

```py

```


## utils/crop_helpers.py

```py
from models import CropInfo, SendLog, db

def load_crop_info_from_db(filename):
    c = CropInfo.query.filter_by(filename=filename).first()
    if not c:
        return None
    return {"x": c.x, "y": c.y, "width": c.width, "height": c.height}

def save_crop_info_to_db(filename, crop_data):
    c = CropInfo.query.filter_by(filename=filename).first()
    if not c:
        c = CropInfo(filename=filename)
        db.session.add(c)
    c.x = crop_data.get("x", 0)
    c.y = crop_data.get("y", 0)
    c.width = crop_data.get("width", 0)
    c.height = crop_data.get("height", 0)
    db.session.commit()

def add_send_log_entry(filename):
    entry = SendLog(filename=filename)
    db.session.add(entry)
    db.session.commit()

def get_last_sent():
    latest = SendLog.query.order_by(SendLog.id.desc()).first()
    return latest.filename if latest else None
```


## utils/image_helpers.py

```py
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
```


## .DS_Store

```
   Bud1           	                                                           i cbwspblob                                                                                                                                                                                                                                                                                                                                                                                                                                           s t a t i cbwspblob   �bplist00�]ShowStatusBar[ShowToolbar[ShowTabView_ContainerShowSidebar\WindowBounds[ShowSidebar		_{{188, 391}, {920, 436}}	#/;R_klmno�                            �    s t a t i cfdscbool     s t a t i cvSrnlong                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            @      �                                        @      �                                          @      �                                          @                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   E  	                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       DSDB                                 `          �                                         @      �                                          @      �                                          @                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              
```


## .export-ignore

```
node_modules/
*.log
dist/
.vscode/
```


## .gitattributes

```
# Auto detect text files and perform LF normalization
* text=auto
```


## app.py

```py
from flask import Flask
import os
from config import Config
from models import db
import pillow_heif
from tasks import celery, start_scheduler

# Register HEIF opener for Pillow
pillow_heif.register_heif_opener()

def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)

    # Ensure required folders exist
    for folder in [app.config['IMAGE_FOLDER'], app.config['THUMBNAIL_FOLDER'], app.config['DATA_FOLDER']]:
        if not os.path.exists(folder):
            os.makedirs(folder)

    # Initialize database without migrations
    db.init_app(app)

    # Register blueprints
    from routes.image_routes import image_bp
    from routes.device_routes import device_bp
    from routes.schedule_routes import schedule_bp
    from routes.settings_routes import settings_bp
    from routes.device_info_routes import device_info_bp
    from routes.ai_tagging_routes import ai_bp

    app.register_blueprint(image_bp)
    app.register_blueprint(device_bp)
    app.register_blueprint(schedule_bp)
    app.register_blueprint(settings_bp)
    app.register_blueprint(device_info_bp)
    app.register_blueprint(ai_bp)

    # Create database tables if they don't exist.
    with app.app_context():
        db.create_all()

    # Configure Celery
    celery.conf.update(
        broker_url='redis://localhost:6379/0',
        result_backend='redis://localhost:6379/0'
    )

    # Start the scheduler for scheduled tasks
    start_scheduler(app)
    
    # Run the fetch_device_metrics task immediately to get initial metrics
    with app.app_context():
        from tasks import fetch_device_metrics
        fetch_device_metrics.delay()

    return app

app = create_app()

# Make the app available to Celery tasks
celery.conf.update(app=app)

if __name__ == '__main__':
    # When running via 'python app.py' this block will execute.
    app.run(host='0.0.0.0', port=5001, debug=True)
```


## config.py

```py
import os

basedir = os.path.abspath(os.path.dirname(__file__))

class Config:
    SECRET_KEY = "super-secret-key"
    # Database: using an absolute path in a “data” folder in the project directory.
    # In the container, basedir will be /app so the DB will be at /app/data/mydb.sqlite.
    SQLALCHEMY_DATABASE_URI = 'sqlite:///' + os.path.join(basedir, 'data', 'mydb.sqlite')
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # Folders for images, thumbnails, and data storage
    IMAGE_FOLDER = os.path.join(basedir, 'images')
    THUMBNAIL_FOLDER = os.path.join(basedir, 'images', 'thumbnails')
    DATA_FOLDER = os.path.join(basedir, 'data')
```


## Dockerfile

```
# Use an official Python image
FROM python:3.13.2-slim

# Set timezone and cache directory for models (persisted in /data/model_cache)
ENV TZ=Europe/Copenhagen
ENV XDG_CACHE_HOME=/app/data/model_cache

# Install system dependencies and redis-server
RUN apt-get update && apt-get install -y \
    curl \
    sqlite3 \
    supervisor \
    tzdata \
    build-essential \
    gcc \
    git \
    libffi-dev \
    libssl-dev \
    libjpeg-dev \
    zlib1g-dev \
    libpng-dev \
    libfreetype6-dev \
    redis-server \
    && ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone \
    && rm -rf /var/lib/apt/lists/*

# Create directory for the database if needed
RUN mkdir -p /data

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files into the container
COPY . .

# Pre-download all CLIP models to ensure they're available
RUN mkdir -p /app/data/model_cache && \
    python -c "import open_clip; \
    model1, _, _ = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai', jit=False, force_quick_gelu=True); \
    model2, _, _ = open_clip.create_model_and_transforms('ViT-B-16', pretrained='openai', jit=False, force_quick_gelu=True); \
    model3, _, _ = open_clip.create_model_and_transforms('ViT-L-14', pretrained='openai', jit=False, force_quick_gelu=True)"

# Increase memory limit for Celery workers to prevent SIGKILL
ENV CELERY_WORKER_MAX_MEMORY_PER_CHILD=500000
ENV CELERY_WORKER_MAX_TASKS_PER_CHILD=1
# Reduce the number of Celery workers to prevent memory issues
ENV CELERY_WORKERS=2

# Expose port 5001
EXPOSE 5001

# Copy entrypoint script and make it executable
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Copy Supervisor configuration file
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Tell Flask which file is our app
ENV FLASK_APP=app.py

# Run entrypoint.sh (which handles migrations and launches Supervisor)
CMD ["/entrypoint.sh"]
```


## entrypoint.sh

```sh
#!/bin/sh
# entrypoint.sh - Auto-create the database tables then launch the app via Supervisor.

echo "Ensuring /app/data folder exists..."
mkdir -p /app/data

echo "Skipping migration steps. Database tables will be created at runtime if missing."

echo "Starting Supervisor..."
exec /usr/bin/supervisord -c /etc/supervisor/conf.d/supervisord.conf
```


## exportconfig.json

```json
{
    "output": "export.md",
    "description": "",
    "ignoreFile": ".export-ignore",
    "includeFile": ".export-whitelist",
    "maxFileSize": 1048576,
    "removeComments": true,
    "includeProjectStructure": true
  }
```


## LICENSE

```
GNU GENERAL PUBLIC LICENSE
                       Version 3, 29 June 2007

 Copyright (C) 2007 Free Software Foundation, Inc. <https://fsf.org/>
 Everyone is permitted to copy and distribute verbatim copies
 of this license document, but changing it is not allowed.

                            Preamble

  The GNU General Public License is a free, copyleft license for
software and other kinds of works.

  The licenses for most software and other practical works are designed
to take away your freedom to share and change the works.  By contrast,
the GNU General Public License is intended to guarantee your freedom to
share and change all versions of a program--to make sure it remains free
software for all its users.  We, the Free Software Foundation, use the
GNU General Public License for most of our software; it applies also to
any other work released this way by its authors.  You can apply it to
your programs, too.

  When we speak of free software, we are referring to freedom, not
price.  Our General Public Licenses are designed to make sure that you
have the freedom to distribute copies of free software (and charge for
them if you wish), that you receive source code or can get it if you
want it, that you can change the software or use pieces of it in new
free programs, and that you know you can do these things.

  To protect your rights, we need to prevent others from denying you
these rights or asking you to surrender the rights.  Therefore, you have
certain responsibilities if you distribute copies of the software, or if
you modify it: responsibilities to respect the freedom of others.

  For example, if you distribute copies of such a program, whether
gratis or for a fee, you must pass on to the recipients the same
freedoms that you received.  You must make sure that they, too, receive
or can get the source code.  And you must show them these terms so they
know their rights.

  Developers that use the GNU GPL protect your rights with two steps:
(1) assert copyright on the software, and (2) offer you this License
giving you legal permission to copy, distribute and/or modify it.

  For the developers' and authors' protection, the GPL clearly explains
that there is no warranty for this free software.  For both users' and
authors' sake, the GPL requires that modified versions be marked as
changed, so that their problems will not be attributed erroneously to
authors of previous versions.

  Some devices are designed to deny users access to install or run
modified versions of the software inside them, although the manufacturer
can do so.  This is fundamentally incompatible with the aim of
protecting users' freedom to change the software.  The systematic
pattern of such abuse occurs in the area of products for individuals to
use, which is precisely where it is most unacceptable.  Therefore, we
have designed this version of the GPL to prohibit the practice for those
products.  If such problems arise substantially in other domains, we
stand ready to extend this provision to those domains in future versions
of the GPL, as needed to protect the freedom of users.

  Finally, every program is threatened constantly by software patents.
States should not allow patents to restrict development and use of
software on general-purpose computers, but in those that do, we wish to
avoid the special danger that patents applied to a free program could
make it effectively proprietary.  To prevent this, the GPL assures that
patents cannot be used to render the program non-free.

  The precise terms and conditions for copying, distribution and
modification follow.

                       TERMS AND CONDITIONS

  0. Definitions.

  "This License" refers to version 3 of the GNU General Public License.

  "Copyright" also means copyright-like laws that apply to other kinds of
works, such as semiconductor masks.

  "The Program" refers to any copyrightable work licensed under this
License.  Each licensee is addressed as "you".  "Licensees" and
"recipients" may be individuals or organizations.

  To "modify" a work means to copy from or adapt all or part of the work
in a fashion requiring copyright permission, other than the making of an
exact copy.  The resulting work is called a "modified version" of the
earlier work or a work "based on" the earlier work.

  A "covered work" means either the unmodified Program or a work based
on the Program.

  To "propagate" a work means to do anything with it that, without
permission, would make you directly or secondarily liable for
infringement under applicable copyright law, except executing it on a
computer or modifying a private copy.  Propagation includes copying,
distribution (with or without modification), making available to the
public, and in some countries other activities as well.

  To "convey" a work means any kind of propagation that enables other
parties to make or receive copies.  Mere interaction with a user through
a computer network, with no transfer of a copy, is not conveying.

  An interactive user interface displays "Appropriate Legal Notices"
to the extent that it includes a convenient and prominently visible
feature that (1) displays an appropriate copyright notice, and (2)
tells the user that there is no warranty for the work (except to the
extent that warranties are provided), that licensees may convey the
work under this License, and how to view a copy of this License.  If
the interface presents a list of user commands or options, such as a
menu, a prominent item in the list meets this criterion.

  1. Source Code.

  The "source code" for a work means the preferred form of the work
for making modifications to it.  "Object code" means any non-source
form of a work.

  A "Standard Interface" means an interface that either is an official
standard defined by a recognized standards body, or, in the case of
interfaces specified for a particular programming language, one that
is widely used among developers working in that language.

  The "System Libraries" of an executable work include anything, other
than the work as a whole, that (a) is included in the normal form of
packaging a Major Component, but which is not part of that Major
Component, and (b) serves only to enable use of the work with that
Major Component, or to implement a Standard Interface for which an
implementation is available to the public in source code form.  A
"Major Component", in this context, means a major essential component
(kernel, window system, and so on) of the specific operating system
(if any) on which the executable work runs, or a compiler used to
produce the work, or an object code interpreter used to run it.

  The "Corresponding Source" for a work in object code form means all
the source code needed to generate, install, and (for an executable
work) run the object code and to modify the work, including scripts to
control those activities.  However, it does not include the work's
System Libraries, or general-purpose tools or generally available free
programs which are used unmodified in performing those activities but
which are not part of the work.  For example, Corresponding Source
includes interface definition files associated with source files for
the work, and the source code for shared libraries and dynamically
linked subprograms that the work is specifically designed to require,
such as by intimate data communication or control flow between those
subprograms and other parts of the work.

  The Corresponding Source need not include anything that users
can regenerate automatically from other parts of the Corresponding
Source.

  The Corresponding Source for a work in source code form is that
same work.

  2. Basic Permissions.

  All rights granted under this License are granted for the term of
copyright on the Program, and are irrevocable provided the stated
conditions are met.  This License explicitly affirms your unlimited
permission to run the unmodified Program.  The output from running a
covered work is covered by this License only if the output, given its
content, constitutes a covered work.  This License acknowledges your
rights of fair use or other equivalent, as provided by copyright law.

  You may make, run and propagate covered works that you do not
convey, without conditions so long as your license otherwise remains
in force.  You may convey covered works to others for the sole purpose
of having them make modifications exclusively for you, or provide you
with facilities for running those works, provided that you comply with
the terms of this License in conveying all material for which you do
not control copyright.  Those thus making or running the covered works
for you must do so exclusively on your behalf, under your direction
and control, on terms that prohibit them from making any copies of
your copyrighted material outside their relationship with you.

  Conveying under any other circumstances is permitted solely under
the conditions stated below.  Sublicensing is not allowed; section 10
makes it unnecessary.

  3. Protecting Users' Legal Rights From Anti-Circumvention Law.

  No covered work shall be deemed part of an effective technological
measure under any applicable law fulfilling obligations under article
11 of the WIPO copyright treaty adopted on 20 December 1996, or
similar laws prohibiting or restricting circumvention of such
measures.

  When you convey a covered work, you waive any legal power to forbid
circumvention of technological measures to the extent such circumvention
is effected by exercising rights under this License with respect to
the covered work, and you disclaim any intention to limit operation or
modification of the work as a means of enforcing, against the work's
users, your or third parties' legal rights to forbid circumvention of
technological measures.

  4. Conveying Verbatim Copies.

  You may convey verbatim copies of the Program's source code as you
receive it, in any medium, provided that you conspicuously and
appropriately publish on each copy an appropriate copyright notice;
keep intact all notices stating that this License and any
non-permissive terms added in accord with section 7 apply to the code;
keep intact all notices of the absence of any warranty; and give all
recipients a copy of this License along with the Program.

  You may charge any price or no price for each copy that you convey,
and you may offer support or warranty protection for a fee.

  5. Conveying Modified Source Versions.

  You may convey a work based on the Program, or the modifications to
produce it from the Program, in the form of source code under the
terms of section 4, provided that you also meet all of these conditions:

    a) The work must carry prominent notices stating that you modified
    it, and giving a relevant date.

    b) The work must carry prominent notices stating that it is
    released under this License and any conditions added under section
    7.  This requirement modifies the requirement in section 4 to
    "keep intact all notices".

    c) You must license the entire work, as a whole, under this
    License to anyone who comes into possession of a copy.  This
    License will therefore apply, along with any applicable section 7
    additional terms, to the whole of the work, and all its parts,
    regardless of how they are packaged.  This License gives no
    permission to license the work in any other way, but it does not
    invalidate such permission if you have separately received it.

    d) If the work has interactive user interfaces, each must display
    Appropriate Legal Notices; however, if the Program has interactive
    interfaces that do not display Appropriate Legal Notices, your
    work need not make them do so.

  A compilation of a covered work with other separate and independent
works, which are not by their nature extensions of the covered work,
and which are not combined with it such as to form a larger program,
in or on a volume of a storage or distribution medium, is called an
"aggregate" if the compilation and its resulting copyright are not
used to limit the access or legal rights of the compilation's users
beyond what the individual works permit.  Inclusion of a covered work
in an aggregate does not cause this License to apply to the other
parts of the aggregate.

  6. Conveying Non-Source Forms.

  You may convey a covered work in object code form under the terms
of sections 4 and 5, provided that you also convey the
machine-readable Corresponding Source under the terms of this License,
in one of these ways:

    a) Convey the object code in, or embodied in, a physical product
    (including a physical distribution medium), accompanied by the
    Corresponding Source fixed on a durable physical medium
    customarily used for software interchange.

    b) Convey the object code in, or embodied in, a physical product
    (including a physical distribution medium), accompanied by a
    written offer, valid for at least three years and valid for as
    long as you offer spare parts or customer support for that product
    model, to give anyone who possesses the object code either (1) a
    copy of the Corresponding Source for all the software in the
    product that is covered by this License, on a durable physical
    medium customarily used for software interchange, for a price no
    more than your reasonable cost of physically performing this
    conveying of source, or (2) access to copy the
    Corresponding Source from a network server at no charge.

    c) Convey individual copies of the object code with a copy of the
    written offer to provide the Corresponding Source.  This
    alternative is allowed only occasionally and noncommercially, and
    only if you received the object code with such an offer, in accord
    with subsection 6b.

    d) Convey the object code by offering access from a designated
    place (gratis or for a charge), and offer equivalent access to the
    Corresponding Source in the same way through the same place at no
    further charge.  You need not require recipients to copy the
    Corresponding Source along with the object code.  If the place to
    copy the object code is a network server, the Corresponding Source
    may be on a different server (operated by you or a third party)
    that supports equivalent copying facilities, provided you maintain
    clear directions next to the object code saying where to find the
    Corresponding Source.  Regardless of what server hosts the
    Corresponding Source, you remain obligated to ensure that it is
    available for as long as needed to satisfy these requirements.

    e) Convey the object code using peer-to-peer transmission, provided
    you inform other peers where the object code and Corresponding
    Source of the work are being offered to the general public at no
    charge under subsection 6d.

  A separable portion of the object code, whose source code is excluded
from the Corresponding Source as a System Library, need not be
included in conveying the object code work.

  A "User Product" is either (1) a "consumer product", which means any
tangible personal property which is normally used for personal, family,
or household purposes, or (2) anything designed or sold for incorporation
into a dwelling.  In determining whether a product is a consumer product,
doubtful cases shall be resolved in favor of coverage.  For a particular
product received by a particular user, "normally used" refers to a
typical or common use of that class of product, regardless of the status
of the particular user or of the way in which the particular user
actually uses, or expects or is expected to use, the product.  A product
is a consumer product regardless of whether the product has substantial
commercial, industrial or non-consumer uses, unless such uses represent
the only significant mode of use of the product.

  "Installation Information" for a User Product means any methods,
procedures, authorization keys, or other information required to install
and execute modified versions of a covered work in that User Product from
a modified version of its Corresponding Source.  The information must
suffice to ensure that the continued functioning of the modified object
code is in no case prevented or interfered with solely because
modification has been made.

  If you convey an object code work under this section in, or with, or
specifically for use in, a User Product, and the conveying occurs as
part of a transaction in which the right of possession and use of the
User Product is transferred to the recipient in perpetuity or for a
fixed term (regardless of how the transaction is characterized), the
Corresponding Source conveyed under this section must be accompanied
by the Installation Information.  But this requirement does not apply
if neither you nor any third party retains the ability to install
modified object code on the User Product (for example, the work has
been installed in ROM).

  The requirement to provide Installation Information does not include a
requirement to continue to provide support service, warranty, or updates
for a work that has been modified or installed by the recipient, or for
the User Product in which it has been modified or installed.  Access to a
network may be denied when the modification itself materially and
adversely affects the operation of the network or violates the rules and
protocols for communication across the network.

  Corresponding Source conveyed, and Installation Information provided,
in accord with this section must be in a format that is publicly
documented (and with an implementation available to the public in
source code form), and must require no special password or key for
unpacking, reading or copying.

  7. Additional Terms.

  "Additional permissions" are terms that supplement the terms of this
License by making exceptions from one or more of its conditions.
Additional permissions that are applicable to the entire Program shall
be treated as though they were included in this License, to the extent
that they are valid under applicable law.  If additional permissions
apply only to part of the Program, that part may be used separately
under those permissions, but the entire Program remains governed by
this License without regard to the additional permissions.

  When you convey a copy of a covered work, you may at your option
remove any additional permissions from that copy, or from any part of
it.  (Additional permissions may be written to require their own
removal in certain cases when you modify the work.)  You may place
additional permissions on material, added by you to a covered work,
for which you have or can give appropriate copyright permission.

  Notwithstanding any other provision of this License, for material you
add to a covered work, you may (if authorized by the copyright holders of
that material) supplement the terms of this License with terms:

    a) Disclaiming warranty or limiting liability differently from the
    terms of sections 15 and 16 of this License; or

    b) Requiring preservation of specified reasonable legal notices or
    author attributions in that material or in the Appropriate Legal
    Notices displayed by works containing it; or

    c) Prohibiting misrepresentation of the origin of that material, or
    requiring that modified versions of such material be marked in
    reasonable ways as different from the original version; or

    d) Limiting the use for publicity purposes of names of licensors or
    authors of the material; or

    e) Declining to grant rights under trademark law for use of some
    trade names, trademarks, or service marks; or

    f) Requiring indemnification of licensors and authors of that
    material by anyone who conveys the material (or modified versions of
    it) with contractual assumptions of liability to the recipient, for
    any liability that these contractual assumptions directly impose on
    those licensors and authors.

  All other non-permissive additional terms are considered "further
restrictions" within the meaning of section 10.  If the Program as you
received it, or any part of it, contains a notice stating that it is
governed by this License along with a term that is a further
restriction, you may remove that term.  If a license document contains
a further restriction but permits relicensing or conveying under this
License, you may add to a covered work material governed by the terms
of that license document, provided that the further restriction does
not survive such relicensing or conveying.

  If you add terms to a covered work in accord with this section, you
must place, in the relevant source files, a statement of the
additional terms that apply to those files, or a notice indicating
where to find the applicable terms.

  Additional terms, permissive or non-permissive, may be stated in the
form of a separately written license, or stated as exceptions;
the above requirements apply either way.

  8. Termination.

  You may not propagate or modify a covered work except as expressly
provided under this License.  Any attempt otherwise to propagate or
modify it is void, and will automatically terminate your rights under
this License (including any patent licenses granted under the third
paragraph of section 11).

  However, if you cease all violation of this License, then your
license from a particular copyright holder is reinstated (a)
provisionally, unless and until the copyright holder explicitly and
finally terminates your license, and (b) permanently, if the copyright
holder fails to notify you of the violation by some reasonable means
prior to 60 days after the cessation.

  Moreover, your license from a particular copyright holder is
reinstated permanently if the copyright holder notifies you of the
violation by some reasonable means, this is the first time you have
received notice of violation of this License (for any work) from that
copyright holder, and you cure the violation prior to 30 days after
your receipt of the notice.

  Termination of your rights under this section does not terminate the
licenses of parties who have received copies or rights from you under
this License.  If your rights have been terminated and not permanently
reinstated, you do not qualify to receive new licenses for the same
material under section 10.

  9. Acceptance Not Required for Having Copies.

  You are not required to accept this License in order to receive or
run a copy of the Program.  Ancillary propagation of a covered work
occurring solely as a consequence of using peer-to-peer transmission
to receive a copy likewise does not require acceptance.  However,
nothing other than this License grants you permission to propagate or
modify any covered work.  These actions infringe copyright if you do
not accept this License.  Therefore, by modifying or propagating a
covered work, you indicate your acceptance of this License to do so.

  10. Automatic Licensing of Downstream Recipients.

  Each time you convey a covered work, the recipient automatically
receives a license from the original licensors, to run, modify and
propagate that work, subject to this License.  You are not responsible
for enforcing compliance by third parties with this License.

  An "entity transaction" is a transaction transferring control of an
organization, or substantially all assets of one, or subdividing an
organization, or merging organizations.  If propagation of a covered
work results from an entity transaction, each party to that
transaction who receives a copy of the work also receives whatever
licenses to the work the party's predecessor in interest had or could
give under the previous paragraph, plus a right to possession of the
Corresponding Source of the work from the predecessor in interest, if
the predecessor has it or can get it with reasonable efforts.

  You may not impose any further restrictions on the exercise of the
rights granted or affirmed under this License.  For example, you may
not impose a license fee, royalty, or other charge for exercise of
rights granted under this License, and you may not initiate litigation
(including a cross-claim or counterclaim in a lawsuit) alleging that
any patent claim is infringed by making, using, selling, offering for
sale, or importing the Program or any portion of it.

  11. Patents.

  A "contributor" is a copyright holder who authorizes use under this
License of the Program or a work on which the Program is based.  The
work thus licensed is called the contributor's "contributor version".

  A contributor's "essential patent claims" are all patent claims
owned or controlled by the contributor, whether already acquired or
hereafter acquired, that would be infringed by some manner, permitted
by this License, of making, using, or selling its contributor version,
but do not include claims that would be infringed only as a
consequence of further modification of the contributor version.  For
purposes of this definition, "control" includes the right to grant
patent sublicenses in a manner consistent with the requirements of
this License.

  Each contributor grants you a non-exclusive, worldwide, royalty-free
patent license under the contributor's essential patent claims, to
make, use, sell, offer for sale, import and otherwise run, modify and
propagate the contents of its contributor version.

  In the following three paragraphs, a "patent license" is any express
agreement or commitment, however denominated, not to enforce a patent
(such as an express permission to practice a patent or covenant not to
sue for patent infringement).  To "grant" such a patent license to a
party means to make such an agreement or commitment not to enforce a
patent against the party.

  If you convey a covered work, knowingly relying on a patent license,
and the Corresponding Source of the work is not available for anyone
to copy, free of charge and under the terms of this License, through a
publicly available network server or other readily accessible means,
then you must either (1) cause the Corresponding Source to be so
available, or (2) arrange to deprive yourself of the benefit of the
patent license for this particular work, or (3) arrange, in a manner
consistent with the requirements of this License, to extend the patent
license to downstream recipients.  "Knowingly relying" means you have
actual knowledge that, but for the patent license, your conveying the
covered work in a country, or your recipient's use of the covered work
in a country, would infringe one or more identifiable patents in that
country that you have reason to believe are valid.

  If, pursuant to or in connection with a single transaction or
arrangement, you convey, or propagate by procuring conveyance of, a
covered work, and grant a patent license to some of the parties
receiving the covered work authorizing them to use, propagate, modify
or convey a specific copy of the covered work, then the patent license
you grant is automatically extended to all recipients of the covered
work and works based on it.

  A patent license is "discriminatory" if it does not include within
the scope of its coverage, prohibits the exercise of, or is
conditioned on the non-exercise of one or more of the rights that are
specifically granted under this License.  You may not convey a covered
work if you are a party to an arrangement with a third party that is
in the business of distributing software, under which you make payment
to the third party based on the extent of your activity of conveying
the work, and under which the third party grants, to any of the
parties who would receive the covered work from you, a discriminatory
patent license (a) in connection with copies of the covered work
conveyed by you (or copies made from those copies), or (b) primarily
for and in connection with specific products or compilations that
contain the covered work, unless you entered into that arrangement,
or that patent license was granted, prior to 28 March 2007.

  Nothing in this License shall be construed as excluding or limiting
any implied license or other defenses to infringement that may
otherwise be available to you under applicable patent law.

  12. No Surrender of Others' Freedom.

  If conditions are imposed on you (whether by court order, agreement or
otherwise) that contradict the conditions of this License, they do not
excuse you from the conditions of this License.  If you cannot convey a
covered work so as to satisfy simultaneously your obligations under this
License and any other pertinent obligations, then as a consequence you may
not convey it at all.  For example, if you agree to terms that obligate you
to collect a royalty for further conveying from those to whom you convey
the Program, the only way you could satisfy both those terms and this
License would be to refrain entirely from conveying the Program.

  13. Use with the GNU Affero General Public License.

  Notwithstanding any other provision of this License, you have
permission to link or combine any covered work with a work licensed
under version 3 of the GNU Affero General Public License into a single
combined work, and to convey the resulting work.  The terms of this
License will continue to apply to the part which is the covered work,
but the special requirements of the GNU Affero General Public License,
section 13, concerning interaction through a network will apply to the
combination as such.

  14. Revised Versions of this License.

  The Free Software Foundation may publish revised and/or new versions of
the GNU General Public License from time to time.  Such new versions will
be similar in spirit to the present version, but may differ in detail to
address new problems or concerns.

  Each version is given a distinguishing version number.  If the
Program specifies that a certain numbered version of the GNU General
Public License "or any later version" applies to it, you have the
option of following the terms and conditions either of that numbered
version or of any later version published by the Free Software
Foundation.  If the Program does not specify a version number of the
GNU General Public License, you may choose any version ever published
by the Free Software Foundation.

  If the Program specifies that a proxy can decide which future
versions of the GNU General Public License can be used, that proxy's
public statement of acceptance of a version permanently authorizes you
to choose that version for the Program.

  Later license versions may give you additional or different
permissions.  However, no additional obligations are imposed on any
author or copyright holder as a result of your choosing to follow a
later version.

  15. Disclaimer of Warranty.

  THERE IS NO WARRANTY FOR THE PROGRAM, TO THE EXTENT PERMITTED BY
APPLICABLE LAW.  EXCEPT WHEN OTHERWISE STATED IN WRITING THE COPYRIGHT
HOLDERS AND/OR OTHER PARTIES PROVIDE THE PROGRAM "AS IS" WITHOUT WARRANTY
OF ANY KIND, EITHER EXPRESSED OR IMPLIED, INCLUDING, BUT NOT LIMITED TO,
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
PURPOSE.  THE ENTIRE RISK AS TO THE QUALITY AND PERFORMANCE OF THE PROGRAM
IS WITH YOU.  SHOULD THE PROGRAM PROVE DEFECTIVE, YOU ASSUME THE COST OF
ALL NECESSARY SERVICING, REPAIR OR CORRECTION.

  16. Limitation of Liability.

  IN NO EVENT UNLESS REQUIRED BY APPLICABLE LAW OR AGREED TO IN WRITING
WILL ANY COPYRIGHT HOLDER, OR ANY OTHER PARTY WHO MODIFIES AND/OR CONVEYS
THE PROGRAM AS PERMITTED ABOVE, BE LIABLE TO YOU FOR DAMAGES, INCLUDING ANY
GENERAL, SPECIAL, INCIDENTAL OR CONSEQUENTIAL DAMAGES ARISING OUT OF THE
USE OR INABILITY TO USE THE PROGRAM (INCLUDING BUT NOT LIMITED TO LOSS OF
DATA OR DATA BEING RENDERED INACCURATE OR LOSSES SUSTAINED BY YOU OR THIRD
PARTIES OR A FAILURE OF THE PROGRAM TO OPERATE WITH ANY OTHER PROGRAMS),
EVEN IF SUCH HOLDER OR OTHER PARTY HAS BEEN ADVISED OF THE POSSIBILITY OF
SUCH DAMAGES.

  17. Interpretation of Sections 15 and 16.

  If the disclaimer of warranty and limitation of liability provided
above cannot be given local legal effect according to their terms,
reviewing courts shall apply local law that most closely approximates
an absolute waiver of all civil liability in connection with the
Program, unless a warranty or assumption of liability accompanies a
copy of the Program in return for a fee.

                     END OF TERMS AND CONDITIONS

            How to Apply These Terms to Your New Programs

  If you develop a new program, and you want it to be of the greatest
possible use to the public, the best way to achieve this is to make it
free software which everyone can redistribute and change under these terms.

  To do so, attach the following notices to the program.  It is safest
to attach them to the start of each source file to most effectively
state the exclusion of warranty; and each file should have at least
the "copyright" line and a pointer to where the full notice is found.

    <one line to give the program's name and a brief idea of what it does.>
    Copyright (C) <year>  <name of author>

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.

Also add information on how to contact you by electronic and paper mail.

  If the program does terminal interaction, make it output a short
notice like this when it starts in an interactive mode:

    <program>  Copyright (C) <year>  <name of author>
    This program comes with ABSOLUTELY NO WARRANTY; for details type `show w'.
    This is free software, and you are welcome to redistribute it
    under certain conditions; type `show c' for details.

The hypothetical commands `show w' and `show c' should show the appropriate
parts of the General Public License.  Of course, your program's commands
might be different; for a GUI interface, you would use an "about box".

  You should also get your employer (if you work as a programmer) or school,
if any, to sign a "copyright disclaimer" for the program, if necessary.
For more information on this, and how to apply and follow the GNU GPL, see
<https://www.gnu.org/licenses/>.

  The GNU General Public License does not permit incorporating your program
into proprietary programs.  If your program is a subroutine library, you
may consider it more useful to permit linking proprietary applications with
the library.  If this is what you want to do, use the GNU Lesser General
Public License instead of this License.  But first, please read
<https://www.gnu.org/licenses/why-not-lgpl.html>.
```


## models.py

```py
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()

class Device(db.Model):
    __tablename__ = 'devices'
    id = db.Column(db.Integer, primary_key=True)
    color = db.Column(db.String(16), nullable=False)
    friendly_name = db.Column(db.String(128), nullable=False)
    orientation = db.Column(db.String(32), nullable=False)
    address = db.Column(db.String(256), nullable=False)
    display_name = db.Column(db.String(128))
    resolution = db.Column(db.String(32))
    online = db.Column(db.Boolean, default=False)
    cpu_usage = db.Column(db.String(16), default="N/A")
    mem_usage = db.Column(db.String(16), default="N/A")
    disk_usage = db.Column(db.String(16), default="N/A")
    last_sent = db.Column(db.String(256))

    def __repr__(self):
        return f"<Device {self.friendly_name} ({self.address})>"

class ImageDB(db.Model):
    __tablename__ = 'images'
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(256), unique=True, nullable=False)
    tags = db.Column(db.String(512), nullable=True)         # comma-separated tags
    description = db.Column(db.Text, nullable=True)           # description text
    favorite = db.Column(db.Boolean, default=False)           # favorite flag

    def __repr__(self):
        return f"<ImageDB {self.filename}>"

class CropInfo(db.Model):
    __tablename__ = 'crop_info'
    filename = db.Column(db.String(256), primary_key=True)
    x = db.Column(db.Float, default=0)
    y = db.Column(db.Float, default=0)
    width = db.Column(db.Float)
    height = db.Column(db.Float)

    def __repr__(self):
        return f"<CropInfo {self.filename}>"

class SendLog(db.Model):
    __tablename__ = 'send_log'
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(256), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<SendLog {self.filename} {self.timestamp}>"

class ScheduleEvent(db.Model):
    __tablename__ = 'schedule_events'
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(256), nullable=False)
    device = db.Column(db.String(256), nullable=False)
    datetime_str = db.Column(db.String(32))
    sent = db.Column(db.Boolean, default=False)
    recurrence = db.Column(db.String(20), default="none")  # Recurrence type

    def __repr__(self):
        return f"<ScheduleEvent {self.filename} on {self.device}>"

class UserConfig(db.Model):
    __tablename__ = 'user_config'
    id = db.Column(db.Integer, primary_key=True)
    location = db.Column(db.String(256))
    openai_api_key = db.Column(db.String(512))
    ollama_address = db.Column(db.String(256))
    ollama_api_key = db.Column(db.String(512))
    ollama_model = db.Column(db.String(64))  # Column for chosen Ollama model
    clip_model = db.Column(db.String(64), default="ViT-B/32")  # Column for chosen CLIP model

    def __repr__(self):
        return f"<UserConfig {self.id} - Location: {self.location}>"

class DeviceMetrics(db.Model):
    __tablename__ = 'device_metrics'
    id = db.Column(db.Integer, primary_key=True)
    device_id = db.Column(db.Integer, db.ForeignKey('devices.id'), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    cpu = db.Column(db.Float)
    memory = db.Column(db.Float)
    disk = db.Column(db.Float)

    def __repr__(self):
        return (f"<DeviceMetrics device_id={self.device_id} at {self.timestamp}: "
                f"cpu={self.cpu}, mem={self.memory}, disk={self.disk}>")
```


## package.json

```json
{
    "name": "inkydocker",
    "version": "1.0.0",
    "description": "InkyDocker static assets",
    "main": "index.js",
    "scripts": {
      "build": "webpack --mode production",
      "dev": "webpack --mode development --watch"
    },
    "dependencies": {
      "@fullcalendar/core": "^6.1.8",
      "@fullcalendar/timegrid": "^6.1.8"
    },
    "devDependencies": {
      "webpack": "^5.0.0",
      "webpack-cli": "^4.0.0",
      "css-loader": "^6.0.0",
      "style-loader": "^3.0.0"
    },
    "author": "",
    "license": "MIT"
  }
```


## README.md

```md
# InkyDocker

InkyDocker is a Flask-based web application designed to manage and display images on e-ink displays. It allows users to upload images, schedule them for display, manage display settings, and monitor device metrics.

## Features

*   **Image Gallery**: Upload, crop, and manage images for display.
*   **Scheduled Display**: Schedule images to be displayed on e-ink displays at specific times.
*   **Device Management**: Configure and manage connected e-ink displays, including setting orientation and fetching display information.
*   **AI Settings**: Configure AI settings, such as OpenAI API key and Ollama model settings.
*   **Device Monitoring**: Monitor real-time device metrics such as CPU usage, memory usage, and disk usage.

## API Endpoints

The application exposes the following API endpoints for interacting with e-ink displays:

*   `POST /send_image`: Upload an image to display on the e-ink screen. Example:

    ```bash
    curl -F "file=@path/to/your/image.jpg" http://<IP_ADDRESS>/send_image
    ```

*   `POST /set_orientation`: Set the display orientation (choose either "horizontal" or "vertical"). Example:

    ```bash
    curl -X POST -d "orientation=vertical" http://<IP_ADDRESS>/set_orientation
    ```

*   `GET /display_info`: Retrieve display information in JSON format. Example:

    ```bash
    curl http://<IP_ADDRESS>/display_info
    ```

*   `POST /system_update`: Trigger a system update and upgrade (which will reboot the device). Example:

    ```bash
    curl -X POST http://<IP_ADDRESS>/system_update
    ```

*   `POST /backup`: Create a compressed backup of the SD card and download it. Example:

    ```bash
    curl -X POST http://<IP_ADDRESS>/backup --output backup.img.gz
    ```

*   `POST /update`: Perform a Git pull to update the application and reboot the device. Example:

    ```bash
    curl -X POST http://<IP_ADDRESS>/update
    ```

*   `GET /stream`: Connect to the SSE stream to receive real-time system metrics. Example:

    ```bash
    curl http://<IP_ADDRESS>/stream
    ```

## Requirements

*   Python 3.6+
*   Flask
*   Flask-Migrate
*   Pillow
*   Other dependencies listed in `requirements.txt`

## Installation

1.  Clone the repository:

    ```bash
    git clone <repository_url>
    cd inkydocker
    ```

2.  Create a virtual environment:

    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

4.  Configure the application:

    *   Create a `config.py` file based on the `config.example.py` file.
    *   Set the necessary environment variables, such as the database URI and API keys.

5.  Run database migrations:

    ```bash
    flask db upgrade
    ```

6.  Run the application:

    ```bash
    python app.py
    ```

## Usage

1.  Access the web application in your browser.
2.  Configure your e-ink displays in the Settings page.
3.  Upload images to the Gallery.
4.  Schedule images for display on the Schedule page.
5.  Monitor device metrics on the Settings page.

## Contributing

Contributions are welcome! Please submit a pull request with your changes.

## License

[Specify the license for your project]
```


## requirements.txt

```txt
Flask
Pillow==11.1.0
pillow-heif==0.21.0
Flask-SQLAlchemy==3.1.1
cryptography
requests
Flask-Migrate
ollama
httpx
APScheduler>=3.9.0
openai
celery
redis
open_clip_torch
chromadb
scikit-learn
gunicorn
psutil>=5.9.0  # For memory monitoring
```


## supervisord.conf

```conf
[supervisord]
nodaemon=true
logfile=/dev/null

[program:flask]
command=gunicorn -w 1 -t 120 --bind 0.0.0.0:5001 app:app
directory=/app
autostart=true
autorestart=true
stdout_logfile=/dev/stdout
stdout_logfile_maxbytes=0
stderr_logfile=/dev/stderr
stderr_logfile_maxbytes=0

[program:celery]
command=celery -A tasks.celery worker --loglevel=info --max-memory-per-child=500000 --max-tasks-per-child=1 -c %(ENV_CELERY_WORKERS)s
directory=/app
autostart=true
autorestart=true
stdout_logfile=/dev/stdout
stdout_logfile_maxbytes=0
stderr_logfile=/dev/stderr
stderr_logfile_maxbytes=0

[program:redis]
command=redis-server
directory=/app
autostart=true
autorestart=true
stdout_logfile=/dev/stdout
stdout_logfile_maxbytes=0
stderr_logfile=/dev/stderr
stderr_logfile_maxbytes=0
```


## tasks.py

```py
import os
import datetime
import subprocess
import time
import uuid

from flask import current_app
from PIL import Image

import torch
import open_clip
from sklearn.metrics.pairwise import cosine_similarity

# Import and initialize Celery and APScheduler
from celery import Celery
from apscheduler.schedulers.background import BackgroundScheduler

# Initialize Celery with memory limits
celery = Celery('tasks', broker='redis://localhost:6379/0', backend='redis://localhost:6379/0')

# Configure Celery to prevent memory leaks
celery.conf.update(
    worker_max_memory_per_child=500000,   # 500MB memory limit per worker
    worker_max_tasks_per_child=1,         # Restart worker after each task
    task_time_limit=600,                  # 10 minute time limit per task
    task_soft_time_limit=300,             # 5 minute soft time limit (sends exception)
    worker_concurrency=2,                 # Limit to 2 concurrent workers
    broker_connection_retry_on_startup=True  # Fix deprecation warning
)

class FlaskTask(celery.Task):
    def __call__(self, *args, **kwargs):
        # Import the app inside the task to avoid circular imports
        import sys
        import os
        # Add the current directory to the path if not already there
        app_dir = os.path.dirname(os.path.abspath(__file__))
        if app_dir not in sys.path:
            sys.path.insert(0, app_dir)
        
        # Now import the app
        from app import app as flask_app
        with flask_app.app_context():
            return self.run(*args, **kwargs)

celery.Task = FlaskTask

# Initialize APScheduler
scheduler = BackgroundScheduler()

# Device setup
device = "cuda" if torch.cuda.is_available() else "cpu"

# Define candidate tags
CANDIDATE_TAGS = [
    "nature", "urban", "people", "animals", "food",
    "technology", "landscape", "portrait", "abstract",
    "night", "day", "indoor", "outdoor", "colorful",
    "monochrome", "black and white", "sepia", "bright", "dark",
    "warm", "cool", "red", "green", "blue", "yellow", "orange",
    "purple", "pink", "brown", "gray", "white", "black", "minimal",
    "detailed", "texture", "pattern", "geometric", "organic", "digital",
    "analog", "illustration", "painting", "photograph", "sketch",
    "drawing", "3D", "flat", "realistic", "surreal", "fantasy",
    "sci-fi", "historical", "futuristic", "retro", "classic"
]

# Model and embeddings cache
clip_models = {}
clip_preprocessors = {}
tag_embeddings = {}

def get_clip_model():
    """Get the CLIP model based on user configuration"""
    from models import UserConfig
    from flask import current_app
    import gc  # Garbage collection
    
    # Get the selected CLIP model from user config
    config = UserConfig.query.first()
    clip_model_name = 'ViT-B-32'  # Default model (smallest and fastest)
    
    # Create config if it doesn't exist
    if not config:
        from models import db
        config = UserConfig(clip_model=clip_model_name)
        db.session.add(config)
        db.session.commit()
    elif config.clip_model:
        clip_model_name = config.clip_model
    
    # If trying to use ViT-L-14 but we're low on memory, fall back to ViT-B-32
    if clip_model_name == 'ViT-L-14':
        try:
            # Check available memory (this is a simple heuristic)
            import psutil
            available_memory = psutil.virtual_memory().available / (1024 * 1024)  # MB
            if available_memory < 1000:  # Less than 1GB available
                current_app.logger.warning(f"Low memory ({available_memory}MB available). Falling back to ViT-B-32 model.")
                clip_model_name = 'ViT-B-32'
        except ImportError:
            # If psutil is not available, just continue
            pass
    
    # Check if model is already loaded
    if clip_model_name in clip_models:
        return clip_model_name, clip_models[clip_model_name], clip_preprocessors[clip_model_name]
    
    # Clear any existing models to free memory
    for model_name in list(clip_models.keys()):
        if model_name != clip_model_name:
            del clip_models[model_name]
            del clip_preprocessors[model_name]
    
    # Force garbage collection
    gc.collect()
    
    # Load the model
    try:
        current_app.logger.info(f"Loading CLIP model: {clip_model_name}")
        model, _, preprocess = open_clip.create_model_and_transforms(clip_model_name, pretrained='openai', jit=False, force_quick_gelu=True)
        model.to(device)
        model.eval()
        
        # Cache the model and preprocessor
        clip_models[clip_model_name] = model
        clip_preprocessors[clip_model_name] = preprocess
        
        # Precompute tag embeddings for this model
        tokenizer = open_clip.get_tokenizer(clip_model_name)
        if clip_model_name not in tag_embeddings:
            tag_embeddings[clip_model_name] = {}
            with torch.no_grad():
                # Process tags in smaller batches to save memory
                batch_size = 10
                for i in range(0, len(CANDIDATE_TAGS), batch_size):
                    batch_tags = CANDIDATE_TAGS[i:i+batch_size]
                    for tag in batch_tags:
                        text_tokens = tokenizer([f"a photo of {tag}"])
                        text_features = model.encode_text(text_tokens)
                        text_features /= text_features.norm(dim=-1, keepdim=True)
                        tag_embeddings[clip_model_name][tag] = text_features.cpu()  # Store on CPU to save GPU memory
                    # Force garbage collection between batches
                    gc.collect()
        
        return clip_model_name, model, preprocess
    except Exception as e:
        current_app.logger.error(f"Error loading CLIP model {clip_model_name}: {e}")
        # Fall back to default model if available
        if 'ViT-B-32' in clip_models:
            return 'ViT-B-32', clip_models['ViT-B-32'], clip_preprocessors['ViT-B-32']
        # Otherwise load default model
        model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai', jit=False, force_quick_gelu=True)
        model.to(device)
        model.eval()
        clip_models['ViT-B-32'] = model
        clip_preprocessors['ViT-B-32'] = preprocess
        return 'ViT-B-32', model, preprocess

def get_image_embedding(image_path):
    try:
        # Get the current CLIP model
        model_name, model, preprocess = get_clip_model()
        
        # Process the image
        image = Image.open(image_path).convert("RGB")
        image_input = preprocess(image).unsqueeze(0).to(device)
        
        # Get image features
        with torch.no_grad():
            embedding = model.encode_image(image_input)
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)
        
        return embedding.cpu().numpy()[0], model_name
    except Exception as e:
        try:
            current_app.logger.error(f"Error processing image {image_path}: {e}")
        except:
            print(f"Error processing image {image_path}: {e}")
        return None, None

def generate_tags_and_description(embedding, model_name):
    """Generate tags and description based on image embedding and model"""
    from flask import current_app
    
    # If model_name is None, use default
    if model_name is None:
        model_name = 'ViT-B-32'
    
    # If model embeddings not found, try to load the model
    if model_name not in tag_embeddings:
        try:
            get_clip_model()
        except Exception as e:
            current_app.logger.error(f"Error loading model for tag generation: {e}")
            # Fall back to any available model
            if len(tag_embeddings) > 0:
                model_name = list(tag_embeddings.keys())[0]
            else:
                return [], "No tags available"
    
    # Calculate similarities with tag embeddings
    scores = {}
    for tag in CANDIDATE_TAGS:
        if tag in tag_embeddings.get(model_name, {}):
            # Get the tag embedding for this model
            tag_emb = tag_embeddings[model_name][tag]
            # Calculate similarity
            similarity = torch.cosine_similarity(
                torch.tensor(embedding).unsqueeze(0),
                tag_emb.cpu(),
                dim=1
            ).item()
            scores[tag] = similarity
    
    # Sort tags by similarity score
    sorted_tags = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    # Get top 10 tags
    top_tags = [tag for tag, _ in sorted_tags[:10]]
    
    # Create description
    description = "This image may contain " + ", ".join(top_tags) + "."
    
    return top_tags, description

@celery.task(bind=True, time_limit=600, soft_time_limit=500, max_retries=3)
def process_image_tagging(self, filename):
    """
    Process an image: compute its embedding, generate tags and a description,
    and update its record in the database.
    """
    from models import ImageDB, db, UserConfig
    from flask import current_app
    import gc  # Garbage collection
    
    # Force garbage collection at the start
    gc.collect()
    
    # Log which CLIP model is being used
    config = UserConfig.query.first()
    clip_model_name = config.clip_model if config and config.clip_model else "ViT-B-32"
    current_app.logger.info(f"Tagging image {filename} with CLIP model: {clip_model_name}")
    
    # If using the large model and we're low on memory, fall back to the smaller model
    if clip_model_name == 'ViT-L-14':
        try:
            import psutil
            available_memory = psutil.virtual_memory().available / (1024 * 1024)  # MB
            if available_memory < 1000:  # Less than 1GB available
                current_app.logger.warning(f"Low memory ({available_memory}MB available). Falling back to ViT-B-32 model.")
                clip_model_name = 'ViT-B-32'
        except ImportError:
            pass
    
    image_folder = current_app.config.get("IMAGE_FOLDER", "./images")
    image_path = os.path.join(image_folder, filename)
    if not os.path.exists(image_path):
        return {"status": "error", "message": "Image file not found", "filename": filename}
    
    try:
        # Get image embedding using the selected model
        embedding, model_name = get_image_embedding(image_path)
        if embedding is None:
            return {"status": "error", "message": "Failed to compute embedding", "filename": filename}
        
        # Force garbage collection after embedding
        gc.collect()
        
        # Generate tags and description
        tags, description = generate_tags_and_description(embedding, model_name)
        
        # Force garbage collection after tag generation
        gc.collect()
        
        # Update database
        img = ImageDB.query.filter_by(filename=filename).first()
        if img:
            img.tags = ", ".join(tags)
            img.description = description
            db.session.commit()
            
            # Force garbage collection to free memory
            gc.collect()
            
            # Clear any cached models to free memory
            if model_name in clip_models and model_name != 'ViT-B-32':
                del clip_models[model_name]
                del clip_preprocessors[model_name]
                gc.collect()
            
            return {
                "status": "success",
                "filename": filename,
                "tags": tags,
                "description": description,
                "model_used": model_name
            }
        else:
            return {"status": "error", "message": "Image not found in database", "filename": filename}
    except Exception as e:
        current_app.logger.error(f"Error in process_image_tagging: {e}")
        # Try to clean up memory
        gc.collect()
        
        # Clear all cached models to free memory
        for model_name in list(clip_models.keys()):
            del clip_models[model_name]
            del clip_preprocessors[model_name]
        gc.collect()
        
        return {"status": "error", "message": str(e), "filename": filename}

def reembed_image(filename):
    """
    Recompute the embedding, tags, and description for an image and update its record.
    """
    from models import ImageDB, db, UserConfig
    from flask import current_app
    
    # Log which CLIP model is being used
    config = UserConfig.query.first()
    clip_model_name = config.clip_model if config and config.clip_model else "ViT-B-32"
    current_app.logger.info(f"Re-tagging image {filename} with CLIP model: {clip_model_name}")
    
    image_folder = current_app.config.get("IMAGE_FOLDER", "./images")
    image_path = os.path.join(image_folder, filename)
    if not os.path.exists(image_path):
        return {"status": "error", "message": "Image file not found", "filename": filename}
    
    try:
        # Get image embedding using the selected model
        embedding, model_name = get_image_embedding(image_path)
        if embedding is None:
            return {"status": "error", "message": "Failed to compute embedding", "filename": filename}
        
        # Generate tags and description
        tags, description = generate_tags_and_description(embedding, model_name)
        
        # Update database
        img = ImageDB.query.filter_by(filename=filename).first()
        if img:
            img.tags = ", ".join(tags)
            img.description = description
            db.session.commit()
            return {
                "status": "success",
                "filename": filename,
                "tags": tags,
                "description": description,
                "model_used": model_name
            }
        else:
            return {"status": "error", "message": "Image not found in database", "filename": filename}
    except Exception as e:
        current_app.logger.error(f"Error in reembed_image: {e}")
        return {"status": "error", "message": str(e), "filename": filename}

# Global dictionary to track bulk tagging progress
BULK_PROGRESS = {}

@celery.task(bind=True)
def bulk_tag_images(self):
    """
    Process all images that do not have tags. Returns a task ID that can be used to query progress.
    """
    from models import ImageDB
    from flask import current_app
    
    try:
        images = ImageDB.query.filter((ImageDB.tags == None) | (ImageDB.tags == "")).all()
        total = len(images)
        if total == 0:
            return None
        task_id = str(uuid.uuid4())
        BULK_PROGRESS[task_id] = 0
        for i, img in enumerate(images, start=1):
            process_image_tagging.delay(img.filename)
            BULK_PROGRESS[task_id] = (i / total) * 100
        return task_id
    except Exception as e:
        current_app.logger.error(f"Error in bulk_tag_images: {e}")
        return None

@celery.task(bind=True, time_limit=1800, soft_time_limit=1500)
def reembed_all_images(self):
    """
    Rerun tagging on all images using the currently selected CLIP model.
    This is useful when changing the CLIP model.
    """
    from models import ImageDB, UserConfig
    from flask import current_app
    import gc  # Garbage collection
    
    try:
        # Get the current CLIP model
        config = UserConfig.query.first()
        clip_model_name = config.clip_model if config and config.clip_model else "ViT-B-32"
        
        # Log the operation
        current_app.logger.info(f"Rerunning tagging on all images with CLIP model: {clip_model_name}")
        
        # Get all images
        images = ImageDB.query.all()
        total = len(images)
        
        if total == 0:
            return {"status": "success", "message": "No images found to tag"}
        
        # Create a task ID for tracking progress
        task_id = str(uuid.uuid4())
        BULK_PROGRESS[task_id] = 0
        
        # Process images one at a time to manage memory better
        for i, img in enumerate(images):
            # Process one image and wait for it to complete before starting the next
            process_image_tagging.delay(img.filename)
            
            # Update progress
            BULK_PROGRESS[task_id] = min(100, (i + 1) / total * 100)
            
            # Force garbage collection after each image
            gc.collect()
            
            # Add a small delay between tasks to allow memory to be freed
            time.sleep(0.5)
            
        return {
            "status": "success",
            "message": f"Started retagging {total} images with model {clip_model_name}",
            "task_id": task_id
        }
    except Exception as e:
        current_app.logger.error(f"Error in reembed_all_images: {e}")
        # Try to clean up memory
        gc.collect()
        return {"status": "error", "message": str(e)}

def send_scheduled_image(event_id):
    """
    Send a scheduled image to a device.
    """
    from models import ScheduleEvent, Device, db
    from utils.crop_helpers import load_crop_info_from_db
    from utils.image_helpers import add_send_log_entry
    from flask import current_app
    
    try:
        event = ScheduleEvent.query.get(event_id)
        if not event:
            current_app.logger.error("Event not found: %s", event_id)
            return
        device_obj = Device.query.filter_by(address=event.device).first()
        if not device_obj:
            current_app.logger.error("Device not found for event %s", event_id)
            return

        image_folder = current_app.config.get("IMAGE_FOLDER", "./images")
        data_folder = current_app.config.get("DATA_FOLDER", "./data")
        filepath = os.path.join(image_folder, event.filename)
        if not os.path.exists(filepath):
            current_app.logger.error("Image file not found: %s", filepath)
            return

        addr = device_obj.address
        if not (addr.startswith("http://") or addr.startswith("https://")):
            addr = "http://" + addr

        with Image.open(filepath) as orig_img:
            orig_w, orig_h = orig_img.size
            parts = device_obj.resolution.split("x")
            dev_width = int(parts[0])
            dev_height = int(parts[1])
            
            # Check if device is in portrait orientation
            is_portrait = device_obj.orientation.lower() == 'portrait'
            
            # If portrait, swap width and height for target ratio calculation
            if is_portrait:
                target_ratio = dev_height / dev_width
            else:
                target_ratio = dev_width / dev_height
                
            cdata = load_crop_info_from_db(event.filename)
            if cdata:
                x = cdata.get("x", 0)
                y = cdata.get("y", 0)
                w = cdata.get("width", orig_w)
                h = cdata.get("height", orig_h)
                cropped = orig_img.crop((x, y, x + w, y + h))
            else:
                orig_ratio = orig_w / orig_h
                if orig_ratio > target_ratio:
                    new_width = int(orig_h * target_ratio)
                    left = (orig_w - new_width) // 2
                    crop_box = (left, 0, left + new_width, orig_h)
                else:
                    new_height = int(orig_w / target_ratio)
                    top = (orig_h - new_height) // 2
                    crop_box = (0, top, orig_w, top + new_height)
                cropped = orig_img.crop(crop_box)
            
            # If portrait, rotate the image 90 degrees clockwise and swap dimensions
            if is_portrait:
                cropped = cropped.rotate(-90, expand=True)  # -90 for clockwise rotation
                final_img = cropped.resize((dev_height, dev_width), Image.LANCZOS)  # Note swapped dimensions
            else:
                final_img = cropped.resize((dev_width, dev_height), Image.LANCZOS)
            temp_dir = os.path.join(data_folder, "temp")
            if not os.path.exists(temp_dir):
                os.makedirs(temp_dir)
            temp_filename = os.path.join(temp_dir, f"temp_{event.filename}")
            final_img.save(temp_filename, format="JPEG", quality=95)
        
        cmd = f'curl "{addr}/send_image" -X POST -F "file=@{temp_filename}"'
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        os.remove(temp_filename)
        
        if result.returncode == 0:
            event.sent = True
            db.session.commit()
            device_obj.last_sent = event.filename
            db.session.commit()
            add_send_log_entry(event.filename)
        else:
            current_app.logger.error("Error sending image: %s", result.stderr)
    except Exception as e:
        current_app.logger.error("Error in send_scheduled_image: %s", e)
        return

    # Reschedule recurring events
    if event.recurrence and event.recurrence.lower() != "none":
        try:
            dt = datetime.datetime.fromisoformat(event.datetime_str)
        except Exception as e:
            current_app.logger.error("Error parsing datetime_str: %s", e)
            return
        if event.recurrence.lower() == "daily":
            next_dt = dt + datetime.timedelta(days=1)
        elif event.recurrence.lower() == "weekly":
            next_dt = dt + datetime.timedelta(weeks=1)
        elif event.recurrence.lower() == "monthly":
            next_dt = dt + datetime.timedelta(days=30)
        else:
            next_dt = None
        if next_dt:
            event.datetime_str = next_dt.isoformat(sep=' ')
            event.sent = False
            db.session.commit()
            scheduler.add_job(send_scheduled_image, 'date', run_date=next_dt, args=[event.id])

@celery.task(bind=True)
def fetch_device_metrics(self):
    """
    Fetch metrics from all devices and update the database.
    This task runs periodically to ensure we have up-to-date metrics.
    """
    from models import Device, DeviceMetrics, db
    from flask import current_app
    import httpx
    import json
    
    devices = Device.query.all()
    for device in devices:
        try:
            # Ensure the address has a scheme
            address = device.address
            if not (address.startswith("http://") or address.startswith("https://")):
                address = "http://" + address
            
            # Try to connect to the device's stream endpoint
            try:
                # Use a short timeout to avoid blocking
                response = httpx.get(f"{address}/stream", timeout=5.0)
                if response.status_code == 200:
                    # Parse the response
                    data_str = response.text
                    if data_str.startswith('data: '):
                        data_str = data_str[6:]
                    
                    try:
                        data = json.loads(data_str)
                        
                        # Update device metrics in the database
                        if 'cpu' in data and ('memory' in data or 'mem' in data) and 'disk' in data:
                            memory_value = data.get('memory', data.get('mem', 0))
                            
                            # Update device status
                            device.cpu_usage = str(data['cpu'])
                            device.mem_usage = str(memory_value)
                            device.disk_usage = str(data['disk'])
                            device.online = True
                            db.session.commit()
                            
                            # Also save to DeviceMetrics table
                            new_metric = DeviceMetrics(
                                device_id=device.id,
                                cpu=data['cpu'],
                                memory=memory_value,
                                disk=data['disk']
                            )
                            db.session.add(new_metric)
                            db.session.commit()
                            
                            current_app.logger.info(f"Updated metrics for device {device.friendly_name}")
                    except json.JSONDecodeError:
                        current_app.logger.warning(f"Invalid JSON from device {device.friendly_name}: {data_str}")
                else:
                    current_app.logger.warning(f"Error connecting to device {device.friendly_name}: {response.status_code}")
            except Exception as e:
                current_app.logger.error(f"Error fetching metrics from device {device.friendly_name}: {str(e)}")
                # Mark device as offline if we can't connect
                device.online = False
                db.session.commit()
        except Exception as e:
            current_app.logger.error(f"Error processing device {device.id}: {str(e)}")
    
    return {"status": "success", "message": f"Fetched metrics from {len(devices)} devices"}

# Flag to track if scheduler has been started
scheduler_started = False

def start_scheduler(app):
    """
    Start the APScheduler with the Flask app context
    """
    global scheduler_started
    
    # Only start the scheduler if it hasn't been started yet
    if not scheduler_started:
        with app.app_context():
            # Schedule the fetch_device_metrics task to run every minute
            try:
                # Check if job already exists
                if not scheduler.get_job('fetch_device_metrics'):
                    scheduler.add_job(
                        lambda: fetch_device_metrics.delay(),
                        'interval',
                        minutes=1,
                        id='fetch_device_metrics'
                    )
                
                # Only start if not already running
                if not scheduler.running:
                    scheduler.start()
                    scheduler_started = True
                    app.logger.info("Scheduler started successfully")
                else:
                    app.logger.info("Scheduler already running")
            except Exception as e:
                app.logger.error(f"Error starting scheduler: {e}")
    else:
        app.logger.info("Scheduler already initialized")
```


## webpack.config.js

```js
const path = require('path');

module.exports = {
  entry: './assets/js/main.js', 
  output: {
    filename: 'bundle.js',
    path: path.resolve(__dirname, 'static', 'js')
  },
  module: {
    rules: [
      {
        test: /\.css$/i,
        use: ['style-loader', 'css-loader'],
      },
    ],
  },
};
```
```


## exportconfig.json

```json
{
    "output": "export.md",
    "description": "",
    "ignoreFile": ".export-ignore",
    "includeFile": ".export-whitelist",
    "maxFileSize": 1048576,
    "removeComments": true,
    "includeProjectStructure": true
  }
```


## LICENSE

```
GNU GENERAL PUBLIC LICENSE
                       Version 3, 29 June 2007

 Copyright (C) 2007 Free Software Foundation, Inc. <https://fsf.org/>
 Everyone is permitted to copy and distribute verbatim copies
 of this license document, but changing it is not allowed.

                            Preamble

  The GNU General Public License is a free, copyleft license for
software and other kinds of works.

  The licenses for most software and other practical works are designed
to take away your freedom to share and change the works.  By contrast,
the GNU General Public License is intended to guarantee your freedom to
share and change all versions of a program--to make sure it remains free
software for all its users.  We, the Free Software Foundation, use the
GNU General Public License for most of our software; it applies also to
any other work released this way by its authors.  You can apply it to
your programs, too.

  When we speak of free software, we are referring to freedom, not
price.  Our General Public Licenses are designed to make sure that you
have the freedom to distribute copies of free software (and charge for
them if you wish), that you receive source code or can get it if you
want it, that you can change the software or use pieces of it in new
free programs, and that you know you can do these things.

  To protect your rights, we need to prevent others from denying you
these rights or asking you to surrender the rights.  Therefore, you have
certain responsibilities if you distribute copies of the software, or if
you modify it: responsibilities to respect the freedom of others.

  For example, if you distribute copies of such a program, whether
gratis or for a fee, you must pass on to the recipients the same
freedoms that you received.  You must make sure that they, too, receive
or can get the source code.  And you must show them these terms so they
know their rights.

  Developers that use the GNU GPL protect your rights with two steps:
(1) assert copyright on the software, and (2) offer you this License
giving you legal permission to copy, distribute and/or modify it.

  For the developers' and authors' protection, the GPL clearly explains
that there is no warranty for this free software.  For both users' and
authors' sake, the GPL requires that modified versions be marked as
changed, so that their problems will not be attributed erroneously to
authors of previous versions.

  Some devices are designed to deny users access to install or run
modified versions of the software inside them, although the manufacturer
can do so.  This is fundamentally incompatible with the aim of
protecting users' freedom to change the software.  The systematic
pattern of such abuse occurs in the area of products for individuals to
use, which is precisely where it is most unacceptable.  Therefore, we
have designed this version of the GPL to prohibit the practice for those
products.  If such problems arise substantially in other domains, we
stand ready to extend this provision to those domains in future versions
of the GPL, as needed to protect the freedom of users.

  Finally, every program is threatened constantly by software patents.
States should not allow patents to restrict development and use of
software on general-purpose computers, but in those that do, we wish to
avoid the special danger that patents applied to a free program could
make it effectively proprietary.  To prevent this, the GPL assures that
patents cannot be used to render the program non-free.

  The precise terms and conditions for copying, distribution and
modification follow.

                       TERMS AND CONDITIONS

  0. Definitions.

  "This License" refers to version 3 of the GNU General Public License.

  "Copyright" also means copyright-like laws that apply to other kinds of
works, such as semiconductor masks.

  "The Program" refers to any copyrightable work licensed under this
License.  Each licensee is addressed as "you".  "Licensees" and
"recipients" may be individuals or organizations.

  To "modify" a work means to copy from or adapt all or part of the work
in a fashion requiring copyright permission, other than the making of an
exact copy.  The resulting work is called a "modified version" of the
earlier work or a work "based on" the earlier work.

  A "covered work" means either the unmodified Program or a work based
on the Program.

  To "propagate" a work means to do anything with it that, without
permission, would make you directly or secondarily liable for
infringement under applicable copyright law, except executing it on a
computer or modifying a private copy.  Propagation includes copying,
distribution (with or without modification), making available to the
public, and in some countries other activities as well.

  To "convey" a work means any kind of propagation that enables other
parties to make or receive copies.  Mere interaction with a user through
a computer network, with no transfer of a copy, is not conveying.

  An interactive user interface displays "Appropriate Legal Notices"
to the extent that it includes a convenient and prominently visible
feature that (1) displays an appropriate copyright notice, and (2)
tells the user that there is no warranty for the work (except to the
extent that warranties are provided), that licensees may convey the
work under this License, and how to view a copy of this License.  If
the interface presents a list of user commands or options, such as a
menu, a prominent item in the list meets this criterion.

  1. Source Code.

  The "source code" for a work means the preferred form of the work
for making modifications to it.  "Object code" means any non-source
form of a work.

  A "Standard Interface" means an interface that either is an official
standard defined by a recognized standards body, or, in the case of
interfaces specified for a particular programming language, one that
is widely used among developers working in that language.

  The "System Libraries" of an executable work include anything, other
than the work as a whole, that (a) is included in the normal form of
packaging a Major Component, but which is not part of that Major
Component, and (b) serves only to enable use of the work with that
Major Component, or to implement a Standard Interface for which an
implementation is available to the public in source code form.  A
"Major Component", in this context, means a major essential component
(kernel, window system, and so on) of the specific operating system
(if any) on which the executable work runs, or a compiler used to
produce the work, or an object code interpreter used to run it.

  The "Corresponding Source" for a work in object code form means all
the source code needed to generate, install, and (for an executable
work) run the object code and to modify the work, including scripts to
control those activities.  However, it does not include the work's
System Libraries, or general-purpose tools or generally available free
programs which are used unmodified in performing those activities but
which are not part of the work.  For example, Corresponding Source
includes interface definition files associated with source files for
the work, and the source code for shared libraries and dynamically
linked subprograms that the work is specifically designed to require,
such as by intimate data communication or control flow between those
subprograms and other parts of the work.

  The Corresponding Source need not include anything that users
can regenerate automatically from other parts of the Corresponding
Source.

  The Corresponding Source for a work in source code form is that
same work.

  2. Basic Permissions.

  All rights granted under this License are granted for the term of
copyright on the Program, and are irrevocable provided the stated
conditions are met.  This License explicitly affirms your unlimited
permission to run the unmodified Program.  The output from running a
covered work is covered by this License only if the output, given its
content, constitutes a covered work.  This License acknowledges your
rights of fair use or other equivalent, as provided by copyright law.

  You may make, run and propagate covered works that you do not
convey, without conditions so long as your license otherwise remains
in force.  You may convey covered works to others for the sole purpose
of having them make modifications exclusively for you, or provide you
with facilities for running those works, provided that you comply with
the terms of this License in conveying all material for which you do
not control copyright.  Those thus making or running the covered works
for you must do so exclusively on your behalf, under your direction
and control, on terms that prohibit them from making any copies of
your copyrighted material outside their relationship with you.

  Conveying under any other circumstances is permitted solely under
the conditions stated below.  Sublicensing is not allowed; section 10
makes it unnecessary.

  3. Protecting Users' Legal Rights From Anti-Circumvention Law.

  No covered work shall be deemed part of an effective technological
measure under any applicable law fulfilling obligations under article
11 of the WIPO copyright treaty adopted on 20 December 1996, or
similar laws prohibiting or restricting circumvention of such
measures.

  When you convey a covered work, you waive any legal power to forbid
circumvention of technological measures to the extent such circumvention
is effected by exercising rights under this License with respect to
the covered work, and you disclaim any intention to limit operation or
modification of the work as a means of enforcing, against the work's
users, your or third parties' legal rights to forbid circumvention of
technological measures.

  4. Conveying Verbatim Copies.

  You may convey verbatim copies of the Program's source code as you
receive it, in any medium, provided that you conspicuously and
appropriately publish on each copy an appropriate copyright notice;
keep intact all notices stating that this License and any
non-permissive terms added in accord with section 7 apply to the code;
keep intact all notices of the absence of any warranty; and give all
recipients a copy of this License along with the Program.

  You may charge any price or no price for each copy that you convey,
and you may offer support or warranty protection for a fee.

  5. Conveying Modified Source Versions.

  You may convey a work based on the Program, or the modifications to
produce it from the Program, in the form of source code under the
terms of section 4, provided that you also meet all of these conditions:

    a) The work must carry prominent notices stating that you modified
    it, and giving a relevant date.

    b) The work must carry prominent notices stating that it is
    released under this License and any conditions added under section
    7.  This requirement modifies the requirement in section 4 to
    "keep intact all notices".

    c) You must license the entire work, as a whole, under this
    License to anyone who comes into possession of a copy.  This
    License will therefore apply, along with any applicable section 7
    additional terms, to the whole of the work, and all its parts,
    regardless of how they are packaged.  This License gives no
    permission to license the work in any other way, but it does not
    invalidate such permission if you have separately received it.

    d) If the work has interactive user interfaces, each must display
    Appropriate Legal Notices; however, if the Program has interactive
    interfaces that do not display Appropriate Legal Notices, your
    work need not make them do so.

  A compilation of a covered work with other separate and independent
works, which are not by their nature extensions of the covered work,
and which are not combined with it such as to form a larger program,
in or on a volume of a storage or distribution medium, is called an
"aggregate" if the compilation and its resulting copyright are not
used to limit the access or legal rights of the compilation's users
beyond what the individual works permit.  Inclusion of a covered work
in an aggregate does not cause this License to apply to the other
parts of the aggregate.

  6. Conveying Non-Source Forms.

  You may convey a covered work in object code form under the terms
of sections 4 and 5, provided that you also convey the
machine-readable Corresponding Source under the terms of this License,
in one of these ways:

    a) Convey the object code in, or embodied in, a physical product
    (including a physical distribution medium), accompanied by the
    Corresponding Source fixed on a durable physical medium
    customarily used for software interchange.

    b) Convey the object code in, or embodied in, a physical product
    (including a physical distribution medium), accompanied by a
    written offer, valid for at least three years and valid for as
    long as you offer spare parts or customer support for that product
    model, to give anyone who possesses the object code either (1) a
    copy of the Corresponding Source for all the software in the
    product that is covered by this License, on a durable physical
    medium customarily used for software interchange, for a price no
    more than your reasonable cost of physically performing this
    conveying of source, or (2) access to copy the
    Corresponding Source from a network server at no charge.

    c) Convey individual copies of the object code with a copy of the
    written offer to provide the Corresponding Source.  This
    alternative is allowed only occasionally and noncommercially, and
    only if you received the object code with such an offer, in accord
    with subsection 6b.

    d) Convey the object code by offering access from a designated
    place (gratis or for a charge), and offer equivalent access to the
    Corresponding Source in the same way through the same place at no
    further charge.  You need not require recipients to copy the
    Corresponding Source along with the object code.  If the place to
    copy the object code is a network server, the Corresponding Source
    may be on a different server (operated by you or a third party)
    that supports equivalent copying facilities, provided you maintain
    clear directions next to the object code saying where to find the
    Corresponding Source.  Regardless of what server hosts the
    Corresponding Source, you remain obligated to ensure that it is
    available for as long as needed to satisfy these requirements.

    e) Convey the object code using peer-to-peer transmission, provided
    you inform other peers where the object code and Corresponding
    Source of the work are being offered to the general public at no
    charge under subsection 6d.

  A separable portion of the object code, whose source code is excluded
from the Corresponding Source as a System Library, need not be
included in conveying the object code work.

  A "User Product" is either (1) a "consumer product", which means any
tangible personal property which is normally used for personal, family,
or household purposes, or (2) anything designed or sold for incorporation
into a dwelling.  In determining whether a product is a consumer product,
doubtful cases shall be resolved in favor of coverage.  For a particular
product received by a particular user, "normally used" refers to a
typical or common use of that class of product, regardless of the status
of the particular user or of the way in which the particular user
actually uses, or expects or is expected to use, the product.  A product
is a consumer product regardless of whether the product has substantial
commercial, industrial or non-consumer uses, unless such uses represent
the only significant mode of use of the product.

  "Installation Information" for a User Product means any methods,
procedures, authorization keys, or other information required to install
and execute modified versions of a covered work in that User Product from
a modified version of its Corresponding Source.  The information must
suffice to ensure that the continued functioning of the modified object
code is in no case prevented or interfered with solely because
modification has been made.

  If you convey an object code work under this section in, or with, or
specifically for use in, a User Product, and the conveying occurs as
part of a transaction in which the right of possession and use of the
User Product is transferred to the recipient in perpetuity or for a
fixed term (regardless of how the transaction is characterized), the
Corresponding Source conveyed under this section must be accompanied
by the Installation Information.  But this requirement does not apply
if neither you nor any third party retains the ability to install
modified object code on the User Product (for example, the work has
been installed in ROM).

  The requirement to provide Installation Information does not include a
requirement to continue to provide support service, warranty, or updates
for a work that has been modified or installed by the recipient, or for
the User Product in which it has been modified or installed.  Access to a
network may be denied when the modification itself materially and
adversely affects the operation of the network or violates the rules and
protocols for communication across the network.

  Corresponding Source conveyed, and Installation Information provided,
in accord with this section must be in a format that is publicly
documented (and with an implementation available to the public in
source code form), and must require no special password or key for
unpacking, reading or copying.

  7. Additional Terms.

  "Additional permissions" are terms that supplement the terms of this
License by making exceptions from one or more of its conditions.
Additional permissions that are applicable to the entire Program shall
be treated as though they were included in this License, to the extent
that they are valid under applicable law.  If additional permissions
apply only to part of the Program, that part may be used separately
under those permissions, but the entire Program remains governed by
this License without regard to the additional permissions.

  When you convey a copy of a covered work, you may at your option
remove any additional permissions from that copy, or from any part of
it.  (Additional permissions may be written to require their own
removal in certain cases when you modify the work.)  You may place
additional permissions on material, added by you to a covered work,
for which you have or can give appropriate copyright permission.

  Notwithstanding any other provision of this License, for material you
add to a covered work, you may (if authorized by the copyright holders of
that material) supplement the terms of this License with terms:

    a) Disclaiming warranty or limiting liability differently from the
    terms of sections 15 and 16 of this License; or

    b) Requiring preservation of specified reasonable legal notices or
    author attributions in that material or in the Appropriate Legal
    Notices displayed by works containing it; or

    c) Prohibiting misrepresentation of the origin of that material, or
    requiring that modified versions of such material be marked in
    reasonable ways as different from the original version; or

    d) Limiting the use for publicity purposes of names of licensors or
    authors of the material; or

    e) Declining to grant rights under trademark law for use of some
    trade names, trademarks, or service marks; or

    f) Requiring indemnification of licensors and authors of that
    material by anyone who conveys the material (or modified versions of
    it) with contractual assumptions of liability to the recipient, for
    any liability that these contractual assumptions directly impose on
    those licensors and authors.

  All other non-permissive additional terms are considered "further
restrictions" within the meaning of section 10.  If the Program as you
received it, or any part of it, contains a notice stating that it is
governed by this License along with a term that is a further
restriction, you may remove that term.  If a license document contains
a further restriction but permits relicensing or conveying under this
License, you may add to a covered work material governed by the terms
of that license document, provided that the further restriction does
not survive such relicensing or conveying.

  If you add terms to a covered work in accord with this section, you
must place, in the relevant source files, a statement of the
additional terms that apply to those files, or a notice indicating
where to find the applicable terms.

  Additional terms, permissive or non-permissive, may be stated in the
form of a separately written license, or stated as exceptions;
the above requirements apply either way.

  8. Termination.

  You may not propagate or modify a covered work except as expressly
provided under this License.  Any attempt otherwise to propagate or
modify it is void, and will automatically terminate your rights under
this License (including any patent licenses granted under the third
paragraph of section 11).

  However, if you cease all violation of this License, then your
license from a particular copyright holder is reinstated (a)
provisionally, unless and until the copyright holder explicitly and
finally terminates your license, and (b) permanently, if the copyright
holder fails to notify you of the violation by some reasonable means
prior to 60 days after the cessation.

  Moreover, your license from a particular copyright holder is
reinstated permanently if the copyright holder notifies you of the
violation by some reasonable means, this is the first time you have
received notice of violation of this License (for any work) from that
copyright holder, and you cure the violation prior to 30 days after
your receipt of the notice.

  Termination of your rights under this section does not terminate the
licenses of parties who have received copies or rights from you under
this License.  If your rights have been terminated and not permanently
reinstated, you do not qualify to receive new licenses for the same
material under section 10.

  9. Acceptance Not Required for Having Copies.

  You are not required to accept this License in order to receive or
run a copy of the Program.  Ancillary propagation of a covered work
occurring solely as a consequence of using peer-to-peer transmission
to receive a copy likewise does not require acceptance.  However,
nothing other than this License grants you permission to propagate or
modify any covered work.  These actions infringe copyright if you do
not accept this License.  Therefore, by modifying or propagating a
covered work, you indicate your acceptance of this License to do so.

  10. Automatic Licensing of Downstream Recipients.

  Each time you convey a covered work, the recipient automatically
receives a license from the original licensors, to run, modify and
propagate that work, subject to this License.  You are not responsible
for enforcing compliance by third parties with this License.

  An "entity transaction" is a transaction transferring control of an
organization, or substantially all assets of one, or subdividing an
organization, or merging organizations.  If propagation of a covered
work results from an entity transaction, each party to that
transaction who receives a copy of the work also receives whatever
licenses to the work the party's predecessor in interest had or could
give under the previous paragraph, plus a right to possession of the
Corresponding Source of the work from the predecessor in interest, if
the predecessor has it or can get it with reasonable efforts.

  You may not impose any further restrictions on the exercise of the
rights granted or affirmed under this License.  For example, you may
not impose a license fee, royalty, or other charge for exercise of
rights granted under this License, and you may not initiate litigation
(including a cross-claim or counterclaim in a lawsuit) alleging that
any patent claim is infringed by making, using, selling, offering for
sale, or importing the Program or any portion of it.

  11. Patents.

  A "contributor" is a copyright holder who authorizes use under this
License of the Program or a work on which the Program is based.  The
work thus licensed is called the contributor's "contributor version".

  A contributor's "essential patent claims" are all patent claims
owned or controlled by the contributor, whether already acquired or
hereafter acquired, that would be infringed by some manner, permitted
by this License, of making, using, or selling its contributor version,
but do not include claims that would be infringed only as a
consequence of further modification of the contributor version.  For
purposes of this definition, "control" includes the right to grant
patent sublicenses in a manner consistent with the requirements of
this License.

  Each contributor grants you a non-exclusive, worldwide, royalty-free
patent license under the contributor's essential patent claims, to
make, use, sell, offer for sale, import and otherwise run, modify and
propagate the contents of its contributor version.

  In the following three paragraphs, a "patent license" is any express
agreement or commitment, however denominated, not to enforce a patent
(such as an express permission to practice a patent or covenant not to
sue for patent infringement).  To "grant" such a patent license to a
party means to make such an agreement or commitment not to enforce a
patent against the party.

  If you convey a covered work, knowingly relying on a patent license,
and the Corresponding Source of the work is not available for anyone
to copy, free of charge and under the terms of this License, through a
publicly available network server or other readily accessible means,
then you must either (1) cause the Corresponding Source to be so
available, or (2) arrange to deprive yourself of the benefit of the
patent license for this particular work, or (3) arrange, in a manner
consistent with the requirements of this License, to extend the patent
license to downstream recipients.  "Knowingly relying" means you have
actual knowledge that, but for the patent license, your conveying the
covered work in a country, or your recipient's use of the covered work
in a country, would infringe one or more identifiable patents in that
country that you have reason to believe are valid.

  If, pursuant to or in connection with a single transaction or
arrangement, you convey, or propagate by procuring conveyance of, a
covered work, and grant a patent license to some of the parties
receiving the covered work authorizing them to use, propagate, modify
or convey a specific copy of the covered work, then the patent license
you grant is automatically extended to all recipients of the covered
work and works based on it.

  A patent license is "discriminatory" if it does not include within
the scope of its coverage, prohibits the exercise of, or is
conditioned on the non-exercise of one or more of the rights that are
specifically granted under this License.  You may not convey a covered
work if you are a party to an arrangement with a third party that is
in the business of distributing software, under which you make payment
to the third party based on the extent of your activity of conveying
the work, and under which the third party grants, to any of the
parties who would receive the covered work from you, a discriminatory
patent license (a) in connection with copies of the covered work
conveyed by you (or copies made from those copies), or (b) primarily
for and in connection with specific products or compilations that
contain the covered work, unless you entered into that arrangement,
or that patent license was granted, prior to 28 March 2007.

  Nothing in this License shall be construed as excluding or limiting
any implied license or other defenses to infringement that may
otherwise be available to you under applicable patent law.

  12. No Surrender of Others' Freedom.

  If conditions are imposed on you (whether by court order, agreement or
otherwise) that contradict the conditions of this License, they do not
excuse you from the conditions of this License.  If you cannot convey a
covered work so as to satisfy simultaneously your obligations under this
License and any other pertinent obligations, then as a consequence you may
not convey it at all.  For example, if you agree to terms that obligate you
to collect a royalty for further conveying from those to whom you convey
the Program, the only way you could satisfy both those terms and this
License would be to refrain entirely from conveying the Program.

  13. Use with the GNU Affero General Public License.

  Notwithstanding any other provision of this License, you have
permission to link or combine any covered work with a work licensed
under version 3 of the GNU Affero General Public License into a single
combined work, and to convey the resulting work.  The terms of this
License will continue to apply to the part which is the covered work,
but the special requirements of the GNU Affero General Public License,
section 13, concerning interaction through a network will apply to the
combination as such.

  14. Revised Versions of this License.

  The Free Software Foundation may publish revised and/or new versions of
the GNU General Public License from time to time.  Such new versions will
be similar in spirit to the present version, but may differ in detail to
address new problems or concerns.

  Each version is given a distinguishing version number.  If the
Program specifies that a certain numbered version of the GNU General
Public License "or any later version" applies to it, you have the
option of following the terms and conditions either of that numbered
version or of any later version published by the Free Software
Foundation.  If the Program does not specify a version number of the
GNU General Public License, you may choose any version ever published
by the Free Software Foundation.

  If the Program specifies that a proxy can decide which future
versions of the GNU General Public License can be used, that proxy's
public statement of acceptance of a version permanently authorizes you
to choose that version for the Program.

  Later license versions may give you additional or different
permissions.  However, no additional obligations are imposed on any
author or copyright holder as a result of your choosing to follow a
later version.

  15. Disclaimer of Warranty.

  THERE IS NO WARRANTY FOR THE PROGRAM, TO THE EXTENT PERMITTED BY
APPLICABLE LAW.  EXCEPT WHEN OTHERWISE STATED IN WRITING THE COPYRIGHT
HOLDERS AND/OR OTHER PARTIES PROVIDE THE PROGRAM "AS IS" WITHOUT WARRANTY
OF ANY KIND, EITHER EXPRESSED OR IMPLIED, INCLUDING, BUT NOT LIMITED TO,
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
PURPOSE.  THE ENTIRE RISK AS TO THE QUALITY AND PERFORMANCE OF THE PROGRAM
IS WITH YOU.  SHOULD THE PROGRAM PROVE DEFECTIVE, YOU ASSUME THE COST OF
ALL NECESSARY SERVICING, REPAIR OR CORRECTION.

  16. Limitation of Liability.

  IN NO EVENT UNLESS REQUIRED BY APPLICABLE LAW OR AGREED TO IN WRITING
WILL ANY COPYRIGHT HOLDER, OR ANY OTHER PARTY WHO MODIFIES AND/OR CONVEYS
THE PROGRAM AS PERMITTED ABOVE, BE LIABLE TO YOU FOR DAMAGES, INCLUDING ANY
GENERAL, SPECIAL, INCIDENTAL OR CONSEQUENTIAL DAMAGES ARISING OUT OF THE
USE OR INABILITY TO USE THE PROGRAM (INCLUDING BUT NOT LIMITED TO LOSS OF
DATA OR DATA BEING RENDERED INACCURATE OR LOSSES SUSTAINED BY YOU OR THIRD
PARTIES OR A FAILURE OF THE PROGRAM TO OPERATE WITH ANY OTHER PROGRAMS),
EVEN IF SUCH HOLDER OR OTHER PARTY HAS BEEN ADVISED OF THE POSSIBILITY OF
SUCH DAMAGES.

  17. Interpretation of Sections 15 and 16.

  If the disclaimer of warranty and limitation of liability provided
above cannot be given local legal effect according to their terms,
reviewing courts shall apply local law that most closely approximates
an absolute waiver of all civil liability in connection with the
Program, unless a warranty or assumption of liability accompanies a
copy of the Program in return for a fee.

                     END OF TERMS AND CONDITIONS

            How to Apply These Terms to Your New Programs

  If you develop a new program, and you want it to be of the greatest
possible use to the public, the best way to achieve this is to make it
free software which everyone can redistribute and change under these terms.

  To do so, attach the following notices to the program.  It is safest
to attach them to the start of each source file to most effectively
state the exclusion of warranty; and each file should have at least
the "copyright" line and a pointer to where the full notice is found.

    <one line to give the program's name and a brief idea of what it does.>
    Copyright (C) <year>  <name of author>

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.

Also add information on how to contact you by electronic and paper mail.

  If the program does terminal interaction, make it output a short
notice like this when it starts in an interactive mode:

    <program>  Copyright (C) <year>  <name of author>
    This program comes with ABSOLUTELY NO WARRANTY; for details type `show w'.
    This is free software, and you are welcome to redistribute it
    under certain conditions; type `show c' for details.

The hypothetical commands `show w' and `show c' should show the appropriate
parts of the General Public License.  Of course, your program's commands
might be different; for a GUI interface, you would use an "about box".

  You should also get your employer (if you work as a programmer) or school,
if any, to sign a "copyright disclaimer" for the program, if necessary.
For more information on this, and how to apply and follow the GNU GPL, see
<https://www.gnu.org/licenses/>.

  The GNU General Public License does not permit incorporating your program
into proprietary programs.  If your program is a subroutine library, you
may consider it more useful to permit linking proprietary applications with
the library.  If this is what you want to do, use the GNU Lesser General
Public License instead of this License.  But first, please read
<https://www.gnu.org/licenses/why-not-lgpl.html>.
```


## models.py

```py
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()

class Device(db.Model):
    __tablename__ = 'devices'
    id = db.Column(db.Integer, primary_key=True)
    color = db.Column(db.String(16), nullable=False)
    friendly_name = db.Column(db.String(128), nullable=False)
    orientation = db.Column(db.String(32), nullable=False)
    address = db.Column(db.String(256), nullable=False)
    display_name = db.Column(db.String(128))
    resolution = db.Column(db.String(32))
    online = db.Column(db.Boolean, default=False)
    last_sent = db.Column(db.String(256))

    def __repr__(self):
        return f"<Device {self.friendly_name} ({self.address})>"

class ImageDB(db.Model):
    __tablename__ = 'images'
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(256), unique=True, nullable=False)
    tags = db.Column(db.String(512), nullable=True)         # comma-separated tags
    description = db.Column(db.Text, nullable=True)           # description text
    favorite = db.Column(db.Boolean, default=False)           # favorite flag

    def __repr__(self):
        return f"<ImageDB {self.filename}>"

class CropInfo(db.Model):
    __tablename__ = 'crop_info'
    filename = db.Column(db.String(256), primary_key=True)
    x = db.Column(db.Float, default=0)
    y = db.Column(db.Float, default=0)
    width = db.Column(db.Float)
    height = db.Column(db.Float)

    def __repr__(self):
        return f"<CropInfo {self.filename}>"

class SendLog(db.Model):
    __tablename__ = 'send_log'
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(256), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<SendLog {self.filename} {self.timestamp}>"

class ScheduleEvent(db.Model):
    __tablename__ = 'schedule_events'
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(256), nullable=False)
    device = db.Column(db.String(256), nullable=False)
    datetime_str = db.Column(db.String(32))
    sent = db.Column(db.Boolean, default=False)
    recurrence = db.Column(db.String(20), default="none")  # Recurrence type

    def __repr__(self):
        return f"<ScheduleEvent {self.filename} on {self.device}>"

class UserConfig(db.Model):
    __tablename__ = 'user_config'
    id = db.Column(db.Integer, primary_key=True)
    location = db.Column(db.String(256))
    openai_api_key = db.Column(db.String(512))
    ollama_address = db.Column(db.String(256))
    ollama_api_key = db.Column(db.String(512))
    ollama_model = db.Column(db.String(64))  # Column for chosen Ollama model
    clip_model = db.Column(db.String(64), default="ViT-B-32")  # Column for chosen CLIP model (using consistent format with tasks.py)

    def __repr__(self):
        return f"<UserConfig {self.id} - Location: {self.location}>"

# DeviceMetrics model removed - we only track online status now
```


## package.json

```json
{
    "name": "inkydocker",
    "version": "1.0.0",
    "description": "InkyDocker static assets",
    "main": "index.js",
    "scripts": {
      "build": "webpack --mode production",
      "dev": "webpack --mode development --watch"
    },
    "dependencies": {
      "@fullcalendar/core": "^6.1.8",
      "@fullcalendar/timegrid": "^6.1.8"
    },
    "devDependencies": {
      "webpack": "^5.0.0",
      "webpack-cli": "^4.0.0",
      "css-loader": "^6.0.0",
      "style-loader": "^3.0.0"
    },
    "author": "",
    "license": "MIT"
  }
```


## README.md

```md
# InkyDocker

InkyDocker is a Flask-based web application designed to manage and display images on e-ink displays. It allows users to upload images, schedule them for display, manage display settings, and monitor device metrics.

## Features

*   **Image Gallery**: Upload, crop, and manage images for display.
*   **Scheduled Display**: Schedule images to be displayed on e-ink displays at specific times.
*   **Device Management**: Configure and manage connected e-ink displays, including setting orientation and fetching display information.
*   **AI Settings**: Configure AI settings, such as OpenAI API key and Ollama model settings.
*   **Device Monitoring**: Monitor real-time device metrics such as CPU usage, memory usage, and disk usage.

## API Endpoints

The application exposes the following API endpoints for interacting with e-ink displays:

*   `POST /send_image`: Upload an image to display on the e-ink screen. Example:

    ```bash
    curl -F "file=@path/to/your/image.jpg" http://<IP_ADDRESS>/send_image
    ```

*   `POST /set_orientation`: Set the display orientation (choose either "horizontal" or "vertical"). Example:

    ```bash
    curl -X POST -d "orientation=vertical" http://<IP_ADDRESS>/set_orientation
    ```

*   `GET /display_info`: Retrieve display information in JSON format. Example:

    ```bash
    curl http://<IP_ADDRESS>/display_info
    ```

*   `POST /system_update`: Trigger a system update and upgrade (which will reboot the device). Example:

    ```bash
    curl -X POST http://<IP_ADDRESS>/system_update
    ```

*   `POST /backup`: Create a compressed backup of the SD card and download it. Example:

    ```bash
    curl -X POST http://<IP_ADDRESS>/backup --output backup.img.gz
    ```

*   `POST /update`: Perform a Git pull to update the application and reboot the device. Example:

    ```bash
    curl -X POST http://<IP_ADDRESS>/update
    ```

*   `GET /stream`: Connect to the SSE stream to receive real-time system metrics. Example:

    ```bash
    curl http://<IP_ADDRESS>/stream
    ```

## Requirements

*   Python 3.6+
*   Flask
*   Flask-Migrate
*   Pillow
*   Other dependencies listed in `requirements.txt`

## Installation

1.  Clone the repository:

    ```bash
    git clone <repository_url>
    cd inkydocker
    ```

2.  Create a virtual environment:

    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

4.  Configure the application:

    *   Create a `config.py` file based on the `config.example.py` file.
    *   Set the necessary environment variables, such as the database URI and API keys.

5.  Run database migrations:

    ```bash
    flask db upgrade
    ```

6.  Run the application:

    ```bash
    python app.py
    ```

## Usage

1.  Access the web application in your browser.
2.  Configure your e-ink displays in the Settings page.
3.  Upload images to the Gallery.
4.  Schedule images for display on the Schedule page.
5.  Monitor device metrics on the Settings page.

## Contributing

Contributions are welcome! Please submit a pull request with your changes.

## License

[Specify the license for your project]
```


## requirements.txt

```txt
Flask
Pillow==11.1.0
pillow-heif==0.21.0
Flask-SQLAlchemy==3.1.1
cryptography
requests
Flask-Migrate
ollama
httpx
APScheduler>=3.9.0
openai
celery
redis
open_clip_torch
scikit-learn
gunicorn
psutil>=5.9.0  # For memory monitoring
```


## scheduler.py

```py
#!/usr/bin/env python3
"""
Dedicated scheduler process for InkyDocker.

This file runs the APScheduler in a dedicated process to avoid starting
multiple schedulers in Celery worker processes. This solves the issue where
each Celery worker was starting its own scheduler, leading to duplicate jobs
and potential race conditions.

The scheduler is responsible for:
1. Running scheduled image sends based on the schedule in the database
2. Periodically checking device online status
"""

import os
import sys
import logging
from apscheduler.schedulers.background import BackgroundScheduler
from flask import Flask
from config import Config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Initialize scheduler at the module level so it can be imported by other modules
scheduler = BackgroundScheduler()

def create_app():
    """Create a minimal Flask app for the scheduler context"""
    app = Flask(__name__)
    app.config.from_object(Config)
    
    # Initialize database
    from models import db
    db.init_app(app)
    
    return app

def start_scheduler(app):
    """
    Start the APScheduler with the Flask app context.
    This function should only be called once in this dedicated process.
    """
    with app.app_context():
        # Import the necessary functions
        from tasks import fetch_device_metrics, send_scheduled_image
        from models import ScheduleEvent
        
        # Schedule the fetch_device_metrics task to run every 60 seconds (reduced from 5 seconds to prevent log flooding)
        scheduler.add_job(
            fetch_device_metrics,
            'interval',
            seconds=60,
            id='fetch_device_metrics'
        )
        
        # Load all scheduled events from the database and schedule them
        try:
            events = ScheduleEvent.query.filter_by(sent=False).all()
            for event in events:
                try:
                    import datetime
                    dt = datetime.datetime.fromisoformat(event.datetime_str)
                    if dt > datetime.datetime.now():
                        scheduler.add_job(
                            send_scheduled_image,
                            'date',
                            run_date=dt,
                            args=[event.id],
                            id=f'event_{event.id}'
                        )
                        logger.info(f"Scheduled event {event.id} for {dt}")
                except Exception as e:
                    logger.error(f"Error scheduling event {event.id}: {e}")
        except Exception as e:
            logger.error(f"Error loading scheduled events: {e}")
            logger.info("Continuing without loading scheduled events")
        
        # Start the scheduler
        scheduler.start()
        logger.info("Scheduler started successfully")
        
        # Keep the process running
        try:
            import time
            while True:
                time.sleep(60)
        except (KeyboardInterrupt, SystemExit):
            logger.info("Scheduler shutting down...")
            scheduler.shutdown()
            logger.info("Scheduler shut down successfully")

if __name__ == "__main__":
    app = create_app()
    start_scheduler(app)
```


## supervisord.conf

```conf
[supervisord]
nodaemon=true
logfile=/dev/null

[program:flask]
command=gunicorn -w 1 -t 120 --bind 0.0.0.0:5001 app:app
directory=/app
autostart=true
autorestart=true
stdout_logfile=/dev/stdout
stdout_logfile_maxbytes=0
stderr_logfile=/dev/stderr
stderr_logfile_maxbytes=0

[program:celery]
command=celery -A tasks.celery worker --loglevel=warning --max-memory-per-child=500000 -c 1
directory=/app
autostart=true
autorestart=true
stdout_logfile=/dev/stdout
stdout_logfile_maxbytes=0
stderr_logfile=/dev/stderr
stderr_logfile_maxbytes=0

# Redis is now started in entrypoint.sh before supervisord

[program:scheduler]
command=python scheduler.py
directory=/app
autostart=true
autorestart=true
stdout_logfile=/dev/stdout
stdout_logfile_maxbytes=0
stderr_logfile=/dev/stderr
stderr_logfile_maxbytes=0
```


## tasks.py

```py
import os
import datetime
import subprocess
import time
import uuid

from flask import current_app
from PIL import Image

import torch
import open_clip
from sklearn.metrics.pairwise import cosine_similarity

# Import and initialize Celery and APScheduler
from celery import Celery
from apscheduler.schedulers.background import BackgroundScheduler

# Initialize Celery with memory limits
celery = Celery('tasks', broker='redis://localhost:6379/0', backend='redis://localhost:6379/0')

# Configure Celery to prevent memory leaks and reduce log flooding
celery.conf.update(
    worker_max_memory_per_child=500000,   # 500MB memory limit per worker
    worker_max_tasks_per_child=1,         # Restart worker after each task
    task_time_limit=600,                  # 10 minute time limit per task
    task_soft_time_limit=300,             # 5 minute soft time limit (sends exception)
    worker_concurrency=1,                 # Limit to 1 concurrent worker to prevent duplicate tasks
    broker_connection_retry_on_startup=True,  # Fix deprecation warning
    worker_hijack_root_logger=False,      # Don't hijack the root logger
    worker_log_format='%(asctime)s [%(levelname)s] %(message)s',  # Simplified log format
    task_track_started=False,             # Don't log when tasks are started
    task_send_sent_event=False,           # Don't send task-sent events
    worker_send_task_events=False,        # Don't send task events
    task_ignore_result=True               # Don't store task results
)

class FlaskTask(celery.Task):
    def __call__(self, *args, **kwargs):
        # Import the app inside the task to avoid circular imports
        import sys
        import os
        # Add the current directory to the path if not already there
        app_dir = os.path.dirname(os.path.abspath(__file__))
        if app_dir not in sys.path:
            sys.path.insert(0, app_dir)
        
        # Now import the app
        from app import app as flask_app
        with flask_app.app_context():
            return self.run(*args, **kwargs)

celery.Task = FlaskTask

# Import APScheduler but don't initialize it here
# The scheduler is now initialized in a dedicated process (scheduler.py)
from apscheduler.schedulers.background import BackgroundScheduler

# Device setup
device = "cuda" if torch.cuda.is_available() else "cpu"

# Define candidate tags
CANDIDATE_TAGS = [
    "nature", "urban", "people", "animals", "food",
    "technology", "landscape", "portrait", "abstract",
    "night", "day", "indoor", "outdoor", "colorful",
    "monochrome", "black and white", "sepia", "bright", "dark",
    "warm", "cool", "red", "green", "blue", "yellow", "orange",
    "purple", "pink", "brown", "gray", "white", "black", "minimal",
    "detailed", "texture", "pattern", "geometric", "organic", "digital",
    "analog", "illustration", "painting", "photograph", "sketch",
    "drawing", "3D", "flat", "realistic", "surreal", "fantasy",
    "sci-fi", "historical", "futuristic", "retro", "classic"
]

# Model and embeddings cache
clip_models = {}
clip_preprocessors = {}
tag_embeddings = {}

def get_clip_model():
    """Get the CLIP model based on user configuration"""
    from models import UserConfig
    from flask import current_app
    import gc  # Garbage collection
    
    # Get the selected CLIP model from user config
    config = UserConfig.query.first()
    clip_model_name = 'ViT-B-32'  # Default model (smallest and fastest)
    
    # Create config if it doesn't exist
    if not config:
        from models import db
        config = UserConfig(clip_model=clip_model_name)
        db.session.add(config)
        db.session.commit()
    elif config.clip_model:
        clip_model_name = config.clip_model
    
    # If trying to use ViT-L-14 but we're low on memory, fall back to ViT-B-32
    if clip_model_name == 'ViT-L-14':
        try:
            # Check available memory (this is a simple heuristic)
            import psutil
            available_memory = psutil.virtual_memory().available / (1024 * 1024)  # MB
            if available_memory < 1000:  # Less than 1GB available
                current_app.logger.warning(f"Low memory ({available_memory}MB available). Falling back to ViT-B-32 model.")
                clip_model_name = 'ViT-B-32'
        except ImportError:
            # If psutil is not available, just continue
            pass
    
    # Check if model is already loaded
    if clip_model_name in clip_models:
        return clip_model_name, clip_models[clip_model_name], clip_preprocessors[clip_model_name]
    
    # Clear any existing models to free memory
    for model_name in list(clip_models.keys()):
        if model_name != clip_model_name:
            del clip_models[model_name]
            del clip_preprocessors[model_name]
    
    # Force garbage collection
    gc.collect()
    
    # Load the model
    try:
        current_app.logger.info(f"Loading CLIP model: {clip_model_name}")
        model, _, preprocess = open_clip.create_model_and_transforms(clip_model_name, pretrained='openai', jit=False, force_quick_gelu=True)
        model.to(device)
        model.eval()
        
        # Cache the model and preprocessor
        clip_models[clip_model_name] = model
        clip_preprocessors[clip_model_name] = preprocess
        
        # Precompute tag embeddings for this model
        tokenizer = open_clip.get_tokenizer(clip_model_name)
        if clip_model_name not in tag_embeddings:
            tag_embeddings[clip_model_name] = {}
            with torch.no_grad():
                # Process tags in smaller batches to save memory
                batch_size = 10
                for i in range(0, len(CANDIDATE_TAGS), batch_size):
                    batch_tags = CANDIDATE_TAGS[i:i+batch_size]
                    for tag in batch_tags:
                        text_tokens = tokenizer([f"a photo of {tag}"])
                        text_features = model.encode_text(text_tokens)
                        text_features /= text_features.norm(dim=-1, keepdim=True)
                        tag_embeddings[clip_model_name][tag] = text_features.cpu()  # Store on CPU to save GPU memory
                    # Force garbage collection between batches
                    gc.collect()
        
        return clip_model_name, model, preprocess
    except Exception as e:
        current_app.logger.error(f"Error loading CLIP model {clip_model_name}: {e}")
        # Fall back to default model if available
        if 'ViT-B-32' in clip_models:
            return 'ViT-B-32', clip_models['ViT-B-32'], clip_preprocessors['ViT-B-32']
        # Otherwise load default model
        model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai', jit=False, force_quick_gelu=True)
        model.to(device)
        model.eval()
        clip_models['ViT-B-32'] = model
        clip_preprocessors['ViT-B-32'] = preprocess
        return 'ViT-B-32', model, preprocess

def get_image_embedding(image_path):
    try:
        # Get the current CLIP model
        model_name, model, preprocess = get_clip_model()
        
        # Process the image
        image = Image.open(image_path).convert("RGB")
        image_input = preprocess(image).unsqueeze(0).to(device)
        
        # Get image features
        with torch.no_grad():
            embedding = model.encode_image(image_input)
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)
        
        return embedding.cpu().numpy()[0], model_name
    except Exception as e:
        try:
            current_app.logger.error(f"Error processing image {image_path}: {e}")
        except:
            print(f"Error processing image {image_path}: {e}")
        return None, None

def generate_tags_and_description(embedding, model_name):
    """Generate tags and description based on image embedding and model"""
    from flask import current_app
    
    # If model_name is None, use default
    if model_name is None:
        model_name = 'ViT-B-32'
    
    # If model embeddings not found, try to load the model
    if model_name not in tag_embeddings:
        try:
            get_clip_model()
        except Exception as e:
            current_app.logger.error(f"Error loading model for tag generation: {e}")
            # Fall back to any available model
            if len(tag_embeddings) > 0:
                model_name = list(tag_embeddings.keys())[0]
            else:
                return [], "No tags available"
    
    # Calculate similarities with tag embeddings
    scores = {}
    for tag in CANDIDATE_TAGS:
        if tag in tag_embeddings.get(model_name, {}):
            # Get the tag embedding for this model
            tag_emb = tag_embeddings[model_name][tag]
            # Calculate similarity
            similarity = torch.cosine_similarity(
                torch.tensor(embedding).unsqueeze(0),
                tag_emb.cpu(),
                dim=1
            ).item()
            scores[tag] = similarity
    
    # Sort tags by similarity score
    sorted_tags = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    # Get top 10 tags
    top_tags = [tag for tag, _ in sorted_tags[:10]]
    
    # Create description
    description = "This image may contain " + ", ".join(top_tags) + "."
    
    return top_tags, description

@celery.task(bind=True, time_limit=600, soft_time_limit=500, max_retries=3)
def process_image_tagging(self, filename):
    """
    Process an image: compute its embedding, generate tags and a description,
    and update its record in the database.
    """
    from models import ImageDB, db, UserConfig
    from flask import current_app
    import gc  # Garbage collection
    
    # Force garbage collection at the start
    gc.collect()
    
    # Log which CLIP model is being used
    config = UserConfig.query.first()
    clip_model_name = config.clip_model if config and config.clip_model else "ViT-B-32"
    current_app.logger.info(f"Tagging image {filename} with CLIP model: {clip_model_name}")
    
    # If using the large model and we're low on memory, fall back to the smaller model
    if clip_model_name == 'ViT-L-14':
        try:
            import psutil
            available_memory = psutil.virtual_memory().available / (1024 * 1024)  # MB
            if available_memory < 1000:  # Less than 1GB available
                current_app.logger.warning(f"Low memory ({available_memory}MB available). Falling back to ViT-B-32 model.")
                clip_model_name = 'ViT-B-32'
        except ImportError:
            pass
    
    image_folder = current_app.config.get("IMAGE_FOLDER", "./images")
    image_path = os.path.join(image_folder, filename)
    if not os.path.exists(image_path):
        return {"status": "error", "message": "Image file not found", "filename": filename}
    
    try:
        # Get image embedding using the selected model
        embedding, model_name = get_image_embedding(image_path)
        if embedding is None:
            return {"status": "error", "message": "Failed to compute embedding", "filename": filename}
        
        # Force garbage collection after embedding
        gc.collect()
        
        # Generate tags and description
        tags, description = generate_tags_and_description(embedding, model_name)
        
        # Force garbage collection after tag generation
        gc.collect()
        
        # Update database
        img = ImageDB.query.filter_by(filename=filename).first()
        if img:
            img.tags = ", ".join(tags)
            img.description = description
            db.session.commit()
            
            # Force garbage collection to free memory
            gc.collect()
            
            # Clear any cached models to free memory
            if model_name in clip_models and model_name != 'ViT-B-32':
                del clip_models[model_name]
                del clip_preprocessors[model_name]
                gc.collect()
            
            return {
                "status": "success",
                "filename": filename,
                "tags": tags,
                "description": description,
                "model_used": model_name
            }
        else:
            return {"status": "error", "message": "Image not found in database", "filename": filename}
    except Exception as e:
        current_app.logger.error(f"Error in process_image_tagging: {e}")
        # Try to clean up memory
        gc.collect()
        
        # Clear all cached models to free memory
        for model_name in list(clip_models.keys()):
            del clip_models[model_name]
            del clip_preprocessors[model_name]
        gc.collect()
        
        return {"status": "error", "message": str(e), "filename": filename}

def reembed_image(filename):
    """
    Recompute the embedding, tags, and description for an image and update its record.
    """
    from models import ImageDB, db, UserConfig
    from flask import current_app
    
    # Log which CLIP model is being used
    config = UserConfig.query.first()
    clip_model_name = config.clip_model if config and config.clip_model else "ViT-B-32"
    current_app.logger.info(f"Re-tagging image {filename} with CLIP model: {clip_model_name}")
    
    image_folder = current_app.config.get("IMAGE_FOLDER", "./images")
    image_path = os.path.join(image_folder, filename)
    if not os.path.exists(image_path):
        return {"status": "error", "message": "Image file not found", "filename": filename}
    
    try:
        # Get image embedding using the selected model
        embedding, model_name = get_image_embedding(image_path)
        if embedding is None:
            return {"status": "error", "message": "Failed to compute embedding", "filename": filename}
        
        # Generate tags and description
        tags, description = generate_tags_and_description(embedding, model_name)
        
        # Update database
        img = ImageDB.query.filter_by(filename=filename).first()
        if img:
            img.tags = ", ".join(tags)
            img.description = description
            db.session.commit()
            return {
                "status": "success",
                "filename": filename,
                "tags": tags,
                "description": description,
                "model_used": model_name
            }
        else:
            return {"status": "error", "message": "Image not found in database", "filename": filename}
    except Exception as e:
        current_app.logger.error(f"Error in reembed_image: {e}")
        return {"status": "error", "message": str(e), "filename": filename}

# Global dictionary to track bulk tagging progress
BULK_PROGRESS = {}

@celery.task(bind=True)
def bulk_tag_images(self):
    """
    Process all images that do not have tags. Returns a task ID that can be used to query progress.
    """
    from models import ImageDB
    from flask import current_app
    
    try:
        images = ImageDB.query.filter((ImageDB.tags == None) | (ImageDB.tags == "")).all()
        total = len(images)
        if total == 0:
            return None
        task_id = str(uuid.uuid4())
        BULK_PROGRESS[task_id] = 0
        for i, img in enumerate(images, start=1):
            process_image_tagging.delay(img.filename)
            BULK_PROGRESS[task_id] = (i / total) * 100
        return task_id
    except Exception as e:
        current_app.logger.error(f"Error in bulk_tag_images: {e}")
        return None

@celery.task(bind=True, time_limit=1800, soft_time_limit=1500)
def reembed_all_images(self):
    """
    Rerun tagging on all images using the currently selected CLIP model.
    This is useful when changing the CLIP model.
    """
    from models import ImageDB, UserConfig
    from flask import current_app
    import gc  # Garbage collection
    
    try:
        # Get the current CLIP model
        config = UserConfig.query.first()
        clip_model_name = config.clip_model if config and config.clip_model else "ViT-B-32"
        
        # Log the operation
        current_app.logger.info(f"Rerunning tagging on all images with CLIP model: {clip_model_name}")
        
        # Get all images
        images = ImageDB.query.all()
        total = len(images)
        
        if total == 0:
            return {"status": "success", "message": "No images found to tag"}
        
        # Create a task ID for tracking progress
        task_id = str(uuid.uuid4())
        BULK_PROGRESS[task_id] = 0
        
        # Process images one at a time to manage memory better
        for i, img in enumerate(images):
            # Process one image and wait for it to complete before starting the next
            process_image_tagging.delay(img.filename)
            
            # Update progress
            BULK_PROGRESS[task_id] = min(100, (i + 1) / total * 100)
            
            # Force garbage collection after each image
            gc.collect()
            
            # Add a small delay between tasks to allow memory to be freed
            time.sleep(0.5)
            
        return {
            "status": "success",
            "message": f"Started retagging {total} images with model {clip_model_name}",
            "task_id": task_id
        }
    except Exception as e:
        current_app.logger.error(f"Error in reembed_all_images: {e}")
        # Try to clean up memory
        gc.collect()
        return {"status": "error", "message": str(e)}

def send_scheduled_image(event_id):
    """
    Send a scheduled image to a device.
    """
    from models import ScheduleEvent, Device, db
    from utils.crop_helpers import load_crop_info_from_db
    from utils.image_helpers import add_send_log_entry
    from flask import current_app
    
    try:
        event = ScheduleEvent.query.get(event_id)
        if not event:
            current_app.logger.error("Event not found: %s", event_id)
            return
        device_obj = Device.query.filter_by(address=event.device).first()
        if not device_obj:
            current_app.logger.error("Device not found for event %s", event_id)
            return

        image_folder = current_app.config.get("IMAGE_FOLDER", "./images")
        data_folder = current_app.config.get("DATA_FOLDER", "./data")
        filepath = os.path.join(image_folder, event.filename)
        if not os.path.exists(filepath):
            current_app.logger.error("Image file not found: %s", filepath)
            return

        addr = device_obj.address
        if not (addr.startswith("http://") or addr.startswith("https://")):
            addr = "http://" + addr

        with Image.open(filepath) as orig_img:
            orig_w, orig_h = orig_img.size
            parts = device_obj.resolution.split("x")
            dev_width = int(parts[0])
            dev_height = int(parts[1])
            
            # Check if device is in portrait orientation
            is_portrait = device_obj.orientation.lower() == 'portrait'
            
            # If portrait, swap width and height for target ratio calculation
            if is_portrait:
                target_ratio = dev_height / dev_width
            else:
                target_ratio = dev_width / dev_height
                
            cdata = load_crop_info_from_db(event.filename)
            if cdata:
                x = cdata.get("x", 0)
                y = cdata.get("y", 0)
                w = cdata.get("width", orig_w)
                h = cdata.get("height", orig_h)
                cropped = orig_img.crop((x, y, x + w, y + h))
            else:
                orig_ratio = orig_w / orig_h
                if orig_ratio > target_ratio:
                    new_width = int(orig_h * target_ratio)
                    left = (orig_w - new_width) // 2
                    crop_box = (left, 0, left + new_width, orig_h)
                else:
                    new_height = int(orig_w / target_ratio)
                    top = (orig_h - new_height) // 2
                    crop_box = (0, top, orig_w, top + new_height)
                cropped = orig_img.crop(crop_box)
            
            # If portrait, rotate the image 90 degrees clockwise and swap dimensions
            if is_portrait:
                cropped = cropped.rotate(-90, expand=True)  # -90 for clockwise rotation
                final_img = cropped.resize((dev_height, dev_width), Image.LANCZOS)  # Note swapped dimensions
            else:
                final_img = cropped.resize((dev_width, dev_height), Image.LANCZOS)
            temp_dir = os.path.join(data_folder, "temp")
            if not os.path.exists(temp_dir):
                os.makedirs(temp_dir)
            temp_filename = os.path.join(temp_dir, f"temp_{event.filename}")
            final_img.save(temp_filename, format="JPEG", quality=95)
        
        cmd = f'curl "{addr}/send_image" -X POST -F "file=@{temp_filename}"'
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        os.remove(temp_filename)
        
        if result.returncode == 0:
            event.sent = True
            db.session.commit()
            device_obj.last_sent = event.filename
            db.session.commit()
            add_send_log_entry(event.filename)
        else:
            current_app.logger.error("Error sending image: %s", result.stderr)
    except Exception as e:
        current_app.logger.error("Error in send_scheduled_image: %s", e)
        return

    # Reschedule recurring events
    if event.recurrence and event.recurrence.lower() != "none":
        try:
            dt = datetime.datetime.fromisoformat(event.datetime_str)
        except Exception as e:
            current_app.logger.error("Error parsing datetime_str: %s", e)
            return
        if event.recurrence.lower() == "daily":
            next_dt = dt + datetime.timedelta(days=1)
        elif event.recurrence.lower() == "weekly":
            next_dt = dt + datetime.timedelta(weeks=1)
        elif event.recurrence.lower() == "monthly":
            next_dt = dt + datetime.timedelta(days=30)
        else:
            next_dt = None
        if next_dt:
            # Update the event in the database with the new date
            event.datetime_str = next_dt.isoformat(sep=' ')
            event.sent = False
            db.session.commit()
            
            # Note: We don't schedule the job here anymore
            # The dedicated scheduler process will pick up this event on its next run
            current_app.logger.info(f"Rescheduled event {event.id} for {next_dt}")

@celery.task(bind=True, ignore_result=True)
def fetch_device_metrics(self):
    """
    Check if devices are online and update their status in the database.
    This task runs periodically to ensure we have up-to-date status information.
    Logging has been minimized to prevent log flooding.
    """
    from models import Device, db
    from flask import current_app
    import httpx
    
    # Disable httpx logging
    import logging
    logging.getLogger("httpx").setLevel(logging.WARNING)
    
    try:
        devices = Device.query.all()
        status_changes = 0
        
        for device in devices:
            try:
                # Ensure the address has a scheme
                address = device.address
                if not (address.startswith("http://") or address.startswith("https://")):
                    address = "http://" + address
                
                # Store previous status to detect changes
                was_online = device.online
                
                # Try to connect to the device's display_info endpoint
                try:
                    # Use a short timeout to avoid blocking
                    response = httpx.get(f"{address}/display_info", timeout=5.0)
                    
                    if response.status_code == 200:
                        # Only update and log if status changed
                        if not was_online:
                            device.online = True
                            db.session.commit()
                            status_changes += 1
                    else:
                        # Only update and log if status changed
                        if was_online:
                            device.online = False
                            db.session.commit()
                            status_changes += 1
                except Exception:
                    # Only update if status changed
                    if was_online:
                        device.online = False
                        db.session.commit()
                        status_changes += 1
            except Exception:
                # Silently continue to next device
                pass
                
        # Only log if there were status changes
        if status_changes > 0:
            current_app.logger.info(f"Updated status for {status_changes} devices")
            
        return {"status": "success", "message": f"Checked status of {len(devices)} devices"}
    except Exception:
        # Catch all exceptions to prevent task failures
        return {"status": "error", "message": "Error checking device status"}

# The scheduler functionality has been moved to scheduler.py
# This function is kept as a stub for backward compatibility
def start_scheduler(app):
    """
    This function is deprecated and no longer starts the scheduler.
    The scheduler is now started in a dedicated process (scheduler.py).
    """
    app.logger.info("Scheduler initialization skipped - now handled by dedicated process")
```


## webpack.config.js

```js
const path = require('path');

module.exports = {
  entry: './assets/js/main.js', 
  output: {
    filename: 'bundle.js',
    path: path.resolve(__dirname, 'static', 'js')
  },
  module: {
    rules: [
      {
        test: /\.css$/i,
        use: ['style-loader', 'css-loader'],
      },
    ],
  },
};
```
```


## exportconfig.json

```json
{
    "output": "export.md",
    "description": "",
    "ignoreFile": ".export-ignore",
    "includeFile": ".export-whitelist",
    "maxFileSize": 1048576,
    "removeComments": true,
    "includeProjectStructure": true
  }
```


## LICENSE

```
GNU GENERAL PUBLIC LICENSE
                       Version 3, 29 June 2007

 Copyright (C) 2007 Free Software Foundation, Inc. <https://fsf.org/>
 Everyone is permitted to copy and distribute verbatim copies
 of this license document, but changing it is not allowed.

                            Preamble

  The GNU General Public License is a free, copyleft license for
software and other kinds of works.

  The licenses for most software and other practical works are designed
to take away your freedom to share and change the works.  By contrast,
the GNU General Public License is intended to guarantee your freedom to
share and change all versions of a program--to make sure it remains free
software for all its users.  We, the Free Software Foundation, use the
GNU General Public License for most of our software; it applies also to
any other work released this way by its authors.  You can apply it to
your programs, too.

  When we speak of free software, we are referring to freedom, not
price.  Our General Public Licenses are designed to make sure that you
have the freedom to distribute copies of free software (and charge for
them if you wish), that you receive source code or can get it if you
want it, that you can change the software or use pieces of it in new
free programs, and that you know you can do these things.

  To protect your rights, we need to prevent others from denying you
these rights or asking you to surrender the rights.  Therefore, you have
certain responsibilities if you distribute copies of the software, or if
you modify it: responsibilities to respect the freedom of others.

  For example, if you distribute copies of such a program, whether
gratis or for a fee, you must pass on to the recipients the same
freedoms that you received.  You must make sure that they, too, receive
or can get the source code.  And you must show them these terms so they
know their rights.

  Developers that use the GNU GPL protect your rights with two steps:
(1) assert copyright on the software, and (2) offer you this License
giving you legal permission to copy, distribute and/or modify it.

  For the developers' and authors' protection, the GPL clearly explains
that there is no warranty for this free software.  For both users' and
authors' sake, the GPL requires that modified versions be marked as
changed, so that their problems will not be attributed erroneously to
authors of previous versions.

  Some devices are designed to deny users access to install or run
modified versions of the software inside them, although the manufacturer
can do so.  This is fundamentally incompatible with the aim of
protecting users' freedom to change the software.  The systematic
pattern of such abuse occurs in the area of products for individuals to
use, which is precisely where it is most unacceptable.  Therefore, we
have designed this version of the GPL to prohibit the practice for those
products.  If such problems arise substantially in other domains, we
stand ready to extend this provision to those domains in future versions
of the GPL, as needed to protect the freedom of users.

  Finally, every program is threatened constantly by software patents.
States should not allow patents to restrict development and use of
software on general-purpose computers, but in those that do, we wish to
avoid the special danger that patents applied to a free program could
make it effectively proprietary.  To prevent this, the GPL assures that
patents cannot be used to render the program non-free.

  The precise terms and conditions for copying, distribution and
modification follow.

                       TERMS AND CONDITIONS

  0. Definitions.

  "This License" refers to version 3 of the GNU General Public License.

  "Copyright" also means copyright-like laws that apply to other kinds of
works, such as semiconductor masks.

  "The Program" refers to any copyrightable work licensed under this
License.  Each licensee is addressed as "you".  "Licensees" and
"recipients" may be individuals or organizations.

  To "modify" a work means to copy from or adapt all or part of the work
in a fashion requiring copyright permission, other than the making of an
exact copy.  The resulting work is called a "modified version" of the
earlier work or a work "based on" the earlier work.

  A "covered work" means either the unmodified Program or a work based
on the Program.

  To "propagate" a work means to do anything with it that, without
permission, would make you directly or secondarily liable for
infringement under applicable copyright law, except executing it on a
computer or modifying a private copy.  Propagation includes copying,
distribution (with or without modification), making available to the
public, and in some countries other activities as well.

  To "convey" a work means any kind of propagation that enables other
parties to make or receive copies.  Mere interaction with a user through
a computer network, with no transfer of a copy, is not conveying.

  An interactive user interface displays "Appropriate Legal Notices"
to the extent that it includes a convenient and prominently visible
feature that (1) displays an appropriate copyright notice, and (2)
tells the user that there is no warranty for the work (except to the
extent that warranties are provided), that licensees may convey the
work under this License, and how to view a copy of this License.  If
the interface presents a list of user commands or options, such as a
menu, a prominent item in the list meets this criterion.

  1. Source Code.

  The "source code" for a work means the preferred form of the work
for making modifications to it.  "Object code" means any non-source
form of a work.

  A "Standard Interface" means an interface that either is an official
standard defined by a recognized standards body, or, in the case of
interfaces specified for a particular programming language, one that
is widely used among developers working in that language.

  The "System Libraries" of an executable work include anything, other
than the work as a whole, that (a) is included in the normal form of
packaging a Major Component, but which is not part of that Major
Component, and (b) serves only to enable use of the work with that
Major Component, or to implement a Standard Interface for which an
implementation is available to the public in source code form.  A
"Major Component", in this context, means a major essential component
(kernel, window system, and so on) of the specific operating system
(if any) on which the executable work runs, or a compiler used to
produce the work, or an object code interpreter used to run it.

  The "Corresponding Source" for a work in object code form means all
the source code needed to generate, install, and (for an executable
work) run the object code and to modify the work, including scripts to
control those activities.  However, it does not include the work's
System Libraries, or general-purpose tools or generally available free
programs which are used unmodified in performing those activities but
which are not part of the work.  For example, Corresponding Source
includes interface definition files associated with source files for
the work, and the source code for shared libraries and dynamically
linked subprograms that the work is specifically designed to require,
such as by intimate data communication or control flow between those
subprograms and other parts of the work.

  The Corresponding Source need not include anything that users
can regenerate automatically from other parts of the Corresponding
Source.

  The Corresponding Source for a work in source code form is that
same work.

  2. Basic Permissions.

  All rights granted under this License are granted for the term of
copyright on the Program, and are irrevocable provided the stated
conditions are met.  This License explicitly affirms your unlimited
permission to run the unmodified Program.  The output from running a
covered work is covered by this License only if the output, given its
content, constitutes a covered work.  This License acknowledges your
rights of fair use or other equivalent, as provided by copyright law.

  You may make, run and propagate covered works that you do not
convey, without conditions so long as your license otherwise remains
in force.  You may convey covered works to others for the sole purpose
of having them make modifications exclusively for you, or provide you
with facilities for running those works, provided that you comply with
the terms of this License in conveying all material for which you do
not control copyright.  Those thus making or running the covered works
for you must do so exclusively on your behalf, under your direction
and control, on terms that prohibit them from making any copies of
your copyrighted material outside their relationship with you.

  Conveying under any other circumstances is permitted solely under
the conditions stated below.  Sublicensing is not allowed; section 10
makes it unnecessary.

  3. Protecting Users' Legal Rights From Anti-Circumvention Law.

  No covered work shall be deemed part of an effective technological
measure under any applicable law fulfilling obligations under article
11 of the WIPO copyright treaty adopted on 20 December 1996, or
similar laws prohibiting or restricting circumvention of such
measures.

  When you convey a covered work, you waive any legal power to forbid
circumvention of technological measures to the extent such circumvention
is effected by exercising rights under this License with respect to
the covered work, and you disclaim any intention to limit operation or
modification of the work as a means of enforcing, against the work's
users, your or third parties' legal rights to forbid circumvention of
technological measures.

  4. Conveying Verbatim Copies.

  You may convey verbatim copies of the Program's source code as you
receive it, in any medium, provided that you conspicuously and
appropriately publish on each copy an appropriate copyright notice;
keep intact all notices stating that this License and any
non-permissive terms added in accord with section 7 apply to the code;
keep intact all notices of the absence of any warranty; and give all
recipients a copy of this License along with the Program.

  You may charge any price or no price for each copy that you convey,
and you may offer support or warranty protection for a fee.

  5. Conveying Modified Source Versions.

  You may convey a work based on the Program, or the modifications to
produce it from the Program, in the form of source code under the
terms of section 4, provided that you also meet all of these conditions:

    a) The work must carry prominent notices stating that you modified
    it, and giving a relevant date.

    b) The work must carry prominent notices stating that it is
    released under this License and any conditions added under section
    7.  This requirement modifies the requirement in section 4 to
    "keep intact all notices".

    c) You must license the entire work, as a whole, under this
    License to anyone who comes into possession of a copy.  This
    License will therefore apply, along with any applicable section 7
    additional terms, to the whole of the work, and all its parts,
    regardless of how they are packaged.  This License gives no
    permission to license the work in any other way, but it does not
    invalidate such permission if you have separately received it.

    d) If the work has interactive user interfaces, each must display
    Appropriate Legal Notices; however, if the Program has interactive
    interfaces that do not display Appropriate Legal Notices, your
    work need not make them do so.

  A compilation of a covered work with other separate and independent
works, which are not by their nature extensions of the covered work,
and which are not combined with it such as to form a larger program,
in or on a volume of a storage or distribution medium, is called an
"aggregate" if the compilation and its resulting copyright are not
used to limit the access or legal rights of the compilation's users
beyond what the individual works permit.  Inclusion of a covered work
in an aggregate does not cause this License to apply to the other
parts of the aggregate.

  6. Conveying Non-Source Forms.

  You may convey a covered work in object code form under the terms
of sections 4 and 5, provided that you also convey the
machine-readable Corresponding Source under the terms of this License,
in one of these ways:

    a) Convey the object code in, or embodied in, a physical product
    (including a physical distribution medium), accompanied by the
    Corresponding Source fixed on a durable physical medium
    customarily used for software interchange.

    b) Convey the object code in, or embodied in, a physical product
    (including a physical distribution medium), accompanied by a
    written offer, valid for at least three years and valid for as
    long as you offer spare parts or customer support for that product
    model, to give anyone who possesses the object code either (1) a
    copy of the Corresponding Source for all the software in the
    product that is covered by this License, on a durable physical
    medium customarily used for software interchange, for a price no
    more than your reasonable cost of physically performing this
    conveying of source, or (2) access to copy the
    Corresponding Source from a network server at no charge.

    c) Convey individual copies of the object code with a copy of the
    written offer to provide the Corresponding Source.  This
    alternative is allowed only occasionally and noncommercially, and
    only if you received the object code with such an offer, in accord
    with subsection 6b.

    d) Convey the object code by offering access from a designated
    place (gratis or for a charge), and offer equivalent access to the
    Corresponding Source in the same way through the same place at no
    further charge.  You need not require recipients to copy the
    Corresponding Source along with the object code.  If the place to
    copy the object code is a network server, the Corresponding Source
    may be on a different server (operated by you or a third party)
    that supports equivalent copying facilities, provided you maintain
    clear directions next to the object code saying where to find the
    Corresponding Source.  Regardless of what server hosts the
    Corresponding Source, you remain obligated to ensure that it is
    available for as long as needed to satisfy these requirements.

    e) Convey the object code using peer-to-peer transmission, provided
    you inform other peers where the object code and Corresponding
    Source of the work are being offered to the general public at no
    charge under subsection 6d.

  A separable portion of the object code, whose source code is excluded
from the Corresponding Source as a System Library, need not be
included in conveying the object code work.

  A "User Product" is either (1) a "consumer product", which means any
tangible personal property which is normally used for personal, family,
or household purposes, or (2) anything designed or sold for incorporation
into a dwelling.  In determining whether a product is a consumer product,
doubtful cases shall be resolved in favor of coverage.  For a particular
product received by a particular user, "normally used" refers to a
typical or common use of that class of product, regardless of the status
of the particular user or of the way in which the particular user
actually uses, or expects or is expected to use, the product.  A product
is a consumer product regardless of whether the product has substantial
commercial, industrial or non-consumer uses, unless such uses represent
the only significant mode of use of the product.

  "Installation Information" for a User Product means any methods,
procedures, authorization keys, or other information required to install
and execute modified versions of a covered work in that User Product from
a modified version of its Corresponding Source.  The information must
suffice to ensure that the continued functioning of the modified object
code is in no case prevented or interfered with solely because
modification has been made.

  If you convey an object code work under this section in, or with, or
specifically for use in, a User Product, and the conveying occurs as
part of a transaction in which the right of possession and use of the
User Product is transferred to the recipient in perpetuity or for a
fixed term (regardless of how the transaction is characterized), the
Corresponding Source conveyed under this section must be accompanied
by the Installation Information.  But this requirement does not apply
if neither you nor any third party retains the ability to install
modified object code on the User Product (for example, the work has
been installed in ROM).

  The requirement to provide Installation Information does not include a
requirement to continue to provide support service, warranty, or updates
for a work that has been modified or installed by the recipient, or for
the User Product in which it has been modified or installed.  Access to a
network may be denied when the modification itself materially and
adversely affects the operation of the network or violates the rules and
protocols for communication across the network.

  Corresponding Source conveyed, and Installation Information provided,
in accord with this section must be in a format that is publicly
documented (and with an implementation available to the public in
source code form), and must require no special password or key for
unpacking, reading or copying.

  7. Additional Terms.

  "Additional permissions" are terms that supplement the terms of this
License by making exceptions from one or more of its conditions.
Additional permissions that are applicable to the entire Program shall
be treated as though they were included in this License, to the extent
that they are valid under applicable law.  If additional permissions
apply only to part of the Program, that part may be used separately
under those permissions, but the entire Program remains governed by
this License without regard to the additional permissions.

  When you convey a copy of a covered work, you may at your option
remove any additional permissions from that copy, or from any part of
it.  (Additional permissions may be written to require their own
removal in certain cases when you modify the work.)  You may place
additional permissions on material, added by you to a covered work,
for which you have or can give appropriate copyright permission.

  Notwithstanding any other provision of this License, for material you
add to a covered work, you may (if authorized by the copyright holders of
that material) supplement the terms of this License with terms:

    a) Disclaiming warranty or limiting liability differently from the
    terms of sections 15 and 16 of this License; or

    b) Requiring preservation of specified reasonable legal notices or
    author attributions in that material or in the Appropriate Legal
    Notices displayed by works containing it; or

    c) Prohibiting misrepresentation of the origin of that material, or
    requiring that modified versions of such material be marked in
    reasonable ways as different from the original version; or

    d) Limiting the use for publicity purposes of names of licensors or
    authors of the material; or

    e) Declining to grant rights under trademark law for use of some
    trade names, trademarks, or service marks; or

    f) Requiring indemnification of licensors and authors of that
    material by anyone who conveys the material (or modified versions of
    it) with contractual assumptions of liability to the recipient, for
    any liability that these contractual assumptions directly impose on
    those licensors and authors.

  All other non-permissive additional terms are considered "further
restrictions" within the meaning of section 10.  If the Program as you
received it, or any part of it, contains a notice stating that it is
governed by this License along with a term that is a further
restriction, you may remove that term.  If a license document contains
a further restriction but permits relicensing or conveying under this
License, you may add to a covered work material governed by the terms
of that license document, provided that the further restriction does
not survive such relicensing or conveying.

  If you add terms to a covered work in accord with this section, you
must place, in the relevant source files, a statement of the
additional terms that apply to those files, or a notice indicating
where to find the applicable terms.

  Additional terms, permissive or non-permissive, may be stated in the
form of a separately written license, or stated as exceptions;
the above requirements apply either way.

  8. Termination.

  You may not propagate or modify a covered work except as expressly
provided under this License.  Any attempt otherwise to propagate or
modify it is void, and will automatically terminate your rights under
this License (including any patent licenses granted under the third
paragraph of section 11).

  However, if you cease all violation of this License, then your
license from a particular copyright holder is reinstated (a)
provisionally, unless and until the copyright holder explicitly and
finally terminates your license, and (b) permanently, if the copyright
holder fails to notify you of the violation by some reasonable means
prior to 60 days after the cessation.

  Moreover, your license from a particular copyright holder is
reinstated permanently if the copyright holder notifies you of the
violation by some reasonable means, this is the first time you have
received notice of violation of this License (for any work) from that
copyright holder, and you cure the violation prior to 30 days after
your receipt of the notice.

  Termination of your rights under this section does not terminate the
licenses of parties who have received copies or rights from you under
this License.  If your rights have been terminated and not permanently
reinstated, you do not qualify to receive new licenses for the same
material under section 10.

  9. Acceptance Not Required for Having Copies.

  You are not required to accept this License in order to receive or
run a copy of the Program.  Ancillary propagation of a covered work
occurring solely as a consequence of using peer-to-peer transmission
to receive a copy likewise does not require acceptance.  However,
nothing other than this License grants you permission to propagate or
modify any covered work.  These actions infringe copyright if you do
not accept this License.  Therefore, by modifying or propagating a
covered work, you indicate your acceptance of this License to do so.

  10. Automatic Licensing of Downstream Recipients.

  Each time you convey a covered work, the recipient automatically
receives a license from the original licensors, to run, modify and
propagate that work, subject to this License.  You are not responsible
for enforcing compliance by third parties with this License.

  An "entity transaction" is a transaction transferring control of an
organization, or substantially all assets of one, or subdividing an
organization, or merging organizations.  If propagation of a covered
work results from an entity transaction, each party to that
transaction who receives a copy of the work also receives whatever
licenses to the work the party's predecessor in interest had or could
give under the previous paragraph, plus a right to possession of the
Corresponding Source of the work from the predecessor in interest, if
the predecessor has it or can get it with reasonable efforts.

  You may not impose any further restrictions on the exercise of the
rights granted or affirmed under this License.  For example, you may
not impose a license fee, royalty, or other charge for exercise of
rights granted under this License, and you may not initiate litigation
(including a cross-claim or counterclaim in a lawsuit) alleging that
any patent claim is infringed by making, using, selling, offering for
sale, or importing the Program or any portion of it.

  11. Patents.

  A "contributor" is a copyright holder who authorizes use under this
License of the Program or a work on which the Program is based.  The
work thus licensed is called the contributor's "contributor version".

  A contributor's "essential patent claims" are all patent claims
owned or controlled by the contributor, whether already acquired or
hereafter acquired, that would be infringed by some manner, permitted
by this License, of making, using, or selling its contributor version,
but do not include claims that would be infringed only as a
consequence of further modification of the contributor version.  For
purposes of this definition, "control" includes the right to grant
patent sublicenses in a manner consistent with the requirements of
this License.

  Each contributor grants you a non-exclusive, worldwide, royalty-free
patent license under the contributor's essential patent claims, to
make, use, sell, offer for sale, import and otherwise run, modify and
propagate the contents of its contributor version.

  In the following three paragraphs, a "patent license" is any express
agreement or commitment, however denominated, not to enforce a patent
(such as an express permission to practice a patent or covenant not to
sue for patent infringement).  To "grant" such a patent license to a
party means to make such an agreement or commitment not to enforce a
patent against the party.

  If you convey a covered work, knowingly relying on a patent license,
and the Corresponding Source of the work is not available for anyone
to copy, free of charge and under the terms of this License, through a
publicly available network server or other readily accessible means,
then you must either (1) cause the Corresponding Source to be so
available, or (2) arrange to deprive yourself of the benefit of the
patent license for this particular work, or (3) arrange, in a manner
consistent with the requirements of this License, to extend the patent
license to downstream recipients.  "Knowingly relying" means you have
actual knowledge that, but for the patent license, your conveying the
covered work in a country, or your recipient's use of the covered work
in a country, would infringe one or more identifiable patents in that
country that you have reason to believe are valid.

  If, pursuant to or in connection with a single transaction or
arrangement, you convey, or propagate by procuring conveyance of, a
covered work, and grant a patent license to some of the parties
receiving the covered work authorizing them to use, propagate, modify
or convey a specific copy of the covered work, then the patent license
you grant is automatically extended to all recipients of the covered
work and works based on it.

  A patent license is "discriminatory" if it does not include within
the scope of its coverage, prohibits the exercise of, or is
conditioned on the non-exercise of one or more of the rights that are
specifically granted under this License.  You may not convey a covered
work if you are a party to an arrangement with a third party that is
in the business of distributing software, under which you make payment
to the third party based on the extent of your activity of conveying
the work, and under which the third party grants, to any of the
parties who would receive the covered work from you, a discriminatory
patent license (a) in connection with copies of the covered work
conveyed by you (or copies made from those copies), or (b) primarily
for and in connection with specific products or compilations that
contain the covered work, unless you entered into that arrangement,
or that patent license was granted, prior to 28 March 2007.

  Nothing in this License shall be construed as excluding or limiting
any implied license or other defenses to infringement that may
otherwise be available to you under applicable patent law.

  12. No Surrender of Others' Freedom.

  If conditions are imposed on you (whether by court order, agreement or
otherwise) that contradict the conditions of this License, they do not
excuse you from the conditions of this License.  If you cannot convey a
covered work so as to satisfy simultaneously your obligations under this
License and any other pertinent obligations, then as a consequence you may
not convey it at all.  For example, if you agree to terms that obligate you
to collect a royalty for further conveying from those to whom you convey
the Program, the only way you could satisfy both those terms and this
License would be to refrain entirely from conveying the Program.

  13. Use with the GNU Affero General Public License.

  Notwithstanding any other provision of this License, you have
permission to link or combine any covered work with a work licensed
under version 3 of the GNU Affero General Public License into a single
combined work, and to convey the resulting work.  The terms of this
License will continue to apply to the part which is the covered work,
but the special requirements of the GNU Affero General Public License,
section 13, concerning interaction through a network will apply to the
combination as such.

  14. Revised Versions of this License.

  The Free Software Foundation may publish revised and/or new versions of
the GNU General Public License from time to time.  Such new versions will
be similar in spirit to the present version, but may differ in detail to
address new problems or concerns.

  Each version is given a distinguishing version number.  If the
Program specifies that a certain numbered version of the GNU General
Public License "or any later version" applies to it, you have the
option of following the terms and conditions either of that numbered
version or of any later version published by the Free Software
Foundation.  If the Program does not specify a version number of the
GNU General Public License, you may choose any version ever published
by the Free Software Foundation.

  If the Program specifies that a proxy can decide which future
versions of the GNU General Public License can be used, that proxy's
public statement of acceptance of a version permanently authorizes you
to choose that version for the Program.

  Later license versions may give you additional or different
permissions.  However, no additional obligations are imposed on any
author or copyright holder as a result of your choosing to follow a
later version.

  15. Disclaimer of Warranty.

  THERE IS NO WARRANTY FOR THE PROGRAM, TO THE EXTENT PERMITTED BY
APPLICABLE LAW.  EXCEPT WHEN OTHERWISE STATED IN WRITING THE COPYRIGHT
HOLDERS AND/OR OTHER PARTIES PROVIDE THE PROGRAM "AS IS" WITHOUT WARRANTY
OF ANY KIND, EITHER EXPRESSED OR IMPLIED, INCLUDING, BUT NOT LIMITED TO,
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
PURPOSE.  THE ENTIRE RISK AS TO THE QUALITY AND PERFORMANCE OF THE PROGRAM
IS WITH YOU.  SHOULD THE PROGRAM PROVE DEFECTIVE, YOU ASSUME THE COST OF
ALL NECESSARY SERVICING, REPAIR OR CORRECTION.

  16. Limitation of Liability.

  IN NO EVENT UNLESS REQUIRED BY APPLICABLE LAW OR AGREED TO IN WRITING
WILL ANY COPYRIGHT HOLDER, OR ANY OTHER PARTY WHO MODIFIES AND/OR CONVEYS
THE PROGRAM AS PERMITTED ABOVE, BE LIABLE TO YOU FOR DAMAGES, INCLUDING ANY
GENERAL, SPECIAL, INCIDENTAL OR CONSEQUENTIAL DAMAGES ARISING OUT OF THE
USE OR INABILITY TO USE THE PROGRAM (INCLUDING BUT NOT LIMITED TO LOSS OF
DATA OR DATA BEING RENDERED INACCURATE OR LOSSES SUSTAINED BY YOU OR THIRD
PARTIES OR A FAILURE OF THE PROGRAM TO OPERATE WITH ANY OTHER PROGRAMS),
EVEN IF SUCH HOLDER OR OTHER PARTY HAS BEEN ADVISED OF THE POSSIBILITY OF
SUCH DAMAGES.

  17. Interpretation of Sections 15 and 16.

  If the disclaimer of warranty and limitation of liability provided
above cannot be given local legal effect according to their terms,
reviewing courts shall apply local law that most closely approximates
an absolute waiver of all civil liability in connection with the
Program, unless a warranty or assumption of liability accompanies a
copy of the Program in return for a fee.

                     END OF TERMS AND CONDITIONS

            How to Apply These Terms to Your New Programs

  If you develop a new program, and you want it to be of the greatest
possible use to the public, the best way to achieve this is to make it
free software which everyone can redistribute and change under these terms.

  To do so, attach the following notices to the program.  It is safest
to attach them to the start of each source file to most effectively
state the exclusion of warranty; and each file should have at least
the "copyright" line and a pointer to where the full notice is found.

    <one line to give the program's name and a brief idea of what it does.>
    Copyright (C) <year>  <name of author>

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.

Also add information on how to contact you by electronic and paper mail.

  If the program does terminal interaction, make it output a short
notice like this when it starts in an interactive mode:

    <program>  Copyright (C) <year>  <name of author>
    This program comes with ABSOLUTELY NO WARRANTY; for details type `show w'.
    This is free software, and you are welcome to redistribute it
    under certain conditions; type `show c' for details.

The hypothetical commands `show w' and `show c' should show the appropriate
parts of the General Public License.  Of course, your program's commands
might be different; for a GUI interface, you would use an "about box".

  You should also get your employer (if you work as a programmer) or school,
if any, to sign a "copyright disclaimer" for the program, if necessary.
For more information on this, and how to apply and follow the GNU GPL, see
<https://www.gnu.org/licenses/>.

  The GNU General Public License does not permit incorporating your program
into proprietary programs.  If your program is a subroutine library, you
may consider it more useful to permit linking proprietary applications with
the library.  If this is what you want to do, use the GNU Lesser General
Public License instead of this License.  But first, please read
<https://www.gnu.org/licenses/why-not-lgpl.html>.
```


## models.py

```py
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()

class Device(db.Model):
    __tablename__ = 'devices'
    id = db.Column(db.Integer, primary_key=True)
    color = db.Column(db.String(16), nullable=False)
    friendly_name = db.Column(db.String(128), nullable=False)
    orientation = db.Column(db.String(32), nullable=False)
    address = db.Column(db.String(256), nullable=False)
    display_name = db.Column(db.String(128))
    resolution = db.Column(db.String(32))
    online = db.Column(db.Boolean, default=False)
    last_sent = db.Column(db.String(256))

    def __repr__(self):
        return f"<Device {self.friendly_name} ({self.address})>"

class ImageDB(db.Model):
    __tablename__ = 'images'
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(256), unique=True, nullable=False)
    tags = db.Column(db.String(512), nullable=True)         # comma-separated tags
    description = db.Column(db.Text, nullable=True)           # description text
    favorite = db.Column(db.Boolean, default=False)           # favorite flag

    def __repr__(self):
        return f"<ImageDB {self.filename}>"

class CropInfo(db.Model):
    __tablename__ = 'crop_info'
    filename = db.Column(db.String(256), primary_key=True)
    x = db.Column(db.Float, default=0)
    y = db.Column(db.Float, default=0)
    width = db.Column(db.Float)
    height = db.Column(db.Float)
    resolution = db.Column(db.String(32))  # Store the display resolution (e.g., "1024x768")

    def __repr__(self):
        return f"<CropInfo {self.filename}>"

class SendLog(db.Model):
    __tablename__ = 'send_log'
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(256), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<SendLog {self.filename} {self.timestamp}>"

class ScheduleEvent(db.Model):
    __tablename__ = 'schedule_events'
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(256), nullable=False)
    device = db.Column(db.String(256), nullable=False)
    datetime_str = db.Column(db.String(32))
    sent = db.Column(db.Boolean, default=False)
    recurrence = db.Column(db.String(20), default="none")  # Recurrence type

    def __repr__(self):
        return f"<ScheduleEvent {self.filename} on {self.device}>"

class UserConfig(db.Model):
    __tablename__ = 'user_config'
    id = db.Column(db.Integer, primary_key=True)
    location = db.Column(db.String(256))
    openai_api_key = db.Column(db.String(512))
    ollama_address = db.Column(db.String(256))
    ollama_api_key = db.Column(db.String(512))
    ollama_model = db.Column(db.String(64))  # Column for chosen Ollama model
    clip_model = db.Column(db.String(64), default="ViT-B-32")  # Column for chosen CLIP model (using consistent format with tasks.py)

    def __repr__(self):
        return f"<UserConfig {self.id} - Location: {self.location}>"

# DeviceMetrics model removed - we only track online status now
```


## package.json

```json
{
    "name": "inkydocker",
    "version": "1.0.0",
    "description": "InkyDocker static assets",
    "main": "index.js",
    "scripts": {
      "build": "webpack --mode production",
      "dev": "webpack --mode development --watch"
    },
    "dependencies": {
      "@fullcalendar/core": "^6.1.8",
      "@fullcalendar/timegrid": "^6.1.8"
    },
    "devDependencies": {
      "webpack": "^5.0.0",
      "webpack-cli": "^4.0.0",
      "css-loader": "^6.0.0",
      "style-loader": "^3.0.0"
    },
    "author": "",
    "license": "MIT"
  }
```


## README.md

```md
# InkyDocker

InkyDocker is a Flask-based web application designed to manage and display images on e-ink displays. It allows users to upload images, schedule them for display, manage display settings, and monitor device metrics.

## Features

*   **Image Gallery**: Upload, crop, and manage images for display.
*   **Scheduled Display**: Schedule images to be displayed on e-ink displays at specific times.
*   **Device Management**: Configure and manage connected e-ink displays, including setting orientation and fetching display information.
*   **AI Settings**: Configure AI settings, such as OpenAI API key and Ollama model settings.
*   **Device Monitoring**: Monitor real-time device metrics such as CPU usage, memory usage, and disk usage.

## API Endpoints

The application exposes the following API endpoints for interacting with e-ink displays:

*   `POST /send_image`: Upload an image to display on the e-ink screen. Example:

    ```bash
    curl -F "file=@path/to/your/image.jpg" http://<IP_ADDRESS>/send_image
    ```

*   `POST /set_orientation`: Set the display orientation (choose either "horizontal" or "vertical"). Example:

    ```bash
    curl -X POST -d "orientation=vertical" http://<IP_ADDRESS>/set_orientation
    ```

*   `GET /display_info`: Retrieve display information in JSON format. Example:

    ```bash
    curl http://<IP_ADDRESS>/display_info
    ```

*   `POST /system_update`: Trigger a system update and upgrade (which will reboot the device). Example:

    ```bash
    curl -X POST http://<IP_ADDRESS>/system_update
    ```

*   `POST /backup`: Create a compressed backup of the SD card and download it. Example:

    ```bash
    curl -X POST http://<IP_ADDRESS>/backup --output backup.img.gz
    ```

*   `POST /update`: Perform a Git pull to update the application and reboot the device. Example:

    ```bash
    curl -X POST http://<IP_ADDRESS>/update
    ```

*   `GET /stream`: Connect to the SSE stream to receive real-time system metrics. Example:

    ```bash
    curl http://<IP_ADDRESS>/stream
    ```

## Requirements

*   Python 3.6+
*   Flask
*   Flask-Migrate
*   Pillow
*   Other dependencies listed in `requirements.txt`

## Installation

1.  Clone the repository:

    ```bash
    git clone <repository_url>
    cd inkydocker
    ```

2.  Create a virtual environment:

    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

4.  Configure the application:

    *   Create a `config.py` file based on the `config.example.py` file.
    *   Set the necessary environment variables, such as the database URI and API keys.

5.  Run database migrations:

    ```bash
    flask db upgrade
    ```

6.  Run the application:

    ```bash
    python app.py
    ```

## Usage

1.  Access the web application in your browser.
2.  Configure your e-ink displays in the Settings page.
3.  Upload images to the Gallery.
4.  Schedule images for display on the Schedule page.
5.  Monitor device metrics on the Settings page.

## Contributing

Contributions are welcome! Please submit a pull request with your changes.

## License

[Specify the license for your project]
```


## requirements.txt

```txt
Flask
Pillow==11.1.0
pillow-heif==0.21.0
Flask-SQLAlchemy==3.1.1
cryptography
requests
Flask-Migrate
ollama
httpx
APScheduler>=3.9.0
openai
celery
redis
open_clip_torch
scikit-learn
gunicorn
psutil>=5.9.0  # For memory monitoring
```


## scheduler.py

```py
#!/usr/bin/env python3
"""
Dedicated scheduler process for InkyDocker.

This file runs the APScheduler in a dedicated process to avoid starting
multiple schedulers in Celery worker processes. This solves the issue where
each Celery worker was starting its own scheduler, leading to duplicate jobs
and potential race conditions.

The scheduler is responsible for:
1. Running scheduled image sends based on the schedule in the database
2. Periodically checking device online status
"""

import os
import sys
import logging
from apscheduler.schedulers.background import BackgroundScheduler
from flask import Flask
from config import Config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Initialize scheduler at the module level so it can be imported by other modules
# Configure with optimized settings for better performance
scheduler = BackgroundScheduler(
    job_defaults={
        'coalesce': True,  # Combine multiple pending executions of the same job into one
        'max_instances': 1,  # Only allow one instance of each job to run at a time
        'misfire_grace_time': 3600  # Allow misfires up to 1 hour
    },
    executor='threadpool',  # Use threadpool executor for better performance
    timezone='UTC'  # Use UTC for consistent timezone handling
)

def create_app():
    """Create a minimal Flask app for the scheduler context"""
    app = Flask(__name__)
    app.config.from_object(Config)
    
    # Initialize database
    from models import db
    db.init_app(app)
    
    return app

def start_scheduler(app):
    """
    Start the APScheduler with the Flask app context.
    This function should only be called once in this dedicated process.
    """
    with app.app_context():
        # Import the necessary functions
        from tasks import fetch_device_metrics, send_scheduled_image
        from models import ScheduleEvent
        
        # Schedule the fetch_device_metrics task to run every 60 seconds (reduced from 5 seconds to prevent log flooding)
        scheduler.add_job(
            fetch_device_metrics,
            'interval',
            seconds=60,
            id='fetch_device_metrics'
        )
        
        # Load all scheduled events from the database and schedule them
        try:
            events = ScheduleEvent.query.filter_by(sent=False).all()
            for event in events:
                try:
                    import datetime
                    dt = datetime.datetime.fromisoformat(event.datetime_str)
                    if dt > datetime.datetime.now():
                        scheduler.add_job(
                            send_scheduled_image,
                            'date',
                            run_date=dt,
                            args=[event.id],
                            id=f'event_{event.id}'
                        )
                        logger.info(f"Scheduled event {event.id} for {dt}")
                except Exception as e:
                    logger.error(f"Error scheduling event {event.id}: {e}")
        except Exception as e:
            logger.error(f"Error loading scheduled events: {e}")
            logger.info("Continuing without loading scheduled events")
        
        # Start the scheduler
        scheduler.start()
        logger.info("Scheduler started successfully")
        
        # Keep the process running
        try:
            import time
            while True:
                time.sleep(60)
        except (KeyboardInterrupt, SystemExit):
            logger.info("Scheduler shutting down...")
            scheduler.shutdown()
            logger.info("Scheduler shut down successfully")

if __name__ == "__main__":
    app = create_app()
    start_scheduler(app)
```


## supervisord.conf

```conf
[supervisord]
nodaemon=true
logfile=/dev/null

[program:flask]
command=gunicorn -w 1 -t 120 --bind 0.0.0.0:5001 app:app
directory=/app
autostart=true
autorestart=true
stdout_logfile=/dev/stdout
stdout_logfile_maxbytes=0
stderr_logfile=/dev/stderr
stderr_logfile_maxbytes=0

[program:celery]
command=celery -A tasks.celery worker --loglevel=warning --max-memory-per-child=500000 -c 1
directory=/app
autostart=true
autorestart=true
stdout_logfile=/dev/stdout
stdout_logfile_maxbytes=0
stderr_logfile=/dev/stderr
stderr_logfile_maxbytes=0

# Redis is now started in entrypoint.sh before supervisord

[program:scheduler]
command=python scheduler.py
directory=/app
autostart=true
autorestart=true
stdout_logfile=/dev/stdout
stdout_logfile_maxbytes=0
stderr_logfile=/dev/stderr
stderr_logfile_maxbytes=0
```


## tasks.py

```py
import os
import datetime
import subprocess
import time
import uuid

from flask import current_app
from PIL import Image

import torch
import open_clip
from sklearn.metrics.pairwise import cosine_similarity

# Import and initialize Celery and APScheduler
from celery import Celery
from apscheduler.schedulers.background import BackgroundScheduler

# Initialize Celery with memory limits
celery = Celery('tasks', broker='redis://localhost:6379/0', backend='redis://localhost:6379/0')

# Configure Celery to prevent memory leaks and reduce log flooding
celery.conf.update(
    worker_max_memory_per_child=500000,   # 500MB memory limit per worker
    worker_max_tasks_per_child=1,         # Restart worker after each task
    task_time_limit=600,                  # 10 minute time limit per task
    task_soft_time_limit=300,             # 5 minute soft time limit (sends exception)
    worker_concurrency=1,                 # Limit to 1 concurrent worker to prevent duplicate tasks
    broker_connection_retry_on_startup=True,  # Fix deprecation warning
    worker_hijack_root_logger=False,      # Don't hijack the root logger
    worker_log_format='%(asctime)s [%(levelname)s] %(message)s',  # Simplified log format
    task_track_started=False,             # Don't log when tasks are started
    task_send_sent_event=False,           # Don't send task-sent events
    worker_send_task_events=False,        # Don't send task events
    task_ignore_result=True               # Don't store task results
)

class FlaskTask(celery.Task):
    def __call__(self, *args, **kwargs):
        # Import the app inside the task to avoid circular imports
        import sys
        import os
        # Add the current directory to the path if not already there
        app_dir = os.path.dirname(os.path.abspath(__file__))
        if app_dir not in sys.path:
            sys.path.insert(0, app_dir)
        
        # Now import the app
        from app import app as flask_app
        with flask_app.app_context():
            return self.run(*args, **kwargs)

celery.Task = FlaskTask

# Import APScheduler but don't initialize it here
# The scheduler is now initialized in a dedicated process (scheduler.py)
from apscheduler.schedulers.background import BackgroundScheduler

# Device setup
device = "cuda" if torch.cuda.is_available() else "cpu"

# Define candidate tags
CANDIDATE_TAGS = [
    "nature", "urban", "people", "animals", "food",
    "technology", "landscape", "portrait", "abstract",
    "night", "day", "indoor", "outdoor", "colorful",
    "monochrome", "black and white", "sepia", "bright", "dark",
    "warm", "cool", "red", "green", "blue", "yellow", "orange",
    "purple", "pink", "brown", "gray", "white", "black", "minimal",
    "detailed", "texture", "pattern", "geometric", "organic", "digital",
    "analog", "illustration", "painting", "photograph", "sketch",
    "drawing", "3D", "flat", "realistic", "surreal", "fantasy",
    "sci-fi", "historical", "futuristic", "retro", "classic"
]

# Model and embeddings cache
clip_models = {}
clip_preprocessors = {}
tag_embeddings = {}

def get_clip_model():
    """Get the CLIP model based on user configuration"""
    from models import UserConfig
    from flask import current_app
    import gc  # Garbage collection
    
    # Get the selected CLIP model from user config
    config = UserConfig.query.first()
    clip_model_name = 'ViT-B-32'  # Default model (smallest and fastest)
    
    # Create config if it doesn't exist
    if not config:
        from models import db
        config = UserConfig(clip_model=clip_model_name)
        db.session.add(config)
        db.session.commit()
    elif config.clip_model:
        clip_model_name = config.clip_model
    
    # If trying to use ViT-L-14 but we're low on memory, fall back to ViT-B-32
    if clip_model_name == 'ViT-L-14':
        try:
            # Check available memory (this is a simple heuristic)
            import psutil
            available_memory = psutil.virtual_memory().available / (1024 * 1024)  # MB
            if available_memory < 1000:  # Less than 1GB available
                current_app.logger.warning(f"Low memory ({available_memory}MB available). Falling back to ViT-B-32 model.")
                clip_model_name = 'ViT-B-32'
        except ImportError:
            # If psutil is not available, just continue
            pass
    
    # Check if model is already loaded
    if clip_model_name in clip_models:
        return clip_model_name, clip_models[clip_model_name], clip_preprocessors[clip_model_name]
    
    # Clear any existing models to free memory
    for model_name in list(clip_models.keys()):
        if model_name != clip_model_name:
            del clip_models[model_name]
            del clip_preprocessors[model_name]
    
    # Force garbage collection
    gc.collect()
    
    # Load the model
    try:
        current_app.logger.info(f"Loading CLIP model: {clip_model_name}")
        model, _, preprocess = open_clip.create_model_and_transforms(clip_model_name, pretrained='openai', jit=False, force_quick_gelu=True)
        model.to(device)
        model.eval()
        
        # Cache the model and preprocessor
        clip_models[clip_model_name] = model
        clip_preprocessors[clip_model_name] = preprocess
        
        # Precompute tag embeddings for this model
        tokenizer = open_clip.get_tokenizer(clip_model_name)
        if clip_model_name not in tag_embeddings:
            tag_embeddings[clip_model_name] = {}
            with torch.no_grad():
                # Process tags in smaller batches to save memory
                batch_size = 10
                for i in range(0, len(CANDIDATE_TAGS), batch_size):
                    batch_tags = CANDIDATE_TAGS[i:i+batch_size]
                    for tag in batch_tags:
                        text_tokens = tokenizer([f"a photo of {tag}"])
                        text_features = model.encode_text(text_tokens)
                        text_features /= text_features.norm(dim=-1, keepdim=True)
                        tag_embeddings[clip_model_name][tag] = text_features.cpu()  # Store on CPU to save GPU memory
                    # Force garbage collection between batches
                    gc.collect()
        
        return clip_model_name, model, preprocess
    except Exception as e:
        current_app.logger.error(f"Error loading CLIP model {clip_model_name}: {e}")
        # Fall back to default model if available
        if 'ViT-B-32' in clip_models:
            return 'ViT-B-32', clip_models['ViT-B-32'], clip_preprocessors['ViT-B-32']
        # Otherwise load default model
        model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai', jit=False, force_quick_gelu=True)
        model.to(device)
        model.eval()
        clip_models['ViT-B-32'] = model
        clip_preprocessors['ViT-B-32'] = preprocess
        return 'ViT-B-32', model, preprocess

def get_image_embedding(image_path):
    try:
        # Get the current CLIP model
        model_name, model, preprocess = get_clip_model()
        
        # Process the image
        image = Image.open(image_path).convert("RGB")
        image_input = preprocess(image).unsqueeze(0).to(device)
        
        # Get image features
        with torch.no_grad():
            embedding = model.encode_image(image_input)
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)
        
        return embedding.cpu().numpy()[0], model_name
    except Exception as e:
        try:
            current_app.logger.error(f"Error processing image {image_path}: {e}")
        except:
            print(f"Error processing image {image_path}: {e}")
        return None, None

def generate_tags_and_description(embedding, model_name):
    """Generate tags and description based on image embedding and model"""
    from flask import current_app
    
    # If model_name is None, use default
    if model_name is None:
        model_name = 'ViT-B-32'
    
    # If model embeddings not found, try to load the model
    if model_name not in tag_embeddings:
        try:
            get_clip_model()
        except Exception as e:
            current_app.logger.error(f"Error loading model for tag generation: {e}")
            # Fall back to any available model
            if len(tag_embeddings) > 0:
                model_name = list(tag_embeddings.keys())[0]
            else:
                return [], "No tags available"
    
    # Calculate similarities with tag embeddings
    scores = {}
    for tag in CANDIDATE_TAGS:
        if tag in tag_embeddings.get(model_name, {}):
            # Get the tag embedding for this model
            tag_emb = tag_embeddings[model_name][tag]
            # Calculate similarity
            similarity = torch.cosine_similarity(
                torch.tensor(embedding).unsqueeze(0),
                tag_emb.cpu(),
                dim=1
            ).item()
            scores[tag] = similarity
    
    # Sort tags by similarity score
    sorted_tags = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    # Get top 10 tags
    top_tags = [tag for tag, _ in sorted_tags[:10]]
    
    # Create description
    description = "This image may contain " + ", ".join(top_tags) + "."
    
    return top_tags, description

@celery.task(bind=True, time_limit=600, soft_time_limit=500, max_retries=3)
def process_image_tagging(self, filename):
    """
    Process an image: compute its embedding, generate tags and a description,
    and update its record in the database.
    """
    from models import ImageDB, db, UserConfig
    from flask import current_app
    import gc  # Garbage collection
    
    # Force garbage collection at the start
    gc.collect()
    
    # Log which CLIP model is being used
    config = UserConfig.query.first()
    clip_model_name = config.clip_model if config and config.clip_model else "ViT-B-32"
    current_app.logger.info(f"Tagging image {filename} with CLIP model: {clip_model_name}")
    
    # If using the large model and we're low on memory, fall back to the smaller model
    if clip_model_name == 'ViT-L-14':
        try:
            import psutil
            available_memory = psutil.virtual_memory().available / (1024 * 1024)  # MB
            if available_memory < 1000:  # Less than 1GB available
                current_app.logger.warning(f"Low memory ({available_memory}MB available). Falling back to ViT-B-32 model.")
                clip_model_name = 'ViT-B-32'
        except ImportError:
            pass
    
    image_folder = current_app.config.get("IMAGE_FOLDER", "./images")
    image_path = os.path.join(image_folder, filename)
    if not os.path.exists(image_path):
        return {"status": "error", "message": "Image file not found", "filename": filename}
    
    try:
        # Get image embedding using the selected model
        embedding, model_name = get_image_embedding(image_path)
        if embedding is None:
            return {"status": "error", "message": "Failed to compute embedding", "filename": filename}
        
        # Force garbage collection after embedding
        gc.collect()
        
        # Generate tags and description
        tags, description = generate_tags_and_description(embedding, model_name)
        
        # Force garbage collection after tag generation
        gc.collect()
        
        # Update database
        img = ImageDB.query.filter_by(filename=filename).first()
        if img:
            img.tags = ", ".join(tags)
            img.description = description
            db.session.commit()
            
            # Force garbage collection to free memory
            gc.collect()
            
            # Clear any cached models to free memory
            if model_name in clip_models and model_name != 'ViT-B-32':
                del clip_models[model_name]
                del clip_preprocessors[model_name]
                gc.collect()
            
            return {
                "status": "success",
                "filename": filename,
                "tags": tags,
                "description": description,
                "model_used": model_name
            }
        else:
            return {"status": "error", "message": "Image not found in database", "filename": filename}
    except Exception as e:
        current_app.logger.error(f"Error in process_image_tagging: {e}")
        # Try to clean up memory
        gc.collect()
        
        # Clear all cached models to free memory
        for model_name in list(clip_models.keys()):
            del clip_models[model_name]
            del clip_preprocessors[model_name]
        gc.collect()
        
        return {"status": "error", "message": str(e), "filename": filename}

def reembed_image(filename):
    """
    Recompute the embedding, tags, and description for an image and update its record.
    """
    from models import ImageDB, db, UserConfig
    from flask import current_app
    
    # Log which CLIP model is being used
    config = UserConfig.query.first()
    clip_model_name = config.clip_model if config and config.clip_model else "ViT-B-32"
    current_app.logger.info(f"Re-tagging image {filename} with CLIP model: {clip_model_name}")
    
    image_folder = current_app.config.get("IMAGE_FOLDER", "./images")
    image_path = os.path.join(image_folder, filename)
    if not os.path.exists(image_path):
        return {"status": "error", "message": "Image file not found", "filename": filename}
    
    try:
        # Get image embedding using the selected model
        embedding, model_name = get_image_embedding(image_path)
        if embedding is None:
            return {"status": "error", "message": "Failed to compute embedding", "filename": filename}
        
        # Generate tags and description
        tags, description = generate_tags_and_description(embedding, model_name)
        
        # Update database
        img = ImageDB.query.filter_by(filename=filename).first()
        if img:
            img.tags = ", ".join(tags)
            img.description = description
            db.session.commit()
            return {
                "status": "success",
                "filename": filename,
                "tags": tags,
                "description": description,
                "model_used": model_name
            }
        else:
            return {"status": "error", "message": "Image not found in database", "filename": filename}
    except Exception as e:
        current_app.logger.error(f"Error in reembed_image: {e}")
        return {"status": "error", "message": str(e), "filename": filename}

# Global dictionary to track bulk tagging progress
BULK_PROGRESS = {}

@celery.task(bind=True)
def bulk_tag_images(self):
    """
    Process all images that do not have tags. Returns a task ID that can be used to query progress.
    """
    from models import ImageDB
    from flask import current_app
    
    try:
        images = ImageDB.query.filter((ImageDB.tags == None) | (ImageDB.tags == "")).all()
        total = len(images)
        if total == 0:
            return None
        task_id = str(uuid.uuid4())
        BULK_PROGRESS[task_id] = 0
        for i, img in enumerate(images, start=1):
            process_image_tagging.delay(img.filename)
            BULK_PROGRESS[task_id] = (i / total) * 100
        return task_id
    except Exception as e:
        current_app.logger.error(f"Error in bulk_tag_images: {e}")
        return None

@celery.task(bind=True, time_limit=1800, soft_time_limit=1500)
def reembed_all_images(self):
    """
    Rerun tagging on all images using the currently selected CLIP model.
    This is useful when changing the CLIP model.
    """
    from models import ImageDB, UserConfig
    from flask import current_app
    import gc  # Garbage collection
    
    try:
        # Get the current CLIP model
        config = UserConfig.query.first()
        clip_model_name = config.clip_model if config and config.clip_model else "ViT-B-32"
        
        # Log the operation
        current_app.logger.info(f"Rerunning tagging on all images with CLIP model: {clip_model_name}")
        
        # Get all images
        images = ImageDB.query.all()
        total = len(images)
        
        if total == 0:
            return {"status": "success", "message": "No images found to tag"}
        
        # Create a task ID for tracking progress
        task_id = str(uuid.uuid4())
        BULK_PROGRESS[task_id] = 0
        
        # Process images one at a time to manage memory better
        for i, img in enumerate(images):
            # Process one image and wait for it to complete before starting the next
            process_image_tagging.delay(img.filename)
            
            # Update progress
            BULK_PROGRESS[task_id] = min(100, (i + 1) / total * 100)
            
            # Force garbage collection after each image
            gc.collect()
            
            # Add a small delay between tasks to allow memory to be freed
            time.sleep(0.5)
            
        return {
            "status": "success",
            "message": f"Started retagging {total} images with model {clip_model_name}",
            "task_id": task_id
        }
    except Exception as e:
        current_app.logger.error(f"Error in reembed_all_images: {e}")
        # Try to clean up memory
        gc.collect()
        return {"status": "error", "message": str(e)}

def send_scheduled_image(event_id):
    """
    Send a scheduled image to a device.
    """
    from models import ScheduleEvent, Device, db
    from utils.crop_helpers import load_crop_info_from_db
    from utils.image_helpers import add_send_log_entry
    from flask import current_app
    
    try:
        event = ScheduleEvent.query.get(event_id)
        if not event:
            current_app.logger.error("Event not found: %s", event_id)
            return
        device_obj = Device.query.filter_by(address=event.device).first()
        if not device_obj:
            current_app.logger.error("Device not found for event %s", event_id)
            return

        image_folder = current_app.config.get("IMAGE_FOLDER", "./images")
        data_folder = current_app.config.get("DATA_FOLDER", "./data")
        filepath = os.path.join(image_folder, event.filename)
        if not os.path.exists(filepath):
            current_app.logger.error("Image file not found: %s", filepath)
            return

        addr = device_obj.address
        if not (addr.startswith("http://") or addr.startswith("https://")):
            addr = "http://" + addr

        with Image.open(filepath) as orig_img:
            orig_w, orig_h = orig_img.size
            parts = device_obj.resolution.split("x")
            dev_width = int(parts[0])
            dev_height = int(parts[1])
            
            # Check if device is in portrait orientation
            is_portrait = device_obj.orientation.lower() == 'portrait'
            
            # If portrait, swap width and height for target ratio calculation
            if is_portrait:
                target_ratio = dev_height / dev_width
            else:
                target_ratio = dev_width / dev_height
                
            cdata = load_crop_info_from_db(event.filename)
            if cdata:
                x = cdata.get("x", 0)
                y = cdata.get("y", 0)
                w = cdata.get("width", orig_w)
                h = cdata.get("height", orig_h)
                cropped = orig_img.crop((x, y, x + w, y + h))
            else:
                orig_ratio = orig_w / orig_h
                if orig_ratio > target_ratio:
                    new_width = int(orig_h * target_ratio)
                    left = (orig_w - new_width) // 2
                    crop_box = (left, 0, left + new_width, orig_h)
                else:
                    new_height = int(orig_w / target_ratio)
                    top = (orig_h - new_height) // 2
                    crop_box = (0, top, orig_w, top + new_height)
                cropped = orig_img.crop(crop_box)
            
            # If portrait, rotate the image 90 degrees clockwise and swap dimensions
            if is_portrait:
                cropped = cropped.rotate(-90, expand=True)  # -90 for clockwise rotation
                final_img = cropped.resize((dev_height, dev_width), Image.LANCZOS)  # Note swapped dimensions
            else:
                final_img = cropped.resize((dev_width, dev_height), Image.LANCZOS)
            temp_dir = os.path.join(data_folder, "temp")
            if not os.path.exists(temp_dir):
                os.makedirs(temp_dir)
            temp_filename = os.path.join(temp_dir, f"temp_{event.filename}")
            final_img.save(temp_filename, format="JPEG", quality=95)
        
        cmd = f'curl "{addr}/send_image" -X POST -F "file=@{temp_filename}"'
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        os.remove(temp_filename)
        
        if result.returncode == 0:
            event.sent = True
            db.session.commit()
            device_obj.last_sent = event.filename
            db.session.commit()
            add_send_log_entry(event.filename)
        else:
            current_app.logger.error("Error sending image: %s", result.stderr)
    except Exception as e:
        current_app.logger.error("Error in send_scheduled_image: %s", e)
        return

    # Reschedule recurring events
    if event.recurrence and event.recurrence.lower() != "none":
        try:
            dt = datetime.datetime.fromisoformat(event.datetime_str)
        except Exception as e:
            current_app.logger.error("Error parsing datetime_str: %s", e)
            return
        if event.recurrence.lower() == "daily":
            next_dt = dt + datetime.timedelta(days=1)
        elif event.recurrence.lower() == "weekly":
            next_dt = dt + datetime.timedelta(weeks=1)
        elif event.recurrence.lower() == "monthly":
            next_dt = dt + datetime.timedelta(days=30)
        else:
            next_dt = None
        if next_dt:
            # Update the event in the database with the new date
            event.datetime_str = next_dt.isoformat(sep=' ')
            event.sent = False
            db.session.commit()
            
            # Note: We don't schedule the job here anymore
            # The dedicated scheduler process will pick up this event on its next run
            current_app.logger.info(f"Rescheduled event {event.id} for {next_dt}")

@celery.task(bind=True, ignore_result=True)
def fetch_device_metrics(self):
    """
    Check if devices are online and update their status in the database.
    This task runs periodically to ensure we have up-to-date status information.
    Logging has been minimized to prevent log flooding.
    """
    from models import Device, db
    from flask import current_app
    import httpx
    
    # Disable httpx logging
    import logging
    logging.getLogger("httpx").setLevel(logging.WARNING)
    
    try:
        devices = Device.query.all()
        status_changes = 0
        
        for device in devices:
            try:
                # Ensure the address has a scheme
                address = device.address
                if not (address.startswith("http://") or address.startswith("https://")):
                    address = "http://" + address
                
                # Store previous status to detect changes
                was_online = device.online
                
                # Try to connect to the device's display_info endpoint
                try:
                    # Use a short timeout to avoid blocking
                    response = httpx.get(f"{address}/display_info", timeout=5.0)
                    
                    if response.status_code == 200:
                        # Only update and log if status changed
                        if not was_online:
                            device.online = True
                            db.session.commit()
                            status_changes += 1
                    else:
                        # Only update and log if status changed
                        if was_online:
                            device.online = False
                            db.session.commit()
                            status_changes += 1
                except Exception:
                    # Only update if status changed
                    if was_online:
                        device.online = False
                        db.session.commit()
                        status_changes += 1
            except Exception:
                # Silently continue to next device
                pass
                
        # Only log if there were status changes
        if status_changes > 0:
            current_app.logger.info(f"Updated status for {status_changes} devices")
            
        return {"status": "success", "message": f"Checked status of {len(devices)} devices"}
    except Exception:
        # Catch all exceptions to prevent task failures
        return {"status": "error", "message": "Error checking device status"}

# The scheduler functionality has been moved to scheduler.py
# This function is kept as a stub for backward compatibility
def start_scheduler(app):
    """
    This function is deprecated and no longer starts the scheduler.
    The scheduler is now started in a dedicated process (scheduler.py).
    """
    app.logger.info("Scheduler initialization skipped - now handled by dedicated process")
```


## webpack.config.js

```js
const path = require('path');

module.exports = {
  entry: './assets/js/main.js', 
  output: {
    filename: 'bundle.js',
    path: path.resolve(__dirname, 'static', 'js')
  },
  module: {
    rules: [
      {
        test: /\.css$/i,
        use: ['style-loader', 'css-loader'],
      },
    ],
  },
};
```
```


## exportconfig.json

```json
{
    "output": "export.md",
    "description": "",
    "ignoreFile": ".export-ignore",
    "includeFile": ".export-whitelist",
    "maxFileSize": 1048576,
    "removeComments": true,
    "includeProjectStructure": true
  }
```


## LICENSE

```
GNU GENERAL PUBLIC LICENSE
                       Version 3, 29 June 2007

 Copyright (C) 2007 Free Software Foundation, Inc. <https://fsf.org/>
 Everyone is permitted to copy and distribute verbatim copies
 of this license document, but changing it is not allowed.

                            Preamble

  The GNU General Public License is a free, copyleft license for
software and other kinds of works.

  The licenses for most software and other practical works are designed
to take away your freedom to share and change the works.  By contrast,
the GNU General Public License is intended to guarantee your freedom to
share and change all versions of a program--to make sure it remains free
software for all its users.  We, the Free Software Foundation, use the
GNU General Public License for most of our software; it applies also to
any other work released this way by its authors.  You can apply it to
your programs, too.

  When we speak of free software, we are referring to freedom, not
price.  Our General Public Licenses are designed to make sure that you
have the freedom to distribute copies of free software (and charge for
them if you wish), that you receive source code or can get it if you
want it, that you can change the software or use pieces of it in new
free programs, and that you know you can do these things.

  To protect your rights, we need to prevent others from denying you
these rights or asking you to surrender the rights.  Therefore, you have
certain responsibilities if you distribute copies of the software, or if
you modify it: responsibilities to respect the freedom of others.

  For example, if you distribute copies of such a program, whether
gratis or for a fee, you must pass on to the recipients the same
freedoms that you received.  You must make sure that they, too, receive
or can get the source code.  And you must show them these terms so they
know their rights.

  Developers that use the GNU GPL protect your rights with two steps:
(1) assert copyright on the software, and (2) offer you this License
giving you legal permission to copy, distribute and/or modify it.

  For the developers' and authors' protection, the GPL clearly explains
that there is no warranty for this free software.  For both users' and
authors' sake, the GPL requires that modified versions be marked as
changed, so that their problems will not be attributed erroneously to
authors of previous versions.

  Some devices are designed to deny users access to install or run
modified versions of the software inside them, although the manufacturer
can do so.  This is fundamentally incompatible with the aim of
protecting users' freedom to change the software.  The systematic
pattern of such abuse occurs in the area of products for individuals to
use, which is precisely where it is most unacceptable.  Therefore, we
have designed this version of the GPL to prohibit the practice for those
products.  If such problems arise substantially in other domains, we
stand ready to extend this provision to those domains in future versions
of the GPL, as needed to protect the freedom of users.

  Finally, every program is threatened constantly by software patents.
States should not allow patents to restrict development and use of
software on general-purpose computers, but in those that do, we wish to
avoid the special danger that patents applied to a free program could
make it effectively proprietary.  To prevent this, the GPL assures that
patents cannot be used to render the program non-free.

  The precise terms and conditions for copying, distribution and
modification follow.

                       TERMS AND CONDITIONS

  0. Definitions.

  "This License" refers to version 3 of the GNU General Public License.

  "Copyright" also means copyright-like laws that apply to other kinds of
works, such as semiconductor masks.

  "The Program" refers to any copyrightable work licensed under this
License.  Each licensee is addressed as "you".  "Licensees" and
"recipients" may be individuals or organizations.

  To "modify" a work means to copy from or adapt all or part of the work
in a fashion requiring copyright permission, other than the making of an
exact copy.  The resulting work is called a "modified version" of the
earlier work or a work "based on" the earlier work.

  A "covered work" means either the unmodified Program or a work based
on the Program.

  To "propagate" a work means to do anything with it that, without
permission, would make you directly or secondarily liable for
infringement under applicable copyright law, except executing it on a
computer or modifying a private copy.  Propagation includes copying,
distribution (with or without modification), making available to the
public, and in some countries other activities as well.

  To "convey" a work means any kind of propagation that enables other
parties to make or receive copies.  Mere interaction with a user through
a computer network, with no transfer of a copy, is not conveying.

  An interactive user interface displays "Appropriate Legal Notices"
to the extent that it includes a convenient and prominently visible
feature that (1) displays an appropriate copyright notice, and (2)
tells the user that there is no warranty for the work (except to the
extent that warranties are provided), that licensees may convey the
work under this License, and how to view a copy of this License.  If
the interface presents a list of user commands or options, such as a
menu, a prominent item in the list meets this criterion.

  1. Source Code.

  The "source code" for a work means the preferred form of the work
for making modifications to it.  "Object code" means any non-source
form of a work.

  A "Standard Interface" means an interface that either is an official
standard defined by a recognized standards body, or, in the case of
interfaces specified for a particular programming language, one that
is widely used among developers working in that language.

  The "System Libraries" of an executable work include anything, other
than the work as a whole, that (a) is included in the normal form of
packaging a Major Component, but which is not part of that Major
Component, and (b) serves only to enable use of the work with that
Major Component, or to implement a Standard Interface for which an
implementation is available to the public in source code form.  A
"Major Component", in this context, means a major essential component
(kernel, window system, and so on) of the specific operating system
(if any) on which the executable work runs, or a compiler used to
produce the work, or an object code interpreter used to run it.

  The "Corresponding Source" for a work in object code form means all
the source code needed to generate, install, and (for an executable
work) run the object code and to modify the work, including scripts to
control those activities.  However, it does not include the work's
System Libraries, or general-purpose tools or generally available free
programs which are used unmodified in performing those activities but
which are not part of the work.  For example, Corresponding Source
includes interface definition files associated with source files for
the work, and the source code for shared libraries and dynamically
linked subprograms that the work is specifically designed to require,
such as by intimate data communication or control flow between those
subprograms and other parts of the work.

  The Corresponding Source need not include anything that users
can regenerate automatically from other parts of the Corresponding
Source.

  The Corresponding Source for a work in source code form is that
same work.

  2. Basic Permissions.

  All rights granted under this License are granted for the term of
copyright on the Program, and are irrevocable provided the stated
conditions are met.  This License explicitly affirms your unlimited
permission to run the unmodified Program.  The output from running a
covered work is covered by this License only if the output, given its
content, constitutes a covered work.  This License acknowledges your
rights of fair use or other equivalent, as provided by copyright law.

  You may make, run and propagate covered works that you do not
convey, without conditions so long as your license otherwise remains
in force.  You may convey covered works to others for the sole purpose
of having them make modifications exclusively for you, or provide you
with facilities for running those works, provided that you comply with
the terms of this License in conveying all material for which you do
not control copyright.  Those thus making or running the covered works
for you must do so exclusively on your behalf, under your direction
and control, on terms that prohibit them from making any copies of
your copyrighted material outside their relationship with you.

  Conveying under any other circumstances is permitted solely under
the conditions stated below.  Sublicensing is not allowed; section 10
makes it unnecessary.

  3. Protecting Users' Legal Rights From Anti-Circumvention Law.

  No covered work shall be deemed part of an effective technological
measure under any applicable law fulfilling obligations under article
11 of the WIPO copyright treaty adopted on 20 December 1996, or
similar laws prohibiting or restricting circumvention of such
measures.

  When you convey a covered work, you waive any legal power to forbid
circumvention of technological measures to the extent such circumvention
is effected by exercising rights under this License with respect to
the covered work, and you disclaim any intention to limit operation or
modification of the work as a means of enforcing, against the work's
users, your or third parties' legal rights to forbid circumvention of
technological measures.

  4. Conveying Verbatim Copies.

  You may convey verbatim copies of the Program's source code as you
receive it, in any medium, provided that you conspicuously and
appropriately publish on each copy an appropriate copyright notice;
keep intact all notices stating that this License and any
non-permissive terms added in accord with section 7 apply to the code;
keep intact all notices of the absence of any warranty; and give all
recipients a copy of this License along with the Program.

  You may charge any price or no price for each copy that you convey,
and you may offer support or warranty protection for a fee.

  5. Conveying Modified Source Versions.

  You may convey a work based on the Program, or the modifications to
produce it from the Program, in the form of source code under the
terms of section 4, provided that you also meet all of these conditions:

    a) The work must carry prominent notices stating that you modified
    it, and giving a relevant date.

    b) The work must carry prominent notices stating that it is
    released under this License and any conditions added under section
    7.  This requirement modifies the requirement in section 4 to
    "keep intact all notices".

    c) You must license the entire work, as a whole, under this
    License to anyone who comes into possession of a copy.  This
    License will therefore apply, along with any applicable section 7
    additional terms, to the whole of the work, and all its parts,
    regardless of how they are packaged.  This License gives no
    permission to license the work in any other way, but it does not
    invalidate such permission if you have separately received it.

    d) If the work has interactive user interfaces, each must display
    Appropriate Legal Notices; however, if the Program has interactive
    interfaces that do not display Appropriate Legal Notices, your
    work need not make them do so.

  A compilation of a covered work with other separate and independent
works, which are not by their nature extensions of the covered work,
and which are not combined with it such as to form a larger program,
in or on a volume of a storage or distribution medium, is called an
"aggregate" if the compilation and its resulting copyright are not
used to limit the access or legal rights of the compilation's users
beyond what the individual works permit.  Inclusion of a covered work
in an aggregate does not cause this License to apply to the other
parts of the aggregate.

  6. Conveying Non-Source Forms.

  You may convey a covered work in object code form under the terms
of sections 4 and 5, provided that you also convey the
machine-readable Corresponding Source under the terms of this License,
in one of these ways:

    a) Convey the object code in, or embodied in, a physical product
    (including a physical distribution medium), accompanied by the
    Corresponding Source fixed on a durable physical medium
    customarily used for software interchange.

    b) Convey the object code in, or embodied in, a physical product
    (including a physical distribution medium), accompanied by a
    written offer, valid for at least three years and valid for as
    long as you offer spare parts or customer support for that product
    model, to give anyone who possesses the object code either (1) a
    copy of the Corresponding Source for all the software in the
    product that is covered by this License, on a durable physical
    medium customarily used for software interchange, for a price no
    more than your reasonable cost of physically performing this
    conveying of source, or (2) access to copy the
    Corresponding Source from a network server at no charge.

    c) Convey individual copies of the object code with a copy of the
    written offer to provide the Corresponding Source.  This
    alternative is allowed only occasionally and noncommercially, and
    only if you received the object code with such an offer, in accord
    with subsection 6b.

    d) Convey the object code by offering access from a designated
    place (gratis or for a charge), and offer equivalent access to the
    Corresponding Source in the same way through the same place at no
    further charge.  You need not require recipients to copy the
    Corresponding Source along with the object code.  If the place to
    copy the object code is a network server, the Corresponding Source
    may be on a different server (operated by you or a third party)
    that supports equivalent copying facilities, provided you maintain
    clear directions next to the object code saying where to find the
    Corresponding Source.  Regardless of what server hosts the
    Corresponding Source, you remain obligated to ensure that it is
    available for as long as needed to satisfy these requirements.

    e) Convey the object code using peer-to-peer transmission, provided
    you inform other peers where the object code and Corresponding
    Source of the work are being offered to the general public at no
    charge under subsection 6d.

  A separable portion of the object code, whose source code is excluded
from the Corresponding Source as a System Library, need not be
included in conveying the object code work.

  A "User Product" is either (1) a "consumer product", which means any
tangible personal property which is normally used for personal, family,
or household purposes, or (2) anything designed or sold for incorporation
into a dwelling.  In determining whether a product is a consumer product,
doubtful cases shall be resolved in favor of coverage.  For a particular
product received by a particular user, "normally used" refers to a
typical or common use of that class of product, regardless of the status
of the particular user or of the way in which the particular user
actually uses, or expects or is expected to use, the product.  A product
is a consumer product regardless of whether the product has substantial
commercial, industrial or non-consumer uses, unless such uses represent
the only significant mode of use of the product.

  "Installation Information" for a User Product means any methods,
procedures, authorization keys, or other information required to install
and execute modified versions of a covered work in that User Product from
a modified version of its Corresponding Source.  The information must
suffice to ensure that the continued functioning of the modified object
code is in no case prevented or interfered with solely because
modification has been made.

  If you convey an object code work under this section in, or with, or
specifically for use in, a User Product, and the conveying occurs as
part of a transaction in which the right of possession and use of the
User Product is transferred to the recipient in perpetuity or for a
fixed term (regardless of how the transaction is characterized), the
Corresponding Source conveyed under this section must be accompanied
by the Installation Information.  But this requirement does not apply
if neither you nor any third party retains the ability to install
modified object code on the User Product (for example, the work has
been installed in ROM).

  The requirement to provide Installation Information does not include a
requirement to continue to provide support service, warranty, or updates
for a work that has been modified or installed by the recipient, or for
the User Product in which it has been modified or installed.  Access to a
network may be denied when the modification itself materially and
adversely affects the operation of the network or violates the rules and
protocols for communication across the network.

  Corresponding Source conveyed, and Installation Information provided,
in accord with this section must be in a format that is publicly
documented (and with an implementation available to the public in
source code form), and must require no special password or key for
unpacking, reading or copying.

  7. Additional Terms.

  "Additional permissions" are terms that supplement the terms of this
License by making exceptions from one or more of its conditions.
Additional permissions that are applicable to the entire Program shall
be treated as though they were included in this License, to the extent
that they are valid under applicable law.  If additional permissions
apply only to part of the Program, that part may be used separately
under those permissions, but the entire Program remains governed by
this License without regard to the additional permissions.

  When you convey a copy of a covered work, you may at your option
remove any additional permissions from that copy, or from any part of
it.  (Additional permissions may be written to require their own
removal in certain cases when you modify the work.)  You may place
additional permissions on material, added by you to a covered work,
for which you have or can give appropriate copyright permission.

  Notwithstanding any other provision of this License, for material you
add to a covered work, you may (if authorized by the copyright holders of
that material) supplement the terms of this License with terms:

    a) Disclaiming warranty or limiting liability differently from the
    terms of sections 15 and 16 of this License; or

    b) Requiring preservation of specified reasonable legal notices or
    author attributions in that material or in the Appropriate Legal
    Notices displayed by works containing it; or

    c) Prohibiting misrepresentation of the origin of that material, or
    requiring that modified versions of such material be marked in
    reasonable ways as different from the original version; or

    d) Limiting the use for publicity purposes of names of licensors or
    authors of the material; or

    e) Declining to grant rights under trademark law for use of some
    trade names, trademarks, or service marks; or

    f) Requiring indemnification of licensors and authors of that
    material by anyone who conveys the material (or modified versions of
    it) with contractual assumptions of liability to the recipient, for
    any liability that these contractual assumptions directly impose on
    those licensors and authors.

  All other non-permissive additional terms are considered "further
restrictions" within the meaning of section 10.  If the Program as you
received it, or any part of it, contains a notice stating that it is
governed by this License along with a term that is a further
restriction, you may remove that term.  If a license document contains
a further restriction but permits relicensing or conveying under this
License, you may add to a covered work material governed by the terms
of that license document, provided that the further restriction does
not survive such relicensing or conveying.

  If you add terms to a covered work in accord with this section, you
must place, in the relevant source files, a statement of the
additional terms that apply to those files, or a notice indicating
where to find the applicable terms.

  Additional terms, permissive or non-permissive, may be stated in the
form of a separately written license, or stated as exceptions;
the above requirements apply either way.

  8. Termination.

  You may not propagate or modify a covered work except as expressly
provided under this License.  Any attempt otherwise to propagate or
modify it is void, and will automatically terminate your rights under
this License (including any patent licenses granted under the third
paragraph of section 11).

  However, if you cease all violation of this License, then your
license from a particular copyright holder is reinstated (a)
provisionally, unless and until the copyright holder explicitly and
finally terminates your license, and (b) permanently, if the copyright
holder fails to notify you of the violation by some reasonable means
prior to 60 days after the cessation.

  Moreover, your license from a particular copyright holder is
reinstated permanently if the copyright holder notifies you of the
violation by some reasonable means, this is the first time you have
received notice of violation of this License (for any work) from that
copyright holder, and you cure the violation prior to 30 days after
your receipt of the notice.

  Termination of your rights under this section does not terminate the
licenses of parties who have received copies or rights from you under
this License.  If your rights have been terminated and not permanently
reinstated, you do not qualify to receive new licenses for the same
material under section 10.

  9. Acceptance Not Required for Having Copies.

  You are not required to accept this License in order to receive or
run a copy of the Program.  Ancillary propagation of a covered work
occurring solely as a consequence of using peer-to-peer transmission
to receive a copy likewise does not require acceptance.  However,
nothing other than this License grants you permission to propagate or
modify any covered work.  These actions infringe copyright if you do
not accept this License.  Therefore, by modifying or propagating a
covered work, you indicate your acceptance of this License to do so.

  10. Automatic Licensing of Downstream Recipients.

  Each time you convey a covered work, the recipient automatically
receives a license from the original licensors, to run, modify and
propagate that work, subject to this License.  You are not responsible
for enforcing compliance by third parties with this License.

  An "entity transaction" is a transaction transferring control of an
organization, or substantially all assets of one, or subdividing an
organization, or merging organizations.  If propagation of a covered
work results from an entity transaction, each party to that
transaction who receives a copy of the work also receives whatever
licenses to the work the party's predecessor in interest had or could
give under the previous paragraph, plus a right to possession of the
Corresponding Source of the work from the predecessor in interest, if
the predecessor has it or can get it with reasonable efforts.

  You may not impose any further restrictions on the exercise of the
rights granted or affirmed under this License.  For example, you may
not impose a license fee, royalty, or other charge for exercise of
rights granted under this License, and you may not initiate litigation
(including a cross-claim or counterclaim in a lawsuit) alleging that
any patent claim is infringed by making, using, selling, offering for
sale, or importing the Program or any portion of it.

  11. Patents.

  A "contributor" is a copyright holder who authorizes use under this
License of the Program or a work on which the Program is based.  The
work thus licensed is called the contributor's "contributor version".

  A contributor's "essential patent claims" are all patent claims
owned or controlled by the contributor, whether already acquired or
hereafter acquired, that would be infringed by some manner, permitted
by this License, of making, using, or selling its contributor version,
but do not include claims that would be infringed only as a
consequence of further modification of the contributor version.  For
purposes of this definition, "control" includes the right to grant
patent sublicenses in a manner consistent with the requirements of
this License.

  Each contributor grants you a non-exclusive, worldwide, royalty-free
patent license under the contributor's essential patent claims, to
make, use, sell, offer for sale, import and otherwise run, modify and
propagate the contents of its contributor version.

  In the following three paragraphs, a "patent license" is any express
agreement or commitment, however denominated, not to enforce a patent
(such as an express permission to practice a patent or covenant not to
sue for patent infringement).  To "grant" such a patent license to a
party means to make such an agreement or commitment not to enforce a
patent against the party.

  If you convey a covered work, knowingly relying on a patent license,
and the Corresponding Source of the work is not available for anyone
to copy, free of charge and under the terms of this License, through a
publicly available network server or other readily accessible means,
then you must either (1) cause the Corresponding Source to be so
available, or (2) arrange to deprive yourself of the benefit of the
patent license for this particular work, or (3) arrange, in a manner
consistent with the requirements of this License, to extend the patent
license to downstream recipients.  "Knowingly relying" means you have
actual knowledge that, but for the patent license, your conveying the
covered work in a country, or your recipient's use of the covered work
in a country, would infringe one or more identifiable patents in that
country that you have reason to believe are valid.

  If, pursuant to or in connection with a single transaction or
arrangement, you convey, or propagate by procuring conveyance of, a
covered work, and grant a patent license to some of the parties
receiving the covered work authorizing them to use, propagate, modify
or convey a specific copy of the covered work, then the patent license
you grant is automatically extended to all recipients of the covered
work and works based on it.

  A patent license is "discriminatory" if it does not include within
the scope of its coverage, prohibits the exercise of, or is
conditioned on the non-exercise of one or more of the rights that are
specifically granted under this License.  You may not convey a covered
work if you are a party to an arrangement with a third party that is
in the business of distributing software, under which you make payment
to the third party based on the extent of your activity of conveying
the work, and under which the third party grants, to any of the
parties who would receive the covered work from you, a discriminatory
patent license (a) in connection with copies of the covered work
conveyed by you (or copies made from those copies), or (b) primarily
for and in connection with specific products or compilations that
contain the covered work, unless you entered into that arrangement,
or that patent license was granted, prior to 28 March 2007.

  Nothing in this License shall be construed as excluding or limiting
any implied license or other defenses to infringement that may
otherwise be available to you under applicable patent law.

  12. No Surrender of Others' Freedom.

  If conditions are imposed on you (whether by court order, agreement or
otherwise) that contradict the conditions of this License, they do not
excuse you from the conditions of this License.  If you cannot convey a
covered work so as to satisfy simultaneously your obligations under this
License and any other pertinent obligations, then as a consequence you may
not convey it at all.  For example, if you agree to terms that obligate you
to collect a royalty for further conveying from those to whom you convey
the Program, the only way you could satisfy both those terms and this
License would be to refrain entirely from conveying the Program.

  13. Use with the GNU Affero General Public License.

  Notwithstanding any other provision of this License, you have
permission to link or combine any covered work with a work licensed
under version 3 of the GNU Affero General Public License into a single
combined work, and to convey the resulting work.  The terms of this
License will continue to apply to the part which is the covered work,
but the special requirements of the GNU Affero General Public License,
section 13, concerning interaction through a network will apply to the
combination as such.

  14. Revised Versions of this License.

  The Free Software Foundation may publish revised and/or new versions of
the GNU General Public License from time to time.  Such new versions will
be similar in spirit to the present version, but may differ in detail to
address new problems or concerns.

  Each version is given a distinguishing version number.  If the
Program specifies that a certain numbered version of the GNU General
Public License "or any later version" applies to it, you have the
option of following the terms and conditions either of that numbered
version or of any later version published by the Free Software
Foundation.  If the Program does not specify a version number of the
GNU General Public License, you may choose any version ever published
by the Free Software Foundation.

  If the Program specifies that a proxy can decide which future
versions of the GNU General Public License can be used, that proxy's
public statement of acceptance of a version permanently authorizes you
to choose that version for the Program.

  Later license versions may give you additional or different
permissions.  However, no additional obligations are imposed on any
author or copyright holder as a result of your choosing to follow a
later version.

  15. Disclaimer of Warranty.

  THERE IS NO WARRANTY FOR THE PROGRAM, TO THE EXTENT PERMITTED BY
APPLICABLE LAW.  EXCEPT WHEN OTHERWISE STATED IN WRITING THE COPYRIGHT
HOLDERS AND/OR OTHER PARTIES PROVIDE THE PROGRAM "AS IS" WITHOUT WARRANTY
OF ANY KIND, EITHER EXPRESSED OR IMPLIED, INCLUDING, BUT NOT LIMITED TO,
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
PURPOSE.  THE ENTIRE RISK AS TO THE QUALITY AND PERFORMANCE OF THE PROGRAM
IS WITH YOU.  SHOULD THE PROGRAM PROVE DEFECTIVE, YOU ASSUME THE COST OF
ALL NECESSARY SERVICING, REPAIR OR CORRECTION.

  16. Limitation of Liability.

  IN NO EVENT UNLESS REQUIRED BY APPLICABLE LAW OR AGREED TO IN WRITING
WILL ANY COPYRIGHT HOLDER, OR ANY OTHER PARTY WHO MODIFIES AND/OR CONVEYS
THE PROGRAM AS PERMITTED ABOVE, BE LIABLE TO YOU FOR DAMAGES, INCLUDING ANY
GENERAL, SPECIAL, INCIDENTAL OR CONSEQUENTIAL DAMAGES ARISING OUT OF THE
USE OR INABILITY TO USE THE PROGRAM (INCLUDING BUT NOT LIMITED TO LOSS OF
DATA OR DATA BEING RENDERED INACCURATE OR LOSSES SUSTAINED BY YOU OR THIRD
PARTIES OR A FAILURE OF THE PROGRAM TO OPERATE WITH ANY OTHER PROGRAMS),
EVEN IF SUCH HOLDER OR OTHER PARTY HAS BEEN ADVISED OF THE POSSIBILITY OF
SUCH DAMAGES.

  17. Interpretation of Sections 15 and 16.

  If the disclaimer of warranty and limitation of liability provided
above cannot be given local legal effect according to their terms,
reviewing courts shall apply local law that most closely approximates
an absolute waiver of all civil liability in connection with the
Program, unless a warranty or assumption of liability accompanies a
copy of the Program in return for a fee.

                     END OF TERMS AND CONDITIONS

            How to Apply These Terms to Your New Programs

  If you develop a new program, and you want it to be of the greatest
possible use to the public, the best way to achieve this is to make it
free software which everyone can redistribute and change under these terms.

  To do so, attach the following notices to the program.  It is safest
to attach them to the start of each source file to most effectively
state the exclusion of warranty; and each file should have at least
the "copyright" line and a pointer to where the full notice is found.

    <one line to give the program's name and a brief idea of what it does.>
    Copyright (C) <year>  <name of author>

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.

Also add information on how to contact you by electronic and paper mail.

  If the program does terminal interaction, make it output a short
notice like this when it starts in an interactive mode:

    <program>  Copyright (C) <year>  <name of author>
    This program comes with ABSOLUTELY NO WARRANTY; for details type `show w'.
    This is free software, and you are welcome to redistribute it
    under certain conditions; type `show c' for details.

The hypothetical commands `show w' and `show c' should show the appropriate
parts of the General Public License.  Of course, your program's commands
might be different; for a GUI interface, you would use an "about box".

  You should also get your employer (if you work as a programmer) or school,
if any, to sign a "copyright disclaimer" for the program, if necessary.
For more information on this, and how to apply and follow the GNU GPL, see
<https://www.gnu.org/licenses/>.

  The GNU General Public License does not permit incorporating your program
into proprietary programs.  If your program is a subroutine library, you
may consider it more useful to permit linking proprietary applications with
the library.  If this is what you want to do, use the GNU Lesser General
Public License instead of this License.  But first, please read
<https://www.gnu.org/licenses/why-not-lgpl.html>.
```


## models.py

```py
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()

class Device(db.Model):
    __tablename__ = 'devices'
    id = db.Column(db.Integer, primary_key=True)
    color = db.Column(db.String(16), nullable=False)
    friendly_name = db.Column(db.String(128), nullable=False)
    orientation = db.Column(db.String(32), nullable=False)
    address = db.Column(db.String(256), nullable=False)
    display_name = db.Column(db.String(128))
    resolution = db.Column(db.String(32))
    online = db.Column(db.Boolean, default=False)
    last_sent = db.Column(db.String(256))

    def __repr__(self):
        return f"<Device {self.friendly_name} ({self.address})>"

class ImageDB(db.Model):
    __tablename__ = 'images'
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(256), unique=True, nullable=False)
    tags = db.Column(db.String(512), nullable=True)         # comma-separated tags
    description = db.Column(db.Text, nullable=True)           # description text
    favorite = db.Column(db.Boolean, default=False)           # favorite flag

    def __repr__(self):
        return f"<ImageDB {self.filename}>"

class CropInfo(db.Model):
    __tablename__ = 'crop_info'
    filename = db.Column(db.String(256), primary_key=True)
    x = db.Column(db.Float, default=0)
    y = db.Column(db.Float, default=0)
    width = db.Column(db.Float)
    height = db.Column(db.Float)
    resolution = db.Column(db.String(32))  # Store the display resolution (e.g., "1024x768")

    def __repr__(self):
        return f"<CropInfo {self.filename}>"

class SendLog(db.Model):
    __tablename__ = 'send_log'
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(256), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<SendLog {self.filename} {self.timestamp}>"

class ScheduleEvent(db.Model):
    __tablename__ = 'schedule_events'
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(256), nullable=False)
    device = db.Column(db.String(256), nullable=False)
    datetime_str = db.Column(db.String(32))
    sent = db.Column(db.Boolean, default=False)
    recurrence = db.Column(db.String(20), default="none")  # Recurrence type

    def __repr__(self):
        return f"<ScheduleEvent {self.filename} on {self.device}>"

class UserConfig(db.Model):
    __tablename__ = 'user_config'
    id = db.Column(db.Integer, primary_key=True)
    location = db.Column(db.String(256))
    openai_api_key = db.Column(db.String(512))
    ollama_address = db.Column(db.String(256))
    ollama_api_key = db.Column(db.String(512))
    ollama_model = db.Column(db.String(64))  # Column for chosen Ollama model
    clip_model = db.Column(db.String(64), default="ViT-B-32")  # Column for chosen CLIP model (using consistent format with tasks.py)

    def __repr__(self):
        return f"<UserConfig {self.id} - Location: {self.location}>"

# DeviceMetrics model removed - we only track online status now
```


## package.json

```json
{
    "name": "inkydocker",
    "version": "1.0.0",
    "description": "InkyDocker static assets",
    "main": "index.js",
    "scripts": {
      "build": "webpack --mode production",
      "dev": "webpack --mode development --watch"
    },
    "dependencies": {
      "@fullcalendar/core": "^6.1.8",
      "@fullcalendar/timegrid": "^6.1.8"
    },
    "devDependencies": {
      "webpack": "^5.0.0",
      "webpack-cli": "^4.0.0",
      "css-loader": "^6.0.0",
      "style-loader": "^3.0.0"
    },
    "author": "",
    "license": "MIT"
  }
```


## README.md

```md
# InkyDocker

InkyDocker is a Flask-based web application designed to manage and display images on e-ink displays. It allows users to upload images, schedule them for display, manage display settings, and monitor device metrics.

## Features

*   **Image Gallery**: Upload, crop, and manage images for display.
*   **Scheduled Display**: Schedule images to be displayed on e-ink displays at specific times.
*   **Device Management**: Configure and manage connected e-ink displays, including setting orientation and fetching display information.
*   **AI Settings**: Configure AI settings, such as OpenAI API key and Ollama model settings.
*   **Device Monitoring**: Monitor real-time device metrics such as CPU usage, memory usage, and disk usage.

## API Endpoints

The application exposes the following API endpoints for interacting with e-ink displays:

*   `POST /send_image`: Upload an image to display on the e-ink screen. Example:

    ```bash
    curl -F "file=@path/to/your/image.jpg" http://<IP_ADDRESS>/send_image
    ```

*   `POST /set_orientation`: Set the display orientation (choose either "horizontal" or "vertical"). Example:

    ```bash
    curl -X POST -d "orientation=vertical" http://<IP_ADDRESS>/set_orientation
    ```

*   `GET /display_info`: Retrieve display information in JSON format. Example:

    ```bash
    curl http://<IP_ADDRESS>/display_info
    ```

*   `POST /system_update`: Trigger a system update and upgrade (which will reboot the device). Example:

    ```bash
    curl -X POST http://<IP_ADDRESS>/system_update
    ```

*   `POST /backup`: Create a compressed backup of the SD card and download it. Example:

    ```bash
    curl -X POST http://<IP_ADDRESS>/backup --output backup.img.gz
    ```

*   `POST /update`: Perform a Git pull to update the application and reboot the device. Example:

    ```bash
    curl -X POST http://<IP_ADDRESS>/update
    ```

*   `GET /stream`: Connect to the SSE stream to receive real-time system metrics. Example:

    ```bash
    curl http://<IP_ADDRESS>/stream
    ```

## Requirements

*   Python 3.6+
*   Flask
*   Flask-Migrate
*   Pillow
*   Other dependencies listed in `requirements.txt`

## Installation

1.  Clone the repository:

    ```bash
    git clone <repository_url>
    cd inkydocker
    ```

2.  Create a virtual environment:

    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

4.  Configure the application:

    *   Create a `config.py` file based on the `config.example.py` file.
    *   Set the necessary environment variables, such as the database URI and API keys.

5.  Run database migrations:

    ```bash
    flask db upgrade
    ```

6.  Run the application:

    ```bash
    python app.py
    ```

## Usage

1.  Access the web application in your browser.
2.  Configure your e-ink displays in the Settings page.
3.  Upload images to the Gallery.
4.  Schedule images for display on the Schedule page.
5.  Monitor device metrics on the Settings page.

## Contributing

Contributions are welcome! Please submit a pull request with your changes.

## License

[Specify the license for your project]
```


## requirements.txt

```txt
Flask
Pillow==11.1.0
pillow-heif==0.21.0
Flask-SQLAlchemy==3.1.1
cryptography
requests
Flask-Migrate
ollama
httpx
APScheduler>=3.9.0
openai
celery
redis
open_clip_torch
scikit-learn
gunicorn
psutil>=5.9.0  # For memory monitoring
```


## scheduler.py

```py
#!/usr/bin/env python3
"""
Dedicated scheduler process for InkyDocker.

This file runs the APScheduler in a dedicated process to avoid starting
multiple schedulers in Celery worker processes. This solves the issue where
each Celery worker was starting its own scheduler, leading to duplicate jobs
and potential race conditions.

The scheduler is responsible for:
1. Running scheduled image sends based on the schedule in the database
2. Periodically checking device online status
"""

import os
import sys
import logging
from apscheduler.schedulers.background import BackgroundScheduler
from flask import Flask
from config import Config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Initialize scheduler at the module level so it can be imported by other modules
# Configure with optimized settings for better performance
scheduler = BackgroundScheduler(
    job_defaults={
        'coalesce': True,  # Combine multiple pending executions of the same job into one
        'max_instances': 1,  # Only allow one instance of each job to run at a time
        'misfire_grace_time': 3600  # Allow misfires up to 1 hour
    },
    executor='threadpool',  # Use threadpool executor for better performance
    timezone='UTC'  # Use UTC for consistent timezone handling
)

def create_app():
    """Create a minimal Flask app for the scheduler context"""
    app = Flask(__name__)
    app.config.from_object(Config)
    
    # Initialize database
    from models import db
    db.init_app(app)
    
    return app

def start_scheduler(app):
    """
    Start the APScheduler with the Flask app context.
    This function should only be called once in this dedicated process.
    """
    with app.app_context():
        # Import the necessary functions
        from tasks import fetch_device_metrics, send_scheduled_image
        from models import ScheduleEvent
        
        # Schedule the fetch_device_metrics task to run every 60 seconds (reduced from 5 seconds to prevent log flooding)
        scheduler.add_job(
            fetch_device_metrics,
            'interval',
            seconds=60,
            id='fetch_device_metrics'
        )
        
        # Load all scheduled events from the database and schedule them
        try:
            events = ScheduleEvent.query.filter_by(sent=False).all()
            for event in events:
                try:
                    import datetime
                    dt = datetime.datetime.fromisoformat(event.datetime_str)
                    if dt > datetime.datetime.now():
                        scheduler.add_job(
                            send_scheduled_image,
                            'date',
                            run_date=dt,
                            args=[event.id],
                            id=f'event_{event.id}'
                        )
                        logger.info(f"Scheduled event {event.id} for {dt}")
                except Exception as e:
                    logger.error(f"Error scheduling event {event.id}: {e}")
        except Exception as e:
            logger.error(f"Error loading scheduled events: {e}")
            logger.info("Continuing without loading scheduled events")
        
        # Start the scheduler
        scheduler.start()
        logger.info("Scheduler started successfully")
        
        # Keep the process running
        try:
            import time
            while True:
                time.sleep(60)
        except (KeyboardInterrupt, SystemExit):
            logger.info("Scheduler shutting down...")
            scheduler.shutdown()
            logger.info("Scheduler shut down successfully")

if __name__ == "__main__":
    app = create_app()
    start_scheduler(app)
```


## supervisord.conf

```conf
[supervisord]
nodaemon=true
logfile=/dev/null

[program:flask]
command=gunicorn -w 1 -t 120 --bind 0.0.0.0:5001 app:app
directory=/app
autostart=true
autorestart=true
stdout_logfile=/dev/stdout
stdout_logfile_maxbytes=0
stderr_logfile=/dev/stderr
stderr_logfile_maxbytes=0

[program:celery]
command=celery -A tasks.celery worker --loglevel=warning --max-memory-per-child=500000 -c 1
directory=/app
autostart=true
autorestart=true
stdout_logfile=/dev/stdout
stdout_logfile_maxbytes=0
stderr_logfile=/dev/stderr
stderr_logfile_maxbytes=0

# Redis is now started in entrypoint.sh before supervisord

[program:scheduler]
command=python scheduler.py
directory=/app
autostart=true
autorestart=true
stdout_logfile=/dev/stdout
stdout_logfile_maxbytes=0
stderr_logfile=/dev/stderr
stderr_logfile_maxbytes=0
```


## tasks.py

```py
import os
import datetime
import subprocess
import time
import uuid

from flask import current_app
from PIL import Image

import torch
import open_clip
from sklearn.metrics.pairwise import cosine_similarity

# Import and initialize Celery and APScheduler
from celery import Celery
from apscheduler.schedulers.background import BackgroundScheduler

# Initialize Celery with memory limits
celery = Celery('tasks', broker='redis://localhost:6379/0', backend='redis://localhost:6379/0')

# Configure Celery to prevent memory leaks and reduce log flooding
celery.conf.update(
    worker_max_memory_per_child=500000,   # 500MB memory limit per worker
    worker_max_tasks_per_child=1,         # Restart worker after each task
    task_time_limit=600,                  # 10 minute time limit per task
    task_soft_time_limit=300,             # 5 minute soft time limit (sends exception)
    worker_concurrency=1,                 # Limit to 1 concurrent worker to prevent duplicate tasks
    broker_connection_retry_on_startup=True,  # Fix deprecation warning
    worker_hijack_root_logger=False,      # Don't hijack the root logger
    worker_log_format='%(asctime)s [%(levelname)s] %(message)s',  # Simplified log format
    task_track_started=False,             # Don't log when tasks are started
    task_send_sent_event=False,           # Don't send task-sent events
    worker_send_task_events=False,        # Don't send task events
    task_ignore_result=True               # Don't store task results
)

class FlaskTask(celery.Task):
    def __call__(self, *args, **kwargs):
        # Import the app inside the task to avoid circular imports
        import sys
        import os
        # Add the current directory to the path if not already there
        app_dir = os.path.dirname(os.path.abspath(__file__))
        if app_dir not in sys.path:
            sys.path.insert(0, app_dir)
        
        # Now import the app
        from app import app as flask_app
        with flask_app.app_context():
            return self.run(*args, **kwargs)

celery.Task = FlaskTask

# Import APScheduler but don't initialize it here
# The scheduler is now initialized in a dedicated process (scheduler.py)
from apscheduler.schedulers.background import BackgroundScheduler

# Device setup
device = "cuda" if torch.cuda.is_available() else "cpu"

# Define candidate tags
CANDIDATE_TAGS = [
    "nature", "urban", "people", "animals", "food",
    "technology", "landscape", "portrait", "abstract",
    "night", "day", "indoor", "outdoor", "colorful",
    "monochrome", "black and white", "sepia", "bright", "dark",
    "warm", "cool", "red", "green", "blue", "yellow", "orange",
    "purple", "pink", "brown", "gray", "white", "black", "minimal",
    "detailed", "texture", "pattern", "geometric", "organic", "digital",
    "analog", "illustration", "painting", "photograph", "sketch",
    "drawing", "3D", "flat", "realistic", "surreal", "fantasy",
    "sci-fi", "historical", "futuristic", "retro", "classic"
]

# Model and embeddings cache
clip_models = {}
clip_preprocessors = {}
tag_embeddings = {}

def get_clip_model():
    """Get the CLIP model based on user configuration"""
    from models import UserConfig
    from flask import current_app
    import gc  # Garbage collection
    
    # Get the selected CLIP model from user config
    config = UserConfig.query.first()
    clip_model_name = 'ViT-B-32'  # Default model (smallest and fastest)
    
    # Create config if it doesn't exist
    if not config:
        from models import db
        config = UserConfig(clip_model=clip_model_name)
        db.session.add(config)
        db.session.commit()
    elif config.clip_model:
        clip_model_name = config.clip_model
    
    # If trying to use ViT-L-14 but we're low on memory, fall back to ViT-B-32
    if clip_model_name == 'ViT-L-14':
        try:
            # Check available memory (this is a simple heuristic)
            import psutil
            available_memory = psutil.virtual_memory().available / (1024 * 1024)  # MB
            if available_memory < 1000:  # Less than 1GB available
                current_app.logger.warning(f"Low memory ({available_memory}MB available). Falling back to ViT-B-32 model.")
                clip_model_name = 'ViT-B-32'
        except ImportError:
            # If psutil is not available, just continue
            pass
    
    # Check if model is already loaded
    if clip_model_name in clip_models:
        return clip_model_name, clip_models[clip_model_name], clip_preprocessors[clip_model_name]
    
    # Clear any existing models to free memory
    for model_name in list(clip_models.keys()):
        if model_name != clip_model_name:
            del clip_models[model_name]
            del clip_preprocessors[model_name]
    
    # Force garbage collection
    gc.collect()
    
    # Load the model
    try:
        current_app.logger.info(f"Loading CLIP model: {clip_model_name}")
        model, _, preprocess = open_clip.create_model_and_transforms(clip_model_name, pretrained='openai', jit=False, force_quick_gelu=True)
        model.to(device)
        model.eval()
        
        # Cache the model and preprocessor
        clip_models[clip_model_name] = model
        clip_preprocessors[clip_model_name] = preprocess
        
        # Precompute tag embeddings for this model
        tokenizer = open_clip.get_tokenizer(clip_model_name)
        if clip_model_name not in tag_embeddings:
            tag_embeddings[clip_model_name] = {}
            with torch.no_grad():
                # Process tags in smaller batches to save memory
                batch_size = 10
                for i in range(0, len(CANDIDATE_TAGS), batch_size):
                    batch_tags = CANDIDATE_TAGS[i:i+batch_size]
                    for tag in batch_tags:
                        text_tokens = tokenizer([f"a photo of {tag}"])
                        text_features = model.encode_text(text_tokens)
                        text_features /= text_features.norm(dim=-1, keepdim=True)
                        tag_embeddings[clip_model_name][tag] = text_features.cpu()  # Store on CPU to save GPU memory
                    # Force garbage collection between batches
                    gc.collect()
        
        return clip_model_name, model, preprocess
    except Exception as e:
        current_app.logger.error(f"Error loading CLIP model {clip_model_name}: {e}")
        # Fall back to default model if available
        if 'ViT-B-32' in clip_models:
            return 'ViT-B-32', clip_models['ViT-B-32'], clip_preprocessors['ViT-B-32']
        # Otherwise load default model
        model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai', jit=False, force_quick_gelu=True)
        model.to(device)
        model.eval()
        clip_models['ViT-B-32'] = model
        clip_preprocessors['ViT-B-32'] = preprocess
        return 'ViT-B-32', model, preprocess

def get_image_embedding(image_path):
    try:
        # Get the current CLIP model
        model_name, model, preprocess = get_clip_model()
        
        # Process the image
        image = Image.open(image_path).convert("RGB")
        image_input = preprocess(image).unsqueeze(0).to(device)
        
        # Get image features
        with torch.no_grad():
            embedding = model.encode_image(image_input)
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)
        
        return embedding.cpu().numpy()[0], model_name
    except Exception as e:
        try:
            current_app.logger.error(f"Error processing image {image_path}: {e}")
        except:
            print(f"Error processing image {image_path}: {e}")
        return None, None

def generate_tags_and_description(embedding, model_name):
    """Generate tags and description based on image embedding and model"""
    from flask import current_app
    
    # If model_name is None, use default
    if model_name is None:
        model_name = 'ViT-B-32'
    
    # If model embeddings not found, try to load the model
    if model_name not in tag_embeddings:
        try:
            get_clip_model()
        except Exception as e:
            current_app.logger.error(f"Error loading model for tag generation: {e}")
            # Fall back to any available model
            if len(tag_embeddings) > 0:
                model_name = list(tag_embeddings.keys())[0]
            else:
                return [], "No tags available"
    
    # Calculate similarities with tag embeddings
    scores = {}
    for tag in CANDIDATE_TAGS:
        if tag in tag_embeddings.get(model_name, {}):
            # Get the tag embedding for this model
            tag_emb = tag_embeddings[model_name][tag]
            # Calculate similarity
            similarity = torch.cosine_similarity(
                torch.tensor(embedding).unsqueeze(0),
                tag_emb.cpu(),
                dim=1
            ).item()
            scores[tag] = similarity
    
    # Sort tags by similarity score
    sorted_tags = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    # Get top 10 tags
    top_tags = [tag for tag, _ in sorted_tags[:10]]
    
    # Create description
    description = "This image may contain " + ", ".join(top_tags) + "."
    
    return top_tags, description

@celery.task(bind=True, time_limit=600, soft_time_limit=500, max_retries=3)
def process_image_tagging(self, filename):
    """
    Process an image: compute its embedding, generate tags and a description,
    and update its record in the database.
    """
    from models import ImageDB, db, UserConfig
    from flask import current_app
    import gc  # Garbage collection
    
    # Force garbage collection at the start
    gc.collect()
    
    # Log which CLIP model is being used
    config = UserConfig.query.first()
    clip_model_name = config.clip_model if config and config.clip_model else "ViT-B-32"
    current_app.logger.info(f"Tagging image {filename} with CLIP model: {clip_model_name}")
    
    # If using the large model and we're low on memory, fall back to the smaller model
    if clip_model_name == 'ViT-L-14':
        try:
            import psutil
            available_memory = psutil.virtual_memory().available / (1024 * 1024)  # MB
            if available_memory < 1000:  # Less than 1GB available
                current_app.logger.warning(f"Low memory ({available_memory}MB available). Falling back to ViT-B-32 model.")
                clip_model_name = 'ViT-B-32'
        except ImportError:
            pass
    
    image_folder = current_app.config.get("IMAGE_FOLDER", "./images")
    image_path = os.path.join(image_folder, filename)
    if not os.path.exists(image_path):
        return {"status": "error", "message": "Image file not found", "filename": filename}
    
    try:
        # Get image embedding using the selected model
        embedding, model_name = get_image_embedding(image_path)
        if embedding is None:
            return {"status": "error", "message": "Failed to compute embedding", "filename": filename}
        
        # Force garbage collection after embedding
        gc.collect()
        
        # Generate tags and description
        tags, description = generate_tags_and_description(embedding, model_name)
        
        # Force garbage collection after tag generation
        gc.collect()
        
        # Update database
        img = ImageDB.query.filter_by(filename=filename).first()
        if img:
            img.tags = ", ".join(tags)
            img.description = description
            db.session.commit()
            
            # Force garbage collection to free memory
            gc.collect()
            
            # Clear any cached models to free memory
            if model_name in clip_models and model_name != 'ViT-B-32':
                del clip_models[model_name]
                del clip_preprocessors[model_name]
                gc.collect()
            
            return {
                "status": "success",
                "filename": filename,
                "tags": tags,
                "description": description,
                "model_used": model_name
            }
        else:
            return {"status": "error", "message": "Image not found in database", "filename": filename}
    except Exception as e:
        current_app.logger.error(f"Error in process_image_tagging: {e}")
        # Try to clean up memory
        gc.collect()
        
        # Clear all cached models to free memory
        for model_name in list(clip_models.keys()):
            del clip_models[model_name]
            del clip_preprocessors[model_name]
        gc.collect()
        
        return {"status": "error", "message": str(e), "filename": filename}

def reembed_image(filename):
    """
    Recompute the embedding, tags, and description for an image and update its record.
    """
    from models import ImageDB, db, UserConfig
    from flask import current_app
    
    # Log which CLIP model is being used
    config = UserConfig.query.first()
    clip_model_name = config.clip_model if config and config.clip_model else "ViT-B-32"
    current_app.logger.info(f"Re-tagging image {filename} with CLIP model: {clip_model_name}")
    
    image_folder = current_app.config.get("IMAGE_FOLDER", "./images")
    image_path = os.path.join(image_folder, filename)
    if not os.path.exists(image_path):
        return {"status": "error", "message": "Image file not found", "filename": filename}
    
    try:
        # Get image embedding using the selected model
        embedding, model_name = get_image_embedding(image_path)
        if embedding is None:
            return {"status": "error", "message": "Failed to compute embedding", "filename": filename}
        
        # Generate tags and description
        tags, description = generate_tags_and_description(embedding, model_name)
        
        # Update database
        img = ImageDB.query.filter_by(filename=filename).first()
        if img:
            img.tags = ", ".join(tags)
            img.description = description
            db.session.commit()
            return {
                "status": "success",
                "filename": filename,
                "tags": tags,
                "description": description,
                "model_used": model_name
            }
        else:
            return {"status": "error", "message": "Image not found in database", "filename": filename}
    except Exception as e:
        current_app.logger.error(f"Error in reembed_image: {e}")
        return {"status": "error", "message": str(e), "filename": filename}

# Global dictionary to track bulk tagging progress
BULK_PROGRESS = {}

@celery.task(bind=True)
def bulk_tag_images(self):
    """
    Process all images that do not have tags. Returns a task ID that can be used to query progress.
    """
    from models import ImageDB
    from flask import current_app
    
    try:
        images = ImageDB.query.filter((ImageDB.tags == None) | (ImageDB.tags == "")).all()
        total = len(images)
        if total == 0:
            return None
        task_id = str(uuid.uuid4())
        BULK_PROGRESS[task_id] = 0
        for i, img in enumerate(images, start=1):
            process_image_tagging.delay(img.filename)
            BULK_PROGRESS[task_id] = (i / total) * 100
        return task_id
    except Exception as e:
        current_app.logger.error(f"Error in bulk_tag_images: {e}")
        return None

@celery.task(bind=True, time_limit=1800, soft_time_limit=1500)
def reembed_all_images(self):
    """
    Rerun tagging on all images using the currently selected CLIP model.
    This is useful when changing the CLIP model.
    """
    from models import ImageDB, UserConfig
    from flask import current_app
    import gc  # Garbage collection
    
    try:
        # Get the current CLIP model
        config = UserConfig.query.first()
        clip_model_name = config.clip_model if config and config.clip_model else "ViT-B-32"
        
        # Log the operation
        current_app.logger.info(f"Rerunning tagging on all images with CLIP model: {clip_model_name}")
        
        # Get all images
        images = ImageDB.query.all()
        total = len(images)
        
        if total == 0:
            return {"status": "success", "message": "No images found to tag"}
        
        # Create a task ID for tracking progress
        task_id = str(uuid.uuid4())
        BULK_PROGRESS[task_id] = 0
        
        # Process images one at a time to manage memory better
        for i, img in enumerate(images):
            # Process one image and wait for it to complete before starting the next
            process_image_tagging.delay(img.filename)
            
            # Update progress
            BULK_PROGRESS[task_id] = min(100, (i + 1) / total * 100)
            
            # Force garbage collection after each image
            gc.collect()
            
            # Add a small delay between tasks to allow memory to be freed
            time.sleep(0.5)
            
        return {
            "status": "success",
            "message": f"Started retagging {total} images with model {clip_model_name}",
            "task_id": task_id
        }
    except Exception as e:
        current_app.logger.error(f"Error in reembed_all_images: {e}")
        # Try to clean up memory
        gc.collect()
        return {"status": "error", "message": str(e)}

def send_scheduled_image(event_id):
    """
    Send a scheduled image to a device.
    """
    from models import ScheduleEvent, Device, db
    from utils.crop_helpers import load_crop_info_from_db
    from utils.image_helpers import add_send_log_entry
    from flask import current_app
    
    try:
        event = ScheduleEvent.query.get(event_id)
        if not event:
            current_app.logger.error("Event not found: %s", event_id)
            return
        device_obj = Device.query.filter_by(address=event.device).first()
        if not device_obj:
            current_app.logger.error("Device not found for event %s", event_id)
            return

        image_folder = current_app.config.get("IMAGE_FOLDER", "./images")
        data_folder = current_app.config.get("DATA_FOLDER", "./data")
        filepath = os.path.join(image_folder, event.filename)
        if not os.path.exists(filepath):
            current_app.logger.error("Image file not found: %s", filepath)
            return

        addr = device_obj.address
        if not (addr.startswith("http://") or addr.startswith("https://")):
            addr = "http://" + addr

        with Image.open(filepath) as orig_img:
            orig_w, orig_h = orig_img.size
            parts = device_obj.resolution.split("x")
            dev_width = int(parts[0])
            dev_height = int(parts[1])
            
            # Check if device is in portrait orientation
            is_portrait = device_obj.orientation.lower() == 'portrait'
            
            # If portrait, swap width and height for target ratio calculation
            if is_portrait:
                target_ratio = dev_height / dev_width
            else:
                target_ratio = dev_width / dev_height
                
            cdata = load_crop_info_from_db(event.filename)
            if cdata:
                x = cdata.get("x", 0)
                y = cdata.get("y", 0)
                w = cdata.get("width", orig_w)
                h = cdata.get("height", orig_h)
                cropped = orig_img.crop((x, y, x + w, y + h))
            else:
                orig_ratio = orig_w / orig_h
                if orig_ratio > target_ratio:
                    new_width = int(orig_h * target_ratio)
                    left = (orig_w - new_width) // 2
                    crop_box = (left, 0, left + new_width, orig_h)
                else:
                    new_height = int(orig_w / target_ratio)
                    top = (orig_h - new_height) // 2
                    crop_box = (0, top, orig_w, top + new_height)
                cropped = orig_img.crop(crop_box)
            
            # If portrait, rotate the image 90 degrees clockwise and swap dimensions
            if is_portrait:
                cropped = cropped.rotate(-90, expand=True)  # -90 for clockwise rotation
                final_img = cropped.resize((dev_height, dev_width), Image.LANCZOS)  # Note swapped dimensions
            else:
                final_img = cropped.resize((dev_width, dev_height), Image.LANCZOS)
            temp_dir = os.path.join(data_folder, "temp")
            if not os.path.exists(temp_dir):
                os.makedirs(temp_dir)
            temp_filename = os.path.join(temp_dir, f"temp_{event.filename}")
            final_img.save(temp_filename, format="JPEG", quality=95)
        
        cmd = f'curl "{addr}/send_image" -X POST -F "file=@{temp_filename}"'
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        os.remove(temp_filename)
        
        if result.returncode == 0:
            event.sent = True
            db.session.commit()
            device_obj.last_sent = event.filename
            db.session.commit()
            add_send_log_entry(event.filename)
        else:
            current_app.logger.error("Error sending image: %s", result.stderr)
    except Exception as e:
        current_app.logger.error("Error in send_scheduled_image: %s", e)
        return

    # Reschedule recurring events
    if event.recurrence and event.recurrence.lower() != "none":
        try:
            dt = datetime.datetime.fromisoformat(event.datetime_str)
        except Exception as e:
            current_app.logger.error("Error parsing datetime_str: %s", e)
            return
        if event.recurrence.lower() == "daily":
            next_dt = dt + datetime.timedelta(days=1)
        elif event.recurrence.lower() == "weekly":
            next_dt = dt + datetime.timedelta(weeks=1)
        elif event.recurrence.lower() == "monthly":
            next_dt = dt + datetime.timedelta(days=30)
        else:
            next_dt = None
        if next_dt:
            # Update the event in the database with the new date
            event.datetime_str = next_dt.isoformat(sep=' ')
            event.sent = False
            db.session.commit()
            
            # Note: We don't schedule the job here anymore
            # The dedicated scheduler process will pick up this event on its next run
            current_app.logger.info(f"Rescheduled event {event.id} for {next_dt}")

@celery.task(bind=True, ignore_result=True)
def fetch_device_metrics(self):
    """
    Check if devices are online and update their status in the database.
    This task runs periodically to ensure we have up-to-date status information.
    Logging has been minimized to prevent log flooding.
    """
    from models import Device, db
    from flask import current_app
    import httpx
    
    # Disable httpx logging
    import logging
    logging.getLogger("httpx").setLevel(logging.WARNING)
    
    try:
        devices = Device.query.all()
        status_changes = 0
        
        for device in devices:
            try:
                # Ensure the address has a scheme
                address = device.address
                if not (address.startswith("http://") or address.startswith("https://")):
                    address = "http://" + address
                
                # Store previous status to detect changes
                was_online = device.online
                
                # Try to connect to the device's display_info endpoint
                try:
                    # Use a short timeout to avoid blocking
                    response = httpx.get(f"{address}/display_info", timeout=5.0)
                    
                    if response.status_code == 200:
                        # Only update and log if status changed
                        if not was_online:
                            device.online = True
                            db.session.commit()
                            status_changes += 1
                    else:
                        # Only update and log if status changed
                        if was_online:
                            device.online = False
                            db.session.commit()
                            status_changes += 1
                except Exception:
                    # Only update if status changed
                    if was_online:
                        device.online = False
                        db.session.commit()
                        status_changes += 1
            except Exception:
                # Silently continue to next device
                pass
                
        # Only log if there were status changes
        if status_changes > 0:
            current_app.logger.info(f"Updated status for {status_changes} devices")
            
        return {"status": "success", "message": f"Checked status of {len(devices)} devices"}
    except Exception:
        # Catch all exceptions to prevent task failures
        return {"status": "error", "message": "Error checking device status"}

# The scheduler functionality has been moved to scheduler.py
# This function is kept as a stub for backward compatibility
def start_scheduler(app):
    """
    This function is deprecated and no longer starts the scheduler.
    The scheduler is now started in a dedicated process (scheduler.py).
    """
    app.logger.info("Scheduler initialization skipped - now handled by dedicated process")
```


## webpack.config.js

```js
const path = require('path');

module.exports = {
  entry: './assets/js/main.js', 
  output: {
    filename: 'bundle.js',
    path: path.resolve(__dirname, 'static', 'js')
  },
  module: {
    rules: [
      {
        test: /\.css$/i,
        use: ['style-loader', 'css-loader'],
      },
    ],
  },
};
```

