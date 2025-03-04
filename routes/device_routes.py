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