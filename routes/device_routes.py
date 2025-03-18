from flask import Blueprint, request, jsonify, current_app, send_file
from models import db, Device
import httpx
import os, datetime, json

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
    
    # Ensure address has HTTP protocol
    if not device_address.startswith(('http://', 'https://')):
        device_address = f'http://{device_address}'
    
    def generate():
        try:
            # Stream the response using httpx
            with httpx.stream('GET', f"{device_address}/stream", timeout=None) as response:
                for line in response.iter_lines():
                    if line:
                        # Return each line from the stream
                        yield line.decode('utf-8') + '\n'
        except Exception as e:
            current_app.logger.error(f"Error streaming metrics: {e}")
            yield f"data: {{\"error\": \"{str(e)}\"}}\n\n"
    
    return current_app.response_class(generate(), mimetype='text/event-stream')

@device_bp.route('/test_device/<int:index>', methods=['GET'])
def test_device(index):
    all_devices = Device.query.order_by(Device.id).all()
    if index < 0 or index >= len(all_devices):
        return jsonify({"status": "error", "message": "Device not found"}), 404
    device = all_devices[index]
    address = device.address
    
    # Ensure address has HTTP protocol
    if not address.startswith(('http://', 'https://')):
        address = f'http://{address}'
    
    try:
        # Use httpx for the HEAD request with a 5-second timeout
        response = httpx.head(address, timeout=5.0, follow_redirects=True)
        current_app.logger.info("HTTP status for %s: %s", address, response.status_code)
        
        if response.status_code == 200:
            device.online = True
            db.session.commit()
            return jsonify({"status": "ok"}), 200
        else:
            device.online = False
            db.session.commit()
            return jsonify({"status": "error", "message": f"HTTP status code: {response.status_code}"}), 500
    except Exception as e:
        current_app.logger.error("Error testing device %s: %s", address, str(e))
        device.online = False
        db.session.commit()
        return jsonify({"status": "error", "message": str(e)}), 500

@device_bp.route('/test_connection_address', methods=['GET'])
def test_connection_address():
    address = request.args.get('address')
    if not address:
        return jsonify({"status": "error", "message": "No address provided"}), 400
    
    # Ensure address has HTTP protocol
    if not address.startswith(('http://', 'https://')):
        address = f'http://{address}'
    
    try:
        # Use httpx for the HEAD request with a 5-second timeout
        response = httpx.head(address, timeout=5.0, follow_redirects=True)
        current_app.logger.info("HTTP status for %s: %s", address, response.status_code)
        
        if response.status_code == 200:
            return jsonify({"status": "ok"}), 200
        else:
            return jsonify({"status": "failed", "message": f"HTTP status code: {response.status_code}"}), 500
    except Exception as e:
        current_app.logger.error("Error testing connection to %s: %s", address, str(e))
        return jsonify({"status": "failed", "message": str(e)}), 500