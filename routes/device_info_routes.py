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