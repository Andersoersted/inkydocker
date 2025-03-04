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
