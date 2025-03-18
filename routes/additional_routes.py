from flask import Blueprint, request, jsonify, current_app
from models import Device, db
import json
import httpx

additional_bp = Blueprint('additional', __name__)

@additional_bp.route('/fetch_display_info', methods=['GET'])
def fetch_display_info():
    address = request.args.get('address')
    if not address:
        return jsonify({"status": "error", "message": "No address provided"}), 400
        
    # Ensure address has HTTP protocol
    if not address.startswith(('http://', 'https://')):
        address = f'http://{address}'
        
    url = f"{address}/display_info"
    
    try:
        # Use httpx with a short timeout to match curl's behavior
        with httpx.Client(timeout=10.0) as client:
            response = client.get(url)
            
        # Check if request was successful
        if response.status_code != 200:
            return jsonify({"status": "error", "message": f"HTTP error: {response.status_code}"}), 500
        
        # Parse the JSON response
        raw_info = response.json()
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
