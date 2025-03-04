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