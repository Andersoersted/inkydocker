from flask import Blueprint, request, render_template, flash, redirect, url_for, jsonify, current_app
from models import db, Device, UserConfig
import logging
import os
import torch
from datetime import datetime
import json
from utils.notification_service import NotificationService, notify_on_completion

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
            # Create a single notification for adding display
            NotificationService.create_info("Adding new E-Ink display...")
            
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
            
            # Create success notification
            NotificationService.create_success(f"E-Ink display '{friendly_name}' added successfully")
        else:
            error_msg = "Missing mandatory fields (color, friendly name, orientation, address)."
            flash(error_msg, "error")
            # Create error notification
            NotificationService.create_error(error_msg)
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
    # Create a single notification for device deletion
    NotificationService.create_info("Deleting E-Ink display...")
    
    all_devices = Device.query.order_by(Device.id).all()
    if 0 <= device_index < len(all_devices):
        device = all_devices[device_index]
        device_name = device.friendly_name
        db.session.delete(device)
        db.session.commit()
        flash("Device deleted", "success")
        # Create a success notification
        NotificationService.create_success(f"E-Ink display '{device_name}' deleted successfully")
    else:
        error_msg = f"Device with index {device_index} not found"
        flash("Device not found", "error")
        # Create error notification
        NotificationService.create_error(error_msg)
    return redirect(url_for("settings.settings"))

@settings_bp.route('/edit_device', methods=['POST'])
def edit_device():
    try:
        # Create notification at the start
        NotificationService.create_info("Updating E-Ink display...")
        
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
            # Create success notification
            NotificationService.create_success(f"E-Ink display '{friendly_name}' updated successfully")
        else:
            error_msg = f"Device with index {index} not found"
            flash("Device index not found", "error")
            # Create error notification
            NotificationService.create_error(error_msg)
    except Exception as e:
        error_msg = f"Error editing device: {str(e)}"
        flash(error_msg, "error")
        # Create error notification
        NotificationService.create_error(error_msg)
    return redirect(url_for("settings.settings"))

@settings_bp.route('/settings/update_clip_model', methods=['POST'])
def update_clip_model():
    # Create info notification at start
    NotificationService.create_info("Updating AI model settings...")
    
    data = request.get_json()
    config = UserConfig.query.first()
    if not config:
        config = UserConfig(location="London")
        db.session.add(config)
    
    updated = False
    
    if "clip_model" in data:
        config.clip_model = data.get("clip_model")
        updated = True
    
    if "min_tags" in data:
        min_tags = data.get("min_tags")
        if isinstance(min_tags, int) and min_tags > 0:
            config.min_tags = min_tags
            updated = True
        else:
            error_msg = "Invalid minimum tags value. Must be a positive integer."
            # Create error notification
            NotificationService.create_error(error_msg)
            return jsonify({"status": "error", "message": error_msg})
    
    if "similarity_threshold" in data:
        threshold = data.get("similarity_threshold")
        valid_thresholds = ["very_high", "high", "medium", "low", "very_low"]
        if threshold in valid_thresholds:
            config.similarity_threshold = threshold
            updated = True
        else:
            error_msg = "Invalid similarity threshold value."
            # Create error notification
            NotificationService.create_error(error_msg)
            return jsonify({"status": "error", "message": error_msg})
    
    if updated:
        db.session.commit()
        # Create more specific success notification
        message = "AI model settings updated successfully"
        if "clip_model" in data:
            message = f"AI model updated to {data.get('clip_model')}"
        NotificationService.create_success(message)
        return jsonify({"status": "success", "message": "Settings updated successfully."})
    else:
        error_msg = "No valid settings provided."
        # Create error notification
        NotificationService.create_error(error_msg)
        return jsonify({"status": "error", "message": error_msg})

@settings_bp.route('/settings/rerun_all_tagging', methods=['POST'])
def rerun_all_tagging():
    try:
        # Create start notification
        NotificationService.create_info("Starting AI retagging of all images...")
        
        # Import the task for rerunning tagging
        from tasks import reembed_all_images
        
        # Start the task
        task = reembed_all_images.delay()
        
        # Create a notification with more details
        NotificationService.create_success(f"AI retagging process initiated (Task ID: {task.id}). This might take a while.")
        
        return jsonify({
            "status": "success",
            "message": "Tagging process started.",
            "task_id": str(task.id)
        })
    except Exception as e:
        logger.error(f"Error starting retagging: {str(e)}")
        
        # Create error notification
        NotificationService.create_error(f"Failed to start AI retagging: {str(e)}")
        
        return jsonify({"status": "error", "message": f"Error: {str(e)}"})

@settings_bp.route('/settings/verify_clip_model', methods=['GET'])
def verify_clip_model():
    """
    Endpoint to verify which CLIP model is currently being used.
    This helps users confirm they're using the correct model for tagging.
    """
    try:
        from models import UserConfig
        
        # Create info notification at start
        NotificationService.create_info("Verifying current AI model...")
        
        # Get the current CLIP model from user config
        config = UserConfig.query.first()
        
        if not config:
            error_msg = "No configuration found"
            NotificationService.create_error(error_msg)
            return jsonify({
                "status": "error",
                "message": error_msg
            }), 404
            
        clip_model_name = config.clip_model if config.clip_model else "ViT-B-32"
        
        # Log the verification request
        logger.info(f"CLIP model verification requested: current model is {clip_model_name}")
        
        # Create success notification
        NotificationService.create_success(f"AI model verification completed: {clip_model_name}")
        
        return jsonify({
            "status": "success",
            "model_name": clip_model_name,
            "message": f"Current CLIP model is {clip_model_name}"
        })
    except Exception as e:
        logger.error(f"Error verifying CLIP model: {str(e)}")
        
        # Create error notification
        NotificationService.create_error(f"Failed to verify AI model: {str(e)}")
        
        return jsonify({"status": "error", "message": f"Error: {str(e)}"})

@settings_bp.route('/settings/test_tagging', methods=['POST'])
def test_tagging():
    """
    Endpoint to test image tagging with the current CLIP model.
    This allows users to verify the model is working correctly, see the generated tags, and view the uploaded image.
    """
    try:
        from tasks import get_image_embedding
        from models import UserConfig
        import os
        import time
        import torch
        from flask import current_app, url_for

        # Check if a file was uploaded
        if 'file' not in request.files:
            return jsonify({
                "status": "error",
                "message": "No file uploaded"
            }), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({
                "status": "error",
                "message": "No file selected"
            }), 400

        # Get the current configuration
        config = UserConfig.query.first()
        if not config:
            return jsonify({
                "status": "error",
                "message": "No configuration found"
            }), 404

        # Allow overriding the CLIP model via form data; fallback to config or default
        clip_model_name = request.form.get("clip_model_override") or (config.clip_model if config.clip_model else "ViT-B-32")
        max_tags = config.min_tags if config.min_tags else 5
        threshold_level = config.similarity_threshold if hasattr(config, 'similarity_threshold') and config.similarity_threshold else "medium"

        # Get the actual cosine threshold value from the level
        from tasks import SIMILARITY_THRESHOLDS, DEFAULT_THRESHOLD
        cosine_threshold = SIMILARITY_THRESHOLDS.get(threshold_level, SIMILARITY_THRESHOLDS[DEFAULT_THRESHOLD])

        # Save the uploaded file to a persistent location in static/uploads
        upload_folder = os.path.join(current_app.root_path, "static", "uploads")
        if not os.path.exists(upload_folder):
            os.makedirs(upload_folder)
        filename = f"test_{int(time.time())}_{file.filename}"
        file_path = os.path.join(upload_folder, filename)
        file.save(file_path)

        # Get the image embedding
        embedding, model_used = get_image_embedding(file_path)
        if embedding is None:
            os.remove(file_path)
            return jsonify({
                "status": "error",
                "message": "Failed to compute image embedding"
            }), 500

        # Calculate similarities with tag embeddings
        from tasks import tag_embeddings, CANDIDATE_TAGS
        scores = {}
        for tag in CANDIDATE_TAGS:
            if tag in tag_embeddings.get(model_used, {}):
                tag_emb = tag_embeddings[model_used][tag]
                similarity = torch.cosine_similarity(
                    torch.tensor(embedding).unsqueeze(0),
                    tag_emb.cpu(),
                    dim=1
                ).item()
                scores[tag] = similarity
        sorted_tags = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        filtered_tags = []
        for tag, score in sorted_tags:
            if score >= cosine_threshold:
                filtered_tags.append({"tag": tag, "score": score})
                if len(filtered_tags) >= max_tags:
                    break
        all_tags_with_scores = [{"tag": tag, "score": score} for tag, score in sorted_tags[:20]]
        threshold_descriptions = {
            "very_high": "Very High - Only exact matches",
            "high": "High - Strong matches",
            "medium": "Medium - Balanced matches",
            "low": "Low - More inclusive matches",
            "very_low": "Very Low - Most inclusive matches"
        }
        threshold_description = threshold_descriptions.get(threshold_level, "Medium - Balanced matches")

        # Build the URL for the uploaded image
        image_url = url_for('static', filename=f"uploads/{filename}", _external=True)

        return jsonify({
            "status": "success",
            "model_used": model_used,
            "tags_with_scores": all_tags_with_scores,
            "filtered_tags": filtered_tags,
            "threshold": cosine_threshold,
            "threshold_level": threshold_level,
            "threshold_description": threshold_description,
            "max_tags": max_tags,
            "uploaded_image": image_url
        })
    except Exception as e:
        logger.error(f"Error in test_tagging: {str(e)}")
        return jsonify({
            "status": "error",
            "message": f"Error: {str(e)}"
        }), 500

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

# Note: Model download routes have been removed as all models are now preloaded with the Docker image