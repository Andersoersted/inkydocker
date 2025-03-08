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
            return jsonify({"status": "error", "message": "Invalid minimum tags value. Must be a positive integer."})
    
    if updated:
        db.session.commit()
        return jsonify({"status": "success", "message": "Settings updated successfully."})
    else:
        return jsonify({"status": "error", "message": "No valid settings provided."})

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

@settings_bp.route('/settings/verify_clip_model', methods=['GET'])
def verify_clip_model():
    """
    Endpoint to verify which CLIP model is currently being used.
    This helps users confirm they're using the correct model for tagging.
    """
    try:
        from models import UserConfig
        
        # Get the current CLIP model from user config
        config = UserConfig.query.first()
        
        if not config:
            return jsonify({
                "status": "error",
                "message": "No configuration found"
            }), 404
            
        clip_model_name = config.clip_model if config.clip_model else "ViT-B-32"
        
        # Log the verification request
        logger.info(f"CLIP model verification requested: current model is {clip_model_name}")
        
        return jsonify({
            "status": "success",
            "model_name": clip_model_name,
            "message": f"Current CLIP model is {clip_model_name}"
        })
    except Exception as e:
        logger.error(f"Error verifying CLIP model: {str(e)}")
        return jsonify({"status": "error", "message": f"Error: {str(e)}"})

@settings_bp.route('/settings/test_tagging', methods=['POST'])
def test_tagging():
    """
    Endpoint to test image tagging with the current CLIP model.
    This allows users to verify the model is working correctly and see the generated tags.
    """
    try:
        from tasks import get_image_embedding, COSINE_THRESHOLD
        from models import UserConfig
        import os
        import tempfile
        import torch
        
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
            
        # Get the current CLIP model and settings
        config = UserConfig.query.first()
        if not config:
            return jsonify({
                "status": "error",
                "message": "No configuration found"
            }), 404
            
        clip_model_name = config.clip_model if config.clip_model else "ViT-B-32"
        max_tags = config.min_tags if config.min_tags else 5
        
        # Save the uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp:
            file.save(temp.name)
            temp_path = temp.name
        
        try:
            # Get the image embedding
            embedding, model_used = get_image_embedding(temp_path)
            if embedding is None:
                os.unlink(temp_path)  # Clean up the temporary file
                return jsonify({
                    "status": "error",
                    "message": "Failed to compute image embedding"
                }), 500
                
            # Calculate similarities with tag embeddings
            from tasks import tag_embeddings, CANDIDATE_TAGS
            
            scores = {}
            for tag in CANDIDATE_TAGS:
                if tag in tag_embeddings.get(model_used, {}):
                    # Get the tag embedding for this model
                    tag_emb = tag_embeddings[model_used][tag]
                    # Calculate similarity
                    similarity = torch.cosine_similarity(
                        torch.tensor(embedding).unsqueeze(0),
                        tag_emb.cpu(),
                        dim=1
                    ).item()
                    scores[tag] = similarity
            
            # Sort tags by similarity score
            sorted_tags = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            
            # Filter tags by cosine similarity threshold and limit to max_tags
            filtered_tags = []
            for tag, score in sorted_tags:
                # Only include tags with similarity above the threshold
                if score >= COSINE_THRESHOLD:
                    filtered_tags.append({"tag": tag, "score": score})
                    # Stop once we reach the maximum number of tags
                    if len(filtered_tags) >= max_tags:
                        break
            
            # Include all tags with scores for display in the UI
            all_tags_with_scores = [{"tag": tag, "score": score} for tag, score in sorted_tags[:20]]
            
            # Log the test results
            logger.info(f"Test tagging completed with model {model_used}: generated {len(filtered_tags)} tags")
            
            # Clean up the temporary file
            os.unlink(temp_path)
            
            return jsonify({
                "status": "success",
                "model_used": model_used,
                "tags_with_scores": all_tags_with_scores,
                "filtered_tags": filtered_tags,
                "threshold": COSINE_THRESHOLD,
                "max_tags": max_tags
            })
            
        except Exception as e:
            # Clean up the temporary file in case of error
            os.unlink(temp_path)
            raise e
            
    except Exception as e:
        logger.error(f"Error in test_tagging: {str(e)}")
        return jsonify({
            "status": "error",
            "message": f"Error: {str(e)}"
        }), 500

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