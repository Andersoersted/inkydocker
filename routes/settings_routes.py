from flask import Blueprint, request, render_template, flash, redirect, url_for, jsonify, current_app
from models import db, Device, UserConfig
import logging
import httpx
import os
import torch
import open_clip
from datetime import datetime
from tqdm import tqdm
import threading
import time
import json

settings_bp = Blueprint('settings', __name__)
logger = logging.getLogger(__name__)

# Dictionary to track model download progress
model_download_progress = {}

def download_model_thread(model_name, task_id, app):
    """
    Thread function to download a model and track progress.
    """
    try:
        # Use the application context
        with app.app_context():
            # Create data directory if it doesn't exist
            data_folder = current_app.config.get("DATA_FOLDER", "./data")
            models_folder = os.path.join(data_folder, "models")
            if not os.path.exists(models_folder):
                os.makedirs(models_folder)
            
            # Update progress to 5%
            model_download_progress[task_id] = {
                "progress": 5,
                "status": "downloading",
                "model_name": model_name
            }
            
            # Load the model - this will download it if not already cached
            logger.info(f"Starting download of model: {model_name}")
            
            # Get available pretrained tags for the model
            available_pretrained_tags = []
            try:
                # Try to get available pretrained tags
                import open_clip
                available_pretrained_tags = open_clip.list_pretrained_tags_by_model(model_name)
                logger.info(f"Available pretrained tags for {model_name}: {available_pretrained_tags}")
            except Exception as e:
                logger.info(f"Could not get pretrained tags for {model_name}: {str(e)}")
                # If we can't get tags, use default tags to try
                available_pretrained_tags = ['openai', 'laion2b_s34b_b79k', 'datacomp1b', 'laion400m_e32']
            
            # If no tags found, use default list
            if not available_pretrained_tags:
                available_pretrained_tags = ['openai', 'laion2b_s34b_b79k', 'datacomp1b', 'laion400m_e32']
            
            # Try different model name formats and pretrained tags
            success = False
            last_error = None
            
            # List of model name variations to try
            model_variations = []
            
            # Original model name
            model_variations.append(model_name)
            
            # If model name contains a slash, try different variations
            if '/' in model_name:
                # Extract the model architecture name (after the slash)
                model_arch = model_name.split('/')[-1]
                model_variations.append(model_arch)
                
                # Try with organization as prefix
                org = model_name.split('/')[0]
                model_variations.append(f"{org}-{model_arch}")
                
                # Try with underscores instead of dashes
                model_variations.append(model_arch.replace('-', '_'))
            
            # Try each model variation with each pretrained tag
            total_attempts = len(model_variations) * len(available_pretrained_tags)
            current_attempt = 0
            
            for model_var in model_variations:
                logger.info(f"Trying model variation: {model_var}")
                
                for pretrained_tag in available_pretrained_tags:
                    current_attempt += 1
                    
                    # Update progress based on attempts (from 5% to 80%)
                    progress_percent = 5 + (current_attempt / total_attempts) * 75
                    model_download_progress[task_id] = {
                        "progress": int(progress_percent),
                        "status": "downloading",
                        "model_name": model_name,
                        "current_attempt": f"Trying {model_var} with {pretrained_tag}"
                    }
                    
                    try:
                        logger.info(f"Attempting to load {model_var} with pretrained tag: {pretrained_tag}")
                        # Set a recursion limit for model downloading
                        import sys
                        old_recursion_limit = sys.getrecursionlimit()
                        sys.setrecursionlimit(3000)  # Increase recursion limit temporarily
                        
                        try:
                            model, _, preprocess = open_clip.create_model_and_transforms(
                                model_var,
                                pretrained=pretrained_tag,
                                cache_dir=models_folder
                            )
                        finally:
                            # Restore original recursion limit
                            sys.setrecursionlimit(old_recursion_limit)
                        # If we get here, the model loaded successfully
                        logger.info(f"Successfully loaded {model_var} with pretrained tag: {pretrained_tag}")
                        success = True
                        
                        # Save which pretrained tag was used
                        model_info = {
                            "name": model_name,
                            "download_date": datetime.now().isoformat(),
                            "path": models_folder,
                            "pretrained_tag": pretrained_tag,
                            "model_variation": model_var
                        }
                        break
                    except Exception as e:
                        last_error = e
                        logger.info(f"Failed to load {model_var} with pretrained tag {pretrained_tag}: {str(e)}")
                
                if success:
                    break
            
            # If all attempts failed, raise the last error
            if not success:
                raise Exception(f"Failed to load model {model_name} with any variation or pretrained tag. Last error: {str(last_error)}")
            
            # Update progress to 90%
            model_download_progress[task_id] = {
                "progress": 90,
                "status": "finalizing",
                "model_name": model_name
            }
            
            # If we didn't set model_info earlier (shouldn't happen), create a default one
            if not 'model_info' in locals():
                model_info = {
                    "name": model_name,
                    "download_date": datetime.now().isoformat(),
                    "path": models_folder
                }
            
            with open(os.path.join(models_folder, f"{model_name.replace('/', '_')}_info.json"), 'w') as f:
                json.dump(model_info, f)
            
            # Update progress to 100%
            model_download_progress[task_id] = {
                "progress": 100,
                "status": "completed",
                "model_name": model_name
            }
            
            logger.info(f"Successfully downloaded model: {model_name}")
            
            # Keep the completed status for a while before removing
            time.sleep(30)
            if task_id in model_download_progress:
                del model_download_progress[task_id]
                
    except Exception as e:
        logger.error(f"Error downloading model {model_name}: {str(e)}")
        model_download_progress[task_id] = {
            "progress": 0,
            "status": "error",
            "model_name": model_name,
            "error": str(e)
        }
        # Keep the error status for a while before removing
        time.sleep(60)
        if task_id in model_download_progress:
            del model_download_progress[task_id]

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
    
    if "similarity_threshold" in data:
        threshold = data.get("similarity_threshold")
        valid_thresholds = ["very_high", "high", "medium", "low", "very_low"]
        if threshold in valid_thresholds:
            config.similarity_threshold = threshold
            updated = True
        else:
            return jsonify({"status": "error", "message": "Invalid similarity threshold value."})
    
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
        from tasks import get_image_embedding
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
        threshold_level = config.similarity_threshold if hasattr(config, 'similarity_threshold') and config.similarity_threshold else "medium"
        
        # Get the actual cosine threshold value from the level
        from tasks import SIMILARITY_THRESHOLDS, DEFAULT_THRESHOLD
        cosine_threshold = SIMILARITY_THRESHOLDS.get(threshold_level, SIMILARITY_THRESHOLDS[DEFAULT_THRESHOLD])
        
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
                if score >= cosine_threshold:
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
            
            # Get human-readable descriptions for the threshold levels
            threshold_descriptions = {
                "very_high": "Very High - Only exact matches",
                "high": "High - Strong matches",
                "medium": "Medium - Balanced matches",
                "low": "Low - More inclusive matches",
                "very_low": "Very Low - Most inclusive matches"
            }
            
            threshold_description = threshold_descriptions.get(threshold_level, "Medium - Balanced matches")
            
            return jsonify({
                "status": "success",
                "model_used": model_used,
                "tags_with_scores": all_tags_with_scores,
                "filtered_tags": filtered_tags,
                "threshold": cosine_threshold,
                "threshold_level": threshold_level,
                "threshold_description": threshold_description,
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

@settings_bp.route('/settings/download_model', methods=['POST'])
def download_model():
    """
    Endpoint to download a custom CLIP model.
    """
    try:
        data = request.get_json()
        model_name = data.get("model_name")
        
        if not model_name:
            return jsonify({
                "status": "error",
                "message": "Model name is required"
            }), 400
        
        # Check if the model is already being downloaded
        for task_id, info in model_download_progress.items():
            if info.get("model_name") == model_name and info.get("status") in ["downloading", "finalizing"]:
                return jsonify({
                    "status": "error",
                    "message": f"Model {model_name} is already being downloaded",
                    "task_id": task_id
                }), 409
        
        # Generate a task ID for tracking progress
        task_id = f"download_{int(time.time())}"
        
        # Initialize progress tracking
        model_download_progress[task_id] = {
            "progress": 0,
            "status": "starting",
            "model_name": model_name
        }
        
        # Get the current app
        from flask import current_app
        app = current_app._get_current_object()  # Get the actual app object, not the proxy
        
        # Start download in a separate thread
        thread = threading.Thread(
            target=download_model_thread,
            args=(model_name, task_id, app)
        )
        thread.daemon = True
        thread.start()
        
        return jsonify({
            "status": "success",
            "message": f"Started downloading model {model_name}",
            "task_id": task_id
        })
        
    except Exception as e:
        logger.error(f"Error in download_model: {str(e)}")
        return jsonify({
            "status": "error",
            "message": f"Error: {str(e)}"
        }), 500

@settings_bp.route('/settings/download_progress/<task_id>', methods=['GET'])
def download_progress(task_id):
    """
    Endpoint to check the progress of a model download.
    """
    if task_id in model_download_progress:
        return jsonify({
            "status": "success",
            "progress": model_download_progress[task_id]
        })
    else:
        return jsonify({
            "status": "error",
            "message": "Download task not found"
        }), 404

@settings_bp.route('/settings/list_custom_models', methods=['GET'])
def list_custom_models():
    """
    Endpoint to list all downloaded custom models.
    """
    try:
        data_folder = current_app.config.get("DATA_FOLDER", "./data")
        models_folder = os.path.join(data_folder, "models")
        
        if not os.path.exists(models_folder):
            return jsonify({
                "status": "success",
                "models": []
            })
        
        # Look for model info files
        models = []
        for filename in os.listdir(models_folder):
            if filename.endswith("_info.json"):
                try:
                    with open(os.path.join(models_folder, filename), 'r') as f:
                        model_info = json.load(f)
                        models.append(model_info)
                except Exception as e:
                    logger.error(f"Error reading model info file {filename}: {str(e)}")
        
        return jsonify({
            "status": "success",
            "models": models
        })
        
    except Exception as e:
        logger.error(f"Error in list_custom_models: {str(e)}")
        return jsonify({
            "status": "error",
            "message": f"Error: {str(e)}"
        }), 500

@settings_bp.route('/settings/list_available_models', methods=['GET'])
def list_available_models():
    """
    Endpoint to list all available models from OpenCLIP.
    """
    try:
        import open_clip
        
        # Get all available model names
        available_models = open_clip.list_models()
        
        return jsonify({
            "status": "success",
            "models": available_models
        })
        
    except Exception as e:
        logger.error(f"Error in list_available_models: {str(e)}")
        return jsonify({
            "status": "error",
            "message": f"Error: {str(e)}"
        }), 500

@settings_bp.route('/settings/delete_model', methods=['POST'])
def delete_model():
    """
    Endpoint to delete a downloaded model.
    """
    try:
        data = request.get_json()
        model_name = data.get("model_name")
        
        if not model_name:
            return jsonify({
                "status": "error",
                "message": "Model name is required"
            }), 400
        
        # Get the data folder
        data_folder = current_app.config.get("DATA_FOLDER", "./data")
        models_folder = os.path.join(data_folder, "models")
        
        # Check if the model info file exists
        model_info_path = os.path.join(models_folder, f"{model_name.replace('/', '_')}_info.json")
        if not os.path.exists(model_info_path):
            return jsonify({
                "status": "error",
                "message": f"Model {model_name} not found"
            }), 404
        
        # Delete the model info file
        os.remove(model_info_path)
        
        # Check if this model is currently selected in the config
        config = UserConfig.query.first()
        if config and config.clip_model == model_name:
            # Reset to default model
            config.clip_model = "ViT-B-32"
            db.session.commit()
            logger.info(f"Reset clip_model to default after deleting {model_name}")
        
        # Also check if it's the custom model
        if config and config.custom_model == model_name:
            # Reset custom model
            config.custom_model = None
            config.custom_model_enabled = False
            db.session.commit()
            logger.info(f"Reset custom_model after deleting {model_name}")
        
        return jsonify({
            "status": "success",
            "message": f"Model {model_name} deleted successfully"
        })
        
    except Exception as e:
        logger.error(f"Error in delete_model: {str(e)}")
        return jsonify({
            "status": "error",
            "message": f"Error: {str(e)}"
        }), 500

@settings_bp.route('/settings/enable_custom_model', methods=['POST'])
def enable_custom_model():
    """
    Endpoint to enable or disable a custom model.
    """
    try:
        data = request.get_json()
        model_name = data.get("model_name")
        enabled = data.get("enabled", False)
        
        if not model_name and enabled:
            return jsonify({
                "status": "error",
                "message": "Model name is required when enabling a custom model"
            }), 400
        
        # Get the current config
        config = UserConfig.query.first()
        if not config:
            config = UserConfig(location="London")
            db.session.add(config)
        
        # Update the config
        config.custom_model = model_name if enabled else None
        config.custom_model_enabled = enabled
        db.session.commit()
        
        return jsonify({
            "status": "success",
            "message": f"Custom model {'enabled' if enabled else 'disabled'} successfully"
        })
        
    except Exception as e:
        logger.error(f"Error in enable_custom_model: {str(e)}")
        return jsonify({
            "status": "error",
            "message": f"Error: {str(e)}"
        }), 500