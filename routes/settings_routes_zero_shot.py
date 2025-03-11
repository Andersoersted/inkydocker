from flask import Blueprint, request, jsonify, current_app
import logging
import threading
import time
import os
import json
from datetime import datetime
from models import db, UserConfig

# Create a blueprint for zero-shot settings routes
zero_shot_bp = Blueprint('zero_shot', __name__)
logger = logging.getLogger(__name__)

# Dictionary to track model download progress
model_download_progress = {}

def download_zero_shot_model_thread(model_name, task_id, app):
    """
    Thread function to download an OpenCLIP model and track progress.
    """
    try:
        # Use the application context
        with app.app_context():
            # Update progress
            model_download_progress[task_id] = {
                "progress": 10,
                "status": "downloading",
                "model_name": model_name,
                "current_attempt": f"Downloading {model_name} from OpenCLIP"
            }
            
            # Import zero-shot tagger and MODELS dictionary
            from utils.zero_shot_tagger import MODELS
            
            # Get model info
            if model_name not in MODELS:
                raise ValueError(f"Unknown zero-shot model: {model_name}")
            
            model_info = MODELS[model_name]
            
            # Set up cache directory
            data_folder = current_app.config.get("DATA_FOLDER", "./data")
            models_folder = os.path.join(data_folder, "models")
            os.makedirs(models_folder, exist_ok=True)
            
            # Download models with appropriate recursion limit
            import sys
            old_recursion_limit = sys.getrecursionlimit()
            try:
                # Increase recursion limit temporarily to handle deep call stacks
                recursion_limit = 15000
                sys.setrecursionlimit(recursion_limit)
                logger.info(f"Temporarily increased recursion limit to {recursion_limit} for loading OpenCLIP model")
                
                # Download the model using open_clip
                import open_clip
                
                # Update progress
                model_download_progress[task_id] = {
                    "progress": 20,
                    "status": "downloading",
                    "model_name": model_name,
                    "current_attempt": f"Downloading OpenCLIP model: {model_info['model_name']} (pretrained={model_info['pretrained']})"
                }
                
                # Download OpenCLIP model
                logger.info(f"Downloading OpenCLIP model: {model_info['model_name']} (pretrained={model_info['pretrained']})")
                model, _, preprocess = open_clip.create_model_and_transforms(
                    model_info['model_name'],
                    pretrained=model_info['pretrained'],
                    cache_dir=models_folder
                )
                logger.info(f"Successfully downloaded OpenCLIP model: {model_info['model_name']}")
                
            finally:
                # Restore original recursion limit
                sys.setrecursionlimit(old_recursion_limit)
                logger.info(f"Restored recursion limit to {old_recursion_limit}")
            
            # Update progress to complete
            model_download_progress[task_id] = {
                "progress": 100,
                "status": "completed",
                "model_name": model_name
            }
            
            # Update the user config to use this model
            config = UserConfig.query.first()
            if not config:
                config = UserConfig()
                db.session.add(config)
            
            config.zero_shot_model = model_name
            config.zero_shot_enabled = True
            db.session.commit()
            
            # Keep the completed status for a while before removing
            time.sleep(30)
            if task_id in model_download_progress:
                del model_download_progress[task_id]
                
    except Exception as e:
        logger.error(f"Error downloading zero-shot model {model_name}: {str(e)}")
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

@zero_shot_bp.route('/settings/update_zero_shot_settings', methods=['POST'])
def update_zero_shot_settings():
    """
    Endpoint to update zero-shot tagging settings.
    """
    try:
        data = request.get_json()
        config = UserConfig.query.first()
        if not config:
            config = UserConfig(zero_shot_enabled=True, zero_shot_model="base", zero_shot_min_confidence=0.3)
            db.session.add(config)
        
        updated = False
        
        # Update zero-shot enabled if provided
        if "zero_shot_enabled" in data:
            zero_shot_enabled = data.get("zero_shot_enabled")
            if isinstance(zero_shot_enabled, bool):
                config.zero_shot_enabled = zero_shot_enabled
                updated = True
        
        # Update zero-shot model if provided
        if "zero_shot_model" in data:
            zero_shot_model = data.get("zero_shot_model")
            from utils.zero_shot_tagger import MODELS
            if zero_shot_model in MODELS:
                config.zero_shot_model = zero_shot_model
                updated = True
            else:
                return jsonify({
                    "status": "error",
                    "message": f"Invalid zero-shot model: {zero_shot_model}"
                }), 400
        
        # Update zero-shot min confidence if provided
        if "zero_shot_min_confidence" in data:
            zero_shot_min_confidence = data.get("zero_shot_min_confidence")
            if isinstance(zero_shot_min_confidence, (int, float)) and 0.0 <= zero_shot_min_confidence <= 1.0:
                config.zero_shot_min_confidence = zero_shot_min_confidence
                updated = True
            else:
                return jsonify({
                    "status": "error",
                    "message": "Invalid minimum confidence value. Must be a number between 0 and 1."
                }), 400
        
        if updated:
            db.session.commit()
            return jsonify({
                "status": "success",
                "message": "Zero-shot tagging settings updated successfully."
            })
        else:
            return jsonify({
                "status": "error",
                "message": "No valid settings provided."
            }), 400
        
    except Exception as e:
        logger.error(f"Error in update_zero_shot_settings: {str(e)}")
        return jsonify({
            "status": "error",
            "message": f"Error: {str(e)}"
        }), 500

@zero_shot_bp.route('/settings/download_zero_shot_model', methods=['POST'])
def download_zero_shot_model():
    """
    Endpoint to download a zero-shot model from Hugging Face.
    """
    try:
        data = request.get_json()
        model_name = data.get("model_name")
        
        if not model_name:
            return jsonify({
                "status": "error",
                "message": "Model name is required"
            }), 400
        
        # Validate zero-shot model
        from utils.zero_shot_tagger import MODELS
        if model_name not in MODELS:
            return jsonify({
                "status": "error",
                "message": f"Invalid zero-shot model: {model_name}"
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
        app = current_app._get_current_object()
        
        # Start download in a separate thread
        thread = threading.Thread(
            target=download_zero_shot_model_thread,
            args=(model_name, task_id, app)
        )
        thread.daemon = True
        thread.start()
        
        return jsonify({
            "status": "success",
            "message": f"Started downloading zero-shot model {model_name}",
            "task_id": task_id
        })
        
    except Exception as e:
        logger.error(f"Error in download_zero_shot_model: {str(e)}")
        return jsonify({
            "status": "error",
            "message": f"Error: {str(e)}"
        }), 500

@zero_shot_bp.route('/settings/zero_shot_download_progress/<task_id>', methods=['GET'])
def zero_shot_download_progress(task_id):
    """
    Endpoint to check the progress of a zero-shot model download.
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

@zero_shot_bp.route('/settings/list_zero_shot_models', methods=['GET'])
def list_zero_shot_models():
    """
    Endpoint to list all available zero-shot models.
    """
    try:
        from utils.zero_shot_tagger import list_available_models
        
        # Get all available models
        available_models = list_available_models()
        
        return jsonify({
            "status": "success",
            "models": available_models
        })
        
    except Exception as e:
        logger.error(f"Error in list_zero_shot_models: {str(e)}")
        return jsonify({
            "status": "error",
            "message": f"Error: {str(e)}"
        }), 500

@zero_shot_bp.route('/settings/test_zero_shot_tagging', methods=['POST'])
def test_zero_shot_tagging():
    """
    Endpoint to test zero-shot image tagging.
    """
    try:
        from utils.zero_shot_tagger import generate_tags_with_zero_shot
        from models import UserConfig
        import os
        import tempfile
        
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
            
        # Get the current zero-shot model and settings
        config = UserConfig.query.first()
        if not config:
            return jsonify({
                "status": "error",
                "message": "No configuration found"
            }), 404
            
        model_size = config.zero_shot_model if hasattr(config, 'zero_shot_model') and config.zero_shot_model else "base"
        max_tags = config.min_tags if hasattr(config, 'min_tags') and config.min_tags is not None else 5
        min_confidence = config.zero_shot_min_confidence if hasattr(config, 'zero_shot_min_confidence') and config.zero_shot_min_confidence is not None else 0.3
        
        # Save the uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp:
            file.save(temp.name)
            temp_path = temp.name
        
        try:
            # Generate tags using zero-shot
            tags, description = generate_tags_with_zero_shot(
                temp_path,
                model_size=model_size,
                max_tags=max_tags,
                min_confidence=min_confidence
            )
            
            # Log the test results
            logger.info(f"Test zero-shot tagging completed with model {model_size}: generated {len(tags)} tags")
            
            # Clean up the temporary file
            os.unlink(temp_path)
            
            return jsonify({
                "status": "success",
                "model_used": f"Zero-Shot: {model_size}",
                "tags": tags,
                "description": description,
                "min_confidence": min_confidence,
                "max_tags": max_tags
            })
            
        except Exception as e:
            # Clean up the temporary file in case of error
            os.unlink(temp_path)
            raise e
            
    except Exception as e:
        logger.error(f"Error in test_zero_shot_tagging: {str(e)}")
        return jsonify({
            "status": "error",
            "message": f"Error: {str(e)}"
        }), 500