from flask import Blueprint, request, jsonify, current_app, url_for
import logging
import os
import json
from datetime import datetime
from models import db, UserConfig

# Create a blueprint for zero-shot settings routes
zero_shot_bp = Blueprint('zero_shot', __name__)
logger = logging.getLogger(__name__)

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

# Note: Model download routes have been removed as all models are now preloaded with the Docker image

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
    Shows all possible tags without filtering by confidence to help users set their confidence level.
    """
    try:
        from utils.zero_shot_tagger import generate_tags_with_zero_shot, MODELS
        from models import UserConfig
        import os
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
            
        # Get the current zero-shot model and settings
        config = UserConfig.query.first()
        if not config:
            return jsonify({
                "status": "error",
                "message": "No configuration found"
            }), 404
        
        # Check if a model override was provided in the form data
        model_size = request.form.get("model_override")
        if not model_size:
            model_size = config.zero_shot_model if hasattr(config, 'zero_shot_model') and config.zero_shot_model else "small"
        
        # Validate the model size
        if model_size not in MODELS:
            logger.warning(f"Invalid model size: {model_size}, using default")
            model_size = "small"
            
        # Get max tags from config - but use a higher value for testing to show more options
        max_tags = 20  # Show more tags in test mode
        
        # Save the uploaded file to a temporary location in static/uploads
        upload_folder = os.path.join(current_app.root_path, "static", "uploads")
        if not os.path.exists(upload_folder):
            os.makedirs(upload_folder)
        filename = f"test_{int(time.time())}_{file.filename}"
        file_path = os.path.join(upload_folder, filename)
        file.save(file_path)
        
        # For testing, use a very low confidence to show all possible tags
        min_confidence = 0.01  # Very low confidence to show all tags
        
        # Generate tags using zero-shot with low confidence to get all possible tags
        all_tags, description, tags_with_scores = generate_tags_with_zero_shot(
            file_path,
            model_size=model_size,
            max_tags=max_tags,
            min_confidence=min_confidence
        )
        
        # Get the actual confidence threshold from config for reference
        actual_confidence = config.zero_shot_min_confidence if hasattr(config, 'zero_shot_min_confidence') and config.zero_shot_min_confidence is not None else 0.3
        actual_max_tags = config.min_tags if hasattr(config, 'min_tags') and config.min_tags is not None else 5
        
        # Log the test results
        logger.info(f"Test zero-shot tagging completed with model {model_size}: generated {len(all_tags)} tags with confidence scores")
        
        # Build the URL for the uploaded image
        try:
            image_url = url_for('static', filename=f"uploads/{filename}", _external=True)
        except Exception as url_error:
            logger.warning(f"Error generating URL for image: {str(url_error)}")
            # Fallback to a relative path if url_for fails
            image_url = f"/static/uploads/{filename}"
        
        return jsonify({
            "status": "success",
            "model_used": f"Zero-Shot: {model_size}",
            "tags": all_tags,
            "tags_with_scores": tags_with_scores,
            "description": description,
            "min_confidence": actual_confidence,  # Return the actual confidence for reference
            "max_tags": actual_max_tags,  # Return the actual max tags for reference
            "uploaded_image": image_url,
            "test_mode": True  # Indicate this is test mode with all tags shown
        })
            
    except Exception as e:
        logger.error(f"Error in test_zero_shot_tagging: {str(e)}")
        return jsonify({
            "status": "error",
            "message": f"Error: {str(e)}"
        }), 500