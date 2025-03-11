import os
import sys
import warnings
import torch
from PIL import Image
from flask import current_app

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Increase recursion limit for model loading
sys.setrecursionlimit(15000)

# Define RAM++ model configuration
RAM_MODELS = {
    "ram_large": {
        "model_type": "ram_swin_large_patch4_window7_224",
        "pretrained": "ram_plus_swin_large_14m",
        "description": "RAM++ Large model (most accurate)"
    }
}

# Default model name
DEFAULT_MODEL = "ram_large"

# Cache for loaded models
ram_model_cache = None

def get_ram_model(model_name=DEFAULT_MODEL):
    """
    Load and return the RAM++ model using the recommended approach from the recognize-anything repository.
    
    Args:
        model_name: Name of the RAM++ model to load (defaults to ram_large)
        
    Returns:
        The RAM model object
    """
    global ram_model_cache
    
    # Return cached model if available
    if ram_model_cache is not None:
        return ram_model_cache
    
    # Get model info
    if model_name not in RAM_MODELS:
        current_app.logger.warning(f"Unknown RAM++ model: {model_name}, using default model")
        model_name = DEFAULT_MODEL
    
    model_info = RAM_MODELS[model_name]
    
    try:
        current_app.logger.info(f"Loading RAM++ model: {model_name}")
        
        # Import the RAM model using the recommended approach
        try:
            from ram.models import ram_plus
            
            # Determine device (use CUDA if available)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            current_app.logger.info(f"Using device: {device} for RAM++ model")
            
            # Check if we have a local model file
            cache_dir = "/app/data/ram_models"
            pth_file = os.path.join(cache_dir, "ram_plus_swin_large_14m.pth")
            
            if os.path.exists(pth_file):
                current_app.logger.info(f"Using local model file: {pth_file}")
                pretrained = pth_file
            else:
                # Use the default pretrained model
                pretrained = model_info.get("pretrained", "ram_plus_swin_large_14m")
                current_app.logger.info(f"Using pretrained model: {pretrained}")
            
            # Create the RAM model
            ram_model = ram_plus(pretrained=pretrained)
            current_app.logger.info(f"Successfully loaded RAM++ model")
            
            # Cache the model for future use
            ram_model_cache = ram_model
            
            return ram_model
            
        except ImportError as e:
            current_app.logger.error(f"ram.models module not found: {e}")
            current_app.logger.info("Please install the recognize-anything package: pip install git+https://github.com/xinyu1205/recognize-anything.git")
            return None
            
    except Exception as e:
        current_app.logger.error(f"Error loading RAM++ model {model_name}: {e}")
        return None

def generate_tags_with_ram(image_path, model_name=DEFAULT_MODEL, max_tags=10, min_confidence=0.3):
    """
    Generate tags for an image using the RAM++ model.
    
    Args:
        image_path: Path to the image file
        model_name: Name of the RAM++ model to use (defaults to ram_large)
        max_tags: Maximum number of tags to return
        min_confidence: Minimum confidence threshold for tags
        
    Returns:
        tuple: (tags, description) where tags is a list of tag strings and
               description is a string description of the image
    """
    try:
        # Get the RAM model
        ram_model = get_ram_model(model_name)
        
        # If model is None, fall back to CLIP
        if ram_model is None:
            current_app.logger.info("RAM model not available, falling back to CLIP")
            from tasks import get_image_embedding, generate_tags_and_description
            embedding, model_used = get_image_embedding(image_path)
            if embedding is None:
                return [], "Error generating tags"
            return generate_tags_and_description(embedding, model_used)
        
        # Generate tags directly using the RAM model
        current_app.logger.info(f"Generating tags for image: {image_path}")
        
        # Use the inference method as shown in the GitHub example
        generated_text = ram_model.generate_tag(image_path)
        current_app.logger.info(f"Generated text: {generated_text}")
        
        # Parse tags (RAM typically returns a comma-separated list)
        all_tags = [tag.strip() for tag in generated_text.split("|")]
        
        # Filter and limit tags
        filtered_tags = all_tags[:max_tags]
        
        # Create description
        description = "This image contains " + ", ".join(filtered_tags) + "."
        
        current_app.logger.info(f"Generated {len(filtered_tags)} tags with RAM++ model {model_name}")
        
        return filtered_tags, description
        
    except Exception as e:
        current_app.logger.error(f"Error generating tags with RAM++: {e}")
        # Fall back to CLIP as a last resort
        try:
            current_app.logger.info("Falling back to CLIP after RAM++ error")
            from tasks import get_image_embedding, generate_tags_and_description
            embedding, model_used = get_image_embedding(image_path)
            if embedding is None:
                return [], "Error generating tags"
            return generate_tags_and_description(embedding, model_used)
        except Exception as clip_error:
            current_app.logger.error(f"CLIP fallback also failed: {clip_error}")
            return [], "Error generating tags"

def list_available_ram_models():
    """
    Return a list of available RAM++ models with their descriptions.
    
    Returns:
        list: List of dictionaries with model information
    """
    return [
        {"name": name, "description": info["description"]}
        for name, info in RAM_MODELS.items()
    ]