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

# Define model configurations
MODELS = {
    "base": {
        "model_name": "ViT-B-32",
        "pretrained": "openai",
        "description": "Base model (faster)"
    },
    "large": {
        "model_name": "ViT-L-14",
        "pretrained": "openai",
        "description": "Large model (more accurate)"
    }
}

# Default model name
DEFAULT_MODEL = "base"

# Cache for loaded models
clip_models = {}

def get_clip_model(model_size=DEFAULT_MODEL):
    """
    Load and return the OpenCLIP model.
    
    Args:
        model_size: Size of the model to load (base or large)
        
    Returns:
        Tuple of (model, preprocess)
    """
    global clip_models
    
    # Return cached model if available
    if model_size in clip_models:
        return clip_models[model_size]
    
    # Get model info
    if model_size not in MODELS:
        current_app.logger.warning(f"Unknown model size: {model_size}, using default model")
        model_size = DEFAULT_MODEL
    
    model_info = MODELS[model_size]
    model_name = model_info["model_name"]
    pretrained = model_info["pretrained"]
    
    try:
        current_app.logger.info(f"Loading OpenCLIP model: {model_name} (pretrained={pretrained})")
        
        # Import the necessary modules
        try:
            import open_clip
            
            # Determine device (use CUDA if available)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            current_app.logger.info(f"Using device: {device} for CLIP model")
            
            # Load the model
            model, _, preprocess = open_clip.create_model_and_transforms(
                model_name,
                pretrained=pretrained,
                device=device
            )
            
            current_app.logger.info(f"Successfully loaded OpenCLIP model")
            
            # Cache the model
            clip_models[model_size] = (model, preprocess)
            
            return model, preprocess
            
        except ImportError as e:
            current_app.logger.error(f"open_clip module not found: {e}")
            return None, None
            
    except Exception as e:
        current_app.logger.error(f"Error loading OpenCLIP model {model_name}: {e}")
        return None, None

def generate_tags_with_zero_shot(image_path, model_size=DEFAULT_MODEL, max_tags=10, min_confidence=0.3, candidate_labels=None):
    """
    Generate tags for an image using OpenCLIP zero-shot classification.
    
    Args:
        image_path: Path to the image file
        model_size: Size of the model to use (base or large)
        max_tags: Maximum number of tags to return
        min_confidence: Minimum confidence threshold for tags
        candidate_labels: Optional list of candidate labels for classification
        
    Returns:
        tuple: (tags, description) where tags is a list of tag strings and
               description is a string description of the image
    """
    try:
        # Default candidate labels if none provided
        if candidate_labels is None:
            candidate_labels = [
                "person", "people", "animal", "vehicle", "building", "furniture", 
                "electronics", "food", "plant", "landscape", "indoor", "outdoor",
                "day", "night", "water", "sky", "mountain", "beach", "city", "rural",
                "portrait", "group photo", "selfie", "action", "event", "document",
                # Add more specific object categories
                "man", "woman", "child", "baby", "dog", "cat", "bird", "car", "truck", 
                "bicycle", "motorcycle", "boat", "airplane", "train", "bus", "chair", "table", 
                "sofa", "bed", "tv", "laptop", "phone", "book", "bottle", "cup", "plate", "bowl",
                "fruit", "vegetable", "tree", "flower", "plant", "mountain", "building", "house",
                "road", "sidewalk", "river", "lake", "ocean", "beach", "cloud", "sun", "moon"
            ]
        
        # Get the CLIP model
        model, preprocess = get_clip_model(model_size)
        
        # If model is None, return empty results
        if model is None or preprocess is None:
            current_app.logger.error("CLIP model not available")
            return [], "Error generating tags"
        
        # Open the image
        image = Image.open(image_path)
        
        # Preprocess the image
        image_input = preprocess(image).unsqueeze(0).to(model.device)
        
        # Generate text features for all candidate labels
        import open_clip
        text_tokens = open_clip.tokenize(candidate_labels).to(model.device)
        
        # Get image and text features
        with torch.no_grad():
            image_features = model.encode_image(image_input)
            text_features = model.encode_text(text_tokens)
            
            # Normalize features
            image_features = image_features / image_features.norm(dim=1, keepdim=True)
            text_features = text_features / text_features.norm(dim=1, keepdim=True)
            
            # Calculate similarity scores
            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            
            # Get the top tags
            values, indices = torch.topk(similarity[0], k=min(len(candidate_labels), max_tags * 2))
            
        # Filter tags by confidence threshold
        tags = []
        for value, idx in zip(values, indices):
            if value >= min_confidence and len(tags) < max_tags:
                tag = candidate_labels[idx]
                if tag not in tags:
                    tags.append(tag)
        
        # Create description
        description = "This image contains " + ", ".join(tags) + "."
        
        current_app.logger.info(f"Generated {len(tags)} tags with OpenCLIP model {model_size}")
        
        return tags, description
        
    except Exception as e:
        current_app.logger.error(f"Error generating tags with OpenCLIP model: {e}")
        return [], "Error generating tags"

def list_available_models():
    """
    Return a list of available models with their descriptions.
    
    Returns:
        list: List of dictionaries with model information
    """
    return [
        {"name": name, "description": info["description"]}
        for name, info in MODELS.items()
    ]