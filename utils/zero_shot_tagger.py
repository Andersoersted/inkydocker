import os
import sys
import warnings
import torch
from PIL import Image
from flask import current_app

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Set recursion limit for model loading
sys.setrecursionlimit(20000)
print(f"Recursion limit set to {sys.getrecursionlimit()}")

# Define model configurations - updated to use available pretrained models
MODELS = {
    "small": {
        "model_name": "ViT-B-32",
        "pretrained": "laion400m_e32",
        "description": "Small model (fastest)"
    },
    "medium": {
        "model_name": "ViT-L-14",
        "pretrained": "openai",
        "description": "Medium model (balanced)"
    },
    "large": {
        "model_name": "ViT-H-14",
        "pretrained": "laion2b_s32b_b79k",
        "description": "Large model (most accurate)"
    }
}

# Default model name
DEFAULT_MODEL = "small"

# Cache for loaded models
clip_models = {}

def get_clip_model(model_size=DEFAULT_MODEL):
    """
    Load and return the OpenCLIP model.
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
        
        import open_clip
        
        # Determine device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load the model
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained,
            device=device
        )
        
        current_app.logger.info(f"Successfully loaded OpenCLIP model on {device}")
        
        # Cache the model
        clip_models[model_size] = (model, preprocess)
        
        return model, preprocess
            
    except Exception as e:
        current_app.logger.error(f"Error loading OpenCLIP model {model_name}: {e}")
        if model_size != DEFAULT_MODEL:
            current_app.logger.warning(f"Attempting to load fallback model: {DEFAULT_MODEL}")
            return get_clip_model(DEFAULT_MODEL)
        return None, None

def generate_tags_with_zero_shot(image_path, model_size=DEFAULT_MODEL, max_tags=10, min_confidence=0.3, candidate_labels=None):
    """
    Generate tags for an image using OpenCLIP zero-shot classification.
    Returns both tags and tags with confidence scores.
    """
    try:
        # Default candidate labels if none provided
        if candidate_labels is None:
            candidate_labels = [
                "person", "people", "animal", "vehicle", "building", "furniture",
                "electronics", "food", "plant", "landscape", "indoor", "outdoor",
                "day", "night", "water", "sky", "mountain", "beach", "city", "rural",
                "portrait", "group photo", "selfie", "action", "event", "document",
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
        image = Image.open(image_path).convert("RGB")
        
        # Determine the device to use
        device = next(model.parameters()).device
        
        # Preprocess the image and ensure it's on the right device
        image_input = preprocess(image).unsqueeze(0).to(device)
        
        # Generate text features for all candidate labels
        import open_clip
        text_tokens = open_clip.tokenize(candidate_labels).to(device)
        
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
            
        # Create a list of tags with their confidence scores
        tags_with_scores = []
        for value, idx in zip(values.cpu().numpy(), indices.cpu().numpy()):
            tag = candidate_labels[idx]
            confidence = float(value)
            tags_with_scores.append({"tag": tag, "confidence": confidence})
        
        # Sort by confidence (highest first)
        tags_with_scores.sort(key=lambda x: x["confidence"], reverse=True)
        
        # Filter tags by confidence threshold for the regular tags list
        tags = []
        for item in tags_with_scores:
            if item["confidence"] >= min_confidence and len(tags) < max_tags:
                if item["tag"] not in tags:
                    tags.append(item["tag"])
        
        # Create description
        description = "This image contains " + ", ".join(tags) + "."
        
        current_app.logger.info(f"Generated {len(tags)} tags with OpenCLIP model {model_size}")
        
        return tags, description, tags_with_scores
        
    except Exception as e:
        current_app.logger.error(f"Error generating tags with OpenCLIP model: {e}")
        return [], f"Error generating tags: {str(e)}", []

def list_available_models():
    """
    Return a list of available models with their descriptions.
    """
    return [
        {"name": name, "description": info["description"]}
        for name, info in MODELS.items()
    ]