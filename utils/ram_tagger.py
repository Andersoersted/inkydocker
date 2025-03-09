import os
import torch
from PIL import Image
from flask import current_app
from huggingface_hub import hf_hub_download
from transformers import AutoProcessor, AutoModelForVision2Seq

# Define available RAM models with their Hugging Face repo IDs
RAM_MODELS = {
    "ram_smallest": {
        "repo_id": "xinyu1205/recognize_anything_model",
        "filename": "ram_tiny_model.pth",  # Smallest model available
        "description": "Smallest RAM model (very fast, basic tagging)",
        "recursion_limit": 3000  # Lower recursion limit for this model
    },
    "ram_small": {
        "repo_id": "xinyu1205/recognize_anything_model",
        "filename": "ram_small_model.pth",
        "description": "Small RAM model (fast, good for basic tagging)",
        "recursion_limit": 5000
    },
    "ram_medium": {
        "repo_id": "xinyu1205/recognize_anything_model",
        "filename": "ram_swin_large_14m.pth",
        "description": "Medium RAM model (balanced performance, recommended)",
        "recursion_limit": 10000
    },
    "ram_large": {
        "repo_id": "xinyu1205/recognize_anything_model",
        "filename": "ram_swin_large_14m.pth",  # Same weights as medium but with higher recursion limit
        "description": "Large RAM model (most accurate, higher memory usage)",
        "recursion_limit": 15000  # Higher recursion limit for this model
    }
}

# Cache for loaded models
ram_models = {}
ram_processors = {}

def get_ram_model(model_name="ram_medium"):
    """
    Load and return the specified RAM model.
    
    Args:
        model_name: Name of the RAM model to load (ram_small, ram_medium, ram_large)
        
    Returns:
        tuple: (model, processor) for the specified RAM model
    """
    # Check if model is already loaded
    if model_name in ram_models:
        return ram_models[model_name], ram_processors[model_name]
    
    # Get model info
    if model_name not in RAM_MODELS:
        current_app.logger.warning(f"Unknown RAM model: {model_name}, falling back to ram_medium")
        model_name = "ram_medium"
    
    model_info = RAM_MODELS[model_name]
    
    try:
        current_app.logger.info(f"Loading RAM model: {model_name}")
        
        # Set up cache directory
        data_folder = current_app.config.get("DATA_FOLDER", "./data")
        models_folder = os.path.join(data_folder, "ram_models")
        os.makedirs(models_folder, exist_ok=True)
        
        # Download model if not already cached
        model_path = os.path.join(models_folder, model_info["filename"])
        if not os.path.exists(model_path):
            current_app.logger.info(f"Downloading RAM model {model_name} from Hugging Face")
            hf_hub_download(
                repo_id=model_info["repo_id"],
                filename=model_info["filename"],
                cache_dir=models_folder,
                force_download=False
            )
        
        # Determine device (use CUDA if available)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        current_app.logger.info(f"Using device: {device} for RAM model")
        
        # Load model and processor with increased recursion limit
        import sys
        old_recursion_limit = sys.getrecursionlimit()
        try:
            # Get model-specific recursion limit or use default
            recursion_limit = model_info.get("recursion_limit", 5000)
            
            # Increase recursion limit temporarily to handle deep call stacks
            sys.setrecursionlimit(recursion_limit)
            current_app.logger.info(f"Temporarily increased recursion limit to {recursion_limit} for loading RAM model {model_name}")
            
            processor = AutoProcessor.from_pretrained(model_info["repo_id"])
            model = AutoModelForVision2Seq.from_pretrained(
                model_info["repo_id"],
                torch_dtype=torch.float16 if device == "cuda" else torch.float32
            )
        finally:
            # Restore original recursion limit
            sys.setrecursionlimit(old_recursion_limit)
            current_app.logger.info(f"Restored recursion limit to {old_recursion_limit}")
        
        # Move model to device
        model.to(device)
        model.eval()
        
        # Cache model and processor
        ram_models[model_name] = model
        ram_processors[model_name] = processor
        
        return model, processor
    
    except Exception as e:
        current_app.logger.error(f"Error loading RAM model {model_name}: {e}")
        raise

def generate_tags_with_ram(image_path, model_name="ram_medium", max_tags=10, min_confidence=0.3):
    """
    Generate tags for an image using the RAM model.
    
    Args:
        image_path: Path to the image file
        model_name: Name of the RAM model to use
        max_tags: Maximum number of tags to return
        min_confidence: Minimum confidence threshold for tags
        
    Returns:
        tuple: (tags, description) where tags is a list of tag strings and 
               description is a string description of the image
    """
    try:
        # Load image
        image = Image.open(image_path).convert("RGB")
        
        # Get model and processor
        model, processor = get_ram_model(model_name)
        
        # Determine device
        device = next(model.parameters()).device
        
        # Process image
        inputs = processor(images=image, return_tensors="pt").to(device)
        
        # Generate tags
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=128,
                num_beams=5,
                early_stopping=True
            )
        
        # Decode tags
        generated_text = processor.batch_decode(outputs, skip_special_tokens=True)[0]
        
        # Parse tags (RAM typically returns a comma-separated list)
        all_tags = [tag.strip() for tag in generated_text.split(",")]
        
        # Filter and limit tags
        filtered_tags = all_tags[:max_tags]
        
        # Create description
        description = "This image contains " + ", ".join(filtered_tags) + "."
        
        current_app.logger.info(f"Generated {len(filtered_tags)} tags with RAM model {model_name}")
        
        return filtered_tags, description
    
    except Exception as e:
        current_app.logger.error(f"Error generating tags with RAM: {e}")
        return [], "Error generating tags"

def list_available_ram_models():
    """
    Return a list of available RAM models with their descriptions.
    
    Returns:
        list: List of dictionaries with model information
    """
    return [
        {"name": name, "description": info["description"]}
        for name, info in RAM_MODELS.items()
    ]