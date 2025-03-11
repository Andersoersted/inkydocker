# Apply PyTorch monkey patches before any other imports
import sys
import os

# Increase recursion limit globally
sys.setrecursionlimit(15000)

# Fix for PyTorch distributed module compatibility issues
import torch
if hasattr(torch.distributed, 'reduce_op'):
    # Create a proper ReduceOp class with RedOpType attribute
    original_reduce_op = torch.distributed.reduce_op
    
    class FixedReduceOp:
        def __init__(self):
            self.SUM = original_reduce_op.SUM
            self.PRODUCT = original_reduce_op.PRODUCT
            self.MIN = original_reduce_op.MIN
            self.MAX = original_reduce_op.MAX
            self.BAND = original_reduce_op.BAND
            self.BOR = original_reduce_op.BOR
            self.BXOR = original_reduce_op.BXOR
            
            # Add RedOpType as a class attribute that references self
            class RedOpType:
                SUM = self.SUM
                PRODUCT = self.PRODUCT
                MIN = self.MIN
                MAX = self.MAX
                BAND = self.BAND
                BOR = self.BOR
                BXOR = self.BXOR
            
            self.RedOpType = RedOpType
    
    # Replace the original reduce_op with our fixed version
    torch.distributed.ReduceOp = FixedReduceOp()
    torch.distributed.reduce_op = torch.distributed.ReduceOp

from PIL import Image
from flask import current_app
from huggingface_hub import hf_hub_download
from transformers import AutoProcessor, AutoModelForVision2Seq

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
ram_models = {}
ram_processors = {}

def get_ram_model(model_name=DEFAULT_MODEL):
    """
    Load and return the RAM++ model.
    
    Args:
        model_name: Name of the RAM++ model to load (defaults to ram_large)
        
    Returns:
        tuple: (model, processor) for the RAM++ model
    """
    # Check if model is already loaded
    if model_name in ram_models:
        return ram_models[model_name], ram_processors[model_name]
    
    # Get model info
    if model_name not in RAM_MODELS:
        current_app.logger.warning(f"Unknown RAM++ model: {model_name}, using default model")
        model_name = DEFAULT_MODEL
    
    model_info = RAM_MODELS[model_name]
    
    try:
        current_app.logger.info(f"Loading RAM++ model: {model_name}")
        
        # Determine device (use CUDA if available)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        current_app.logger.info(f"Using device: {device} for RAM++ model")
        
        # Load model and processor with increased recursion limit
        import sys
        old_recursion_limit = sys.getrecursionlimit()
        try:
            # Use a fixed high recursion limit for all models
            recursion_limit = 15000
            
            # Increase recursion limit temporarily to handle deep call stacks
            sys.setrecursionlimit(recursion_limit)
            current_app.logger.info(f"Temporarily increased recursion limit to {recursion_limit} for loading RAM++ model {model_name}")
            
            # Try to import the recognize-anything package
            try:
                from recognize_anything import ram
                
                # Load the model using the recognize-anything package
                model_type = model_info["model_type"]
                current_app.logger.info(f"Loading RAM++ model type: {model_type}")
                
                try:
                    # Try to create the RAM++ model
                    pretrained = model_info.get("pretrained", "ram_plus_swin_large_14m")
                    ram_model = ram.ram(pretrained=pretrained, model_type=model_type)
                    
                    # Extract the model and processor
                    model = ram_model.model
                    processor = ram_model.processor
                    
                    current_app.logger.info(f"Successfully loaded RAM++ model using recognize-anything package")
                except Exception as e:
                    current_app.logger.warning(f"Error loading RAM++ model using recognize-anything package: {e}")
                    current_app.logger.warning("Falling back to direct loading from Hugging Face")
                    # Fall back to direct loading if the RAM++ model is not available
                    processor = AutoProcessor.from_pretrained("xinyu1205/recognize_anything_plus_model")
                    model = AutoModelForVision2Seq.from_pretrained(
                        "xinyu1205/recognize_anything_plus_model",
                        torch_dtype=torch.float16 if device == "cuda" else torch.float32
                    )
            
            except ImportError:
                current_app.logger.warning("recognize-anything package not found, falling back to direct loading")
                # Fall back to direct loading if the package is not available
                try:
                    # Try to load from local cache first
                    cache_dir = "/app/data/ram_models"
                    if os.path.exists(cache_dir) and len(os.listdir(cache_dir)) > 0:
                        current_app.logger.info(f"Loading RAM++ model from local cache: {cache_dir}")
                        
                        # Check if we have the .pth file
                        pth_file = os.path.join(cache_dir, "ram_plus_swin_large_14m.pth")
                        if os.path.exists(pth_file):
                            current_app.logger.info(f"Found RAM++ .pth file: {pth_file}")
                            # Use the recognize-anything package to load the .pth file
                            from recognize_anything import ram
                            ram_model = ram.ram(pretrained=pth_file, model_type=model_type)
                            model = ram_model.model
                            processor = ram_model.processor
                        else:
                            # Try to load using the transformers API
                            processor = AutoProcessor.from_pretrained(cache_dir)
                            model = AutoModelForVision2Seq.from_pretrained(
                                cache_dir,
                                torch_dtype=torch.float16 if device == "cuda" else torch.float32
                            )
                    else:
                        # If local cache is not available, try to download from Hugging Face
                        current_app.logger.info("Local cache not available, downloading from Hugging Face")
                        processor = AutoProcessor.from_pretrained("xinyu1205/recognize_anything_plus_model")
                        model = AutoModelForVision2Seq.from_pretrained(
                            "xinyu1205/recognize_anything_plus_model",
                            torch_dtype=torch.float16 if device == "cuda" else torch.float32
                        )
                except Exception as e:
                    current_app.logger.error(f"Error loading RAM++ model: {e}")
                    # Fall back to CLIP model if RAM++ is not available
                    current_app.logger.warning("Falling back to CLIP model for tagging")
                    # Return None to signal that we should use CLIP instead
                    return None, None
                
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
        current_app.logger.error(f"Error loading RAM++ model {model_name}: {e}")
        raise

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
        # Try to use the recognize-anything package directly if available
        try:
            from recognize_anything import ram
            
            # Check if we should use the cached model
            if model_name in ram_models:
                current_app.logger.info(f"Using cached RAM++ model: {model_name}")
                # Use the cached model and processor
                model, processor = get_ram_model(model_name)
                
                # If model is None, fall back to CLIP
                if model is None:
                    current_app.logger.info("RAM model not available, falling back to CLIP")
                    # Use CLIP model from tasks.py
                    from tasks import get_image_embedding, generate_tags_and_description
                    embedding, model_used = get_image_embedding(image_path)
                    if embedding is None:
                        return [], "Error generating tags"
                    return generate_tags_and_description(embedding, model_used)
                
                # Load image
                image = Image.open(image_path).convert("RGB")
                
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
                
            else:
                # Create a new RAM++ model instance
                model_info = RAM_MODELS[model_name]
                model_type = model_info["model_type"]
                pretrained = model_info.get("pretrained", "ram_plus_swin_large_14m")
                
                try:
                    # Check if we have the .pth file
                    cache_dir = "/app/data/ram_models"
                    pth_file = os.path.join(cache_dir, "ram_plus_swin_large_14m.pth")
                    if os.path.exists(pth_file):
                        current_app.logger.info(f"Using RAM++ .pth file: {pth_file}")
                        # Create the RAM++ model with the .pth file
                        ram_model = ram.ram(pretrained=pth_file, model_type=model_type)
                    else:
                        # Create the RAM++ model with the default pretrained model
                        ram_model = ram.ram(pretrained=pretrained, model_type=model_type)
                    
                    # Generate tags directly using the RAM++ model
                    generated_text = ram_model.generate_caption(image_path)
                    
                    # Cache the model and processor for future use
                    ram_models[model_name] = ram_model.model
                    ram_processors[model_name] = ram_model.processor
                    
                    current_app.logger.info(f"Successfully generated tags using RAM++ model")
                except Exception as e:
                    current_app.logger.warning(f"Error generating tags using RAM++ model: {e}")
                    current_app.logger.warning("Falling back to direct loading from Hugging Face")
                    
                    # Fall back to using the get_ram_model function
                    model, processor = get_ram_model(model_name)
                    
                    # If model is None, fall back to CLIP
                    if model is None:
                        current_app.logger.info("RAM model not available, falling back to CLIP")
                        # Use CLIP model from tasks.py
                        from tasks import get_image_embedding, generate_tags_and_description
                        embedding, model_used = get_image_embedding(image_path)
                        if embedding is None:
                            return [], "Error generating tags"
                        return generate_tags_and_description(embedding, model_used)
                    
                    # Load image
                    image = Image.open(image_path).convert("RGB")
                    
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
            
        except ImportError:
            # Fall back to using our custom implementation
            current_app.logger.info("recognize-anything package not found, using custom implementation")
            
            # Get model and processor
            model, processor = get_ram_model(model_name)
            
            # If model is None, fall back to CLIP
            if model is None:
                current_app.logger.info("RAM model not available, falling back to CLIP")
                # Use CLIP model from tasks.py
                from tasks import get_image_embedding, generate_tags_and_description
                embedding, model_used = get_image_embedding(image_path)
                if embedding is None:
                    return [], "Error generating tags"
                return generate_tags_and_description(embedding, model_used)
            
            # Load image
            image = Image.open(image_path).convert("RGB")
            
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