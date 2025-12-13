"""
AI/ML Model management for InkyDocker.
Handles OpenCLIP model loading, caching, embedding generation, and tag generation.
"""
import os
import gc
import sys
import torch
import psutil
import open_clip
from PIL import Image
from flask import current_app
from models import UserConfig, db
from utils.constants import CANDIDATE_TAGS, SIMILARITY_THRESHOLDS, DEFAULT_THRESHOLD


# Device setup with environment variable override for large models
device = "cuda" if torch.cuda.is_available() else "cpu"

# Check if we should force CPU for large models based on system memory
FORCE_CPU_FOR_LARGE_MODELS = os.environ.get('FORCE_CPU_FOR_LARGE_MODELS', '0') == '1'
# Handle empty string case for SYSTEM_MEMORY_GB
system_memory_str = os.environ.get('SYSTEM_MEMORY_GB', '0')
SYSTEM_MEMORY_GB = float(system_memory_str) if system_memory_str else 0.0

if FORCE_CPU_FOR_LARGE_MODELS:
    print(f"System configured to force CPU for large models due to limited memory ({SYSTEM_MEMORY_GB}GB)")

# Model and embeddings cache
clip_models = {}
clip_preprocessors = {}
tag_embeddings = {}


def get_better_prompt_for_tag(tag):
    """
    Generate a more specific prompt for a tag to improve object detection.
    Different types of tags need different prompt structures for best results.
    """
    # People and portrait prompts
    if tag in ["person", "people", "man", "woman", "child", "children", "baby", "group of people",
               "crowd", "portrait", "selfie", "face"]:
        return f"a photograph of {tag} in the image"

    # Object prompts
    elif tag in ["car", "vehicle", "bicycle", "motorcycle", "airplane", "boat", "building", "house",
                "furniture", "chair", "table", "bed", "computer", "phone", "television", "book",
                "clock", "bottle", "cup", "plate", "food", "fruit", "vegetable", "meal"]:
        return f"a clear photo of a {tag}"

    # Animal prompts
    elif tag in ["animal", "dog", "cat", "bird", "fish", "horse", "cow", "sheep", "wildlife", "pet"]:
        return f"a photo of a {tag} clearly visible"

    # Nature prompts
    elif tag in ["tree", "flower", "plant", "mountain", "river", "lake", "ocean", "beach", "forest",
                "sky", "cloud", "sun", "moon", "stars"]:
        return f"a landscape showing {tag}"

    # Environment prompts
    elif tag in ["city", "urban", "rural", "indoor", "outdoor", "street", "park", "garden", "office",
                "home", "kitchen", "bedroom", "bathroom"]:
        return f"a scene of a {tag} environment"

    # Activity prompts
    elif tag in ["walking", "running", "swimming", "eating", "drinking", "reading", "writing",
                "working", "playing", "dancing", "singing"]:
        return f"a photo of someone {tag}"

    # Default prompt format
    else:
        return f"a photo of {tag}"


def get_clip_model():
    """Get the CLIP model based on user configuration using a simplified approach"""
    # Get the selected CLIP model from user config
    config = UserConfig.query.first()
    clip_model_name = 'ViT-B-32'  # Default model (smallest and fastest)
    use_custom_model = False
    custom_model_name = None

    # Create config if it doesn't exist
    if not config:
        config = UserConfig(clip_model=clip_model_name)
        db.session.add(config)
        db.session.commit()
    elif config.clip_model:
        clip_model_name = config.clip_model

    # Check if custom model is enabled
    if config and config.custom_model_enabled and config.custom_model:
        use_custom_model = True
        custom_model_name = config.custom_model

    # If using custom model, use that instead of the standard model
    model_key = custom_model_name if use_custom_model else clip_model_name

    # Check if model is already loaded
    if model_key in clip_models:
        return model_key, clip_models[model_key], clip_preprocessors[model_key]

    # Check available memory before loading a new model
    mem = psutil.virtual_memory()
    available_gb = mem.available / (1024 * 1024 * 1024)

    # Define large models that might cause memory issues
    large_models = ['ViT-SO400M-16-SigLIP2-512', 'ViT-L-14', 'ViT-H-14', 'ViT-g-14']

    # Force CPU for large models if memory is limited
    force_cpu_for_model = False
    if model_key in large_models and (FORCE_CPU_FOR_LARGE_MODELS or available_gb < 8.0) and device == "cuda":
        current_app.logger.warning(f"Model {model_key} is large and available memory is limited ({available_gb:.2f} GB). Forcing CPU usage.")
        force_cpu_for_model = True

    # Clear models to free memory before loading a new one
    if len(clip_models) > 2:
        current_app.logger.info(f"Clearing model cache to free memory before loading {model_key}")
        for model_name in list(clip_models.keys()):
            if model_name != model_key:
                current_app.logger.info(f"Removing model {model_name} from cache")
                del clip_models[model_name]
                del clip_preprocessors[model_name]
                if model_name in tag_embeddings:
                    del tag_embeddings[model_name]

        # Force garbage collection
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

    # Load the model
    try:
        # Determine which device to use for this model
        model_device = "cpu" if force_cpu_for_model else device

        # Set recursion limit for model loading
        old_recursion_limit = sys.getrecursionlimit()
        sys.setrecursionlimit(20000)  # Increase recursion limit temporarily

        try:
            # Determine cache directory
            data_folder = current_app.config.get("DATA_FOLDER", "./data")
            models_folder = os.path.join(data_folder, "models")
            cache_dir = models_folder if os.path.exists(models_folder) else "/app/data/model_cache"

            # Determine pretrained tag to use based on model
            if model_key == 'ViT-H-14':
                pretrained_tag = 'laion2b_s32b_b79k'  # Use the correct tag for ViT-H-14
            else:
                pretrained_tag = 'openai'  # Default pretrained tag for other models

            # Load the model
            model, _, preprocess = open_clip.create_model_and_transforms(
                model_key,
                pretrained=pretrained_tag,
                jit=False,
                force_quick_gelu=True,  # Enable QuickGELU to match pretrained weights
                cache_dir=cache_dir
            )
            current_app.logger.info(f"Successfully loaded {model_key} with pretrained tag: {pretrained_tag}")
        finally:
            # Restore original recursion limit
            sys.setrecursionlimit(old_recursion_limit)

        # Move model to the appropriate device (CPU or CUDA)
        model.to(model_device)
        model.eval()

        # Cache the model and preprocessor
        clip_models[model_key] = model
        clip_preprocessors[model_key] = preprocess

        # Precompute tag embeddings for this model
        tokenizer = open_clip.get_tokenizer(model_key)
        if model_key not in tag_embeddings:
            tag_embeddings[model_key] = {}
            with torch.no_grad():
                # Process tags in smaller batches to save memory
                batch_size = 10
                for i in range(0, len(CANDIDATE_TAGS), batch_size):
                    batch_tags = CANDIDATE_TAGS[i:i+batch_size]
                    for tag in batch_tags:
                        # Use more specific prompts for better object detection
                        prompt = get_better_prompt_for_tag(tag)
                        text_tokens = tokenizer([prompt])
                        text_features = model.encode_text(text_tokens)
                        text_features /= text_features.norm(dim=-1, keepdim=True)
                        tag_embeddings[model_key][tag] = text_features.cpu()  # Store on CPU to save GPU memory
                    # Force garbage collection between batches
                    gc.collect()

        return model_key, model, preprocess
    except Exception as e:
        current_app.logger.error(f"Error loading model {model_key}: {e}")
        # Fall back to default model if available
        if 'ViT-B-32' in clip_models:
            return 'ViT-B-32', clip_models['ViT-B-32'], clip_preprocessors['ViT-B-32']

        # Otherwise load default model with simplified approach
        try:
            # Use the pre-downloaded model from the Docker build
            cache_dir = "/app/data/model_cache"
            model, _, preprocess = open_clip.create_model_and_transforms(
                'ViT-B-32',
                pretrained='openai',
                jit=False,
                force_quick_gelu=True,  # Enable QuickGELU to match pretrained weights
                cache_dir=cache_dir
            )
            model.to(device)
            model.eval()
            clip_models['ViT-B-32'] = model
            clip_preprocessors['ViT-B-32'] = preprocess
            return 'ViT-B-32', model, preprocess
        except Exception as fallback_error:
            current_app.logger.error(f"Error loading fallback model: {fallback_error}")
            raise Exception("Failed to load any CLIP model")


def get_image_embedding(image_path):
    """
    Generate an embedding for an image using the current CLIP model.

    Args:
        image_path: Path to the image file

    Returns:
        tuple: (embedding array, model_name) or (None, None) on error
    """
    try:
        # Get the current CLIP model
        model_name, model, preprocess = get_clip_model()

        # Determine which device the model is on
        model_device = next(model.parameters()).device
        current_app.logger.info(f"Model {model_name} is on device: {model_device}")

        # Process the image and ensure it's on the same device as the model
        image = Image.open(image_path).convert("RGB")
        image_input = preprocess(image).unsqueeze(0).to(model_device)

        # Get image features
        with torch.no_grad():
            embedding = model.encode_image(image_input)
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)

        return embedding.cpu().numpy()[0], model_name
    except Exception as e:
        try:
            current_app.logger.error(f"Error processing image {image_path}: {e}")
        except:
            print(f"Error processing image {image_path}: {e}")
        return None, None


def generate_tags_and_description(embedding, model_name):
    """
    Generate tags and description based on image embedding and model.

    Args:
        embedding: Image embedding array
        model_name: Name of the model used to generate the embedding

    Returns:
        tuple: (list of tags, description string)
    """
    # If model_name is None, use default
    if model_name is None:
        model_name = 'ViT-B-32'

    # If model embeddings not found, try to load the model
    if model_name not in tag_embeddings:
        try:
            get_clip_model()
        except Exception as e:
            current_app.logger.error(f"Error loading model for tag generation: {e}")
            # Fall back to any available model
            if len(tag_embeddings) > 0:
                model_name = list(tag_embeddings.keys())[0]
            else:
                return [], "No tags available"

    # Get user config for tags setting and similarity threshold
    config = UserConfig.query.first()
    max_tags = 5  # Default maximum number of tags
    threshold_level = DEFAULT_THRESHOLD  # Default threshold level

    if config:
        if hasattr(config, 'min_tags') and config.min_tags is not None:
            max_tags = config.min_tags
        if hasattr(config, 'similarity_threshold') and config.similarity_threshold is not None:
            threshold_level = config.similarity_threshold

    # Get the actual cosine threshold value from the level
    cosine_threshold = SIMILARITY_THRESHOLDS.get(threshold_level, SIMILARITY_THRESHOLDS[DEFAULT_THRESHOLD])

    # Calculate similarities with tag embeddings
    scores = {}
    for tag in CANDIDATE_TAGS:
        if tag in tag_embeddings.get(model_name, {}):
            # Get the tag embedding for this model
            tag_emb = tag_embeddings[model_name][tag]
            # Ensure both tensors are on the same device (CPU)
            embedding_tensor = torch.tensor(embedding, device='cpu').unsqueeze(0)
            tag_emb_cpu = tag_emb.cpu()
            # Calculate similarity
            similarity = torch.cosine_similarity(
                embedding_tensor,
                tag_emb_cpu,
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
            filtered_tags.append(tag)
            # Stop once we reach the maximum number of tags
            if len(filtered_tags) >= max_tags:
                break

    # Create description
    description = "This image may contain " + ", ".join(filtered_tags) + "."

    return filtered_tags, description
