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

# Now import the rest of the modules
import datetime
import subprocess
import time
import uuid
import multiprocessing

from flask import current_app
from PIL import Image

import open_clip
from sklearn.metrics.pairwise import cosine_similarity

# Set multiprocessing start method to 'spawn' to fix CUDA issues
# This needs to be done before any multiprocessing operations
try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    # Method already set, ignore
    pass

# Set up signal handlers for graceful error handling
import signal
import traceback
import sys

def handle_segfault(signum, frame):
    """Handle segmentation fault (SIGSEGV) more gracefully"""
    error_msg = f"CRITICAL ERROR: Segmentation fault (SIGSEGV) detected in process {os.getpid()}"
    stack_trace = ''.join(traceback.format_stack(frame))
    
    # Log the error
    print(f"{error_msg}\nStack trace:\n{stack_trace}", file=sys.stderr)
    
    # Try to log to file as well
    try:
        with open('/tmp/segfault_log.txt', 'a') as f:
            f.write(f"\n{'-'*80}\n{datetime.datetime.now()}: {error_msg}\n")
            f.write(f"Stack trace:\n{stack_trace}\n")
    except:
        pass
    
    # Exit more gracefully than a hard crash
    os._exit(1)

# Register the signal handler
signal.signal(signal.SIGSEGV, handle_segfault)

# Import and initialize Celery and APScheduler
from celery import Celery
from apscheduler.schedulers.background import BackgroundScheduler

# Initialize Celery with memory limits
celery = Celery('tasks', broker='redis://localhost:6379/0', backend='redis://localhost:6379/0')

# Configure Celery with optimized settings but no memory limits
celery.conf.update(
    worker_max_tasks_per_child=10,        # Restart worker after 10 tasks (increased from 1)
    task_time_limit=1800,                 # 30 minute time limit per task (increased from 10 minutes)
    task_soft_time_limit=1500,            # 25 minute soft time limit (increased from 5 minutes)
    worker_concurrency=1,                 # Limit to 1 concurrent worker to prevent duplicate tasks
    broker_connection_retry_on_startup=True,  # Fix deprecation warning
    worker_hijack_root_logger=False,      # Don't hijack the root logger
    worker_log_format='%(asctime)s [%(levelname)s] %(message)s',  # Simplified log format
    task_track_started=False,             # Don't log when tasks are started
    task_send_sent_event=False,           # Don't send task-sent events
    worker_send_task_events=False,        # Don't send task events
    task_ignore_result=True               # Don't store task results
)

class FlaskTask(celery.Task):
    def __call__(self, *args, **kwargs):
        # Import the app inside the task to avoid circular imports
        import sys
        import os
        # Add the current directory to the path if not already there
        app_dir = os.path.dirname(os.path.abspath(__file__))
        if app_dir not in sys.path:
            sys.path.insert(0, app_dir)
        
        # Now import the app
        from app import app as flask_app
        with flask_app.app_context():
            return self.run(*args, **kwargs)

celery.Task = FlaskTask

# Import APScheduler but don't initialize it here
# The scheduler is now initialized in a dedicated process (scheduler.py)
from apscheduler.schedulers.background import BackgroundScheduler

# Device setup with environment variable override for large models
import os
device = "cuda" if torch.cuda.is_available() else "cpu"

# Check if we should force CPU for large models based on system memory
FORCE_CPU_FOR_LARGE_MODELS = os.environ.get('FORCE_CPU_FOR_LARGE_MODELS', '0') == '1'
# Handle empty string case for SYSTEM_MEMORY_GB
system_memory_str = os.environ.get('SYSTEM_MEMORY_GB', '0')
SYSTEM_MEMORY_GB = float(system_memory_str) if system_memory_str else 0.0

if FORCE_CPU_FOR_LARGE_MODELS:
    print(f"System configured to force CPU for large models due to limited memory ({SYSTEM_MEMORY_GB}GB)")

# Define candidate tags with more focus on objects and content
CANDIDATE_TAGS = [
    # People and portraits
    "person", "people", "man", "woman", "child", "children", "baby", "group of people", "crowd", "portrait", "selfie", "face",
    
    # Objects
    "car", "vehicle", "bicycle", "motorcycle", "airplane", "boat", "building", "house", "furniture", "chair", "table", "bed",
    "computer", "phone", "television", "book", "clock", "bottle", "cup", "plate", "food", "fruit", "vegetable", "meal",
    
    # Animals
    "animal", "dog", "cat", "bird", "fish", "horse", "cow", "sheep", "wildlife", "pet",
    
    # Nature
    "tree", "flower", "plant", "mountain", "river", "lake", "ocean", "beach", "forest", "sky", "cloud", "sun", "moon", "stars",
    
    # Environments
    "city", "urban", "rural", "indoor", "outdoor", "street", "park", "garden", "office", "home", "kitchen", "bedroom", "bathroom",
    
    # Activities
    "walking", "running", "swimming", "eating", "drinking", "reading", "writing", "working", "playing", "dancing", "singing",
    
    # Time
    "day", "night", "sunrise", "sunset", "morning", "evening",
    
    # Styles (fewer than before)
    "colorful", "monochrome", "black and white", "bright", "dark",
    
    # Qualities
    "natural", "artificial", "modern", "vintage", "detailed", "minimal", "realistic", "abstract"
]

# Similarity threshold levels mapped to cosine values
SIMILARITY_THRESHOLDS = {
    "very_high": 0.5,    # Only very strong matches (highest precision)
    "high": 0.4,         # Strong matches (high precision)
    "medium": 0.3,       # Balanced matches (default)
    "low": 0.2,          # More inclusive matches (higher recall)
    "very_low": 0.1      # Most inclusive matches (highest recall)
}

# Default threshold (will be overridden by user settings)
DEFAULT_THRESHOLD = "medium"

# Function to generate better prompts for object detection
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

# Model and embeddings cache
clip_models = {}
clip_preprocessors = {}
tag_embeddings = {}

def get_clip_model():
    """Get the CLIP model based on user configuration"""
    from models import UserConfig
    from flask import current_app
    import gc  # Garbage collection
    import psutil  # For memory monitoring
    
    # Get the selected CLIP model from user config
    config = UserConfig.query.first()
    clip_model_name = 'ViT-B-32'  # Default model (smallest and fastest)
    use_custom_model = False
    custom_model_name = None
    
    # Create config if it doesn't exist
    if not config:
        from models import db
        config = UserConfig(clip_model=clip_model_name)
        db.session.add(config)
        db.session.commit()
    elif config.clip_model:
        clip_model_name = config.clip_model
    
    # Check if custom model is enabled
    if config and config.custom_model_enabled and config.custom_model:
        use_custom_model = True
        custom_model_name = config.custom_model
        current_app.logger.info(f"🔍 CUSTOM MODEL SELECTION: Using custom model {custom_model_name} for image tagging")
    else:
        current_app.logger.info(f"🔍 CLIP MODEL SELECTION: Using {clip_model_name} for image tagging")
    
    # If using custom model, use that instead of the standard model
    model_key = custom_model_name if use_custom_model else clip_model_name
    
    # Check if model is already loaded
    if model_key in clip_models:
        return model_key, clip_models[model_key], clip_preprocessors[model_key]
    
    # Check available memory before loading a new model
    mem = psutil.virtual_memory()
    available_gb = mem.available / (1024 * 1024 * 1024)
    current_app.logger.info(f"Available memory before loading model: {available_gb:.2f} GB")
    
    # Define large models that might cause memory issues
    large_models = ['ViT-SO400M-16-SigLIP2-512', 'ViT-L-14', 'ViT-H-14', 'ViT-g-14']
    
    # Models that are known to cause segmentation faults with CUDA
    problematic_models = ['ViT-SO400M-16-SigLIP2-512']
    
    # Force CPU for large models if memory is limited or if configured to do so
    force_cpu_for_model = False
    
    # Always force CPU for problematic models that cause segmentation faults
    if model_key in problematic_models and device == "cuda":
        current_app.logger.warning(f"Model {model_key} is known to cause segmentation faults with CUDA. Forcing CPU usage for this model.")
        force_cpu_for_model = True
    # Force CPU for large models based on memory or configuration
    elif model_key in large_models and (FORCE_CPU_FOR_LARGE_MODELS or available_gb < 8.0) and device == "cuda":
        current_app.logger.warning(f"Model {model_key} is large and available memory is limited ({available_gb:.2f} GB) or FORCE_CPU_FOR_LARGE_MODELS is enabled. Forcing CPU usage for this model.")
        force_cpu_for_model = True
    
    # Clear models to free memory before loading a new one
    if len(clip_models) > 3 or (model_key in large_models and len(clip_models) > 1):
        current_app.logger.info(f"Clearing model cache to free memory before loading {model_key}")
        # Keep fewer models in memory when dealing with large models
        models_to_keep = 1 if model_key in large_models else 2
        models_to_remove = list(clip_models.keys())[:-models_to_keep] if len(clip_models) > models_to_keep else list(clip_models.keys())
        for model_name in models_to_remove:
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
        current_app.logger.info(f"Loading model {model_key} on device: {model_device}")
        
        # Set the default device for PyTorch operations in this context
        # This helps ensure all new tensors are created on the right device
        if model_device == "cpu":
            # If we're forcing CPU, make sure CUDA isn't used by default
            old_device = torch.cuda.current_device() if torch.cuda.is_available() else -1
            torch.cuda.set_device(-1)  # Set to CPU
            current_app.logger.info(f"Temporarily setting default CUDA device to CPU for model loading")
        
        if use_custom_model:
            current_app.logger.info(f"Loading custom model: {custom_model_name}")
            # Set the cache directory for custom models
            data_folder = current_app.config.get("DATA_FOLDER", "./data")
            models_folder = os.path.join(data_folder, "models")
            
            # Check if we have a record of which pretrained tag was used for this model
            data_folder = current_app.config.get("DATA_FOLDER", "./data")
            models_folder = os.path.join(data_folder, "models")
            model_info_path = os.path.join(models_folder, f"{custom_model_name.replace('/', '_')}_info.json")
            
            # Default pretrained tags to try
            available_pretrained_tags = []
            
            # Try to load the model info to get the pretrained tag
            if os.path.exists(model_info_path):
                try:
                    import json
                    with open(model_info_path, 'r') as f:
                        model_info = json.load(f)
                        if 'pretrained_tag' in model_info and model_info['pretrained_tag']:
                            # If we have a record of the pretrained tag, use it first
                            available_pretrained_tags.append(model_info['pretrained_tag'])
                            current_app.logger.info(f"Using recorded pretrained tag for {custom_model_name}: {model_info['pretrained_tag']}")
                except Exception as e:
                    current_app.logger.error(f"Error reading model info file: {str(e)}")
            
            # Get available pretrained tags for the model
            try:
                # Try to get available pretrained tags from open_clip
                model_pretrained_tags = open_clip.list_pretrained_tags_by_model(custom_model_name)
                current_app.logger.info(f"Available pretrained tags for {custom_model_name}: {model_pretrained_tags}")
                
                # Add any tags we don't already have
                for tag in model_pretrained_tags:
                    if tag not in available_pretrained_tags:
                        available_pretrained_tags.append(tag)
            except Exception as e:
                current_app.logger.info(f"Could not get pretrained tags for {custom_model_name}: {str(e)}")
            
            # If still no tags, use default list
            if not available_pretrained_tags:
                available_pretrained_tags = ['openai', 'laion2b_s34b_b79k', 'datacomp1b', 'laion400m_e32', 'laion2b']
            
            # Try different model name formats and pretrained tags
            success = False
            last_error = None
            
            # List of model name variations to try
            model_variations = []
            
            # Original model name
            model_variations.append(custom_model_name)
            
            # If model name contains a slash, try different variations
            if '/' in custom_model_name:
                # Extract the model architecture name (after the slash)
                model_arch = custom_model_name.split('/')[-1]
                model_variations.append(model_arch)
                
                # Try with organization as prefix
                org = custom_model_name.split('/')[0]
                model_variations.append(f"{org}-{model_arch}")
                
                # Try with underscores instead of dashes
                model_variations.append(model_arch.replace('-', '_'))
            
            # Try each model variation with each pretrained tag
            for model_var in model_variations:
                current_app.logger.info(f"Trying model variation: {model_var}")
                
                for pretrained_tag in available_pretrained_tags:
                    try:
                        current_app.logger.info(f"Attempting to load {model_var} with pretrained tag: {pretrained_tag}")
                        
                        # Set a timeout for model loading to prevent hanging
                        import signal
                        
                        def timeout_handler(signum, frame):
                            raise TimeoutError(f"Loading model {model_var} timed out after 60 seconds")
                        
                        # Set timeout for large models
                        if model_var in large_models:
                            signal.signal(signal.SIGALRM, timeout_handler)
                            signal.alarm(60)  # 60 second timeout for large models
                        
                        try:
                            # Set a recursion limit for model downloading
                            import sys
                            old_recursion_limit = sys.getrecursionlimit()
                            sys.setrecursionlimit(15000)  # Increase recursion limit temporarily
                            
                            try:
                                model, _, preprocess = open_clip.create_model_and_transforms(
                                    model_var,
                                    pretrained=pretrained_tag,
                                    jit=False,
                                    force_quick_gelu=False,  # Don't force QuickGELU to avoid warnings
                                    cache_dir=models_folder if os.path.exists(models_folder) else "/app/data/model_cache"
                                )
                            finally:
                                # Restore original recursion limit
                                sys.setrecursionlimit(old_recursion_limit)
                            
                            # Cancel timeout if successful
                            if model_var in large_models:
                                signal.alarm(0)
                                
                            # If we get here, the model loaded successfully
                            current_app.logger.info(f"Successfully loaded {model_var} with pretrained tag: {pretrained_tag}")
                            success = True
                            break
                        except TimeoutError as te:
                            current_app.logger.error(f"Timeout loading model {model_var}: {str(te)}")
                            # Cancel timeout
                            if model_var in large_models:
                                signal.alarm(0)
                            raise
                            
                    except Exception as e:
                        last_error = e
                        current_app.logger.info(f"Failed to load {model_var} with pretrained tag {pretrained_tag}: {str(e)}")
                
                if success:
                    break
            
            # If all attempts failed, provide a more helpful error message
            if not success:
                # Check if the last error was a recursion error
                if isinstance(last_error, RecursionError) or "maximum recursion depth exceeded" in str(last_error):
                    raise Exception(f"Failed to download model {custom_model_name} due to recursion limits. Please try a different model like ViT-B-32 or RN50 which are smaller and more compatible.")
                else:
                    raise Exception(f"Failed to load model {custom_model_name} with any variation or pretrained tag. Last error: {str(last_error)}")
        else:
            current_app.logger.info(f"Loading CLIP model: {clip_model_name}")
            
            # Get available pretrained tags for the model
            available_pretrained_tags = []
            try:
                # Try to get available pretrained tags
                available_pretrained_tags = open_clip.list_pretrained_tags_by_model(clip_model_name)
                current_app.logger.info(f"Available pretrained tags for {clip_model_name}: {available_pretrained_tags}")
            except Exception as e:
                current_app.logger.info(f"Could not get pretrained tags for {clip_model_name}: {str(e)}")
                # If we can't get tags, use default tags to try
                available_pretrained_tags = ['openai', 'laion2b_s34b_b79k', 'datacomp1b', 'laion400m_e32', 'laion2b', 'merged2b_s6b_b61k']
            
            # If no tags found, use default list
            if not available_pretrained_tags:
                available_pretrained_tags = ['openai', 'laion2b_s34b_b79k', 'datacomp1b', 'laion400m_e32', 'laion2b', 'merged2b_s6b_b61k']
            
            # Try each pretrained tag
            success = False
            last_error = None
            
            for pretrained_tag in available_pretrained_tags:
                try:
                    current_app.logger.info(f"Attempting to load {clip_model_name} with pretrained tag: {pretrained_tag}")
                    
                    # Set a timeout for model loading to prevent hanging
                    import signal
                    
                    def timeout_handler(signum, frame):
                        raise TimeoutError(f"Loading model {clip_model_name} timed out after 60 seconds")
                    
                    # Set timeout for large models
                    if clip_model_name in large_models:
                        signal.signal(signal.SIGALRM, timeout_handler)
                        signal.alarm(60)  # 60 second timeout for large models
                    
                    try:
                        # Set a recursion limit for model downloading
                        import sys
                        old_recursion_limit = sys.getrecursionlimit()
                        sys.setrecursionlimit(15000)  # Increase recursion limit temporarily
                        
                        try:
                            # Use the pre-downloaded model from the Docker build
                            cache_dir = "/app/data/model_cache"
                            model, _, preprocess = open_clip.create_model_and_transforms(
                                clip_model_name,
                                pretrained=pretrained_tag,
                                jit=False,
                                force_quick_gelu=False,  # Don't force QuickGELU to avoid warnings
                                cache_dir=cache_dir
                            )
                        finally:
                            # Restore original recursion limit
                            sys.setrecursionlimit(old_recursion_limit)
                        
                        # Cancel timeout if successful
                        if clip_model_name in large_models:
                            signal.alarm(0)
                            
                        # If we get here, the model loaded successfully
                        current_app.logger.info(f"Successfully loaded {clip_model_name} with pretrained tag: {pretrained_tag}")
                        success = True
                        break
                    except TimeoutError as te:
                        current_app.logger.error(f"Timeout loading model {clip_model_name}: {str(te)}")
                        # Cancel timeout
                        if clip_model_name in large_models:
                            signal.alarm(0)
                        raise
                        
                except Exception as e:
                    last_error = e
                    current_app.logger.info(f"Failed to load {clip_model_name} with pretrained tag {pretrained_tag}: {str(e)}")
            
            # If all attempts failed, provide a more helpful error message
            if not success:
                # Check if the last error was a recursion error
                if isinstance(last_error, RecursionError) or "maximum recursion depth exceeded" in str(last_error):
                    raise Exception(f"Failed to download model {clip_model_name} due to recursion limits. Please try a different model like ViT-B-32 or RN50 which are smaller and more compatible.")
                else:
                    raise Exception(f"Failed to load model {clip_model_name} with any pretrained tag. Last error: {str(last_error)}")
        
        # Move model to the appropriate device (CPU or CUDA)
        model.to(model_device)
        model.eval()
        
        # Verify the model is on the expected device
        actual_device = next(model.parameters()).device
        if str(actual_device) != str(model_device):
            current_app.logger.warning(f"Model device mismatch! Requested: {model_device}, Actual: {actual_device}")
            # Force the model to the correct device again
            model = model.to(model_device)
            # Verify again
            actual_device = next(model.parameters()).device
            current_app.logger.info(f"After correction, model is on device: {actual_device}")
        
        # Restore the default CUDA device if we changed it
        if model_device == "cpu" and torch.cuda.is_available() and 'old_device' in locals():
            if old_device >= 0:
                torch.cuda.set_device(old_device)
                current_app.logger.info(f"Restored default CUDA device to {old_device}")
        
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
        # Otherwise load default model
        import sys
        old_recursion_limit = sys.getrecursionlimit()
        sys.setrecursionlimit(15000)  # Increase recursion limit temporarily
        
        try:
            # Use the pre-downloaded model from the Docker build
            cache_dir = "/app/data/model_cache"
            model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai', jit=False, force_quick_gelu=False, cache_dir=cache_dir)
            model.to(device)
            model.eval()
            clip_models['ViT-B-32'] = model
            clip_preprocessors['ViT-B-32'] = preprocess
            return 'ViT-B-32', model, preprocess
        finally:
            # Restore original recursion limit
            sys.setrecursionlimit(old_recursion_limit)

def get_image_embedding(image_path):
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
    """Generate tags and description based on image embedding and model"""
    from flask import current_app
    from models import UserConfig
    
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
    
    # Log the threshold, number of tags, and their similarity scores
    current_app.logger.info(f"📊 TAG GENERATION: Using model {model_name}")
    current_app.logger.info(f"📊 TAG GENERATION: Generated {len(filtered_tags)} tags with similarity threshold {threshold_level} ({cosine_threshold}) (max: {max_tags})")
    
    # Log the selected tags with their similarity scores
    tag_scores = [(tag, scores[tag]) for tag in filtered_tags]
    tag_score_str = ", ".join([f"{tag} ({score:.3f})" for tag, score in tag_scores])
    current_app.logger.info(f"📊 TAG GENERATION: Selected tags with scores: {tag_score_str}")
    
    # Create description
    description = "This image may contain " + ", ".join(filtered_tags) + "."
    
    return filtered_tags, description

@celery.task(bind=True, time_limit=600, soft_time_limit=500, max_retries=3)
def process_image_tagging(self, filename):
    """
    Process an image: compute its embedding, generate tags and a description,
    and update its record in the database.
    """
    from models import ImageDB, db, UserConfig
    from flask import current_app
    import gc  # Garbage collection
    
    # Get user configuration
    config = UserConfig.query.first()
    if not config:
        from models import db
        config = UserConfig(ram_enabled=True, ram_model="ram_large")
        db.session.add(config)
        db.session.commit()
    
    # Determine image path
    image_folder = current_app.config.get("IMAGE_FOLDER", "./images")
    image_path = os.path.join(image_folder, filename)
    if not os.path.exists(image_path):
        return {"status": "error", "message": "Image file not found", "filename": filename}
    
    try:
        # Check if this is a screenshot (which should use CLIP) or a regular image (which should use RAM)
        is_screenshot = filename.startswith("screenshot_")
        
        if is_screenshot:
            # For screenshots, use the existing CLIP model
            clip_model_name = config.clip_model if config and config.clip_model else "ViT-B-32"
            current_app.logger.info(f"🏷️ TAGGING: Processing screenshot {filename} with CLIP model: {clip_model_name}")
            
            # Get image embedding using the selected CLIP model
            embedding, model_name = get_image_embedding(image_path)
            if embedding is None:
                return {"status": "error", "message": "Failed to compute embedding", "filename": filename}
            
            # Generate tags and description using CLIP
            tags, description = generate_tags_and_description(embedding, model_name)
            model_used = f"CLIP: {model_name}"
        else:
            # For regular images, use RAM if enabled
            if config.ram_enabled:
                try:
                    # Import RAM tagger
                    from utils.ram_tagger import generate_tags_with_ram
                    
                    # Get RAM model name from config
                    ram_model = config.ram_model if config and config.ram_model else "ram_large"
                    current_app.logger.info(f"🏷️ TAGGING: Processing image {filename} with RAM model: {ram_model}")
                    
                    # Generate tags and description using RAM
                    tags, description = generate_tags_with_ram(
                        image_path,
                        model_name=ram_model,
                        max_tags=config.min_tags,
                        min_confidence=config.ram_min_confidence
                    )
                    
                    # Check if tags were generated successfully
                    if tags and len(tags) > 0:
                        model_used = f"RAM: {ram_model}"
                    else:
                        # If RAM failed to generate tags, fall back to CLIP
                        current_app.logger.warning(f"RAM failed to generate tags for {filename}, falling back to CLIP")
                        clip_model_name = config.clip_model if config and config.clip_model else "ViT-B-32"
                        current_app.logger.info(f"🏷️ TAGGING: Processing image {filename} with CLIP model: {clip_model_name} (RAM fallback)")
                        
                        # Get image embedding using the selected CLIP model
                        embedding, model_name = get_image_embedding(image_path)
                        if embedding is None:
                            return {"status": "error", "message": "Failed to compute embedding", "filename": filename}
                        
                        # Generate tags and description using CLIP
                        tags, description = generate_tags_and_description(embedding, model_name)
                        model_used = f"CLIP: {model_name}"
                except Exception as e:
                    # If RAM fails with an exception, fall back to CLIP
                    current_app.logger.error(f"Error using RAM for {filename}: {e}, falling back to CLIP")
                    clip_model_name = config.clip_model if config and config.clip_model else "ViT-B-32"
                    current_app.logger.info(f"🏷️ TAGGING: Processing image {filename} with CLIP model: {clip_model_name} (RAM error fallback)")
                    
                    # Get image embedding using the selected CLIP model
                    embedding, model_name = get_image_embedding(image_path)
                    if embedding is None:
                        return {"status": "error", "message": "Failed to compute embedding", "filename": filename}
                    
                    # Generate tags and description using CLIP
                    tags, description = generate_tags_and_description(embedding, model_name)
                    model_used = f"CLIP: {model_name}"
            else:
                # If RAM is disabled, fall back to CLIP
                clip_model_name = config.clip_model if config and config.clip_model else "ViT-B-32"
                current_app.logger.info(f"🏷️ TAGGING: Processing image {filename} with CLIP model: {clip_model_name} (RAM disabled)")
                
                # Get image embedding using the selected CLIP model
                embedding, model_name = get_image_embedding(image_path)
                if embedding is None:
                    return {"status": "error", "message": "Failed to compute embedding", "filename": filename}
                
                # Generate tags and description using CLIP
                tags, description = generate_tags_and_description(embedding, model_name)
                model_used = f"CLIP: {model_name}"
        
        # Update database
        img = ImageDB.query.filter_by(filename=filename).first()
        if img:
            img.tags = ", ".join(tags)
            img.description = description
            db.session.commit()
            
            return {
                "status": "success",
                "filename": filename,
                "tags": tags,
                "description": description,
                "model_used": model_used
            }
        else:
            return {"status": "error", "message": "Image not found in database", "filename": filename}
    except Exception as e:
        current_app.logger.error(f"Error in process_image_tagging: {e}")
        # Don't clear models on error to avoid reloading them
        
        return {"status": "error", "message": str(e), "filename": filename}

def reembed_image(filename):
    """
    Recompute the embedding, tags, and description for an image and update its record.
    """
    from models import ImageDB, db, UserConfig
    from flask import current_app
    
    # Get user configuration
    config = UserConfig.query.first()
    if not config:
        from models import db
        config = UserConfig(ram_enabled=True, ram_model="ram_large")
        db.session.add(config)
        db.session.commit()
    
    # Determine image path
    image_folder = current_app.config.get("IMAGE_FOLDER", "./images")
    image_path = os.path.join(image_folder, filename)
    if not os.path.exists(image_path):
        return {"status": "error", "message": "Image file not found", "filename": filename}
    
    try:
        # Check if this is a screenshot (which should use CLIP) or a regular image (which should use RAM)
        is_screenshot = filename.startswith("screenshot_")
        
        if is_screenshot:
            # For screenshots, use the existing CLIP model
            clip_model_name = config.clip_model if config and config.clip_model else "ViT-B-32"
            current_app.logger.info(f"🔄 RE-TAGGING: Processing screenshot {filename} with CLIP model: {clip_model_name}")
            
            # Get image embedding using the selected CLIP model
            embedding, model_name = get_image_embedding(image_path)
            if embedding is None:
                return {"status": "error", "message": "Failed to compute embedding", "filename": filename}
            
            # Generate tags and description using CLIP
            tags, description = generate_tags_and_description(embedding, model_name)
            model_used = f"CLIP: {model_name}"
        else:
            # For regular images, use RAM if enabled
            if config.ram_enabled:
                try:
                    # Import RAM tagger
                    from utils.ram_tagger import generate_tags_with_ram
                    
                    # Get RAM model name from config
                    ram_model = config.ram_model if config and config.ram_model else "ram_large"
                    current_app.logger.info(f"🔄 RE-TAGGING: Processing image {filename} with RAM model: {ram_model}")
                    
                    # Generate tags and description using RAM
                    tags, description = generate_tags_with_ram(
                        image_path,
                        model_name=ram_model,
                        max_tags=config.min_tags,
                        min_confidence=config.ram_min_confidence
                    )
                    
                    # Check if tags were generated successfully
                    if tags and len(tags) > 0:
                        model_used = f"RAM: {ram_model}"
                    else:
                        # If RAM failed to generate tags, fall back to CLIP
                        current_app.logger.warning(f"RAM failed to generate tags for {filename}, falling back to CLIP")
                        clip_model_name = config.clip_model if config and config.clip_model else "ViT-B-32"
                        current_app.logger.info(f"🔄 RE-TAGGING: Processing image {filename} with CLIP model: {clip_model_name} (RAM fallback)")
                        
                        # Get image embedding using the selected CLIP model
                        embedding, model_name = get_image_embedding(image_path)
                        if embedding is None:
                            return {"status": "error", "message": "Failed to compute embedding", "filename": filename}
                        
                        # Generate tags and description using CLIP
                        tags, description = generate_tags_and_description(embedding, model_name)
                        model_used = f"CLIP: {model_name}"
                except Exception as e:
                    # If RAM fails with an exception, fall back to CLIP
                    current_app.logger.error(f"Error using RAM for {filename}: {e}, falling back to CLIP")
                    clip_model_name = config.clip_model if config and config.clip_model else "ViT-B-32"
                    current_app.logger.info(f"🔄 RE-TAGGING: Processing image {filename} with CLIP model: {clip_model_name} (RAM error fallback)")
                    
                    # Get image embedding using the selected CLIP model
                    embedding, model_name = get_image_embedding(image_path)
                    if embedding is None:
                        return {"status": "error", "message": "Failed to compute embedding", "filename": filename}
                    
                    # Generate tags and description using CLIP
                    tags, description = generate_tags_and_description(embedding, model_name)
                    model_used = f"CLIP: {model_name}"
            else:
                # If RAM is disabled, fall back to CLIP
                clip_model_name = config.clip_model if config and config.clip_model else "ViT-B-32"
                current_app.logger.info(f"🔄 RE-TAGGING: Processing image {filename} with CLIP model: {clip_model_name} (RAM disabled)")
                
                # Get image embedding using the selected CLIP model
                embedding, model_name = get_image_embedding(image_path)
                if embedding is None:
                    return {"status": "error", "message": "Failed to compute embedding", "filename": filename}
                
                # Generate tags and description using CLIP
                tags, description = generate_tags_and_description(embedding, model_name)
                model_used = f"CLIP: {model_name}"
        
        # Update database
        img = ImageDB.query.filter_by(filename=filename).first()
        if img:
            img.tags = ", ".join(tags)
            img.description = description
            db.session.commit()
            return {
                "status": "success",
                "filename": filename,
                "tags": tags,
                "description": description,
                "model_used": model_used
            }
        else:
            return {"status": "error", "message": "Image not found in database", "filename": filename}
    except Exception as e:
        current_app.logger.error(f"Error in reembed_image: {e}")
        return {"status": "error", "message": str(e), "filename": filename}

# Global dictionary to track bulk tagging progress
BULK_PROGRESS = {}

@celery.task(bind=True)
def bulk_tag_images(self):
    """
    Process all images that do not have tags. Returns a task ID that can be used to query progress.
    """
    from models import ImageDB
    from flask import current_app
    
    try:
        images = ImageDB.query.filter((ImageDB.tags == None) | (ImageDB.tags == "")).all()
        total = len(images)
        if total == 0:
            return None
        task_id = str(uuid.uuid4())
        BULK_PROGRESS[task_id] = 0
        for i, img in enumerate(images, start=1):
            process_image_tagging.delay(img.filename)
            BULK_PROGRESS[task_id] = (i / total) * 100
        return task_id
    except Exception as e:
        current_app.logger.error(f"Error in bulk_tag_images: {e}")
        return None

@celery.task(bind=True, time_limit=1800, soft_time_limit=1500)
def reembed_all_images(self):
    """
    Rerun tagging on all images using the currently selected models.
    This is useful when changing the tagging models or settings.
    """
    from models import ImageDB, UserConfig
    from flask import current_app
    import gc  # Garbage collection
    
    try:
        # Get user configuration
        config = UserConfig.query.first()
        if not config:
            from models import db
            config = UserConfig(ram_enabled=True, ram_model="ram_large")
            db.session.add(config)
            db.session.commit()
        
        # Determine which models will be used
        clip_model_name = config.clip_model if config and config.clip_model else "ViT-B-32"
        ram_model = config.ram_model if config and config.ram_model else "ram_large"
        ram_enabled = config.ram_enabled if config else True
        
        # Log the operation
        if ram_enabled:
            current_app.logger.info(f"Rerunning tagging on all images with RAM model: {ram_model} (screenshots will use CLIP: {clip_model_name})")
        else:
            current_app.logger.info(f"Rerunning tagging on all images with CLIP model: {clip_model_name}")
        
        # Get all images
        images = ImageDB.query.all()
        total = len(images)
        
        if total == 0:
            return {"status": "success", "message": "No images found to tag"}
        
        # Create a task ID for tracking progress
        task_id = str(uuid.uuid4())
        BULK_PROGRESS[task_id] = 0
        
        # First, clear all existing tags
        current_app.logger.info("Clearing all existing tags before retagging")
        from models import db  # Import db here to avoid 'db not defined' error
        for img in images:
            img.tags = ""
            img.description = ""
        db.session.commit()
        
        # Process images one at a time to manage memory better
        for i, img in enumerate(images):
            # Process one image and wait for it to complete before starting the next
            process_image_tagging.delay(img.filename)
            
            # Update progress
            BULK_PROGRESS[task_id] = min(100, (i + 1) / total * 100)
            
            # Add a small delay between tasks
            time.sleep(0.1)
            
        return {
            "status": "success",
            "message": f"Started retagging {total} images with appropriate models",
            "task_id": task_id
        }
    except Exception as e:
        current_app.logger.error(f"Error in reembed_all_images: {e}")
        # No need to clean up memory
        return {"status": "error", "message": str(e)}

def send_scheduled_image(event_id):
    """
    Send a scheduled image to a device.
    """
    # Import the app and create an application context
    # This is needed when the function is called directly by the scheduler
    import logging
    import asyncio
    logger = logging.getLogger(__name__)
    logger.info(f"Starting send_scheduled_image for event {event_id}")
    
    # Create a Flask app context to use current_app
    from app import app as flask_app
    with flask_app.app_context():
        from models import ScheduleEvent, Device, db, Screenshot
        from utils.crop_helpers import load_crop_info_from_db
        from utils.image_helpers import add_send_log_entry
        from flask import current_app
        from routes.browserless_routes import take_screenshot_with_puppeteer
        
        current_app.logger.info(f"Created Flask app context for event {event_id}")
        
        try:
            event = ScheduleEvent.query.get(event_id)
            if not event:
                current_app.logger.error("Event not found: %s", event_id)
                return
            
            current_app.logger.info(f"Found event: {event.id}, filename: {event.filename}, device: {event.device}")
            
            device_obj = Device.query.filter_by(address=event.device).first()
            if not device_obj:
                current_app.logger.error("Device not found for event %s", event_id)
                return
            
            current_app.logger.info(f"Found device: {device_obj.friendly_name}, address: {device_obj.address}")
    
            # Check if this is a screenshot that needs to be refreshed
            if event.refresh_screenshot:
                # Check if the filename exists in the screenshots table
                screenshot = Screenshot.query.filter_by(filename=event.filename).first()
                if screenshot:
                    current_app.logger.info(f"Refreshing screenshot {screenshot.name} before sending")
                    try:
                        # Get browserless config
                        from models import BrowserlessConfig
                        config = BrowserlessConfig.query.filter_by(active=True).first()
                        if not config:
                            current_app.logger.error("No active browserless configuration found")
                        else:
                            # Generate a new filename for the refreshed screenshot
                            timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
                            new_filename = f"screenshot_{timestamp}.jpg"
                            current_app.logger.info(f"New filename for refreshed screenshot: {new_filename}")
                            
                            # Create screenshots folder if it doesn't exist
                            screenshots_folder = os.path.join(current_app.config.get('DATA_FOLDER', './data'), 'screenshots')
                            if not os.path.exists(screenshots_folder):
                                os.makedirs(screenshots_folder)
                            
                            # Path for the new screenshot file
                            filepath = os.path.join(screenshots_folder, new_filename)
                            
                            # Take the new screenshot
                            asyncio.run(take_screenshot_with_puppeteer(
                                url=screenshot.url,
                                config=config,
                                filepath=filepath
                            ))
                            
                            current_app.logger.info(f"Screenshot refreshed successfully: {filepath}")
                            
                            # Copy crop info from old screenshot to new one
                            from models import ScreenshotCropInfo
                            old_crop_info = ScreenshotCropInfo.query.filter_by(filename=event.filename).first()
                            if old_crop_info:
                                current_app.logger.info(f"Copying crop info from {event.filename} to {new_filename}")
                                # Check if crop info already exists for the new filename
                                new_crop_info = ScreenshotCropInfo.query.filter_by(filename=new_filename).first()
                                if not new_crop_info:
                                    new_crop_info = ScreenshotCropInfo(filename=new_filename)
                                    db.session.add(new_crop_info)
                                
                                # Copy all crop data
                                new_crop_info.x = old_crop_info.x
                                new_crop_info.y = old_crop_info.y
                                new_crop_info.width = old_crop_info.width
                                new_crop_info.height = old_crop_info.height
                                new_crop_info.resolution = old_crop_info.resolution
                                db.session.commit()
                            
                            # Update the screenshot record in the database
                            screenshot.filename = new_filename
                            screenshot.last_updated = datetime.datetime.utcnow()
                            
                            # Update the event to use the new filename
                            event.filename = new_filename
                            db.session.commit()
                            
                            current_app.logger.info(f"Updated event to use new screenshot filename: {new_filename}")
                    except Exception as e:
                        current_app.logger.error(f"Error refreshing screenshot: {e}")
    
            image_folder = current_app.config.get("IMAGE_FOLDER", "./images")
            data_folder = current_app.config.get("DATA_FOLDER", "./data")
            screenshots_folder = os.path.join(data_folder, 'screenshots')
            
            # Check if the file is a screenshot or a regular image
            filepath = os.path.join(screenshots_folder, event.filename)
            if not os.path.exists(filepath):
                # Try in the regular images folder
                filepath = os.path.join(image_folder, event.filename)
                if not os.path.exists(filepath):
                    current_app.logger.error("Image file not found: %s", filepath)
                    return
    
            addr = device_obj.address
            if not (addr.startswith("http://") or addr.startswith("https://")):
                addr = "http://" + addr
    
            with Image.open(filepath) as orig_img:
                orig_w, orig_h = orig_img.size
                parts = device_obj.resolution.split("x")
                dev_width = int(parts[0])
                dev_height = int(parts[1])
                
                # Check if device is in portrait orientation
                is_portrait = device_obj.orientation.lower() == 'portrait'
                current_app.logger.info(f"Device orientation from database: '{device_obj.orientation}', is_portrait: {is_portrait}")
                
                # Calculate aspect ratio based on device orientation
                # This ratio is used for cropping to ensure the image fits the display correctly
                if is_portrait:
                    # For portrait displays, use height/width (taller than wide)
                    device_ratio = dev_height / dev_width
                    current_app.logger.info(f"Portrait display: using height/width ratio = {device_ratio}")
                else:
                    # For landscape displays, use width/height (wider than tall)
                    device_ratio = dev_width / dev_height
                    current_app.logger.info(f"Landscape display: using width/height ratio = {device_ratio}")
                
                # Log the original image dimensions and device info
                current_app.logger.info(f"Original image dimensions: {orig_w}x{orig_h}, device orientation: {device_obj.orientation}, device resolution: {device_obj.resolution}, device ratio: {device_ratio}")
                    
                # First check for ScreenshotCropInfo for screenshots
                from models import ScreenshotCropInfo
                screenshot_crop = None
                if event.filename.startswith("screenshot_"):
                    screenshot_crop = ScreenshotCropInfo.query.filter_by(filename=event.filename).first()
                    if screenshot_crop:
                        current_app.logger.info(f"Found ScreenshotCropInfo for {event.filename}")
                
                # If screenshot crop info exists, use it
                if screenshot_crop:
                    x = screenshot_crop.x
                    y = screenshot_crop.y
                    w = screenshot_crop.width
                    h = screenshot_crop.height
                    current_app.logger.info(f"Using screenshot crop data: x={x}, y={y}, w={w}, h={h}")
                    cropped = orig_img.crop((x, y, x + w, y + h))
                else:
                    # Otherwise check for regular CropInfo
                    cdata = load_crop_info_from_db(event.filename)
                    if cdata:
                        x = cdata.get("x", 0)
                        y = cdata.get("y", 0)
                        w = cdata.get("width", orig_w)
                        h = cdata.get("height", orig_h)
                        current_app.logger.info(f"Using regular crop data: x={x}, y={y}, w={w}, h={h}")
                        cropped = orig_img.crop((x, y, x + w, y + h))
                    else:
                        # If no crop data, create an auto-centered crop
                        current_app.logger.info("No crop data found, using auto-centered crop")
                        orig_ratio = orig_w / orig_h
                        
                        # Log the ratios for debugging
                        current_app.logger.info(f"Original image ratio: {orig_ratio}, device ratio: {device_ratio}")
                        
                        if orig_ratio > device_ratio:
                            # Image is wider than device ratio, use full height
                            new_width = int(orig_h * device_ratio)
                            left = (orig_w - new_width) // 2
                            crop_box = (left, 0, left + new_width, orig_h)
                        else:
                            # Image is taller than device ratio, use full width
                            new_height = int(orig_w / device_ratio)
                            top = (orig_h - new_height) // 2
                            crop_box = (0, top, orig_w, top + new_height)
                        
                        current_app.logger.info(f"Auto crop box: {crop_box}")
                        cropped = orig_img.crop(crop_box)
                        current_app.logger.info(f"Auto-cropped image dimensions: {cropped.size}")
                
                # Step 2: Resize and rotate the cropped image to match the target resolution and orientation
                current_app.logger.info(f"Cropped image size before resize/rotation: {cropped.size}")
                
                # IMPORTANT: The rotation is applied based on the device orientation in the database
                # If the eInk display itself is also rotating the image, this might cause double rotation
                
                # If portrait, rotate the image 90 degrees clockwise
                if is_portrait:
                    current_app.logger.info(f"Device is in PORTRAIT mode, rotating image 90° clockwise")
                    cropped = cropped.rotate(-90, expand=True)  # -90 for clockwise rotation
                    current_app.logger.info(f"After rotation size: {cropped.size}")
                    
                    # For portrait displays, we swap width and height in the final resize
                    # This is because the physical display is rotated, but the native resolution
                    # is still reported as if it were in landscape
                    current_app.logger.info(f"Swapping dimensions for portrait mode: {dev_width}x{dev_height} -> {dev_height}x{dev_width}")
                    final_img = cropped.resize((dev_height, dev_width), Image.LANCZOS)
                    current_app.logger.info(f"Final image size after portrait resize: {final_img.size}")
                else:
                    current_app.logger.info(f"Device is in LANDSCAPE mode, no rotation needed")
                    # For landscape displays, we use the normal dimensions
                    final_img = cropped.resize((dev_width, dev_height), Image.LANCZOS)
                    current_app.logger.info(f"Final image size after landscape resize: {final_img.size}")
                
                current_app.logger.info(f"Final image size: {final_img.size}, target device resolution: {device_obj.resolution}")
                temp_dir = os.path.join(data_folder, "temp")
                if not os.path.exists(temp_dir):
                    os.makedirs(temp_dir)
                # Create a unique temporary filename to avoid any caching issues
                import uuid
                unique_id = uuid.uuid4().hex[:8]
                temp_filename = os.path.join(temp_dir, f"temp_{unique_id}_{event.filename}")
                
                # Save the final image with high quality
                final_img.save(temp_filename, format="JPEG", quality=95)
                current_app.logger.info(f"Original image path: {filepath}")
                current_app.logger.info(f"Saved temporary file: {temp_filename}")
                current_app.logger.info(f"Final image dimensions being sent: {final_img.size}")
            
            # Verify the temporary file exists and has the correct dimensions
            try:
                with Image.open(temp_filename) as verify_img:
                    current_app.logger.info(f"Verifying temporary file: {temp_filename}, dimensions: {verify_img.size}")
            except Exception as e:
                current_app.logger.error(f"Error verifying temporary file: {e}")
            
            # Send the temporary file to the device using curl with verbose output
            cmd = f'curl -v "{addr}/send_image" -X POST -F "file=@{temp_filename}"'
            current_app.logger.info(f"Sending image with command: {cmd}")
            
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            
            # Log the curl response in detail
            current_app.logger.info(f"Curl stdout: {result.stdout}")
            current_app.logger.info(f"Curl stderr (includes request details): {result.stderr}")
            current_app.logger.info(f"Curl return code: {result.returncode}")
            
            # Delete the temporary file after sending
            os.remove(temp_filename)
            current_app.logger.info(f"Temporary file deleted: {temp_filename}")
            
            if result.returncode == 0:
                current_app.logger.info(f"Successfully sent image to device {device_obj.friendly_name}")
                event.sent = True
                db.session.commit()
                device_obj.last_sent = event.filename
                db.session.commit()
                add_send_log_entry(event.filename)
                current_app.logger.info(f"Updated database: event {event.id} marked as sent")
            else:
                current_app.logger.error(f"Error sending image: {result.stderr}")
        except Exception as e:
            current_app.logger.error("Error in send_scheduled_image: %s", e)
            return
    
        # Reschedule recurring events
        if event.recurrence and event.recurrence.lower() != "none":
            try:
                import pytz
                copenhagen_tz = pytz.timezone('Europe/Copenhagen')
                
                # Parse the datetime string
                dt = datetime.datetime.fromisoformat(event.datetime_str)
                
                # Ensure the datetime is in Copenhagen timezone
                if dt.tzinfo is None:
                    dt = copenhagen_tz.localize(dt)
                else:
                    dt = dt.astimezone(copenhagen_tz)
                    
            except Exception as e:
                current_app.logger.error("Error parsing datetime_str: %s", e)
                return
                
            if event.recurrence.lower() == "daily":
                next_dt = dt + datetime.timedelta(days=1)
            elif event.recurrence.lower() == "weekly":
                next_dt = dt + datetime.timedelta(weeks=1)
            elif event.recurrence.lower() == "monthly":
                next_dt = dt + datetime.timedelta(days=30)
            else:
                next_dt = None
                
            if next_dt:
                # Update the event in the database with the new date
                event.datetime_str = next_dt.isoformat()
                event.sent = False
                db.session.commit()
                
                # Note: We don't schedule the job here anymore
                # The dedicated scheduler process will pick up this event on its next run
                current_app.logger.info(f"Rescheduled event {event.id} for {next_dt}")

@celery.task(bind=True, ignore_result=True)
def fetch_device_metrics(self=None):
    """
    Check if devices are online and update their status in the database.
    This task runs periodically to ensure we have up-to-date status information.
    Logging has been minimized to prevent log flooding.
    
    Note: The self parameter is optional to allow this function to be called
    both as a Celery task and directly by the scheduler.
    """
    # Import the app and create an application context if needed
    # This is needed when the function is called directly by the scheduler
    import logging
    logger = logging.getLogger(__name__)
    
    # Create a Flask app context to use current_app
    from app import app as flask_app
    with flask_app.app_context():
        from models import Device, db
        from flask import current_app
        import httpx
        
        # Disable httpx logging
        logging.getLogger("httpx").setLevel(logging.WARNING)
    
        try:
            devices = Device.query.all()
            status_changes = 0
            
            for device in devices:
                try:
                    # Ensure the address has a scheme
                    address = device.address
                    if not (address.startswith("http://") or address.startswith("https://")):
                        address = "http://" + address
                    
                    # Store previous status to detect changes
                    was_online = device.online
                    
                    # Try to connect to the device's display_info endpoint
                    try:
                        # Use a short timeout to avoid blocking
                        response = httpx.get(f"{address}/display_info", timeout=5.0)
                        
                        if response.status_code == 200:
                            # Only update and log if status changed
                            if not was_online:
                                device.online = True
                                db.session.commit()
                                status_changes += 1
                        else:
                            # Only update and log if status changed
                            if was_online:
                                device.online = False
                                db.session.commit()
                                status_changes += 1
                    except Exception:
                        # Only update if status changed
                        if was_online:
                            device.online = False
                            db.session.commit()
                            status_changes += 1
                except Exception:
                    # Silently continue to next device
                    pass
                    
            # Only log if there were status changes
            if status_changes > 0:
                current_app.logger.info(f"Updated status for {status_changes} devices")
                
            return {"status": "success", "message": f"Checked status of {len(devices)} devices"}
        except Exception:
            # Catch all exceptions to prevent task failures
            return {"status": "error", "message": "Error checking device status"}

# The scheduler functionality has been moved to scheduler.py
# The start_scheduler function has been removed as it's no longer needed
