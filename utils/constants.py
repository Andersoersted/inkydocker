"""
Application constants for InkyDocker.
Contains candidate tags, similarity thresholds, and other configuration values.
"""

# Define candidate tags with focus on objects and content
CANDIDATE_TAGS = [
    # People and portraits
    "person", "people", "man", "woman", "child", "children", "baby", "group of people",
    "crowd", "portrait", "selfie", "face",

    # Objects
    "car", "vehicle", "bicycle", "motorcycle", "airplane", "boat", "building", "house",
    "furniture", "chair", "table", "bed", "computer", "phone", "television", "book",
    "clock", "bottle", "cup", "plate", "food", "fruit", "vegetable", "meal",

    # Animals
    "animal", "dog", "cat", "bird", "fish", "horse", "cow", "sheep", "wildlife", "pet",

    # Nature
    "tree", "flower", "plant", "mountain", "river", "lake", "ocean", "beach", "forest",
    "sky", "cloud", "sun", "moon", "stars",

    # Environments
    "city", "urban", "rural", "indoor", "outdoor", "street", "park", "garden", "office",
    "home", "kitchen", "bedroom", "bathroom",

    # Activities
    "walking", "running", "swimming", "eating", "drinking", "reading", "writing",
    "working", "playing", "dancing", "singing",

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
