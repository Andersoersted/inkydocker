from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()

class BrowserlessConfig(db.Model):
    __tablename__ = 'browserless_config'
    id = db.Column(db.Integer, primary_key=True)
    address = db.Column(db.String(256), nullable=False)
    port = db.Column(db.Integer, nullable=False)
    token = db.Column(db.String(256), nullable=True)
    active = db.Column(db.Boolean, default=True)

class Screenshot(db.Model):
    __tablename__ = 'screenshots'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(256), nullable=False)
    url = db.Column(db.String(1024), nullable=False)
    filename = db.Column(db.String(256), unique=True, nullable=False)
    last_updated = db.Column(db.DateTime, default=datetime.utcnow)

class ScreenshotCropInfo(db.Model):
    __tablename__ = 'screenshot_crop_info'
    filename = db.Column(db.String(256), primary_key=True)
    x = db.Column(db.Float, default=0)
    y = db.Column(db.Float, default=0)
    width = db.Column(db.Float)
    height = db.Column(db.Float)
    resolution = db.Column(db.String(32))  # Store the display resolution (e.g., "1024x768")

class Device(db.Model):
    __tablename__ = 'devices'
    id = db.Column(db.Integer, primary_key=True)
    color = db.Column(db.String(16), nullable=False)
    friendly_name = db.Column(db.String(128), nullable=False)
    orientation = db.Column(db.String(32), nullable=False)
    address = db.Column(db.String(256), nullable=False)
    display_name = db.Column(db.String(128))
    resolution = db.Column(db.String(32))
    online = db.Column(db.Boolean, default=False)
    last_sent = db.Column(db.String(256))

    def __repr__(self):
        return f"<Device {self.friendly_name} ({self.address})>"

class ImageDB(db.Model):
    __tablename__ = 'images'
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(256), unique=True, nullable=False)
    tags = db.Column(db.String(512), nullable=True)         # comma-separated tags
    description = db.Column(db.Text, nullable=True)           # description text
    favorite = db.Column(db.Boolean, default=False)           # favorite flag

    def __repr__(self):
        return f"<ImageDB {self.filename}>"

class CropInfo(db.Model):
    __tablename__ = 'crop_info'
    filename = db.Column(db.String(256), primary_key=True)
    x = db.Column(db.Float, default=0)
    y = db.Column(db.Float, default=0)
    width = db.Column(db.Float)
    height = db.Column(db.Float)
    resolution = db.Column(db.String(32))  # Store the display resolution (e.g., "1024x768")
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self):
        return f"<CropInfo {self.filename}>"

class SendLog(db.Model):
    __tablename__ = 'send_log'
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(256), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<SendLog {self.filename} {self.timestamp}>"

class ScheduleEvent(db.Model):
    __tablename__ = 'schedule_events'
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(256), nullable=False)
    device = db.Column(db.String(256), nullable=False)
    datetime_str = db.Column(db.String(32))
    sent = db.Column(db.Boolean, default=False)
    recurrence = db.Column(db.String(20), default="none")  # Recurrence type
    refresh_screenshot = db.Column(db.Boolean, default=False)  # Whether to refresh screenshot before sending

    def __repr__(self):
        return f"<ScheduleEvent {self.filename} on {self.device}>"

class UserConfig(db.Model):
    __tablename__ = 'user_config'
    id = db.Column(db.Integer, primary_key=True)
    location = db.Column(db.String(256))
    
    # CLIP model settings (kept for screenshot processing)
    clip_model = db.Column(db.String(64), default="ViT-B-32")  # Column for chosen CLIP model (using consistent format with tasks.py)
    min_tags = db.Column(db.Integer, default=5)  # Maximum number of tags to generate for images (with similarity threshold)
    custom_model = db.Column(db.String(256), nullable=True)  # Custom model name (e.g., "openai/clip-vit-base-patch32")
    custom_model_enabled = db.Column(db.Boolean, default=False)  # Whether to use the custom model
    similarity_threshold = db.Column(db.String(20), default="medium")  # Similarity threshold level (very_high, high, medium, low, very_low)
    
    # Zero-shot model settings (for image tagging)
    zero_shot_enabled = db.Column(db.Boolean, default=True)  # Whether to use zero-shot for image tagging (default to True)
    zero_shot_model = db.Column(db.String(64), default="base")  # Selected zero-shot model (base or large)
    zero_shot_min_confidence = db.Column(db.Float, default=0.1)  # Minimum confidence threshold for zero-shot tags

    def __repr__(self):
        return f"<UserConfig {self.id} - Location: {self.location}>"

# DeviceMetrics model removed - we only track online status now
