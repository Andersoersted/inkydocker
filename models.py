from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import json

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
    recurrence_details = db.Column(db.Text, nullable=True)  # JSON field for recurrence details
    refresh_screenshot = db.Column(db.Boolean, default=False)  # Whether to refresh screenshot before sending
    created_at = db.Column(db.DateTime, default=datetime.utcnow)  # When the event was created

    def __repr__(self):
        return f"<ScheduleEvent {self.filename} on {self.device}>"
    
    def get_recurrence_details(self):
        """Parse and return the recurrence details as a dictionary"""
        if not self.recurrence_details:
            return {}
        try:
            return json.loads(self.recurrence_details)
        except:
            return {}
    
    def set_recurrence_details(self, details):
        """Set recurrence details from a dictionary"""
        if details:
            self.recurrence_details = json.dumps(details)
        else:
            self.recurrence_details = None

class UserConfig(db.Model):
    __tablename__ = 'user_config'
    id = db.Column(db.Integer, primary_key=True)
    location = db.Column(db.String(256))
    
    # CLIP model settings (kept for screenshot processing)
    clip_model = db.Column(db.String(64), default="ViT-B-32")  # Column for chosen CLIP model
    min_tags = db.Column(db.Integer, default=5)  # Maximum number of tags to generate for images
    custom_model = db.Column(db.String(256), nullable=True)  # Custom model name
    similarity_threshold = db.Column(db.Float, default=0.2)  # Similarity threshold for tag selection
    
    # Zero Shot settings
    zero_shot_enabled = db.Column(db.Boolean, default=True)  # Enable zero shot tagging
    zero_shot_model = db.Column(db.String(256), default="facebook/bart-large-mnli")  # Zero shot model name
    zero_shot_min_confidence = db.Column(db.Float, default=0.5)  # Confidence threshold for zero shot classification
    ram_model = db.Column(db.String(256), default="facebook/ram-14b")  # RAM model name
