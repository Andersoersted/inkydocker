from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()

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

    def __repr__(self):
        return f"<ScheduleEvent {self.filename} on {self.device}>"

class UserConfig(db.Model):
    __tablename__ = 'user_config'
    id = db.Column(db.Integer, primary_key=True)
    location = db.Column(db.String(256))
    openai_api_key = db.Column(db.String(512))
    ollama_address = db.Column(db.String(256))
    ollama_api_key = db.Column(db.String(512))
    ollama_model = db.Column(db.String(64))  # Column for chosen Ollama model
    clip_model = db.Column(db.String(64), default="ViT-B-32")  # Column for chosen CLIP model (using consistent format with tasks.py)

    def __repr__(self):
        return f"<UserConfig {self.id} - Location: {self.location}>"

# DeviceMetrics model removed - we only track online status now
