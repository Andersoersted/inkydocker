from flask import Flask
import os
from config import Config
from models import db
import pillow_heif
from tasks import celery, start_scheduler

# Register HEIF opener for Pillow
pillow_heif.register_heif_opener()

def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)

    # Ensure required folders exist
    for folder in [app.config['IMAGE_FOLDER'], app.config['THUMBNAIL_FOLDER'], app.config['DATA_FOLDER']]:
        if not os.path.exists(folder):
            os.makedirs(folder)

    # Initialize database without migrations
    db.init_app(app)

    # Register blueprints
    from routes.image_routes import image_bp
    from routes.device_routes import device_bp
    from routes.schedule_routes import schedule_bp
    from routes.settings_routes import settings_bp
    from routes.device_info_routes import device_info_bp
    from routes.ai_tagging_routes import ai_bp

    app.register_blueprint(image_bp)
    app.register_blueprint(device_bp)
    app.register_blueprint(schedule_bp)
    app.register_blueprint(settings_bp)
    app.register_blueprint(device_info_bp)
    app.register_blueprint(ai_bp)

    # Create database tables if they don't exist.
    with app.app_context():
        db.create_all()

    # Configure Celery
    celery.conf.update(
        broker_url='redis://localhost:6379/0',
        result_backend='redis://localhost:6379/0'
    )

    # Note: Scheduler is now started in a dedicated process (scheduler.py)
    # We still call the function for backward compatibility, but it doesn't start the scheduler
    start_scheduler(app)
    
    # We no longer run fetch_device_metrics immediately here
    # The dedicated scheduler process will handle this

    return app

app = create_app()

# Make the app available to Celery tasks
celery.conf.update(app=app)

if __name__ == '__main__':
    # When running via 'python app.py' this block will execute.
    app.run(host='0.0.0.0', port=5001, debug=True)
