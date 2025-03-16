# Suppress warnings
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*torch.distributed.reduce_op.*")

from flask import Flask, send_from_directory, request
import os
import multiprocessing
import logging
from config import Config
from models import db
from flask_migrate import Migrate
import pillow_heif

# Set multiprocessing start method to 'spawn' to fix CUDA issues
try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    # Method already set, ignore
    pass

# Import celery after setting multiprocessing start method
from tasks import celery

# Register HEIF opener for Pillow
pillow_heif.register_heif_opener()

def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)
    
    # Configure Flask logging to reduce verbosity
    app.logger.setLevel(logging.WARNING)
    logging.getLogger('werkzeug').setLevel(logging.WARNING)

    # Ensure required folders exist
    for folder in [app.config['IMAGE_FOLDER'], app.config['THUMBNAIL_FOLDER'], app.config['DATA_FOLDER']]:
        if not os.path.exists(folder):
            os.makedirs(folder)

    # Initialize database with migrations
    db.init_app(app)
    migrate = Migrate(app, db)

    # Register blueprints
    from routes.image_routes import image_bp
    from routes.device_routes import device_bp
    from routes.schedule_routes import schedule_bp
    from routes.settings_routes import settings_bp
    from routes.device_info_routes import device_info_bp
    from routes.ai_tagging_routes import ai_bp
    from routes.browserless_routes import browserless_bp
    from routes.settings_routes_zero_shot import zero_shot_bp

    app.register_blueprint(image_bp)
    app.register_blueprint(device_bp)
    app.register_blueprint(schedule_bp)
    app.register_blueprint(settings_bp)
    app.register_blueprint(device_info_bp)
    app.register_blueprint(ai_bp)
    app.register_blueprint(browserless_bp)
    app.register_blueprint(zero_shot_bp)

    # Create database tables if they don't exist
    with app.app_context():
        db.create_all()

    # Configure Celery with Redis settings
    celery.conf.update(
        broker_url='redis://localhost:6379/0',
        result_backend='redis://localhost:6379/0',
        broker_connection_retry_on_startup=True,
        result_expires=3600,  # Results expire after 1 hour
        task_acks_late=True  # Only acknowledge tasks after they are completed
    )

    # Add cache control for static files
    @app.after_request
    def add_cache_headers(response):
        if request.path.startswith('/static/'):
            response.cache_control.max_age = 604800  # 1 week in seconds
            response.cache_control.public = True
        return response

    # Serve static files with cache headers
    @app.route('/static/<path:filename>')
    def serve_static(filename):
        response = send_from_directory(app.static_folder, filename)
        response.cache_control.max_age = 604800  # 1 week in seconds
        response.cache_control.public = True
        return response

    return app

app = create_app()

# Make the app available to Celery tasks
celery.conf.update(app=app)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=False, threaded=True)
