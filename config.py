"""
Configuration settings for InkyDocker application.
Supports environment variable overrides for flexible deployment.
"""
import os
import secrets

basedir = os.path.abspath(os.path.dirname(__file__))


class Config:
    """Base configuration with sensible defaults."""

    # Security
    # Generate a strong random key if not provided via environment variable
    SECRET_KEY = os.environ.get('SECRET_KEY') or secrets.token_hex(32)

    # CSRF Protection
    WTF_CSRF_ENABLED = True
    WTF_CSRF_TIME_LIMIT = None  # CSRF tokens don't expire

    # File Upload Security
    MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50MB max file size

    # Session Security
    SESSION_COOKIE_SECURE = os.environ.get('SESSION_COOKIE_SECURE', 'False') == 'True'  # Set True in production with HTTPS
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Lax'

    # Database configuration
    # In the container, basedir will be /app so the DB will be at /app/data/mydb.sqlite
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or \
        'sqlite:///' + os.path.join(basedir, 'data', 'mydb.sqlite')
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # Folder paths for images, thumbnails, and data storage
    IMAGE_FOLDER = os.path.join(basedir, 'images')
    THUMBNAIL_FOLDER = os.path.join(basedir, 'images', 'thumbnails')
    DATA_FOLDER = os.path.join(basedir, 'data')
    SCREENSHOTS_FOLDER = os.path.join(basedir, 'data', 'screenshots')

    # Redis configuration for Celery
    REDIS_URL = os.environ.get('REDIS_URL') or 'redis://localhost:6379/0'

    # Application settings
    TIMEZONE = os.environ.get('TZ') or 'Europe/Copenhagen'