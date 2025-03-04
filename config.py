import os

basedir = os.path.abspath(os.path.dirname(__file__))

class Config:
    SECRET_KEY = "super-secret-key"
    # Database: using an absolute path in a “data” folder in the project directory.
    # In the container, basedir will be /app so the DB will be at /app/data/mydb.sqlite.
    SQLALCHEMY_DATABASE_URI = 'sqlite:///' + os.path.join(basedir, 'data', 'mydb.sqlite')
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # Folders for images, thumbnails, and data storage
    IMAGE_FOLDER = os.path.join(basedir, 'images')
    THUMBNAIL_FOLDER = os.path.join(basedir, 'images', 'thumbnails')
    DATA_FOLDER = os.path.join(basedir, 'data')