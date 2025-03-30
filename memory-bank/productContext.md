# Product Context

This file provides a high-level overview of the project and the expected product that will be created. Initially it is based upon projectBrief.md (if provided) and all other available project-related information in the working directory. This file is intended to be updated as the project evolves, and should be used to inform all other modes of the project's goals and context.
2025-03-29 11:11:55 - Log of updates made will be appended as footnotes to the end of this file.

*

## Project Goal

*   

## Key Features

*   

## Overall Architecture


*[2025-03-29 11:19:06] - Initial project summary added based on README.md.*
Manage and display images on multiple e-ink devices, including features like AI tagging, website screenshot capture, and scheduling.

## Key Features

*   
- Image gallery & management (upload, search, favorites)
- AI-powered tagging (CLIP, Zero-Shot)
- E-Ink display integration (multi-device, status, optimization, direct send)
- Website screenshot capture (scheduled refresh, cropping)
- Scheduling system (timed, recurring, calendar view)
- Image processing (cropping, conversion, optimization)

## Overall Architecture

*   - **Framework:** Flask Web Application (served via Gunicorn)
- **Database:** SQLAlchemy ORM with Flask-Migrate (Models: Device, ImageDB, Screenshot, ScheduleEvent, CropInfo, Config, etc.)
- **Task Queue:** Celery with Redis broker/backend
- **Scheduling:** APScheduler (potentially for triggering Celery tasks)
- **AI/ML:** PyTorch, Transformers, Scikit-learn
- **Screenshotting:** Pyppeteer
- **Routing:** Modular structure using Flask Blueprints (`routes/`)
- **Configuration:** Centralized (`config.py`)
- **Image Handling:** Pillow with HEIF support
- **Deployment:** Docker / Docker Compose