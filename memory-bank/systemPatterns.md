# System Patterns *Optional*

This file documents recurring patterns and standards used in the project.
It is optional, but recommended to be updated as the project evolves.
2025-03-29 11:15:56 - Log of updates made.

*

## Coding Patterns

*   

## Architectural Patterns

*   

## Testing Patterns


*[2025-03-29 11:24:54] - Added core architectural patterns based on app.py analysis.*
- **Flask:** Microframework used for the web application core.
- **SQLAlchemy:** ORM used for database interaction.
- **Celery:** Distributed task queue used for background processing (e.g., image tagging, sending to devices).
- **Flask Blueprints:** Used to modularize application routes (`routes/` directory).
## Architectural Patterns

*   

## Testing Patterns

*[2025-03-29 11:53:45] - Added database model patterns and entities based on models.py analysis.*

- **SQLAlchemy Model Patterns:**
  - Composite Primary Keys used (e.g., `CropInfo`, `ScreenshotCropInfo`).
  - JSON storage for complex data (e.g., `ScheduleEvent.recurrence_details`).
  - Lack of explicitly defined `db.relationship` for foreign keys.
- **Core Data Entities:** `Device`, `ImageDB`, `Screenshot`, `ScheduleEvent`, `CropInfo`, `ScreenshotCropInfo`, `UserConfig`, `BrowserlessConfig`, `Notification`, `SendLog`.
## Testing Patterns


*[2025-03-29 11:24:54] - Added core architectural patterns based on app.py analysis.*
- **Flask:** Microframework used for the web application core.
- **SQLAlchemy:** ORM used for database interaction.
- **Celery:** Distributed task queue used for background processing (e.g., image tagging, sending to devices).
- **Flask Blueprints:** Used to modularize application routes (`routes/` directory).
## Architectural Patterns

*   

## Testing Patterns

*