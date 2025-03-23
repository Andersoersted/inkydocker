# Simplified Database Setup for Device-Specific Crop Functionality

This document explains the approach taken to implement device-specific crop functionality in the database.

## Original Problem

We encountered migration issues when attempting to implement device-specific crop functionality with Alembic migrations, particularly with fresh database installations. The migration errors occurred due to missing references and inconsistent naming conventions.

## Simplified Solution

Since the application is still in active development and there's no production data to preserve, we've adopted a simpler approach:

### Drop and Recreate the Database on Startup

We've modified the `entrypoint.sh` script to:
1. Drop all existing tables
2. Create all tables from scratch based on the current model definitions

```bash
python -c "from app import app; from models import db; app.app_context().push(); db.drop_all(); db.create_all()"
```

This approach:
- Eliminates the need for complex migrations
- Ensures the database schema always matches the latest model definitions
- Simplifies development by removing migration dependencies
- Prevents errors when starting with a fresh database

### Database Model Changes

The multi-device crop functionality is implemented through the following model changes:

1. Added `device_address` column to `CropInfo` table:
   ```python
   device_address = db.Column(db.String(256), primary_key=True)
   ```

2. Added `device_address` column to `ScreenshotCropInfo` table:
   ```python
   device_address = db.Column(db.String(256), primary_key=True)
   ```

3. Made both columns part of the composite primary key alongside `filename`

## Implementation Details

This simplified approach:
- Creates all necessary tables with the correct structure
- Ensures the device_address column exists in crop tables
- Sets up proper composite primary keys for device-specific crops
- Avoids complex migration chains and dependencies

## Development Note

This approach is appropriate during active development but would need to be
revisited before moving to production where database migrations would be needed
to preserve existing data.