# Multi-Device Crop Feature Implementation

This feature allows storing separate crop coordinates for each e-ink display, enforcing appropriate aspect ratios for different device orientations and resolutions.

## Key Implementation Components

### Database

- Added `device_address` column to `CropInfo` and `ScreenshotCropInfo` tables
- Created composite primary keys to support multiple crops per image
- Simplified database setup by dropping and recreating all tables on Docker startup

### Backend

1. **Image Routes**
   - Updated `/api/get_crop_info` to accept a device parameter
   - Modified crop saving to maintain separate crop data per device
   - Enhanced send_image to use device-specific crops

2. **Browserless Routes**
   - Updated screenshot-related endpoints to support device-specific crops
   - Improved crop coordinate handling for different display resolutions

3. **Scheduling**
   - Enhanced refresh functionality to maintain device-specific crops
   - Ensured recurring events use correct device-specific crop data

4. **Utils**
   - Updated crop_helpers.py to handle device-specific crop storage and retrieval
   - Added proper fallback logic to default device when needed

### Frontend

1. **Crop Interface**
   - Enforces correct aspect ratio based on the device's orientation and resolution
   - Shows available crops for different devices
   - Provides refresh functionality for screenshots

2. **User Experience**
   - Displays device information during cropping
   - Handles device-specific preview rendering
   - Ensures crop boxes maintain proper device aspect ratios

## Code Changes

The main files modified:

- `models.py`: Added device_address to crop models
- `routes/image_routes.py`: Updated endpoints to handle device-specific crops
- `routes/browserless_routes.py`: Enhanced screenshot handling for multiple devices
- `tasks.py`: Updated scheduled sending to support device-specific crops
- `static/device_crop.js`: Added new frontend functionality for device selection
- `templates/index.html`: Updated crop UI to show device information
- `entrypoint.sh`: Simplified database initialization for development

## Usage

1. When cropping an image, the crop box will automatically enforce the aspect ratio of the selected device
2. Each device can have its own crop of the same image
3. When sending an image, the system uses the device-specific crop
4. Screenshots can be refreshed while maintaining the original crop coordinates

## Considerations

For portrait displays (where height > width), the system automatically handles the rotation and aspect ratio to ensure proper display on the e-ink device.