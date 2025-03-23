# Multi-Device Crop Implementation Notes

This document provides technical details about the implementation of multi-device cropping functionality in the InkyDocker application.

## Overview

The multi-device cropping feature allows users to define different crop settings for the same image or screenshot for different e-ink displays. Each display can have its own crop configuration tailored to its specific resolution and aspect ratio.

## Database Changes

1. Added `device_address` column to both `crop_info` and `screenshot_crop_info` tables
2. Made `device_address` part of the composite primary key alongside `filename` to support multiple crop settings per image
3. Added indices for efficient lookups based on filename and device combinations
4. Set default value of 'default_device' for existing crop records to maintain backward compatibility

## Backend API Changes

### Image Routes
1. Updated `/api/get_crop_info/<filename>` endpoint to:
   - Accept an optional `device` query parameter
   - Return available device crops for a given image
   - Include device_address in the response
   - Fall back to default device crop if the requested device crop doesn't exist

2. Updated `save_crop_info` endpoint to:
   - Store the device address alongside crop coordinates
   - Replace only that device's crop instead of all crops for an image

3. Updated `send_image` function to:
   - Fetch the device-specific crop information
   - Apply appropriate crop and resize based on the selected device's configuration

### Browserless Routes
1. Updated `get_screenshot_crop_info` endpoint with similar device-specific capabilities
2. Modified `save_screenshot_crop_info` to handle device-specific crops
3. Adapted `get_screenshot` and `send_screenshot` functions to utilize device-specific crops
4. Updated error handling and logging to include device information

### Utilities
1. Enhanced `load_crop_info_from_db` function in `crop_helpers.py` to:
   - Accept an optional device_address parameter
   - Fetch device-specific crop data with proper fallback mechanism
   - Return device address in the result

2. Modified `save_crop_info_to_db` to:
   - Maintain device-specific crop records
   - Handle device resolution information correctly

## Scheduler Enhancement

1. Updated `send_scheduled_image` function to:
   - Support refreshing screenshots while maintaining device-specific crop information
   - Copy all device-specific crop configurations when refreshing a screenshot
   - Properly use device-specific crops when sending images on schedule

## Frontend Changes

1. Enhanced crop modal to:
   - Display device selection information
   - Enforce the correct aspect ratio based on device orientation and resolution
   - Show which device is being cropped for

2. Added UI for device-specific crop management:
   - Display available crops for different devices
   - Allow switching between devices to see their specific crop settings

3. Added refresh functionality:
   - Added "Refresh" button for screenshots to capture a fresh version
   - Implemented logic to apply saved crop settings to refreshed screenshots

4. Updated crop previews to:
   - Show the correct aspect ratio guides based on device
   - Display device-specific crop information when available

## Key Implementation Considerations

1. **Aspect Ratio Enforcement**: 
   - The cropping tool enforces the correct aspect ratio based on the device's resolution and orientation
   - For portrait displays, the physical display is rotated 90°, so width and height are swapped for aspect ratio calculation

2. **Backward Compatibility**:
   - Existing crop records are assigned a default device address
   - System falls back to 'default_device' crops when device-specific crops aren't available

3. **Send vs. Refresh Behavior**:
   - Send: Uses the stored crop info for the selected device to process the current image
   - Refresh: Captures a new screenshot, then applies the previously stored crop settings

4. **Docker Compatibility**:
   - All changes are compatible with the Docker deployment
   - Database migration is included to update schema automatically

## Testing Recommendations

1. Test creating crops for different devices for the same image
2. Verify that each device properly uses its specific crop settings
3. Test refresh functionality to ensure crop settings are properly transferred
4. Verify scheduled events with refreshed screenshots maintain proper cropping
5. Test aspect ratio enforcement for different device orientations