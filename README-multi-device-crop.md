# Multi-Device Cropping Functionality

This document explains the enhanced cropping functionality that supports multiple e-ink displays, each with its own resolution and aspect ratio.

## Overview

The system now supports saving and applying device-specific crops for both regular images and screenshots. This means each image can have different crop configurations for different e-ink displays, and the system will automatically use the appropriate crop when sending to a specific device.

## Key Features

1. **Device-Specific Crops**: Each image can have multiple crop configurations, one for each e-ink display.
2. **Aspect Ratio Enforcement**: When cropping for a specific device, the crop interface enforces that device's aspect ratio.
3. **Fallback Mechanism**: If a crop hasn't been defined for a specific device, the system falls back to a default crop.
4. **Refresh Before Send**: For screenshots, users can refresh the screenshot before sending, which captures a new screenshot and applies the stored crop information.

## Implementation Details

### Database Changes

- Added `device_address` column to both `crop_info` and `screenshot_crop_info` tables
- Created composite primary keys using (`filename`, `device_address`) to support multiple crops per image
- Added appropriate indices for efficient lookups

### Backend API Changes

- Modified `/api/get_crop_info/<filename>` to accept an optional `device` query parameter
- Updated `/api/get_screenshot_crop_info/<filename>` to support device-specific crops
- Enhanced `load_crop_info_from_db` function to fetch device-specific crops with proper fallback
- Updated `send_image` and `send_screenshot` functions to use the appropriate device-specific crop

### Frontend Changes

- Added UI for displaying available device-specific crops for an image
- Enhanced crop modal to show which device is being cropped for
- Added support for enforcing device-specific aspect ratios in the crop interface
- Implemented a refresh button for screenshots to capture a new screenshot before sending

## Usage Flow

1. **Setting Up Device-Specific Crops**:
   - Select a device from the device list
   - Open an image and click "Crop Image"
   - The crop interface will enforce the aspect ratio for the selected device
   - Save the crop configuration, which will be associated with the selected device

2. **Sending Images to Devices**:
   - When sending an image to a device, the system will automatically use the crop configuration specific to that device
   - If no device-specific crop exists, it will fall back to a default crop

3. **Refreshing Screenshots**:
   - For screenshots, users can click the "Refresh" button in the crop modal
   - This will capture a new screenshot of the webpage
   - Then apply the existing crop configuration to the new screenshot

## Technical Notes

- The aspect ratio calculation takes into account the device's orientation (portrait vs. landscape)
- For portrait displays, the physical display is rotated 90°, so the width and height are swapped when calculating the aspect ratio
- The crop interface provides visual cues about the device's orientation and resolution
- All crop coordinates are stored in relation to the original full-size image, not the displayed preview

## Docker Deployment

The system includes a migration file (`add_device_specific_crop_info.py`) that will automatically update the database schema when the Docker container starts. No manual intervention is required for the database changes.