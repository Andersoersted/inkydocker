# Notification-Triggering Actions

## Image Operations
1. **Image Sending**
   - When image sending process is initiated - INFO notification
   - When image is successfully sent - SUCCESS notification
   - When image sending fails - ERROR notification

2. **Image Capture/Screenshot**
   - When screenshot capture is initiated - INFO notification
   - When screenshot is successfully captured - SUCCESS notification
   - When screenshot capture fails - ERROR notification

3. **Image Processing**
   - When crop or resize operation starts - INFO notification
   - When crop or resize operation completes - SUCCESS notification
   - When crop or resize operation fails - ERROR notification

## Scheduled Events
1. **Event Creation/Deletion**
   - When a new scheduled event is created - SUCCESS notification
   - When a scheduled event is deleted - INFO notification

2. **Event Execution**
   - When a scheduled event execution starts - INFO notification
   - When execution completes successfully - SUCCESS notification
   - When execution fails - ERROR notification

## System Operations
1. **Refresh Actions**
   - When a manual refresh is initiated - INFO notification
   - When a refresh is completed - SUCCESS notification

2. **Settings Updates**
   - When system settings are updated - SUCCESS notification
   - When settings update fails - ERROR notification