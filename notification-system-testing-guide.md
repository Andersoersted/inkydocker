# Notification System Testing Guide

This guide provides instructions for testing the new centralized notification system that has been implemented to replace all spinners and progress bars with unified, persistent notifications.

## What Was Implemented

1. **Notification Database Model**
   - Added `Notification` model to store all notifications with 30-day expiry
   - Notifications include: message, type, read status, created/expiry timestamps

2. **Notification API**
   - Added RESTful endpoints for notification CRUD operations
   - Endpoints handle creating, retrieving, marking as read, and deleting notifications

3. **Notification Service**
   - Created a centralized service for consistent notification creation
   - Added helper functions and decorators for common notification patterns

4. **Unified UI Components**
   - Integrated Bootstrap 5 Toasts for real-time notifications
   - Created a notification center accessible via bell icon
   - Added unread notification counter

5. **Live Notification Enhancements**
   - Created specialized code to integrate with existing UI elements
   - Added progress reporting for uploads
   - Added status updates for image sending
   - Added notifications for scheduled events

## How to Test

### 1. Image Upload Notifications

1. Go to the Gallery page
2. Click "Choose File" and select one or more images
3. Click "Upload"
4. **Expected behavior:**
   - You should see "Starting image upload..." notification
   - Progress notifications should appear as upload progresses (every 10%)
   - A success notification should appear when upload is complete
   - The notifications should be visible in the notification center (bell icon)

### 2. Image Sending Notifications

1. Go to the Gallery page
2. Select a device using the radio buttons
3. Click "Send" on any image
4. **Expected behavior:**
   - You should see "Sending image..." notification
   - The Send button should still show a spinner (backward compatibility)
   - A success notification should appear when send is complete
   - The notifications should be visible in the notification center

### 3. Screenshot Sending

1. Go to the Gallery page
2. Select a device
3. Click "Send" on a screenshot
4. **Expected behavior:**
   - You should see "Sending screenshot..." notification
   - A success notification should appear when send is complete

### 4. Scheduled Events

1. Go to the Schedule page
2. Create a new scheduled event
3. **Expected behavior:**
   - You should see "Creating scheduled event..." notification
   - A success notification should appear when event is created
   - When the event executes, notifications for start and completion should appear

### 5. E-Ink Display Addition

1. Go to Settings
2. Add a new E-Ink display
3. **Expected behavior:**
   - You should see a notification about the display addition
   - Only one notification should appear (no duplicates)

### 6. Notification Center

1. Click the bell icon in the navigation bar
2. **Expected behavior:**
   - All recent notifications should be listed
   - Unread notifications should be highlighted
   - You can mark notifications as read individually or all at once

### 7. Error Handling

1. Try to trigger errors (e.g., send to an offline device)
2. **Expected behavior:**
   - Error notifications should appear
   - The error notifications should persist in the notification center

## Known Limitations

1. The notification system uses polling rather than WebSockets, so there may be a delay of up to 30 seconds before notifications appear in the notification center when triggered from another device or server-side process.

2. For backward compatibility, some original UI elements (spinners, progress bars) are maintained alongside the new notification system. These will be phased out in a future update once the notification system is proven reliable.

## Docker Testing Instructions

Since the application runs in a Docker container, testing requires rebuilding and redeploying:

```bash
# Rebuild the Docker image
docker-compose build

# Restart the container
docker-compose down
docker-compose up -d

# Check logs for any errors
docker-compose logs -f
```

The notification system is designed to work with a fresh database, so no migration scripts were created. When you start the app without an existing database, the necessary tables will be created automatically.