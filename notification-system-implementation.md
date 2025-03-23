# Notification System Implementation

This document details the centralized notification system that has been implemented for InkyDocker. The system provides both real-time and persistent notifications for all significant operations.

## Overview

The notification system consists of several components:

1. **Database Layer**: Notifications are stored in a dedicated table with automatic 30-day expiry
2. **Backend API**: RESTful endpoints for creating, retrieving, and managing notifications
3. **Notification Service**: A centralized service for creating notifications from any part of the application
4. **UI Components**: Bootstrap 5 Toasts for real-time notifications and a notification center for persistent ones

## Features

- **Real-Time Notifications**: Pop-up toast notifications for active users
- **Persistent Storage**: All notifications are stored in the database
- **Notification Center**: Access all notifications via a bell icon in the navigation bar
- **Notification Types**: Support for info, success, warning, and error notifications
- **Automatic Expiry**: Notifications automatically expire after 30 days
- **Mark as Read**: Users can mark notifications as read individually or all at once
- **Badge Counter**: Unread notification count displayed on the bell icon

## Database Schema

The `Notification` model includes:

- `id`: Primary key
- `message`: Notification message
- `type`: Notification type (info, success, warning, error)
- `is_read`: Boolean for read status
- `created_at`: Creation timestamp
- `expires_at`: Expiration timestamp (30 days after creation)

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/notifications` | GET | Get all non-expired notifications |
| `/api/notifications` | POST | Create a new notification |
| `/api/notifications/mark-read` | POST | Mark notification(s) as read |
| `/api/notifications/delete-expired` | POST | Delete all expired notifications |
| `/api/notifications/<id>` | DELETE | Delete a specific notification |

## Integration Points

The notification system has been integrated at key points in the application:

### Image Operations
- Image upload initiation and completion
- Image sending initiation and completion
- Image processing operations

### Scheduled Events
- Event creation, deletion, and updates
- Scheduled event execution

### System Operations
- Error conditions and warnings
- Successful operations

## Notification Service Usage

The notification service can be used in two ways:

### Direct Method Calls

```python
from utils.notification_service import NotificationService

# Create various types of notifications
NotificationService.create_info("Processing started...")
NotificationService.create_success("Operation completed successfully")
NotificationService.create_warning("Resource is running low")
NotificationService.create_error("Failed to complete operation")
```

### Decorator Pattern

```python
from utils.notification_service import notify_on_completion

@notify_on_completion(
    message_start="Starting operation...",
    message_success="Operation completed successfully",
    message_error="Operation failed: {error}"
)
def some_function():
    # Function code here
    pass
```

## Frontend Components

The notification system includes:

1. **Toast Notifications**: Pop-up notifications using Bootstrap 5 Toasts
2. **Notification Center**: Sidebar for viewing all notifications
3. **Notification Badge**: Counter showing unread notifications
4. **Bell Icon**: Access point for the notification center

## Automatic Cleanup

Expired notifications (older than 30 days) are automatically cleaned up through a scheduled task that runs every 12 hours.

## How It Works

1. When an action that should trigger a notification occurs, the application calls the Notification Service
2. The service creates a notification record in the database
3. If users are actively viewing the page, a toast notification appears
4. All notifications are accessible through the notification center
5. Notifications older than 30 days are automatically removed

## Testing the Notification System

You can test the notification system by:

1. Uploading an image (creates an "upload started" and "upload completed" notification)
2. Sending an image to a device (creates "sending" and "sent successfully" notifications)
3. Creating a scheduled event (creates a "scheduled successfully" notification)
4. Accessing the notification center by clicking the bell icon
5. Marking notifications as read

## Future Improvements

- **WebSocket Support**: Replace polling with WebSockets for real-time updates
- **Notification Categories**: Group notifications by category for better organization
- **User Preferences**: Allow users to customize notification settings
- **Mobile Notifications**: Extend to support mobile push notifications