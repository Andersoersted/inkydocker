# Notification System Migration Plan

## Overview

This document outlines the plan to migrate all legacy notification mechanisms in the InkyDocker application to the new centralized notification system. The goal is to eliminate redundant notification implementations and ensure a consistent user experience with both real-time and persistent notifications.

## 1. Legacy Notification Mechanisms Identified

### 1.1. Image Sending Notifications (templates/index.html)

- **Lines ~550-560**: Spinner in buttons during image sending
- **Lines ~590-620**: Toast notifications for successful image sends
- **Lines ~625-650**: Toast notifications for failed image sends
- **Lines ~665-705**: Toast notifications for network errors during sending
- **Lines ~705-745**: Toast notifications for timeouts during sending

### 1.2. Image Upload Notifications (templates/index.html)

- **Lines ~485-501**: Upload form initialization
- **Lines ~502-510**: Progress bar updates during upload
- **Lines ~512-525**: Upload status updates in DOM elements
- **Lines ~527-530**: Error status updates in DOM elements

### 1.3. Scheduled Event Notifications (static/fullcalendar-init.js)

- Custom toast notifications for event creation, updating, and deletion
- No persistent notifications for event execution

### 1.4. E-Ink Display Addition (routes/device_routes.py)

- Likely contains duplicate status updates when adding a display
- Needs to be examined and consolidated

### 1.5. Screenshot Operations (routes/browserless_routes.py)

- Status updates for screenshot capture and processing
- No persistent notifications

## 2. Migration Strategy

### 2.1. Core Components (Already Implemented)

1. **Notification Database Model** (`models.py`)
   - `Notification` class with appropriate fields and 30-day expiry logic

2. **Notification API** (`routes/notification_routes.py`)
   - RESTful endpoints for notification CRUD operations

3. **Notification Service** (`utils/notification_service.py`)
   - Service for creating notifications from any part of the application
   - Decorator for wrapping functions with notification creation

4. **Notification UI** (`static/js/notifications.js` and `static/css/notifications.css`)
   - Frontend components for displaying and managing notifications

### 2.2. Migration Steps

1. **Add Integration Layer** (`static/js/live-notifications.js`)
   - Helper functions to integrate existing code with the notification system
   - Replace UI updates with notification creation

2. **Update Templates**
   - Modify HTML templates to use the new notification system
   - Remove redundant notification elements
   - Keep some progress indicators for visual feedback but ensure all events create persistent notifications

3. **Update Backend Routes**
   - Apply the `@notify_on_completion` decorator to appropriate route handlers
   - Replace direct status updates with calls to `NotificationService`

4. **Add Documentation**
   - Add inline comments explaining the notification system integration
   - Update documentation files with implementation details and testing instructions

## 3. File-by-File Migration Plan

### 3.1. Frontend Files

#### 3.1.1. `templates/index.html`

- **Replace Progress Bar Logic**:
  - Keep the progress bar for visual feedback
  - Add notification creation at start/progress/completion
  - Remove redundant status text updates

- **Replace Image Sending Notifications**:
  - Keep the button spinner for visual feedback
  - Replace all toast creation code with calls to `createNotification()`
  - Ensure one notification per event (no duplicates)

#### 3.1.2. `templates/schedule.html` and related JS

- **Replace Event Notifications**:
  - Identify event creation, update, delete operations
  - Replace custom toast implementations with `createNotification()`
  - Add notifications for event execution

#### 3.1.3. `static/js/live-notifications.js` (New File)

- Create new integration layer to handle:
  - Upload progress notification
  - Image sending notification
  - Event notifications
  - Integration with existing UI elements

### 3.2. Backend Files

#### 3.2.1. `routes/image_routes.py`

- Apply `@notify_on_completion` to:
  - `upload_file` function
  - `send_image` function
  - Other image processing functions

#### 3.2.2. `routes/device_routes.py`

- Apply `@notify_on_completion` to:
  - Device addition functions
  - Device update functions
  - Remove duplicate notifications

#### 3.2.3. `routes/browserless_routes.py`

- Apply `@notify_on_completion` to:
  - Screenshot capture functions
  - Screenshot processing functions

#### 3.2.4. `routes/schedule_routes.py`

- Apply `@notify_on_completion` to:
  - Event creation/update/delete functions
  - Add notification creation to event execution code

#### 3.2.5. `tasks.py`

- Add notification creation to:
  - Asynchronous image processing tasks
  - Scheduled tasks
  - Event execution logic

## 4. Specific Code Locations and Modifications

### 4.1. Frontend Modifications

#### 4.1.1. `templates/index.html`

| Line Numbers | Current Implementation | Migration Approach |
|--------------|------------------------|-------------------|
| ~486-510 | Upload form with progress bar | Keep progress bar but add notification creation at start, progress milestones, and completion |
| ~512-530 | Upload status messages | Replace with notification creation |
| ~534-556 | Image send button with spinner | Keep spinner but add notification creation |
| ~586-623 | Success toast for image sending | Replace with `createNotification()` |
| ~625-660 | Error toast for image sending | Replace with `createNotification()` |
| ~665-705 | Network error toast | Replace with `createNotification()` |
| ~705-745 | Timeout error toast | Replace with `createNotification()` |

#### 4.1.2. `templates/schedule.html` and JS

| Location | Current Implementation | Migration Approach |
|----------|------------------------|-------------------|
| FullCalendar init | Event creation/update toasts | Replace with notification creation |

### 4.2. Backend Modifications

#### 4.2.1. `routes/settings_routes.py`

| Line Numbers | Current Implementation | Migration Approach |
|--------------|------------------------|-------------------|
| ~213 | `flash("Device added successfully", "success")` | Add `NotificationService.create_success("E-Ink display added successfully")` |
| ~215 | `flash("Missing mandatory fields...", "error")` | Add `NotificationService.create_error("Missing fields for E-Ink display addition")` |
| ~239 | `flash("Device deleted", "success")` | Add `NotificationService.create_success("E-Ink display deleted")` |
| ~241 | `flash("Device not found", "error")` | Add `NotificationService.create_error("E-Ink display not found")` |
| ~260 | `flash("Device updated successfully", "success")` | Add `NotificationService.create_success("E-Ink display updated successfully")` |
| ~264 | `flash("Error editing device: " + str(e), "error")` | Add `NotificationService.create_error(f"Error updating E-Ink display: {str(e)}")` |

#### 4.2.2. `routes/image_routes.py`

| Functions | Migration Approach |
|-----------|-------------------|
| `upload_file()` | Apply `@notify_on_completion` decorator |
| `send_image()` | Apply `@notify_on_completion` decorator |
| `save_crop_info_endpoint()` | Apply `@notify_on_completion` decorator |

#### 4.2.3. `routes/browserless_routes.py`

| Functions | Migration Approach |
|-----------|-------------------|
| Screenshot-related functions | Apply `@notify_on_completion` decorator |

#### 4.2.4. `tasks.py`

| Functions | Migration Approach |
|-----------|-------------------|
| `send_scheduled_image()` | Add notification creation at key points |
| Tagging functions | Add notification creation at key points |

## 5. Testing Strategy

1. **Unit Testing**:
   - Verify notification creation from various parts of the application
   - Ensure notifications are properly persisted

2. **Integration Testing**:
   - Test each user flow that should trigger notifications
   - Verify notifications appear in real-time and persist

3. **Specific Test Cases**:
   - Image upload with progress
   - Image sending success/failure
   - E-Ink display addition
   - Screenshot operations
   - Scheduled event creation and execution

## 5. Implementation Timeline

1. **Phase 1**: Update frontend templates to use notification system
   - Focus on `index.html` and `schedule.html`

2. **Phase 2**: Update backend routes with notification service
   - Apply decorators to route handlers
   - Remove duplicate notification logic

3. **Phase 3**: Test and refine
   - Identify any missed notification paths
   - Ensure consistent user experience

## 6. Success Criteria

- All user actions produce exactly one notification (no duplicates)
- All notifications are visible in real-time and persisted to history
- No legacy toast or status update mechanisms remain active
- Consistent notification style and behavior throughout the application