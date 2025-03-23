# Notification System Integration Plan

## System Components

### 1. Database Schema (SQLAlchemy)
- `Notification` model with fields:
  - `id`: Primary key
  - `message`: The notification message
  - `type`: Type (info, success, warning, error)
  - `is_read`: Boolean flag for read status
  - `created_at`: Creation timestamp
  - `expires_at`: Expiration timestamp (30 days after creation)

### 2. Backend API (Flask Routes)
- `/api/notifications`: 
  - GET: Retrieve all non-expired notifications
  - POST: Create a new notification
- `/api/notifications/mark-read`: 
  - POST: Mark notification(s) as read
- `/api/notifications/delete`: 
  - POST: Delete specific notification(s)

### 3. Notification Service
- Create a centralized service to handle notification creation
- Implement a background job to automatically expire notifications older than 30 days

### 4. Front-end Components
- **Toast Notifications**: 
  - Bootstrap 5 toasts for real-time notifications
  - Automatically appear when new notifications are created
  - Different colors for different notification types
- **Notification Center**: 
  - Accessible from a bell icon in the navigation bar
  - Displays all non-expired notifications
  - Provides options to mark notifications as read or delete them
  - Badge counter showing unread notification count

## Implementation Strategy

### Phase 1: Database and Backend
1. Create the `Notification` model using SQLAlchemy
2. Add migration script to create the notifications table
3. Implement API endpoints for CRUD operations
4. Create a notification service for centralized notification creation
5. Implement automatic expiry mechanism for notifications older than 30 days

### Phase 2: Frontend Implementation
1. Create a toast notification component using Bootstrap 5
2. Implement a notification center sidebar/modal
3. Add a notification bell icon with badge counter in the navigation bar
4. Set up polling mechanism to check for new notifications every 30 seconds

### Phase 3: Integration
1. Identify all parts of the application that currently send notifications
2. Replace existing notification mechanisms with calls to the new notification service
3. Update frontend to display both real-time toasts and persistent notifications

## Technical Implementation Details

### Real-time Notification Mechanism
- Simple polling mechanism that checks for new notifications every 30 seconds
- Toast notifications only appear for notifications created after the user's session started

### 30-Day Expiry Implementation
- Set `expires_at` to 30 days from creation when creating a notification
- Backend API automatically filters out expired notifications
- Background job regularly cleans up expired notifications from the database

### Docker Considerations
- Include any new Python packages in requirements.txt
- Ensure all new components are properly initialized in app.py
- Update Docker-related files if necessary