# Notification System Troubleshooting

This document provides solutions for common issues that might arise with the notification system.

## Common Issues

### 1. Duplicate Notifications

If you're seeing duplicate notifications for the same action, this might be caused by:

- **Multiple Trigger Points**: The same action might be triggering notifications from multiple places in the code.
  - **Solution**: Check if you're using both the `@notify_on_completion` decorator and direct `NotificationService` calls for the same operation.

- **Frontend Event Binding**: Multiple event handlers might be triggering for the same action.
  - **Solution**: Make sure event handlers are properly bound and use event delegation where appropriate.

- **Polling Overlap**: If polling intervals overlap with manual refreshes, you might see duplicates.
  - **Solution**: Implement debouncing in the notification polling system.

### 2. Missing Notifications

If notifications aren't appearing:

- **Check Database**: Verify that notifications are being created in the database.
  - **Solution**: Check server logs for any errors in notification creation.

- **Frontend Issues**: If notifications are in the database but not showing in the UI:
  - **Solution**: Check browser console for JavaScript errors and verify that the polling is working.

- **Expired Notifications**: Notifications older than 30 days are automatically removed.
  - **Solution**: Check the `expires_at` field in the database.

### 3. Notification Center Not Showing

If the notification center doesn't appear when clicking the bell icon:

- **Bootstrap JS**: Verify that Bootstrap 5 JS is properly loaded.
  - **Solution**: Check the browser console for any errors related to Bootstrap.

- **DOM Structure**: The notification center relies on a specific DOM structure.
  - **Solution**: Verify that the offcanvas component is properly initialized.

### 4. Notification Styling Issues

If notifications don't appear correctly styled:

- **CSS Loading**: Verify that the notification CSS file is being loaded.
  - **Solution**: Check that `/static/css/notifications.css` is included in the page.

- **Bootstrap Compatibility**: The styling depends on Bootstrap 5.
  - **Solution**: Make sure you're using a compatible Bootstrap version.

## Advanced Troubleshooting

### Backend Debugging

To debug notification creation on the backend:

```python
from flask import current_app

# Add this to the point where you suspect notifications aren't being created
current_app.logger.debug(f"Attempting to create notification: {message}")
```

### Frontend Debugging

To debug notification display on the frontend:

```javascript
// Add this to notifications.js to see detailed polling information
const originalFetchNotifications = NotificationSystem.fetchNotifications;
NotificationSystem.fetchNotifications = function() {
    console.log("Fetching notifications at", new Date());
    return originalFetchNotifications.apply(this, arguments).then(result => {
        console.log("Fetch result:", result);
        return result;
    });
};
```

## Getting Help

If you continue to experience issues with the notification system, check:

1. Application logs for error messages
2. Browser console for JavaScript errors
3. Database entries to confirm notification creation

For persistent issues, you might need to consider rebuilding the container to ensure all components are properly installed and initialized.