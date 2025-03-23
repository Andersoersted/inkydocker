from models import db, Notification
from datetime import datetime, timedelta
from flask import current_app
import functools

class NotificationService:
    """
    Centralized service for creating and managing notifications.
    """
    
    @staticmethod
    def create_notification(message, notification_type='info'):
        """
        Create a new notification with the specified message and type.
        
        Args:
            message (str): The notification message
            notification_type (str): Type of notification (info, success, warning, error)
            
        Returns:
            Notification: The created notification object
        """
        try:
            notification = Notification(message=message, type=notification_type)
            db.session.add(notification)
            db.session.commit()
            
            current_app.logger.info(f"Created notification: {notification_type} - {message}")
            return notification
        except Exception as e:
            current_app.logger.error(f"Error creating notification: {str(e)}")
            db.session.rollback()
            return None
    
    @staticmethod
    def create_info(message):
        """Shorthand for creating an info notification"""
        return NotificationService.create_notification(message, 'info')
    
    @staticmethod
    def create_success(message):
        """Shorthand for creating a success notification"""
        return NotificationService.create_notification(message, 'success')
    
    @staticmethod
    def create_warning(message):
        """Shorthand for creating a warning notification"""
        return NotificationService.create_notification(message, 'warning')
    
    @staticmethod
    def create_error(message):
        """Shorthand for creating an error notification"""
        return NotificationService.create_notification(message, 'error')
    
    @staticmethod
    def delete_expired():
        """Delete all expired notifications"""
        return Notification.delete_expired()

# Helper function to use as a decorator for operations that need notifications
def notify_on_completion(message_start=None, message_success=None, message_error=None):
    """
    Decorator that creates notifications when a function starts, succeeds, or fails.
    
    Usage:
        @notify_on_completion(
            message_start="Starting image processing...",
            message_success="Image processing completed successfully",
            message_error="Image processing failed: {error}"
        )
        def process_image(image_path):
            # Process the image
            return result
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Create start notification if provided
            if message_start:
                NotificationService.create_info(message_start)
            
            try:
                # Call the original function
                result = func(*args, **kwargs)
                
                # Create success notification if provided
                if message_success:
                    NotificationService.create_success(message_success)
                
                return result
            except Exception as e:
                # Create error notification if provided
                if message_error:
                    error_msg = message_error
                    if "{error}" in message_error:
                        error_msg = message_error.format(error=str(e))
                    NotificationService.create_error(error_msg)
                
                # Re-raise the exception
                raise
        
        return wrapper
    
    return decorator