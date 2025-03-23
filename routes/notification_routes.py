from flask import Blueprint, request, jsonify, current_app
from models import db, Notification
from datetime import datetime, timedelta
from sqlalchemy import desc

notification_bp = Blueprint('notification', __name__)

@notification_bp.route('/api/notifications', methods=['GET'])
def get_notifications():
    """
    Get all non-expired notifications.
    
    Query parameters:
    - include_read: Whether to include read notifications (default: false)
    - limit: Maximum number of notifications to return (default: 50)
    """
    include_read = request.args.get('include_read', 'false').lower() == 'true'
    limit = min(int(request.args.get('limit', 50)), 100)  # Cap at 100 notifications max
    
    # Get all non-expired notifications
    query = Notification.query.filter(Notification.expires_at > datetime.utcnow())
    
    if not include_read:
        query = query.filter_by(is_read=False)
    
    notifications = query.order_by(desc(Notification.created_at)).limit(limit).all()
    
    result = []
    for notification in notifications:
        result.append({
            'id': notification.id,
            'message': notification.message,
            'type': notification.type,
            'is_read': notification.is_read,
            'created_at': notification.created_at.isoformat(),
            'expires_at': notification.expires_at.isoformat()
        })
    
    return jsonify({'notifications': result})

@notification_bp.route('/api/notifications', methods=['POST'])
def create_notification():
    """
    Create a new notification.
    
    Request body:
    {
        "message": "Notification message",
        "type": "info|success|warning|error"
    }
    """
    data = request.get_json()
    
    if not data or 'message' not in data:
        return jsonify({'error': 'Message is required'}), 400
    
    message = data.get('message')
    notification_type = data.get('type', 'info')
    
    # Validate notification type
    if notification_type not in ['info', 'success', 'warning', 'error']:
        return jsonify({'error': 'Invalid notification type'}), 400
    
    # Create notification
    notification = Notification(message=message, type=notification_type)
    db.session.add(notification)
    db.session.commit()
    
    current_app.logger.info(f"Created new notification: {notification_type} - {message}")
    
    return jsonify({
        'id': notification.id,
        'message': notification.message,
        'type': notification.type,
        'is_read': notification.is_read,
        'created_at': notification.created_at.isoformat(),
        'expires_at': notification.expires_at.isoformat()
    }), 201

@notification_bp.route('/api/notifications/mark-read', methods=['POST'])
def mark_notifications_read():
    """
    Mark notifications as read.
    
    Request body:
    {
        "ids": [1, 2, 3]  # Optional - if not provided, marks all as read
    }
    """
    data = request.get_json() or {}
    notification_ids = data.get('ids', [])
    
    if notification_ids:
        # Mark specific notifications as read
        notifications = Notification.query.filter(
            Notification.id.in_(notification_ids),
            Notification.expires_at > datetime.utcnow()
        ).all()
        
        for notification in notifications:
            notification.is_read = True
        
        db.session.commit()
        return jsonify({'message': f'Marked {len(notifications)} notifications as read'})
    else:
        # Mark all notifications as read
        count = Notification.query.filter(
            Notification.is_read == False,
            Notification.expires_at > datetime.utcnow()
        ).update({'is_read': True})
        
        db.session.commit()
        return jsonify({'message': f'Marked {count} notifications as read'})

@notification_bp.route('/api/notifications/delete-expired', methods=['POST'])
def delete_expired_notifications():
    """
    Delete all expired notifications.
    Admin-only endpoint.
    """
    count = Notification.delete_expired()
    return jsonify({'message': f'Deleted {count} expired notifications'})

@notification_bp.route('/api/notifications/<int:notification_id>', methods=['DELETE'])
def delete_notification(notification_id):
    """
    Delete a specific notification.
    """
    notification = Notification.query.get_or_404(notification_id)
    db.session.delete(notification)
    db.session.commit()
    
    return jsonify({'message': f'Notification {notification_id} deleted'})