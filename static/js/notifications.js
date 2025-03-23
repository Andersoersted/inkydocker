/**
 * Notification System
 * 
 * Handles real-time and persistent notifications using Bootstrap 5 toasts
 * and a centralized notification center.
 */

// Initialize the notification system
document.addEventListener('DOMContentLoaded', function() {
    // Create notification system
    NotificationSystem.init();
});

const NotificationSystem = {
    // Configuration
    config: {
        pollInterval: 30000, // 30 seconds
        maxToasts: 5,
        notificationLifetime: 30, // 30 days
    },
    
    // State
    state: {
        lastFetchTime: null,
        pollingTimer: null,
        unreadCount: 0,
        isInitialized: false,
    },
    
    // Cache DOM elements
    elements: {
        toastContainer: null,
        notificationCenter: null,
        notificationBody: null,
        notificationBadge: null,
        bellIcon: null,
    },
    
    /**
     * Initialize the notification system
     */
    init: function() {
        if (this.state.isInitialized) return;
        
        // Add notification bell to the navbar if it doesn't exist
        this.createNotificationElements();
        
        // Initialize elements
        this.elements.toastContainer = document.querySelector('.toast-container');
        if (!this.elements.toastContainer) {
            this.elements.toastContainer = document.createElement('div');
            this.elements.toastContainer.className = 'toast-container position-fixed bottom-0 end-0 p-3';
            document.body.appendChild(this.elements.toastContainer);
        }
        
        this.elements.bellIcon = document.getElementById('notification-bell');
        this.elements.notificationBadge = document.getElementById('notification-badge');
        this.elements.notificationCenter = document.getElementById('notification-center');
        this.elements.notificationBody = document.getElementById('notification-body');
        
        // Setup event listeners
        this.setupEventListeners();
        
        // Start polling for new notifications
        this.startPolling();
        
        // Mark as initialized
        this.state.isInitialized = true;
        
        // Initial fetch of notifications
        this.fetchNotifications();
    },
    
    /**
     * Create notification UI elements
     */
    createNotificationElements: function() {
        // Check if notification bell already exists
        if (document.getElementById('notification-bell')) return;
        
        // Create notification bell in the navbar
        const navbarNav = document.querySelector('#navbarNav .navbar-nav');
        if (!navbarNav) return;
        
        const notificationLi = document.createElement('li');
        notificationLi.className = 'nav-item notification-badge';
        notificationLi.innerHTML = `
            <a class="nav-link" href="#" id="notification-bell" data-bs-toggle="offcanvas" data-bs-target="#notification-center">
                <i class="fas fa-bell"></i>
                <span class="badge bg-danger" id="notification-badge" style="display: none;">0</span>
            </a>
        `;
        navbarNav.appendChild(notificationLi);
        
        // Create notification center
        const notificationCenter = document.createElement('div');
        notificationCenter.className = 'offcanvas offcanvas-end';
        notificationCenter.id = 'notification-center';
        notificationCenter.setAttribute('tabindex', '-1');
        notificationCenter.setAttribute('aria-labelledby', 'notification-center-label');
        
        notificationCenter.innerHTML = `
            <div class="offcanvas-header notification-header">
                <h5 class="offcanvas-title" id="notification-center-label">Notifications</h5>
                <button type="button" class="btn-close text-reset" data-bs-dismiss="offcanvas" aria-label="Close"></button>
            </div>
            <div class="offcanvas-body p-0">
                <div id="notification-body" class="notification-body">
                    <div class="notification-empty">
                        <i class="fas fa-bell fa-2x mb-3"></i>
                        <p>No notifications yet</p>
                    </div>
                </div>
                <div class="notification-footer">
                    <button class="btn btn-sm btn-outline-secondary" id="mark-all-read">Mark all as read</button>
                </div>
            </div>
        `;
        
        document.body.appendChild(notificationCenter);
        
        // Add CSS link if it doesn't exist
        if (!document.querySelector('link[href*="notifications.css"]')) {
            const cssLink = document.createElement('link');
            cssLink.rel = 'stylesheet';
            cssLink.href = '/static/css/notifications.css';
            document.head.appendChild(cssLink);
        }
    },
    
    /**
     * Set up event listeners
     */
    setupEventListeners: function() {
        // Mark all as read button
        const markAllReadBtn = document.getElementById('mark-all-read');
        if (markAllReadBtn) {
            markAllReadBtn.addEventListener('click', () => this.markAllAsRead());
        }
        
        // Set up notification center events
        if (this.elements.notificationCenter) {
            this.elements.notificationCenter.addEventListener('show.bs.offcanvas', () => {
                // Refresh notifications when opening the center
                this.fetchNotifications();
            });
        }
        
        // Delegate click events for notification items
        if (this.elements.notificationBody) {
            this.elements.notificationBody.addEventListener('click', (e) => {
                // Check if the clicked element or its parent is a notification item
                const notificationItem = e.target.closest('.notification-item');
                if (!notificationItem) return;
                
                // Check if the clicked element is the mark-read button
                const markReadBtn = e.target.closest('.mark-read-btn');
                if (markReadBtn) {
                    const notificationId = notificationItem.dataset.id;
                    this.markAsRead(notificationId);
                    e.preventDefault();
                    return;
                }
                
                // Mark as read when clicking the notification itself
                const notificationId = notificationItem.dataset.id;
                if (notificationId && !notificationItem.classList.contains('read')) {
                    this.markAsRead(notificationId);
                }
            });
        }
    },
    
    /**
     * Start polling for new notifications
     */
    startPolling: function() {
        if (this.state.pollingTimer) {
            clearInterval(this.state.pollingTimer);
        }
        
        this.state.pollingTimer = setInterval(() => {
            this.fetchNotifications();
        }, this.config.pollInterval);
    },
    
    /**
     * Fetch notifications from the server
     */
    fetchNotifications: function() {
        fetch('/api/notifications')
            .then(response => response.json())
            .then(data => {
                this.handleNotifications(data.notifications);
            })
            .catch(error => {
                console.error('Error fetching notifications:', error);
            });
    },
    
    /**
     * Handle notifications data from server
     */
    handleNotifications: function(notifications) {
        // Update unread count
        this.state.unreadCount = notifications.length;
        this.updateBadge();
        
        // Update notification center
        this.updateNotificationCenter(notifications);
        
        // Show toasts for new notifications
        if (this.state.lastFetchTime) {
            const newNotifications = notifications.filter(notification => {
                const notificationTime = new Date(notification.created_at);
                return notificationTime > this.state.lastFetchTime;
            });
            
            // Show toasts for new notifications (in reverse order so newest appears on top)
            newNotifications.reverse().forEach(notification => {
                this.showToast(notification);
            });
        }
        
        // Update last fetch time
        this.state.lastFetchTime = new Date();
    },
    
    /**
     * Update the notification badge count
     */
    updateBadge: function() {
        if (!this.elements.notificationBadge) return;
        
        if (this.state.unreadCount > 0) {
            this.elements.notificationBadge.textContent = this.state.unreadCount > 99 ? '99+' : this.state.unreadCount;
            this.elements.notificationBadge.style.display = 'inline-block';
        } else {
            this.elements.notificationBadge.style.display = 'none';
        }
    },
    
    /**
     * Update the notification center with notifications
     */
    updateNotificationCenter: function(notifications) {
        if (!this.elements.notificationBody) return;
        
        if (notifications.length === 0) {
            this.elements.notificationBody.innerHTML = `
                <div class="notification-empty">
                    <i class="fas fa-bell fa-2x mb-3"></i>
                    <p>No notifications yet</p>
                </div>
            `;
            return;
        }
        
        // Clear existing notifications
        this.elements.notificationBody.innerHTML = '';
        
        // Add each notification
        notifications.forEach(notification => {
            const notificationItem = document.createElement('div');
            notificationItem.className = `notification-item ${notification.type}`;
            notificationItem.dataset.id = notification.id;
            
            // Format the date
            const notificationDate = new Date(notification.created_at);
            const formattedDate = notificationDate.toLocaleDateString() + ' ' + notificationDate.toLocaleTimeString();
            
            // Icon based on notification type
            const iconClass = this.getIconForType(notification.type);
            
            notificationItem.innerHTML = `
                <div class="d-flex align-items-start">
                    <div class="notification-icon ${notification.type}">
                        <i class="${iconClass}"></i>
                    </div>
                    <div class="flex-grow-1">
                        <div class="notification-message">${notification.message}</div>
                        <div class="notification-time">${formattedDate}</div>
                    </div>
                    <button class="btn btn-sm mark-read-btn" title="Mark as read">
                        <i class="fas fa-check"></i>
                    </button>
                </div>
            `;
            
            this.elements.notificationBody.appendChild(notificationItem);
        });
    },
    
    /**
     * Show a toast notification
     */
    showToast: function(notification) {
        if (!this.elements.toastContainer) return;
        
        // Create toast element
        const toast = document.createElement('div');
        toast.className = `toast notification-toast ${notification.type} show`;
        toast.setAttribute('role', 'alert');
        toast.setAttribute('aria-live', 'assertive');
        toast.setAttribute('aria-atomic', 'true');
        
        // Icon based on notification type
        const iconClass = this.getIconForType(notification.type);
        
        toast.innerHTML = `
            <div class="toast-header">
                <i class="${iconClass} me-2"></i>
                <strong class="me-auto">${this.getHeaderForType(notification.type)}</strong>
                <small>Just now</small>
                <button type="button" class="btn-close" data-bs-dismiss="toast" aria-label="Close"></button>
            </div>
            <div class="toast-body">
                ${notification.message}
            </div>
        `;
        
        // Add to container
        this.elements.toastContainer.appendChild(toast);
        
        // Initialize Bootstrap toast
        const bsToast = new bootstrap.Toast(toast, {
            autohide: true,
            delay: 5000
        });
        
        // Remove from DOM after hiding
        toast.addEventListener('hidden.bs.toast', function() {
            toast.remove();
        });
        
        // Show the toast
        bsToast.show();
        
        // Limit maximum number of toasts
        const toasts = this.elements.toastContainer.querySelectorAll('.toast');
        if (toasts.length > this.config.maxToasts) {
            // Remove oldest toasts
            for (let i = 0; i < toasts.length - this.config.maxToasts; i++) {
                const oldToast = toasts[i];
                const bsOldToast = bootstrap.Toast.getInstance(oldToast);
                if (bsOldToast) {
                    bsOldToast.hide();
                } else {
                    oldToast.remove();
                }
            }
        }
    },
    
    /**
     * Mark a notification as read
     */
    markAsRead: function(notificationId) {
        fetch('/api/notifications/mark-read', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ ids: [notificationId] }),
        })
        .then(response => response.json())
        .then(data => {
            // Refresh notifications
            this.fetchNotifications();
        })
        .catch(error => {
            console.error('Error marking notification as read:', error);
        });
    },
    
    /**
     * Mark all notifications as read
     */
    markAllAsRead: function() {
        fetch('/api/notifications/mark-read', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({}),  // Empty object will mark all as read
        })
        .then(response => response.json())
        .then(data => {
            // Refresh notifications
            this.fetchNotifications();
        })
        .catch(error => {
            console.error('Error marking all notifications as read:', error);
        });
    },
    
    /**
     * Get appropriate icon class for notification type
     */
    getIconForType: function(type) {
        switch (type) {
            case 'info':
                return 'fas fa-info-circle';
            case 'success':
                return 'fas fa-check-circle';
            case 'warning':
                return 'fas fa-exclamation-triangle';
            case 'error':
                return 'fas fa-times-circle';
            default:
                return 'fas fa-bell';
        }
    },
    
    /**
     * Get appropriate header text for notification type
     */
    getHeaderForType: function(type) {
        switch (type) {
            case 'info':
                return 'Information';
            case 'success':
                return 'Success';
            case 'warning':
                return 'Warning';
            case 'error':
                return 'Error';
            default:
                return 'Notification';
        }
    }
};

/**
 * Helper function to create a notification
 * 
 * @param {string} message - The notification message
 * @param {string} type - The notification type: 'info', 'success', 'warning', 'error'
 * @returns {Promise} - A promise that resolves when the notification is created
 */
function createNotification(message, type = 'info') {
    return fetch('/api/notifications', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ message, type }),
    })
    .then(response => response.json())
    .catch(error => {
        console.error('Error creating notification:', error);
    });
}