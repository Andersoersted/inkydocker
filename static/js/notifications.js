/**
 * Notification System
 *
 * Handles:
 * 1. Fetching/displaying persistent notifications from the backend via polling.
 * 2. Displaying these persistent notifications in an offcanvas center.
 * 3. Showing transient 'live' toasts for ongoing operations (uploads, sending, etc.)
 *    with spinners or progress bars, managed client-side.
 */

const NotificationSystem = {
    // --- Configuration ---
    config: {
        pollInterval: 30000, // 30 seconds for backend polling
        maxToasts: 5,        // Max simultaneous standard toasts
        liveToastDefaults: {
            autohide: false,
            delay: 5000 // Default delay for hiding completed live toasts
        }
    },

    // --- State ---
    state: {
        lastFetchTime: null,
        pollingTimer: null,
        unreadCount: 0,
        isInitialized: false,
        activeLiveToasts: {} // Stores references to active live toasts { operationId: { element: toastElement, bsToast: bootstrapToastInstance } }
    },

    // --- DOM Elements Cache ---
    elements: {
        toastContainer: null,
        notificationCenter: null,
        notificationBody: null,
        notificationBadge: null,
        bellIcon: null,
    },

    // --- Initialization ---
    init: function() {
        if (this.state.isInitialized) return;
        console.log("Initializing NotificationSystem...");

        // Ensure toast container exists (moved from app.js for self-containment)
        this.elements.toastContainer = document.querySelector('.toast-container');
        if (!this.elements.toastContainer) {
            console.log("Creating toast container.");
            this.elements.toastContainer = document.createElement('div');
            this.elements.toastContainer.className = 'toast-container position-fixed bottom-0 end-0 p-3';
            this.elements.toastContainer.style.zIndex = '1100'; // Ensure toasts appear above most elements
            document.body.appendChild(this.elements.toastContainer);
        } else {
             console.log("Toast container found.");
        }

        // Create notification center UI if needed
        this.createNotificationCenterElements();

        // Cache other elements
        this.elements.bellIcon = document.getElementById('notification-bell');
        this.elements.notificationBadge = document.getElementById('notification-badge');
        this.elements.notificationCenter = document.getElementById('notification-center');
        this.elements.notificationBody = document.getElementById('notification-body');

        // Setup event listeners (polling, center, live operations)
        this.setupBackendPollingListeners();
        this.setupNotificationCenterListeners();
        this.setupLiveOperationListeners(); // Integrate listeners from live-notifications.js

        this.state.isInitialized = true;
        console.log("NotificationSystem Initialized.");

        // Initial fetch of persistent notifications
        this.fetchPersistentNotifications();
    },

    // --- Backend Persistent Notification Handling ---

    createNotificationCenterElements: function() {
        // Check if notification bell already exists
        if (document.getElementById('notification-bell')) return;
        console.log("Creating notification center elements.");

        const navbarNav = document.querySelector('#navbarNav .navbar-nav');
        if (!navbarNav) {
            console.error("Navbar nav container not found for notification bell.");
            return;
        }

        const notificationLi = document.createElement('li');
        notificationLi.className = 'nav-item'; // Removed notification-badge class, badge is inside
        notificationLi.innerHTML = `
            <a class="nav-link position-relative" href="#" id="notification-bell" data-bs-toggle="offcanvas" data-bs-target="#notification-center" aria-label="Notifications">
                <i class="fas fa-bell" aria-hidden="true"></i>
                <span class="badge bg-danger rounded-pill position-absolute top-0 start-100 translate-middle" id="notification-badge" style="display: none;">0</span>
            </a>
        `;
        navbarNav.appendChild(notificationLi);

        const notificationCenter = document.createElement('div');
        notificationCenter.className = 'offcanvas offcanvas-end';
        notificationCenter.id = 'notification-center';
        notificationCenter.setAttribute('tabindex', '-1');
        notificationCenter.setAttribute('aria-labelledby', 'notification-center-label');

        notificationCenter.innerHTML = `
            <div class="offcanvas-header notification-header">
                <h5 class="offcanvas-title" id="notification-center-label">Notifications</h5>
                <button type="button" class="btn-close" data-bs-dismiss="offcanvas" aria-label="Close"></button>
            </div>
            <div class="offcanvas-body p-0">
                <div id="notification-body" class="notification-body">
                    <!-- Content populated by JS -->
                </div>
                <div class="notification-footer">
                    <button class="btn btn-sm btn-outline-secondary" id="mark-all-read">Mark all as read</button>
                </div>
            </div>
        `;
        document.body.appendChild(notificationCenter);

        // Add CSS link dynamically if needed (though it should be in base.html now)
        if (!document.querySelector('link[href*="notifications.css"]')) {
            console.warn("Notifications CSS not found in head, adding dynamically.");
            const cssLink = document.createElement('link');
            cssLink.rel = 'stylesheet';
            cssLink.href = '/static/css/notifications.css'; // Adjust path if necessary
            document.head.appendChild(cssLink);
        }
    },

    setupBackendPollingListeners: function() {
        if (this.state.pollingTimer) {
            clearInterval(this.state.pollingTimer);
        }
        this.state.pollingTimer = setInterval(() => {
            this.fetchPersistentNotifications();
        }, this.config.pollInterval);
        console.log("Backend polling started.");
    },

    setupNotificationCenterListeners: function() {
        const markAllReadBtn = document.getElementById('mark-all-read');
        if (markAllReadBtn) {
            markAllReadBtn.addEventListener('click', () => this.markAllAsRead());
        }

        if (this.elements.notificationCenter) {
            // Use Bootstrap's event system
            this.elements.notificationCenter.addEventListener('show.bs.offcanvas', () => {
                console.log("Notification center opening, fetching persistent notifications.");
                this.fetchPersistentNotifications(); // Refresh on open
            });
        }

        if (this.elements.notificationBody) {
            this.elements.notificationBody.addEventListener('click', (e) => {
                const notificationItem = e.target.closest('.notification-item');
                if (!notificationItem) return;

                const markReadBtn = e.target.closest('.mark-read-btn');
                const notificationId = notificationItem.dataset.id;

                if (markReadBtn && notificationId) {
                    console.log(`Marking notification ${notificationId} as read via button.`);
                    this.markAsRead(notificationId);
                    e.preventDefault(); // Prevent other actions if button clicked
                } else if (notificationId && !notificationItem.classList.contains('read')) {
                    // Mark as read when clicking the notification body itself (if not already read)
                    console.log(`Marking notification ${notificationId} as read via item click.`);
                    this.markAsRead(notificationId);
                }
            });
        }
         console.log("Notification center listeners set up.");
    },

    fetchPersistentNotifications: function() {
        console.log("Fetching persistent notifications...");
        fetch('/api/notifications') // Assuming this endpoint returns only unread persistent notifications
            .then(response => {
                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
                return response.json();
            })
            .then(data => {
                console.log("Received persistent notifications:", data);
                this.handlePersistentNotifications(data.notifications || []);
            })
            .catch(error => {
                console.error('Error fetching persistent notifications:', error);
                // Optionally show an error toast
                // this.showToast({ message: 'Could not fetch notifications.', type: 'error' });
            });
    },

    handlePersistentNotifications: function(notifications) {
        // Update unread count based on fetched *persistent* notifications
        this.state.unreadCount = notifications.filter(n => !n.read).length; // Assuming API returns read status
        this.updateBadge();

        // Update notification center UI
        this.updateNotificationCenterUI(notifications);

        // Show standard toasts ONLY for *new* persistent notifications since last fetch
        if (this.state.lastFetchTime) {
            const newNotifications = notifications.filter(notification => {
                // Ensure created_at is valid before creating Date object
                return notification.created_at && new Date(notification.created_at) > this.state.lastFetchTime;
            });

            if (newNotifications.length > 0) {
                 console.log(`Showing ${newNotifications.length} new persistent notification toasts.`);
                 // Show toasts (newest first)
                 newNotifications.reverse().forEach(notification => {
                     this.showStandardToast(notification); // Use a separate function for standard toasts
                 });
            }
        }

        // Update last fetch time only if fetch was successful
        this.state.lastFetchTime = new Date();
    },

    updateBadge: function() {
        if (!this.elements.notificationBadge) return;
        if (this.state.unreadCount > 0) {
            this.elements.notificationBadge.textContent = this.state.unreadCount > 99 ? '99+' : this.state.unreadCount;
            this.elements.notificationBadge.style.display = ''; // Use '' to reset to default display
        } else {
            this.elements.notificationBadge.style.display = 'none';
        }
    },

    updateNotificationCenterUI: function(notifications) {
        if (!this.elements.notificationBody) return;

        if (!notifications || notifications.length === 0) {
            this.elements.notificationBody.innerHTML = `
                <div class="notification-empty">
                    <i class="fas fa-bell fa-2x mb-3" aria-hidden="true"></i>
                    <p>No notifications yet</p>
                </div>
            `;
            return;
        }

        this.elements.notificationBody.innerHTML = ''; // Clear existing
        notifications.sort((a, b) => new Date(b.created_at) - new Date(a.created_at)); // Sort newest first

        notifications.forEach(notification => {
            const item = document.createElement('div');
            item.className = `notification-item ${notification.type || 'info'} ${notification.read ? 'read' : ''}`;
            item.dataset.id = notification.id;

            const date = notification.created_at ? new Date(notification.created_at) : new Date();
            // More robust date formatting
            const formattedDate = date.toLocaleDateString(undefined, { year: 'numeric', month: 'short', day: 'numeric' }) + ' ' +
                                  date.toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit' });

            const iconClass = this.getIconForType(notification.type);

            item.innerHTML = `
                <div class="notification-icon ${notification.type || 'info'}">
                    <i class="${iconClass}" aria-hidden="true"></i>
                </div>
                <div class="notification-content flex-grow-1">
                    <div class="notification-message">${notification.message || 'No message'}</div>
                    <div class="notification-time">${formattedDate}</div>
                </div>
                ${!notification.read ? `
                <button class="btn btn-sm btn-outline-secondary mark-read-btn ms-2" title="Mark as read">
                    <i class="fas fa-check" aria-hidden="true"></i>
                </button>` : ''}
            `;
            this.elements.notificationBody.appendChild(item);
        });
    },

    markAsRead: function(notificationId) {
        console.log(`API call: Mark notification ${notificationId} as read.`);
        fetch('/api/notifications/mark-read', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ ids: [notificationId] }),
        })
        .then(response => response.ok ? response.json() : Promise.reject(`HTTP error! status: ${response.status}`))
        .then(() => {
            console.log(`Notification ${notificationId} marked as read successfully.`);
            this.fetchPersistentNotifications(); // Refresh list
        })
        .catch(error => {
            console.error(`Error marking notification ${notificationId} as read:`, error);
            this.showStandardToast({ message: 'Failed to mark notification as read.', type: 'error' });
        });
    },

    markAllAsRead: function() {
        console.log("API call: Mark all notifications as read.");
        fetch('/api/notifications/mark-read', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({}), // Empty body means mark all
        })
        .then(response => response.ok ? response.json() : Promise.reject(`HTTP error! status: ${response.status}`))
        .then(() => {
            console.log("All notifications marked as read successfully.");
            this.fetchPersistentNotifications(); // Refresh list
        })
        .catch(error => {
            console.error('Error marking all notifications as read:', error);
            this.showStandardToast({ message: 'Failed to mark all notifications as read.', type: 'error' });
        });
    },

    // --- Standard Toast Handling (for new persistent notifications) ---
    showStandardToast: function(notification) {
        if (!this.elements.toastContainer) return;

        const toastElement = this.createToastElement(notification.message, notification.type);
        this.elements.toastContainer.appendChild(toastElement);

        const bsToast = new bootstrap.Toast(toastElement, {
            autohide: true,
            delay: this.config.liveToastDefaults.delay // Use same default delay
        });

        toastElement.addEventListener('hidden.bs.toast', () => toastElement.remove());
        bsToast.show();
        this.limitStandardToastCount(); // Limit standard toasts
    },

    limitStandardToastCount: function() {
        const standardToasts = this.elements.toastContainer.querySelectorAll('.toast:not([data-operation-id])'); // Select only standard toasts
        if (standardToasts.length > this.config.maxToasts) {
            console.log(`Limiting standard toasts, removing ${standardToasts.length - this.config.maxToasts} oldest.`);
            for (let i = 0; i < standardToasts.length - this.config.maxToasts; i++) {
                const oldToast = standardToasts[i];
                const bsOldToast = bootstrap.Toast.getInstance(oldToast);
                if (bsOldToast) {
                    bsOldToast.hide(); // This will trigger removal via 'hidden.bs.toast' listener
                } else {
                    oldToast.remove(); // Fallback removal
                }
            }
        }
    },


    // --- Live Operation Toast Handling ---

    /**
     * Shows a persistent toast for an ongoing operation.
     * @param {string} operationId - A unique ID for this operation.
     * @param {string} message - The initial message.
     * @param {string} type - Notification type (info, success, warning, error).
     * @param {boolean} showSpinner - Whether to show a spinner.
     * @param {number|null} progress - Initial progress value (0-100), null for no progress bar.
     */
    showLiveToast: function(operationId, message, type = 'info', showSpinner = false, progress = null) {
        if (!this.elements.toastContainer) {
             console.error("Toast container not available for live toast.");
             return;
        }
        if (this.state.activeLiveToasts[operationId]) {
            console.warn(`Live toast with ID ${operationId} already exists. Updating instead.`);
            this.updateLiveToast(operationId, message, type, showSpinner, progress);
            return;
        }
         console.log(`Showing live toast: ${operationId} - ${message}`);

        const toastElement = this.createToastElement(message, type, showSpinner, progress);
        toastElement.dataset.operationId = operationId; // Mark as live toast

        this.elements.toastContainer.appendChild(toastElement);

        const bsToast = new bootstrap.Toast(toastElement, {
            autohide: false // Live toasts do not autohide initially
        });

        // Store reference
        this.state.activeLiveToasts[operationId] = { element: toastElement, bsToast: bsToast };

        // Add listener for manual removal case (though completeLiveToast is preferred)
        toastElement.addEventListener('hidden.bs.toast', () => {
            toastElement.remove();
            delete this.state.activeLiveToasts[operationId];
             console.log(`Live toast ${operationId} removed.`);
        });

        bsToast.show();
    },

    /**
     * Updates an existing live toast.
     * @param {string} operationId - The ID of the toast to update.
     * @param {string} message - The new message.
     * @param {string} [type] - Optional new type.
     * @param {boolean|null} [showSpinner] - Optional new spinner state (null to keep current).
     * @param {number|null} [progress] - Optional new progress value (null to keep current).
     */
    updateLiveToast: function(operationId, message, type = null, showSpinner = null, progress = null) {
        const liveToast = this.state.activeLiveToasts[operationId];
        if (!liveToast) {
            console.warn(`Cannot update live toast: ID ${operationId} not found.`);
            return;
        }
         console.log(`Updating live toast: ${operationId} - ${message}`);

        const toastElement = liveToast.element;
        const bodyElement = toastElement.querySelector('.toast-body');
        const headerElement = toastElement.querySelector('.toast-header');
        const iconElement = headerElement?.querySelector('i');
        const titleElement = headerElement?.querySelector('.me-auto');
        const messageSpan = bodyElement?.querySelector('.live-toast-message');
        const spinnerElement = bodyElement?.querySelector('.spinner-border');
        const progressElement = bodyElement?.querySelector('.progress');
        const progressBar = progressElement?.querySelector('.progress-bar');

        // Update Type (Icon, Header Text, Border)
        if (type && toastElement.classList.contains(type)) {
             const currentType = [...toastElement.classList].find(cls => ['info', 'success', 'warning', 'error'].includes(cls));
             if (currentType) toastElement.classList.remove(currentType);
             toastElement.classList.add(type);
             if (iconElement) iconElement.className = `${this.getIconForType(type)} me-2`;
             if (titleElement) titleElement.textContent = this.getHeaderForType(type);
        }

        // Update Message
        if (messageSpan) {
            messageSpan.textContent = message;
        } else if (bodyElement) {
             // Fallback if structure wasn't as expected initially
             bodyElement.innerHTML = `<span class="live-toast-message">${message}</span>` + (bodyElement.innerHTML.includes('spinner') ? this.getSpinnerHtml() : '') + (bodyElement.innerHTML.includes('progress') ? this.getProgressHtml(progress ?? 0) : '');
        }


        // Update Spinner
        if (showSpinner !== null) {
            if (showSpinner && !spinnerElement) {
                bodyElement?.insertAdjacentHTML('beforeend', this.getSpinnerHtml());
            } else if (!showSpinner && spinnerElement) {
                spinnerElement.remove();
            }
        }

        // Update Progress Bar
        if (progress !== null) {
            if (progress >= 0 && progress <= 100) {
                if (!progressElement && bodyElement) {
                     bodyElement.insertAdjacentHTML('beforeend', this.getProgressHtml(progress));
                } else if (progressBar) {
                    progressBar.style.width = `${progress}%`;
                    progressBar.setAttribute('aria-valuenow', progress);
                    progressBar.textContent = `${progress}%`; // Optional: show text
                    if (progressElement) progressElement.style.display = ''; // Ensure visible
                }
            } else if (progressElement) {
                 progressElement.style.display = 'none'; // Hide if progress is invalid/nullified
            }
        } else if (progressElement && showSpinner === true) {
             // Hide progress if spinner is explicitly shown and progress is not updated
             progressElement.style.display = 'none';
        }

    },

    /**
     * Completes a live toast, updating its message/type and making it autohide.
     * @param {string} operationId - The ID of the toast to complete.
     * @param {string} message - The final message (e.g., "Success!").
     * @param {string} type - The final type (e.g., 'success' or 'error').
     * @param {number} [delay] - Optional hide delay (defaults to config).
     */
    completeLiveToast: function(operationId, message, type = 'success', delay = null) {
        const liveToast = this.state.activeLiveToasts[operationId];
        if (!liveToast) {
            console.warn(`Cannot complete live toast: ID ${operationId} not found.`);
            // Optionally create a standard toast if the live one is missing
            this.showStandardToast({ message: message, type: type });
            return;
        }
         console.log(`Completing live toast: ${operationId} - ${message}`);

        // Update final state (remove spinner/progress)
        this.updateLiveToast(operationId, message, type, false, -1); // Progress -1 hides bar

        // Make it autohide
        const hideDelay = delay ?? this.config.liveToastDefaults.delay;
        // liveToast.bsToast.update({ autohide: true, delay: hideDelay }); // .update() is not a valid BS5 method
        // Attempt to re-show the toast to apply autohide. Note: BS5 might require dispose/recreate for option changes.
        // We might need to manually set a timeout to hide if this doesn't work reliably.
        liveToast.bsToast.show();

        // Reference will be removed by the 'hidden.bs.toast' listener added in showLiveToast
    },

    // --- Live Operation Listeners (Integrated from live-notifications.js) ---
    setupLiveOperationListeners: function() {
        // Removed call to setupImageUploadListener
        // Removed call to setupImageSendingListener
        this.setupScheduleEventsListener();
         console.log("Live operation listeners set up.");
    },

    // setupImageUploadListener function removed

    // setupImageSendingListener function removed

    setupScheduleEventsListener: function() {
        // This relies on other parts of the application using window.postMessage
        window.addEventListener('message', (event) => {
            if (event.data && event.data.type === 'schedule-status') {
                const { action, details, operationId } = event.data;
                const id = operationId || `schedule-${Date.now()}`; // Use provided ID or generate one

                if (action === 'start') {
                    this.showLiveToast(id, `Scheduled task started: ${details}`, 'info', true);
                } else if (action === 'progress') {
                     this.updateLiveToast(id, `Scheduled task progress: ${details}`, 'info', true); // Keep spinner for generic progress
                } else if (action === 'complete') {
                    this.completeLiveToast(id, `Scheduled task finished: ${details}`, 'success');
                } else if (action === 'error') {
                    this.completeLiveToast(id, `Scheduled task failed: ${details}`, 'error', 10000);
                }
            }
        });
         console.log("Schedule event listener set up.");
    },


    // --- Helper Functions ---
    createToastElement: function(message, type = 'info', showSpinner = false, progress = null) {
        const toast = document.createElement('div');
        // Base classes + type class
        toast.className = `toast notification-toast ${type}`;
        toast.setAttribute('role', 'alert');
        toast.setAttribute('aria-live', 'assertive');
        toast.setAttribute('aria-atomic', 'true');

        const iconClass = this.getIconForType(type);
        const headerText = this.getHeaderForType(type);

        let bodyContent = `<span class="live-toast-message">${message}</span>`;
        if (showSpinner) {
            bodyContent += this.getSpinnerHtml();
        }
        if (progress !== null && progress >= 0 && progress <= 100) {
            bodyContent += this.getProgressHtml(progress);
        }

        toast.innerHTML = `
            <div class="toast-header">
                <i class="${iconClass} me-2" aria-hidden="true"></i>
                <strong class="me-auto">${headerText}</strong>
                <small class="text-muted">Just now</small> <!-- Consider updating this dynamically -->
                <button type="button" class="btn-close" data-bs-dismiss="toast" aria-label="Close"></button>
            </div>
            <div class="toast-body">
                ${bodyContent}
            </div>
        `;
        return toast;
    },

    getSpinnerHtml: function() {
        return ` <span class="spinner-border spinner-border-sm ms-2" role="status" aria-hidden="true"></span>`;
    },

    getProgressHtml: function(progressValue) {
         const clampedProgress = Math.max(0, Math.min(100, progressValue));
         return `
            <div class="progress mt-2" style="height: 6px;">
                <div class="progress-bar" role="progressbar" style="width: ${clampedProgress}%;"
                     aria-valuenow="${clampedProgress}" aria-valuemin="0" aria-valuemax="100">${clampedProgress}%</div>
            </div>
        `;
    },

    getIconForType: function(type) {
        switch (type) {
            case 'info': return 'fas fa-info-circle';
            case 'success': return 'fas fa-check-circle';
            case 'warning': return 'fas fa-exclamation-triangle';
            case 'error': return 'fas fa-times-circle';
            default: return 'fas fa-bell';
        }
    },

    getHeaderForType: function(type) {
        switch (type) {
            case 'info': return 'Information';
            case 'success': return 'Success';
            case 'warning': return 'Warning';
            case 'error': return 'Error';
            default: return 'Notification';
        }
    }
};

// --- Global Helper (Optional but convenient) ---
/**
 * Creates a persistent notification via the backend API.
 * Use NotificationSystem.showLiveToast for transient operation status.
 * @param {string} message - The notification message.
 * @param {string} type - The notification type: 'info', 'success', 'warning', 'error'.
 * @returns {Promise} - A promise that resolves with the created notification data or rejects on error.
 */
function createPersistentNotification(message, type = 'info') {
    console.log(`API call: Create persistent notification - ${message}`);
    return fetch('/api/notifications', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message, type }),
    })
    .then(response => {
        if (!response.ok) {
             return response.json().then(err => Promise.reject(err.error || `HTTP error! status: ${response.status}`));
        }
        return response.json();
    })
    .catch(error => {
        console.error('Error creating persistent notification:', error);
        // Optionally show an error toast *here* using the live system
        NotificationSystem.showStandardToast({ message: `Failed to save notification: ${error}`, type: 'error' });
        return Promise.reject(error); // Re-reject so calling code knows it failed
    });
}


// --- Initialize ---
// Use a slight delay or ensure this runs after the main DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => NotificationSystem.init());
} else {
    NotificationSystem.init(); // DOMContentLoaded has already fired
}