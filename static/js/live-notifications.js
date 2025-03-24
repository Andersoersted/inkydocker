/**
 * Live Notifications System
 * 
 * Extends the notification system to replace spinners and progress bars with
 * live, persistent notifications.
 */

document.addEventListener('DOMContentLoaded', function() {
    // Initialize the live notification system
    LiveNotifications.init();
});

const LiveNotifications = {
    // Configuration
    config: {
        uploadUpdateFrequency: 10, // Update progress notification every 10% during upload
        progressClasses: "progress-notification",
        uploadNotificationId: null,
        sendingNotificationId: null,
        screenshotNotificationId: null,
        displayAdditionNotificationId: null,
        scheduledEventNotificationId: null,
    },

    // Active operation tracking
    operations: {
        activeUploads: {},
        activeSends: {},
        activeScreenshots: {},
        activeSchedules: {},
    },
    
    /**
     * Initialize the live notification system
     */
    init: function() {
        // Initialize only after NotificationSystem is ready
        if (typeof NotificationSystem === 'undefined') {
            setTimeout(() => this.init(), 100);
            return;
        }

        // Override upload form behavior
        this.setupImageUpload();
        
        // Override image sending behavior
        this.setupImageSending();
        
        // Override Bootstrap modal events to handle schedule events
        this.setupScheduleEvents();
    },
    
    /**
     * Set up image upload with notification instead of progress bar
     */
    setupImageUpload: function() {
        const uploadForm = document.getElementById('uploadForm');
        if (!uploadForm) return;
        
        // Replace the original submit handler
        uploadForm.addEventListener('submit', function(e) {
            e.preventDefault();
            
            const fileInput = document.getElementById('fileInput');
            if (!fileInput.files.length) return;
            
            // Create notification for upload start
            createNotification("Starting image upload...", "info")
                .then(response => {
                    LiveNotifications.config.uploadNotificationId = response.id;
                });
            
            const formData = new FormData();
            for (let i = 0; i < fileInput.files.length; i++) {
                formData.append('file', fileInput.files[i]);
            }
            
            const xhr = new XMLHttpRequest();
            xhr.open('POST', uploadForm.action, true);
            
            // Track progress
            let lastProgressPercentage = 0;
            xhr.upload.addEventListener("progress", function(e) {
                if (e.lengthComputable) {
                    const percentComplete = Math.round((e.loaded / e.total) * 100);
                    
                    // Update the progress bar (keep it for backward compatibility)
                    const progressBar = document.getElementById('progressBar');
                    if (progressBar) {
                        progressBar.style.width = percentComplete + '%';
                        progressBar.textContent = percentComplete + '%';
                        document.getElementById('progressContainer').style.display = 'block';
                    }
                    
                    // Update notification only when progress increases by 10%
                    if (percentComplete >= lastProgressPercentage + 10) {
                        lastProgressPercentage = percentComplete;
                        const message = `Uploading images: ${percentComplete}% complete`;
                        
                        createNotification(message, "info");
                    }
                }
            });
            
            xhr.onload = function() {
                if (xhr.status === 200) {
                    // Create success notification
                    createNotification("Images uploaded successfully!", "success");
                    
                    // Hide progress container
                    const progressContainer = document.getElementById('progressContainer');
                    if (progressContainer) {
                        progressContainer.style.display = 'none';
                    }
                    
                    // Refresh gallery
                    setTimeout(function() {
                        if (typeof currentPage !== 'undefined' && typeof loadImages === 'function') {
                            currentPage = 1;
                            loadImages(1);
                        }
                    }, 1500);
                } else {
                    createNotification("Error uploading images", "error");
                }
            };
            
            xhr.onerror = function() {
                createNotification("Error uploading images", "error");
            };
            
            xhr.send(formData);
        }, { capture: true }); // Use capture to ensure this runs before other handlers
    },
    
    /**
     * Set up image sending with notification instead of spinner
     */
    setupImageSending: function() {
        // Intercept clicks on send buttons
        document.addEventListener('click', function(e) {
            if (e.target && e.target.classList.contains('send-button')) {
                // Get the original handler to execute
                const imageFilename = e.target.getAttribute('data-image');
                const selectedDevice = document.querySelector('input[name="device"]:checked');
                if (!selectedDevice) return;
                
                // Don't stop propagation, let the original handler run but add our notification
                const deviceFriendly = selectedDevice.getAttribute('data-friendly');
                
                // Create notification for sending start
                createNotification(`Sending image: ${imageFilename} to ${deviceFriendly}...`, "info")
                    .then(response => {
                        LiveNotifications.config.sendingNotificationId = response.id;
                    });
                
                // Find the original handler and let it continue
                // The success/error notification will be created by the original handler's callback
            }
        });
    },
    
    /**
     * Set up scheduled event handling
     */
    setupScheduleEvents: function() {
        // Listen for schedule event execution
        // This typically happens in a separate handler that's already implemented
        // but we'll set up a global message listener to handle this
        
        window.addEventListener('message', function(event) {
            // Check if this is a scheduled event message
            if (event.data && event.data.type === 'schedule') {
                if (event.data.action === 'start') {
                    createNotification(`Scheduled event started: ${event.data.details}`, "info")
                        .then(response => {
                            LiveNotifications.config.scheduledEventNotificationId = response.id;
                        });
                } else if (event.data.action === 'complete') {
                    createNotification(`Scheduled event completed: ${event.data.details}`, "success");
                }
            }
        });
        
        // For demo/testing purposes, create a helper to trigger these events
        window.triggerScheduleEvent = function(action, details) {
            window.postMessage({
                type: 'schedule',
                action: action,
                details: details
            }, '*');
        };
    }
};

/**
 * Create a live toast notification that updates automatically
 * 
 * @param {string} id - A unique ID for the notification
 * @param {string} message - The notification message
 * @param {number} progress - Progress value (0-100)
 * @param {string} type - Notification type (info, success, warning, error)
 */
function createLiveNotification(id, message, progress, type = 'info') {
    // Check if notification exists with this ID
    const existingToast = document.querySelector(`.toast[data-notification-id="${id}"]`);
    
    if (existingToast) {
        // Update existing notification
        const progressBar = existingToast.querySelector('.progress-bar');
        if (progressBar) {
            progressBar.style.width = `${progress}%`;
            progressBar.setAttribute('aria-valuenow', progress);
        }
        
        const messageElement = existingToast.querySelector('.toast-body .message');
        if (messageElement) {
            messageElement.textContent = message;
        }
        
        // If progress is 100, prepare to remove
        if (progress >= 100) {
            setTimeout(() => {
                // Remove toast
                const bsToast = bootstrap.Toast.getInstance(existingToast);
                if (bsToast) bsToast.hide();
            }, 1000);
        }
    } else {
        // Create new notification
        createNotification(`${message} (${progress}% complete)`, type);
    }
    
    return true;
}