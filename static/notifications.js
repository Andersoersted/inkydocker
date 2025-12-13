/**
 * InkyDocker Notification System
 * Centralized notification handling for consistent user feedback
 */

const InkyNotify = {
  /**
   * Show a notification toast
   * @param {string} message - Message to display
   * @param {string} type - Notification type: 'success', 'error', 'info', 'warning'
   * @param {number} duration - Auto-hide duration in ms (0 = no auto-hide)
   */
  show: function(message, type = 'info', duration = 5000) {
    const toastContainer = this.getOrCreateContainer();
    const toastEl = this.createToast(message, type);

    toastContainer.appendChild(toastEl);

    const toast = new bootstrap.Toast(toastEl, {
      autohide: duration > 0,
      delay: duration
    });

    toast.show();

    // Remove from DOM after hidden
    toastEl.addEventListener('hidden.bs.toast', function() {
      toastEl.remove();
    });
  },

  /**
   * Show success notification
   */
  success: function(message, duration = 5000) {
    this.show(message, 'success', duration);
  },

  /**
   * Show error notification
   */
  error: function(message, duration = 8000) {
    this.show(message, 'error', duration);
  },

  /**
   * Show info notification
   */
  info: function(message, duration = 5000) {
    this.show(message, 'info', duration);
  },

  /**
   * Show warning notification
   */
  warning: function(message, duration = 6000) {
    this.show(message, 'warning', duration);
  },

  /**
   * Get or create the toast container
   */
  getOrCreateContainer: function() {
    let container = document.getElementById('inky-toast-container');
    if (!container) {
      container = document.createElement('div');
      container.id = 'inky-toast-container';
      container.className = 'toast-container position-fixed bottom-0 end-0 p-3';
      container.style.zIndex = '1100';
      document.body.appendChild(container);
    }
    return container;
  },

  /**
   * Create a toast element
   */
  createToast: function(message, type) {
    const toastEl = document.createElement('div');
    toastEl.className = 'toast align-items-center border-0';
    toastEl.setAttribute('role', 'alert');
    toastEl.setAttribute('aria-live', 'assertive');
    toastEl.setAttribute('aria-atomic', 'true');

    const config = this.getTypeConfig(type);
    toastEl.classList.add(...config.classes);

    toastEl.innerHTML = `
      <div class="d-flex">
        <div class="toast-body">
          <i class="${config.icon} me-2"></i>${message}
        </div>
        <button type="button" class="btn-close ${config.closeClass} me-2 m-auto"
                data-bs-dismiss="toast" aria-label="Close"></button>
      </div>
    `;

    return toastEl;
  },

  /**
   * Get configuration for notification type
   */
  getTypeConfig: function(type) {
    const configs = {
      'success': {
        classes: ['text-white', 'bg-success'],
        icon: 'fas fa-check-circle',
        closeClass: 'btn-close-white'
      },
      'error': {
        classes: ['text-white', 'bg-danger'],
        icon: 'fas fa-exclamation-circle',
        closeClass: 'btn-close-white'
      },
      'warning': {
        classes: ['text-dark', 'bg-warning'],
        icon: 'fas fa-exclamation-triangle',
        closeClass: ''
      },
      'info': {
        classes: ['text-white', 'bg-info'],
        icon: 'fas fa-info-circle',
        closeClass: 'btn-close-white'
      }
    };

    return configs[type] || configs['info'];
  },

  /**
   * Update inline status element (for loading states)
   * @param {HTMLElement|string} element - Element or selector
   * @param {string} message - Message to display
   * @param {string} type - Status type: 'loading', 'success', 'error', 'info'
   */
  updateStatus: function(element, message, type = 'info') {
    const el = typeof element === 'string' ? document.querySelector(element) : element;
    if (!el) return;

    const alertClass = {
      'loading': 'alert-info',
      'success': 'alert-success',
      'error': 'alert-danger',
      'info': 'alert-info',
      'warning': 'alert-warning'
    }[type] || 'alert-info';

    const icon = {
      'loading': '<span class="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true"></span>',
      'success': '<i class="fas fa-check-circle me-2"></i>',
      'error': '<i class="fas fa-exclamation-circle me-2"></i>',
      'info': '<i class="fas fa-info-circle me-2"></i>',
      'warning': '<i class="fas fa-exclamation-triangle me-2"></i>'
    }[type] || '';

    el.innerHTML = `<div class="alert ${alertClass}">${icon}${message}</div>`;
  },

  /**
   * Clear inline status element
   */
  clearStatus: function(element) {
    const el = typeof element === 'string' ? document.querySelector(element) : element;
    if (el) el.innerHTML = '';
  }
};

// Make globally available
window.InkyNotify = InkyNotify;

// Auto-display Flask flash messages on page load
document.addEventListener('DOMContentLoaded', function() {
  // Flask flash messages will be rendered as data attributes
  const flashContainer = document.getElementById('flask-flash-messages');
  if (flashContainer) {
    const messages = JSON.parse(flashContainer.dataset.messages || '[]');
    messages.forEach(function(msg) {
      InkyNotify.show(msg.message, msg.category);
    });
  }
});
