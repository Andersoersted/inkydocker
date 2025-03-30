// Function to initialize lazy loading
function initLazyLoading() {
  if ('IntersectionObserver' in window) {
    const lazyImageObserver = new IntersectionObserver((entries, observer) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          const lazyImage = entry.target;
          if (lazyImage.dataset.src) {
            // Set up the onload handler before changing src
            lazyImage.onload = function() {
              // Add loaded class for fade-in effect
              lazyImage.classList.add('loaded');
              // Add fade-in class for additional animation
              lazyImage.classList.add('fade-in');
              // Remove the onload handler to prevent memory leaks
              this.onload = null;
            };

            // Set the src to trigger loading
            lazyImage.src = lazyImage.dataset.src;
            lazyImageObserver.unobserve(lazyImage);
          }
        }
      });
    });

    // Observe all images with the 'lazy' class
    // Query within the function scope to pick up newly added images too
    document.querySelectorAll('img.lazy:not(.loaded)').forEach(lazyImage => {
      lazyImageObserver.observe(lazyImage);
    });
  } else {
    // Fallback for browsers that don't support IntersectionObserver
    document.querySelectorAll('img.lazy:not(.loaded)').forEach(img => {
      if (img.dataset.src) {
        // Set up the onload handler before changing src
        img.onload = function() {
          // Add loaded class for fade-in effect
          img.classList.add('loaded');
          // Add fade-in class for additional animation
          img.classList.add('fade-in');
          // Remove the onload handler to prevent memory leaks
          this.onload = null;
        };

        // Set the src to trigger loading
        img.src = img.dataset.src;
        img.classList.remove('lazy'); // Remove lazy class after setting src in fallback
      }
    });
  }
}

// Make the function globally available
window.initLazyLoading = initLazyLoading;


document.addEventListener('DOMContentLoaded', function() {
  // Initialize lazy loading on initial page load
  initLazyLoading();

  // Add Bootstrap 5 Toasts container (if not already handled by NotificationSystem)
  if (!document.querySelector('.toast-container')) {
      console.log("app.js: Creating toast container.");
      const toastContainer = document.createElement('div');
      toastContainer.className = 'toast-container position-fixed bottom-0 end-0 p-3';
      toastContainer.style.zIndex = '1100'; // Ensure toasts appear above most elements
      document.body.appendChild(toastContainer);
  }
});