/**
 * Enhanced Gallery with Masonry Layout, Lazy Loading, and Virtual Scrolling
 */
document.addEventListener('DOMContentLoaded', function() {
  // Initialize masonry layout
  setTimeout(function() {
    initMasonryLayout();
    // Initialize tag filtering
    initTagFiltering();
    // Initialize lazy loading
    initLazyLoading();
    // Initialize virtual scrolling
    initVirtualScrolling();
  }, 100);
});

/**
 * Initialize virtual scrolling for automatic loading of more images
 */
function initVirtualScrolling() {
  const gallery = document.querySelector('.gallery');
  if (!gallery) return;
  
  // Create variables to track loading state
  window.isLoading = false;
  window.allImagesLoaded = false;
  
  // Create a sentinel element that will trigger loading more images
  const bottomSentinel = document.createElement('div');
  bottomSentinel.id = 'bottom-sentinel';
  bottomSentinel.style.height = '10px';
  gallery.after(bottomSentinel);
  
  // Create intersection observer for bottom sentinel
  const bottomObserver = new IntersectionObserver((entries) => {
    if (entries[0].isIntersecting && !window.isLoading && !window.allImagesLoaded) {
      // When sentinel is visible, load more images
      const loadMoreBtn = document.getElementById('loadMoreBtn');
      if (loadMoreBtn) {
        // Simulate click on load more button
        loadMoreBtn.click();
      }
    }
  });
  
  bottomObserver.observe(bottomSentinel);
  
  // Modify the loadImages function to update loading state
  const loadMoreBtn = document.getElementById('loadMoreBtn');
  if (loadMoreBtn) {
    const originalLoadMoreClick = loadMoreBtn.onclick;
    loadMoreBtn.onclick = function() {
      window.isLoading = true;
      
      // Only call the original click handler if it exists
      if (originalLoadMoreClick) {
        originalLoadMoreClick.apply(this);
      } else {
        // Fallback behavior if no click handler exists yet
        if (typeof window.currentPage !== 'undefined') {
          window.currentPage++;
          if (typeof window.loadImages === 'function') {
            window.loadImages(window.currentPage, true);
          }
        }
      }
      
      // Hide the button while loading
      this.style.display = 'none';
    };
  }
  
  // Patch the existing code to update loading state after images are loaded
  const originalLoadImages = window.loadImages;
  if (originalLoadImages) {
    window.loadImages = function(page, append) {
      window.isLoading = true;
      
      // Call original function
      originalLoadImages(page, append);
      
      // Add event listener to detect when loading is complete
      const loadingSpinner = document.getElementById('loadingSpinner');
      
      // Create a mutation observer to watch for changes to the loading spinner
      const observer = new MutationObserver((mutations) => {
        mutations.forEach((mutation) => {
          if (mutation.attributeName === 'style' &&
              loadingSpinner.style.display === 'none') {
            // Loading is complete
            window.isLoading = false;
            
            // Check if all images are loaded
            const loadMoreBtn = document.getElementById('loadMoreBtn');
            window.allImagesLoaded = loadMoreBtn.style.display === 'none';
            
            // Show the button again if there are more images
            if (!window.allImagesLoaded) {
              loadMoreBtn.style.display = 'block';
            }
            
            // Disconnect the observer
            observer.disconnect();
          }
        });
      });
      
      // Start observing the loading spinner
      observer.observe(loadingSpinner, { attributes: true });
    };
  }
}

/**
 * Initialize lazy loading for images with fade-in effect and optimized loading
 */
function initLazyLoading() {
  // Check if IntersectionObserver is supported
  if ('IntersectionObserver' in window) {
    // Use a more efficient root margin to start loading images before they enter the viewport
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
              
              // Notify the masonry layout that an image has loaded
              if (typeof initMasonryLayout === 'function') {
                // Use a small timeout to ensure the image is fully rendered
                setTimeout(() => {
                  initMasonryLayout();
                }, 50);
              }
            };
            
            // Set the src to trigger loading
            lazyImage.src = lazyImage.dataset.src;
            lazyImageObserver.unobserve(lazyImage);
          }
        }
      });
    }, {
      rootMargin: '200px 0px', // Start loading images when they're 200px from entering the viewport
      threshold: 0.01 // Trigger when just 1% of the image is visible
    });

    // Observe all images with the 'lazy' class
    const lazyImages = document.querySelectorAll('img.lazy');
    lazyImages.forEach(lazyImage => {
      lazyImageObserver.observe(lazyImage);
    });
  } else {
    // Fallback for browsers that don't support IntersectionObserver
    document.querySelectorAll('img.lazy').forEach(img => {
      if (img.dataset.src) {
        // Set up the onload handler before changing src
        img.onload = function() {
          // Add loaded class for fade-in effect
          img.classList.add('loaded');
          // Add fade-in class for additional animation
          img.classList.add('fade-in');
          // Remove the onload handler to prevent memory leaks
          this.onload = null;
          
          // Notify the masonry layout that an image has loaded
          if (typeof initMasonryLayout === 'function') {
            setTimeout(() => {
              initMasonryLayout();
            }, 50);
          }
        };
        
        // Set the src to trigger loading
        img.src = img.dataset.src;
      }
    });
  }
}

/**
 * Initialize masonry layout for gallery items
 */
function initMasonryLayout() {
  const gallery = document.querySelector('.gallery');
  if (!gallery) return;
  
  const items = gallery.querySelectorAll('.gallery-item');
  
  // Check if this is a modal gallery (in schedule.html) or main gallery (in index.html)
  const isModalGallery = gallery.id === 'galleryModal';
  
  // For modal galleries, we don't need the masonry layout
  if (isModalGallery) {
    // Just make sure all images are visible
    items.forEach(item => {
      const img = item.querySelector('img');
      if (img && img.getAttribute('data-src')) {
        img.src = img.getAttribute('data-src');
        img.classList.remove('lazy');
      }
      
      // Make sure overlay is visible on hover
      const overlay = item.querySelector('.overlay');
      if (overlay) {
        item.addEventListener('mouseenter', () => {
          overlay.style.opacity = '1';
        });
        item.addEventListener('mouseleave', () => {
          overlay.style.opacity = '0';
        });
      }
    });
    return;
  }
  
  // For the main gallery, use a simpler layout
  // Observe each gallery item
  items.forEach(item => {
    // Make sure overlay is visible on hover
    const overlay = item.querySelector('.overlay');
    if (overlay) {
      item.addEventListener('mouseenter', () => {
        overlay.style.opacity = '1';
      });
      item.addEventListener('mouseleave', () => {
        overlay.style.opacity = '0';
      });
    }
  });
}


/**
 * Initialize tag filtering for gallery
 */
function initTagFiltering() {
  const gallery = document.querySelector('.gallery');
  if (!gallery) return;
  
  // Skip tag filtering for modal galleries
  if (gallery.id === 'galleryModal') return;
  
  const items = gallery.querySelectorAll('.gallery-item');
  
  // Extract all unique tags from gallery items
  const allTags = new Set();
  items.forEach(item => {
    const tags = item.getAttribute('data-tags');
    if (tags) {
      tags.split(',').forEach(tag => {
        tag = tag.trim();
        if (tag) allTags.add(tag);
      });
    }
  });
  
  // Create filter buttons if we have tags
  if (allTags.size > 0) {
    const filterContainer = document.createElement('div');
    filterContainer.className = 'gallery-filters';
    
    // Add "All" filter
    const allFilter = document.createElement('div');
    allFilter.className = 'tag-filter active';
    allFilter.textContent = 'All';
    allFilter.addEventListener('click', () => {
      // Remove active class from all filters
      document.querySelectorAll('.tag-filter').forEach(f => f.classList.remove('active'));
      // Add active class to this filter
      allFilter.classList.add('active');
      // Show all items
      items.forEach(item => item.style.display = '');
      // Recalculate layout
      setTimeout(initMasonryLayout, 100);
    });
    filterContainer.appendChild(allFilter);
    
    // Add filter for each tag
    allTags.forEach(tag => {
      const filter = document.createElement('div');
      filter.className = 'tag-filter';
      filter.textContent = tag;
      filter.addEventListener('click', () => {
        // Remove active class from all filters
        document.querySelectorAll('.tag-filter').forEach(f => f.classList.remove('active'));
        // Add active class to this filter
        filter.classList.add('active');
        // Filter items
        items.forEach(item => {
          const itemTags = item.getAttribute('data-tags') || '';
          if (itemTags.includes(tag)) {
            item.style.display = '';
          } else {
            item.style.display = 'none';
          }
        });
        // Recalculate layout
        setTimeout(initMasonryLayout, 100);
      });
      filterContainer.appendChild(filter);
    });
    
    // Insert filter container before gallery
    gallery.parentNode.insertBefore(filterContainer, gallery);
  }
}