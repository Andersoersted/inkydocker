/**
 * Enhanced Gallery with Masonry Layout and Lazy Loading
 */
document.addEventListener('DOMContentLoaded', function() {
  // Initialize masonry layout
  initMasonryLayout();
  
  // Initialize lazy loading
  initLazyLoading();
  
  // Initialize tag filtering
  initTagFiltering();
});

/**
 * Initialize masonry layout for gallery items
 */
function initMasonryLayout() {
  const gallery = document.querySelector('.gallery');
  if (!gallery) return;
  
  const items = gallery.querySelectorAll('.gallery-item');
  
  // Add resize observer to recalculate layout when images load
  const resizeObserver = new ResizeObserver(entries => {
    for (let entry of entries) {
      const item = entry.target;
      const height = item.getBoundingClientRect().height;
      
      // Calculate how many rows this item should span
      // Each row is 10px (defined in CSS)
      const rowSpan = Math.ceil(height / 10);
      item.style.gridRowEnd = `span ${rowSpan}`;
    }
  });
  
  // Observe each gallery item
  items.forEach(item => {
    // Add placeholder while image loads
    const placeholder = document.createElement('div');
    placeholder.className = 'img-placeholder';
    item.appendChild(placeholder);
    
    // Start observing for size changes
    resizeObserver.observe(item);
  });
}

/**
 * Initialize lazy loading for gallery images
 */
function initLazyLoading() {
  const images = document.querySelectorAll('.gallery .img-container img');
  
  if ('IntersectionObserver' in window) {
    const imageObserver = new IntersectionObserver((entries, observer) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          const img = entry.target;
          const src = img.getAttribute('data-src') || img.getAttribute('src');
          
          // Create a new image to preload
          const newImg = new Image();
          newImg.src = src;
          
          newImg.onload = function() {
            // When image is loaded, update the visible image
            img.src = src;
            img.classList.add('loaded');
            
            // Remove placeholder
            const placeholder = img.closest('.gallery-item').querySelector('.img-placeholder');
            if (placeholder) {
              placeholder.style.opacity = 0;
              setTimeout(() => {
                if (placeholder.parentNode) {
                  placeholder.parentNode.removeChild(placeholder);
                }
              }, 300);
            }
            
            // Stop observing the image
            imageObserver.unobserve(img);
          };
        }
      });
    }, {
      rootMargin: '50px 0px',
      threshold: 0.01
    });
    
    // Start observing each image
    images.forEach(img => {
      imageObserver.observe(img);
      
      // Set initial state
      if (!img.getAttribute('data-src')) {
        img.setAttribute('data-src', img.getAttribute('src'));
      }
      img.removeAttribute('src');
    });
  } else {
    // Fallback for browsers that don't support IntersectionObserver
    images.forEach(img => {
      img.classList.add('loaded');
    });
  }
}

/**
 * Initialize tag filtering for gallery
 */
function initTagFiltering() {
  const gallery = document.querySelector('.gallery');
  if (!gallery) return;
  
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