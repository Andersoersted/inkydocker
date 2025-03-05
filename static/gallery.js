/**
 * Enhanced Gallery with Masonry Layout and Lazy Loading
 */
document.addEventListener('DOMContentLoaded', function() {
  // Initialize masonry layout
  setTimeout(function() {
    initMasonryLayout();
    // Initialize tag filtering
    initTagFiltering();
  }, 100);
});

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
  
  // For the main gallery, use the masonry layout
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
    // Make sure all images are visible
    const img = item.querySelector('img');
    if (img && img.getAttribute('data-src')) {
      img.src = img.getAttribute('data-src');
    }
    
    // Start observing for size changes
    resizeObserver.observe(item);
    
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