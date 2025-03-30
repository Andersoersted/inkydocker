console.log("gallery_page.js: Script start");
// --- Global Scope Variables & Functions ---

// Debounce helper
function debounce(func, wait) {
  let timeout;
  return function() {
    var context = this;
    var args = arguments;
    clearTimeout(timeout);
    timeout = setTimeout(function() {
      func.apply(context, args);
    }, wait);
  };
}

// Lightbox functions
function openLightbox(src, alt) {
  const lightboxModal = new bootstrap.Modal(document.getElementById('lightboxModal'));
  const lightboxImage = document.getElementById('lightboxImage');
  const lightboxCaption = document.getElementById('lightboxCaption');
  if (!lightboxModal || !lightboxImage || !lightboxCaption) {
      console.error("Lightbox elements not found");
      return;
  }
  lightboxImage.dataset.src = src;
  lightboxImage.src = src; // Also set src directly for immediate display
  lightboxImage.classList.remove('lazy');
  lightboxCaption.innerText = alt;
  lightboxModal.show();
}
function closeLightbox() {
  const lightboxModalEl = document.getElementById('lightboxModal');
  const lightboxModal = bootstrap.Modal.getInstance(lightboxModalEl);
  if (lightboxModal) {
    lightboxModal.hide();
  }
}

// Gallery state variables
let currentPage = 1;
const imagesPerPage = 20;
let isSearchMode = false;
let searchQuery = '';

// Function to create gallery item HTML
function createGalleryItemHTML(image) {
  const tags = image.tags ? image.tags.join(', ') : '';
  // Use the thumbnail route for the gallery view data-src
  const galleryUrl = `/thumbnail/${encodeURIComponent(image.filename)}`;

  return `
    <div class="gallery-item" data-tags="${tags}">
      <div class="img-container">
        <img src="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 300 200' width='300' height='200'%3E%3Crect width='300' height='200' fill='%23f8f9fa'/%3E%3Cpath d='M145,80 L155,80 L155,120 L145,120 Z' fill='%23dee2e6'/%3E%3Ccircle cx='150' cy='60' r='10' fill='%23dee2e6'/%3E%3Cpath d='M130,140 L170,140 L170,150 L130,150 Z' fill='%23dee2e6'/%3E%3C/svg%3E" data-src="${galleryUrl}" alt="${image.filename}" data-filename="${image.filename}" loading="lazy" class="lazy">
        <div class="overlay">
          <div class="favorite-icon ${image.favorite ? 'favorited' : ''}" title="Favorite" data-image="${image.filename}">
            <i class="fa fa-heart"></i>
          </div>
          <button class="send-button" data-image="${image.filename}">Send</button>
          <button class="info-button" data-image="${image.filename}">Info</button> <!-- Ensure info_modal.js handles this -->
          <div class="delete-icon" title="Delete" data-image="${image.filename}">
            <i class="fa fa-trash"></i>
          </div>
        </div>
      </div>
    </div>
  `;
}

// Function to load images
function loadImages(page = 1, append = false) {
  const gallery = document.getElementById('gallery');
  const loadingSpinner = document.getElementById('loadingSpinner');
  const loadMoreBtn = document.getElementById('loadMoreBtn');

  if (!gallery) {
      console.error("Gallery element not found. Cannot load images.");
      if (loadingSpinner) loadingSpinner.style.display = 'none'; // Hide spinner if gallery missing
      return;
  }
  if (!append) {
    gallery.innerHTML = ''; // Clear gallery only if not appending
  }

  if (loadingSpinner) loadingSpinner.style.display = 'block';
  if (loadMoreBtn) loadMoreBtn.style.display = 'none'; // Hide load more button while loading

  const url = isSearchMode
    ? `/api/search_images?q=${encodeURIComponent(searchQuery)}&page=${page}&per_page=${imagesPerPage}`
    : `/api/get_images?page=${page}&per_page=${imagesPerPage}`;

  fetch(url)
    .then(response => {
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        return response.json();
    })
    .then(data => {
      if (loadingSpinner) loadingSpinner.style.display = 'none';

      if (data.status === "success") {
        const images = isSearchMode ? data.results.ids : data.images; // Adjust based on API response structure

        if (images.length === 0 && page === 1) {
          gallery.innerHTML = "<p>No images found.</p>";
          // Keep load more hidden
          return;
        }

        images.forEach(imageInfo => {
          // Handle potential differences between search results (filenames) and get_images (objects)
          const filename = isSearchMode ? imageInfo : imageInfo.filename;
          const imageData = {
            filename: filename,
            url: `/images/${encodeURIComponent(filename)}`, // Keep original URL for other uses if needed
            // Provide defaults if search results don't include these details
            favorite: isSearchMode ? false : (imageInfo.favorite || false),
            tags: isSearchMode ? [] : (imageInfo.tags || [])
          };
          const itemHTML = createGalleryItemHTML(imageData);
          gallery.insertAdjacentHTML('beforeend', itemHTML);
        });

        // Show 'Load More' button only if a full page was loaded
        if (loadMoreBtn) {
            if (images.length >= imagesPerPage) {
              loadMoreBtn.style.display = 'block';
            } else {
              loadMoreBtn.style.display = 'none';
            }
        }

        // Initialize masonry layout (ensure masonry.js is loaded and initMasonryLayout is global or accessible)
        if (typeof initMasonryLayout === 'function') {
          setTimeout(initMasonryLayout, 100); // Delay slightly for images to potentially render
        } else {
            console.warn("initMasonryLayout function not found.");
        }

        // Initialize lazy loading (ensure app.js with initLazyLoading is loaded)
        if (typeof initLazyLoading === 'function') {
          initLazyLoading(); // Call directly if available
        } else {
            console.warn("initLazyLoading function not found. Lazy loading might not work.");
            // Basic fallback (less efficient)
            gallery.querySelectorAll('img.lazy[data-src]').forEach(img => {
                img.src = img.dataset.src;
                img.classList.remove('lazy');
            });
        }
      } else {
        console.error("Error loading images API response:", data.message || 'Unknown error');
        gallery.innerHTML = "<p>Error loading images.</p>"; // Show error in gallery
      }
    })
    .catch(err => {
      if (loadingSpinner) loadingSpinner.style.display = 'none';
      console.error("Fetch error loading images:", err);
      if (gallery) gallery.innerHTML = "<p>Error loading images.</p>"; // Show error in gallery
    });
}


// Search function (debounced)
const performSearch = debounce(function() {
  const searchInput = document.getElementById('gallerySearch');
  const searchSpinner = document.getElementById('searchSpinner');
  if (!searchInput) return; // Guard against missing element

  searchQuery = searchInput.value.trim();
  isSearchMode = !!searchQuery; // Set search mode based on query presence
  currentPage = 1; // Reset page when searching

  if (searchSpinner) searchSpinner.style.display = 'block';
  loadImages(1); // Call the globally defined loadImages, page 1, don't append
  // Spinner is hidden within loadImages
}, 500);

/* Function to update current image display */
function updateCurrentImageDisplay() {
  fetch('/api/get_current_image')
    .then(response => response.json())
    .then(data => {
      const currentImage = document.getElementById('currentImage');
      const currentImageTitle = document.getElementById('currentImageTitle');
      const currentImagePlaceholder = document.getElementById('currentImagePlaceholder');

      if (!currentImage || !currentImageTitle || !currentImagePlaceholder) {
          console.warn("Current image display elements not found.");
          return;
      }

      if (data.status === "success" && data.devices && data.devices.length > 0) {
          const device = data.devices[0]; // Assuming first device for now
          currentImageTitle.textContent = `Current image on ${device.device_name || 'N/A'}`;

          if (device.current_image) {
              // Determine if it's a screenshot or regular image based on your logic
              // This might need adjustment based on how screenshots are identified
              const isScreenshot = device.current_image.startsWith('screenshot_'); // Example check
              const imageUrl = isScreenshot
                  ? `/screenshots/${encodeURIComponent(device.current_image)}?cropped=true` // Adjust endpoint if needed
                  : device.display_url || `/images/${encodeURIComponent(device.current_image)}`;

              currentImage.src = imageUrl;
              currentImage.style.display = 'block';
              currentImagePlaceholder.style.display = 'none';
          } else {
              currentImage.style.display = 'none';
              currentImagePlaceholder.style.display = 'block';
              currentImagePlaceholder.textContent = 'No image sent yet.';
          }
      } else {
          // Handle no devices or error case
          currentImage.style.display = 'none';
          currentImageTitle.textContent = 'Current Image';
          currentImagePlaceholder.style.display = 'block';
          currentImagePlaceholder.textContent = (data.status !== "success")
              ? 'Error fetching status.'
              : 'No devices configured or no image sent.';
      }
    })
    .catch(error => {
      console.error("Error updating current image:", error);
      const currentImageTitle = document.getElementById('currentImageTitle');
      const currentImagePlaceholder = document.getElementById('currentImagePlaceholder');
      if (currentImageTitle) currentImageTitle.textContent = 'Current Image';
      if (currentImagePlaceholder) {
          currentImagePlaceholder.style.display = 'block';
          currentImagePlaceholder.textContent = 'Error fetching status.';
      }
    });
}

// Explicitly expose functions needed by other modules
window.loadImages = loadImages;
window.currentPage = currentPage; // Expose if needed, though modifying global vars directly is often discouraged


// --- DOMContentLoaded Listener ---
document.addEventListener("DOMContentLoaded", function() {
  console.log("gallery_page.js: DOMContentLoaded listener start");

  try {
    // Inject dynamic CSS for favorite icon
    const styleTag = document.createElement('style');
    styleTag.innerHTML = `
      .favorite-icon i { font-size: 1.5em; color: #ccc; transition: color 0.3s; }
      .favorite-icon.favorited i { color: red; }
    `;
    document.head.appendChild(styleTag);

    // Get DOM element references
    const searchInput = document.getElementById('gallerySearch');
    const loadMoreBtn = document.getElementById('loadMoreBtn');
    const uploadForm = document.getElementById('uploadForm');
    // Ensure gallery element exists before attaching listeners that depend on it
    const galleryElement = document.getElementById('gallery');

    // Initial actions
    console.log("gallery_page.js: Initializing gallery...");
    if (galleryElement) {
        console.log("gallery_page.js: Calling initial loadImages(1)");
        loadImages(1); // Load initial images only if gallery exists
    } else {
        console.error("Gallery element not found on DOMContentLoaded.");
    }
    updateCurrentImageDisplay(); // Update current image display on load

    // --- Event Listeners ---

    // Load More Button
    if (loadMoreBtn) {
      loadMoreBtn.addEventListener('click', function() {
        currentPage++;
        loadImages(currentPage, true); // Call global loadImages, append = true
      });
    } else {
        console.warn("Load More button not found.");
    }

    // Search Input
    if (searchInput) {
      searchInput.addEventListener('input', performSearch); // Call global performSearch
    } else {
        console.warn("Search input not found.");
    }

    // Upload Form - Handles its own notifications
    if (uploadForm) {
        uploadForm.addEventListener('submit', function(e) {
          e.preventDefault();
          const fileInput = document.getElementById('fileInput');
          if (!fileInput || !fileInput.files.length) return;

          const fileCount = fileInput.files.length;
          const operationId = `upload-${Date.now()}`;
          fileInput.dataset.uploadId = operationId; // Store for progress updates

          // Start notification using NotificationSystem
          if (typeof NotificationSystem !== 'undefined' && typeof NotificationSystem.showLiveToast === 'function') {
              NotificationSystem.showLiveToast(operationId, `Starting upload of ${fileCount} image(s)...`, 'info', false, 0);
          } else {
              console.warn("NotificationSystem.showLiveToast not available for upload start.");
          }

          const formData = new FormData(); // Use the form directly
          for (let i = 0; i < fileInput.files.length; i++) {
            formData.append('file', fileInput.files[i]);
          }

          const xhr = new XMLHttpRequest();
          xhr.open('POST', uploadForm.action, true);

          let lastReportedProgress = -1;
          xhr.upload.addEventListener("progress", function(event) {
            if (event.lengthComputable) {
              const percentComplete = Math.round((event.loaded / event.total) * 100);
              // Update legacy progress bar (optional)
              const progressBar = document.getElementById('progressBar');
              if (progressBar) {
                  progressBar.style.width = percentComplete + '%';
                  progressBar.textContent = percentComplete + '%';
                  const progressContainer = document.getElementById('progressContainer');
                  if (progressContainer) progressContainer.style.display = 'block';
              }
              // Update notification progress (throttled) using NotificationSystem
              if (percentComplete > lastReportedProgress) {
                   lastReportedProgress = percentComplete;
                   if (typeof NotificationSystem !== 'undefined' && typeof NotificationSystem.updateLiveToast === 'function') {
                       NotificationSystem.updateLiveToast(operationId, `Uploading ${fileCount} image(s)...`, 'info', false, percentComplete);
                   }
              }
            }
          });

          xhr.onload = function() {
            fileInput.value = ''; // Clear file input
            const legacyProgress = document.getElementById('progressContainer');
            if(legacyProgress) legacyProgress.style.display = 'none'; // Hide legacy progress bar
            const uploadStatus = document.getElementById('uploadStatus');
            if(uploadStatus) uploadStatus.textContent = ''; // Clear legacy status

            if (xhr.status >= 200 && xhr.status < 300) {
               // Complete notification with success using NotificationSystem
               if (typeof NotificationSystem !== 'undefined' && typeof NotificationSystem.completeLiveToast === 'function') {
                  NotificationSystem.completeLiveToast(operationId, `Successfully uploaded ${fileCount} image(s)!`, 'success');
               } else {
                   console.warn("NotificationSystem.completeLiveToast not available for upload success.");
               }
               // Refresh gallery after a short delay
               setTimeout(function() {
                 currentPage = 1;
                 loadImages(1); // Call global loadImages
                 console.log("Gallery refresh triggered by gallery_page.js after upload.");
               }, 1500);
            } else {
               // Handle upload error & complete notification with error using NotificationSystem
               let errorMsg = `Upload failed (Status: ${xhr.status})`;
               try {
                   const jsonResponse = JSON.parse(xhr.responseText);
                   errorMsg = jsonResponse.error || errorMsg;
               } catch (parseError) { /* Ignore */ }

               if (typeof NotificationSystem !== 'undefined' && typeof NotificationSystem.completeLiveToast === 'function') {
                  NotificationSystem.completeLiveToast(operationId, errorMsg, "error", 10000);
               } else {
                   console.error("NotificationSystem.completeLiveToast not available for upload error.");
                   if(uploadStatus) uploadStatus.textContent = 'Error uploading image.'; // Fallback status
               }
            }
          };

          xhr.onerror = function() {
            fileInput.value = '';
            const legacyProgress = document.getElementById('progressContainer');
            if(legacyProgress) legacyProgress.style.display = 'none';
            const uploadStatus = document.getElementById('uploadStatus');
            if(uploadStatus) uploadStatus.textContent = '';

             // Complete notification with network error using NotificationSystem
             if (typeof NotificationSystem !== 'undefined' && typeof NotificationSystem.completeLiveToast === 'function') {
                NotificationSystem.completeLiveToast(operationId, "Upload failed due to network error.", "error", 10000);
             } else {
                 console.error("NotificationSystem.completeLiveToast not available for upload network error.");
                 if(uploadStatus) uploadStatus.textContent = 'Error uploading image.'; // Fallback status
             }
          };

          xhr.send(formData);
        });
    } else {
        console.warn("Upload form not found.");
    }

    // Delegated Event Listener for Gallery Actions (Send, Delete, Favorite)
    // Info button click is handled by info_modal.js
    document.addEventListener('click', function(e) {

      // Send button
      if (e.target && e.target.classList.contains('send-button')) {
        e.stopPropagation();
        const button = e.target;
        const imageFilename = button.getAttribute('data-image');
        const selectedDevice = document.querySelector('input[name="device"]:checked');

        if (!selectedDevice) {
            if (typeof NotificationSystem !== 'undefined' && typeof NotificationSystem.showStandardToast === 'function') {
                NotificationSystem.showStandardToast({ message: "Please select a device first.", type: 'warning' });
            } else { alert("Please select a device first."); }
            return;
        }
        if (!imageFilename) {
            console.error("Send button clicked without data-image attribute.");
            return;
        }

        const originalText = button.textContent;
        button.disabled = true;
        button.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Sending...';

        const deviceFriendly = selectedDevice.getAttribute('data-friendly');
        const deviceId = selectedDevice.value;
        const operationId = `send-${imageFilename}-${deviceId}`;

        // Show sending notification using NotificationSystem
        if (typeof NotificationSystem !== 'undefined' && typeof NotificationSystem.showLiveToast === 'function') {
            NotificationSystem.showLiveToast(operationId, `Sending image: ${imageFilename} to ${deviceFriendly}...`, "info", true);
        } else { console.warn("NotificationSystem.showLiveToast not available."); }

        const formData = new FormData();
        formData.append("device", deviceId);
        formData.append("filename", imageFilename);
        const finalUrl = `/send_image?t=${Date.now()}`; // Add timestamp to prevent caching

        console.log(`Sending image: ${imageFilename} to device: ${deviceFriendly} (${deviceId})`);
        console.log(`Request URL: ${finalUrl}`);

        const xhr = new XMLHttpRequest();
        xhr.open('POST', finalUrl, true);
        xhr.timeout = 120000;

        xhr.onload = function() {
          button.disabled = false; // Re-enable button regardless of outcome
          console.log(`Send Response: Status ${xhr.status}, Text: ${xhr.responseText}`);
          if (xhr.status === 202) {
            button.innerHTML = '<i class="fas fa-hourglass-half"></i> Queued';
             // Show temporary queued toast (optional, as backend should send persistent one)
             if (typeof NotificationSystem !== 'undefined' && typeof NotificationSystem.showStandardToast === 'function') {
                 NotificationSystem.showStandardToast({ message: `Image send task queued for ${deviceFriendly}.`, type: 'info', delay: 3000 });
             }
             // Note: We don't complete the live toast here, assuming backend/Celery will send a final status notification.
          } else {
            button.innerHTML = '<i class="fas fa-times"></i> Failed';
            let errorMsg = `Error sending image ${imageFilename} to ${deviceFriendly}`;
             try { errorMsg = JSON.parse(xhr.responseText).message || errorMsg; } catch (parseError) { /* Ignore */ }
             // Complete the live toast with error using NotificationSystem
            if (typeof NotificationSystem !== 'undefined' && typeof NotificationSystem.completeLiveToast === 'function') {
                NotificationSystem.completeLiveToast(operationId, errorMsg, "error", 10000);
            } else { console.error("NotificationSystem.completeLiveToast not available."); }
          }
          setTimeout(() => { button.innerHTML = originalText; }, 2000); // Restore button text
        };
        xhr.onerror = function() {
          button.disabled = false; button.innerHTML = '<i class="fas fa-times"></i> Failed';
          const errorMsg = "Network error while sending image.";
           // Complete the live toast with error using NotificationSystem
           if (typeof NotificationSystem !== 'undefined' && typeof NotificationSystem.completeLiveToast === 'function') {
               NotificationSystem.completeLiveToast(operationId, errorMsg, "error", 10000);
           } else { console.error("NotificationSystem.completeLiveToast not available."); }
          setTimeout(() => { button.innerHTML = originalText; }, 2000);
        };
        xhr.ontimeout = function() {
          button.disabled = false; button.innerHTML = '<i class="fas fa-times"></i> Timeout';
          const errorMsg = "Request timed out. The eInk display might be busy or offline.";
           // Complete the live toast with warning using NotificationSystem
           if (typeof NotificationSystem !== 'undefined' && typeof NotificationSystem.completeLiveToast === 'function') {
               NotificationSystem.completeLiveToast(operationId, errorMsg, "warning", 10000);
           } else { console.error("NotificationSystem.completeLiveToast not available."); }
          setTimeout(() => { button.innerHTML = originalText; }, 2000);
        };
        xhr.send(formData);
      }

      // Delete button
      else if (e.target && e.target.closest('.delete-icon')) {
        e.stopPropagation();
        const deleteIcon = e.target.closest('.delete-icon');
        const imageFilename = deleteIcon.getAttribute('data-image');
        if (!imageFilename) {
            console.error("Delete icon clicked without data-image attribute.");
            return;
        }

        if (!confirm(`Are you sure you want to delete ${imageFilename}? This cannot be undone.`)) {
            return;
        }

        const deleteUrl = `/delete_image/${encodeURIComponent(imageFilename)}`;
        fetch(deleteUrl, { method: 'POST' })
          .then(response => response.json())
          .then(data => {
            if (data.status === "success") {
              currentPage = 1; // Reset to first page after delete
              loadImages(1); // Refresh gallery
               if (typeof NotificationSystem !== 'undefined' && typeof NotificationSystem.showStandardToast === 'function') {
                   NotificationSystem.showStandardToast({ message: data.message || "Image deleted.", type: 'success' });
               }
            } else {
              console.error("Error deleting image:", data.message);
               if (typeof NotificationSystem !== 'undefined' && typeof NotificationSystem.showStandardToast === 'function') {
                   NotificationSystem.showStandardToast({ message: `Error deleting image: ${data.message || 'Unknown error'}`, type: 'error' });
               }
            }
          })
          .catch(error => {
            console.error("Fetch error deleting image:", error);
             if (typeof NotificationSystem !== 'undefined' && typeof NotificationSystem.showStandardToast === 'function') {
                 NotificationSystem.showStandardToast({ message: `Error deleting image: ${error}`, type: 'error' });
             }
          });
      }

      // Favorite toggle
      else if (e.target && e.target.closest('.favorite-icon')) {
        e.stopPropagation();
        const favIcon = e.target.closest('.favorite-icon');
        const imageFilename = favIcon.getAttribute('data-image');
         if (!imageFilename) {
            console.error("Favorite icon clicked without data-image attribute.");
            return;
        }

        favIcon.classList.toggle('favorited');
        const isFavorited = favIcon.classList.contains('favorited');

        fetch("/api/update_image_metadata", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ filename: imageFilename, favorite: isFavorited }) // Only send favorite status
        })
          .then(resp => resp.json())
          .then(data => {
            if (data.status !== "success") {
              console.error("Error updating favorite:", data.message);
               favIcon.classList.toggle('favorited'); // Revert UI on error
               if (typeof NotificationSystem !== 'undefined' && typeof NotificationSystem.showStandardToast === 'function') {
                   NotificationSystem.showStandardToast({ message: `Error updating favorite: ${data.message || 'Unknown error'}`, type: 'error' });
               }
            } else {
                 if (typeof NotificationSystem !== 'undefined' && typeof NotificationSystem.showStandardToast === 'function') {
                   NotificationSystem.showStandardToast({ message: `Favorite status updated for ${imageFilename}`, type: 'success', delay: 1500 });
               }
            }
          })
          .catch(err => {
            console.error("Fetch error updating favorite:", err);
             favIcon.classList.toggle('favorited'); // Revert UI on error
             if (typeof NotificationSystem !== 'undefined' && typeof NotificationSystem.showStandardToast === 'function') {
                 NotificationSystem.showStandardToast({ message: `Error updating favorite: ${err}`, type: 'error' });
             }
          });
      }
    });

  } catch (error) {
      console.error("Error during gallery_page.js DOMContentLoaded initialization:", error);
      // Optionally display a user-friendly error message on the page
      const galleryDiv = document.getElementById('gallery');
      if (galleryDiv) {
          galleryDiv.innerHTML = '<p class="text-danger">Error initializing gallery. Please check the console.</p>';
      }
  }
}); // End of DOMContentLoaded listener
