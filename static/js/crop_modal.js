// Variable to store the cropper instance
let cropper = null;

// Store original image dimensions for scaling
let originalImageWidth = 0;
let originalImageHeight = 0;

function openCropModal() {
  // Set up crop image
  const cropImage = document.getElementById('cropImage');
  // Ensure currentInfoFilename is accessible (might need to be passed or global)
  if (!currentInfoFilename) {
      console.error("Cannot open crop modal: currentInfoFilename is not set.");
      // Optionally show a user notification
      if (typeof NotificationSystem !== 'undefined' && typeof NotificationSystem.showStandardToast === 'function') {
          NotificationSystem.showStandardToast({ message: "Please open the info modal for an image first.", type: 'warning' });
      }
      return;
  }
  cropImage.dataset.src = `/images/${encodeURIComponent(currentInfoFilename)}`;
  cropImage.src = `/images/${encodeURIComponent(currentInfoFilename)}`;

  // Reset original dimensions
  originalImageWidth = 0;
  originalImageHeight = 0;

  // Get the crop modal element
  const cropModalEl = document.getElementById('cropModal');

  // Initialize the crop data flag to false - will be updated when data is fetched
  cropModalEl.setAttribute('data-has-crop-data', 'false');

  // Clear any previous saved crop position
  delete cropModalEl.dataset.cropPosition;

  // Reset the save button state
  const saveBtn = document.querySelector('#cropModal .btn-primary');
  if (saveBtn) {
    saveBtn.disabled = false;
    saveBtn.textContent = 'Save Crop';
  }

  // Get selected device information
  const selectedDevice = document.querySelector('input[name="device"]:checked');
  let deviceOrientation = 'landscape'; // Default to landscape
  let deviceResolution = '800x600'; // Default resolution

  if (selectedDevice) {
    // Get device resolution from data attribute
    deviceResolution = selectedDevice.getAttribute('data-resolution') || '800x600';

    // Get device orientation from data attribute
    deviceOrientation = selectedDevice.getAttribute('data-orientation') || 'landscape';

    console.log(`Selected device: ${selectedDevice.getAttribute('data-friendly')}, orientation: ${deviceOrientation}, resolution: ${deviceResolution}`);
  }

  // Add device info to the modal title
  const modalTitle = document.getElementById('cropModalLabel');
  modalTitle.textContent = `Crop Image for ${deviceOrientation} display (${deviceResolution})`;

  // First prefetch crop info to have it ready when modal opens
  fetch(`/api/get_crop_info/${encodeURIComponent(currentInfoFilename)}`)
    .then(response => response.json())
    .then(data => {
      if (data.status === "success" && data.crop_info) {
        // Mark modal as having crop data
        cropModalEl.setAttribute('data-has-crop-data', 'true');

        // Store original dimensions for scaling
        if (data.original_dimensions) {
          originalImageWidth = data.original_dimensions.width;
          originalImageHeight = data.original_dimensions.height;
        }

        // Store the crop position in a dataset attribute for reference
        cropModalEl.dataset.cropPosition = JSON.stringify({
          x: data.crop_info.x || 0,
          y: data.crop_info.y || 0,
          width: data.crop_info.width || 0,
          height: data.crop_info.height || 0
        });
      }

      // Now show the modal after getting the data
      const cropModal = new bootstrap.Modal(cropModalEl);
      cropModal.show();

      // Initialize cropper after the modal is shown - pass needed parameters
      // Use one-time event listener to avoid stacking multiple listeners
      const initCropperOnce = function() {
        // Ensure initializeCropper is defined (might need to move it here or ensure it's global)
        if (typeof initializeCropper === 'function') {
            initializeCropper(deviceOrientation, deviceResolution);
        } else {
            console.error("initializeCropper function not found.");
        }
        cropModalEl.removeEventListener('shown.bs.modal', initCropperOnce);
      };
      cropModalEl.addEventListener('shown.bs.modal', initCropperOnce);

      // Update the orientation note
      updateOrientationNote(deviceOrientation, deviceResolution);
    })
    .catch(error => {
      console.error("Error pre-fetching crop info:", error);

      // Show modal anyway even if prefetch fails
      const cropModal = new bootstrap.Modal(cropModalEl);
      cropModal.show();

      // Initialize cropper after the modal is shown
      const initCropperOnce = function() {
         if (typeof initializeCropper === 'function') {
            initializeCropper(deviceOrientation, deviceResolution);
        } else {
            console.error("initializeCropper function not found.");
        }
        cropModalEl.removeEventListener('shown.bs.modal', initCropperOnce);
      };
      cropModalEl.addEventListener('shown.bs.modal', initCropperOnce);

      // Update the orientation note
      updateOrientationNote(deviceOrientation, deviceResolution);
    });
}

// Add a note about device orientation to the modal function
function updateOrientationNote(deviceOrientation, deviceResolution) {
  const modalBody = document.querySelector('#cropModal .modal-body');
  let orientationNote = document.getElementById('orientationNote');

  if (!orientationNote) {
    orientationNote = document.createElement('div');
    orientationNote.id = 'orientationNote';
    orientationNote.className = 'alert alert-info mt-3';
    // Find a suitable place to append, e.g., after the crop container wrapper
    const wrapper = modalBody.querySelector('.crop-container-wrapper');
    if (wrapper && wrapper.parentNode === modalBody) {
        modalBody.insertBefore(orientationNote, wrapper.nextSibling);
    } else {
        modalBody.appendChild(orientationNote); // Fallback
    }
  }

  // Prepare the orientation-specific message
  let orientationMsg = '';
  if (deviceOrientation === 'portrait') {
    orientationMsg = 'Since your display is in portrait mode, the image will be rotated 90° clockwise after cropping to fit correctly.';
  } else {
    orientationMsg = 'Since your display is in landscape mode, the image will be sent exactly as cropped, without rotation.';
  }

  orientationNote.innerHTML = `
    <strong>Note:</strong> You are cropping for a <strong>${deviceOrientation}</strong> display (${deviceResolution}).
    <br>The crop area is locked to the correct aspect ratio for your display and starts at the maximum possible size.
    <br>You can move and resize the crop area to select which part of the image to send.
    <br>${orientationMsg}
  `;
}

function closeCropModal() {
  const cropModalEl = document.getElementById('cropModal');
  const cropModal = bootstrap.Modal.getInstance(cropModalEl);
  if (cropModal) {
    cropModal.hide();
  }

  // Reset the crop data flag
  if (cropModalEl) {
      cropModalEl.setAttribute('data-has-crop-data', 'false');
  }


  // Destroy cropper instance when modal is closed
  if (cropper) {
    cropper.destroy();
    cropper = null;
  }

  // Reset the save button state
  const saveBtn = document.querySelector('#cropModal .btn-primary');
  if (saveBtn) {
    saveBtn.disabled = false;
    saveBtn.textContent = 'Save Crop';
  }
}

// Global variable to store the current aspect ratio
let currentAspectRatio = null;

// Simple function to set default crop box
function globalSetDefaultCropBox(cropperInstance) {
  if (!cropperInstance) return;

  try {
    // Set crop mode first
    cropperInstance.setDragMode('crop');

    // Check if we have saved crop position in the dataset
    const cropModal = document.getElementById('cropModal');
    const savedCropPosition = cropModal.dataset.cropPosition;

    if (savedCropPosition && cropModal.getAttribute('data-has-crop-data') === 'true') {
      try {
        // Use saved crop position if available
        console.log("Using saved crop position from dataset");
        const cropPosition = JSON.parse(savedCropPosition);

        // Get image data to determine if this is a full-width crop
        const imageData = cropperInstance.getImageData();
        const canvasData = cropperInstance.getCanvasData();
        const isFullWidth = cropPosition.width >= imageData.naturalWidth * 0.95; // Consider it full width if >95% of image width

        if (isFullWidth) {
          console.log("Detected a full-width crop, using maximum possible width");

          // Use the aspect ratio to calculate the maximum box
          const aspectRatio = currentAspectRatio || 1.6666667;

          // For a full-width crop, we want to make sure it takes the full width
          const maxWidth = canvasData.width;
          const maxHeight = maxWidth / aspectRatio;

          // Try multiple approaches to ensure the cropper sets the full width

          // Approach 1: Use setCropBoxData directly
          cropperInstance.setCropBoxData({
            left: canvasData.left,
            top: canvasData.top + (canvasData.height - maxHeight) / 2,
            width: maxWidth,
            height: maxHeight
          });

          // Approach 2: Also try setData
          cropperInstance.setData({
            x: 0,
            y: (imageData.height - imageData.width / aspectRatio) / 2,
            width: imageData.width,
            height: imageData.width / aspectRatio,
            rotate: 0,
            scaleX: 1,
            scaleY: 1
          });

          // Approach 3: Force a center crop with the proper aspect ratio
          setTimeout(() => {
            // One more attempt to ensure it takes the full width
            cropperInstance.setCropBoxData({
              left: canvasData.left,
              top: canvasData.top + (canvasData.height - maxHeight) / 2,
              width: maxWidth,
              height: maxHeight
            });
          }, 100);
        } else {
          // Normal crop - just use the saved position
          // First try with setData which is more reliable
          cropperInstance.setData({
            x: cropPosition.x,
            y: cropPosition.y,
            width: cropPosition.width,
            height: cropPosition.height,
            rotate: 0,
            scaleX: 1,
            scaleY: 1
          });

          // Also try setting the crop box directly
          cropperInstance.setCropBoxData({
            left: cropPosition.x + canvasData.left,
            top: cropPosition.y + canvasData.top,
            width: cropPosition.width,
            height: cropPosition.height
          });
        }

        console.log("Applied saved crop position:", cropPosition);
        return;
      } catch (parseErr) {
        console.error("Error applying saved crop position:", parseErr);
      }
    }

    // First check if there's already a valid crop box
    const currentBox = cropperInstance.getCropBoxData();
    if (currentBox && currentBox.width > 0 && currentBox.height > 0) {
      console.log("Crop box already exists:", currentBox);
      return; // Keep existing box
    }

    // Get current aspect ratio
    let aspectRatio = currentAspectRatio || 1.6666667;

    // If we got here, we need to create a crop box
    // Use a simple approach - create a centered crop box at maximum size
    const imageData = cropperInstance.getImageData();

    // Calculate maximum crop size that fits the aspect ratio
    let cropWidth, cropHeight;

    if (imageData.width / imageData.height > aspectRatio) {
      // Image is wider than the target ratio - constrain by height
      cropHeight = imageData.height;
      cropWidth = cropHeight * aspectRatio;
    } else {
      // Image is taller than target ratio - constrain by width
      cropWidth = imageData.width;
      cropHeight = cropWidth / aspectRatio;
    }

    // Let cropper handle the positioning logic
    console.log(`Setting crop box to ${cropWidth}x${cropHeight}`);

    // Use setData instead of setCropBoxData - more reliable
    cropperInstance.setData({
      x: (imageData.width - cropWidth) / 2,
      y: (imageData.height - cropHeight) / 2,
      width: cropWidth,
      height: cropHeight,
      rotate: 0
    });

    // Verify it worked
    console.log("Current crop box:", cropperInstance.getCropBoxData());

  } catch (err) {
    console.error("Error in globalSetDefaultCropBox:", err);
  }
}

// Simple event listener for when the modal is shown
document.getElementById('cropModal').addEventListener('shown.bs.modal', function() {
  // Give time for cropper to initialize
  setTimeout(function() {
    if (!cropper) return;

    // Always ensure crop mode is active
    cropper.setDragMode('crop');

    // Get current box
    const cropBox = cropper.getCropBoxData();

    // Only create a default box if we have no existing crop data
    // Check if crop box is missing or has invalid dimensions
    if ((!cropBox || !cropBox.width || !cropBox.height ||
        cropBox.width <= 0 || cropBox.height <= 0) &&
        // Add this condition to prevent overriding saved crop data
        document.getElementById('cropModal').getAttribute('data-has-crop-data') !== 'true') {

      console.log("Creating default crop box");

      // Create using simpler approach directly with the cropper API
      const imgData = cropper.getImageData();
      const canvasData = cropper.getCanvasData();
      const aspect = currentAspectRatio || 1.6666667;

      // Always default to maximum width for the best user experience
      // This creates a crop box that takes the full width of the canvas
      const maxWidth = canvasData.width;
      const maxHeight = maxWidth / aspect;

      // First try setting the crop box directly
      cropper.setCropBoxData({
        left: canvasData.left,
        top: canvasData.top + (canvasData.height - maxHeight) / 2,
        width: maxWidth,
        height: maxHeight
      });

      // Also try using setData for reliability
      cropper.setData({
        x: 0,
        y: (imgData.height - imgData.width / aspect) / 2,
        width: imgData.width,
        height: imgData.width / aspect
      });

      console.log("Created default full-width crop box");
    }
  }, 300);
});