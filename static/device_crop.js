/**
 * Device-specific cropping functionality
 * This file contains the functions needed to support multi-device crop definitions
 * where each e-ink display can have its own crop of the same image
 */

// Variable to store available device crops for the current image
let availableDeviceCrops = [];

/**
 * Initialize device-specific cropping
 * This is called when the page loads
 */
document.addEventListener('DOMContentLoaded', function() {
  // Set up event listener for the Refresh button in the crop modal
  const refreshBtn = document.getElementById('refreshBtn');
  if (refreshBtn) {
    refreshBtn.addEventListener('click', refreshAndCrop);
  }
});

/**
 * Function to refresh a screenshot and then apply the crop
 * This is called when the user clicks the "Refresh" button in the crop modal
 */
function refreshAndCrop() {
  const refreshBtn = document.getElementById('refreshBtn');
  if (!refreshBtn) return;
  
  const originalText = refreshBtn.textContent;
  refreshBtn.disabled = true;
  refreshBtn.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Refreshing...';
  
  // Get the selected device
  const selectedDevice = document.querySelector('input[name="device"]:checked');
  if (!selectedDevice) {
    alert("Please select a device first");
    refreshBtn.disabled = false;
    refreshBtn.textContent = originalText;
    return;
  }
  
  // Determine if this is a screenshot or regular image
  const isScreenshot = currentInfoFilename.startsWith('screenshot_');
  
  if (!isScreenshot) {
    alert("Refresh functionality is only available for screenshots");
    refreshBtn.disabled = false;
    refreshBtn.textContent = originalText;
    return;
  }
  
  // Get the screenshot details to refresh it
  fetch(`/api/get_screenshot_info/${encodeURIComponent(currentInfoFilename)}`)
    .then(response => response.json())
    .then(data => {
      if (data.status === "success" && data.screenshot) {
        // Now refresh the screenshot using the URL
        return fetch('/api/browserless/screenshot', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            url: data.screenshot.url,
            name: data.screenshot.name
          })
        });
      } else {
        throw new Error("Could not find screenshot details");
      }
    })
    .then(response => response.json())
    .then(result => {
      if (result.status === "success") {
        // Screenshot was refreshed, now reopen the crop modal with the new screenshot
        refreshBtn.innerHTML = '<i class="fas fa-check"></i> Success!';
        setTimeout(() => {
          closeCropModal();
          // Reopen the crop modal with the refreshed image
          currentInfoFilename = result.filename;
          openCropModal();
        }, 1000);
      } else {
        throw new Error(result.message || "Failed to refresh screenshot");
      }
    })
    .catch(error => {
      console.error("Error refreshing screenshot:", error);
      refreshBtn.disabled = false;
      refreshBtn.textContent = originalText;
      alert("Error refreshing screenshot: " + error.message);
    });
}

/**
 * Load crop information for a specific device
 * @param {string} filename - The image filename
 * @param {string} deviceAddress - The device address to load crop info for
 * @returns {Promise} - A promise that resolves to the crop info
 */
function loadDeviceCropInfo(filename, deviceAddress) {
  const url = `/api/get_crop_info/${encodeURIComponent(filename)}${deviceAddress ? `?device=${encodeURIComponent(deviceAddress)}` : ''}`;
  return fetch(url)
    .then(response => response.json())
    .then(data => {
      if (data.available_devices && data.available_devices.length > 0) {
        availableDeviceCrops = data.available_devices;
        displayAvailableDeviceCrops(data.available_devices, data.crop_info);
      }
      return data;
    });
}

/**
 * Display available device crops in the crop modal
 * @param {Array} devices - Array of device objects
 * @param {Object} currentCropInfo - The current crop info
 */
function displayAvailableDeviceCrops(devices, currentCropInfo) {
  // Show available device crops section
  const availableDeviceCropsDiv = document.getElementById('availableDeviceCrops');
  const deviceCropsList = document.getElementById('deviceCropsList');
  
  if (!availableDeviceCropsDiv || !deviceCropsList) return;
  
  if (devices && devices.length > 0) {
    availableDeviceCropsDiv.style.display = 'block';
    deviceCropsList.innerHTML = '';
    
    // Add buttons for each device crop
    devices.forEach(device => {
      const button = document.createElement('button');
      button.className = 'btn btn-sm btn-outline-primary';
      button.textContent = device.address;
      button.type = 'button';
      button.onclick = function() {
        // Load this device's crop by reloading the crop modal
        const selectedDevice = document.querySelector(`input[name="device"][value="${device.address}"]`);
        if (selectedDevice) {
          selectedDevice.checked = true;
        }
        
        // Reload crop data for this device
        closeCropModal();
        setTimeout(() => {
          openCropModal();
        }, 100);
      };
      
      // Mark current device
      if (currentCropInfo && currentCropInfo.device_address === device.address) {
        button.classList.remove('btn-outline-primary');
        button.classList.add('btn-primary');
        button.innerHTML = `<i class="fas fa-check-circle mr-1"></i> ${device.address}`;
      }
      
      deviceCropsList.appendChild(button);
    });
  } else {
    availableDeviceCrops.style.display = 'none';
  }
  
  // Show the refresh button if this is a screenshot
  const refreshBtn = document.getElementById('refreshBtn');
  if (refreshBtn && currentInfoFilename && currentInfoFilename.startsWith('screenshot_')) {
    refreshBtn.style.display = 'inline-block';
  }
}

/**
 * Save crop data for a specific device
 * @param {Object} cropData - The crop data to save
 * @param {string} filename - The image filename
 * @returns {Promise} - A promise that resolves when the crop is saved
 */
function saveDeviceCropData(cropData, filename) {
  // Get selected device for resolution info
  const selectedDevice = document.querySelector('input[name="device"]:checked');
  const deviceAddress = selectedDevice ? selectedDevice.value : 'default_device';
  
  // Combine crop data with device info
  const data = {
    ...cropData,
    device: deviceAddress,
    device_address: deviceAddress
  };
  
  // Send crop data to server
  return fetch(`/save_crop_info/${encodeURIComponent(filename)}`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify(data)
  });
}

// Modify the original saveCropData function to use our device-specific function
const originalSaveCropData = window.saveCropData;
window.saveCropData = function() {
  if (!cropper) {
    console.error("Cropper not initialized");
    alert("Error: Cropper not initialized. Please try again.");
    return;
  }
  
  try {
    // Make sure the cropper is valid and ready
    if (!cropper.ready) {
      console.error("Cropper not ready");
      alert("Error: Cropper not ready. Please try again.");
      return;
    }
    
    // Get the image dimensions from the cropper
    const cropperImageData = cropper.getImageData();
    const displayWidth = cropperImageData.naturalWidth;
    const displayHeight = cropperImageData.naturalHeight;
    
    // It's critical to get the image dimensions correctly here
    const displayedImage = cropper.getImageData();
    const displayedWidth = displayedImage.width;
    const displayedHeight = displayedImage.height;
    
    // Get crop data from cropper (relative to the currently displayed image)
    let cropData;
    try {
      cropData = cropper.getData(true); // rounded to integers
    } catch (error) {
      console.error("Error getting crop data:", error);
      alert("Error getting crop data. Please try again.");
      return;
    }
    
    // Calculate scaling factors between displayed image and original full-res image
    let scaleX = 1;
    let scaleY = 1;
    
    if (originalImageWidth > 0 && originalImageHeight > 0 &&
        displayedWidth > 0 && displayedHeight > 0) {
      // Scale from display size back to original full-res size
      scaleX = originalImageWidth / displayedWidth;
      scaleY = originalImageHeight / displayedHeight;
      
      console.log(`Save scaling factors (displayed → original): ${scaleX}x, ${scaleY}y`);
      console.log(`Displayed image: ${displayedWidth}x${displayedHeight}`);
      console.log(`Original image: ${originalImageWidth}x${originalImageHeight}`);
      console.log(`Crop on display: x=${cropData.x}, y=${cropData.y}, w=${cropData.width}, h=${cropData.height}`);
      
      // Scale the crop coordinates to match the original image
      cropData.x = Math.round(cropData.x * scaleX);
      cropData.y = Math.round(cropData.y * scaleY);
      cropData.width = Math.round(cropData.width * scaleX);
      cropData.height = Math.round(cropData.height * scaleY);
      
      console.log(`Scaled crop for original: x=${cropData.x}, y=${cropData.y}, w=${cropData.width}, h=${cropData.height}`);
    } else {
      console.warn("Could not scale crop coordinates - missing dimension information");
    }
    
    // Show saving indicator
    const saveBtn = document.querySelector('#cropModal .btn-primary');
    const originalText = saveBtn.textContent;
    saveBtn.disabled = true;
    saveBtn.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Saving...';
    
    // Use our device-specific save function
    saveDeviceCropData(cropData, currentInfoFilename)
      .then(response => response.json())
      .then(result => {
        if (result.status === "success") {
          // Show success message
          const infoStatus = document.getElementById('infoStatus');
          if (infoStatus) {
            infoStatus.textContent = 'Crop saved successfully!';
          }
          
          // Reset button state before closing modal
          saveBtn.disabled = false;
          saveBtn.textContent = originalText;
          
          // Show the Refresh button if it's a screenshot
          const refreshBtn = document.getElementById('refreshBtn');
          if (refreshBtn && currentInfoFilename && currentInfoFilename.startsWith('screenshot_')) {
            refreshBtn.style.display = 'inline-block';
          }
          
          // Close the crop modal
          closeCropModal();
        } else {
          console.error("Error saving crop data:", result.message);
          saveBtn.disabled = false;
          saveBtn.textContent = originalText;
          alert("Error saving crop data: " + (result.message || "Unknown error"));
        }
      })
      .catch(error => {
        console.error("Error saving crop data:", error);
        saveBtn.disabled = false;
        saveBtn.textContent = originalText;
        alert("Error saving crop data: " + error.message);
      });
  } catch (error) {
    console.error("Error in saveCropData:", error);
    alert("An error occurred while processing crop data. Please try again.");
    
    // Reset button state
    const saveBtn = document.querySelector('#cropModal .btn-primary');
    if (saveBtn) {
      saveBtn.disabled = false;
      saveBtn.textContent = 'Save Crop';
    }
  }
};

/**
 * Initialize the cropper with device-specific settings
 * @param {string} deviceOrientation - The device orientation (landscape or portrait)
 * @param {string} deviceResolution - The device resolution (e.g., 800x600)
 */
function initializeCropper(deviceOrientation, deviceResolution) {
  console.log(`Initializing cropper with device: orientation=${deviceOrientation}, resolution=${deviceResolution}`);
  // Destroy existing cropper if it exists
  if (cropper) {
    cropper.destroy();
  }
  
  console.log("Initializing cropper for modal...");
  console.log(`Using device orientation: ${deviceOrientation}, resolution: ${deviceResolution}`);
  
  // Get the crop image element
  const cropImage = document.getElementById('cropImage');
  if (!cropImage) {
    console.error("Crop image element not found");
    return;
  }
  
  // First check if there are existing crop settings for this image and device
  const deviceAddress = document.querySelector('input[name="device"]:checked')?.value;
  console.log(`Checking for existing crop settings for ${currentInfoFilename} with device ${deviceAddress}`);
  
  const url = `/api/get_crop_info/${encodeURIComponent(currentInfoFilename)}${deviceAddress ? `?device=${encodeURIComponent(deviceAddress)}` : ''}`;
  fetch(url)
    .then(response => response.json())
    .then(data => {
      console.log('Crop info response:', data);
      
      // Store original dimensions for scaling
      if (data.original_dimensions) {
        originalImageWidth = data.original_dimensions.width;
        originalImageHeight = data.original_dimensions.height;
        console.log(`Original image dimensions: ${originalImageWidth}x${originalImageHeight}`);
      }
      
      // Check if data exists and has the expected structure
      const hasCropData = data.status === "success" && data.crop_info !== null;
      console.log('Has existing crop data:', hasCropData);
      
      // Mark the modal with a data attribute to indicate we have crop data
      document.getElementById('cropModal').setAttribute('data-has-crop-data', hasCropData.toString());
      
      // Display available device crops if any
      if (data.available_devices && data.available_devices.length > 0) {
        displayAvailableDeviceCrops(data.available_devices, data.crop_info);
      }
      
      // Calculate the aspect ratio based on device resolution and orientation
      let aspectRatio = NaN; // Default to free aspect ratio
      
      if (deviceResolution) {
        const parts = deviceResolution.split('x');
        if (parts.length === 2) {
          const width = parseInt(parts[0], 10);
          const height = parseInt(parts[1], 10);
          
          if (!isNaN(width) && !isNaN(height) && width > 0 && height > 0) {
            // Set aspect ratio based on device orientation
            if (deviceOrientation === 'portrait') {
              // For portrait displays, the physical display is rotated 90°
              // So we need to swap the dimensions and use height/width (taller than wide)
              console.log(`Portrait display: swapping dimensions ${width}x${height} -> ${height}x${width}`);
              aspectRatio = height / width;
            } else {
              // For landscape displays, use width/height (wider than tall)
              aspectRatio = width / height;
            }
            // Save aspect ratio to global variable so it can be used later
            window.currentAspectRatio = aspectRatio;
            console.log(`Setting aspect ratio to ${aspectRatio} for ${deviceOrientation} display (${deviceResolution})`);
          }
        }
      }
      
      // Initialize Cropper.js with the calculated aspect ratio
      cropper = new Cropper(cropImage, {
        viewMode: 1,
        dragMode: 'crop',
        aspectRatio: aspectRatio, // Use the calculated aspect ratio based on device
        autoCropArea: 1, // Use maximum possible area (1 = 100%)
        restore: false,
        guides: true,
        center: true,
        highlight: false,
        cropBoxMovable: true,
        cropBoxResizable: true,
        toggleDragModeOnDblclick: false,
        responsive: true,     // Make it responsive to window resizing
        wheelZoomRatio: 0.1,  // Slower zoom for more precision
        ready: function() {
          // Log when cropper is ready
          console.log('Cropper instance initialized');
          
          // When the cropper is ready, check if we have existing crop data
          if (hasCropData) {
            try {
              console.log('Applying existing crop data:', data.crop_info);
              
              // Get key measurements from the cropper
              const imageData = this.cropper.getImageData(); // Image as displayed
              const canvasData = this.cropper.getCanvasData(); // Canvas containing the image
              const containerData = this.cropper.getContainerData(); // Outer container
              
              console.log("Image data:", imageData);
              console.log("Canvas data:", canvasData);
              console.log("Container data:", containerData);
              
              // IMPORTANT: Use the ACTUAL displayed size, not the natural size
              const displayedWidth = imageData.width; // Actual width of image as displayed
              const displayedHeight = imageData.height; // Actual height of image as displayed
              
              // Calculate scaling factors from original full-res to display size
              let scaleX = 1;
              let scaleY = 1;
              
              if (originalImageWidth > 0 && originalImageHeight > 0 &&
                  displayedWidth > 0 && displayedHeight > 0) {
                // Scale from original full-res to currently displayed size
                scaleX = displayedWidth / originalImageWidth;
                scaleY = displayedHeight / originalImageHeight;
                
                console.log(`Scaling factors (original → displayed): ${scaleX}x, ${scaleY}y`);
                console.log(`Original crop from database: x=${data.crop_info.x}, y=${data.crop_info.y}, w=${data.crop_info.width}, h=${data.crop_info.height}`);
              }
              
              // Scale the coordinates from original image size to displayed size
              const scaledX = Math.round(data.crop_info.x * scaleX);
              const scaledY = Math.round(data.crop_info.y * scaleY);
              const scaledWidth = Math.min(Math.round(data.crop_info.width * scaleX), displayedWidth);
              const scaledHeight = Math.min(Math.round(data.crop_info.height * scaleY), displayedHeight);
              
              console.log(`Scaled for display view: x=${scaledX}, y=${scaledY}, w=${scaledWidth}, h=${scaledHeight}`);
              
              // The canvas may be offset within the container
              // We need to account for these offsets when positioning the crop box
              const canvasLeft = canvasData.left;
              const canvasTop = canvasData.top;
              
              // Calculate the final position of the crop box relative to canvas
              const cropBoxData = {
                left: canvasLeft + scaledX,  // Position relative to canvas left
                top: canvasTop + scaledY,    // Position relative to canvas top
                width: scaledWidth,
                height: scaledHeight
              };
              
              console.log("Setting crop box data:", cropBoxData);
              
              // Set up a timeout to ensure the crop box is applied after any automatic positioning
              setTimeout(() => {
                // Make sure we're in crop mode
                this.cropper.setDragMode('crop');
                
                // Apply the crop box
                this.cropper.setCropBoxData(cropBoxData);
                
                // Also try setting the data directly
                this.cropper.setData({
                  x: scaledX,
                  y: scaledY,
                  width: scaledWidth,
                  height: scaledHeight,
                  rotate: 0,
                  scaleX: 1,
                  scaleY: 1
                });
                
                console.log('Successfully applied existing crop data');
              }, 100);
            } catch (error) {
              console.error('Error applying existing crop data:', error);
              // Fall back to default behavior
              globalSetDefaultCropBox(this.cropper);
            }
          } else {
            // No existing crop data, use default behavior
            console.log('No existing crop data found, using default settings');
            globalSetDefaultCropBox(this.cropper);
          }
        }
      });
    })
    .catch(error => {
      console.error('Error fetching crop info:', error);
      
      // Initialize Cropper.js with default settings if fetch fails
      cropper = new Cropper(cropImage, {
        viewMode: 1,
        dragMode: 'crop',
        aspectRatio: getAspectRatioFromDevice(deviceOrientation, deviceResolution),
        autoCropArea: 1,
        restore: false,
        guides: true,
        center: true,
        highlight: false,
        cropBoxMovable: true,
        cropBoxResizable: true,
        toggleDragModeOnDblclick: false,
        responsive: true,
        ready: function() {
          // Set default crop box
          globalSetDefaultCropBox(this.cropper);
        }
      });
    });
}

/**
 * Set default crop box for new crops or when crop data is not found
 * This is called when there's no existing crop data for the image
 * @param {Object} cropperInstance - The cropper instance
 */
function globalSetDefaultCropBox(cropperInstance) {
  if (!cropperInstance) return;

  try {
    // Set crop mode first
    cropperInstance.setDragMode('crop');
    
    // Check if there's already a valid crop box
    const currentBox = cropperInstance.getCropBoxData();
    if (currentBox && currentBox.width > 0 && currentBox.height > 0) {
      console.log("Crop box already exists:", currentBox);
      return; // Keep existing box
    }
    
    // Get current aspect ratio
    let aspectRatio = window.currentAspectRatio || 1.6666667;
    
    // If we got here, we need to create a crop box
    // Use a simple approach - create a centered crop box at maximum size
    const imageData = cropperInstance.getImageData();
    const canvasData = cropperInstance.getCanvasData();
    
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
    
    // Position the crop box in the center
    cropperInstance.setCropBoxData({
      left: canvasData.left + (canvasData.width - cropWidth) / 2,
      top: canvasData.top + (canvasData.height - cropHeight) / 2,
      width: cropWidth,
      height: cropHeight
    });
    
    console.log(`Set default crop box: ${cropWidth}x${cropHeight}`);
    
    // Also try using setData for more reliable results
    cropperInstance.setData({
      x: (imageData.width - cropWidth) / 2,
      y: (imageData.height - cropHeight) / 2,
      width: cropWidth,
      height: cropHeight,
      rotate: 0
    });
  } catch (err) {
    console.error("Error in globalSetDefaultCropBox:", err);
  }
}

/**
 * Helper function to calculate aspect ratio from device specs
 */
function getAspectRatioFromDevice(deviceOrientation, deviceResolution) {
  let aspectRatio = NaN; // Default to free aspect ratio
  
  if (deviceResolution) {
    const parts = deviceResolution.split('x');
    if (parts.length === 2) {
      const width = parseInt(parts[0], 10);
      const height = parseInt(parts[1], 10);
      
      if (!isNaN(width) && !isNaN(height) && width > 0 && height > 0) {
        // Set aspect ratio based on device orientation
        if (deviceOrientation === 'portrait') {
          // For portrait displays, use height/width
          console.log(`Portrait display: ${height}/${width} = ${height/width}`);
          aspectRatio = height / width;
        } else {
          // For landscape displays, use width/height
          console.log(`Landscape display: ${width}/${height} = ${width/height}`);
          aspectRatio = width / height;
        }
      }
    }
  }
  
  // Default to 16:9 if calculation fails
  return aspectRatio || 16/9;
}

// Modify the original openCropModal function to use our device-specific function
const originalOpenCropModal = window.openCropModal;
window.openCropModal = function() {
  // Run the original function first
  if (typeof originalOpenCropModal === 'function') {
    originalOpenCropModal();
  }
  
  // Get selected device information
  const selectedDevice = document.querySelector('input[name="device"]:checked');
  const deviceAddress = selectedDevice ? selectedDevice.value : null;
  
  // Modify the crop modal to show device information
  document.getElementById('cropModal').setAttribute('data-device', deviceAddress || 'default');
  
  // Load crop info for this device
  if (currentInfoFilename) {
    loadDeviceCropInfo(currentInfoFilename, deviceAddress);
  }
};