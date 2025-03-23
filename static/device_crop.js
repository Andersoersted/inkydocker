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
  const availableDeviceCrops = document.getElementById('availableDeviceCrops');
  const deviceCropsList = document.getElementById('deviceCropsList');
  
  if (!availableDeviceCrops || !deviceCropsList) return;
  
  if (devices && devices.length > 0) {
    availableDeviceCrops.style.display = 'block';
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