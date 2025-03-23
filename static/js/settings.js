// Settings page specific JavaScript
document.addEventListener('DOMContentLoaded', function() {
  // Apply colors to the device color indicators
  function applyDeviceColors() {
    const colorIndicators = document.querySelectorAll('.device-color-indicator');
    colorIndicators.forEach(indicator => {
      const color = indicator.getAttribute('data-color');
      if (color) {
        indicator.style.backgroundColor = color;
      }
    });
  }

  // Initial application of colors
  applyDeviceColors();
  
  // Also apply colors after table updates
  setInterval(applyDeviceColors, 2000);
  
  // Fix for Advanced Actions modal
  // Make sure the elements exist before trying to update them
  window.openAdvancedModal = function(index, friendlyName) {
    const titleElement = document.getElementById('advancedDeviceTitle');
    const modal = document.getElementById('advancedActionsModal');
    
    if (titleElement) {
      titleElement.textContent = "Advanced Actions for " + friendlyName;
    }
    
    if (modal) {
      modal.setAttribute('data-device-index', index);
      
      // Get the device's IP address for the command examples (if elements exist)
      const deviceRow = document.querySelector('tr[data-index="' + index + '"]');
      if (deviceRow) {
        const deviceAddress = deviceRow.getAttribute('data-address');
        if (deviceAddress) {
          // Remove http:// or https:// prefix if present
          let cleanAddress = deviceAddress;
          if (cleanAddress.startsWith("http://")) {
            cleanAddress = cleanAddress.substring(7);
          } else if (cleanAddress.startsWith("https://")) {
            cleanAddress = cleanAddress.substring(8);
          }
          
          // Update elements if they exist
          const sysUpdateIP = document.getElementById('systemUpdateIP');
          const appUpdateIP = document.getElementById('appUpdateIP');
          if (sysUpdateIP) sysUpdateIP.textContent = cleanAddress;
          if (appUpdateIP) appUpdateIP.textContent = cleanAddress;
        }
      }
      
      // Show the modal
      const modalInstance = new bootstrap.Modal(modal);
      modalInstance.show();
    }
  };
});