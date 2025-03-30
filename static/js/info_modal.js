/* Info Modal Logic */
let currentInfoFilename = null;

document.addEventListener('click', function(e) {
  if (e.target && e.target.classList.contains('info-button')) {
    e.stopPropagation();
    const imageFilename = e.target.getAttribute('data-image');
    openInfoModal(imageFilename);
  }
});

/* Function to open info modal */
function openInfoModal(filename) {
  currentInfoFilename = filename;

  // Reset the form
  document.getElementById('tagContainer').innerHTML = '';
  document.getElementById('infoStatus').textContent = '';

  // Show loading state
  document.getElementById('infoFilename').textContent = 'Loading...';
  document.getElementById('infoResolution').textContent = 'Loading...';
  document.getElementById('infoFilesize').textContent = 'Loading...';

  // Show the modal
  const infoModal = new bootstrap.Modal(document.getElementById('infoModal'));
  infoModal.show();

  // Load image data
  fetch(`/api/get_image_metadata?filename=${encodeURIComponent(filename)}`)
    .then(response => response.json())
    .then(data => {
      if (data.status === "success") {
        // Update basic info
        document.getElementById('infoFilename').textContent = filename;
        document.getElementById('infoResolution').textContent = data.resolution || 'N/A';
        document.getElementById('infoFilesize').textContent = data.filesize || 'N/A';
        document.getElementById('infoFavorite').checked = data.favorite || false;

        // Update preview image
        const previewImg = document.getElementById('infoImagePreview');
        previewImg.dataset.src = `/images/${encodeURIComponent(filename)}`;
        previewImg.src = `/images/${encodeURIComponent(filename)}`;

        // Update tags
        const tagContainer = document.getElementById('tagContainer');
        tagContainer.innerHTML = '';

        if (data.tags && data.tags.length > 0) {
          data.tags.forEach(tag => {
            const tagEl = document.createElement('span');
            tagEl.className = 'badge bg-primary me-1 mb-1 p-2';
            tagEl.textContent = tag;

            const removeBtn = document.createElement('button');
            removeBtn.className = 'btn-close btn-close-white ms-1';
            removeBtn.setAttribute('aria-label', 'Remove');
            removeBtn.style.fontSize = '0.5rem';
            removeBtn.onclick = function() {
              removeTag(tag);
            };

            tagEl.appendChild(removeBtn);
            tagContainer.appendChild(tagEl);
          });

          document.getElementById('infoTags').value = data.tags.join(',');
        }
      } else {
        console.error("Error loading image info:", data.message);
      }
    })
    .catch(error => {
      console.error("Error loading image info:", error);
    });
}

/* Tag management functions */
function addTag() {
  const newTagInput = document.getElementById('newTagInput');
  const tag = newTagInput.value.trim();

  if (!tag) return;

  // Get current tags
  const infoTags = document.getElementById('infoTags');
  const currentTags = infoTags.value ? infoTags.value.split(',') : [];

  // Check if tag already exists
  if (currentTags.includes(tag)) {
    newTagInput.value = '';
    return;
  }

  // Add new tag
  currentTags.push(tag);
  infoTags.value = currentTags.join(',');

  // Add tag element
  const tagContainer = document.getElementById('tagContainer');
  const tagEl = document.createElement('span');
  tagEl.className = 'badge bg-primary me-1 mb-1 p-2';
  tagEl.textContent = tag;

  const removeBtn = document.createElement('button');
  removeBtn.className = 'btn-close btn-close-white ms-1';
  removeBtn.setAttribute('aria-label', 'Remove');
  removeBtn.style.fontSize = '0.5rem';
  removeBtn.onclick = function() {
    removeTag(tag);
  };

  tagEl.appendChild(removeBtn);
  tagContainer.appendChild(tagEl);

  // Clear input
  newTagInput.value = '';
}

function removeTag(tag) {
  // Get current tags
  const infoTags = document.getElementById('infoTags');
  let currentTags = infoTags.value ? infoTags.value.split(',') : [];

  // Remove tag
  currentTags = currentTags.filter(t => t !== tag);
  infoTags.value = currentTags.join(',');

  // Update UI
  const tagContainer = document.getElementById('tagContainer');
  const tagElements = tagContainer.querySelectorAll('.badge');

  tagElements.forEach(el => {
    if (el.textContent.includes(tag)) {
      el.remove();
    }
  });
}

function saveInfoEdits() {
  const infoStatus = document.getElementById('infoStatus');
  infoStatus.textContent = 'Saving...';

  const tags = document.getElementById('infoTags').value ?
    document.getElementById('infoTags').value.split(',') : [];
  const favorite = document.getElementById('infoFavorite').checked;

  fetch("/api/update_image_metadata", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      filename: currentInfoFilename,
      tags: tags,
      favorite: favorite
    })
  })
    .then(response => response.json())
    .then(data => {
      if (data.status === "success") {
        infoStatus.textContent = 'Saved successfully!';

        // Refresh gallery to show updated metadata
        // Note: loadImages is defined in gallery_page.js, ensure it's loaded
        if (typeof loadImages === 'function') {
            setTimeout(() => {
              loadImages(currentPage); // currentPage is also from gallery_page.js
            }, 1000);
        } else {
            console.warn("loadImages function not found. Cannot refresh gallery from info_modal.js");
        }
      } else {
        infoStatus.textContent = 'Error saving: ' + data.message;
      }
    })
    .catch(error => {
      infoStatus.textContent = 'Error saving metadata';
      console.error("Error saving metadata:", error);
    });
}

function runOpenClip() {
  const infoStatus = document.getElementById('infoStatus');
  infoStatus.textContent = 'Running AI tagging...';

  fetch(`/api/run_openclip/${encodeURIComponent(currentInfoFilename)}`, {
    method: "POST"
  })
    .then(response => response.json())
    .then(data => {
      if (data.status === "success") {
        infoStatus.textContent = 'AI tagging complete!';

        // Reload the info modal to show new tags
        openInfoModal(currentInfoFilename);
      } else {
        infoStatus.textContent = 'Error running AI tagging: ' + data.message;
      }
    })
    .catch(error => {
      infoStatus.textContent = 'Error running AI tagging';
      console.error("Error running AI tagging:", error);
    });
}

function closeInfoModal() {
  const infoModalEl = document.getElementById('infoModal');
  const infoModal = bootstrap.Modal.getInstance(infoModalEl);
  if (infoModal) {
    infoModal.hide();
  }
}