let currentGallery = null;

document.addEventListener('DOMContentLoaded', () => {
  const configForm = document.getElementById('immichConfigForm');
  if (configForm) {
    configForm.addEventListener('submit', async e => {
      e.preventDefault();
      const address = document.getElementById('immichAddress').value;
      const apiKey = document.getElementById('immichApiKey').value;
      const res = await fetch('/api/immich/config', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ address, api_key: apiKey })
      });
      const data = await res.json();
      document.getElementById('immichConfigStatus').textContent = data.status;
    });
  }

  const refreshBtn = document.getElementById('refreshGalleries');
  if (refreshBtn) {
    refreshBtn.addEventListener('click', loadGalleries);
  }
  loadGalleryClicks();
});

async function loadGalleries() {
  const res = await fetch('/api/immich/galleries');
  const data = await res.json();
  if (data.galleries) {
    const ul = document.getElementById('galleryList');
    ul.innerHTML = '';
    data.galleries.forEach(g => {
      const li = document.createElement('li');
      li.className = 'list-group-item';
      li.dataset.id = g.id || g.gallery_id;
      li.textContent = g.albumName || g.name;
      ul.appendChild(li);
    });
    loadGalleryClicks();
  }
}

function loadGalleryClicks() {
  document.querySelectorAll('#galleryList li').forEach(li => {
    li.addEventListener('click', () => {
      currentGallery = li.dataset.id;
      document.querySelectorAll('#galleryList li').forEach(el => el.classList.remove('active'));
      li.classList.add('active');
      loadAssets(1);
    });
  });
}

async function loadAssets(page=1) {
  if (!currentGallery) return;
  const res = await fetch(`/api/immich/assets/${currentGallery}?page=${page}`);
  const data = await res.json();
  const container = document.getElementById('immichAssets');
  container.innerHTML = '';
  if (data.assets) {
    data.assets.forEach(a => {
      const div = document.createElement('div');
      div.className = 'gallery-item';
      const img = document.createElement('img');
      img.src = `/immich/image/${a.id}?thumbnail=true`;
      img.loading = 'lazy';
      img.addEventListener('click', () => {
        openCropModal(a.id);
      });
      div.appendChild(img);
      container.appendChild(div);
    });
  }
}

function openCropModal(assetId) {
  // Placeholder hook for crop modal integration
  console.log('crop', assetId);
}
