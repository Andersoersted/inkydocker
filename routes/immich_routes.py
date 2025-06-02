from flask import Blueprint, request, jsonify, render_template, current_app, send_file
from models import db, ImmichConfig, ImmichGallery, CropInfo, Device
from utils.crop_helpers import save_crop_info_to_db, load_crop_info_from_db
import httpx
import os
import tempfile
import subprocess
from PIL import Image

immich_bp = Blueprint('immich', __name__)

@immich_bp.route('/immich')
def immich_page():
    config = ImmichConfig.query.filter_by(active=True).first()
    galleries = ImmichGallery.query.all()
    return render_template('immich.html', config=config, galleries=galleries)

@immich_bp.route('/api/immich/config', methods=['POST'])
def save_config():
    data = request.get_json()
    address = data.get('address', '').rstrip('/')
    api_key = data.get('api_key')
    if not address or not api_key:
        return jsonify({'status': 'error', 'message': 'Missing parameters'}), 400
    existing = ImmichConfig.query.filter_by(active=True).first()
    if not existing:
        existing = ImmichConfig(address=address, api_key=api_key, active=True)
        db.session.add(existing)
    else:
        existing.address = address
        existing.api_key = api_key
    db.session.commit()
    return jsonify({'status': 'success'})

@immich_bp.route('/api/immich/galleries')
def fetch_galleries():
    config = ImmichConfig.query.filter_by(active=True).first()
    if not config:
        return jsonify({'status': 'error', 'message': 'No config'}), 400
    url = f"{config.address}/api/albums"
    try:
        resp = httpx.get(url, headers={'x-api-key': config.api_key}, timeout=10)
        resp.raise_for_status()
        albums = resp.json()
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500
    # Update database
    ImmichGallery.query.delete()
    for a in albums:
        gallery = ImmichGallery(gallery_id=str(a.get('id')), name=a.get('albumName', 'Album'))
        db.session.add(gallery)
    db.session.commit()
    return jsonify({'status': 'success', 'galleries': albums})

@immich_bp.route('/api/immich/assets/<gallery_id>')
def get_assets(gallery_id):
    page = int(request.args.get('page', 1))
    limit = int(request.args.get('limit', 50))
    config = ImmichConfig.query.filter_by(active=True).first()
    if not config:
        return jsonify({'status': 'error', 'message': 'No config'}), 400
    url = f"{config.address}/api/albums/{gallery_id}/assets?page={page}&limit={limit}"
    try:
        resp = httpx.get(url, headers={'x-api-key': config.api_key}, timeout=15)
        resp.raise_for_status()
        assets = resp.json()
        return jsonify({'status': 'success', 'assets': assets})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@immich_bp.route('/immich/image/<asset_id>')
def proxy_image(asset_id):
    thumbnail = request.args.get('thumbnail') == 'true'
    config = ImmichConfig.query.filter_by(active=True).first()
    if not config:
        return "No config", 404
    if thumbnail:
        url = f"{config.address}/api/assets/{asset_id}/thumbnail"
    else:
        url = f"{config.address}/api/assets/{asset_id}"
    try:
        r = httpx.get(url, headers={'x-api-key': config.api_key}, timeout=30)
        r.raise_for_status()
    except Exception as e:
        return f"Error: {e}", 500
    temp = tempfile.NamedTemporaryFile(delete=False)
    temp.write(r.content)
    temp.close()
    return send_file(temp.name, mimetype='image/jpeg')

@immich_bp.route('/immich/save_crop_info/<asset_id>', methods=['POST'])
def save_crop(asset_id):
    crop_data = request.get_json()
    key = f"immich:{asset_id}"
    save_crop_info_to_db(key, crop_data)
    saved = load_crop_info_from_db(key)
    return jsonify({'status': 'success', 'data': saved})

@immich_bp.route('/immich/send_image/<asset_id>', methods=['POST'])
def send_image(asset_id):
    device_addr = request.form.get('device')
    if not device_addr:
        return "No device", 400
    device = Device.query.filter_by(address=device_addr).first()
    if not device:
        return "Device not found", 404
    config = ImmichConfig.query.filter_by(active=True).first()
    if not config:
        return "No config", 404
    url = f"{config.address}/api/assets/{asset_id}"
    try:
        r = httpx.get(url, headers={'x-api-key': config.api_key}, timeout=30)
        r.raise_for_status()
    except Exception as e:
        return f"Error: {e}", 500
    temp = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
    temp.write(r.content)
    temp.close()
    # reuse logic from gallery send_image
    filename_key = f"immich:{asset_id}"
    try:
        result = _process_and_send(temp.name, filename_key, device)
    finally:
        try:
            os.remove(temp.name)
        except Exception:
            pass
    if result:
        return jsonify({'status': 'success'})
    return jsonify({'status': 'error', 'message': 'send failed'}), 500

def _process_and_send(filepath, key, device_obj):
    try:
        with Image.open(filepath) as orig_img:
            orig_w, orig_h = orig_img.size
            parts = device_obj.resolution.split('x')
            dev_width = int(parts[0])
            dev_height = int(parts[1])
            is_portrait = device_obj.orientation.lower() == 'portrait'
            device_ratio = dev_height / dev_width if is_portrait else dev_width / dev_height
            crop = load_crop_info_from_db(key)
            if crop:
                x = crop.get('x', 0)
                y = crop.get('y', 0)
                w = crop.get('width', orig_w)
                h = crop.get('height', orig_h)
                cropped = orig_img.crop((x, y, x + w, y + h))
            else:
                orig_ratio = orig_w / orig_h
                if orig_ratio > device_ratio:
                    new_width = int(orig_h * device_ratio)
                    left = (orig_w - new_width) // 2
                    crop_box = (left, 0, left + new_width, orig_h)
                else:
                    new_height = int(orig_w / device_ratio)
                    top = (orig_h - new_height) // 2
                    crop_box = (0, top, orig_w, top + new_height)
                cropped = orig_img.crop(crop_box)
            if is_portrait:
                cropped = cropped.rotate(-90, expand=True)
                final_img = cropped.resize((dev_height, dev_width), Image.LANCZOS)
            else:
                final_img = cropped.resize((dev_width, dev_height), Image.LANCZOS)
            temp_dir = os.path.join(current_app.config.get('DATA_FOLDER', './data'), 'temp')
            os.makedirs(temp_dir, exist_ok=True)
            out_path = os.path.join(temp_dir, os.path.basename(filepath))
            final_img.save(out_path, format='JPEG', quality=95)
    except Exception:
        return False
    url = device_obj.address
    if not url.startswith('http'):
        url = 'http://' + url
    url = f"{url}/send_image"
    curl_cmd = ['curl', '-s', '-F', f"file=@{out_path}", url]
    try:
        subprocess.run(curl_cmd, timeout=120)
    except Exception:
        return False
    os.remove(out_path)
    return True
