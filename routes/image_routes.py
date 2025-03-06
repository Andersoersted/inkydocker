from flask import Blueprint, request, redirect, url_for, render_template, flash, send_from_directory, send_file, jsonify, current_app, abort
from models import db, ImageDB, CropInfo, SendLog, Device
import os
from PIL import Image
import subprocess
from utils.image_helpers import allowed_file, convert_to_jpeg
from utils.crop_helpers import load_crop_info_from_db, save_crop_info_to_db, add_send_log_entry, get_last_sent

image_bp = Blueprint('image', __name__)

@image_bp.route('/thumbnail/<filename>')
def thumbnail(filename):
    image_folder = current_app.config['IMAGE_FOLDER']
    thumbnail_folder = current_app.config['THUMBNAIL_FOLDER']
    thumb_path = os.path.join(thumbnail_folder, filename)
    image_path = os.path.join(image_folder, filename)
    if not os.path.exists(image_path):
        return "Not Found", 404
    if not os.path.exists(thumb_path):
        try:
            with Image.open(image_path) as img:
                img.thumbnail((200, 200))
                if img.mode in ("RGBA", "P"):
                    img = img.convert("RGB")
                img.save(thumb_path, "JPEG")
        except Exception as e:
            current_app.logger.error("Error generating thumbnail for %s: %s", filename, e)
            return "Error generating thumbnail", 500
    return send_from_directory(thumbnail_folder, filename)

@image_bp.route('/', methods=['GET', 'POST'])
def upload_file():
    image_folder = current_app.config['IMAGE_FOLDER']
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        files = request.files.getlist('file')
        if not files or all(file.filename == '' for file in files):
            flash('No file selected')
            return redirect(request.url)
        for file in files:
            if file and allowed_file(file.filename):
                from werkzeug.utils import secure_filename
                original_filename = secure_filename(file.filename)
                ext = original_filename.rsplit('.', 1)[1].lower()
                if ext in ['heic', 'nef', 'cr2', 'arw', 'dng']:
                    base = os.path.splitext(original_filename)[0]
                    new_filename = convert_to_jpeg(file, base, image_folder)
                    if new_filename is None:
                        flash(f"Error converting {original_filename} to JPEG.")
                    else:
                        if not ImageDB.query.filter_by(filename=new_filename).first():
                            db.session.add(ImageDB(filename=new_filename))
                            db.session.commit()
                            # Trigger automatic image tagging
                            from tasks import process_image_tagging
                            process_image_tagging.delay(new_filename)
                else:
                    filepath = os.path.join(image_folder, original_filename)
                    file.save(filepath)
                    if not ImageDB.query.filter_by(filename=original_filename).first():
                        db.session.add(ImageDB(filename=original_filename))
                        db.session.commit()
                        # Trigger automatic image tagging
                        from tasks import process_image_tagging
                        process_image_tagging.delay(original_filename)
        return redirect(url_for('image.upload_file'))
    
    images_db = ImageDB.query.all()
    images = [img.filename for img in images_db]
    devices_db = Device.query.all()
    devices = []
    for d in devices_db:
        devices.append({
            "color": d.color,
            "friendly_name": d.friendly_name,
            "orientation": d.orientation,
            "address": d.address,
            "display_name": d.display_name,
            "resolution": d.resolution,
            "online": d.online,
            "last_sent": d.last_sent
        })
    
    # Get all screenshots filenames for checking if an image is a screenshot
    from models import Screenshot
    screenshots = Screenshot.query.all()
    screenshots_filenames = [s.filename for s in screenshots]
    
    last_sent = get_last_sent()
    return render_template('index.html', images=images, devices=devices, last_sent=last_sent, screenshots_filenames=screenshots_filenames)

@image_bp.route('/images/<filename>')
def uploaded_file(filename):
    image_folder = current_app.config['IMAGE_FOLDER']
    filepath = os.path.join(image_folder, filename)
    if not os.path.exists(filepath):
        return "File not found", 404
    if request.args.get("size") == "info":
        try:
            with Image.open(filepath) as img:
                max_width = 300
                w, h = img.size
                if w > max_width:
                    ratio = max_width / float(w)
                    new_size = (max_width, int(h * ratio))
                    img = img.resize(new_size, Image.LANCZOS)
                from io import BytesIO
                buf = BytesIO()
                img.save(buf, format="JPEG")
                buf.seek(0)
                return send_file(buf, mimetype='image/jpeg')
        except Exception as e:
            current_app.logger.error("Error processing image %s for info: %s", filename, e)
            return "Error processing image", 500
    ext = filename.rsplit('.', 1)[1].lower()
    if ext == "heic":
        try:
            from io import BytesIO
            with Image.open(filepath) as img:
                buf = BytesIO()
                img.save(buf, format="JPEG")
                buf.seek(0)
                return send_file(buf, mimetype='image/jpeg')
        except Exception as e:
            current_app.logger.error("Error processing HEIC image %s: %s", filename, e)
            return "Error processing image", 500
    else:
        return send_from_directory(image_folder, filename)

@image_bp.route('/save_crop_info/<filename>', methods=['POST'])
def save_crop_info_endpoint(filename):
    crop_data = request.get_json()
    if not crop_data:
        return jsonify({"status": "error", "message": "No crop data provided"}), 400
    
    # Validate crop data
    required_fields = ["x", "y", "width", "height"]
    for field in required_fields:
        if field not in crop_data or not isinstance(crop_data[field], (int, float)) or crop_data[field] < 0:
            return jsonify({"status": "error", "message": f"Invalid crop data: {field} is missing or invalid"}), 400
    
    # Get the selected device resolution if provided
    if "device" in crop_data:
        device_addr = crop_data.get("device")
        device_obj = Device.query.filter_by(address=device_addr).first()
        if device_obj and device_obj.resolution:
            crop_data["resolution"] = device_obj.resolution
            current_app.logger.info(f"Saving crop with resolution: {device_obj.resolution}")
        else:
            current_app.logger.warning(f"Device not found or missing resolution: {device_addr}")
    else:
        current_app.logger.warning("No device provided for crop data")
    
    save_crop_info_to_db(filename, crop_data)
    return jsonify({"status": "success"}), 200

@image_bp.route('/send_image/<filename>', methods=['POST'])
def send_image(filename):
    image_folder = current_app.config['IMAGE_FOLDER']
    data_folder = current_app.config['DATA_FOLDER']
    filepath = os.path.join(image_folder, filename)
    if not os.path.exists(filepath):
        return "File not found", 404
    device_addr = request.form.get("device")
    if not device_addr:
        return "No device specified", 400

    from models import Device
    device_obj = Device.query.filter_by(address=device_addr).first()
    if not device_obj:
        return "Device not found in DB", 500
    dev_width = None
    dev_height = None
    if device_obj.resolution:
        parts = device_obj.resolution.split("x")
        if len(parts) == 2:
            try:
                dev_width = int(parts[0])
                dev_height = int(parts[1])
            except:
                pass
    if not (dev_width and dev_height):
        return "Target resolution not found", 500

    try:
        with Image.open(filepath) as orig_img:
            orig_w, orig_h = orig_img.size
            
            # Check if device is in portrait orientation
            is_portrait = device_obj.orientation.lower() == 'portrait'
            
            # Store device dimensions in their natural orientation
            if is_portrait:
                # For portrait displays, the natural orientation is height > width
                natural_dev_width = dev_width
                natural_dev_height = dev_height
            else:
                # For landscape displays, the natural orientation is width > height
                natural_dev_width = dev_width
                natural_dev_height = dev_height
            
            # Calculate target ratio - this is only used for auto-cropping when no crop data exists
            # We always calculate it as width/height regardless of orientation
            target_ratio = natural_dev_width / natural_dev_height
                
            # Log the original image dimensions and device info
            current_app.logger.info(f"Original image dimensions: {orig_w}x{orig_h}, device orientation: {device_obj.orientation}, device resolution: {device_obj.resolution}")
            
            # Step 1: Apply crop if available
            cdata = load_crop_info_from_db(filename)
            if cdata and all(key in cdata for key in ["x", "y", "width", "height"]):
                x = cdata.get("x", 0)
                y = cdata.get("y", 0)
                w = cdata.get("width", orig_w)
                h = cdata.get("height", orig_h)
                
                # Validate crop coordinates
                if x < 0 or y < 0 or w <= 0 or h <= 0 or x + w > orig_w or y + h > orig_h:
                    current_app.logger.warning(f"Invalid crop coordinates: ({x}, {y}, {w}, {h}) for image {orig_w}x{orig_h}")
                    # Fall back to auto-centered crop
                    cdata = None
                else:
                    # If we have stored resolution and it matches the current device,
                    # use the stored crop data directly
                    stored_resolution = cdata.get("resolution")
                    current_app.logger.info(f"Stored resolution: {stored_resolution}, device resolution: {device_obj.resolution}")
                    
                    if stored_resolution and stored_resolution == device_obj.resolution:
                        current_app.logger.info(f"Using stored crop data: ({x}, {y}, {w}, {h})")
                        cropped = orig_img.crop((x, y, x+w, y+h))
                    else:
                        # Always use the user's crop selection directly without any adjustments
                        # This respects the user's crop choice regardless of device orientation
                        current_app.logger.info(f"Using stored crop data directly: ({x}, {y}, {w}, {h})")
                        cropped = orig_img.crop((x, y, x+w, y+h))
                        crop_w, crop_h = cropped.size
                        current_app.logger.info(f"Cropped dimensions: {crop_w}x{crop_h}")
            
            # If no valid crop data, create an auto-centered crop
            if not cdata or "x" not in cdata:
                current_app.logger.info(f"No crop data found, using auto-centered crop")
                orig_ratio = orig_w / orig_h
                
                # For auto-cropping, we use the device's natural aspect ratio
                # This ensures the image fits properly on the device
                device_ratio = natural_dev_width / natural_dev_height
                
                # Log the ratios for debugging
                current_app.logger.info(f"Original image ratio: {orig_ratio}, device ratio: {device_ratio}")
                
                if orig_ratio > device_ratio:
                    # Image is wider than device ratio, use full height
                    new_width = int(orig_h * device_ratio)
                    left = (orig_w - new_width) // 2
                    crop_box = (left, 0, left + new_width, orig_h)
                else:
                    # Image is taller than device ratio, use full width
                    new_height = int(orig_w / device_ratio)
                    top = (orig_h - new_height) // 2
                    crop_box = (0, top, orig_w, top + new_height)
                
                current_app.logger.info(f"Auto crop box: {crop_box}")
                cropped = orig_img.crop(crop_box)
                current_app.logger.info(f"Auto-cropped image dimensions: {cropped.size}")

            # Step 2: Resize the cropped image to match the target resolution
            current_app.logger.info(f"Cropped image size before resize/rotation: {cropped.size}")
            
            # If portrait, rotate the image 90 degrees clockwise
            # This rotation happens AFTER cropping, so it doesn't affect the crop selection
            if is_portrait:
                current_app.logger.info("Rotating image for portrait orientation")
                cropped = cropped.rotate(-90, expand=True)  # -90 for clockwise rotation
                current_app.logger.info(f"After rotation size: {cropped.size}")
                # For portrait displays, we swap width and height in the final resize
                final_img = cropped.resize((dev_height, dev_width), Image.LANCZOS)
            else:
                # For landscape displays, we use the normal dimensions
                final_img = cropped.resize((dev_width, dev_height), Image.LANCZOS)
            
            current_app.logger.info(f"Final image size: {final_img.size}, target device resolution: {device_obj.resolution}")
            
            # Save the processed image as a temporary file
            temp_dir = os.path.join(data_folder, "temp")
            if not os.path.exists(temp_dir):
                os.makedirs(temp_dir)
            temp_filename = os.path.join(temp_dir, f"temp_{filename}")
            final_img.save(temp_filename, format="JPEG", quality=95)
            current_app.logger.info(f"Saved temporary file: {temp_filename}")

        cmd = f'curl "{device_addr}/send_image" -X POST -F "file=@{temp_filename}"'
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        os.remove(temp_filename)
        if result.returncode != 0:
            return f"Error sending image: {result.stderr}", 500

        device_obj.last_sent = filename
        db.session.commit()
        add_send_log_entry(filename)
        return f"Image sent successfully: {result.stdout}", 200
    except Exception as e:
        current_app.logger.error("Error resizing/cropping image: %s", e)
        return f"Error processing image: {e}", 500

@image_bp.route('/api/get_images', methods=['GET'])
def get_images():
    try:
        page = int(request.args.get('page', 1))
        per_page = int(request.args.get('per_page', 20))
        
        if page < 1 or per_page < 1 or per_page > 100:
            return jsonify({"status": "error", "message": "Invalid pagination parameters"}), 400
        
        # Calculate offset for pagination
        offset = (page - 1) * per_page
        
        # Get images with pagination
        images_query = ImageDB.query.order_by(ImageDB.id.desc())
        total_images = images_query.count()
        images_page = images_query.offset(offset).limit(per_page).all()
        
        # Format the response
        images_data = []
        for img in images_page:
            image_data = {
                "filename": img.filename,
                "favorite": img.favorite if hasattr(img, 'favorite') else False,
                "tags": img.tags.split(',') if hasattr(img, 'tags') and img.tags else []
            }
            images_data.append(image_data)
        
        return jsonify({
            "status": "success",
            "total": total_images,
            "page": page,
            "per_page": per_page,
            "images": images_data
        }), 200
    except Exception as e:
        current_app.logger.error(f"Error getting images: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500

@image_bp.route('/delete_image/<filename>', methods=['POST'])
def delete_image_endpoint(filename):
    image_folder = current_app.config['IMAGE_FOLDER']
    thumbnail_folder = current_app.config['THUMBNAIL_FOLDER']
    filepath = os.path.join(image_folder, filename)
    thumb_path = os.path.join(thumbnail_folder, filename)
    if os.path.exists(filepath):
        try:
            os.remove(filepath)
            if os.path.exists(thumb_path):
                os.remove(thumb_path)
            img_obj = ImageDB.query.filter_by(filename=filename).first()
            if img_obj:
                db.session.delete(img_obj)
                db.session.commit()
            return jsonify({"status": "success", "message": "Image deleted"}), 200
        except Exception as e:
            current_app.logger.error("Error removing file %s: %s", filepath, e)
            return jsonify({"status": "error", "message": str(e)}), 500
    else:
        return jsonify({"status": "error", "message": "File not found"}), 404
