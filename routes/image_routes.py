from flask import Blueprint, request, redirect, url_for, render_template, flash, send_from_directory, send_file, jsonify, current_app, abort
from models import db, ImageDB, CropInfo, SendLog, Device
import os
import datetime
from PIL import Image
import subprocess
import httpx
from utils.image_helpers import allowed_file, convert_to_jpeg
from utils.crop_helpers import load_crop_info_from_db, save_crop_info_to_db, add_send_log_entry, get_last_sent

image_bp = Blueprint('image', __name__)

@image_bp.route('/thumbnail/<filename>')
def thumbnail(filename):
    image_folder = current_app.config['IMAGE_FOLDER']
    thumbnail_folder = current_app.config['THUMBNAIL_FOLDER']
    thumb_path = os.path.join(thumbnail_folder, filename)
    webp_thumb_path = os.path.join(thumbnail_folder, os.path.splitext(filename)[0] + '.webp')
    image_path = os.path.join(image_folder, filename)
    
    if not os.path.exists(image_path):
        return "Not Found", 404
    
    # Check for WebP thumbnail first
    if os.path.exists(webp_thumb_path):
        response = send_from_directory(thumbnail_folder, os.path.basename(webp_thumb_path))
        response.headers['Cache-Control'] = 'public, max-age=31536000'
        return response
    
    # If WebP doesn't exist, check for regular thumbnail
    if os.path.exists(thumb_path):
        return send_from_directory(thumbnail_folder, filename)
    
    # If neither exists, create both thumbnails
    try:
        with Image.open(image_path) as img:
            # Get EXIF data to check for orientation
            exif = None
            try:
                exif = img._getexif()
            except (AttributeError, KeyError, IndexError):
                # Not all image formats have EXIF data
                pass
            
            # Auto-orient the image based on EXIF data if available
            # This ensures portrait images display correctly without rotation
            if exif:
                # EXIF orientation tag is 274
                orientation_tag = 274
                if orientation_tag in exif:
                    orientation = exif[orientation_tag]
                    # Apply orientation correction
                    if orientation == 2:
                        img = img.transpose(Image.FLIP_LEFT_RIGHT)
                    elif orientation == 3:
                        img = img.transpose(Image.ROTATE_180)
                    elif orientation == 4:
                        img = img.transpose(Image.FLIP_TOP_BOTTOM)
                    elif orientation == 5:
                        img = img.transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.ROTATE_90)
                    elif orientation == 6:
                        img = img.transpose(Image.ROTATE_270)
                    elif orientation == 7:
                        img = img.transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.ROTATE_270)
                    elif orientation == 8:
                        img = img.transpose(Image.ROTATE_90)
            
            # Create thumbnail
            img.thumbnail((200, 200))
            if img.mode in ("RGBA", "P"):
                img = img.convert("RGB")
            
            # Save regular JPEG thumbnail for compatibility
            img.save(thumb_path, "JPEG")
            
            # Save WebP version for better performance
            img.save(webp_thumb_path, "WEBP", quality=80)
            
            current_app.logger.debug(f"Created thumbnails for {filename}, size: {img.size}")
        
        # Return WebP version
        response = send_from_directory(thumbnail_folder, os.path.basename(webp_thumb_path))
        response.headers['Cache-Control'] = 'public, max-age=31536000'
        return response
    except Exception as e:
        current_app.logger.error("Error generating thumbnail for %s: %s", filename, e)
        return "Error generating thumbnail", 500

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
    webp_folder = os.path.join(current_app.config['DATA_FOLDER'], 'webp_cache')
    
    # Create WebP cache folder if it doesn't exist
    if not os.path.exists(webp_folder):
        os.makedirs(webp_folder)
    
    filepath = os.path.join(image_folder, filename)
    if not os.path.exists(filepath):
        return "File not found", 404
    
    # For info modal preview, create a smaller version
    if request.args.get("size") == "info":
        try:
            with Image.open(filepath) as img:
                # Get EXIF data to check for orientation
                exif = None
                try:
                    exif = img._getexif()
                except (AttributeError, KeyError, IndexError):
                    # Not all image formats have EXIF data
                    pass
                
                # Auto-orient the image based on EXIF data if available
                if exif:
                    # EXIF orientation tag is 274
                    orientation_tag = 274
                    if orientation_tag in exif:
                        orientation = exif[orientation_tag]
                        # Apply orientation correction
                        if orientation == 2:
                            img = img.transpose(Image.FLIP_LEFT_RIGHT)
                        elif orientation == 3:
                            img = img.transpose(Image.ROTATE_180)
                        elif orientation == 4:
                            img = img.transpose(Image.FLIP_TOP_BOTTOM)
                        elif orientation == 5:
                            img = img.transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.ROTATE_90)
                        elif orientation == 6:
                            img = img.transpose(Image.ROTATE_270)
                        elif orientation == 7:
                            img = img.transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.ROTATE_270)
                        elif orientation == 8:
                            img = img.transpose(Image.ROTATE_90)
                
                # Resize for info modal
                max_width = 300
                w, h = img.size
                if w > max_width:
                    ratio = max_width / float(w)
                    new_size = (max_width, int(h * ratio))
                    img = img.resize(new_size, Image.LANCZOS)
                
                # Convert to RGB if needed
                if img.mode in ("RGBA", "P"):
                    img = img.convert("RGB")
                    
                from io import BytesIO
                buf = BytesIO()
                img.save(buf, format="WEBP", quality=85)
                buf.seek(0)
                return send_file(buf, mimetype='image/webp')
        except Exception as e:
            current_app.logger.error("Error processing image %s for info: %s", filename, e)
            return "Error processing image", 500
    
    # For gallery display, use WebP if available or create it
    if request.args.get("for") == "gallery" or request.headers.get('Accept', '').find('image/webp') != -1:
        webp_path = os.path.join(webp_folder, os.path.splitext(filename)[0] + '.webp')
        
        # If WebP doesn't exist or is older than original, create it
        if not os.path.exists(webp_path) or os.path.getmtime(webp_path) < os.path.getmtime(filepath):
            try:
                with Image.open(filepath) as img:
                    # Get EXIF data to check for orientation
                    exif = None
                    try:
                        exif = img._getexif()
                    except (AttributeError, KeyError, IndexError):
                        # Not all image formats have EXIF data
                        pass
                    
                    # Auto-orient the image based on EXIF data if available
                    # This ensures portrait images display correctly without rotation
                    if exif:
                        # EXIF orientation tag is 274
                        orientation_tag = 274
                        if orientation_tag in exif:
                            orientation = exif[orientation_tag]
                            # Apply orientation correction
                            if orientation == 2:
                                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                            elif orientation == 3:
                                img = img.transpose(Image.ROTATE_180)
                            elif orientation == 4:
                                img = img.transpose(Image.FLIP_TOP_BOTTOM)
                            elif orientation == 5:
                                img = img.transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.ROTATE_90)
                            elif orientation == 6:
                                img = img.transpose(Image.ROTATE_270)
                            elif orientation == 7:
                                img = img.transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.ROTATE_270)
                            elif orientation == 8:
                                img = img.transpose(Image.ROTATE_90)
                    
                    # Resize for gallery if image is very large
                    w, h = img.size
                    max_dimension = 800  # Reasonable size for gallery display
                    if w > max_dimension or h > max_dimension:
                        if w > h:
                            ratio = max_dimension / float(w)
                            new_size = (max_dimension, int(h * ratio))
                        else:
                            ratio = max_dimension / float(h)
                            new_size = (int(w * ratio), max_dimension)
                        img = img.resize(new_size, Image.LANCZOS)
                    
                    # Convert to RGB if needed
                    if img.mode in ("RGBA", "P"):
                        img = img.convert("RGB")
                    
                    # Save as WebP - DO NOT rotate portrait images for gallery view
                    img.save(webp_path, "WEBP", quality=85)
                    current_app.logger.debug(f"Created WebP version for gallery: {webp_path}, size: {img.size}")
            except Exception as e:
                current_app.logger.error("Error creating WebP for %s: %s", filename, e)
                # Fall back to original if WebP creation fails
                return send_from_directory(image_folder, filename)
        
        # Serve WebP with caching headers
        response = send_from_directory(webp_folder, os.path.basename(webp_path))
        response.headers['Cache-Control'] = 'public, max-age=31536000'
        return response
    
    # For HEIC files, convert to JPEG on-the-fly
    ext = filename.rsplit('.', 1)[1].lower() if '.' in filename else ''
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
    
    # For original image requests (sending to eInk display), serve the original
    return send_from_directory(image_folder, filename)

@image_bp.route('/save_crop_info/<filename>', methods=['POST'])
def save_crop_info_endpoint(filename):
    import json
    crop_data = request.get_json()
    current_app.logger.info(f"Received crop data for {filename}: {json.dumps(crop_data)}")
    
    if not crop_data:
        current_app.logger.error(f"No crop data provided for {filename}")
        return jsonify({"status": "error", "message": "No crop data provided"}), 400
    
    # Validate crop data
    required_fields = ["x", "y", "width", "height"]
    for field in required_fields:
        if field not in crop_data or not isinstance(crop_data[field], (int, float)) or crop_data[field] < 0:
            current_app.logger.error(f"Invalid crop data for {filename}: {field} is missing or invalid")
            return jsonify({"status": "error", "message": f"Invalid crop data: {field} is missing or invalid"}), 400
    
    # Get the selected device resolution if provided
    if "device" in crop_data:
        device_addr = crop_data.get("device")
        device_obj = Device.query.filter_by(address=device_addr).first()
        if device_obj and device_obj.resolution:
            crop_data["resolution"] = device_obj.resolution
            current_app.logger.info(f"Saving crop for {filename} with resolution: {device_obj.resolution}")
        else:
            current_app.logger.warning(f"Device not found or missing resolution: {device_addr} for {filename}")
    else:
        current_app.logger.warning(f"No device provided for crop data for {filename}")
    
    # Check for existing crop data before saving
    existing = CropInfo.query.filter_by(filename=filename).first()
    if existing:
        current_app.logger.info(f"Updating existing crop for {filename}. Old values: x={existing.x}, y={existing.y}, w={existing.width}, h={existing.height}")
    else:
        current_app.logger.info(f"Creating new crop record for {filename}")
    
    # Save the crop data
    try:
        save_crop_info_to_db(filename, crop_data)
        
        # Verify that the data was actually saved correctly by reloading it
        db.session.expire_all()  # Force reload from database
        saved_data = load_crop_info_from_db(filename)
        if saved_data:
            current_app.logger.info(f"Verified crop data for {filename} was saved: x={saved_data['x']}, y={saved_data['y']}, w={saved_data['width']}, h={saved_data['height']}")
            
            # Check if values match what was submitted
            all_match = True
            for field in required_fields:
                if abs(saved_data[field] - crop_data[field]) > 0.01:  # Allow for small floating point differences
                    current_app.logger.warning(f"Mismatch in saved crop data for {filename}: {field} should be {crop_data[field]} but is {saved_data[field]}")
                    all_match = False
            
            if all_match:
                current_app.logger.info(f"All crop values for {filename} match the submitted data")
            
            return jsonify({
                "status": "success",
                "message": "Crop info saved successfully",
                "updated_at": saved_data.get("updated_at")
            }), 200
        else:
            current_app.logger.error(f"Failed to verify crop data was saved for {filename}")
            return jsonify({"status": "error", "message": "Failed to verify crop data was saved"}), 500
    except Exception as e:
        current_app.logger.error(f"Error saving crop data for {filename}: {str(e)}")
        return jsonify({"status": "error", "message": f"Database error: {str(e)}"}), 500

@image_bp.route('/send_image/<filename>', methods=['POST'])
@image_bp.route('/send_image', methods=['POST'])
def send_image(filename=None):
    # If filename is not provided in the URL, get it from the form data
    if not filename:
        filename = request.form.get("filename")
        if not filename:
            current_app.logger.error("[GALLERY] No filename specified in request")
            return "No filename specified", 400
    image_folder = current_app.config['IMAGE_FOLDER']
    data_folder = current_app.config['DATA_FOLDER']
    
    # Log only basic request details
    current_app.logger.debug(f"[GALLERY] Send image request received for filename: {filename}")
    
    filepath = os.path.join(image_folder, filename)
    if not os.path.exists(filepath):
        current_app.logger.error(f"[GALLERY] File not found: {filepath}")
        return "File not found", 404
    
    device_addr = request.form.get("device")
    if not device_addr:
        current_app.logger.error("[GALLERY] No device specified in request")
        return "No device specified", 400

    current_app.logger.debug(f"[GALLERY] Sending to device: {device_addr}")
    
    from models import Device
    device_obj = Device.query.filter_by(address=device_addr).first()
    if not device_obj:
        current_app.logger.error(f"[GALLERY] Device not found in DB: {device_addr}")
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
            current_app.logger.debug(f"Device orientation from database: '{device_obj.orientation}', is_portrait: {is_portrait}")
            
            # Calculate aspect ratio based on device orientation
            # This ratio is used for cropping to ensure the image fits the display correctly
            if is_portrait:
                # For portrait displays, use height/width (taller than wide)
                device_ratio = dev_height / dev_width
                current_app.logger.debug(f"Portrait display: using height/width ratio = {device_ratio}")
            else:
                # For landscape displays, use width/height (wider than tall)
                device_ratio = dev_width / dev_height
                current_app.logger.debug(f"Landscape display: using width/height ratio = {device_ratio}")
                
            # Log the original image dimensions and device info
            current_app.logger.debug(f"Original image dimensions: {orig_w}x{orig_h}, device orientation: {device_obj.orientation}, device resolution: {device_obj.resolution}, device ratio: {device_ratio}")
            
            # IMPROVED CROPPING AND RESIZING LOGIC
            current_app.logger.debug(f"Processing image using improved cropping and resizing")
            
            # Step 1: First determine the exact target aspect ratio based on device orientation
            # We need to be very precise here as this ratio determines our crop
            is_portrait = device_obj.orientation.lower() == 'portrait'
            if is_portrait:
                # For portrait displays, height > width (taller than wide)
                # But the native device resolution is always in landscape mode, so we swap
                device_ratio = dev_height / dev_width  # This is now > 1.0 for portrait
                target_width = dev_height  # The width of the final image should be the device's height
                target_height = dev_width  # The height of the final image should be the device's width
                current_app.logger.debug(f"Portrait display: using height/width ratio = {device_ratio}")
                current_app.logger.debug(f"Target dimensions (after rotation): {target_width}x{target_height}")
            else:
                # For landscape displays, width > height (wider than tall)
                device_ratio = dev_width / dev_height  # This is > 1.0 for landscape
                target_width = dev_width
                target_height = dev_height
                current_app.logger.debug(f"Landscape display: using width/height ratio = {device_ratio}")
                current_app.logger.debug(f"Target dimensions: {target_width}x{target_height}")
            
            # Step 2: Apply crop if available or do auto-crop to match the target aspect ratio
            # Force refresh from database to ensure we get the latest crop info
            db.session.expire_all()
            cdata = load_crop_info_from_db(filename)
            
            # Log when the crop data was last updated if available
            if cdata and "updated_at" in cdata and cdata["updated_at"]:
                current_app.logger.info(f"Crop data for {filename} was last updated at: {cdata['updated_at']}")
            
            if cdata and all(key in cdata for key in ["x", "y", "width", "height"]):
                # We have manual crop data from the user
                x = cdata.get("x", 0)
                y = cdata.get("y", 0)
                w = cdata.get("width", orig_w)
                h = cdata.get("height", orig_h)
                
                # Log the crop data we're using
                current_app.logger.info(f"Using crop data for {filename}: x={x}, y={y}, w={w}, h={h}, resolution={cdata.get('resolution')}")
                
                # Validate crop coordinates
                if x < 0 or y < 0 or w <= 0 or h <= 0 or x + w > orig_w or y + h > orig_h:
                    current_app.logger.warning(f"Invalid crop coordinates: ({x}, {y}, {w}, {h}) for image {orig_w}x{orig_h}")
                    # Fall back to auto-centered crop (handled in else branch below)
                    cdata = None
                else:
                    # We have valid crop coordinates
                    stored_resolution = cdata.get("resolution")
                    current_app.logger.debug(f"Stored crop info: x={x}, y={y}, width={w}, height={h} for resolution: {stored_resolution}")
                    
                    # CRITICAL ISSUE: If the crop dimensions match exactly the device resolution,
                    # it suggests the crop data is actually representing a small portion of a larger image,
                    # so we need to interpret it differently
                    is_device_res_crop = (stored_resolution == device_obj.resolution and
                                         w == dev_width and h == dev_height)
                    
                    if is_device_res_crop:
                        current_app.logger.debug(f"Detected crop with exact device resolution dimensions")
                        # This needs special handling - the stored crop is actually representing a region of interest
                        # We need to scale it according to the original image size
                        
                        # First, get the full dimensions of the original image
                        current_app.logger.debug(f"Original image dimensions: {orig_w}x{orig_h}")
                        
                        # Calculate the actual crop coordinates based on the aspect ratio
                        # This ensures we get the full image while maintaining the correct aspect ratio
                        if is_portrait:
                            # For portrait mode, we need a crop with aspect ratio of height/width
                            target_ratio = dev_height / dev_width  # E.g., 800/480 = 1.667
                        else:
                            # For landscape mode, we need a crop with aspect ratio of width/height
                            target_ratio = dev_width / dev_height  # E.g., 800/480 = 1.667
                        
                        current_app.logger.debug(f"Target aspect ratio: {target_ratio}")
                        
                        # Calculate dimensions to keep the entire original image visible
                        # but cropped to the correct aspect ratio
                        if orig_w / orig_h > target_ratio:  # Image is wider than needed
                            # Use full height, calculate width based on aspect ratio
                            crop_height = orig_h
                            crop_width = int(crop_height * target_ratio)
                            # Center the crop horizontally
                            crop_x = (orig_w - crop_width) // 2
                            crop_y = 0
                        else:  # Image is taller than needed
                            # Use full width, calculate height based on aspect ratio
                            crop_width = orig_w
                            crop_height = int(crop_width / target_ratio)
                            # Center the crop vertically
                            crop_x = 0
                            crop_y = (orig_h - crop_height) // 2
                        
                        # Apply the calculated crop
                        new_crop_box = (crop_x, crop_y, crop_x + crop_width, crop_y + crop_height)
                        current_app.logger.debug(f"Full image crop box: {new_crop_box}")
                        cropped = orig_img.crop(new_crop_box)
                        current_app.logger.debug(f"Cropped to: {cropped.size}, ratio: {crop_width/crop_height:.4f}")
                        
                    else:
                        # Normal case: Apply the user's crop selection
                        cropped = orig_img.crop((x, y, x+w, y+h))
                        crop_w, crop_h = cropped.size
                        crop_ratio = crop_w / crop_h
                        current_app.logger.debug(f"User crop dimensions: {crop_w}x{crop_h}, ratio: {crop_ratio}")
                        
                        # Check if the crop matches our target aspect ratio
                        # If the aspect ratio is off by more than 1%, we need to adjust it
                        # This ensures the image will properly fit the device's display
                        ratio_diff = abs(crop_ratio - device_ratio) / device_ratio
                        current_app.logger.debug(f"Aspect ratio difference: {ratio_diff * 100:.2f}%")
                        
                        if is_portrait:
                            # For portrait mode, we need to compare reciprocals
                            portrait_crop_ratio = crop_h / crop_w
                            portrait_device_ratio = 1 / device_ratio
                            ratio_diff = abs(portrait_crop_ratio - portrait_device_ratio) / portrait_device_ratio
                            current_app.logger.debug(f"Portrait aspect ratio difference: {ratio_diff * 100:.2f}%")
                        
                        if ratio_diff > 0.01:  # More than 1% difference
                            current_app.logger.debug(f"Crop aspect ratio ({crop_ratio:.4f}) doesn't match device ratio ({device_ratio:.4f}), adjusting")
                            
                            # We need to re-crop to match the target aspect ratio
                            # While preserving as much of the user's original crop as possible
                            if crop_ratio > device_ratio:  # Crop is too wide
                                # Need to make it narrower - adjust width while keeping center
                                new_crop_w = int(crop_h * device_ratio)
                                diff = crop_w - new_crop_w
                                new_x = x + (diff // 2)
                                new_crop_box = (new_x, y, new_x + new_crop_w, y + crop_h)
                                current_app.logger.debug(f"Adjusting width: old={crop_w}, new={new_crop_w}")
                            else:  # Crop is too tall
                                # Need to make it shorter - adjust height while keeping center
                                new_crop_h = int(crop_w / device_ratio)
                                diff = crop_h - new_crop_h
                                new_y = y + (diff // 2)
                                new_crop_box = (x, new_y, x + crop_w, new_y + new_crop_h)
                                current_app.logger.debug(f"Adjusting height: old={crop_h}, new={new_crop_h}")
                                
                            current_app.logger.debug(f"Adjusted crop box: {new_crop_box}")
                            cropped = orig_img.crop(new_crop_box)
                        else:
                            current_app.logger.debug(f"User crop matches target aspect ratio within 1% tolerance")
            
            # If no valid crop data, create an auto-centered crop to match the target aspect ratio
            if not cdata or "x" not in cdata:
                current_app.logger.debug(f"No crop data found, using auto-centered crop")
                orig_ratio = orig_w / orig_h
                current_app.logger.debug(f"Original image ratio: {orig_ratio}, target device ratio: {device_ratio}")
                
                if orig_ratio > device_ratio:
                    # Image is wider than device ratio, use full height and center width
                    new_width = int(orig_h * device_ratio)
                    left = (orig_w - new_width) // 2
                    crop_box = (left, 0, left + new_width, orig_h)
                else:
                    # Image is taller than device ratio, use full width and center height
                    new_height = int(orig_w / device_ratio)
                    top = (orig_h - new_height) // 2
                    crop_box = (0, top, orig_w, top + new_height)
                
                current_app.logger.debug(f"Auto crop box: {crop_box}")
                cropped = orig_img.crop(crop_box)
                crop_w, crop_h = cropped.size
                current_app.logger.debug(f"Auto-cropped image dimensions: {crop_w}x{crop_h}, ratio: {crop_w/crop_h:.4f}")
            
            # Step 3: Now we have a properly cropped image with the correct aspect ratio
            # Resize it to match the exact target resolution for the device
            # We always resize BEFORE rotation for better quality
            current_app.logger.debug(f"Cropped image size before final resize: {cropped.size}")
            
            if is_portrait:
                # For portrait mode, we have a critical detail to manage:
                # 1. We need to crop to the PORTRAIT aspect ratio (already done in previous step)
                # 2. Resize to match the target dimensions (swapped for portrait)
                # 3. Rotate the image afterward to match device orientation
                
                # First, double check our target dimensions for portrait mode
                current_app.logger.debug(f"Portrait target dimensions (before rotation): {target_width}x{target_height}")
                
                # Resize to our target (still in landscape orientation)
                final_img = cropped.resize((target_width, target_height), Image.LANCZOS)
                resize_w, resize_h = final_img.size
                current_app.logger.debug(f"Resized to: {resize_w}x{resize_h}")
                
                # Verify resize worked as expected
                if abs(resize_w - target_width) > 1 or abs(resize_h - target_height) > 1:
                    current_app.logger.warning(f"Resize dimensions don't match target: got {resize_w}x{resize_h}, expected {target_width}x{target_height}")
                
                # Rotate 90 degrees clockwise for portrait display
                current_app.logger.debug(f"Rotating image 90° clockwise for portrait display")
                final_img = final_img.rotate(-90, expand=True)  # -90 degrees = clockwise rotation
                rotated_w, rotated_h = final_img.size
                current_app.logger.debug(f"Rotated image dimensions: {rotated_w}x{rotated_h}")
                
                # After rotation, dimensions should be swapped
                # In portrait mode, the final dimensions should be (height, width)
                expected_w, expected_h = target_height, target_width
                if abs(rotated_w - expected_w) > 1 or abs(rotated_h - expected_h) > 1:
                    current_app.logger.warning(f"Rotated dimensions are incorrect: got {rotated_w}x{rotated_h}, expected {expected_w}x{expected_h}")
                    # Attempt to fix by explicit resize after rotation if needed
                    if abs(rotated_w - expected_w) > 5 or abs(rotated_h - expected_h) > 5:
                        current_app.logger.debug(f"Fixing rotated dimensions with explicit resize")
                        final_img = final_img.resize((expected_w, expected_h), Image.LANCZOS)
                        current_app.logger.debug(f"Fixed dimensions: {final_img.size}")
            else:
                # For landscape orientation, simply resize to target dimensions
                final_img = cropped.resize((target_width, target_height), Image.LANCZOS)
                resize_w, resize_h = final_img.size
                current_app.logger.debug(f"Landscape final size: {resize_w}x{resize_h}")
                
                # Verify resize worked as expected
                if abs(resize_w - target_width) > 1 or abs(resize_h - target_height) > 1:
                    current_app.logger.warning(f"Resize dimensions don't match target: got {resize_w}x{resize_h}, expected {target_width}x{target_height}")
                    # Attempt to fix by explicit resize if needed
                    if abs(resize_w - target_width) > 5 or abs(resize_h - target_height) > 5:
                        current_app.logger.debug(f"Fixing dimensions with explicit resize")
                        final_img = final_img.resize((target_width, target_height), Image.LANCZOS)
                        current_app.logger.debug(f"Fixed dimensions: {final_img.size}")
            
            current_app.logger.debug(f"Final image size: {final_img.size}, target device resolution: {device_obj.resolution}")
            
            # Save the processed image as a temporary file
            temp_dir = os.path.join(data_folder, "temp")
            if not os.path.exists(temp_dir):
                os.makedirs(temp_dir)
            
            # Create a unique temporary filename to avoid any caching issues
            import uuid
            unique_id = uuid.uuid4().hex[:8]
            temp_filename = os.path.join(temp_dir, f"temp_{unique_id}_{filename}")
            
            # Save the final image with high quality
            final_img.save(temp_filename, format="JPEG", quality=95)
            current_app.logger.debug(f"Original image path: {filepath}")
            current_app.logger.debug(f"Saved temporary file: {temp_filename}")
            current_app.logger.debug(f"Final image dimensions being sent: {final_img.size}")

        # Verify the temporary file exists and has the correct dimensions
        try:
            with Image.open(temp_filename) as verify_img:
                current_app.logger.debug(f"Verifying temporary file: {temp_filename}, dimensions: {verify_img.size}")
        except Exception as e:
            current_app.logger.error(f"Error verifying temporary file: {e}")
        # Create a unique identifier for this gallery send
        import uuid
        send_id = uuid.uuid4().hex[:8]
        
        # Log the image details before sending
        current_app.logger.debug(f"[GALLERY-{send_id}] Sending image {filename} to device {device_obj.friendly_name} at {device_addr}")
        current_app.logger.debug(f"[GALLERY-{send_id}] Temporary file path: {temp_filename}")
        
        # Ensure device address has HTTP protocol
        if not device_addr.startswith(('http://', 'https://')):
            device_addr = f'http://{device_addr}'
        
        # Prepare URL for the request - ensure we're using the correct endpoint
        # The e-ink display is expecting just '/send_image', not '/send_image/filename'
        url = f"{device_addr}/send_image"
        current_app.logger.debug(f"[GALLERY-{send_id}] Sending request to: {url}")
        
        try:
            # Fix: Use exactly the curl command that works but with greater verbosity for debugging
            # First, make sure the URL doesn't have any query parameters
            base_url = url.split('?')[0]
            
            # Add verbose diagnostic logging
            current_app.logger.debug(f"[GALLERY-{send_id}] Image properties: temp_file={temp_filename}")
            current_app.logger.debug(f"[GALLERY-{send_id}] Target device: {device_obj.friendly_name} at {device_addr}")
            current_app.logger.debug(f"[GALLERY-{send_id}] Using URL: {base_url}")
            
            # Get the file size for logging
            file_size = os.path.getsize(temp_filename)
            current_app.logger.debug(f"[GALLERY-{send_id}] File size: {file_size} bytes")
            
            # Save a copy of the image for debugging purposes
            debug_copy = os.path.join(temp_dir, f"debug_{unique_id}_{filename}")
            import shutil
            shutil.copy2(temp_filename, debug_copy)
            current_app.logger.debug(f"[GALLERY-{send_id}] Debug copy saved to: {debug_copy}")
            
            # Verify the image with Pillow to ensure it's not corrupted
            try:
                with Image.open(temp_filename) as verify_img:
                    current_app.logger.debug(f"[GALLERY-{send_id}] Image verification: format={verify_img.format}, size={verify_img.size}, mode={verify_img.mode}")
            except Exception as e:
                current_app.logger.error(f"[GALLERY-{send_id}] Image verification failed: {e}")
                return f"Error with processed image: {e}", 500
                
            # Execute the exact curl command that we know works
            curl_cmd = ["curl", "-v", "-F", f"file=@{temp_filename}", base_url]
            current_app.logger.debug(f"[GALLERY-{send_id}] Executing curl command: {' '.join(curl_cmd)}")
            
            try:
                # Run curl with verbose output to see exactly what's happening
                result = subprocess.run(curl_cmd, capture_output=True, text=True, timeout=120)
                
                # Log comprehensive information about the curl command result
                current_app.logger.debug(f"[GALLERY-{send_id}] Curl exit code: {result.returncode}")
                current_app.logger.debug(f"[GALLERY-{send_id}] Curl stdout: {result.stdout}")
                current_app.logger.debug(f"[GALLERY-{send_id}] Curl stderr: {result.stderr}")
                
                # Write to a separate log file for easier debugging
                with open('/tmp/curl_debug_log.txt', 'a') as f:
                    f.write(f"\n{'-'*80}\n{datetime.datetime.now()}: [GALLERY-{send_id}] Curl debug log\n")
                    f.write(f"Command: {' '.join(curl_cmd)}\n")
                    f.write(f"Exit code: {result.returncode}\n")
                    f.write(f"Stdout:\n{result.stdout}\n")
                    f.write(f"Stderr:\n{result.stderr}\n")
                
                # Create a response-like object
                class CurlResponse:
                    def __init__(self, text, code):
                        self.text = text
                        self.status_code = code
                
                # Use the curl exit code to determine success or failure
                status_code = 200 if result.returncode == 0 else 500
                
                # Check for known error patterns in the output
                if "Error" in result.stdout or "error" in result.stdout:
                    current_app.logger.error(f"[GALLERY-{send_id}] Error detected in curl stdout: {result.stdout}")
                    response = CurlResponse(result.stdout, 500)
                else:
                    response = CurlResponse(result.stdout, status_code)
                
                # Delete the debug copy after 10 minutes to avoid filling disk
                def delete_debug_copy():
                    try:
                        if os.path.exists(debug_copy):
                            os.remove(debug_copy)
                            current_app.logger.debug(f"Debug copy {debug_copy} auto-deleted after 10 minutes")
                    except Exception as e:
                        current_app.logger.error(f"Error deleting debug copy: {e}")
                
                import threading
                timer = threading.Timer(600, delete_debug_copy)
                timer.daemon = True
                timer.start()
                
                # Return error if curl failed
                if result.returncode != 0:
                    return f"Error sending image with curl: {result.stderr}", 500
            
            except subprocess.TimeoutExpired:
                current_app.logger.error(f"[GALLERY-{send_id}] Curl command timed out after 120 seconds")
                response = CurlResponse("Curl command timed out", 500)
                return "Request timed out while sending the image to the device", 500
            
            except Exception as e:
                current_app.logger.error(f"[GALLERY-{send_id}] Exception executing curl: {e}")
                response = CurlResponse(str(e), 500)
                return f"Error sending image: {str(e)}", 500
                
                # Log the response details
                current_app.logger.debug(f"[GALLERY-{send_id}] Response status code: {response.status_code}")
                current_app.logger.debug(f"[GALLERY-{send_id}] Response headers: {response.headers}")
                current_app.logger.debug(f"[GALLERY-{send_id}] Response content: {response.text}")
                
                # Write to a separate log file for easier debugging
                with open('/tmp/gallery_send_log.txt', 'a') as f:
                    f.write(f"\n{'-'*80}\n{datetime.datetime.now()}: [GALLERY-{send_id}] Sending image to {device_addr}\n")
                    f.write(f"Filename: {filename}\n")
                    f.write(f"URL: {url}\n")
                    f.write(f"Status code: {response.status_code}\n")
                    f.write(f"Response: {response.text}\n")
                
                # Delete the temporary file after sending
                try:
                    os.remove(temp_filename)
                    current_app.logger.debug(f"[GALLERY-{send_id}] Temporary file deleted: {temp_filename}")
                except Exception as e:
                    current_app.logger.error(f"[GALLERY-{send_id}] Error deleting temporary file: {e}")
                
                # Check if the request was successful
                if response.status_code != 200:
                    current_app.logger.error(f"[GALLERY-{send_id}] Error sending image: {response.text}")
                    with open('/tmp/gallery_send_log.txt', 'a') as f:
                        f.write(f"ERROR: Failed to send image. Status code: {response.status_code}\n")
                    return f"Error sending image: {response.text}", 500
                
                # If we get here, the request was successful

            # Update the device's last_sent field
            device_obj.last_sent = filename
            db.session.commit()
            current_app.logger.debug(f"[GALLERY-{send_id}] Updated device {device_obj.friendly_name} last_sent to {filename}")
            
            # Add a log entry for this send operation
            add_send_log_entry(filename)
            current_app.logger.debug(f"[GALLERY-{send_id}] Added send log entry for {filename}")
            
            # Write completion to log file
            with open('/tmp/gallery_send_log.txt', 'a') as f:
                f.write(f"{datetime.datetime.now()}: [GALLERY-{send_id}] Successfully completed gallery send\n")
                
            return "Image sent successfully", 200
            
        except httpx.TimeoutException:
            current_app.logger.error(f"[GALLERY-{send_id}] HTTP request timed out after 120 seconds")
            try:
                os.remove(temp_filename)
            except:
                pass
            return "Request timed out while sending the image to the device", 500
        except httpx.RequestError as e:
            current_app.logger.error(f"[GALLERY-{send_id}] HTTP request error: {e}")
            try:
                os.remove(temp_filename)
            except:
                pass
            return f"Network error while sending the image: {str(e)}", 500
        except Exception as e:
            current_app.logger.error(f"[GALLERY-{send_id}] Unexpected error: {e}")
            try:
                os.remove(temp_filename)
            except:
                pass
            return f"Error sending image: {str(e)}", 500
    except Exception as e:
        current_app.logger.error("Error resizing/cropping image: %s", e)
        return f"Error processing image: {e}", 500

@image_bp.route('/api/get_current_image', methods=['GET'])
def get_current_image():
    """Get information about the currently displayed image on a device."""
    try:
        # Get the device address from query parameters, or use the first device if not specified
        device_addr = request.args.get("device")
        
        if device_addr:
            # Find the specific device
            device = Device.query.filter_by(address=device_addr).first()
            if not device:
                return jsonify({
                    "status": "error",
                    "message": f"Device not found with address: {device_addr}"
                }), 404
            
            # Get the last sent image for this specific device
            filename = device.last_sent
            if not filename:
                return jsonify({
                    "status": "info",
                    "message": "No image has been sent to this device yet"
                }), 200
                
            # Return information about the current image
            return jsonify({
                "status": "success",
                "current_image": {
                    "filename": filename,
                    "device": device.friendly_name,
                    "sent_at": datetime.datetime.now().isoformat(),  # Ideally this would be stored in the DB
                    "display_url": f"/images/{filename}"
                }
            }), 200
        else:
            # No device specified, return info for all devices
            devices = Device.query.all()
            devices_info = []
            
            for device in devices:
                if device.last_sent:
                    devices_info.append({
                        "device_name": device.friendly_name,
                        "device_address": device.address,
                        "current_image": device.last_sent,
                        "display_url": f"/images/{device.last_sent}" if device.last_sent else None
                    })
            
            return jsonify({
                "status": "success",
                "devices": devices_info
            }), 200
    
    except Exception as e:
        current_app.logger.error(f"Error retrieving current image info: {e}")
        return jsonify({
            "status": "error",
            "message": f"Error retrieving current image information: {str(e)}"
        }), 500

@image_bp.route('/api/get_crop_info/<filename>', methods=['GET'])
def get_crop_info(filename):
    """Get crop information for an image."""
    # Force a refresh from database to ensure we get the latest crop info
    db.session.expire_all()
    
    # Check if crop info exists for this filename
    crop_info = CropInfo.query.filter_by(filename=filename).first()
    
    if crop_info:
        # Log the crop data we're returning
        current_app.logger.info(f"Returning crop info for {filename}: x={crop_info.x}, y={crop_info.y}, w={crop_info.width}, h={crop_info.height}")
        
        # Return the crop info as JSON with updated_at timestamp
        return jsonify({
            "status": "success",
            "crop_info": {
                "x": crop_info.x,
                "y": crop_info.y,
                "width": crop_info.width,
                "height": crop_info.height,
                "resolution": crop_info.resolution,
                "updated_at": crop_info.updated_at.isoformat() if hasattr(crop_info, 'updated_at') and crop_info.updated_at else None
            }
        }), 200
    else:
        # No crop info found
        current_app.logger.info(f"No crop information found for {filename}")
        return jsonify({
            "status": "error",
            "message": "No crop information found for this image"
        }), 404

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
