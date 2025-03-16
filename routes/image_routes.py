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
            
            current_app.logger.info(f"Created thumbnails for {filename}, size: {img.size}")
        
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
                    current_app.logger.info(f"Created WebP version for gallery: {webp_path}, size: {img.size}")
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
            current_app.logger.info(f"Device orientation from database: '{device_obj.orientation}', is_portrait: {is_portrait}")
            
            # Calculate aspect ratio based on device orientation
            # This ratio is used for cropping to ensure the image fits the display correctly
            if is_portrait:
                # For portrait displays, use height/width (taller than wide)
                device_ratio = dev_height / dev_width
                current_app.logger.info(f"Portrait display: using height/width ratio = {device_ratio}")
            else:
                # For landscape displays, use width/height (wider than tall)
                device_ratio = dev_width / dev_height
                current_app.logger.info(f"Landscape display: using width/height ratio = {device_ratio}")
                
            # Log the original image dimensions and device info
            current_app.logger.info(f"Original image dimensions: {orig_w}x{orig_h}, device orientation: {device_obj.orientation}, device resolution: {device_obj.resolution}, device ratio: {device_ratio}")
            
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
                
                # For auto-cropping, we use the device's aspect ratio based on orientation
                # This ensures the image fits properly on the device
                # device_ratio is already calculated above based on orientation
                
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

            # Step 2: Resize and rotate the cropped image to match the target resolution and orientation
            current_app.logger.info(f"Cropped image size before resize/rotation: {cropped.size}")
            
            # Check if device is in portrait orientation directly from the database
            is_portrait = device_obj.orientation.lower() == 'portrait'
            current_app.logger.info(f"Device orientation from database: '{device_obj.orientation}', is_portrait: {is_portrait}")
            current_app.logger.info(f"Device resolution from database: {device_obj.resolution} (width x height)")
            # IMPORTANT: The rotation is applied based on the device orientation in the database
            # If the eInk display itself is also rotating the image, this might cause double rotation
            
            # If portrait, rotate the image 90 degrees clockwise
            if is_portrait:
                current_app.logger.info(f"Device is in PORTRAIT mode, rotating image 90° clockwise")
                cropped = cropped.rotate(-90, expand=True)  # -90 for clockwise rotation
                current_app.logger.info(f"After rotation size: {cropped.size}")
                
                # For portrait displays, we swap width and height in the final resize
                # This is because the physical display is rotated, but the native resolution
                # is still reported as if it were in landscape
                current_app.logger.info(f"Swapping dimensions for portrait mode: {dev_width}x{dev_height} -> {dev_height}x{dev_width}")
                final_img = cropped.resize((dev_height, dev_width), Image.LANCZOS)
                current_app.logger.info(f"Final image size after portrait resize: {final_img.size}")
            else:
                current_app.logger.info(f"Device is in LANDSCAPE mode, no rotation needed")
                # For landscape displays, we use the normal dimensions
                final_img = cropped.resize((dev_width, dev_height), Image.LANCZOS)
                current_app.logger.info(f"Final image size after landscape resize: {final_img.size}")
                current_app.logger.info(f"Final image size after landscape resize: {final_img.size}")
            
            current_app.logger.info(f"Final image size: {final_img.size}, target device resolution: {device_obj.resolution}")
            
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
            current_app.logger.info(f"Original image path: {filepath}")
            current_app.logger.info(f"Saved temporary file: {temp_filename}")
            current_app.logger.info(f"Final image dimensions being sent: {final_img.size}")

        # Verify the temporary file exists and has the correct dimensions
        try:
            with Image.open(temp_filename) as verify_img:
                current_app.logger.info(f"Verifying temporary file: {temp_filename}, dimensions: {verify_img.size}")
        except Exception as e:
            current_app.logger.error(f"Error verifying temporary file: {e}")
        
        # Send the temporary file to the device using curl with verbose output
        cmd = f'curl -v "{device_addr}/send_image" -X POST -F "file=@{temp_filename}"'
        current_app.logger.info(f"Sending image with command: {cmd}")
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        # Log the curl response in detail
        current_app.logger.info(f"Curl stdout: {result.stdout}")
        current_app.logger.info(f"Curl stderr (includes request details): {result.stderr}")
        current_app.logger.info(f"Curl return code: {result.returncode}")
            
        # Delete the temporary file after sending
        os.remove(temp_filename)
        current_app.logger.info(f"Temporary file deleted: {temp_filename}")
        
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
