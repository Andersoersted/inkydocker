from flask import Blueprint, request, redirect, url_for, render_template, flash, send_from_directory, send_file, jsonify, current_app, abort
from models import db, ImageDB, CropInfo, SendLog, Device
from utils.notification_service import NotificationService, notify_on_completion
import os
import datetime
from PIL import Image
import subprocess
import httpx
import traceback # Ensure traceback is imported
import uuid # Ensure uuid is imported
import asyncio # Add asyncio import
import shutil # Ensure shutil is imported
from utils.image_helpers import allowed_file, convert_to_jpeg
from utils.crop_helpers import load_crop_info_from_db, save_crop_info_to_db, add_send_log_entry, get_last_sent
from io import BytesIO # Ensure BytesIO is imported
from tasks import send_image_via_httpx_task # Import the new Celery task

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
            if exif:
                orientation_tag = 274
                if orientation_tag in exif:
                    orientation = exif[orientation_tag]
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
        NotificationService.create_info("Processing image upload...")
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
                ext = original_filename.rsplit('.', 1)[1].lower() if '.' in original_filename else ''
                if ext in ['heic', 'nef', 'cr2', 'arw', 'dng']:
                    base = os.path.splitext(original_filename)[0]
                    new_filename = convert_to_jpeg(file, base, image_folder)
                    if new_filename is None:
                        flash(f"Error converting {original_filename} to JPEG.")
                    else:
                        if not ImageDB.query.filter_by(filename=new_filename).first():
                            db.session.add(ImageDB(filename=new_filename))
                            db.session.commit()
                            from tasks import process_image_tagging
                            process_image_tagging.delay(new_filename)
                else:
                    filepath = os.path.join(image_folder, original_filename)
                    file.save(filepath)
                    if not ImageDB.query.filter_by(filename=original_filename).first():
                        db.session.add(ImageDB(filename=original_filename))
                        db.session.commit()
                        from tasks import process_image_tagging
                        process_image_tagging.delay(original_filename)

        NotificationService.create_success("Images successfully uploaded")
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

    from models import Screenshot
    screenshots = Screenshot.query.all()
    screenshots_filenames = [s.filename for s in screenshots]

    last_sent = get_last_sent()
    return render_template('index.html', images=images, devices=devices, last_sent=last_sent, screenshots_filenames=screenshots_filenames)

@image_bp.route('/images/<filename>')
def uploaded_file(filename):
    image_folder = current_app.config['IMAGE_FOLDER']
    webp_folder = os.path.join(current_app.config['DATA_FOLDER'], 'webp_cache')

    if not os.path.exists(webp_folder):
        os.makedirs(webp_folder)

    filepath = os.path.join(image_folder, filename)
    if not os.path.exists(filepath):
        return "File not found", 404

    if request.args.get("size") == "info":
        try:
            with Image.open(filepath) as img:
                exif = None
                try:
                    exif = img._getexif()
                except (AttributeError, KeyError, IndexError):
                    pass
                if exif:
                    orientation_tag = 274
                    if orientation_tag in exif:
                        orientation = exif[orientation_tag]
                        if orientation == 2: img = img.transpose(Image.FLIP_LEFT_RIGHT)
                        elif orientation == 3: img = img.transpose(Image.ROTATE_180)
                        elif orientation == 4: img = img.transpose(Image.FLIP_TOP_BOTTOM)
                        elif orientation == 5: img = img.transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.ROTATE_90)
                        elif orientation == 6: img = img.transpose(Image.ROTATE_270)
                        elif orientation == 7: img = img.transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.ROTATE_270)
                        elif orientation == 8: img = img.transpose(Image.ROTATE_90)

                max_width = 300
                w, h = img.size
                if w > max_width:
                    ratio = max_width / float(w)
                    new_size = (max_width, int(h * ratio))
                    img = img.resize(new_size, Image.LANCZOS)

                if img.mode in ("RGBA", "P"):
                    img = img.convert("RGB")

                buf = BytesIO()
                img.save(buf, format="WEBP", quality=85)
                buf.seek(0)
                return send_file(buf, mimetype='image/webp')
        except Exception as e:
            current_app.logger.error("Error processing image %s for info: %s", filename, e)
            return "Error processing image", 500

    if request.args.get("for") == "gallery" or request.headers.get('Accept', '').find('image/webp') != -1:
        webp_path = os.path.join(webp_folder, os.path.splitext(filename)[0] + '.webp')

        if not os.path.exists(webp_path) or os.path.getmtime(webp_path) < os.path.getmtime(filepath):
            try:
                with Image.open(filepath) as img:
                    exif = None
                    try:
                        exif = img._getexif()
                    except (AttributeError, KeyError, IndexError):
                        pass
                    if exif:
                        orientation_tag = 274
                        if orientation_tag in exif:
                            orientation = exif[orientation_tag]
                            if orientation == 2: img = img.transpose(Image.FLIP_LEFT_RIGHT)
                            elif orientation == 3: img = img.transpose(Image.ROTATE_180)
                            elif orientation == 4: img = img.transpose(Image.FLIP_TOP_BOTTOM)
                            elif orientation == 5: img = img.transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.ROTATE_90)
                            elif orientation == 6: img = img.transpose(Image.ROTATE_270)
                            elif orientation == 7: img = img.transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.ROTATE_270)
                            elif orientation == 8: img = img.transpose(Image.ROTATE_90)

                    w, h = img.size
                    max_dimension = 800
                    if w > max_dimension or h > max_dimension:
                        if w > h:
                            ratio = max_dimension / float(w)
                            new_size = (max_dimension, int(h * ratio))
                        else:
                            ratio = max_dimension / float(h)
                            new_size = (int(w * ratio), max_dimension)
                        img = img.resize(new_size, Image.LANCZOS)

                    if img.mode in ("RGBA", "P"):
                        img = img.convert("RGB")

                    img.save(webp_path, "WEBP", quality=85)
                    current_app.logger.debug(f"Created WebP version for gallery: {webp_path}, size: {img.size}")
            except Exception as e:
                current_app.logger.error("Error creating WebP for %s: %s", filename, e)
                return send_from_directory(image_folder, filename)

        response = send_from_directory(webp_folder, os.path.basename(webp_path))
        response.headers['Cache-Control'] = 'public, max-age=31536000'
        return response

    ext = filename.rsplit('.', 1)[1].lower() if '.' in filename else ''
    if ext == "heic":
        try:
            with Image.open(filepath) as img:
                buf = BytesIO()
                img.save(buf, format="JPEG")
                buf.seek(0)
                return send_file(buf, mimetype='image/jpeg')
        except Exception as e:
            current_app.logger.error("Error processing HEIC image %s: %s", filename, e)
            return "Error processing image", 500

    return send_from_directory(image_folder, filename)

@image_bp.route('/save_crop_info/<filename>', methods=['POST'])
@notify_on_completion(
    message_start="Saving image crop settings...",
    message_success="Image crop settings saved successfully",
    message_error="Error saving image crop settings: {error}"
)
def save_crop_info_endpoint(filename):
    import json
    crop_data = request.get_json()
    current_app.logger.info(f"Received crop data for {filename}: {json.dumps(crop_data)}")

    if not crop_data:
        current_app.logger.error(f"No crop data provided for {filename}")
        NotificationService.create_error(f"No crop data provided for {filename}")
        return jsonify({"status": "error", "message": "No crop data provided"}), 400

    required_fields = ["x", "y", "width", "height"]
    for field in required_fields:
        if field not in crop_data or not isinstance(crop_data[field], (int, float)) or crop_data[field] < 0:
            error_msg = f"Invalid crop data for {filename}: {field} is missing or invalid"
            current_app.logger.error(error_msg)
            NotificationService.create_error(error_msg)
            return jsonify({"status": "error", "message": f"Invalid crop data: {field} is missing or invalid"}), 400

    device_addr = crop_data.get("device")
    if not device_addr:
        device_addr = "default_device"
        current_app.logger.warning(f"No device provided for crop data for {filename}, using default_device")
        crop_data["device"] = device_addr

    device_obj = Device.query.filter_by(address=device_addr).first()
    if device_obj and device_obj.resolution:
        crop_data["resolution"] = device_obj.resolution
        current_app.logger.info(f"Saving crop for {filename} with device: {device_addr}, resolution: {device_obj.resolution}")
    else:
        current_app.logger.warning(f"Device not found or missing resolution: {device_addr} for {filename}")

    existing = CropInfo.query.filter_by(filename=filename, device_address=device_addr).first()
    if existing:
        current_app.logger.info(f"Updating existing crop for {filename} and device {device_addr}.")
    else:
        current_app.logger.info(f"Creating new crop record for {filename} and device {device_addr}")

    try:
        save_crop_info_to_db(filename, crop_data)
        db.session.expire_all()
        saved_data = load_crop_info_from_db(filename, device_addr)
        if saved_data:
            current_app.logger.info(f"Verified crop data for {filename} with device {device_addr} was saved.")
            all_match = True
            for field in required_fields:
                if abs(saved_data[field] - crop_data[field]) > 0.01:
                    current_app.logger.warning(f"Mismatch in saved crop data for {filename} with device {device_addr}: {field}")
                    all_match = False
            if all_match:
                current_app.logger.info(f"All crop values for {filename} with device {device_addr} match.")
                device_name = device_obj.friendly_name if device_obj else "unknown device"
                NotificationService.create_success(f"Crop settings saved for {filename} for {device_name}")

            return jsonify({
                "status": "success",
                "message": "Crop info saved successfully",
                "updated_at": saved_data.get("updated_at"),
                "device_address": saved_data.get("device_address")
            }), 200
        else:
            error_msg = f"Failed to verify crop data was saved for {filename} with device {device_addr}"
            current_app.logger.error(error_msg)
            NotificationService.create_error(error_msg)
            return jsonify({"status": "error", "message": "Failed to verify crop data was saved"}), 500
    except Exception as e:
        current_app.logger.error(f"Error saving crop data for {filename} with device {device_addr}: {str(e)}")
        return jsonify({"status": "error", "message": f"Database error: {str(e)}"}), 500

@image_bp.route('/send_image/<filename>', methods=['POST'])
@image_bp.route('/send_image', methods=['POST'])
def send_image(filename=None): # Synchronous function
    current_app.logger.debug(f"[SEND_IMAGE_DEBUG] Entering send_image function. Filename (initial): {filename}")
    NotificationService.create_info("Image send process initiated...")

    if not filename:
        filename = request.form.get("filename")
        if not filename:
            current_app.logger.error("[GALLERY] No filename specified in request")
            NotificationService.create_error("No filename specified in image send request")
            return "No filename specified", 400
    image_folder = current_app.config['IMAGE_FOLDER']
    data_folder = current_app.config['DATA_FOLDER']

    current_app.logger.debug(f"[GALLERY] Send image request received for filename: {filename}")
    current_app.logger.debug(f"[SEND_IMAGE_DEBUG] Filename resolved to: {filename}")

    filepath = os.path.join(image_folder, filename)
    if not os.path.exists(filepath):
        current_app.logger.error(f"[GALLERY] File not found: {filepath}")
        return "File not found", 404

    device_addr = request.form.get("device")
    if not device_addr:
        current_app.logger.error("[GALLERY] No device specified in request")
        return "No device specified", 400

    current_app.logger.debug(f"[GALLERY] Sending to device: {device_addr}")
    current_app.logger.debug(f"[SEND_IMAGE_DEBUG] Device address from form: {device_addr}")

    device_obj = Device.query.filter_by(address=device_addr).first()
    if not device_obj:
        current_app.logger.error(f"[GALLERY] Device not found in DB: {device_addr}")
        return "Device not found in DB", 500
    current_app.logger.debug(f"[SEND_IMAGE_DEBUG] Found device object: ID={device_obj.id}, Name={device_obj.friendly_name}, Res={device_obj.resolution}, Orient={device_obj.orientation}")
    dev_width = None
    dev_height = None
    if device_obj.resolution:
        parts = device_obj.resolution.split("x")
        if len(parts) == 2:
            try:
                dev_width = int(parts[0])
                dev_height = int(parts[1])
            except ValueError:
                pass
    if not (dev_width and dev_height):
        current_app.logger.error(f"Target resolution invalid for device {device_addr}: {device_obj.resolution}")
        return "Target resolution not found or invalid", 500
    current_app.logger.debug(f"[SEND_IMAGE_DEBUG] Parsed device dimensions: Width={dev_width}, Height={dev_height}")

    temp_filename = None
    debug_copy = None
    unique_id = uuid.uuid4().hex[:8]
    temp_dir = os.path.join(data_folder, "temp")

    def delete_debug_copy():
        nonlocal debug_copy
        try:
            if debug_copy and os.path.exists(debug_copy):
                os.remove(debug_copy)
                current_app.logger.debug(f"Debug copy {debug_copy} deleted")
        except Exception as e:
            current_app.logger.error(f"Error deleting debug copy: {e}")

    # Outer try...except for image processing
    try:
        current_app.logger.debug(f"[SEND_IMAGE_DEBUG] Attempting to open image file: {filepath}")
        with Image.open(filepath) as orig_img:
            current_app.logger.debug(f"[SEND_IMAGE_DEBUG] Image opened successfully. Original size: {orig_img.size}, Mode: {orig_img.mode}")
            # --- Start Image Processing ---
            orig_w, orig_h = orig_img.size
            is_portrait = device_obj.orientation.lower() == 'portrait'

            if is_portrait:
                device_ratio = dev_height / dev_width if dev_width > 0 else 0
                target_width = dev_height
                target_height = dev_width
            else:
                device_ratio = dev_width / dev_height if dev_height > 0 else 0
                target_width = dev_width
                target_height = dev_height

            current_app.logger.debug(f"Original: {orig_w}x{orig_h}, Target: {target_width}x{target_height}, Ratio: {device_ratio:.4f}, Portrait: {is_portrait}")
            current_app.logger.debug(f"[SEND_IMAGE_DEBUG] Calculated target: {target_width}x{target_height}, Ratio: {device_ratio:.4f}, Portrait: {is_portrait}")

            db.session.expire_all()
            current_app.logger.debug(f"[SEND_IMAGE_DEBUG] Attempting to load crop info for device: {device_addr}")
            cdata = load_crop_info_from_db(filename, device_addr)
            if not cdata or not all(key in cdata for key in ["x", "y", "width", "height"]):
                 cdata = load_crop_info_from_db(filename) # Load default if device specific fails
                 current_app.logger.debug(f"[SEND_IMAGE_DEBUG] No device-specific crop info found or invalid, trying default crop info.")

            cropped = None
            current_app.logger.debug(f"[SEND_IMAGE_DEBUG] Crop data loaded: {cdata}")
            if cdata and all(key in cdata for key in ["x", "y", "width", "height"]):
                x, y, w, h = cdata.get("x", 0), cdata.get("y", 0), cdata.get("width", orig_w), cdata.get("height", orig_h)
                if x < 0 or y < 0 or w <= 0 or h <= 0 or x + w > orig_w or y + h > orig_h:
                    current_app.logger.warning(f"Invalid crop coordinates. Falling back to auto-crop.")
                    cdata = None
                else:
                    current_app.logger.info(f"Using crop data: x={x}, y={y}, w={w}, h={h}")
                    current_app.logger.debug(f"[SEND_IMAGE_DEBUG] Applying manual crop: Box=({x}, {y}, {x+w}, {y+h})")
                    cropped = orig_img.crop((x, y, x+w, y+h))
                    # Aspect ratio adjustment logic
                    crop_w, crop_h = cropped.size
                    crop_ratio = crop_w / crop_h if crop_h > 0 else 0
                    ratio_diff = abs(crop_ratio - device_ratio) / device_ratio if device_ratio > 0 else 0
                    if is_portrait:
                         portrait_crop_ratio = crop_h / crop_w if crop_w > 0 else 0
                         portrait_device_ratio = 1 / device_ratio if device_ratio > 0 else 0
                         ratio_diff = abs(portrait_crop_ratio - portrait_device_ratio) / portrait_device_ratio if portrait_device_ratio > 0 else 0
                    if ratio_diff > 0.01:
                         current_app.logger.debug("Adjusting crop aspect ratio.")
                         if crop_ratio > device_ratio:
                             new_crop_w = int(crop_h * device_ratio)
                             diff = crop_w - new_crop_w
                             new_x = x + (diff // 2)
                             new_crop_box = (new_x, y, new_x + new_crop_w, y + crop_h)
                         else:
                             new_crop_h = int(crop_w / device_ratio) if device_ratio > 0 else crop_h
                             diff = crop_h - new_crop_h
                             new_y = y + (diff // 2)
                             new_crop_box = (x, new_y, x + crop_w, new_y + new_crop_h)
                         cropped = orig_img.crop(new_crop_box)
                         current_app.logger.debug(f"[SEND_IMAGE_DEBUG] Adjusted crop box for aspect ratio: {new_crop_box}")

            if cropped is None: # Auto-crop if no valid cdata or crop failed
                current_app.logger.debug("[SEND_IMAGE_DEBUG] No valid manual crop data found or applied.")
                current_app.logger.debug("Using auto-centered crop.")
                orig_ratio = orig_w / orig_h if orig_h > 0 else 0
                if orig_ratio > device_ratio:
                    new_width = int(orig_h * device_ratio)
                    left = (orig_w - new_width) // 2
                    crop_box = (left, 0, left + new_width, orig_h)
                else:
                    new_height = int(orig_w / device_ratio) if device_ratio > 0 else orig_h
                    top = (orig_h - new_height) // 2
                    crop_box = (0, top, orig_w, top + new_height)
                current_app.logger.debug(f"[SEND_IMAGE_DEBUG] Applying auto-centered crop: Box={crop_box}")
                cropped = orig_img.crop(crop_box)

            current_app.logger.debug(f"[SEND_IMAGE_DEBUG] Image cropped. Size after crop: {cropped.size if cropped else 'None'}")
            # Resize and rotate
            if is_portrait:
                final_img = cropped.resize((target_width, target_height), Image.LANCZOS)
                final_img = final_img.rotate(-90, expand=True)
                expected_w, expected_h = target_height, target_width
                if abs(final_img.width - expected_w) > 5 or abs(final_img.height - expected_h) > 5:
                     final_img = final_img.resize((expected_w, expected_h), Image.LANCZOS)
            else:
                final_img = cropped.resize((target_width, target_height), Image.LANCZOS)
                if abs(final_img.width - target_width) > 5 or abs(final_img.height - target_height) > 5:
                     final_img = final_img.resize((target_width, target_height), Image.LANCZOS)
            current_app.logger.debug(f"[SEND_IMAGE_DEBUG] Image resized and rotated (if needed). Size before save: {final_img.size}")

            current_app.logger.debug(f"Final image size: {final_img.size}")

            # Save temporary file
            if not os.path.exists(temp_dir): os.makedirs(temp_dir)
            temp_filename = os.path.join(temp_dir, f"temp_{unique_id}_{filename}")
            current_app.logger.debug(f"[SEND_IMAGE_DEBUG] Attempting to save final image to temporary file: {temp_filename}")
            final_img.save(temp_filename, format="JPEG", quality=95)
            current_app.logger.debug(f"[SEND_IMAGE_DEBUG] Successfully saved temporary file: {temp_filename}")
            current_app.logger.debug(f"Saved temporary file: {temp_filename}")
            debug_copy = os.path.join(temp_dir, f"debug_{unique_id}_{filename}")
            shutil.copy2(temp_filename, debug_copy)
            current_app.logger.debug(f"Debug copy saved to: {debug_copy}")
            # --- End Image Processing ---

        # Verify the temporary file exists before dispatching task
        if not temp_filename or not os.path.exists(temp_filename):
             current_app.logger.error(f"Temporary file {temp_filename} not found after processing.")
             return jsonify({"status": "error", "message": "Error processing image: Temporary file not found."}), 500

        send_id = uuid.uuid4().hex[:8]
        current_app.logger.debug(f"[GALLERY-{send_id}] Preparing to dispatch Celery task for {filename} to {device_addr}")
        current_app.logger.debug(f"[SEND_IMAGE_DEBUG] Temporary file verified to exist: {temp_filename}")

        # Dispatch Celery task
        try:
            send_image_via_httpx_task.delay(
                temp_filepath=temp_filename,
                filename=filename,
                device_addr=device_addr, # Pass original address
                device_friendly_name=device_obj.friendly_name,
                send_id=send_id
            )
            current_app.logger.info(f"[GALLERY-{send_id}] Dispatched Celery task to send {filename} to {device_obj.friendly_name}")

            # Keep debug copy for Celery task, don't delete here
            # delete_debug_copy()

            # Return 202 Accepted
            return jsonify({"status": "accepted", "message": f"Image send task queued for {filename} to {device_obj.friendly_name}"}), 202

        except Exception as celery_err:
            # Handle errors dispatching the Celery task
            current_app.logger.error(f"[GALLERY-{send_id}] Error dispatching Celery task: {celery_err}")
            current_app.logger.error(traceback.format_exc())
            NotificationService.create_error(f"Failed to queue send task for {filename}")
            # Cleanup temp files if task dispatch fails
            if temp_filename and os.path.exists(temp_filename):
                try: os.remove(temp_filename)
                except Exception as e: current_app.logger.error(f"Error deleting temp file after Celery dispatch error: {e}")
            delete_debug_copy()
            return jsonify({"status": "error", "message": "Failed to queue image send task"}), 500

    # Outer except for image processing errors
    except Exception as e:
        current_app.logger.error("Error processing image before send: %s", e)
        current_app.logger.error(f"[SEND_IMAGE_DEBUG] Exception occurred during image processing: {e}")
        current_app.logger.error(traceback.format_exc())
        error_msg = f"Error processing image: {e}"
        NotificationService.create_error(error_msg)
        # Ensure cleanup happens even if image processing fails
        if temp_filename and os.path.exists(temp_filename):
             try:
                 os.remove(temp_filename)
             except Exception as del_e:
                 current_app.logger.error(f"Error deleting temp file during outer exception handling: {del_e}")
        delete_debug_copy()
        return jsonify({"status": "error", "message": error_msg}), 500

@image_bp.route('/api/get_current_image', methods=['GET'])
def get_current_image():
    """Get information about the currently displayed image on a device."""
    try:
        device_addr = request.args.get("device")
        if device_addr:
            device = Device.query.filter_by(address=device_addr).first()
            if not device:
                return jsonify({"status": "error", "message": f"Device not found: {device_addr}"}), 404
            filename = device.last_sent
            if not filename:
                return jsonify({"status": "info", "message": "No image sent yet"}), 200
            return jsonify({
                "status": "success",
                "current_image": {
                    "filename": filename,
                    "device": device.friendly_name,
                    "sent_at": datetime.datetime.now().isoformat(), # Placeholder
                    "display_url": f"/images/{filename}"
                }
            }), 200
        else:
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
            return jsonify({"status": "success", "devices": devices_info}), 200
    except Exception as e:
        current_app.logger.error(f"Error retrieving current image info: {e}")
        return jsonify({"status": "error", "message": f"Error: {str(e)}"}), 500

@image_bp.route('/api/get_crop_info/<filename>', methods=['GET'])
def get_crop_info(filename):
    """Get crop information for an image."""
    db.session.expire_all()
    device_address = request.args.get('device', None)
    image_folder = current_app.config['IMAGE_FOLDER']
    filepath = os.path.join(image_folder, filename)
    original_width, original_height = 0, 0
    try:
        with Image.open(filepath) as img:
            original_width, original_height = img.size
    except Exception as e:
        current_app.logger.error(f"Error getting image dimensions for {filename}: {e}")

    devices = []
    try:
        all_device_crops = CropInfo.query.filter_by(filename=filename).all()
        if all_device_crops:
            devices = [{'address': crop.device_address} for crop in all_device_crops]
    except Exception as e:
        current_app.logger.error(f"Error fetching device list for {filename}: {e}")

    query = CropInfo.query.filter_by(filename=filename)
    if device_address:
        crop_info = query.filter_by(device_address=device_address).first()
        if not crop_info:
            crop_info = query.filter_by(device_address='default_device').first()
    else:
        crop_info = query.first()

    if crop_info:
        current_app.logger.info(f"Returning crop info for {filename} device {crop_info.device_address}")
        return jsonify({
            "status": "success",
            "crop_info": {
                "x": crop_info.x, "y": crop_info.y, "width": crop_info.width, "height": crop_info.height,
                "resolution": crop_info.resolution, "device_address": crop_info.device_address,
                "updated_at": crop_info.updated_at.isoformat() if hasattr(crop_info, 'updated_at') and crop_info.updated_at else None
            },
            "available_devices": devices,
            "original_dimensions": {"width": original_width, "height": original_height}
        }), 200
    else:
        current_app.logger.info(f"No crop information found for {filename}")
        return jsonify({
            "status": "success", "message": "No crop information found", "crop_info": None,
            "available_devices": devices,
            "original_dimensions": {"width": original_width, "height": original_height}
        }), 200

@image_bp.route('/api/get_images', methods=['GET'])
def get_images():
    try:
        page = int(request.args.get('page', 1))
        per_page = int(request.args.get('per_page', 20))
        if page < 1 or per_page < 1 or per_page > 100:
            return jsonify({"status": "error", "message": "Invalid pagination parameters"}), 400
        offset = (page - 1) * per_page
        images_query = ImageDB.query.order_by(ImageDB.id.desc())
        total_images = images_query.count()
        images_page = images_query.offset(offset).limit(per_page).all()
        images_data = []
        for img in images_page:
            images_data.append({
                "filename": img.filename,
                "favorite": img.favorite if hasattr(img, 'favorite') else False,
                "tags": img.tags.split(',') if hasattr(img, 'tags') and img.tags else []
            })
        return jsonify({
            "status": "success", "total": total_images, "page": page, "per_page": per_page, "images": images_data
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
    webp_thumb_path = os.path.join(thumbnail_folder, os.path.splitext(filename)[0] + '.webp')
    webp_cache_path = os.path.join(current_app.config['DATA_FOLDER'], 'webp_cache', os.path.splitext(filename)[0] + '.webp')

    file_existed = os.path.exists(filepath)
    try:
        if file_existed: os.remove(filepath)
        if os.path.exists(thumb_path): os.remove(thumb_path)
        if os.path.exists(webp_thumb_path): os.remove(webp_thumb_path)
        if os.path.exists(webp_cache_path): os.remove(webp_cache_path)

        img_obj = ImageDB.query.filter_by(filename=filename).first()
        if img_obj:
            CropInfo.query.filter_by(filename=filename).delete()
            db.session.delete(img_obj)
            db.session.commit()
            message = "Image deleted" if file_existed else "Image DB entry removed (file was already missing)"
            return jsonify({"status": "success", "message": message}), 200
        elif file_existed: # File existed but no DB entry
             return jsonify({"status": "success", "message": "Image file deleted (no DB entry found)"}), 200
        else: # Neither file nor DB entry existed
             return jsonify({"status": "error", "message": "File not found"}), 404
    except Exception as e:
        current_app.logger.error("Error removing file %s: %s", filepath, e)
        return jsonify({"status": "error", "message": str(e)}), 500

@image_bp.route('/api/update_image_metadata', methods=['POST'])
def update_image_metadata():
    """Update tags and favorite status for an image."""
    data = request.get_json()
    filename = data.get('filename')
    tags = data.get('tags', [])
    favorite = data.get('favorite', False)

    if not filename:
        return jsonify({"status": "error", "message": "Filename is required"}), 400

    image = ImageDB.query.filter_by(filename=filename).first()
    if not image:
        return jsonify({"status": "error", "message": "Image not found"}), 404

    try:
        if isinstance(tags, list):
            image.tags = ",".join(tag.strip() for tag in tags if tag.strip())
        if isinstance(favorite, bool):
            image.favorite = favorite
        db.session.commit()
        return jsonify({"status": "success", "message": "Metadata updated"}), 200
    except Exception as e:
        db.session.rollback()
        current_app.logger.error(f"Error updating metadata for {filename}: {e}")
        return jsonify({"status": "error", "message": f"Database error: {str(e)}"}), 500
