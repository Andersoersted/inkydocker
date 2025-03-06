from flask import Blueprint, request, render_template, jsonify, current_app, abort, send_from_directory
from models import db, BrowserlessConfig, Screenshot, ScreenshotCropInfo, Device
import os
from datetime import datetime
import requests
from PIL import Image
from io import BytesIO
from utils.crop_helpers import load_crop_info_from_db, save_crop_info_to_db, add_send_log_entry
import subprocess
import base64

browserless_bp = Blueprint('browserless', __name__)

# Create screenshots folder when the blueprint is registered
@browserless_bp.record_once
def setup_screenshots_folder(state):
    app = state.app
    with app.app_context():
        screenshots_folder = os.path.join(app.config['DATA_FOLDER'], 'screenshots')
        if not os.path.exists(screenshots_folder):
            os.makedirs(screenshots_folder)

@browserless_bp.route('/browserless', methods=['GET'])
def browserless_page():
    # Get browserless config
    config = BrowserlessConfig.query.filter_by(active=True).first()
    
    # Get all screenshots
    screenshots = Screenshot.query.all()
    
    # Get all devices
    devices = Device.query.all()
    
    return render_template('browserless.html', 
                          config=config, 
                          screenshots=screenshots, 
                          devices=devices)

@browserless_bp.route('/api/browserless/config', methods=['POST'])
def save_browserless_config():
    data = request.get_json()
    
    if not data or 'address' not in data or 'port' not in data:
        return jsonify({"status": "error", "message": "Missing required fields"}), 400
    
    # Deactivate all existing configs
    BrowserlessConfig.query.update({BrowserlessConfig.active: False})
    
    # Create new config
    config = BrowserlessConfig(
        address=data['address'],
        port=data['port'],
        token=data.get('token', ''),  # Token is optional but recommended
        active=True
    )
    
    db.session.add(config)
    db.session.commit()
    
    return jsonify({"status": "success", "message": "Configuration saved successfully"}), 200

@browserless_bp.route('/api/browserless/screenshot', methods=['POST'])
def take_screenshot():
    data = request.get_json()
    
    if not data or 'url' not in data or 'name' not in data:
        return jsonify({"status": "error", "message": "Missing required fields"}), 400
    
    # Get active browserless config
    config = BrowserlessConfig.query.filter_by(active=True).first()
    if not config:
        return jsonify({"status": "error", "message": "No active browserless configuration found"}), 400
    
    # Generate filename
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    filename = f"screenshot_{timestamp}.jpg"
    
    # Prepare the browserless API URL
    browserless_url = f"http://{config.address}:{config.port}/screenshot"
    
    # Prepare the payload for browserless
    payload = {
        "url": data['url'],
        "options": {
            "fullPage": True,
            "type": "jpeg",
            "quality": 90
        },
        "cookies": [],  # Empty array to initialize cookies
        "gotoOptions": {
            "waitUntil": "networkidle2",
            "timeout": 30000
        },
        "stealth": False,
        "addScriptTag": [{
            "content": """
                // Auto-accept cookies by clicking common cookie consent buttons
                function autoAcceptCookies() {
                    // Common cookie accept button selectors
                    const selectors = [
                        'button[id*="accept"], button[class*="accept"]',
                        'button[id*="cookie"], button[class*="cookie"]',
                        'a[id*="accept"], a[class*="accept"]',
                        'a[id*="cookie"], a[class*="cookie"]',
                        'button:contains("Accept"), button:contains("Accept All")',
                        'button:contains("I agree"), button:contains("Agree")',
                        'button:contains("OK"), button:contains("Ok")',
                        'button:contains("Got it")',
                        'button:contains("Allow")',
                        'button:contains("Close")',
                        '.cc-accept, .cc-allow, .cc-dismiss',
                        '#cookie-notice .button, #cookie-policy .button',
                        '.cookie-banner button, .cookie-notice button'
                    ];
                    
                    // Try each selector
                    selectors.forEach(selector => {
                        try {
                            const buttons = document.querySelectorAll(selector);
                            buttons.forEach(button => {
                                if (button.offsetParent !== null) { // Check if visible
                                    button.click();
                                    console.log('Clicked cookie button:', selector);
                                }
                            });
                        } catch (e) {
                            // Ignore errors for individual selectors
                        }
                    });
                }
                
                // Run immediately and after a delay to catch late-appearing banners
                autoAcceptCookies();
                setTimeout(autoAcceptCookies, 2000);
                setTimeout(autoAcceptCookies, 5000);
            """
        }]
    }
    
    # Prepare headers with token if available
    headers = {}
    if config.token:
        headers['Authorization'] = f'Bearer {config.token}'
    
    try:
        # Make request to browserless
        response = requests.post(browserless_url, json=payload, headers=headers)
        
        if response.status_code != 200:
            return jsonify({
                "status": "error", 
                "message": f"Browserless API error: {response.status_code} - {response.text}"
            }), 500
        
        # Save the screenshot to file
        screenshots_folder = os.path.join(current_app.config['DATA_FOLDER'], 'screenshots')
        filepath = os.path.join(screenshots_folder, filename)
        
        with open(filepath, 'wb') as f:
            f.write(response.content)
        
        # Create or update screenshot record
        existing_screenshot = Screenshot.query.filter_by(name=data['name']).first()
        
        if existing_screenshot:
            # Delete old file if it exists and is different from the new one
            if existing_screenshot.filename != filename:
                old_filepath = os.path.join(screenshots_folder, existing_screenshot.filename)
                if os.path.exists(old_filepath):
                    os.remove(old_filepath)
            
            # Update existing record
            existing_screenshot.url = data['url']
            existing_screenshot.filename = filename
            existing_screenshot.last_updated = datetime.utcnow()
        else:
            # Create new record
            screenshot = Screenshot(
                name=data['name'],
                url=data['url'],
                filename=filename
            )
            db.session.add(screenshot)
        
        db.session.commit()
        
        return jsonify({
            "status": "success", 
            "message": "Screenshot taken successfully",
            "filename": filename
        }), 200
        
    except Exception as e:
        current_app.logger.error(f"Error taking screenshot: {str(e)}")
        return jsonify({"status": "error", "message": f"Error: {str(e)}"}), 500

@browserless_bp.route('/screenshots/<filename>')
def get_screenshot(filename):
    screenshots_folder = os.path.join(current_app.config['DATA_FOLDER'], 'screenshots')
    return send_from_directory(screenshots_folder, filename)

@browserless_bp.route('/api/browserless/delete/<int:screenshot_id>', methods=['POST'])
def delete_screenshot(screenshot_id):
    screenshot = Screenshot.query.get_or_404(screenshot_id)
    
    # Delete the file
    screenshots_folder = os.path.join(current_app.config['DATA_FOLDER'], 'screenshots')
    filepath = os.path.join(screenshots_folder, screenshot.filename)
    
    if os.path.exists(filepath):
        os.remove(filepath)
    
    # Delete crop info if exists
    crop_info = ScreenshotCropInfo.query.filter_by(filename=screenshot.filename).first()
    if crop_info:
        db.session.delete(crop_info)
    
    # Delete the database record
    db.session.delete(screenshot)
    db.session.commit()
    
    return jsonify({"status": "success", "message": "Screenshot deleted successfully"}), 200

@browserless_bp.route('/save_screenshot_crop_info/<filename>', methods=['POST'])
def save_screenshot_crop_info(filename):
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
    
    # Save crop info
    crop_info = ScreenshotCropInfo.query.filter_by(filename=filename).first()
    if not crop_info:
        crop_info = ScreenshotCropInfo(filename=filename)
        db.session.add(crop_info)
    
    crop_info.x = crop_data.get("x", 0)
    crop_info.y = crop_data.get("y", 0)
    crop_info.width = crop_data.get("width", 0)
    crop_info.height = crop_data.get("height", 0)
    if "resolution" in crop_data:
        crop_info.resolution = crop_data.get("resolution")
    
    db.session.commit()
    
    return jsonify({"status": "success"}), 200

@browserless_bp.route('/send_screenshot/<filename>', methods=['POST'])
def send_screenshot(filename):
    screenshots_folder = os.path.join(current_app.config['DATA_FOLDER'], 'screenshots')
    filepath = os.path.join(screenshots_folder, filename)
    
    if not os.path.exists(filepath):
        return jsonify({"status": "error", "message": "File not found"}), 404
    
    device_addr = request.form.get("device")
    if not device_addr:
        return jsonify({"status": "error", "message": "No device specified"}), 400

    device_obj = Device.query.filter_by(address=device_addr).first()
    if not device_obj:
        return jsonify({"status": "error", "message": "Device not found in DB"}), 500
    
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
        return jsonify({"status": "error", "message": "Target resolution not found"}), 500

    try:
        with Image.open(filepath) as orig_img:
            orig_w, orig_h = orig_img.size
            
            # Check if device is in portrait orientation
            is_portrait = device_obj.orientation.lower() == 'portrait'
            
            # If portrait, swap width and height for target ratio calculation
            if is_portrait:
                target_ratio = dev_height / dev_width
            else:
                target_ratio = dev_width / dev_height
                
            # Log the original image dimensions and target ratio
            current_app.logger.info(f"Original image dimensions: {orig_w}x{orig_h}, target ratio: {target_ratio}")
            
            # Step 1: Apply crop if available
            crop_info = ScreenshotCropInfo.query.filter_by(filename=filename).first()
            cdata = None
            
            if crop_info:
                cdata = {
                    "x": crop_info.x,
                    "y": crop_info.y,
                    "width": crop_info.width,
                    "height": crop_info.height,
                    "resolution": crop_info.resolution
                }
            
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
                        # If resolutions don't match, we need to recalculate the crop
                        # to maintain the correct aspect ratio
                        current_app.logger.info(f"Recalculating crop to match target ratio")
                        cropped = orig_img.crop((x, y, x+w, y+h))
                        crop_w, crop_h = cropped.size
                        crop_ratio = crop_w / crop_h
                        
                        # Adjust the crop to match the target ratio
                        if crop_ratio > target_ratio:
                            new_width = int(crop_h * target_ratio)
                            left = (crop_w - new_width) // 2
                            crop_box = (left, 0, left + new_width, crop_h)
                        else:
                            new_height = int(crop_w / target_ratio)
                            top = (crop_h - new_height) // 2
                            crop_box = (0, top, crop_w, top + new_height)
                        cropped = cropped.crop(crop_box)
            
            # If no valid crop data, create an auto-centered crop with the correct aspect ratio
            if not cdata or "x" not in cdata:
                current_app.logger.info(f"No crop data found, using auto-centered crop")
                orig_ratio = orig_w / orig_h
                
                if orig_ratio > target_ratio:
                    # Image is wider than target ratio, use full height
                    new_width = int(orig_h * target_ratio)
                    left = (orig_w - new_width) // 2
                    crop_box = (left, 0, left + new_width, orig_h)
                else:
                    # Image is taller than target ratio, use full width
                    new_height = int(orig_w / target_ratio)
                    top = (orig_h - new_height) // 2
                    crop_box = (0, top, orig_w, top + new_height)
                
                current_app.logger.info(f"Auto crop box: {crop_box}")
                cropped = orig_img.crop(crop_box)

            # Step 2: Resize the cropped image to match the target resolution
            current_app.logger.info(f"Cropped image size: {cropped.size}")
            
            # If portrait, rotate the image 90 degrees clockwise and swap dimensions
            if is_portrait:
                current_app.logger.info("Rotating image for portrait orientation")
                cropped = cropped.rotate(-90, expand=True)  # -90 for clockwise rotation
                current_app.logger.info(f"After rotation size: {cropped.size}")
                final_img = cropped.resize((dev_height, dev_width), Image.LANCZOS)  # Note swapped dimensions
            else:
                final_img = cropped.resize((dev_width, dev_height), Image.LANCZOS)
            
            current_app.logger.info(f"Final image size: {final_img.size}")
            
            # Save the processed image as a temporary file
            temp_dir = os.path.join(current_app.config['DATA_FOLDER'], "temp")
            if not os.path.exists(temp_dir):
                os.makedirs(temp_dir)
            temp_filename = os.path.join(temp_dir, f"temp_{filename}")
            final_img.save(temp_filename, format="JPEG", quality=95)
            current_app.logger.info(f"Saved temporary file: {temp_filename}")

        cmd = f'curl "{device_addr}/send_image" -X POST -F "file=@{temp_filename}"'
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        os.remove(temp_filename)
        if result.returncode != 0:
            return jsonify({"status": "error", "message": f"Error sending image: {result.stderr}"}), 500

        device_obj.last_sent = filename
        db.session.commit()
        add_send_log_entry(filename)
        return jsonify({"status": "success", "message": "Screenshot sent successfully"}), 200
    except Exception as e:
        current_app.logger.error(f"Error processing screenshot: {str(e)}")
        return jsonify({"status": "error", "message": f"Error processing screenshot: {str(e)}"}), 500