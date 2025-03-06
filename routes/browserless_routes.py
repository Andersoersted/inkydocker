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
import asyncio
import pyppeteer
import logging

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
    
    # Create screenshots folder if it doesn't exist
    screenshots_folder = os.path.join(current_app.config['DATA_FOLDER'], 'screenshots')
    if not os.path.exists(screenshots_folder):
        os.makedirs(screenshots_folder)
    
    # Path for the screenshot file
    filepath = os.path.join(screenshots_folder, filename)
    
    # Use asyncio to run the pyppeteer code
    try:
        # Run the screenshot function in an asyncio event loop
        screenshot_data = asyncio.run(take_screenshot_with_puppeteer(
            url=data['url'],
            config=config,
            filepath=filepath
        ))
        
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

# Pyppeteer function to take screenshot using browserless
async def take_screenshot_with_puppeteer(url, config, filepath):
    current_app.logger.info(f"Connecting to browserless at ws://{config.address}:{config.port}")
    
    # Construct the WebSocket endpoint with token if available
    ws_endpoint = f"ws://{config.address}:{config.port}"
    if config.token:
        ws_endpoint += f"?token={config.token}"
    
    try:
        # Connect to browserless instance
        browser = await pyppeteer.connect(browserWSEndpoint=ws_endpoint)
        
        # Create a new page
        page = await browser.newPage()
        
        # Set viewport size
        await page.setViewport({'width': 1280, 'height': 900})
        
        # Extract domain for cookie handling
        domain = url.split('//')[-1].split('/')[0]
        if domain.startswith('www.'):
            domain = domain[4:]
        
        # Set common cookie consent cookies before navigation
        await set_consent_cookies(page, domain)
        
        # Navigate to the URL with a longer timeout
        current_app.logger.info(f"Navigating to {url}")
        await page.goto(url, {
            'waitUntil': 'networkidle2',
            'timeout': 60000  # 60 seconds timeout
        })
        
        # Execute cookie consent handling script
        current_app.logger.info("Executing cookie consent script")
        await page.evaluate('''
        () => {
            // Universal cookie acceptance function
            function handleCookieConsent() {
                console.log('Starting cookie consent handling...');
                
                // 1. Direct cookie manipulation
                try {
                    // Common cookie names
                    const cookieNames = [
                        'cookieConsent', 'cookie_consent', 'cookies_accepted', 'gdpr_consent',
                        'CookieConsent', 'cookieAccepted', 'cookies_policy', 'cookie_notice'
                    ];
                    
                    cookieNames.forEach(name => {
                        document.cookie = `${name}=true; path=/; max-age=31536000`;
                        document.cookie = `${name}=1; path=/; max-age=31536000`;
                    });
                } catch (e) {
                    console.log('Error setting cookies:', e);
                }
                
                // 2. Click cookie consent buttons
                
                // Helper function to click elements by text content
                function clickByText(text) {
                    const elements = Array.from(document.querySelectorAll('button, a, div[role="button"], [tabindex="0"]'));
                    for (const el of elements) {
                        if (!el || !el.offsetParent) continue; // Skip invisible elements
                        
                        const elText = (el.textContent || el.innerText || '').toLowerCase();
                        if (elText.includes(text)) {
                            console.log(`Clicking element with text: ${text}`);
                            el.click();
                            return true;
                        }
                    }
                    return false;
                }
                
                // Common selectors for cookie consent buttons
                const selectors = [
                    // ID-based selectors
                    '#accept-cookies', '#acceptCookies', '#cookie-accept', '#accept-all-cookies',
                    '#acceptAllCookies', '#cookies-accept-all', '#cookie-accept-all', '#gdpr-accept',
                    '#accept', '#accept_all', '#acceptAll', '#cookie_accept', '#cookie-consent-accept',
                    
                    // Class-based selectors
                    '.cookie-accept', '.accept-cookies', '.accept-all-cookies', '.acceptAllCookies',
                    '.cookie-consent-accept', '.cookie-banner__accept', '.cookie-notice__accept',
                    '.gdpr-accept', '.accept-button', '.cookie-accept-button', '.consent-accept',
                    
                    // Framework-specific selectors
                    '#CybotCookiebotDialogBodyLevelButtonLevelOptinAllowAll',
                    '#onetrust-accept-btn-handler',
                    '.cc-accept', '.cc-allow', '.cc-dismiss',
                    
                    // Attribute-based selectors
                    '[data-action="accept-cookies"]', '[data-role="accept-cookies"]',
                    '[data-consent="accept"]', '[data-cookie-accept="all"]',
                    '[aria-label*="accept cookies"]', '[aria-label*="Accept all"]'
                ];
                
                // Try all selectors
                selectors.forEach(selector => {
                    try {
                        const elements = document.querySelectorAll(selector);
                        elements.forEach(el => {
                            if (el && el.offsetParent !== null) {
                                console.log(`Clicking element with selector: ${selector}`);
                                el.click();
                            }
                        });
                    } catch (e) {
                        // Ignore errors for individual selectors
                    }
                });
                
                // Try clicking by common text patterns in multiple languages
                const textPatterns = [
                    // English
                    'accept', 'accept all', 'accept cookies', 'allow', 'allow all', 'i agree', 'ok', 'got it',
                    // Danish
                    'accepter', 'acceptér', 'tillad', 'tillad alle', 'ja tak', 'accepter alle',
                    // German
                    'akzeptieren', 'alle akzeptieren', 'zustimmen', 'einverstanden',
                    // French
                    'accepter', 'j\'accepte', 'tout accepter',
                    // Spanish
                    'aceptar', 'aceptar todo', 'permitir'
                ];
                
                textPatterns.forEach(pattern => clickByText(pattern));
                
                // 3. Try to handle iframes
                try {
                    const iframes = document.querySelectorAll('iframe');
                    iframes.forEach(iframe => {
                        try {
                            const iframeDoc = iframe.contentDocument || iframe.contentWindow.document;
                            const buttons = iframeDoc.querySelectorAll('button, a');
                            buttons.forEach(button => {
                                const text = (button.textContent || button.innerText || '').toLowerCase();
                                if (text.includes('accept') || text.includes('agree') || text.includes('allow')) {
                                    button.click();
                                }
                            });
                        } catch (e) {
                            // Cross-origin iframe access will fail - this is expected
                        }
                    });
                } catch (e) {
                    console.log('Error handling iframes:', e);
                }
                
                // 4. Try to remove cookie banners directly
                const bannerSelectors = [
                    '[class*="cookie-banner"]', '[id*="cookie-banner"]',
                    '[class*="cookie-consent"]', '[id*="cookie-consent"]',
                    '.cc-window', '.cc-banner', '#cookie-law-info-bar',
                    '#cookiebanner', '#cookieConsent', '#cookie-consent',
                    '#CybotCookiebotDialog', '#onetrust-banner-sdk'
                ];
                
                bannerSelectors.forEach(selector => {
                    try {
                        const elements = document.querySelectorAll(selector);
                        elements.forEach(el => {
                            if (el && el.offsetParent !== null) {
                                el.style.display = 'none';
                                el.style.visibility = 'hidden';
                                try { el.remove(); } catch (e) { /* ignore */ }
                            }
                        });
                    } catch (e) {
                        // Ignore errors
                    }
                });
            }
            
            // Run cookie handling
            handleCookieConsent();
            
            // Run again after delays to catch late-appearing banners
            setTimeout(handleCookieConsent, 1000);
            setTimeout(handleCookieConsent, 3000);
        }
        ''')
        
        # Wait for cookie banners to be handled
        current_app.logger.info("Waiting for cookie banners to be handled")
        await page.waitForTimeout(5000)
        
        # Check if any cookie banners are still visible and try again if needed
        banner_visible = await page.evaluate('''
        () => {
            const bannerSelectors = [
                '[class*="cookie"]', '[id*="cookie"]',
                '[class*="consent"]', '[id*="consent"]',
                '[class*="gdpr"]', '[id*="gdpr"]',
                '.cc-window', '.cc-banner'
            ];
            
            for (const selector of bannerSelectors) {
                const elements = document.querySelectorAll(selector);
                for (const el of elements) {
                    if (el && el.offsetParent !== null &&
                        el.getBoundingClientRect().height > 20 &&
                        window.getComputedStyle(el).display !== 'none') {
                        return true;
                    }
                }
            }
            return false;
        }
        ''')
        
        if banner_visible:
            current_app.logger.info("Cookie banners still visible, trying again")
            await page.evaluate('handleCookieConsent')
            await page.waitForTimeout(3000)
        
        # Take the screenshot
        current_app.logger.info(f"Taking screenshot and saving to {filepath}")
        await page.screenshot({'path': filepath, 'type': 'jpeg', 'quality': 90, 'fullPage': True})
        
        # Close the browser connection
        await browser.close()
        
        return True
        
    except Exception as e:
        current_app.logger.error(f"Error in pyppeteer: {str(e)}")
        raise e

# Helper function to set consent cookies
async def set_consent_cookies(page, domain):
    # Common cookie consent name patterns
    cookie_names = [
        "cookieConsent", "cookie_consent", "cookies_accepted", "cookies_consent",
        "gdpr_consent", "CookieConsent", "CybotCookiebotDialogConsent", "euconsent-v2"
    ]
    
    # Set cookies with different values
    for name in cookie_names:
        await page.setCookie({
            'name': name,
            'value': 'true',
            'domain': domain,
            'path': '/'
        })
        
        await page.setCookie({
            'name': name,
            'value': '1',
            'domain': domain,
            'path': '/'
        })
    
    # Set some specific framework cookies
    await page.setCookie({
        'name': 'CookieConsent',
        'value': 'stamp:-1|necessary:true|preferences:true|statistics:true|marketing:true|method:explicit|ver:1',
        'domain': domain,
        'path': '/'
    })
    
    await page.setCookie({
        'name': 'OptanonConsent',
        'value': 'isGpcEnabled=0&datestamp=Wed+Mar+06+2024+10%3A00%3A00+GMT%2B0100&version=202209.1.0&isIABGlobal=false&hosts=&consentId=47bcd4dd-f4c4-4b04-b78b-37f7e1484595&interactionCount=1&landingPath=NotLandingPage&groups=C0001%3A1%2CC0002%3A1%2CC0003%3A1%2CC0004%3A1',
        'domain': domain,
        'path': '/'
    })

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