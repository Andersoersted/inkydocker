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
            "quality": 90,
            "waitUntil": "networkidle2"  # Wait until network is idle
        },
        "cookies": [],  # Empty array to initialize cookies
        "gotoOptions": {
            "waitUntil": "networkidle2",
            "timeout": 60000  # Increased timeout to 60 seconds
        }
    }
    
    # Try to set common cookie consent cookies for the domain
    domain = data['url'].split('//')[-1].split('/')[0]
    if domain.startswith('www.'):
        domain = domain[4:]  # Remove www. prefix
    
    # Set common cookie consent cookies that work across many sites
    payload["cookies"] = [
        # General cookie consent cookies
        {"name": "cookieConsent", "value": "true", "domain": f".{domain}"},
        {"name": "cookie_consent", "value": "true", "domain": f".{domain}"},
        {"name": "cookies_accepted", "value": "true", "domain": f".{domain}"},
        {"name": "cookies_consent", "value": "true", "domain": f".{domain}"},
        {"name": "cookieConsent", "value": "1", "domain": f".{domain}"},
        {"name": "cookie_consent", "value": "1", "domain": f".{domain}"},
        {"name": "gdpr_consent", "value": "true", "domain": f".{domain}"},
        {"name": "gdpr_consent", "value": "1", "domain": f".{domain}"},
        {"name": "cookieConsentDate", "value": datetime.now().strftime("%Y-%m-%d"), "domain": f".{domain}"},
        
        # Common cookie banner framework cookies
        {"name": "CookieConsent", "value": "{stamp:%27-1%27%2Cnecessary:true%2Cpreferences:true%2Cstatistics:true%2Cmarketing:true%2Cmethod:%27explicit%27%2Cver:1}", "domain": f".{domain}"},
        {"name": "euconsent-v2", "value": "CPwqAEAPwqAEAAHABBENDECsAP_AAH_AAAAAJNNf_X__b3_j-_5_f_t0eY1P9_7_v-0zjhfdt-8N3f_X_L8X42M7vF36pq4KuR4Eu3LBIQdlHOHcTUmw6okVrzPsbk2cr7NKJ7PEmnMbO2dYGH9_n93TuZKY7__8___z_v-v_v____f_7-3_3__5_X---_e_V399zLv9____39nP___9v-_9_____4IhgEmGpeQBdiWODJtGlUKIEYVhIdAKACigGFoisIHVwU7K4CPUEDABAagIwIgQYgoxYBAAIBAEhEQEgB4IBEARAIAAQAqwEIACNgEFgBYGAQACgGhYgRQBCBIQZHBUcpgQFSLRQT2ViCUHexphCGWeBFAo_oqEBGs0QLAyEhYOY4AkBLxZIHmKF8gAAAAA.YAAAAAAAAAAA", "domain": f".{domain}"},
        {"name": "OptanonConsent", "value": "isGpcEnabled=0&datestamp=Wed+Mar+06+2024+09%3A45%3A00+GMT%2B0100+(Central+European+Standard+Time)&version=202209.1.0&isIABGlobal=false&hosts=&consentId=47bcd4dd-f4c4-4b04-b78b-37f7e1484595&interactionCount=1&landingPath=NotLandingPage&groups=C0001%3A1%2CC0002%3A1%2CC0003%3A1%2CC0004%3A1&geolocation=DK%3B84&AwaitingReconsent=false", "domain": f".{domain}"},
        
        # Site-specific cookies for common Danish sites
        {"name": "CybotCookiebotDialogConsent", "value": "00000000000000000000000000000000", "domain": f".{domain}"},
        {"name": "cb_user_accepted_cookies", "value": "1", "domain": f".{domain}"},
        {"name": "cookie_consent_level", "value": "all", "domain": f".{domain}"}
    ]
    
    # Add a comprehensive cookie handling script
    payload["addScriptTag"] = [{
        "content": """
            // Universal cookie acceptance function
            function handleCookieConsent() {
                console.log('Starting cookie consent handling...');
                
                // 1. Direct cookie manipulation - set common cookie consent flags
                try {
                    // Common cookie names used for consent
                    const commonCookieNames = [
                        'cookieConsent', 'cookie_consent', 'cookies_accepted', 'gdpr_consent',
                        'CookieConsent', 'cookieAccepted', 'cookies_policy', 'cookie_notice',
                        'cookie-notice-dismissed', 'allowCookies', 'acceptCookies'
                    ];
                    
                    // Set all these cookies to accepted values
                    commonCookieNames.forEach(name => {
                        document.cookie = `${name}=true; path=/; max-age=31536000`;
                        document.cookie = `${name}=1; path=/; max-age=31536000`;
                        document.cookie = `${name}=accepted; path=/; max-age=31536000`;
                    });
                    
                    console.log('Set consent cookies directly');
                } catch (e) {
                    console.log('Error setting cookies directly:', e);
                }
                
                // 2. Try to find and click cookie consent buttons
                
                // Common cookie banner frameworks and their selectors
                const frameworkSelectors = {
                    // CookieBot
                    'CookieBot': ['#CybotCookiebotDialogBodyLevelButtonLevelOptinAllowAll',
                                 '.CybotCookiebotDialogBodyButton',
                                 '[id*="CybotCookiebot"][id*="Button"][id*="Accept"]'],
                    // OneTrust
                    'OneTrust': ['#onetrust-accept-btn-handler',
                                '.onetrust-close-btn-handler',
                                '.ot-sdk-button[aria-label*="Accept"]'],
                    // Cookie-Script
                    'CookieScript': ['.cookiescript_accept',
                                    '#cookiescript_accept',
                                    '[data-cs-accept-cookies]'],
                    // Cookielaw.org
                    'Cookielaw': ['.optanon-allow-all',
                                 '#optanon-popup-bottom .accept-cookies-button'],
                    // Cookie Notice
                    'CookieNotice': ['.cn-set-cookie',
                                    '.cn-accept-cookie',
                                    '#cn-accept-cookie'],
                    // GDPR Cookie Consent
                    'GDPRCookieConsent': ['.cli_action_button[data-cli_action="accept"]',
                                         '#cookie_action_close_header',
                                         '.cli-accept-all-btn'],
                    // Cookiefirst
                    'Cookiefirst': ['.cookiefirst-accept-all',
                                   '[data-cookiefirst-action="accept"]'],
                    // Complianz
                    'Complianz': ['.cmplz-accept',
                                 '.cc-accept-all',
                                 '.cmplz-btn.cmplz-accept'],
                    // Borlabs Cookie
                    'Borlabs': ['#CookieBoxSaveButton',
                               '.borlabs-cookie-accept',
                               '#borlabs-cookie-btn-accept-all'],
                    // Termly
                    'Termly': ['.termly-styles-btn-accept',
                              '[data-role="accept-button"]'],
                    // Osano
                    'Osano': ['.osano-cm-accept-all',
                             '.osano-cm-button--type_accept'],
                    // Quantcast
                    'Quantcast': ['.qc-cmp2-summary-buttons button:last-child',
                                 '[aria-label*="AGREE"]',
                                 '.qc-cmp-button']
                };
                
                // Try all framework selectors
                for (const [framework, selectors] of Object.entries(frameworkSelectors)) {
                    selectors.forEach(selector => {
                        try {
                            const elements = document.querySelectorAll(selector);
                            if (elements.length > 0) {
                                console.log(`Found ${framework} consent elements with selector: ${selector}`);
                                elements.forEach(el => {
                                    if (el && el.offsetParent !== null) {
                                        console.log(`Clicking ${framework} element:`, selector);
                                        el.click();
                                    }
                                });
                            }
                        } catch (e) {
                            console.log(`Error with ${framework} selector:`, e);
                        }
                    });
                }
                
                // 3. Generic selectors based on common patterns
                const genericSelectors = [
                    // ID-based selectors
                    '#accept-cookies', '#acceptCookies', '#cookie-accept', '#accept-all-cookies',
                    '#acceptAllCookies', '#cookies-accept-all', '#cookie-accept-all', '#gdpr-accept',
                    '#accept', '#accept_all', '#acceptAll', '#cookie_accept', '#cookie-consent-accept',
                    
                    // Class-based selectors
                    '.cookie-accept', '.accept-cookies', '.accept-all-cookies', '.acceptAllCookies',
                    '.cookie-consent-accept', '.cookie-banner__accept', '.cookie-notice__accept',
                    '.gdpr-accept', '.accept-button', '.cookie-accept-button', '.consent-accept',
                    
                    // Attribute-based selectors
                    '[data-action="accept-cookies"]', '[data-role="accept-cookies"]',
                    '[data-consent="accept"]', '[data-cookie-accept="all"]',
                    '[aria-label*="accept cookies"]', '[aria-label*="Accept all"]',
                    '[title*="Accept cookies"]', '[title*="accept all"]'
                ];
                
                genericSelectors.forEach(selector => {
                    try {
                        const elements = document.querySelectorAll(selector);
                        if (elements.length > 0) {
                            console.log(`Found generic consent elements with selector: ${selector}`);
                            elements.forEach(el => {
                                if (el && el.offsetParent !== null) {
                                    console.log('Clicking generic element:', selector);
                                    el.click();
                                }
                            });
                        }
                    } catch (e) {
                        console.log('Error with generic selector:', e);
                    }
                });
                
                // 4. Text-based approach for buttons and links
                const textPatterns = {
                    // Common accept text patterns in various languages
                    acceptPatterns: [
                        'accept', 'accept all', 'accept cookies', 'allow', 'allow all', 'allow cookies',
                        'agree', 'agree to all', 'i agree', 'consent', 'i consent', 'ok', 'okay',
                        'got it', 'understood', 'continue', 'proceed', 'save', 'confirm',
                        // Danish
                        'accepter', 'acceptér', 'tillad', 'tillad alle', 'ja tak', 'accepter alle',
                        // German
                        'akzeptieren', 'alle akzeptieren', 'zustimmen', 'einverstanden',
                        // French
                        'accepter', 'j\'accepte', 'tout accepter', 'autoriser',
                        // Spanish
                        'aceptar', 'aceptar todo', 'permitir', 'estoy de acuerdo',
                        // Italian
                        'accetto', 'accetta tutto', 'consento', 'va bene'
                    ],
                    // Common cookie-related text patterns
                    cookiePatterns: [
                        'cookie', 'cookies', 'gdpr', 'privacy', 'consent', 'data',
                        // Danish
                        'samtykke', 'privat', 'privatlivspolitik',
                        // German
                        'datenschutz', 'einwilligung',
                        // French
                        'confidentialité', 'consentement',
                        // Spanish
                        'privacidad', 'consentimiento',
                        // Italian
                        'privacy', 'consenso'
                    ]
                };
                
                // Find all clickable elements
                const clickableElements = document.querySelectorAll('button, a, div[role="button"], span[role="button"], [tabindex="0"]');
                
                clickableElements.forEach(el => {
                    if (el && el.offsetParent !== null) { // Check if visible
                        const text = (el.textContent || el.innerText || '').toLowerCase();
                        
                        // Check if element contains both accept and cookie related text
                        const hasAcceptText = textPatterns.acceptPatterns.some(pattern => text.includes(pattern));
                        const hasCookieText = textPatterns.cookiePatterns.some(pattern => text.includes(pattern));
                        
                        if (hasAcceptText && hasCookieText) {
                            console.log('Clicking text-matched element:', text);
                            el.click();
                        } else if (hasAcceptText && (el.id || el.className || '').toLowerCase().includes('cookie')) {
                            // Also click if it has accept text and cookie in id/class
                            console.log('Clicking element with accept text and cookie in id/class:', text);
                            el.click();
                        }
                    }
                });
                
                // 5. Try to handle iframes that might contain cookie banners
                try {
                    const iframes = document.querySelectorAll('iframe');
                    iframes.forEach(iframe => {
                        // Check if iframe might be a cookie banner
                        const src = iframe.src || '';
                        const id = iframe.id || '';
                        const className = iframe.className || '';
                        
                        if (src.includes('cookie') || id.includes('cookie') || className.includes('cookie') ||
                            src.includes('consent') || id.includes('consent') || className.includes('consent') ||
                            src.includes('privacy') || id.includes('privacy') || className.includes('privacy')) {
                            
                            console.log('Found potential cookie iframe:', src || id || className);
                            
                            try {
                                // Try to access iframe content (will fail for cross-origin)
                                const iframeDoc = iframe.contentDocument || iframe.contentWindow.document;
                                
                                // Look for accept buttons in the iframe
                                const acceptButtons = iframeDoc.querySelectorAll('button, a');
                                acceptButtons.forEach(button => {
                                    const text = (button.textContent || button.innerText || '').toLowerCase();
                                    if (textPatterns.acceptPatterns.some(pattern => text.includes(pattern))) {
                                        console.log('Clicking iframe button:', text);
                                        button.click();
                                    }
                                });
                            } catch (e) {
                                // Cross-origin iframe access will fail - this is expected
                                console.log('Could not access iframe content (likely cross-origin)');
                            }
                        }
                    });
                } catch (e) {
                    console.log('Error handling iframes:', e);
                }
                
                // 6. Try to handle shadow DOM elements
                try {
                    function processShadowDOM(root) {
                        // Get all elements with shadow roots
                        const elementsWithShadowRoots = Array.from(root.querySelectorAll('*'))
                            .filter(el => el.shadowRoot);
                        
                        elementsWithShadowRoots.forEach(el => {
                            console.log('Processing shadow DOM element:', el.tagName);
                            
                            // Process buttons in this shadow root
                            const shadowButtons = el.shadowRoot.querySelectorAll('button, a');
                            shadowButtons.forEach(button => {
                                const text = (button.textContent || button.innerText || '').toLowerCase();
                                if (textPatterns.acceptPatterns.some(pattern => text.includes(pattern)) &&
                                    textPatterns.cookiePatterns.some(pattern => text.includes(pattern))) {
                                    console.log('Clicking shadow DOM button:', text);
                                    button.click();
                                }
                            });
                            
                            // Recursively process nested shadow roots
                            processShadowDOM(el.shadowRoot);
                        });
                    }
                    
                    // Start processing from document root
                    processShadowDOM(document);
                } catch (e) {
                    console.log('Error handling shadow DOM:', e);
                }
                
                console.log('Completed cookie consent handling');
                window.cookiesHandled = true;
            }
            
            // Run cookie handling immediately
            handleCookieConsent();
            
            // Run again after a delay to catch banners that appear later
            setTimeout(handleCookieConsent, 2000);
            setTimeout(handleCookieConsent, 5000);
        """
    }]
    
    # Add a delay script to ensure we wait after cookie handling
    payload["addScriptTag"].append({
        "content": """
            // Add a delay to ensure cookie banners are handled
            function waitForCookieHandling() {
                return new Promise(resolve => {
                    // Check if cookies have been handled
                    if (window.cookiesHandled) {
                        console.log('Cookies already handled, continuing...');
                        // Even if cookies are handled, wait a bit to ensure banners are gone
                        setTimeout(() => {
                            console.log('Extra wait after cookie handling complete...');
                            resolve();
                        }, 5000); // Wait 5 more seconds after handling
                        return;
                    }
                    
                    // Wait for cookie handling to complete
                    console.log('Waiting for cookie handling...');
                    setTimeout(() => {
                        console.log('Wait complete, continuing...');
                        resolve();
                    }, 12000); // Wait 12 seconds for cookie handling
                });
            }
            
            // Function to check if page has cookie banners still visible
            function hasCookieBanners() {
                // Common cookie banner selectors
                const bannerSelectors = [
                    '[class*="cookie"]', '[id*="cookie"]',
                    '[class*="consent"]', '[id*="consent"]',
                    '[class*="gdpr"]', '[id*="gdpr"]',
                    '[class*="privacy"]', '[id*="privacy"]',
                    '.cc-window', '.cc-banner', '.cookie-banner',
                    '#cookie-notice', '#cookie-banner', '#cookie-law-info-bar',
                    '#cookiebanner', '#cookieConsent', '#cookie-consent'
                ];
                
                // Check if any banner is visible
                for (const selector of bannerSelectors) {
                    const elements = document.querySelectorAll(selector);
                    for (const el of elements) {
                        if (el && el.offsetParent !== null &&
                            el.getBoundingClientRect().height > 20 && // Must have some height
                            window.getComputedStyle(el).display !== 'none' &&
                            window.getComputedStyle(el).visibility !== 'hidden') {
                            console.log('Found visible cookie banner:', selector);
                            return true;
                        }
                    }
                }
                
                return false;
            }
            
            // This will be awaited by browserless before taking the screenshot
            window.waitForScreenshot = async () => {
                // First wait for cookie handling
                await waitForCookieHandling();
                
                // Then check if any banners are still visible
                if (hasCookieBanners()) {
                    console.log('Cookie banners still visible, trying to handle again...');
                    // Try one more time to handle cookies
                    if (typeof handleCookieConsent === 'function') {
                        handleCookieConsent();
                    }
                    
                    // Wait a bit more
                    await new Promise(resolve => setTimeout(resolve, 5000));
                }
                
                return true;
            };
        """
    })
    
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