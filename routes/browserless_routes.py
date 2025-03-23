from flask import Blueprint, request, render_template, jsonify, current_app, abort, send_from_directory
import sqlalchemy as sa
from models import db, BrowserlessConfig, Screenshot, ScreenshotCropInfo, Device
import os
from datetime import datetime
import requests
from PIL import Image
from io import BytesIO
import subprocess
import httpx
import base64
import asyncio
import pyppeteer
import logging
import torch
import open_clip
from PIL import Image
import io

# Function to log when images are sent to devices
def add_send_log_entry(filename):
    """
    Add a log entry when an image is sent to a device.
    This is a simplified version that just logs to the application logger.
    """
    try:
        current_app.logger.info(f"Image sent: {filename} at {datetime.utcnow()}")
        # In a future implementation, this could write to a database table
    except Exception as e:
        current_app.logger.error(f"Failed to add send log entry: {str(e)}")
        # Do not raise the exception to avoid breaking the main flow

# Create blueprint
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
        screenshot_taken = asyncio.run(take_screenshot_with_puppeteer(
            url=data['url'],
            config=config,
            filepath=filepath
        ))
        
        if not screenshot_taken:
            return jsonify({"status": "error", "message": "Failed to take screenshot"}), 500
            
        # Create or update screenshot record
        existing_screenshot = Screenshot.query.filter_by(name=data['name']).first()
        
        if existing_screenshot:
            # Delete old file if it exists and is different from the new one
            if existing_screenshot.filename != filename:
                old_filepath = os.path.join(screenshots_folder, existing_screenshot.filename)
                if os.path.exists(old_filepath):
                    os.remove(old_filepath)
            
            # Save the old filename to copy crop info if needed
            old_filename = existing_screenshot.filename
            
            # Update existing record
            existing_screenshot.url = data['url']
            existing_screenshot.filename = filename
            existing_screenshot.last_updated = datetime.utcnow()
            
            # Copy crop info from old screenshot to new one if it exists
            old_crop_info = ScreenshotCropInfo.query.filter_by(filename=old_filename).first()
            if old_crop_info:
                current_app.logger.info(f"Copying crop info from {old_filename} to {filename}")
                # Check if crop info already exists for the new filename
                new_crop_info = ScreenshotCropInfo.query.filter_by(filename=filename).first()
                if not new_crop_info:
                    new_crop_info = ScreenshotCropInfo(filename=filename)
                    db.session.add(new_crop_info)
                
                # Copy all crop data
                new_crop_info.x = old_crop_info.x
                new_crop_info.y = old_crop_info.y
                new_crop_info.width = old_crop_info.width
                new_crop_info.height = old_crop_info.height
                new_crop_info.resolution = old_crop_info.resolution
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
        error_message = str(e)
        current_app.logger.error(f"Error taking screenshot: {error_message}")
        
        # Provide more user-friendly error messages for common issues
        if "Navigation Timeout Exceeded" in error_message:
            user_message = "The website took too long to load. This could be due to the site being slow or unresponsive. You can try again later."
        elif "net::ERR_NAME_NOT_RESOLVED" in error_message:
            user_message = "Could not resolve the website address. Please check that the URL is correct."
        elif "net::ERR_CONNECTION_REFUSED" in error_message:
            user_message = "Connection to the website was refused. The site might be down or blocking automated access."
        elif "browserWSEndpoint" in error_message:
            user_message = "Could not connect to the browserless service. Please check your browserless configuration."
        else:
            user_message = f"Error: {error_message}"
            
        return jsonify({
            "status": "error",
            "message": user_message,
            "technical_details": error_message
        }), 500

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
        
        # Navigate to the URL with a longer timeout and better error handling
        current_app.logger.info(f"Navigating to {url}")
        try:
            # Increase timeout to 120 seconds and use a more lenient waitUntil option
            await page.goto(url, {
                'waitUntil': 'domcontentloaded',  # Less strict than networkidle2
                'timeout': 120000  # 120 seconds timeout
            })
            
            # After initial load, wait for network to be idle with a separate timeout
            current_app.logger.info("Page loaded, waiting for network idle...")
            try:
                await page.waitForNavigation({
                    'waitUntil': 'networkidle2',
                    'timeout': 30000  # 30 seconds additional timeout for network idle
                })
            except Exception as e:
                # If waiting for network idle times out, we can still proceed
                current_app.logger.warning(f"Network idle timeout, but continuing: {str(e)}")
        except Exception as e:
            current_app.logger.error(f"Navigation error: {str(e)}")
            # Try to proceed anyway - we might still be able to take a screenshot
            current_app.logger.info("Attempting to continue despite navigation error")
        
        # First take a screenshot of the page with potential cookie banners
        current_app.logger.info("Taking initial screenshot")
        initial_screenshot_path = os.path.join(os.path.dirname(filepath), f"initial_{os.path.basename(filepath)}")
        await page.screenshot({'path': initial_screenshot_path, 'type': 'jpeg', 'quality': 80})
        
        # Use DOM-based approach to handle cookie consent - simpler and more reliable
        current_app.logger.info("Attempting to handle cookie consent using DOM approach")
        consent_handled = await handle_cookie_consent_dom(page)
        
        if consent_handled:
            current_app.logger.info("Cookie consent handled, waiting for page to stabilize...")
            # Crucial: Wait long enough for the cookie banner to disappear and page to rerender
            await page.waitFor(5000)  # 5 seconds wait after successful consent handling
            
            # Reload the page to ensure clean view without cookie banners
            current_app.logger.info("Reloading page to get clean view after handling cookie popup")
            try:
                await page.reload({'waitUntil': 'domcontentloaded', 'timeout': 60000})
                # Another critical wait after reload
                await page.waitFor(5000)
            except Exception as e:
                current_app.logger.warning(f"Error reloading page: {str(e)}, continuing anyway")
        else:
            current_app.logger.info("No cookie consent handling was needed or possible")
        
        # Take the final screenshot with additional error handling
        current_app.logger.info(f"Taking final clean screenshot and saving to {filepath}")
        try:
            await page.screenshot({'path': filepath, 'type': 'jpeg', 'quality': 90, 'fullPage': True})
        except Exception as e:
            current_app.logger.error(f"Error taking screenshot: {str(e)}")
            # Try with fullPage=False as a fallback
            current_app.logger.info("Trying fallback screenshot method without fullPage option")
            await page.screenshot({'path': filepath, 'type': 'jpeg', 'quality': 90, 'fullPage': False})
        
        # Compare before and after images if there was consent handling
        if consent_handled and os.path.exists(initial_screenshot_path) and os.path.exists(filepath):
            current_app.logger.info("Comparing initial and final screenshots to verify cookie banner removal")
            try:
                # Simple check - images shouldn't be identical if banner was removed
                from PIL import Image, ImageChops
                
                with Image.open(initial_screenshot_path) as img1, Image.open(filepath) as img2:
                    # Check if images are different
                    diff = ImageChops.difference(img1, img2)
                    if diff.getbbox():
                        current_app.logger.info("Screenshots are different - cookie banner likely removed successfully")
                    else:
                        current_app.logger.warning("Screenshots are identical - cookie banner may not have been removed")
                
                # Cleanup initial screenshot
                os.remove(initial_screenshot_path)
            except Exception as e:
                current_app.logger.error(f"Error comparing screenshots: {str(e)}")
        
        # Close the browser connection
        await browser.close()
        
        return True
        
    except Exception as e:
        current_app.logger.error(f"Error in pyppeteer: {str(e)}")
        raise e

# Function to handle cookie consent using DOM-based approach
async def handle_cookie_consent_dom(page):
    """A simplified DOM-based approach to handle cookie consent popups"""
    current_app.logger.info("Starting DOM-based cookie consent handling")
    
    # Track if we successfully handled consent
    handled = False
    
    try:
        # First ensure the page has had plenty of time to fully render
        current_app.logger.info("Waiting for page to fully render before handling cookie popups (15 seconds)...")
        await page.waitFor(15000)  # Extended initial wait to ensure all JS has loaded and banners have appeared
        
        # Sometimes cookie banners appear after additional time or user interaction
        try:
            # Simulate scroll which often triggers cookie banners
            await page.evaluate('''
                () => {
                    window.scrollBy(0, 100);
                    window.scrollBy(0, -100);
                    
                    // Force additional delay to ensure everything is loaded
                    return new Promise(resolve => setTimeout(resolve, 3000));
                }
            ''')
            current_app.logger.info("Scrolled page to trigger any lazy-loaded cookie banners")
        except Exception as e:
            current_app.logger.warning(f"Failed to scroll page: {str(e)}")
        
        current_app.logger.info("Page should be fully rendered now, proceeding with cookie consent handling")
        
        # 1. First try direct JavaScript click on consent buttons
        consent_handled = await page.evaluate('''
            () => {
                // These are button texts that would indicate a cookie acceptance button
                const acceptTexts = [
                    // English
                    'accept', 'accept all', 'accept cookies', 'allow', 'allow all', 'ok', 'got it', 'agree',
                    // Danish
                    'accepter', 'acceptér', 'tillad', 'tillad alle', 'ja tak', 
                    // German
                    'akzeptieren', 'alle akzeptieren', 'zustimmen', 'einverstanden',
                    // French
                    'accepter', 'tout accepter', 'jaccepte',
                    // Spanish
                    'aceptar', 'aceptar todo', 'permitir'
                ];
                
                const buttonSelectors = [
                    // Most common selectors
                    'button', 'a.button', 'a.btn', 'input[type="button"]', 'input[type="submit"]', 
                    '[role="button"]', '.btn', '[tabindex="0"]'
                ];
                
                // Try to find any visible button with text containing any of our acceptance terms
                let clickedSomething = false;
                
                // Function to check if a text contains any acceptance term
                const hasAcceptText = (text) => {
                    if (!text) return false;
                    text = text.toLowerCase().trim();
                    return acceptTexts.some(term => text.includes(term));
                };
                
                // Function to click one element and remember we did it
                const trySingleClick = (element, reason) => {
                    try {
                        console.log(`Cookie consent: clicking ${reason}`);
                        element.click();
                        clickedSomething = true;
                        return true;
                    } catch (e) {
                        return false;
                    }
                };
                
                // 1. First try most common framework-specific selectors
                const commonSelectors = [
                    '#CybotCookiebotDialogBodyLevelButtonLevelOptinAllowAll',
                    '#onetrust-accept-btn-handler',
                    '.cc-accept', '.cc-allow', '.cc-dismiss',
                    '#accept-cookies', '#acceptCookies', '#cookie-accept', '#accept-all-cookies',
                    '#acceptAllCookies', '#cookies-accept-all', '#cookie-accept-all', '#gdpr-accept',
                    '.cookie-accept', '.accept-cookies', '.accept-all-cookies', '.acceptAllCookies'
                ];
                
                for (const selector of commonSelectors) {
                    const element = document.querySelector(selector);
                    if (element && element.offsetParent !== null) { // Check if visible
                        if (trySingleClick(element, `framework selector: ${selector}`)) return true;
                    }
                }
                
                // 2. Look through all potential buttons
                for (const selector of buttonSelectors) {
                    const buttons = document.querySelectorAll(selector);
                    for (const button of buttons) {
                        // Skip invisible elements
                        if (!button || !button.offsetParent) continue;
                        
                        // Check button text
                        const text = button.textContent || button.innerText || button.value || '';
                        if (hasAcceptText(text)) {
                            if (trySingleClick(button, `text match: ${text}`)) return true;
                        }
                        
                        // Check aria-label
                        const ariaLabel = button.getAttribute('aria-label');
                        if (hasAcceptText(ariaLabel)) {
                            if (trySingleClick(button, `aria-label: ${ariaLabel}`)) return true;
                        }
                        
                        // Check for certain class names or attributes that might indicate cookie consent
                        const classList = button.classList ? Array.from(button.classList) : [];
                        const hasCookieClass = classList.some(cls => cls.toLowerCase().includes('cookie') || cls.toLowerCase().includes('consent'));
                        
                        if (hasCookieClass) {
                            if (trySingleClick(button, `cookie-related class`)) return true;
                        }
                    }
                }
                
                // 3. Handle fixed banners common in cookie consent UIs by looking at position
                const fixedElements = document.querySelectorAll('div[style*="position: fixed"]');
                for (const el of fixedElements) {
                    if (!el || !el.offsetParent) continue;
                    
                    // Check if this fixed element contains any buttons
                    const buttonsInFixed = el.querySelectorAll('button, a, [role="button"]');
                    for (const btn of buttonsInFixed) {
                        const text = btn.textContent || btn.innerText || '';
                        if (hasAcceptText(text)) {
                            if (trySingleClick(btn, `button in fixed element: ${text}`)) return true;
                        }
                    }
                }
                
                // 4. If we've clicked something, report success
                return clickedSomething;
            }
        ''')
        
        if consent_handled:
            current_app.logger.info("Successfully clicked cookie consent button via JavaScript")
            handled = True
            # Critical wait time after clicking
            await page.waitFor(3000)
        else:
            current_app.logger.info("No cookie consent button clicked via JavaScript approach")
            
            # 2. Try direct selectors with Puppeteer's click method
            common_selectors = [
                '#CybotCookiebotDialogBodyLevelButtonLevelOptinAllowAll',
                '#onetrust-accept-btn-handler',
                '.cc-accept', '.cc-allow', '.cc-dismiss',
                '#accept-cookies', '#acceptCookies', '#cookie-accept', '#accept-all-cookies',
                '#acceptAllCookies', '#cookies-accept-all', '#cookie-accept-all', '#gdpr-accept',
                '.cookie-accept', '.accept-cookies', '.accept-all-cookies', '.acceptAllCookies'
            ]
            
            for selector in common_selectors:
                try:
                    # Check if element exists and is visible
                    visible = await page.evaluate('''
                        (selector) => {
                            const el = document.querySelector(selector);
                            return el && el.offsetParent !== null;
                        }
                    ''', selector)
                    
                    if visible:
                        current_app.logger.info(f"Found visible cookie consent button: {selector}")
                        await page.click(selector, {'timeout': 2000})
                        current_app.logger.info(f"Clicked cookie consent button: {selector}")
                        handled = True
                        # Wait longer after clicking
                        await page.waitFor(3000)
                        break
                except Exception as e:
                    current_app.logger.debug(f"Error clicking selector {selector}: {str(e)}")
        
        # 3. If still not handled, try DOM removal of banners
        if not handled:
            current_app.logger.info("Attempting to remove cookie banners via DOM manipulation")
            
            removed = await page.evaluate('''
                () => {
                    // Selectors for common cookie banners
                    const bannerSelectors = [
                        '[class*="cookie-banner"]', '[id*="cookie-banner"]',
                        '[class*="cookie-consent"]', '[id*="cookie-consent"]',
                        '[class*="cookie-notice"]', '[id*="cookie-notice"]',
                        '.cc-window', '.cc-banner', '#cookie-law-info-bar',
                        'div[style*="position: fixed"][style*="bottom"]',
                        'div[style*="position: fixed"][style*="top"]',
                        '[class*="gdpr"]', '[id*="gdpr"]'
                    ];
                    
                    let removed = 0;
                    
                    // Find and remove banner elements
                    for (const selector of bannerSelectors) {
                        const elements = document.querySelectorAll(selector);
                        elements.forEach(el => {
                            if (el && el.offsetParent !== null) {
                                // Hide element with CSS
                                el.style.display = 'none !important';
                                el.style.visibility = 'hidden !important';
                                el.style.opacity = '0 !important';
                                el.style.pointerEvents = 'none !important';
                                
                                // Attempt to remove from DOM
                                try {
                                    if (el.parentNode) {
                                        el.parentNode.removeChild(el);
                                    } else {
                                        el.remove();
                                    }
                                    removed++;
                                } catch (e) {
                                    // If removal fails, at least we've hidden it
                                }
                            }
                        });
                    }
                    
                    // Add CSS to force page to be scrollable and hide other banners
                    const style = document.createElement('style');
                    style.innerHTML = `
                        body { 
                            overflow: auto !important; 
                            height: auto !important; 
                        }
                        
                        /* Hide cookie banners */
                        [class*="cookie-banner"], [id*="cookie-banner"],
                        [class*="cookie-consent"], [id*="cookie-consent"],
                        [class*="cookie-notice"], [id*="cookie-notice"],
                        .cc-window, .cc-banner, #cookie-law-info-bar,
                        div[style*="position: fixed"][style*="bottom"],
                        div[style*="position: fixed"][style*="top"],
                        [class*="gdpr"], [id*="gdpr"] {
                            display: none !important;
                            visibility: hidden !important;
                            opacity: 0 !important;
                            height: 0 !important;
                            pointer-events: none !important;
                        }
                    `;
                    document.head.appendChild(style);
                    
                    return removed;
                }
            ''')
            
            if removed > 0:
                current_app.logger.info(f"Removed {removed} cookie banner elements")
                handled = True
                # Wait after DOM manipulation
                await page.waitFor(3000)
        
        # 4. Try checking iframes as a last resort
        if not handled:
            current_app.logger.info("Checking for cookie consent buttons in iframes")
            
            iframe_handled = await page.evaluate('''
                () => {
                    // Try to find and access all iframes
                    const iframes = document.querySelectorAll('iframe');
                    let clicked = false;
                    
                    // Function to check if a text contains acceptance terms
                    const hasAcceptText = (text) => {
                        if (!text) return false;
                        text = text.toLowerCase().trim();
                        const terms = ['accept', 'agree', 'allow', 'ok', 'got it', 'accepter', 'accepte', 'akzeptieren', 'aceptar'];
                        return terms.some(term => text.includes(term));
                    };
                    
                    // Try to access each iframe
                    for (const iframe of iframes) {
                        try {
                            // Skip invisible iframes
                            if (!iframe || !iframe.offsetParent) continue;
                            
                            // Try to access the iframe's content
                            const iframeDoc = iframe.contentDocument || iframe.contentWindow.document;
                            
                            // Look for buttons in the iframe
                            const buttons = iframeDoc.querySelectorAll('button, a, [role="button"]');
                            
                            for (const btn of buttons) {
                                if (!btn || !btn.offsetParent) continue;
                                
                                const text = btn.textContent || btn.innerText || '';
                                if (hasAcceptText(text)) {
                                    console.log(`Clicking button in iframe: ${text}`);
                                    btn.click();
                                    clicked = true;
                                    break;
                                }
                            }
                            
                            if (clicked) break;
                        } catch (e) {
                            // Security restrictions may prevent accessing iframe content
                            // Just continue to the next iframe
                        }
                    }
                    
                    return clicked;
                }
            ''')
            
            if iframe_handled:
                current_app.logger.info("Successfully clicked button in iframe")
                handled = True
                # Wait after iframe handling
                await page.waitFor(3000)
        
        return handled
            
    except Exception as e:
        current_app.logger.error(f"Error in handle_cookie_consent_dom: {str(e)}")
        return False

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
    
    # Check if we should return the cropped version
    show_cropped = request.args.get('cropped', 'false').lower() == 'true'
    
    # Get device address if provided
    device_address = request.args.get('device', None)
    
    if show_cropped:
        # Query for crop info for this screenshot with the specific device if provided
        query = ScreenshotCropInfo.query.filter_by(filename=filename)
        if device_address:
            crop_info = query.filter_by(device_address=device_address).first()
            if not crop_info:
                # If not found for this device, try the default device
                crop_info = query.filter_by(device_address='default_device').first()
        else:
            # If no device specified, get the first crop info (could be any device)
            crop_info = query.first()
        
        if crop_info and all(getattr(crop_info, attr, None) is not None for attr in ['x', 'y', 'width', 'height']):
            try:
                # Create a temporary cropped version
                filepath = os.path.join(screenshots_folder, filename)
                with Image.open(filepath) as img:
                    # Log the crop dimensions for debugging
                    current_app.logger.info(f"Cropping image with dimensions: x={crop_info.x}, y={crop_info.y}, w={crop_info.width}, h={crop_info.height}, device={crop_info.device_address}")
                    
                    # Get original image size
                    orig_width, orig_height = img.size
                    current_app.logger.info(f"Original image dimensions: {orig_width}x{orig_height}")
                    
                    # Ensure crop coordinates are within the image bounds
                    x1 = max(0, int(crop_info.x))
                    y1 = max(0, int(crop_info.y))
                    x2 = min(orig_width, int(crop_info.x + crop_info.width))
                    y2 = min(orig_height, int(crop_info.y + crop_info.height))
                    
                    # Log the adjusted crop box
                    current_app.logger.info(f"Adjusted crop box for device {crop_info.device_address}: ({x1}, {y1}, {x2}, {y2})")
                    
                    # Perform the crop with exact pixel coordinates
                    cropped = img.crop((x1, y1, x2, y2))
                    cropped = img.crop((x1, y1, x2, y2))
                    
                    # Create a temporary file
                    temp_dir = os.path.join(current_app.config['DATA_FOLDER'], "temp")
                    if not os.path.exists(temp_dir):
                        os.makedirs(temp_dir)
                    
                    # Save with maximum quality and no compression to preserve detail
                    temp_filename = os.path.join(temp_dir, f"cropped_{filename}")
                    cropped.save(temp_filename, format="JPEG", quality=100, optimize=True, subsampling=0)
                    
                    # Log the final cropped dimensions
                    current_app.logger.info(f"Final cropped image dimensions: {cropped.width}x{cropped.height}")
                    
                    # Use Flask's send_file instead of send_from_directory
                    from flask import send_file, after_this_request
                    
                    @after_this_request
                    def remove_file(response):
                        try:
                            os.remove(temp_filename)
                        except Exception as error:
                            current_app.logger.error(f"Error removing temporary file: {error}")
                        return response
                    
                    return send_file(temp_filename, mimetype='image/jpeg')
            except Exception as e:
                current_app.logger.error(f"Error creating cropped image: {str(e)}")
                # Fall back to original image if cropping fails
                pass
    
    # Return the original image if no cropping requested or if cropping failed
    return send_from_directory(screenshots_folder, filename)

@browserless_bp.route('/api/get_screenshot_crop_info/<filename>', methods=['GET'])
def get_screenshot_crop_info(filename):
    """Get crop information for a screenshot."""
    try:
        # Check if a specific device is requested
        device_address = request.args.get('device', None)
        
        # First get a list of all available devices for this screenshot
        devices = []
        with db.engine.connect() as conn:
            device_result = conn.execute(sa.text("""
                SELECT DISTINCT device_address
                FROM screenshot_crop_info
                WHERE filename = :filename
            """), {"filename": filename})
            
            for device_row in device_result:
                devices.append({"address": device_row[0]})
        
        # Check if crop info exists using raw SQL to avoid ORM issues
        with db.engine.connect() as conn:
            query_params = {"filename": filename}
            sql_query = """
                SELECT filename, device_address, x, y, width, height, resolution
                FROM screenshot_crop_info
                WHERE filename = :filename
            """
            
            # Add device filter if specified
            if device_address:
                sql_query += " AND device_address = :device_address"
                query_params["device_address"] = device_address
            else:
                # If no device specified, get the first entry (or filter for default device if needed)
                sql_query += " LIMIT 1"
            
            result = conn.execute(sa.text(sql_query), query_params)
            row = result.fetchone()
            
            if row:
                # Return the crop info as JSON with basic data
                response_data = {
                    "status": "success",
                    "crop_info": {
                        "x": float(row[2]) if row[2] is not None else 0,          # x is now the 3rd column
                        "y": float(row[3]) if row[3] is not None else 0,          # y is now the 4th column
                        "width": float(row[4]) if row[4] is not None else 0,      # width is now the 5th column
                        "height": float(row[5]) if row[5] is not None else 0,     # height is now the 6th column
                        "resolution": row[6],                                     # resolution is now the 7th column
                        "device_address": row[1]                                  # device_address is the 2nd column
                    },
                    "available_devices": devices
                }
                
                current_app.logger.info(f"Retrieved crop info for {filename} with device {row[1]} using raw SQL")
                return jsonify(response_data), 200
            else:
                # No crop info found
                current_app.logger.info(f"No crop info found for {filename}" +
                                        (f" with device {device_address}" if device_address else ""))
                return jsonify({
                    "status": "success",
                    "message": "No crop information found for this screenshot",
                    "crop_info": None,
                    "available_devices": devices
                }), 200  # Return 200 with empty crop_info instead of 404
    except Exception as e:
        current_app.logger.error(f"Error retrieving crop info: {str(e)}")
        return jsonify({
            "status": "error",
            "message": f"Error retrieving crop information: {str(e)}"
        }), 500

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
    
    # Get device address - either directly provided or from the device
    device_addr = None
    if "device" in crop_data:
        device_addr = crop_data.get("device")
        device_obj = Device.query.filter_by(address=device_addr).first()
        if device_obj and device_obj.resolution:
            crop_data["resolution"] = device_obj.resolution
            current_app.logger.info(f"Saving crop with device: {device_addr}, resolution: {device_obj.resolution}")
        else:
            current_app.logger.warning(f"Device not found or missing resolution: {device_addr}")
    
    # If no device address is provided, use a default
    if not device_addr:
        device_addr = "default_device"
        current_app.logger.warning(f"No device provided for crop data, using default_device")
    
    # Log the received crop data
    current_app.logger.info(f"Saving crop data for {filename} with device {device_addr}: {crop_data}")
    
    try:
        # Use a direct raw SQL approach that only uses the columns we know exist
        conn = db.engine.connect()
        
        # Delete only this specific device's crop info, not all crops for this filename
        conn.execute(sa.text("""
            DELETE FROM screenshot_crop_info
            WHERE filename = :filename AND device_address = :device_address
        """), {
            "filename": filename,
            "device_address": device_addr
        })
        
        # Insert using all required columns including device_address
        conn.execute(sa.text("""
            INSERT INTO screenshot_crop_info (filename, device_address, x, y, width, height, resolution)
            VALUES (:filename, :device_address, :x, :y, :width, :height, :resolution)
        """), {
            "filename": filename,
            "device_address": device_addr,
            "x": crop_data.get("x", 0),
            "y": crop_data.get("y", 0),
            "width": crop_data.get("width", 0),
            "height": crop_data.get("height", 0),
            "resolution": crop_data.get("resolution", "")
        })
        
        # Make sure to commit the transaction
        db.session.commit()
        
        current_app.logger.info(f"Crop data saved successfully for {filename} with device {device_addr} using direct SQL")
        return jsonify({"status": "success", "device_address": device_addr}), 200
        
    except Exception as e:
        db.session.rollback()
        current_app.logger.error(f"Error saving crop info (using RAW SQL) for {filename} with device {device_addr}: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500
    
    current_app.logger.info(f"Crop info successfully saved for {filename}")
    
    return jsonify({"status": "success"}), 200

@browserless_bp.route('/send_screenshot/<filename>', methods=['POST'])
def send_screenshot(filename):
    screenshots_folder = os.path.join(current_app.config['DATA_FOLDER'], 'screenshots')
    filepath = os.path.join(screenshots_folder, filename)
    
    # Log the request details for debugging
    current_app.logger.info(f"Send screenshot request received for filename: {filename}")
    
    if not os.path.exists(filepath):
        current_app.logger.error(f"File not found: {filepath}")
        return jsonify({"status": "error", "message": "File not found"}), 404
    
    device_addr = request.form.get("device")
    if not device_addr:
        current_app.logger.error("No device specified in request")
        return jsonify({"status": "error", "message": "No device specified"}), 400

    current_app.logger.info(f"Sending to device: {device_addr}")
    device_obj = Device.query.filter_by(address=device_addr).first()
    if not device_obj:
        current_app.logger.error(f"Device not found in DB: {device_addr}")
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
            # Step 1: Apply crop if available, preferring device-specific crop info
            # First try to get crop info specific to this device
            crop_info = ScreenshotCropInfo.query.filter_by(
                filename=filename,
                device_address=device_addr
            ).first()
            
            # If not found, try to fall back to default device crop
            if not crop_info:
                current_app.logger.info(f"No crop info found for {filename} with device {device_addr}, trying default")
                crop_info = ScreenshotCropInfo.query.filter_by(
                    filename=filename,
                    device_address='default_device'
                ).first()
            
            # If still not found, try any crop info for this screenshot
            if not crop_info:
                current_app.logger.info(f"No default crop info found for {filename}, using first available crop")
                crop_info = ScreenshotCropInfo.query.filter_by(filename=filename).first()
            
            cdata = None
            
            if crop_info:
                cdata = {
                    "x": crop_info.x,
                    "y": crop_info.y,
                    "width": crop_info.width,
                    "height": crop_info.height,
                    "resolution": crop_info.resolution,
                    "device_address": crop_info.device_address
                }
                current_app.logger.info(f"Using crop info for {filename} with device {crop_info.device_address}")
            
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
                final_img = cropped.resize((dev_height, dev_width), Image.Resampling.LANCZOS)  # Note swapped dimensions
            else:
                final_img = cropped.resize((dev_width, dev_height), Image.Resampling.LANCZOS)
            
            current_app.logger.info(f"Final image size: {final_img.size}")
            
            # Save the processed image as a temporary file
            temp_dir = os.path.join(current_app.config['DATA_FOLDER'], "temp")
            if not os.path.exists(temp_dir):
                os.makedirs(temp_dir)
            temp_filename = os.path.join(temp_dir, f"temp_{filename}")
            final_img.save(temp_filename, format="JPEG", quality=95)
            current_app.logger.info(f"Saved temporary file: {temp_filename}")

        # Log the image details before sending
        current_app.logger.info(f"Sending image {filename} to device {device_obj.friendly_name} at {device_addr}")
        current_app.logger.info(f"Temporary file path: {temp_filename}")
        
        # Ensure device address has HTTP protocol
        if not device_addr.startswith(('http://', 'https://')):
            device_addr = f'http://{device_addr}'
            
        # Prepare URL for the request
        url = f"{device_addr}/send_image"
        current_app.logger.info(f"Sending request to: {url}")
        
        try:
            # Use httpx to send a multipart form request
            with open(temp_filename, 'rb') as file_obj:
                # Prepare files for the multipart request
                files = {'file': (filename, file_obj, 'image/jpeg')}
                # Add data parameters including filename parameter
                data = {
                    'source': 'browserless',
                    'filename': filename  # Add filename parameter
                }
                
                # Send the request with a timeout of 2 minutes
                current_app.logger.info(f"Sending httpx POST request")
                with httpx.Client(timeout=120.0) as client:
                    response = client.post(url, files=files, data=data)
                
                # Log the response details
                current_app.logger.info(f"Response status code: {response.status_code}")
                current_app.logger.info(f"Response headers: {response.headers}")
                current_app.logger.info(f"Response content: {response.text}")
            
            # Clean up the temporary file
            os.remove(temp_filename)
            current_app.logger.info(f"Temporary file deleted: {temp_filename}")
            
            if response.status_code != 200:
                current_app.logger.error(f"Error sending image: {response.text}")
                return jsonify({"status": "error", "message": f"Error sending image: {response.text}"}), 500

            # Update the device's last_sent field with the current filename
            device_obj.last_sent = filename
            db.session.commit()
            current_app.logger.info(f"Updated device {device_obj.friendly_name} last_sent to {filename}")
            
            # Add a log entry for this send operation
            add_send_log_entry(filename)
            current_app.logger.info(f"Added send log entry for {filename}")
            
            return jsonify({"status": "success", "message": "Screenshot sent successfully"}), 200
        except httpx.TimeoutException:
            current_app.logger.error(f"HTTP request timed out after 120 seconds")
            try:
                os.remove(temp_filename)
            except:
                pass
            return jsonify({"status": "error", "message": "Request timed out while sending the image to the device"}), 500
        except httpx.RequestError as e:
            current_app.logger.error(f"HTTP request error: {e}")
            try:
                os.remove(temp_filename)
            except:
                pass
            return jsonify({"status": "error", "message": f"Network error while sending the image: {str(e)}"}), 500
        except Exception as e:
            current_app.logger.error(f"Unexpected error during image sending: {e}")
            try:
                os.remove(temp_filename)
            except:
                pass
            return jsonify({"status": "error", "message": f"Error sending image: {str(e)}"}), 500
    except Exception as e:
        current_app.logger.error(f"Error processing screenshot: {str(e)}")
        return jsonify({"status": "error", "message": f"Error processing screenshot: {str(e)}"}), 500
        return jsonify({"status": "error", "message": f"Error processing screenshot: {str(e)}"}), 500