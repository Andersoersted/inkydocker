from flask import Blueprint, request, render_template, jsonify, current_app, abort, send_from_directory
from models import db, BrowserlessConfig, Screenshot, ScreenshotCropInfo, Device
import os
from datetime import datetime
import requests
from PIL import Image
from io import BytesIO
from utils.crop_helpers import load_crop_info_from_db, save_crop_info_to_db, add_send_log_entry
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
        
        # First, take an initial screenshot to analyze for cookie banners
        current_app.logger.info("Taking initial screenshot to analyze for cookie banners")
        initial_screenshot = await page.screenshot({'type': 'jpeg', 'quality': 80})
        
        # Use OpenCLIP to detect if a cookie banner is present
        current_app.logger.info("Using OpenCLIP to detect cookie banners")
        has_banner, similarity_score, matched_prompt = await detect_cookie_banner_with_clip(page)
        
        if has_banner:
            current_app.logger.info(f"Cookie banner detected with {similarity_score:.2f} similarity to '{matched_prompt}'")
            # Handle cookie consent using pyppeteer's native methods
            current_app.logger.info("Handling cookie consent with pyppeteer")
            await handle_cookie_consent(page)
            
            # Wait a moment for any animations to complete
            await page.waitFor(2000)
            
            # Check if the banner is still detected after handling
            has_banner_after, similarity_after, _ = await detect_cookie_banner_with_clip(page)
            if has_banner_after:
                current_app.logger.info(f"Cookie banner still detected after handling (similarity: {similarity_after:.2f}). Trying again.")
                await handle_cookie_consent(page)
                await page.waitFor(2000)  # Wait again after second attempt
            else:
                current_app.logger.info("Cookie banner successfully handled")
                
            # Reload the page to ensure we get a clean view without cookie banners
            current_app.logger.info("Reloading page to get clean view")
            try:
                await page.reload({'waitUntil': 'domcontentloaded', 'timeout': 60000})
                await page.waitFor(3000)  # Wait for page to stabilize
            except Exception as e:
                current_app.logger.warning(f"Error reloading page: {str(e)}, continuing anyway")
        else:
            current_app.logger.info("No cookie banner detected, proceeding with screenshot")
        
        # Take the screenshot with additional error handling
        current_app.logger.info(f"Taking screenshot and saving to {filepath}")
        try:
            await page.screenshot({'path': filepath, 'type': 'jpeg', 'quality': 90, 'fullPage': True})
        except Exception as e:
            current_app.logger.error(f"Error taking screenshot: {str(e)}")
            # Try with fullPage=False as a fallback
            current_app.logger.info("Trying fallback screenshot method without fullPage option")
            await page.screenshot({'path': filepath, 'type': 'jpeg', 'quality': 90, 'fullPage': False})
        
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

# Function to detect cookie banners using OpenCLIP
async def detect_cookie_banner_with_clip(page):
    """
    Use OpenCLIP to detect if a cookie banner is present on the page.
    Returns True if a cookie banner is detected, False otherwise.
    """
    current_app.logger.info("Using OpenCLIP to detect cookie banners")
    
    try:
        # Take a screenshot of the current page
        screenshot_bytes = await page.screenshot({'type': 'jpeg', 'quality': 80})
        
        # Import the get_clip_model function from tasks
        from tasks import get_clip_model, clip_models, clip_preprocessors
        
        # Always use the small model (ViT-B-32) for cookie detection
        model_name = 'ViT-B-32'  # Small model
        
        # Check if the small model is already loaded in cache
        if model_name in clip_models:
            model = clip_models[model_name]
            preprocess = clip_preprocessors[model_name]
        else:
            # Load the small model directly
            # Use the get_clip_model function but override the result to always use ViT-B-32
            model_name, model, preprocess = get_clip_model()
            # If the returned model is not ViT-B-32, force loading it
            if model_name != 'ViT-B-32':
                current_app.logger.info(f"Forcing small model (ViT-B-32) for cookie detection instead of {model_name}")
                cache_dir = "/app/data/model_cache"
                if not os.path.exists(cache_dir):
                    # Try data folder if app folder doesn't exist
                    data_folder = current_app.config.get("DATA_FOLDER", "./data")
                    cache_dir = os.path.join(data_folder, "models")
                
                model, _, preprocess = open_clip.create_model_and_transforms(
                    'ViT-B-32',
                    pretrained='openai',
                    jit=False,
                    force_quick_gelu=True,  # Enable QuickGELU to match pretrained weights
                    cache_dir=cache_dir
                )
                # Set device based on availability first
                device = "cuda" if torch.cuda.is_available() else "cpu"
                
                model.to(device)
                model.eval()
                model_name = 'ViT-B-32'
                
                # Store in cache for future use
                clip_models[model_name] = model
                clip_preprocessors[model_name] = preprocess
                
        current_app.logger.info(f"Using small CLIP model (ViT-B-32) for cookie detection")
        
        # Set device based on availability
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load and preprocess the screenshot
        image = Image.open(io.BytesIO(screenshot_bytes))
        image_input = preprocess(image).unsqueeze(0).to(device)
        
        # Prepare text prompts for cookie banners in multiple languages - expanded for better detection
        prompts = [
            # English prompts
            "a cookie consent banner",
            "accept cookies button",
            "cookie policy notification",
            "GDPR consent dialog",
            "privacy settings popup",
            "cookie preferences",
            "cookie banner with accept button",
            "website cookie acceptance dialog",
            "privacy consent banner",
            "cookie notice with buttons",
            "website privacy notice overlay",
            "cookie compliance banner",
            
            # Danish prompts
            "accepter cookies",
            "accepter alle cookies",
            "cookie samtykke banner",
            "privatlivspolitik popup",
            "godkend cookies knap",
            
            # German prompts
            "akzeptieren cookies",
            "cookie einstellungen",
            "datenschutz-banner",
            "cookie-zustimmungsdialog",
            "alle cookies akzeptieren",
            
            # French prompts
            "accepter les cookies",
            "bannière de consentement aux cookies",
            "paramètres de confidentialité",
            "consentement RGPD",
            
            # Spanish prompts
            "aceptar cookies",
            "banner de consentimiento de cookies",
            "política de privacidad",
            "configuración de cookies"
        ]
        
        # Always use the tokenizer for ViT-B-32 for consistency
        tokenizer = open_clip.get_tokenizer('ViT-B-32')
        text_tokens = tokenizer(prompts)
        
        # Get image and text features
        with torch.no_grad():
            image_features = model.encode_image(image_input)
            text_features = model.encode_text(text_tokens)
            
            # Normalize the features
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            
            # Compute similarity scores
            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            
            # Get the highest similarity score
            max_similarity = similarity.max().item()
            max_index = similarity.argmax().item()
            
            current_app.logger.info(f"Cookie banner detection: highest similarity {max_similarity:.2f} for '{prompts[max_index]}'")
            
            # Lower threshold for detection to increase sensitivity (may need calibration)
            threshold = 0.25
            return max_similarity > threshold, max_similarity, prompts[max_index]
    
    except Exception as e:
        current_app.logger.error(f"Error in OpenCLIP cookie banner detection: {str(e)}")
        return False, 0.0, None

# Function to handle cookie consent using pyppeteer's native methods
async def handle_cookie_consent(page):
    current_app.logger.info("Starting cookie consent handling with pyppeteer")
    
    # Common selectors for cookie consent buttons
    selectors = [
        # ID-based selectors
        '#accept-cookies', '#acceptCookies', '#cookie-accept', '#accept-all-cookies',
        '#acceptAllCookies', '#cookies-accept-all', '#cookie-accept-all', '#gdpr-accept',
        '#accept', '#accept_all', '#acceptAll', '#cookie_accept', '#cookie-consent-accept',
        '#cookieConsent', '#cookieAccept', '#btn-cookie-accept', '#gdpr-consent-accept',
        '#cookie-banner-accept', '#cookieConsentAcceptAllButton', '#accept-cookie-policy',
        
        # Class-based selectors
        '.cookie-accept', '.accept-cookies', '.accept-all-cookies', '.acceptAllCookies',
        '.cookie-consent-accept', '.cookie-banner__accept', '.cookie-notice__accept',
        '.gdpr-accept', '.accept-button', '.cookie-accept-button', '.consent-accept',
        '.cookie-accept-all', '.cookie-banner-accept', '.cookie-consent-button',
        '.cookie-notice-accept', '.gdpr-banner-accept', '.privacy-accept-button',
        
        # Framework-specific selectors
        '#CybotCookiebotDialogBodyLevelButtonLevelOptinAllowAll',
        '#onetrust-accept-btn-handler',
        '.cc-accept', '.cc-allow', '.cc-dismiss',
        '.js-cookie-banner-accept', '.js-accept-cookie-policy',
        
        # Attribute-based selectors
        '[data-action="accept-cookies"]', '[data-role="accept-cookies"]',
        '[data-consent="accept"]', '[data-cookie-accept="all"]',
        '[aria-label*="accept cookies"]', '[aria-label*="Accept all"]',
        '[data-element-id*="cookie-accept"]', '[data-element-id*="cookie-banner"]',
        '[data-testid*="cookie-accept"]', '[data-testid*="cookie-banner-accept"]',
        
        # Additional compound selectors
        'button[class*="cookie"][class*="accept"]',
        'button[class*="gdpr"][class*="accept"]',
        'button[class*="cookie"][class*="allow"]',
        'button[id*="cookie"][id*="accept"]',
        'a[class*="cookie"][class*="accept"]',
        'div[role="button"][class*="cookie"]'
    ]
    
    # Try clicking each selector
    for selector in selectors:
        try:
            # Check if the element exists and is visible
            visible = await page.evaluate('''
                (selector) => {
                    const el = document.querySelector(selector);
                    return el && el.offsetParent !== null;
                }
            ''', selector)
            
            if visible:
                current_app.logger.info(f"Found visible cookie consent button: {selector}")
                await page.click(selector, {'timeout': 1000})
                current_app.logger.info(f"Clicked cookie consent button: {selector}")
                await page.waitFor(500)  # Short wait after clicking
        except Exception as e:
            # Ignore errors for individual selectors
            pass
    
    # Try to find and click buttons by text content - expanded to include more variations
    text_patterns = [
        # English
        'accept', 'accept all', 'accept cookies', 'allow', 'allow all', 'i agree', 'ok', 'got it',
        'agree', 'agree to all', 'continue', 'understood', 'consent', 'confirm', 'save', 'close',
        'agree and close', 'accept and continue', 'accept and proceed', 'accept all cookies',
        'i understand', 'yes', 'agree to cookies', 'accept & continue',
        
        # Danish
        'accepter', 'acceptér', 'tillad', 'tillad alle', 'ja tak', 'accepter alle',
        'forstået', 'fortsæt', 'godkend', 'acceptér alle cookies', 'luk',
        
        # German
        'akzeptieren', 'alle akzeptieren', 'zustimmen', 'einverstanden',
        'ich stimme zu', 'verstanden', 'alle cookies akzeptieren', 'cookies zulassen',
        'erlauben', 'weiter', 'fortfahren', 'bestätigen', 'speichern', 'schließen',
        
        # French
        'accepter', 'tout accepter', 'j\'accepte', 'accepter tous les cookies',
        'continuer', 'compris', 'je comprends', 'consentir', 'fermer',
        'accepter et continuer', 'accepter et fermer',
        
        # Spanish
        'aceptar', 'aceptar todo', 'permitir', 'aceptar cookies',
        'entendido', 'acepto', 'continuar', 'estoy de acuerdo', 'cerrar',
        'aceptar todas', 'aceptar y continuar'
    ]
    
    # Also try with uppercase first letter or all caps
    capitalized_patterns = []
    for pattern in text_patterns:
        capitalized_patterns.append(pattern.capitalize())
        capitalized_patterns.append(pattern.upper())
    
    text_patterns.extend(capitalized_patterns)
    
    for pattern in text_patterns:
        try:
            # Find elements containing the text
            elements = await page.evaluate('''
                (pattern) => {
                    const elements = Array.from(document.querySelectorAll('button, a, div[role="button"], [tabindex="0"]'));
                    return elements
                        .filter(el => {
                            if (!el || !el.offsetParent) return false; // Skip invisible elements
                            const text = (el.textContent || el.innerText || '').toLowerCase();
                            return text.includes(pattern);
                        })
                        .map(el => {
                            const rect = el.getBoundingClientRect();
                            return {
                                x: rect.left + rect.width / 2,
                                y: rect.top + rect.height / 2
                            };
                        });
                }
            ''', pattern)
            
            # Click each element at its center coordinates
            for element in elements:
                current_app.logger.info(f"Found element with text '{pattern}' at coordinates: {element}")
                await page.mouse.click(element['x'], element['y'])
                current_app.logger.info(f"Clicked element with text: {pattern}")
                await page.waitFor(500)  # Short wait after clicking
        except Exception as e:
            # Ignore errors for individual text patterns
            pass
    
    # Try to handle iframes
    try:
        # Get all iframes
        iframes = await page.querySelectorAll('iframe')
        
        for i, iframe in enumerate(iframes):
            try:
                # Try to access the iframe's content
                frame = page.frames[i + 1]  # +1 because the first frame is the main page
                
                if frame:
                    # Try to click accept buttons in the iframe
                    for selector in selectors:
                        try:
                            visible = await frame.evaluate('''
                                (selector) => {
                                    const el = document.querySelector(selector);
                                    return el && el.offsetParent !== null;
                                }
                            ''', selector)
                            
                            if visible:
                                current_app.logger.info(f"Found visible cookie consent button in iframe: {selector}")
                                await frame.click(selector, {'timeout': 1000})
                                current_app.logger.info(f"Clicked cookie consent button in iframe: {selector}")
                        except Exception:
                            # Ignore errors for individual selectors in iframes
                            pass
            except Exception:
                # Ignore errors for individual iframes
                pass
    except Exception as e:
        current_app.logger.info(f"Error handling iframes: {str(e)}")
    
    # Wait a bit for any animations to complete
    await page.waitFor(2000)  # pyppeteer uses waitFor instead of waitForTimeout
    
    # Try to hide any remaining cookie banners - enhanced with more comprehensive selectors
    banner_selectors = [
        # Cookie-specific selectors
        '[class*="cookie-banner"]', '[id*="cookie-banner"]',
        '[class*="cookie-consent"]', '[id*="cookie-consent"]',
        '[class*="cookie-notice"]', '[id*="cookie-notice"]',
        '[class*="cookie-popup"]', '[id*="cookie-popup"]',
        '[class*="cookie-alert"]', '[id*="cookie-alert"]',
        '[class*="cookie-modal"]', '[id*="cookie-modal"]',
        '[class*="cookie-message"]', '[id*="cookie-message"]',
        
        # GDPR-specific selectors
        '[class*="gdpr-banner"]', '[id*="gdpr-banner"]',
        '[class*="gdpr-consent"]', '[id*="gdpr-consent"]',
        '[class*="gdpr-notice"]', '[id*="gdpr-notice"]',
        '[class*="gdpr-popup"]', '[id*="gdpr-popup"]',
        
        # Privacy-specific selectors
        '[class*="privacy-banner"]', '[id*="privacy-banner"]',
        '[class*="privacy-consent"]', '[id*="privacy-consent"]',
        '[class*="privacy-notice"]', '[id*="privacy-notice"]',
        '[class*="privacy-popup"]', '[id*="privacy-popup"]',
        
        # Framework-specific selectors
        '.cc-window', '.cc-banner', '#cookie-law-info-bar',
        '#cookiebanner', '#cookieConsent', '#cookie-consent',
        '#CybotCookiebotDialog', '#onetrust-banner-sdk',
        '.js-cookie-banner', '.js-cookie-consent'
    ]
    
    # First try direct clicks on any remaining accept buttons
    try:
        current_app.logger.info("Executing JavaScript to directly click any remaining cookie acceptance buttons")
        await page.evaluate('''
            () => {
                function containsAcceptText(text) {
                    text = text.toLowerCase();
                    const acceptTerms = ['accept', 'accept all', 'agree', 'allow', 'continue', 'ok',
                                        'yes', 'consent', 'got it', 'understand', 'accepter', 'akzeptieren',
                                        'tillad', 'aceptar'];
                    return acceptTerms.some(term => text.includes(term));
                }
                
                // Find all buttons, links, and clickable elements
                const elements = Array.from(document.querySelectorAll('button, a, div[role="button"], [tabindex="0"], input[type="button"], input[type="submit"]'));
                
                // Try to click any that contain acceptance text and are visible
                elements.forEach(el => {
                    if (el && el.offsetParent !== null) {
                        const text = (el.textContent || el.innerText || el.value || '').trim();
                        if (containsAcceptText(text)) {
                            console.log("Directly clicking element with text: " + text);
                            try { el.click(); } catch (e) { /* ignore */ }
                        }
                    }
                });
            }
        ''')
        await page.waitFor(1000)  # Wait for click actions to process
    except Exception as e:
        current_app.logger.info(f"JavaScript direct click failed: {str(e)}")
    
    # Then try to hide all cookie banners with multiple techniques
    for selector in banner_selectors:
        try:
            await page.evaluate('''
                (selector) => {
                    const elements = document.querySelectorAll(selector);
                    elements.forEach(el => {
                        if (el && el.offsetParent !== null) {
                            // Try multiple methods to hide the element
                            el.style.display = 'none';
                            el.style.visibility = 'hidden';
                            el.style.opacity = '0';
                            el.style.pointerEvents = 'none';
                            el.style.height = '0px';
                            el.style.maxHeight = '0px';
                            el.style.overflow = 'hidden';
                            el.setAttribute('aria-hidden', 'true');
                            
                            // If possible, remove it entirely
                            try { el.remove(); } catch (e) { /* ignore */ }
                            
                            // If it has a parent, try to remove from parent
                            if (el.parentNode) {
                                try { el.parentNode.removeChild(el); } catch (e) { /* ignore */ }
                            }
                        }
                    });
                }
            ''', selector)
        except Exception:
            # Ignore errors for individual banner selectors
            pass
    
    # Finally, inject CSS to hide common patterns
    try:
        await page.evaluate('''
            () => {
                // Add a style tag to hide common cookie banner patterns
                const style = document.createElement('style');
                style.innerHTML = `
                    /* Hide common cookie banners */
                    div[class*="cookie-banner"], div[id*="cookie-banner"],
                    div[class*="cookie-consent"], div[id*="cookie-consent"],
                    div[class*="cookie-notice"], div[id*="cookie-notice"],
                    div[class*="cookie-popup"], div[id*="cookie-popup"],
                    div[class*="gdpr"], div[id*="gdpr"],
                    .cc-window, .cc-banner, #cookie-law-info-bar,
                    div[class*="fixed"][class*="bottom"],
                    div[style*="position: fixed"][style*="bottom"],
                    div[style*="position: fixed"][style*="top"] {
                        display: none !important;
                        visibility: hidden !important;
                        opacity: 0 !important;
                        height: 0 !important;
                        max-height: 0 !important;
                        overflow: hidden !important;
                        pointer-events: none !important;
                    }
                    
                    /* Force body to be scrollable */
                    body {
                        overflow: auto !important;
                        height: auto !important;
                    }
                `;
                document.head.appendChild(style);
            }
        ''')
    except Exception as e:
        current_app.logger.info(f"Could not inject CSS to hide cookie banners: {str(e)}")

@browserless_bp.route('/screenshots/<filename>')
def get_screenshot(filename):
    screenshots_folder = os.path.join(current_app.config['DATA_FOLDER'], 'screenshots')
    
    # Check if we should return the cropped version
    show_cropped = request.args.get('cropped', 'false').lower() == 'true'
    
    if show_cropped:
        # Check if crop info exists for this screenshot
        crop_info = ScreenshotCropInfo.query.filter_by(filename=filename).first()
        
        if crop_info and all(getattr(crop_info, attr, None) is not None for attr in ['x', 'y', 'width', 'height']):
            try:
                # Create a temporary cropped version
                filepath = os.path.join(screenshots_folder, filename)
                with Image.open(filepath) as img:
                    # Crop the image
                    cropped = img.crop((
                        crop_info.x,
                        crop_info.y,
                        crop_info.x + crop_info.width,
                        crop_info.y + crop_info.height
                    ))
                    
                    # Create a temporary file
                    temp_dir = os.path.join(current_app.config['DATA_FOLDER'], "temp")
                    if not os.path.exists(temp_dir):
                        os.makedirs(temp_dir)
                    
                    temp_filename = os.path.join(temp_dir, f"cropped_{filename}")
                    cropped.save(temp_filename, format="JPEG", quality=95)
                    
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
    # Check if crop info exists for this filename
    crop_info = ScreenshotCropInfo.query.filter_by(filename=filename).first()
    
    if crop_info:
        # Return the crop info as JSON
        return jsonify({
            "status": "success",
            "crop_info": {
                "x": crop_info.x,
                "y": crop_info.y,
                "width": crop_info.width,
                "height": crop_info.height,
                "resolution": crop_info.resolution
            }
        }), 200
    else:
        # No crop info found
        return jsonify({
            "status": "error",
            "message": "No crop information found for this screenshot"
        }), 404

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