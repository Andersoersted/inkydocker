from flask import Blueprint, request, jsonify, render_template
from models import db, ScheduleEvent, Device, ImageDB, Screenshot
import datetime
from tasks import send_scheduled_image
# Import the scheduler from the dedicated scheduler module instead of tasks
# This ensures we're using the scheduler that runs in a dedicated process
from scheduler import scheduler

schedule_bp = Blueprint('schedule', __name__)

@schedule_bp.route('/schedule')
def schedule_page():
    devs = Device.query.all()
    devices = []
    for d in devs:
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
    
    # Get all images and their tags
    imgs_db = ImageDB.query.all()
    images = [i.filename for i in imgs_db]
    
    # Create a dictionary of image tags for the template
    image_tags = {}
    for img in imgs_db:
        if img.tags:
            image_tags[img.filename] = img.tags
    
    # Get all browserless screenshots
    screenshots = Screenshot.query.all()
    
    return render_template("schedule.html", devices=devices, images=images, image_tags=image_tags, screenshots=screenshots)

@schedule_bp.route('/schedule/events')
def get_events():
    # Ensure we're getting the latest data from the database
    db.session.expire_all()
    events = ScheduleEvent.query.all()
    event_list = []
    
    # Use timezone-aware datetime objects with Copenhagen timezone
    import pytz
    copenhagen_tz = pytz.timezone('Europe/Copenhagen')
    now = datetime.datetime.now(copenhagen_tz)
    future_horizon = now + datetime.timedelta(days=90)
    past_horizon = now - datetime.timedelta(days=90)  # Show past events up to 90 days back
    print(f"Loading {len(events)} events from database")
    
    # Create a device lookup dictionary for colors and friendly names
    device_lookup = {}
    devices = Device.query.all()
    for device in devices:
        device_lookup[device.address] = {
            "color": device.color,
            "friendly_name": device.friendly_name
        }
    
    # Create an image lookup for thumbnails
    image_lookup = {}
    images = ImageDB.query.all()
    for img in images:
        image_lookup[img.filename] = {
            "tags": img.tags or ""
        }
    
    for ev in events:
        # Get device color and friendly name
        device_info = device_lookup.get(ev.device, {"color": "#cccccc", "friendly_name": ev.device})
        device_color = device_info["color"]
        device_name = device_info["friendly_name"]
        
        if ev.recurrence.lower() == "none":
            # Check if the file is a screenshot
            is_screenshot = False
            thumbnail_url = f"/thumbnail/{ev.filename}"
            current_filename = ev.filename  # Keep track of the current filename
            
            # Check if this is a screenshot
            screenshot = Screenshot.query.filter_by(filename=ev.filename).first()
            if screenshot:
                is_screenshot = True
                # Use the current filename from the Screenshot record, which could be updated during refresh
                thumbnail_url = f"/screenshots/{screenshot.filename}?cropped=true"
                current_filename = screenshot.filename  # Update to the current screenshot filename
            
            event_list.append({
                "id": ev.id,
                "title": f"{ev.filename}",
                "start": ev.datetime_str,
                "device": ev.device,
                "deviceName": device_name,
                "filename": ev.filename,
                "recurrence": ev.recurrence,
                "series": False,
                "backgroundColor": device_color,
                "borderColor": device_color,
                "textColor": "#ffffff",
                "extendedProps": {
                    "thumbnail": thumbnail_url,
                    "isScreenshot": is_screenshot,
                    "refreshScreenshot": ev.refresh_screenshot,
                    "currentFilename": current_filename
                }
            })
        else:
            # Generate recurring occurrences from the next occurrence up to the horizon
            try:
                # Parse the datetime string and ensure it's timezone-aware with Copenhagen timezone
                import pytz
                copenhagen_tz = pytz.timezone('Europe/Copenhagen')
                
                start_dt = datetime.datetime.fromisoformat(ev.datetime_str)
                
                # Ensure start_dt is timezone-aware for proper comparison with now
                if start_dt.tzinfo is None:
                    # If the datetime is naive, assume it's in Copenhagen timezone
                    start_dt = copenhagen_tz.localize(start_dt)
                else:
                    # If the datetime already has timezone info, convert it to Copenhagen timezone
                    start_dt = start_dt.astimezone(copenhagen_tz)
            except Exception as e:
                print(f"Error parsing datetime for event {ev.id}: {str(e)}")
                continue
                
            # For recurring events, generate occurrences both in the past and future
            rec = ev.recurrence.lower()
            
            # First, find the earliest occurrence we want to show (past_horizon)
            occurrence = start_dt
            earliest_occurrence = occurrence
            
            # Go backwards to find past occurrences until we reach past_horizon
            while earliest_occurrence > past_horizon:
                if rec == "daily":
                    temp = earliest_occurrence - datetime.timedelta(days=1)
                elif rec == "weekly":
                    temp = earliest_occurrence - datetime.timedelta(weeks=1)
                elif rec == "monthly":
                    temp = earliest_occurrence - datetime.timedelta(days=30)
                else:
                    break
                
                # If we're still after past_horizon, update earliest_occurrence
                if temp >= past_horizon:
                    earliest_occurrence = temp
                else:
                    break
            
            # Now generate all occurrences from earliest_occurrence to future_horizon
            occurrence = earliest_occurrence
            while occurrence <= future_horizon:
                # Check if the file is a screenshot
                is_screenshot = False
                thumbnail_url = f"/thumbnail/{ev.filename}"
                current_filename = ev.filename  # Keep track of the current filename
                
                # Check if this is a screenshot
                screenshot = Screenshot.query.filter_by(filename=ev.filename).first()
                if screenshot:
                    is_screenshot = True
                    # Use the current filename from the Screenshot record, which could be updated during refresh
                    thumbnail_url = f"/screenshots/{screenshot.filename}?cropped=true"
                    current_filename = screenshot.filename  # Update to the current screenshot filename
                
                event_list.append({
                    "id": ev.id,  # same series id
                    "title": f"{ev.filename} (Recurring)",  # Mark as recurring in title
                    "start": occurrence.isoformat(),  # FIXED: Use standard ISO format without sep parameter
                    "device": ev.device,
                    "deviceName": device_name,
                    "filename": ev.filename,
                    "recurrence": ev.recurrence,
                    "series": True,
                    "backgroundColor": device_color,
                    "borderColor": device_color,
                    "textColor": "#ffffff",
                    "classNames": ["recurring-event"],  # Add a class for styling
                    "extendedProps": {
                        "thumbnail": thumbnail_url,
                        "isRecurring": True,  # Flag for frontend to identify recurring events
                        "isScreenshot": is_screenshot,
                        "refreshScreenshot": ev.refresh_screenshot,
                        "currentFilename": current_filename
                    }
                })
                if rec == "daily":
                    occurrence += datetime.timedelta(days=1)
                elif rec == "weekly":
                    occurrence += datetime.timedelta(weeks=1)
                elif rec == "monthly":
                    occurrence += datetime.timedelta(days=30)
                else:
                    break
    return jsonify(event_list)

@schedule_bp.route('/schedule/add', methods=['POST'])
def add_event():
    data = request.get_json()
    datetime_str = data.get("datetime")
    device = data.get("device")
    filename = data.get("filename")
    recurrence = data.get("recurrence", "none")
    refresh_screenshot = data.get("refresh_screenshot", False)
    timezone_offset = data.get("timezone_offset", 0)  # Minutes offset from UTC
    
    print(f"Adding event with datetime: {datetime_str}, timezone offset: {timezone_offset}, refresh_screenshot: {refresh_screenshot}")
    if not (datetime_str and device and filename):
        return jsonify({"status": "error", "message": "Missing parameters"}), 400
    try:
        # FIXED: Use a more robust regex method to detect timezone information
        # This ensures we only detect timezone info that appears after the 'T' in the ISO format
        import re
        has_timezone = bool(re.search(r'(?<=T)[0-9:]+(?:[+-][0-9]{2}:[0-9]{2}|Z)$', datetime_str))
        
        print(f"Datetime string: {datetime_str}, Has timezone: {has_timezone}")
        
        # FIXED: Handle datetime parsing differently based on whether it has timezone info
        if 'Z' in datetime_str:
            # Handle UTC timezone indicator ('Z')
            dt = datetime.datetime.fromisoformat(datetime_str.replace('Z', '+00:00'))
            print(f"Parsed UTC datetime with Z: {dt}")
        else:
            # Parse the datetime string
            dt = datetime.datetime.fromisoformat(datetime_str)
            
            # IMPORTANT: We're now ignoring timezone offsets completely
            # We'll use the time exactly as entered, in the Docker container's timezone
            print(f"Using datetime exactly as entered: {dt}")
        
        # Ensure the datetime is timezone-aware and in Europe/Copenhagen
        import pytz
        copenhagen_tz = pytz.timezone('Europe/Copenhagen')
        
        if dt.tzinfo is None:
            # If the datetime is naive (no timezone info), explicitly set it to Copenhagen timezone
            dt = copenhagen_tz.localize(dt)
        else:
            # If the datetime already has timezone info, convert it to Copenhagen timezone
            dt = dt.astimezone(copenhagen_tz)
            
        # FIXED: Convert to proper UTC ISO8601 string for FullCalendar
        # This ensures FullCalendar can automatically convert it to the client's local time
        formatted_dt_str = dt.isoformat()
        print(f"Parsed datetime: {datetime_str} -> {formatted_dt_str}")
    except Exception as e:
        return jsonify({"status": "error", "message": f"Invalid datetime format: {str(e)}"}), 400
    new_event = ScheduleEvent(
        filename=filename,
        device=device,
        datetime_str=formatted_dt_str,
        sent=False,
        recurrence=recurrence,
        refresh_screenshot=refresh_screenshot
    )
    db.session.add(new_event)
    db.session.commit()
    
    # Schedule the event using the scheduler from scheduler.py
    # This will be picked up by the dedicated scheduler process
    # Use a unique job ID to avoid conflicts and make scheduling faster
    job_id = f"event_{new_event.id}"
    try:
        scheduler.add_job(
            send_scheduled_image,
            'date',
            run_date=dt,
            args=[new_event.id],
            id=job_id,
            replace_existing=True,
            misfire_grace_time=3600  # Allow misfires up to 1 hour
        )
    except Exception as e:
        print(f"Warning: Could not schedule job: {str(e)}")
        # Continue anyway - the scheduler process will pick up the event later
    
    return jsonify({"status": "success", "event": {
        "id": new_event.id,
        "filename": new_event.filename,
        "datetime": new_event.datetime_str,
        "recurrence": recurrence
    }})

@schedule_bp.route('/schedule/remove/<int:event_id>', methods=['POST'])
def remove_event(event_id):
    ev = ScheduleEvent.query.get(event_id)
    if ev:
        db.session.delete(ev)
        db.session.commit()
    return jsonify({"status": "success"})

@schedule_bp.route('/schedule/update', methods=['POST'])
def update_event():
    data = request.get_json()
    event_id = data.get("event_id")
    new_datetime = data.get("datetime")
    timezone_offset = data.get("timezone_offset", 0)  # Minutes offset from UTC
    
    # Optional parameters for full event update
    device = data.get("device")
    filename = data.get("filename")
    recurrence = data.get("recurrence")
    refresh_screenshot = data.get("refresh_screenshot")
    
    print(f"Updating event {event_id} to {new_datetime} with timezone offset {timezone_offset}")
    
    if not (event_id and new_datetime):
        return jsonify({"status": "error", "message": "Missing parameters"}), 400
    
    try:
        # Convert event_id to integer if it's a string
        if isinstance(event_id, str):
            event_id = int(event_id)
            
        ev = ScheduleEvent.query.get(event_id)
        if not ev:
            return jsonify({"status": "error", "message": "Event not found"}), 404
            
        # Format the datetime string properly
        try:
            # IMPROVED: Use a more robust regex method to detect timezone information
            # This ensures we only detect timezone info that appears after the 'T' in the ISO format
            import re
            has_timezone = bool(re.search(r'(?<=T)[0-9:]+(?:[+-][0-9]{2}:[0-9]{2}|Z)$', new_datetime))
            
            print(f"Datetime string: {new_datetime}, Has timezone: {has_timezone}")
            
            if 'Z' in new_datetime:
                # Handle UTC timezone indicator ('Z')
                dt = datetime.datetime.fromisoformat(new_datetime.replace('Z', '+00:00'))
                print(f"Parsed UTC datetime with Z: {dt}")
            else:
                # Parse the datetime string
                dt = datetime.datetime.fromisoformat(new_datetime)
                
                # IMPORTANT: We're now ignoring timezone offsets completely
                # We'll use the time exactly as entered, in the Docker container's timezone
                print(f"Using datetime exactly as entered: {dt}")
            
            # Ensure the datetime is timezone-aware and in Europe/Copenhagen
            import pytz
            copenhagen_tz = pytz.timezone('Europe/Copenhagen')
            
            if dt.tzinfo is None:
                # If the datetime is naive (no timezone info), explicitly set it to Copenhagen timezone
                dt = copenhagen_tz.localize(dt)
            else:
                # If the datetime already has timezone info, convert it to Copenhagen timezone
                dt = dt.astimezone(copenhagen_tz)
                
            # Convert to proper UTC ISO8601 string for FullCalendar
            formatted_dt_str = dt.isoformat()
            print(f"Parsed datetime: {new_datetime} -> {formatted_dt_str}")
        except Exception as e:
            return jsonify({"status": "error", "message": f"Invalid datetime format: {str(e)}"}), 400
            
        # Update the event in the database
        ev.datetime_str = formatted_dt_str
        
        # Update other fields if provided
        if device:
            ev.device = device
        if filename:
            ev.filename = filename
        if recurrence:
            ev.recurrence = recurrence
        if refresh_screenshot is not None:
            ev.refresh_screenshot = refresh_screenshot
        
        # Make sure changes are committed to the database
        try:
            db.session.commit()
            print(f"Successfully updated event {event_id} in database to {formatted_dt_str}")
        except Exception as e:
            db.session.rollback()
            print(f"Error committing changes to database: {str(e)}")
            raise
        
        # If this is a recurring event, we need to update the base date
        if ev.recurrence.lower() != "none":
            # Also update the scheduler if needed
            try:
                scheduler.reschedule_job(
                    job_id=str(event_id),
                    trigger='date',
                    run_date=dt
                )
            except Exception as e:
                print(f"Warning: Could not reschedule job: {str(e)}")
        
        # Log the update for debugging
        print(f"Updated event {event_id} to {formatted_dt_str}")
        
        return jsonify({"status": "success"})
    except Exception as e:
        db.session.rollback()
        return jsonify({"status": "error", "message": f"Error updating event: {str(e)}"}), 500

@schedule_bp.route('/schedule/skip/<int:event_id>', methods=['POST'])
def skip_event_occurrence(event_id):
    ev = ScheduleEvent.query.get(event_id)
    if not ev:
        return jsonify({"status": "error", "message": "Event not found"}), 404
    if ev.recurrence.lower() == "none":
        return jsonify({"status": "error", "message": "Not a recurring event"}), 400
    try:
        import pytz
        copenhagen_tz = pytz.timezone('Europe/Copenhagen')
        
        # Parse the datetime string
        dt = datetime.datetime.fromisoformat(ev.datetime_str)
        
        # Ensure the datetime is in Copenhagen timezone
        if dt.tzinfo is None:
            dt = copenhagen_tz.localize(dt)
        else:
            dt = dt.astimezone(copenhagen_tz)
            
        if ev.recurrence.lower() == "daily":
            next_dt = dt + datetime.timedelta(days=1)
        elif ev.recurrence.lower() == "weekly":
            next_dt = dt + datetime.timedelta(weeks=1)
        elif ev.recurrence.lower() == "monthly":
            next_dt = dt + datetime.timedelta(days=30)
        else:
            return jsonify({"status": "error", "message": "Unknown recurrence type"}), 400
        
        # Update the event in the database with the new date
        ev.datetime_str = next_dt.isoformat()
        ev.sent = False
        db.session.commit()
        
        # Schedule the event using the scheduler from scheduler.py with improved configuration
        job_id = f"event_{ev.id}"
        try:
            scheduler.add_job(
                send_scheduled_image,
                'date',
                run_date=next_dt,
                args=[ev.id],
                id=job_id,
                replace_existing=True,
                misfire_grace_time=3600  # Allow misfires up to 1 hour
            )
        except Exception as e:
            print(f"Warning: Could not schedule job: {str(e)}")
            # Continue anyway - the scheduler process will pick up the event later
        
        return jsonify({"status": "success", "message": "Occurrence skipped"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500
