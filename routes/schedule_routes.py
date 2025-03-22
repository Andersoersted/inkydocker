from flask import Blueprint, request, jsonify, render_template
from models import db, ScheduleEvent, Device, ImageDB, Screenshot
import datetime
import json
import pytz
import re
from tasks import send_scheduled_image
from scheduler import scheduler
from dateutil.rrule import rrule, DAILY, WEEKLY, MONTHLY
from dateutil.relativedelta import relativedelta
import uuid

schedule_bp = Blueprint('schedule', __name__)

# Set the timezone for consistent handling - using Copenhagen timezone
TIMEZONE = pytz.timezone('Europe/Copenhagen')

@schedule_bp.route('/schedule')
def schedule_page():
    """Render the schedule management page."""
    # Get all devices with their info
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
    
    # Create a dictionary of image tags for template
    image_tags = {}
    for img in imgs_db:
        if img.tags:
            image_tags[img.filename] = img.tags
    
    # Get all browserless screenshots
    screenshots = Screenshot.query.all()
    
    return render_template("schedule.html", devices=devices, images=images, 
                          image_tags=image_tags, screenshots=screenshots)

@schedule_bp.route('/schedule/events')
def get_events():
    """Get all events for the calendar view."""
    # Ensure we're getting the latest data from the database
    db.session.expire_all()
    events = ScheduleEvent.query.all()
    
    # Set the visible time range (90 days in past and future)
    now = datetime.datetime.now(TIMEZONE)
    future_horizon = now + datetime.timedelta(days=90)
    past_horizon = now - datetime.timedelta(days=90)
    
    # Create lookup dictionaries for efficiency
    device_lookup = {d.address: {"color": d.color, "friendly_name": d.friendly_name} 
                     for d in Device.query.all()}
    
    screenshot_lookup = {s.filename: s for s in Screenshot.query.all()}
    
    event_list = []
    
    for ev in events:
        # Get device color and friendly name
        device_info = device_lookup.get(ev.device, {"color": "#cccccc", "friendly_name": ev.device})
        device_color = device_info["color"]
        device_name = device_info["friendly_name"]
        
        # Check if the file is a screenshot
        is_screenshot = ev.filename in screenshot_lookup
        thumbnail_url = f"/screenshots/{ev.filename}?cropped=true" if is_screenshot else f"/thumbnail/{ev.filename}"
        
        # Parse the event datetime
        try:
            start_dt = parse_datetime(ev.datetime_str)
        except ValueError:
            # Skip events with invalid datetime
            continue
        
        # Process non-recurring events
        if ev.recurrence.lower() == "none":
            event_list.append({
                "id": ev.id,
                "title": f"{ev.filename}",
                "start": ev.datetime_str,
                "device": ev.device,
                "deviceName": device_name,
                "filename": ev.filename,
                "recurrence": "none",
                "series": False,
                "backgroundColor": device_color,
                "borderColor": device_color,
                "textColor": "#ffffff",
                "extendedProps": {
                    "thumbnail": thumbnail_url,
                    "isScreenshot": is_screenshot,
                    "refreshScreenshot": ev.refresh_screenshot,
                    "currentFilename": ev.filename
                }
            })
        else:
            # For recurring events, generate occurrences based on the pattern
            recurrence_details = ev.get_recurrence_details()
            
            # Ensure we only generate occurrences from event creation date onwards
            start_from = recurrence_details.get('start_from') or ev.created_at.isoformat()
            earliest_allowed = parse_datetime(start_from)
            
            # Generate occurrences based on recurrence type
            occurrences = generate_occurrences(
                event=ev,
                start_dt=start_dt,
                earliest_allowed=earliest_allowed,
                past_horizon=past_horizon,
                future_horizon=future_horizon
            )
            
            # Add each occurrence to the event list
            for occurrence in occurrences:
                event_list.append({
                    "id": f"{ev.id}_{occurrence.isoformat()}",  # Unique ID for each occurrence
                    "title": f"{ev.filename} (Recurring)",
                    "start": occurrence.isoformat(),
                    "device": ev.device,
                    "deviceName": device_name,
                    "filename": ev.filename,
                    "recurrence": ev.recurrence,
                    "series": True,
                    "backgroundColor": device_color,
                    "borderColor": device_color,
                    "textColor": "#ffffff",
                    "classNames": ["recurring-event"],
                    "extendedProps": {
                        "parentEventId": ev.id,  # Reference to the parent recurring event
                        "thumbnail": thumbnail_url,
                        "isRecurring": True,
                        "isScreenshot": is_screenshot,
                        "refreshScreenshot": ev.refresh_screenshot,
                        "currentFilename": ev.filename
                    }
                })
    
    return jsonify(event_list)

def parse_datetime(dt_string):
    """Parse a datetime string with proper error handling."""
    if 'Z' in dt_string:
        # Convert Z (UTC) format to +00:00 format that fromisoformat can handle
        dt_string = dt_string.replace('Z', '+00:00')
    
    dt = datetime.datetime.fromisoformat(dt_string)
    
    # Ensure the datetime is timezone-aware and in the application timezone
    if dt.tzinfo is None:
        dt = TIMEZONE.localize(dt)
    else:
        dt = dt.astimezone(TIMEZONE)
        
    return dt

def generate_occurrences(event, start_dt, earliest_allowed, past_horizon, future_horizon):
    """Generate occurrences for a recurring event."""
    occurrences = []
    rec_type = event.recurrence.lower()
    details = event.get_recurrence_details()
    
    if rec_type == "daily":
        # Process daily recurrence
        interval = details.get("interval", 1)
        weekdays = details.get("weekdays", [])  # Optional specific weekdays
        
        # Create a sequence of dates
        dates = []
        current = start_dt
        
        # If we have specific weekdays, use the dateutil rrule
        if weekdays:
            # Convert weekdays to dateutil format (0=Monday in our UI, but 0=Monday in dateutil)
            byweekday = [int(day) for day in weekdays]
            
            # Generate occurrences using rrule
            rule = rrule(
                DAILY,
                interval=interval,
                dtstart=max(start_dt, earliest_allowed),
                until=future_horizon,
                byweekday=byweekday
            )
            
            # Convert to list of datetimes
            dates = list(rule)
        else:
            # Simple daily interval without day constraints
            while current <= future_horizon:
                if current >= earliest_allowed:
                    dates.append(current)
                current += datetime.timedelta(days=interval)
        
        # Filter to the range we want - ONLY include events after earliest_allowed
        # Remove past_horizon from the filter to prevent showing events in the past
        occurrences = [d for d in dates
                     if d >= earliest_allowed and d <= future_horizon]
                
    elif rec_type == "weekly":
        # Process weekly recurrence
        interval = details.get("interval", 1)
        weekdays = details.get("weekdays", [])
        
        if not weekdays:
            # If no weekdays specified, use the day of week from the start date
            weekdays = [start_dt.weekday()]
        
        # Generate occurrences using rrule
        rule = rrule(
            WEEKLY,
            interval=interval,
            dtstart=max(start_dt, earliest_allowed),
            until=future_horizon,
            byweekday=[int(day) for day in weekdays]
        )
        
        # Filter to the range we want - ONLY include events after earliest_allowed
        # Remove past_horizon from the filter to prevent showing events in the past
        occurrences = [d for d in rule
                     if d >= earliest_allowed and d <= future_horizon]
        
    elif rec_type == "monthly":
        # Process monthly recurrence
        interval = details.get("interval", 1)
        rec_type = details.get("type", "day")
        
        if rec_type == "day":
            # Monthly by day of month (e.g., 15th of each month)
            day = start_dt.day
            
            # Generate dates using a loop
            current = start_dt
            while current <= future_horizon:
                # Only check against earliest_allowed to prevent past events
                if current >= earliest_allowed:
                    occurrences.append(current)
                # Move to next month with relative delta
                current += relativedelta(months=interval)
                
                # Ensure we keep the same day of month, handling edge cases
                day_in_month = min(day, (current + relativedelta(months=1, day=1) - relativedelta(days=1)).day)
                current = current.replace(day=day_in_month)
                
        elif rec_type == "position":
            # Monthly by position (e.g., 2nd Monday)
            position = details.get("position", 1)
            weekday = details.get("day", 0)
            
            # This is more complex - use a custom function to calculate
            current = start_dt
            while current <= future_horizon:
                # Only check against earliest_allowed to prevent past events
                if current >= earliest_allowed:
                    occurrences.append(current)
                
                # Move to next occurrence using relativedelta for complex cases
                if position > 0:
                    # For 1st, 2nd, 3rd, 4th occurrences
                    current += relativedelta(months=interval)
                    current = get_nth_weekday_of_month(current.year, current.month, weekday, position)
                else:
                    # For last occurrence
                    current += relativedelta(months=interval)
                    current = get_last_weekday_of_month(current.year, current.month, weekday)
    
    return occurrences

def get_nth_weekday_of_month(year, month, weekday, n):
    """Get the nth occurrence of a weekday in a month."""
    # Start from the first day of the month
    date = datetime.datetime(year, month, 1, tzinfo=TIMEZONE.zone)
    
    # Find the first occurrence of the weekday
    days_to_add = (weekday - date.weekday()) % 7
    date += datetime.timedelta(days=days_to_add)
    
    # Add (n-1) weeks to get to the nth occurrence
    date += datetime.timedelta(weeks=(n-1))
    
    # Check if we're still in the same month
    if date.month != month:
        return None
    
    return date

def get_last_weekday_of_month(year, month, weekday):
    """Get the last occurrence of a weekday in a month."""
    # Start from the last day of the month
    next_month = month + 1 if month < 12 else 1
    next_month_year = year if month < 12 else year + 1
    last_day = (datetime.datetime(next_month_year, next_month, 1) - 
                datetime.timedelta(days=1)).day
    date = datetime.datetime(year, month, last_day, tzinfo=TIMEZONE.zone)
    
    # Find the last occurrence of the weekday, going backwards
    days_to_subtract = (date.weekday() - weekday) % 7
    date -= datetime.timedelta(days=days_to_subtract)
    
    # Check if we're still in the same month
    if date.month != month:
        return None
    
    return date

@schedule_bp.route('/schedule/add', methods=['POST'])
def add_event():
    """Add a new scheduled event."""
    data = request.get_json()
    datetime_str = data.get("datetime")
    device = data.get("device")
    filename = data.get("filename")
    recurrence = data.get("recurrence", "none")
    refresh_screenshot = data.get("refresh_screenshot", False)
    recurrence_details = data.get("recurrence_details", {})
    
    if not (datetime_str and device and filename):
        return jsonify({"status": "error", "message": "Missing parameters"}), 400
    
    try:
        # Parse the datetime
        dt = parse_datetime(datetime_str)
        formatted_dt_str = dt.isoformat()
        
        # Create the event
        new_event = ScheduleEvent(
            filename=filename,
            device=device,
            datetime_str=formatted_dt_str,
            sent=False,
            recurrence=recurrence,
            refresh_screenshot=refresh_screenshot,
            created_at=datetime.datetime.now(TIMEZONE)
        )
        
        # Store recurrence details
        if recurrence != "none" and recurrence_details:
            # Always add start_from, and FORCE it to be today to ensure recurring events start from now
            # This is important even if the user provided a different start_from
            recurrence_details['start_from'] = datetime.datetime.now(TIMEZONE).isoformat()
            new_event.set_recurrence_details(recurrence_details)
        
        # Save to database
        db.session.add(new_event)
        db.session.commit()
        
        # Schedule the event
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
        
        return jsonify({
            "status": "success", 
            "event": {
                "id": new_event.id,
                "filename": new_event.filename,
                "datetime": new_event.datetime_str,
                "recurrence": recurrence
            }
        })
    except Exception as e:
        return jsonify({"status": "error", "message": f"Error creating event: {str(e)}"}), 400

@schedule_bp.route('/schedule/remove/<int:event_id>', methods=['POST'])
def remove_event(event_id):
    """Remove a scheduled event."""
    ev = ScheduleEvent.query.get(event_id)
    if ev:
        db.session.delete(ev)
        db.session.commit()
        
        # Also remove from scheduler if it exists
        try:
            scheduler.remove_job(f"event_{event_id}")
        except:
            pass  # Job might not exist, just continue
            
    return jsonify({"status": "success"})

@schedule_bp.route('/schedule/update', methods=['POST'])
def update_event():
    """Update a scheduled event."""
    data = request.get_json()
    event_id = data.get("event_id")
    new_datetime = data.get("datetime")
    
    # Optional parameters for full event update
    device = data.get("device")
    filename = data.get("filename")
    recurrence = data.get("recurrence")
    refresh_screenshot = data.get("refresh_screenshot")
    recurrence_details = data.get("recurrence_details")
    
    if not (event_id and new_datetime):
        return jsonify({"status": "error", "message": "Missing parameters"}), 400
    
    try:
        # Convert event_id to integer if it's a string
        if isinstance(event_id, str):
            event_id = int(event_id)
            
        ev = ScheduleEvent.query.get(event_id)
        if not ev:
            return jsonify({"status": "error", "message": "Event not found"}), 404
            
        # Parse the datetime
        dt = parse_datetime(new_datetime)
        formatted_dt_str = dt.isoformat()
            
        # Update the event
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
            
        # Update recurrence details if provided
        if recurrence_details:
            # Always FORCE start_from to be today, even if user provided a different value
            # This ensures recurring events only start from now, not from the past
            recurrence_details['start_from'] = datetime.datetime.now(TIMEZONE).isoformat()
            ev.set_recurrence_details(recurrence_details)
            
        # Save changes
        db.session.commit()
        
        # Reschedule the job if needed
        job_id = f"event_{event_id}"
        try:
            scheduler.reschedule_job(
                job_id=job_id,
                trigger='date',
                run_date=dt
            )
        except Exception as e:
            print(f"Warning: Could not reschedule job: {str(e)}")
            # Try to add as a new job if reschedule fails
            try:
                scheduler.add_job(
                    send_scheduled_image,
                    'date',
                    run_date=dt,
                    args=[event_id],
                    id=job_id,
                    replace_existing=True,
                    misfire_grace_time=3600
                )
            except Exception as e2:
                print(f"Warning: Could not add job: {str(e2)}")
        
        return jsonify({"status": "success"})
    except Exception as e:
        db.session.rollback()
        return jsonify({"status": "error", "message": f"Error updating event: {str(e)}"}), 400

@schedule_bp.route('/schedule/skip/<int:event_id>', methods=['POST'])
def skip_event(event_id):
    """Skip a single occurrence of a recurring event."""
    # In a simple implementation, we just return success
    # In a more complex system, we could track skipped occurrences
    return jsonify({"status": "success"})

# Migration helper route to update existing event records
@schedule_bp.route('/schedule/migrate', methods=['POST'])
def migrate_events():
    """Migrate existing events to use the new recurrence details format."""
    events = ScheduleEvent.query.all()
    count = 0
    
    for ev in events:
        if ev.recurrence.lower() != "none" and not ev.recurrence_details:
            # Create new recurrence details
            details = {
                "start_from": ev.created_at.isoformat() if ev.created_at else datetime.datetime.now(TIMEZONE).isoformat(),
            }
            
            # Add pattern-specific details
            if ev.recurrence.lower() == "daily":
                details["interval"] = 1
            elif ev.recurrence.lower() == "weekly":
                details["interval"] = 1
                # Default to the day of week from the start date
                try:
                    start_dt = parse_datetime(ev.datetime_str)
                    details["weekdays"] = [start_dt.weekday()]
                except:
                    details["weekdays"] = [0]  # Monday as default
            elif ev.recurrence.lower() == "monthly":
                details["interval"] = 1
                details["type"] = "day"
            
            # Save details
            ev.set_recurrence_details(details)
            count += 1
    
    # Commit all changes
    db.session.commit()
    
    return jsonify({
        "status": "success",
        "message": f"Successfully migrated {count} events"
    })
