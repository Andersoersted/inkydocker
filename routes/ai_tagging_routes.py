from flask import Blueprint, request, jsonify, current_app
import os
from tasks import (
    get_image_embedding,
    generate_tags_and_description,
    reembed_image,
    bulk_tag_images,
    BULK_PROGRESS
)
from models import db, ImageDB
from PIL import Image

ai_bp = Blueprint("ai_tagging", __name__)

@ai_bp.route("/api/ai_tag_image", methods=["POST"])
def ai_tag_image():
    data = request.get_json()
    filename = data.get("filename")
    if not filename:
        return jsonify({"status": "error", "message": "Filename is required"}), 400

    image_folder = current_app.config.get("IMAGE_FOLDER", "./images")
    image_path = os.path.join(image_folder, filename)
    if not os.path.exists(image_path):
        return jsonify({"status": "error", "message": "Image file not found"}), 404

    image_embedding = get_image_embedding(image_path)
    if image_embedding is None:
        return jsonify({"status": "error", "message": "Failed to get embedding"}), 500

    tags, description = generate_tags_and_description(image_embedding)
    # Update the ImageDB record with generated tags
    image_record = ImageDB.query.filter_by(filename=filename).first()
    if image_record:
        image_record.tags = ", ".join(tags)
        db.session.commit()
    else:
        image_record = ImageDB(filename=filename, tags=", ".join(tags))
        db.session.add(image_record)
        db.session.commit()

    return jsonify({
        "status": "success",
        "filename": filename,
        "tags": tags
    }), 200

@ai_bp.route("/api/search_images", methods=["GET"])
def search_images():
    q = request.args.get("q", "").strip()
    if not q:
        return jsonify({"status": "error", "message": "Missing query parameter"}), 400
    
    # Get pagination parameters
    try:
        page = int(request.args.get('page', 1))
        per_page = int(request.args.get('per_page', 20))
        
        if page < 1 or per_page < 1 or per_page > 100:
            return jsonify({"status": "error", "message": "Invalid pagination parameters"}), 400
    except ValueError:
        return jsonify({"status": "error", "message": "Invalid pagination parameters"}), 400
    
    # Calculate offset for pagination
    offset = (page - 1) * per_page
    
    # Get images with pagination
    query = ImageDB.query.filter(ImageDB.tags.ilike(f"%{q}%"))
    total_images = query.count()
    images = query.order_by(ImageDB.id.desc()).offset(offset).limit(per_page).all()
    
    results = {
        "ids": [img.filename for img in images],
        "tags": [img.tags for img in images],
        "total": total_images,
        "page": page,
        "per_page": per_page
    }
    return jsonify({"status": "success", "results": results}), 200

@ai_bp.route("/api/get_image_metadata", methods=["GET"])
def get_image_metadata():
    filename = request.args.get("filename", "").strip()
    if not filename:
        return jsonify({"status": "error", "message": "Missing filename"}), 400

    image_folder = current_app.config.get("IMAGE_FOLDER", "./images")
    image_path = os.path.join(image_folder, filename)
    resolution_str = "N/A"
    filesize_str = "N/A"
    if os.path.exists(image_path):
        try:
            size_bytes = os.path.getsize(image_path)
            filesize_mb = size_bytes / (1024 * 1024)
            filesize_str = f"{filesize_mb:.2f} MB"
            with Image.open(image_path) as im:
                w, h = im.size
                resolution_str = f"{w}x{h}"
        except Exception as ex:
            current_app.logger.warning(f"Could not read file info for {filename}: {ex}")

    image_record = ImageDB.query.filter_by(filename=filename).first()
    if image_record:
        tags = [t.strip() for t in image_record.tags.split(",")] if image_record.tags else []
        favorite = image_record.favorite
    else:
        tags = []
        favorite = False

    return jsonify({
        "status": "success",
        "tags": tags,
        "favorite": favorite,
        "resolution": resolution_str,
        "filesize": filesize_str
    }), 200

@ai_bp.route("/api/update_image_metadata", methods=["POST"])
def update_image_metadata():
    data = request.get_json() or {}
    filename = data.get("filename", "").strip()
    new_tags = data.get("tags", [])
    if isinstance(new_tags, list):
        tags_str = ", ".join(new_tags)
    else:
        tags_str = new_tags
    favorite = data.get("favorite", None)
    if not filename:
        return jsonify({"status": "error", "message": "Missing filename"}), 400

    image_record = ImageDB.query.filter_by(filename=filename).first()
    if not image_record:
        return jsonify({"status": "error", "message": "Image not found"}), 404

    image_record.tags = tags_str
    if favorite is not None:
        image_record.favorite = bool(favorite)
    db.session.commit()
    return jsonify({"status": "success"}), 200

@ai_bp.route("/api/reembed_image", methods=["GET"])
def reembed_image_endpoint():
    filename = request.args.get("filename", "").strip()
    if not filename:
        return jsonify({"status": "error", "message": "Missing filename"}), 400
    result = reembed_image(filename)
    return jsonify(result)

@ai_bp.route("/api/reembed_all_images", methods=["GET"])
def reembed_all_images_endpoint():
    task_id = bulk_tag_images.delay()
    if not task_id:
        return jsonify({"status": "error", "message": "No images found"}), 404
    return jsonify({"status": "success", "message": f"Reembedding images in background. Task ID: {task_id}"}), 200

@ai_bp.route("/api/run_openclip/<filename>", methods=["POST"])
def run_openclip_endpoint(filename):
    """
    Re-run tagging for a specific image using the currently selected CLIP model.
    This endpoint is called from the info modal in the gallery.
    """
    if not filename:
        return jsonify({"status": "error", "message": "Missing filename"}), 400
    
    try:
        # Use the reembed_image function to re-tag the image
        result = reembed_image(filename)
        
        # Check if the operation was successful
        if result.get("status") == "success":
            return jsonify({
                "status": "success",
                "message": "Image tagged successfully",
                "tags": result.get("tags", [])
            }), 200
        else:
            return jsonify({
                "status": "error",
                "message": result.get("message", "Unknown error")
            }), 500
    except Exception as e:
        current_app.logger.error(f"Error in run_openclip_endpoint: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500
