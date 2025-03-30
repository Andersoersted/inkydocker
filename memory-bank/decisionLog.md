# Decision Log

This file records architectural and implementation decisions using a list format.
2025-03-29 11:14:56 - Log of updates made.

*

## Decision

*

## Rationale 

*

## Implementation Details


## Decision

* [2025-03-29 13:16:37] - Add detailed logging to the `/send_image` route handler in `routes/image_routes.py`.

## Rationale

* The application fails to create the temporary processed image file before dispatching the Celery task for sending to the e-ink display.
* Existing logs are insufficient to pinpoint the exact failure point within the image processing steps (loading, cropping, resizing, saving).
* Detailed logging will trace the execution flow and identify the specific operation causing the error.

## Implementation Details

* Add `current_app.logger.debug()` statements at key points within the `send_image` function's try/except block.
* Ensure the main exception handler logs the full traceback.
* Requires switching to Code mode for implementation.
* Requires rebuilding and redeploying the Docker container after code changes.

*