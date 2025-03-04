# InkyDocker

InkyDocker is a Flask-based web application designed to manage and display images on e-ink displays. It allows users to upload images, schedule them for display, manage display settings, and monitor device metrics.

## Features

*   **Image Gallery**: Upload, crop, and manage images for display.
*   **Scheduled Display**: Schedule images to be displayed on e-ink displays at specific times.
*   **Device Management**: Configure and manage connected e-ink displays, including setting orientation and fetching display information.
*   **AI Settings**: Configure AI settings, such as OpenAI API key and Ollama model settings.
*   **Device Monitoring**: Monitor real-time device metrics such as CPU usage, memory usage, and disk usage.

## API Endpoints

The application exposes the following API endpoints for interacting with e-ink displays:

*   `POST /send_image`: Upload an image to display on the e-ink screen. Example:

    ```bash
    curl -F "file=@path/to/your/image.jpg" http://<IP_ADDRESS>/send_image
    ```

*   `POST /set_orientation`: Set the display orientation (choose either "horizontal" or "vertical"). Example:

    ```bash
    curl -X POST -d "orientation=vertical" http://<IP_ADDRESS>/set_orientation
    ```

*   `GET /display_info`: Retrieve display information in JSON format. Example:

    ```bash
    curl http://<IP_ADDRESS>/display_info
    ```

*   `POST /system_update`: Trigger a system update and upgrade (which will reboot the device). Example:

    ```bash
    curl -X POST http://<IP_ADDRESS>/system_update
    ```

*   `POST /backup`: Create a compressed backup of the SD card and download it. Example:

    ```bash
    curl -X POST http://<IP_ADDRESS>/backup --output backup.img.gz
    ```

*   `POST /update`: Perform a Git pull to update the application and reboot the device. Example:

    ```bash
    curl -X POST http://<IP_ADDRESS>/update
    ```

*   `GET /stream`: Connect to the SSE stream to receive real-time system metrics. Example:

    ```bash
    curl http://<IP_ADDRESS>/stream
    ```

## Requirements

*   Python 3.6+
*   Flask
*   Flask-Migrate
*   Pillow
*   Other dependencies listed in `requirements.txt`

## Installation

1.  Clone the repository:

    ```bash
    git clone <repository_url>
    cd inkydocker
    ```

2.  Create a virtual environment:

    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

4.  Configure the application:

    *   Create a `config.py` file based on the `config.example.py` file.
    *   Set the necessary environment variables, such as the database URI and API keys.

5.  Run database migrations:

    ```bash
    flask db upgrade
    ```

6.  Run the application:

    ```bash
    python app.py
    ```

## Usage

1.  Access the web application in your browser.
2.  Configure your e-ink displays in the Settings page.
3.  Upload images to the Gallery.
4.  Schedule images for display on the Schedule page.
5.  Monitor device metrics on the Settings page.

## Contributing

Contributions are welcome! Please submit a pull request with your changes.

## License

[Specify the license for your project]