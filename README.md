# InkyDocker

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python 3.6+](https://img.shields.io/badge/python-3.6+-blue.svg)](https://www.python.org/downloads/)
[![Flask](https://img.shields.io/badge/Flask-2.0+-green.svg)](https://flask.palletsprojects.com/)
[![Docker](https://img.shields.io/badge/Docker-Supported-blue.svg)](https://www.docker.com/)

InkyDocker is a Flask-based web application designed to manage and display images on e-ink displays. It allows users to upload images, schedule them for display, manage display settings, and monitor device metrics.

## Table of Contents

- [Features](#features)
- [Screenshots](#screenshots)
- [Requirements](#requirements)
- [Installation](#installation)
  - [Standard Installation](#standard-installation)
  - [Docker Installation](#docker-installation)
- [Usage](#usage)
- [API Endpoints](#api-endpoints)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

## Features

- **Image Gallery**: Upload, crop, and manage images for display.
- **Scheduled Display**: Schedule images to be displayed on e-ink displays at specific times.
- **Device Management**: Configure and manage connected e-ink displays, including setting orientation and fetching display information.
- **AI Settings**: Configure AI settings, such as OpenAI API key and Ollama model settings.
- **Device Monitoring**: Monitor real-time device metrics such as CPU usage, memory usage, and disk usage.

## Screenshots

*[Add screenshots of your application here]*

## Requirements

- Python 3.6+
- Flask
- Flask-Migrate
- Pillow
- SQLAlchemy
- Celery
- Redis
- Other dependencies listed in `requirements.txt`

## Installation

### Standard Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/inkydocker.git
   cd inkydocker
   ```

2. Create a virtual environment:

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Configure the application:

   - Create a `config.py` file based on the `config.example.py` file.
   - Set the necessary environment variables, such as the database URI and API keys.

5. Run database migrations:

   ```bash
   flask db upgrade
   ```

6. Run the application:

   ```bash
   python app.py
   ```

### Docker Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/inkydocker.git
   cd inkydocker
   ```

2. Build and start the Docker container:

   ```bash
   docker-compose up -d
   ```

   This will:
   - Build the Docker image
   - Start the container
   - Map port 5001 to your host
   - Mount the data and images directories as volumes
   - Set the container to restart automatically unless stopped manually

3. Access the application at `http://localhost:5001`

## Usage

1. Access the web application in your browser at `http://localhost:5001`.
2. Configure your e-ink displays in the Settings page.
3. Upload images to the Gallery.
4. Schedule images for display on the Schedule page.
5. Monitor device metrics on the Settings page.

## API Endpoints

The application exposes the following API endpoints for interacting with e-ink displays:

- `POST /send_image`: Upload an image to display on the e-ink screen.

  ```bash
  curl -F "file=@path/to/your/image.jpg" http://<IP_ADDRESS>/send_image
  ```

- `POST /set_orientation`: Set the display orientation (choose either "horizontal" or "vertical").

  ```bash
  curl -X POST -d "orientation=vertical" http://<IP_ADDRESS>/set_orientation
  ```

- `GET /display_info`: Retrieve display information in JSON format.

  ```bash
  curl http://<IP_ADDRESS>/display_info
  ```

- `POST /system_update`: Trigger a system update and upgrade (which will reboot the device).

  ```bash
  curl -X POST http://<IP_ADDRESS>/system_update
  ```

- `POST /backup`: Create a compressed backup of the SD card and download it.

  ```bash
  curl -X POST http://<IP_ADDRESS>/backup --output backup.img.gz
  ```

- `POST /update`: Perform a Git pull to update the application and reboot the device.

  ```bash
  curl -X POST http://<IP_ADDRESS>/update
  ```

- `GET /stream`: Connect to the SSE stream to receive real-time system metrics.

  ```bash
  curl http://<IP_ADDRESS>/stream
  ```

## Project Structure

```
inkydocker/
├── app.py                  # Main Flask application
├── config.py               # Configuration settings
├── models.py               # Database models
├── scheduler.py            # Scheduling functionality
├── tasks.py                # Celery tasks
├── requirements.txt        # Python dependencies
├── Dockerfile              # Docker configuration
├── docker-compose.yml      # Docker Compose configuration
├── entrypoint.sh           # Docker entrypoint script
├── supervisord.conf        # Supervisor configuration
├── routes/                 # API routes
│   ├── __init__.py
│   ├── image_routes.py
│   ├── device_routes.py
│   ├── schedule_routes.py
│   ├── settings_routes.py
│   └── ai_tagging_routes.py
├── templates/              # HTML templates
│   ├── base.html
│   ├── index.html
│   ├── schedule.html
│   └── settings.html
├── static/                 # Static assets
│   └── style.css
└── utils/                  # Utility functions
    ├── __init__.py
    ├── image_helpers.py
    └── crop_helpers.py
```

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/your-feature`)
3. Make your changes
4. Commit your changes (`git commit -m 'Add some feature'`)
5. Push to the branch (`git push origin feature/your-feature`)
6. Open a Pull Request

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

The GNU GPL is a copyleft license that ensures that the software remains free and open source. It requires that any derivative works or modifications to the code must also be released under the same license terms.

For more information about the GNU GPL v3.0, visit [https://www.gnu.org/licenses/gpl-3.0.en.html](https://www.gnu.org/licenses/gpl-3.0.en.html).