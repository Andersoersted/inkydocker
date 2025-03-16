# InkyDocker

InkyDocker is a powerful web application for managing and displaying images on e-ink displays. It provides a comprehensive solution for organizing your image collection, automatically tagging images using AI, capturing screenshots from websites, and scheduling image displays on your e-ink devices.

![InkyDocker Screenshot](https://via.placeholder.com/800x450.png?text=InkyDocker+Screenshot)

## Features

### 🖼️ Image Management

- **Image Gallery**: Browse, search, and manage your image collection with an intuitive interface
- **Batch Upload**: Upload multiple images at once with drag-and-drop support
- **Image Tagging**: Automatically tag images using AI or add custom tags manually
- **Favorites**: Mark images as favorites for quick access
- **Search**: Find images by tags, filenames, or other metadata

### 🤖 AI-Powered Image Tagging

- **CLIP Models**: Utilize OpenAI's CLIP models for automatic image tagging
- **Zero-Shot Classification**: Advanced image understanding without specific training
- **Customizable Models**: Choose between different models based on your needs:
  - Small, fast models for quick tagging
  - Larger, more accurate models for detailed analysis
- **Adjustable Confidence Thresholds**: Control the precision of automatic tagging

### 📱 E-Ink Display Integration

- **Multi-Device Support**: Connect and manage multiple e-ink displays
- **Device Status Monitoring**: Track which devices are online and what they're displaying
- **Resolution Optimization**: Automatically resize and crop images to match your display's resolution
- **Orientation Support**: Handle both landscape and portrait display orientations
- **Direct Sending**: Send images directly to your e-ink displays with a single click

### 🌐 Screenshot Capture

- **Website Screenshots**: Capture screenshots from any website
- **Scheduled Refreshes**: Automatically refresh screenshots at specified intervals
- **Cropping Tools**: Select specific portions of websites to display
- **Browserless Integration**: Headless browser support for efficient screenshot capture

### ⏱️ Scheduling

- **Timed Displays**: Schedule images to be sent to displays at specific times
- **Recurring Schedules**: Set up daily, weekly, or monthly recurring image displays
- **Calendar View**: Visualize your scheduled displays in a calendar interface
- **Flexible Recurrence**: Configure custom recurrence patterns for your needs

### 🔧 Image Processing

- **Smart Cropping**: Intelligently crop images to fit your display's aspect ratio
- **Manual Cropping**: Fine-tune image crops with an interactive cropping tool
- **Format Conversion**: Automatic conversion of various image formats (HEIC, RAW, etc.) to compatible formats
- **Optimization**: Resize and optimize images for e-ink display

### ⚙️ System Features

- **Docker Deployment**: Easy deployment using Docker and Docker Compose
- **Responsive Design**: Access the application from desktop or mobile devices
- **API Access**: Programmatic access to all functionality
- **Logging**: Comprehensive logging for troubleshooting

## Getting Started

### Prerequisites

- Docker and Docker Compose
- E-ink displays with network connectivity
- Sufficient storage space for your image collection

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/inkydocker.git
   cd inkydocker
   ```

2. Configure your environment:
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

3. Build and start the containers:
   ```bash
   docker-compose up -d
   ```

4. Access the web interface at `http://localhost:5001`

## Configuration

### E-Ink Display Setup

1. Navigate to the Settings page
2. Click "Add Device"
3. Enter your device's IP address, display name, and specifications
4. Click "Save" to add the device

### AI Tagging Configuration

1. Go to Settings > AI Tagging
2. Choose your preferred model (small, medium, or large)
3. Adjust the confidence threshold as needed
4. Enable or disable Zero-Shot tagging

### Screenshot Configuration

1. Navigate to Settings > Screenshots
2. Configure the browserless settings
3. Set default refresh intervals if desired

## Usage

### Uploading Images

1. Go to the main Gallery page
2. Click "Upload" or drag and drop images onto the upload area
3. Wait for the upload and automatic tagging to complete

### Sending Images to Displays

1. Browse your gallery and find an image you want to display
2. Select the target e-ink display from the dropdown
3. Click "Send" on the image
4. The image will be processed and sent to the display

### Capturing Screenshots

1. Navigate to the Screenshots page
2. Enter the URL of the website you want to capture
3. Click "Capture"
4. Once captured, you can crop and send the screenshot like any other image

### Scheduling Displays

1. Go to the Schedule page
2. Click on a date/time slot
3. Select an image and target display
4. Configure recurrence if desired
5. Save the schedule

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [OpenAI CLIP](https://github.com/openai/CLIP) for image understanding
- [Flask](https://flask.palletsprojects.com/) web framework
- [Celery](https://docs.celeryproject.org/) for task processing
- [Puppeteer](https://pptr.dev/) for screenshot capture