# Use an official Python image
FROM python:3.13.2-slim

# Install curl
RUN apt-get update && apt-get install -y curl

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files into the container
COPY . .

# Expose port 5001
EXPOSE 5001

# Run the Flask application
CMD ["python", "app.py"]