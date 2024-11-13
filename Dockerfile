# Dockerfile
# Start from the base image
FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy the requirements file and install Python dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire application code to the container
COPY . /app

# Set PYTHONPATH to include /app
ENV PYTHONPATH="/app"

# Expose the port
EXPOSE 5001

# Command to run the app
CMD ["python", "api/app.py"]

