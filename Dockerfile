# Use the official Python image from the slim variant
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1  # Prevent Python from writing .pyc files
ENV PYTHONUNBUFFERED 1  # Prevent Python output buffering
ENV PYTHONPATH=/app 

# Set the working directory inside the container
WORKDIR /app

# Copy only the requirements file first to leverage Docker's caching mechanism
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy the entire application to the container
COPY . .

# Install pytest for running tests
RUN pip install pytest

# Run tests inside the Docker build process
RUN pytest tests/

# Expose the port FastAPI will be running on
EXPOSE 5050

# Set up the command to run the FastAPI app using Uvicorn
CMD ["uvicorn", "api.fastapi_app:app", "--host", "0.0.0.0", "--port", "5050"]
