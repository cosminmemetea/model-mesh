# Dockerfile
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Copy the requirements.txt file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the project files
COPY . .

# Expose the correct port
EXPOSE 5050

# Run the FastAPI app
CMD ["uvicorn", "api.fastapi_app:app", "--host", "0.0.0.0", "--port", "5050"]
