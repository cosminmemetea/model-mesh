# Model-Mesh

**Model-Mesh** is a sentiment analysis web application powered by FastAPI. It provides a simple API to predict the sentiment of text (positive or negative) using models like BERT. The app also includes a frontend UI that allows users to input text and view sentiment predictions in a user-friendly way.

### Features:
- Sentiment analysis with BERT model integration.
- FastAPI for serving the API.
- Simple, minimalist frontend for user interaction.
- Dockerized for easy deployment.

### Deployed
The application is deployed as a web service on Render: [Model-Mesh on Render](https://dashboard.render.com).

## Getting Started

### Prerequisites
- Docker installed on your system.

### Run Locally with Docker

1. **Clone the Repository**:

   ```
   git clone https://github.com/your-username/model-mesh.git
   cd model-mesh
   ```
2. **Build the Docker Image**:

   Build the Docker image using the Dockerfile:

   ```
   docker build -t model-mesh:latest . 
   ```
3. **Run the Docker Container**:

   Run the container on port 5050 (or one of your choice, but make sure you update docker related files and app.py):

   ```
   docker run -d -p 5050:5050 model-mesh:latest
   ```
4. **Access the Application**:

    API (FastAPI): Once the container is running, you can access the FastAPI Swagger UI at:

    ```
    http://localhost:5050/docs
    ```

    Frontend UI: To check the mood prediction UI, open your browser and go to:

    ```
    http://localhost:5050/
    ```

## Technologies Used

- FastAPI: For building the API.
- Docker: For containerization.
- BERT: Pre-trained model for sentiment analysis.
- Frontend: Minimal HTML/CSS/JS for the user interface.