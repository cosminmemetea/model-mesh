# Model-Mesh

**Model-Mesh** is a sentiment analysis web application powered by FastAPI. It provides a simple API to predict the sentiment of text (positive or negative) using models like BERT. The app also includes a frontend UI that allows users to input text and view sentiment predictions in a user-friendly way. 
We are using **DVC** for version controlling datasets and models, along with **AWS S3** as the remote storage.
Overall this project includes an API, model manager, and adapters for managing and running various machine learning models.


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

   For easy build and redeployment in the docker container:
   
   ```
   docker-compose up --build modelmesh
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


## Setup Instructions

### 1. Prerequisites

Before starting, ensure you have the following installed:

- Python 3.x
- Pip (Python package manager)
- AWS CLI (for managing AWS resources)
- DVC (for data version control)
- AWS account and access to S3

### 2. AWS S3 Configuration

1. **Create an S3 Bucket**:
   - Log in to your AWS account and navigate to the [S3 Console](https://s3.console.aws.amazon.com/s3).
   - Create a new S3 bucket (ensure it has a unique name).

2. **Configure AWS CLI**:
   - Install the AWS CLI if you haven't already:
     ```
     pip install awscli
     ```
   - Configure your AWS credentials:
     ```
     aws configure
     ```
     You’ll be prompted to enter your:
     - **AWS Access Key ID**
     - **AWS Secret Access Key**
     - **Default region name**
     - **Default output format** (e.g., `json`)

### 3. Installing and Setting Up DVC with S3

1. **Install DVC** with S3 support:
   ```
   pip install "dvc[s3]"
   ```

2. **Initialize DVC in the project: Navigate to your project directory and initialize DVC:**
   ```
   dvc init
   ```

3. **Add data files or directories to DVC: Organize your data into raw/ and processed/ folders, and then track them with DVC:**
   ```
   dvc add data/raw/ data/processed/
   ```
4. **Configure the S3 remote storage for DVC: Replace your-bucket-name and path/to/folder with your actual S3 bucket name and folder path:**
   ```
   dvc remote add -d s3remote s3://your-bucket-name/path/to/folder
   ```
5. **Push the data to S3: Once you've added data files to DVC, push them to the S3 bucket:**
   ```
   dvc push
   ```
### 4. DVC Commands Overview

- Track new data: After adding new data files or directories, use:
    ```
    dvc add <path_to_data>
    ```
- Check DVC status: View the state of your tracked files and their storage locations:
    ```
    dvc status
    ```
- Pull data from S3: If you or another collaborator needs to retrieve the data tracked in DVC from S3:
    ```
    dvc pull
    ```

- Push data to S3: After adding or modifying datasets, push them to S3:
    ```
    dvc push
    ```
- Check DVC remotes: List all configured remotes:
    ```
    dvc remote list
    ```