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
   git clone https://github.com/cosminmemetea/model-mesh.git
   cd model-mesh
   ```
2. **Build the Docker Image**:

   Build the Docker image using the Dockerfile:

   ```
   docker build -t model-mesh:latest . 
   ```

   For easy build and redeployment in the docker container:
   
   ```
   docker compose up --build modelmesh
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

Prerequisites

Before starting, ensure you have the following installed:

- Python 3.x - https://www.python.org/downloads/release/python-3110/
- Pip (Python package manager)
- AWS CLI (for managing AWS resources)
- DVC (for data version control)
- AWS account and access to S3

### 1 Steps to Set Python 3.11 as Default with pip

**1. Open Your `.zshrc` (or `.bash_profile` for bash):**
```
nano ~/.zshrc   # For zsh users
```

**2. Add the Path to Python 3.11 at the Top:**
Add the following line to your `.zshrc` (or/and `.bash_profile`) to make Python 3.11 and its `pip` the default:

```
export PATH="/usr/local/bin/python3.11:$PATH"
export PATH="/Library/Frameworks/Python.framework/Versions/3.11/bin:$PATH"

```

If the path to Python 3.11 is different, adjust accordingly.

**3. Reload Your Shell Configuration:**
After saving the file, reload the configuration:

```
source ~/.zshrc   # Or/And source ~/.bash_profile
```

**4. Verify the Default Python and Pip:**
After reloading the shell, verify that Python 3.11 and its `pip` are being used:

```
python --version
pip --version
```


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
   pip install "dvc[s3]" # use gdrive instead of s3 for google drive support
   ```

2. **Initialize DVC in the project: Navigate to your project directory and initialize DVC:**

   Optinally 

   Remove DVC Metadata and Configurations
   You can remove all DVC metadata by deleting the .dvc/ folder and dvc.yaml file:

   ```   
   rm -rf .dvc
   rm dvc.yaml
   rm .dvcignore
   ```
   
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
   dvc remote add -d gdriveremote gdrive://<your-google-drive-folder-id>
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
- Unlink Old Remotes

If you have an old DVC remote (like AWS S3), you can remove it by running:

```
dvc remote remove <remote_name>
```

###  5. DVC Files and S3
DVC tracks files by their MD5 hashes for efficient version control. When pushing to S3, files will appear as hashed filenames (e.g., md5/hex/hash), which are mapped to the original filenames in your Git repository. This ensures that no duplicate files are stored and changes are versioned.

### 6. Data Versioning and Sharing
When collaborators clone this repository, they can pull the necessary data using DVC. After setting up DVC and configuring the S3 remote, run:
   ```
   dvc pull
   ```
To version control new data, add the files using dvc add and then commit the changes in Git. Push the updated .dvc files to the repository, and use dvc push to sync the data to S3.
Add the following DVC files to Git after every change:
   ```
   git add data/.gitignore data/raw.dvc data/processed.dvc
   git commit -m "Tracked new data with DVC"
   git push
   ```
Never commit large datasets directly to Git; always use DVC to track them.

### 7. Sentiment Analysis Datasets

Below are the datasets used for this project:

- **Sentiment140** (Twitter data)
  - **Description**: A dataset of 1.6 million tweets labeled with sentiment (positive, negative, neutral).
  - **Source**: [Sentiment140 Dataset](http://help.sentiment140.com/for-students/)
  
- **IMDb Movie Reviews**
  - **Description**: 50,000 highly polarized movie reviews labeled with positive and negative sentiments.
  - **Source**: [IMDb Movie Reviews Dataset](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)

- **Amazon Product Reviews**
  - **Description**: Millions of Amazon product reviews labeled with positive/negative sentiments.
  - **Source**: [Amazon Product Reviews](https://www.kaggle.com/datasets/bittlingmayer/amazonreviews)

- **Yelp Reviews**
  - **Description**: Reviews from Yelp labeled with positive and negative sentiments.
  - **Source**: [Yelp Reviews Dataset](https://www.kaggle.com/datasets/yelp-dataset/yelp-dataset)

### 8. Kaggle Setup

To download datasets from Kaggle, you will need to set up the Kaggle API on your machine.

#### Step 1: Install Kaggle API

Install the Kaggle API using pip:

```
pip install kaggle
```

#### Step 2: Authenticate with Kaggle

1. Go to your [Kaggle Account Settings](https://www.kaggle.com/account).
2. Under **API**, click on **Create New API Token** to download a `kaggle.json` file.
3. Move `kaggle.json` to the appropriate directory:
   - **On macOS/Linux**:
     ```
     mkdir -p ~/.kaggle
     mv ~/Downloads/kaggle.json ~/.kaggle/
     chmod 600 ~/.kaggle/kaggle.json
     ```
   - **On Windows**: Place it in `C:\Users\<YourUsername>\.kaggle\kaggle.json`.

#### Step 3: Download Datasets with Kaggle API

After the Kaggle API is configured, you can download datasets using the following commands:

- **IMDb Movie Reviews**:
   ```
   kaggle datasets download -d lakshmi25npathi/imdb-dataset-of-50k-movie-reviews
   unzip imdb-dataset-of-50k-movie-reviews.zip -d data/raw/
   ```

- **Amazon Reviews**:
   ```
   kaggle datasets download -d bittlingmayer/amazonreviews
   unzip amazonreviews.zip -d data/raw/
   ```

### 9. How to Publish Model Mesh on Docker Hub

To publish **Model Mesh** or any Docker image to Docker Hub, follow the steps below:

#### Step 1: Log in to Docker Hub
Ensure you have a Docker Hub account and are logged in locally.

```
docker login
```
You will be prompted to enter your Docker Hub username and password.

#### Step 2: Build the Docker Image
Navigate to the root of your **Model Mesh** project and build the Docker image using the following command:

```
docker build -t model-mesh .
```
This command will create a Docker image tagged as `model-mesh`.

#### Step 3: Tag the Image for Docker Hub
Docker Hub requires a specific format for image tags: `dockerhub-username/repository-name:tag`.

Tag your image with your Docker Hub username and repository name (e.g., `model-mesh`):

```
docker tag model-mesh your-dockerhub-username/model-mesh:latest
```

#### Step 4: Push the Image to Docker Hub
Once tagged, push the image to Docker Hub:

```
docker push your-dockerhub-username/model-mesh:latest
```
This command uploads your Docker image to the Docker Hub repository.

#### Step 5: Make the Image Public (Optional)
If you want your image to be available for others to pull, make the repository public:

1. Go to [Docker Hub](https://hub.docker.com/) and log in.
2. Navigate to your profile and select your repository (e.g., `model-mesh`).
3. Under **Repository Settings**, switch the visibility from **Private** to **Public**.

#### Step 6: Pulling the Image
You can now pull the image from Docker Hub on any machine with Docker installed using:

```
docker pull your-dockerhub-username/model-mesh:latest
```