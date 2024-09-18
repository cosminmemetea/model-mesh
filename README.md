# Running the Application

docker build -t model-mesh:latest .
docker run -d -p 5050:5050 model-mesh:latest

http://localhost:5050/docs#/default/predict_sentiment_predict_post