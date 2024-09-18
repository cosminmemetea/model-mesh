from fastapi import FastAPI, HTTPException
from models.model_manager import ModelManager
from pydantic import BaseModel
from fastapi.responses import JSONResponse

app = FastAPI(
    title="Model-Mesh API",
    description="🚀 A Minimalist, Futuristic API for Sentiment Prediction",
    version="1.0.0",
    docs_url="/docs",  # Swagger
    redoc_url=None     # Disabling ReDoc for clean minimalism
)

# Initialize the model manager
model_manager = ModelManager()

# Define the text request model
class TextRequest(BaseModel):
    text: str
    model: str = "bert"  # Default model is BERT, but this can be overridden

@app.post("/predict", summary="🌟 Predict sentiment using AI", tags=["Prediction"])
async def predict_sentiment(request: TextRequest):
    """
    Predicts the sentiment of the given text using the specified model.
    - Default model is BERT.
    - You can specify another model via the `model` parameter.
    """
    try:
        model = model_manager.get_model(request.model)
        result = model.predict(request.text)
        return JSONResponse(content={"result": result, "model_used": request.model, "input_text": request.text})
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

# Root endpoint for a futuristic welcome
@app.get("/", summary="🚀 Welcome to Model-Mesh API", tags=["General"])
async def root():
    return JSONResponse(content={"message": "🚀 Welcome to Model-Mesh: AI Sentiment Predictions in the Future."})

# Health check for the API
@app.get("/health", summary="✨ Check API health", tags=["General"])
async def health():
    return JSONResponse(content={"status": "OK", "message": "✨ All systems operational."})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.fastapi_app:app", host="0.0.0.0", port=5050, reload=True)
