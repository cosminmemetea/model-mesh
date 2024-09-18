from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from models.model_manager import ModelManager


app = FastAPI(
    title="Model-Mesh API",
    description="🚀 A Minimalist, Futuristic API for Sentiment Prediction",
    version="1.0.0",
    docs_url="/docs",  # Swagger
    redoc_url=None     # Disabling ReDoc for clean minimalism
)

# Allow CORS for all origins (for development purposes)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can restrict this to specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount the 'static' directory for serving static files like images, CSS, and HTML
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    file_path = Path(__file__).parent.parent / "static/index.html"
    return HTMLResponse(content=file_path.read_text(), status_code=200)

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
