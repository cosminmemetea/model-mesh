# api/fastapi_app.py
from fastapi import FastAPI, HTTPException
from models.model_manager import ModelManager
from pydantic import BaseModel

app = FastAPI()

# Initialize the model manager
model_manager = ModelManager()

class TextRequest(BaseModel):
    text: str
    model: str = "bert"  # Default model is BERT, but you can switch

@app.post("/predict")
async def predict_sentiment(request: TextRequest):
    try:
        model = model_manager.get_model(request.model)
        result = model.predict(request.text)
        return {"result": result}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
