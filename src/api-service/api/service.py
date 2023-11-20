import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "./"))

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from starlette.middleware.cors import CORSMiddleware
import base64
from PIL import Image
import io
from inference_engine import InferenceEngine  # Assuming this is your class from the model code

app = FastAPI(title="BioViL Model API", description="API for BioViL Model", version="1.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

inference_engine = InferenceEngine()

@app.on_event("startup")
async def startup_event():
    print("Starting up...")


@app.get("/")
async def root():
    return {"message": "BioViL Model API"}


class PredictRequest(BaseModel):
    image: str
    text: str


@app.post("/predict")
async def predict(request: PredictRequest):
    # Decode the Base64 image
    try:
        image_data = base64.b64decode(request.image.split(",")[1])  # Assuming the format is "data:image/png;base64,<data>"
        image = Image.open(io.BytesIO(image_data))
        text_prompt = request.text
    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid image / text data")
    
    # Call the inference method
    try:
        result = inference_engine.inference(image, text_prompt)

        # Convert Figure to Bytes
        buf = io.BytesIO()
        result.savefig(buf, format='png')  # Save figure to the buffer
        buf.seek(0)  # Rewind the buffer
        encoded_image = base64.b64encode(buf.getvalue()).decode('utf-8')

        # # Create a StreamingResponse, setting the correct media type
        # result = StreamingResponse(buf, media_type="image/png")
        return f"data:image/png;base64,{encoded_image}"
    
    except Exception as e:
        return "Error occurred: " + str(e)
