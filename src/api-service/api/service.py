import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "./"))

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import StreamingResponse
from starlette.middleware.cors import CORSMiddleware
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

@app.post("/predict")
async def predict(file: UploadFile, text_prompt: str=Form(...)):
    # Read image file
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data))
    
    # Call the inference method
    try:
        result = inference_engine.inference(image, text_prompt)

        # Convert Figure to Bytes
        buf = io.BytesIO()
        result.savefig(buf, format='png')  # Save figure to the buffer
        buf.seek(0)  # Rewind the buffer

        # Create a StreamingResponse, setting the correct media type
        result = StreamingResponse(buf, media_type="image/png")
        return result
    
    except Exception as e:
        return "Error occurred: " + str(e)
