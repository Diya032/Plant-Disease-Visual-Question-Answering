from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import torch
import io
import os
import time
import logging
from functools import lru_cache
from typing import Dict, Any
import uvicorn
from starlette.requests import Request
from starlette.status import HTTP_429_TOO_MANY_REQUESTS
import asyncio
from pathlib import Path

# Import our VQA model
from enhanced_vqa import EnhancedPlantDiseaseVQA

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("api.log")]
)
logger = logging.getLogger(__name__)

# Define paths
CHECKPOINT_PATH = os.environ.get(
    "CHECKPOINT_PATH", 
    r"checkpoints\best_checkpoint_epoch_0.pth"  # Default path, change as needed
)
PROCESSOR_PATH = "Salesforce/blip-vqa-base"
ENHANCER_MODEL_NAME = os.environ.get("ENHANCER_MODEL", "google/flan-t5-base")
UPLOAD_DIR = "uploads"

# Create upload directory
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Rate limiting
class RateLimiter:
    def __init__(self, calls: int = 10, period: int = 60):
        self.calls = calls
        self.period = period
        self.timestamps = {}
        
    async def is_rate_limited(self, client_ip: str) -> bool:
        now = time.time()
        
        # Initialize if new client
        if client_ip not in self.timestamps:
            self.timestamps[client_ip] = []
        
        # Remove old timestamps
        self.timestamps[client_ip] = [ts for ts in self.timestamps[client_ip] if now - ts < self.period]
        
        # Check if rate limit is exceeded
        if len(self.timestamps[client_ip]) >= self.calls:
            return True
        
        # Add new timestamp
        self.timestamps[client_ip].append(now)
        return False

rate_limiter = RateLimiter(calls=5, period=60)  # 5 calls per minute per IP

# Create a single instance of the VQA model to avoid reloading
@lru_cache(maxsize=1)
def get_vqa_model():
    """Initialize and cache the VQA model."""
    try:
        logger.info(f"Loading VQA model with checkpoint: {CHECKPOINT_PATH}")
        model = EnhancedPlantDiseaseVQA(
            checkpoint_path=CHECKPOINT_PATH,
            processor_path=PROCESSOR_PATH,
            enhancer_model_name=ENHANCER_MODEL_NAME
        )
        logger.info("VQA model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Failed to load VQA model: {str(e)}")
        raise e

# Handle background image cleanup
def cleanup_old_images():
    """Remove images older than 1 hour."""
    try:
        now = time.time()
        for file_path in Path(UPLOAD_DIR).glob('*.jpg'):
            if now - file_path.stat().st_mtime > 3600:  # 1 hour
                file_path.unlink()
                logger.info(f"Removed old image: {file_path}")
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")

# Initialize FastAPI app
app = FastAPI(title="Plant Disease VQA API", 
              description="API for Visual Question Answering on plant diseases")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Startup event
@app.on_event("startup")
async def startup_event():
    logger.info("Starting API server")
    # Pre-load model to avoid cold start
    try:
        get_vqa_model()
    except Exception as e:
        logger.error(f"Error pre-loading model: {e}")

# Rate limiter middleware
@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    client_ip = request.client.host
    if request.url.path == "/predict/":  # Only rate limit the predict endpoint
        is_limited = await rate_limiter.is_rate_limited(client_ip)
        if is_limited:
            return JSONResponse(
                status_code=HTTP_429_TOO_MANY_REQUESTS,
                content={"error": "Rate limit exceeded. Please try again later."}
            )
    response = await call_next(request)
    return response

# Endpoints
@app.get("/")
async def root():
    return {"message": "Plant Disease VQA API is running"}

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        # Check if model is loaded
        model = get_vqa_model()
        return {"status": "healthy", "model_loaded": True}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}

@app.post("/predict/")
async def predict(
    background_tasks: BackgroundTasks,
    image: UploadFile = File(...), 
    question: str = Form(...),
):
    """
    Get answer to a question about a plant disease image.
    
    - **image**: Plant image file (JPG, PNG)
    - **question**: Question about the plant disease
    """
    try:
        start_time = time.time()
        
        # Input validation
        if not question:
            raise HTTPException(status_code=400, detail="Question cannot be empty")
            
        if not image.filename:
            raise HTTPException(status_code=400, detail="No image file provided")
            
        # File size validation (limit to 5MB)
        image_data = await image.read()
        if len(image_data) > 5 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="Image file too large (max 5MB)")
        
        # Check file type
        file_ext = image.filename.split('.')[-1].lower()
        if file_ext not in ['jpg', 'jpeg', 'png']:
            raise HTTPException(status_code=400, detail="Only JPG and PNG files are allowed")
        
        # Save image for debugging/auditing (optional)
        timestamp = int(time.time())
        safe_filename = f"{timestamp}_{os.path.basename(image.filename)}"
        image_path = os.path.join(UPLOAD_DIR, safe_filename)
        with open(image_path, "wb") as f:
            f.write(image_data)
        
        # Load image
        image_pil = Image.open(io.BytesIO(image_data)).convert("RGB")
        
        # Get model and make prediction
        vqa_model = get_vqa_model()
        answer = vqa_model.ask_question(image_pil, question)
        
        # Schedule cleanup of old images
        background_tasks.add_task(cleanup_old_images)
        
        # Calculate processing time
        process_time = time.time() - start_time
        
        # Return result
        return {
            "answer": answer,
            "processing_time_seconds": round(process_time, 2)
        }
        
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        logger.error("GPU out of memory error")
        raise HTTPException(status_code=503, detail="Server is temporarily unavailable. Try again later.")
        
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("inference_api:app", host="0.0.0.0", port=8000, reload=True)