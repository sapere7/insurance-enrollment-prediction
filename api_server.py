# api_server.py
# REST API for insurance enrollment prediction using FastAPI.

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator 
from typing import Optional, Literal, Dict, Any
from sklearn.pipeline import Pipeline 
import joblib 
import pandas as pd
import numpy as np
import logging
import os
import uvicorn
import datetime 

# --- Logging Setup ---
def setup_logging(log_dir: str = "logs/api", level: int = logging.INFO) -> logging.Logger: 
    """Configures logging to console and a timestamped file for the API server."""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"api_run_{timestamp}.log")

    # Configure logger for this script
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logger = logging.getLogger("api_server") 
    logger.setLevel(level)
    logger.propagate = False 
    
    # Remove existing handlers to avoid duplication
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Add new handlers
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter(log_format))
    logger.addHandler(file_handler)
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(log_format))
    logger.addHandler(console_handler)

    logger.info(f"API Server logging configured. Log file: {log_file}")
    return logger

logger = setup_logging() 

# --- FastAPI App ---
app = FastAPI(
    title="Insurance Enrollment Prediction API",
    description="API for predicting employee insurance enrollment likelihood.",
    version="1.0.0"
)

# --- Data Models ---
class Employee(BaseModel):
    """Input data model for a single employee prediction request."""
    age: int = Field(..., example=35, description="Employee age in years")
    gender: Literal["Male", "Female", "Other"] = Field(..., example="Female", description="Employee gender")
    marital_status: Literal["Single", "Married", "Divorced", "Widowed"] = Field(..., example="Married", description="Marital status")
    salary: float = Field(..., example=75000, description="Annual salary in USD")
    employment_type: Literal["Full-time", "Part-time", "Contract"] = Field(..., example="Full-time", description="Employment type")
    region: str = Field(..., example="Northeast", description="Geographic region")
    has_dependents: bool = Field(..., example=True, description="Whether employee has dependents")
    tenure_years: float = Field(..., example=5.5, description="Years of employment at company")
    
    # Pydantic V2 Validators
    @field_validator('age')
    @classmethod
    def age_must_be_positive(cls, v: int) -> int:
        if v < 18 or v > 80: # Example reasonable age range
            raise ValueError('Age must be between 18 and 80')
        return v
    
    @field_validator('salary')
    @classmethod
    def salary_must_be_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError('Salary must be positive')
        return v
    
    @field_validator('tenure_years')
    @classmethod
    def tenure_must_be_positive(cls, v: float) -> float:
        if v < 0:
            raise ValueError('Tenure years must be non-negative')
        return v

class PredictionResponse(BaseModel):
    """Response model for the prediction endpoint."""
    employee_data: Employee
    enrollment_probability: float = Field(..., example=0.75, description="Probability of enrollment (0 to 1)")
    predicted_enrollment: bool = Field(..., example=True, description="Predicted enrollment status (True if probability >= 0.5)")

# --- Global Model Variable ---
# Stores the loaded pipeline after startup
model_pipeline: Optional[Pipeline] = None

# --- Utility Functions ---
def get_model_path() -> str:
    """
    Determines the model file path to load.
    Uses MODEL_PATH environment variable if set, otherwise defaults to 
    the tuned Gradient Boosting model, falling back to the initial one.
    """
    default_tuned_path = "output/models/GradientBoosting_tuned_pipeline.joblib"
    default_initial_path = "output/models/GradientBoosting_pipeline.joblib"
    
    # Determine default path based on file existence
    if os.path.exists(default_tuned_path):
        default_path = default_tuned_path
        logger.debug(f"Defaulting to tuned model: {default_path}")
    else:
        default_path = default_initial_path
        logger.debug(f"Tuned model not found, defaulting to initial model: {default_path}")
        
    # Check environment variable override
    model_path_env = os.environ.get("MODEL_PATH")
    if model_path_env:
        logger.info(f"Using model path from MODEL_PATH environment variable: {model_path_env}")
        return model_path_env
    else:
        logger.info(f"Using default model path: {default_path}")
        return default_path

# --- FastAPI Events ---
@app.on_event("startup")
async def startup_event():
    """Loads the scikit-learn pipeline model when the API starts."""
    global model_pipeline
    model_path = get_model_path()
    logger.info(f"Attempting to load model from: {model_path}")
    try:
        if not os.path.exists(model_path):
            logger.error(f"Model file not found at specified path: {model_path}")
            raise FileNotFoundError(f"Model file not found at {model_path}")
        
        model_pipeline = joblib.load(model_path)
        logger.info(f"Model loaded successfully from {model_path}")
        
        # Log basic info about the loaded pipeline
        if isinstance(model_pipeline, Pipeline) and hasattr(model_pipeline, 'steps'):
             logger.info(f"Model pipeline steps: {[step[0] for step in model_pipeline.steps]}")
        else:
             logger.info(f"Loaded object type: {type(model_pipeline)}")

    except Exception as e:
        logger.error(f"FATAL: Failed to load model during startup: {e}", exc_info=True)
        model_pipeline = None # Ensure model is None if loading fails

# --- API Endpoints ---
@app.get("/", summary="API Root")
async def root():
    """Provides basic information about the API."""
    return {
        "message": "Insurance Enrollment Prediction API",
        "version": app.version,
        "documentation": app.docs_url
    }

@app.post("/predict", response_model=PredictionResponse, summary="Predict Enrollment")
async def predict(employee: Employee):
    """
    Predicts insurance enrollment probability for a single employee.

    Takes employee data as input and returns the enrollment probability 
    and a binary prediction based on a 0.5 threshold.
    """
    global model_pipeline 
    
    if model_pipeline is None:
        logger.error("Prediction failed: Model is not loaded.")
        raise HTTPException(status_code=503, detail="Model not available. Check server logs for loading errors.")
    
    try:
        # Convert input Pydantic model to a DataFrame suitable for the pipeline
        input_df = pd.DataFrame([employee.dict()])
        logger.info(f"Received prediction request for employee data.") # Avoid logging PII if sensitive
        logger.debug(f"Input data: {employee.dict()}")
        
        # Make prediction using the loaded pipeline
        enrollment_prob = model_pipeline.predict_proba(input_df)[0, 1] # Probability of class 1
        prediction = enrollment_prob >= 0.5 
        
        logger.info(f"Prediction successful: Probability={enrollment_prob:.4f}, Predicted={'Enroll' if prediction else 'Not Enroll'}")
        
        # Return prediction result
        return PredictionResponse(
            employee_data=employee,
            enrollment_probability=float(enrollment_prob),
            predicted_enrollment=bool(prediction)
        )
    
    except Exception as e:
        logger.error(f"Error during prediction: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error during prediction.")

@app.get("/health", summary="API Health Check")
async def health_check():
    """Checks if the API is running and the model was loaded successfully."""
    model_loaded_status = model_pipeline is not None
    if not model_loaded_status:
        logger.warning("Health check failed: Model not loaded.")
        raise HTTPException(status_code=503, detail="Model not loaded or failed to load.") 
    logger.debug("Health check successful.")
    return {"status": "healthy", "model_loaded": model_loaded_status}

# --- Main Execution ---
if __name__ == "__main__":
    logger.info("Starting API server using Uvicorn...")
    # Run the FastAPI app using Uvicorn
    # reload=True is useful for development but should be False in production
    uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=True)
