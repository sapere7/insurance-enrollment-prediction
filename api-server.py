# api.py
# REST API for insurance enrollment prediction using FastAPI

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
from typing import Optional, Literal
import pickle
import pandas as pd
import numpy as np
import logging
import os
import uvicorn

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Insurance Enrollment Prediction API",
    description="API for predicting employee insurance enrollment likelihood",
    version="1.0.0"
)

# Define input data model
class Employee(BaseModel):
    age: int = Field(..., example=35, description="Employee age in years")
    gender: Literal["Male", "Female", "Other"] = Field(..., example="Female", description="Employee gender")
    marital_status: Literal["Single", "Married", "Divorced", "Widowed"] = Field(..., example="Married", description="Marital status")
    salary: float = Field(..., example=75000, description="Annual salary in USD")
    employment_type: Literal["Full-time", "Part-time", "Contract"] = Field(..., example="Full-time", description="Employment type")
    region: str = Field(..., example="Northeast", description="Geographic region")
    has_dependents: bool = Field(..., example=True, description="Whether employee has dependents")
    tenure_years: float = Field(..., example=5.5, description="Years of employment at company")
    
    # Validators for input data
    @validator('age')
    def age_must_be_positive(cls, v):
        if v < 18 or v > 80:
            raise ValueError('Age must be between 18 and 80')
        return v
    
    @validator('salary')
    def salary_must_be_positive(cls, v):
        if v <= 0:
            raise ValueError('Salary must be positive')
        return v
    
    @validator('tenure_years')
    def tenure_must_be_positive(cls, v):
        if v < 0:
            raise ValueError('Tenure years must be non-negative')
        return v

class PredictionResponse(BaseModel):
    employee_data: Employee
    enrollment_probability: float = Field(..., example=0.75, description="Probability of enrollment")
    predicted_enrollment: bool = Field(..., example=True, description="Predicted enrollment status (True = will enroll)")
    feature_importance: dict = Field(..., description="Importance of each feature for this prediction")

# Global variables to store model and feature names
model = None
feature_names = None

@app.on_event("startup")
async def load_model():
    """Load model and feature names on startup"""
    global model, feature_names
    
    try:
        logger.info("Loading model and feature names")
        model_path = os.environ.get("MODEL_PATH", "output/models/GradientBoosting_pipeline.pkl")
        
        if not os.path.exists(model_path):
            logger.error(f"Model file not found at {model_path}")
            raise FileNotFoundError(f"Model file not found at {model_path}")
        
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        # Get feature names
        preprocessor = model.named_steps["preprocessor"]
        try:
            # Try to get feature names from the pipeline
            feature_names = []
            for name, trans, cols in preprocessor.transformers_:
                if name != 'drop':
                    if hasattr(trans, 'get_feature_names_out'):
                        feature_names.extend(trans.get_feature_names_out(cols))
                    else:
                        feature_names.extend(cols)
        except Exception as e:
            logger.warning(f"Could not extract feature names from the pipeline: {e}")
            feature_names = None
        
        logger.info("Model and feature names loaded successfully")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Insurance Enrollment Prediction API",
        "version": "1.0.0",
        "documentation": "/docs"
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(employee: Employee):
    """
    Predict insurance enrollment for an employee
    
    This endpoint takes employee information and returns:
    - The probability of the employee enrolling in the insurance program
    - The binary prediction (will enroll or not)
    - Feature importance for this specific prediction
    """
    global model, feature_names
    
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Convert input to pandas DataFrame
        input_df = pd.DataFrame([employee.dict()])
        
        # Make prediction
        enrollment_prob = model.predict_proba(input_df)[0, 1]
        prediction = enrollment_prob >= 0.5
        
        # Feature importance
        importance_dict = {}
        
        # Get feature importance if the model supports it
        if hasattr(model.named_steps["model"], 'feature_importances_') and feature_names is not None:
            # Get feature importance from the model (e.g., RandomForest, GradientBoosting)
            importances = model.named_steps["model"].feature_importances_
            
            # Sort importance values and get top 10
            if len(importances) == len(feature_names):
                sorted_indices = np.argsort(importances)[::-1][:10]
                for i in sorted_indices:
                    importance_dict[feature_names[i]] = float(importances[i])
        
        # Return prediction result
        return {
            "employee_data": employee,
            "enrollment_probability": float(enrollment_prob),
            "predicted_enrollment": bool(prediction),
            "feature_importance": importance_dict
        }
    
    except Exception as e:
        logger.error(f"Error making prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Error making prediction: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    return {"status": "healthy", "model_loaded": True}

def main():
    """Run the API server"""
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)

if __name__ == "__main__":
    main()
