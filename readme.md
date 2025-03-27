# Insurance Enrollment Prediction

This repository contains a machine learning solution for predicting employee enrollment in a voluntary insurance product. The project includes data preprocessing, exploratory data analysis, model development, hyperparameter tuning, and a REST API for serving predictions.

## Project Structure

```
insurance-enrollment-prediction/
├── main.py                    # Core ML pipeline
├── eda.py                     # Exploratory data analysis
├── feature_importance.py      # Feature importance analysis
├── hyperparameter_tuning.py   # Hyperparameter tuning
├── api.py                     # FastAPI prediction server (Note: Not modified in this session)
├── config.yaml                # Configuration file for the pipeline
├── employee_data.csv          # Input data (synthetic)
├── requirements.txt           # Project dependencies
├── README.md                  # This file
├── report.md                  # Project report and findings
├── logs/                      # Directory for log files
└── output/                    # Generated outputs
    ├── figures/               # Visualizations and plots
    ├── models/                # Trained model files (initial and tuned)
    ├── processed_data/        # Processed data splits and preprocessor
    └── tuning/                # Hyperparameter tuning results and plots
```

## Setup Instructions

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

### Installation

1. Clone this repository:
```bash
git clone https://github.com/sapere7/insurance-enrollment-prediction.git
cd insurance-enrollment-prediction
```

2. Create and activate a virtual environment (optional but recommended):
```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Configuration

The pipeline behavior (file paths, model parameters, tuning settings) is controlled by the `config.yaml` file. Modify this file to change settings.

## Usage

### Running the ML Pipeline

This script performs data loading, preprocessing, splitting, saving processed data, initial model training/evaluation, and saves the best initial model.

```bash
python main.py
```
*(Ensure `config.yaml` points to the correct `data_file`)*

### Exploratory Data Analysis

To generate EDA visualizations (reads `employee_data.csv` directly and saves plots to `output/figures`):

```bash
python eda.py 
```
*(Optionally add `--data_file path/to/data.csv` and `--output_dir path/to/save/figures`)*

### Hyperparameter Tuning

This script loads the processed training data saved by `main.py`, performs hyperparameter tuning based on `config.yaml`, and saves the best *tuned* model pipeline for each model type.

```bash
python hyperparameter_tuning.py
```
*(Run `main.py` first to generate the processed data)*

### Feature Importance Analysis

To analyze and visualize feature importance for a specific model (e.g., GradientBoosting), loading the saved model and test data:

```bash
python feature_importance.py --model_name GradientBoosting
```
*(Ensure `main.py` has been run to generate processed data and models. Add `--config path/to/config.yaml` if not using the default. Plots and CSVs are saved to the configured `figures_dir`.)*

### Testing the Model

To test a saved model pipeline with sample data and optionally evaluate on the full dataset:

```bash
python test_script.py --model_path output/models/GradientBoosting_tuned_pipeline.joblib
```
*(Adjust `--model_path` as needed. Add `--data_file path/to/data.csv` to evaluate on specific data. Add `--output_dir path/to/save/test_results`.)*

### Running the Orchestrator

To run the main pipeline and optionally the tuning pipeline sequentially:

```bash
# Run main pipeline only
python run_all.py

# Run main pipeline AND hyperparameter tuning
python run_all.py --run_tuning 
```
*(Uses `config.yaml` by default. Add `--config path/to/config.yaml` if needed.)*

### Logging

All pipeline runs (`main.py`, `hyperparameter_tuning.py`, `run_all.py`, `test_script.py`, `feature_importance.py`) will generate timestamped log files in the `logs/` directory (or subdirectories like `logs/tuning`, `logs/test`, `logs/analysis`).

### Starting the API Server

To start the FastAPI prediction server:

```bash
python api_server.py
```

- The server will automatically try to load the tuned Gradient Boosting model (`output/models/GradientBoosting_tuned_pipeline.joblib`) if it exists.
- If not found, it will fall back to the initial Gradient Boosting model (`output/models/GradientBoosting_pipeline.joblib`).
- You can override this by setting the `MODEL_PATH` environment variable:
  ```bash
  export MODEL_PATH=output/models/Your_Chosen_Model.joblib # Linux/macOS
  # set MODEL_PATH=output\models\Your_Chosen_Model.joblib # Windows
  python api-server.py 
  ```

Once the server is running, you can access:
- API documentation: http://localhost:8000/docs
- Health check: http://localhost:8000/health
- Prediction endpoint: http://localhost:8000/predict (POST request)

### Making Predictions with the API

Here's an example of how to call the prediction API using curl:

```bash
curl -X 'POST' \
  'http://localhost:8000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "age": 35,
  "gender": "Female",
  "marital_status": "Married",
  "salary": 75000,
  "employment_type": "Full-time",
  "region": "Northeast",
  "has_dependents": true,
  "tenure_years": 5.5
}'
```

Or using Python with the requests library:

```python
import requests
import json

data = {
  "age": 35,
  "gender": "Female",
  "marital_status": "Married",
  "salary": 75000,
  "employment_type": "Full-time",
  "region": "Northeast",
  "has_dependents": True,
  "tenure_years": 5.5
}

response = requests.post("http://localhost:8000/predict", json=data)

if response.status_code == 200:
    result = response.json()
    print(json.dumps(result, indent=2))
    # Example Output:
    # {
    #   "employee_data": { ... input data ... },
    #   "enrollment_probability": 0.85,
    #   "predicted_enrollment": true
    # }
else:
    print(f"Error: {response.status_code} - {response.text}")
```
*(Note: Feature importance per prediction is currently not included in the API response for performance reasons.)*

## Project Report

For a detailed analysis of the approach, findings, and next steps, please see the [project report](report.md).

## License

This project is licensed under the MIT License - see the LICENSE file for details.
