# run_all.py
# Orchestrator script to run the main ML pipeline and optional hyperparameter tuning.

import os
import logging
import argparse
import time
import yaml
import datetime
from typing import Dict, Any

# Import project modules
import main as main_pipeline
import hyperparameter_tuning as tuning_pipeline
# Note: EDA and Feature Importance are run separately via their own scripts or commands.

# --- Logging Setup ---
def setup_logging(log_dir: str = "logs", level: int = logging.INFO) -> logging.Logger:
    """Configures logging to console and a timestamped file for the orchestrator."""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"run_all_{timestamp}.log")

    # Configure logger for this script
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logger = logging.getLogger("run_all")
    logger.setLevel(level)
    logger.propagate = False # Prevent duplicate logs if other modules also configure root logger
    
    # Remove existing handlers for this logger to avoid duplication on re-runs
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Add new handlers
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter(log_format))
    logger.addHandler(file_handler)
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(log_format))
    logger.addHandler(console_handler)

    logger.info(f"Orchestrator logging configured. Log file: {log_file}")
    return logger

logger = setup_logging()

# --- Main Orchestration ---
def run_full_process(config_path: str, run_tuning: bool = False):
    """
    Runs the main ML pipeline and optionally the hyperparameter tuning pipeline.

    Args:
        config_path: Path to the configuration YAML file.
        run_tuning: If True, runs the hyperparameter tuning script after the main pipeline.
    """
    start_time = time.time()
    logger.info(f"--- Starting Full Process at {time.strftime('%Y-%m-%d %H:%M:%S')} ---")
    
    # Load configuration using the function from main_pipeline
    try:
        logger.info(f"Loading configuration from: {config_path}")
        config = main_pipeline.load_config(config_path) 
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}", exc_info=True)
        return # Stop execution if config fails to load

    # --- Run Main Pipeline ---
    main_success = False
    try:
        logger.info("--- Running Main Pipeline (main.py) ---")
        main_results = main_pipeline.main(config)
        logger.info("--- Main Pipeline Finished ---")
        logger.info(f"Main pipeline results summary: {main_results}")
        main_success = True
    except Exception as e:
        logger.error(f"Main pipeline failed: {e}", exc_info=True)
        logger.warning("Skipping tuning due to main pipeline failure.")
        run_tuning = False # Ensure tuning doesn't run

    # --- Run Hyperparameter Tuning (Optional and only if main succeeded) ---
    if run_tuning and main_success:
        try:
            logger.info("--- Running Hyperparameter Tuning (hyperparameter_tuning.py) ---")
            # The tuning script now handles loading its required data based on config
            # We just need to ensure the config is passed correctly (implicitly via load_config inside tuning script)
            # Or explicitly pass the loaded config if tuning script is refactored to accept it.
            # Current tuning script loads config itself, so we just execute it.
            
            # Re-load config within tuning's scope (or pass config dict if refactored)
            tuning_config = tuning_pipeline.load_config(config_path) 
            proc_data_dir = tuning_config.get('processed_data_dir')
            X_train, y_train = tuning_pipeline.load_processed_data(proc_data_dir)
            preprocessor = tuning_pipeline.define_preprocessor(X_train)
            
            # Run the main tuning function
            tuning_results = tuning_pipeline.tune_all_models(X_train, y_train, preprocessor, tuning_config)
            logger.info("--- Hyperparameter Tuning Finished ---")
            logger.info("Tuning process completed. Check tuning logs for details and saved models.")

        except Exception as e:
            logger.error(f"Hyperparameter tuning failed: {e}", exc_info=True)
    elif run_tuning and not main_success:
         logger.info("Skipping hyperparameter tuning because the main pipeline failed.")
    else:
        logger.info("Skipping hyperparameter tuning as requested.")

    elapsed_time = time.time() - start_time
    logger.info(f"--- Full Process Finished in {elapsed_time:.2f} seconds ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run ML pipeline stages based on config file.')
    parser.add_argument('--config', type=str, default='config.yaml', 
                        help='Path to the configuration YAML file.')
    parser.add_argument('--run_tuning', action='store_true', 
                        help='Run hyperparameter tuning after the main pipeline.')
    
    args = parser.parse_args()
    
    run_full_process(args.config, args.run_tuning)
