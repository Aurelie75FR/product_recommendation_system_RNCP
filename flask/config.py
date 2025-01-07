# flask/config.py
from dotenv import load_dotenv
import os
from pathlib import Path

base_path = Path(__file__).parent.parent
env_path = base_path / '.env'
load_dotenv(env_path)

class Config:
    # BigQuery config
    GOOGLE_APPLICATION_CREDENTIALS = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
    PROJECT_ID = os.getenv('PROJECT_ID')
    DATASET_ID = os.getenv('DATASET_ID')
    
    # Flask config
    DEBUG = os.getenv('FLASK_DEBUG', True)
    TESTING = os.getenv('FLASK_TESTING', False)

    # Debug log
    print(f"GOOGLE_APPLICATION_CREDENTIALS: {GOOGLE_APPLICATION_CREDENTIALS}")
    print(f"PROJECT_ID: {PROJECT_ID}")
    print(f"DATASET_ID: {DATASET_ID}")