import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

class Config:
    PDF_DATA_PATH = os.path.join(BASE_DIR, 'data', 'pdfs')
    FAISS_INDEX_PATH = os.path.join(BASE_DIR, 'data', 'faiss_index')
    
    #huggingface model to use
    #HUGGINGFACE_MODEL_NAME = "distilgpt2"
    
    # flask app configurations
    
    DEBUG = True
    PORT = 5000
    HOST = '0.0.0.0'
    # Load Google API key from environment variable, fallback to placeholder if not set
    GOOGLE_API_KEY = os.getenv("API_KEY")  
    
    