import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

class Config:
    PDF_DATA_PATH = os.path.join(BASE_DIR, 'data', 'pdfs')
    FAISS_INDEX_PATH = os.path.join(BASE_DIR, 'data', 'faiss_index')
    MODEL_NAME = "gemini-2.5-flash"
    TEMPERATURE = 0.3
    PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")
    
    # flask app configurations
    
    DEBUG = True
    PORT = 5000
    HOST = '0.0.0.0'
    # Load Google API key from environment variable, fallback to placeholder if not set
    GOOGLE_API_KEY = os.getenv("API_KEY")  
    
    