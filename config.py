import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

class Config:
    PDF_DATA_PATH = os.path.join(BASE_DIR, 'data', 'pdfs')
    FAISS_INDEX_PATH = os.path.join(BASE_DIR, 'data', 'faiss_index')
    
    #huggingface model to use
    HUGGINGFACE_MODEL_NAME = "distilgpt2"
    
    # flask app configurations
    
    DEBUG = True
    PORT = 5000
    HOST = '0.0.0.0'
    GOOGLE_API_KEY = "AIzaSyCMDMzTUoLLrkTH7ALeNYdrX4sJ6U786_0"
    