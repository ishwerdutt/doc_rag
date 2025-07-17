# app.py
from dotenv import load_dotenv
load_dotenv()
from flask import Flask
from config import Config
from routes import main_bp
from rag_pipeline import setup_rag_components
import os

def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    # Register blueprints
    app.register_blueprint(main_bp)

    # Ensure necessary directories exist
    os.makedirs(app.config['PDF_DATA_PATH'], exist_ok=True)
    os.makedirs(app.config['FAISS_INDEX_PATH'], exist_ok=True)

    # Setup RAG components when the app starts
    # initializing rag components
    with app.app_context():
        setup_rag_components()

    return app

# Create the app instance at the module level so Gunicorn can find it
app = create_app()


if __name__ == '__main__':
    app.run(debug=app.config['DEBUG'], host=app.config['HOST'], port=app.config['PORT'])