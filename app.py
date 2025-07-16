# app.py
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
    # This will load/create the vectorstore and initialize the LLM
    # We use app.app_context() to ensure Flask's context is available
    # during RAG component setup, which is necessary for accessing app.config.
    with app.app_context():
        setup_rag_components()

    return app

# Create the app instance at the module level so Gunicorn can find it
app = create_app()


if __name__ == '__main__':
    # When running with `python app.py`, we explicitly run the Flask development server.
    app.run(debug=app.config['DEBUG'], host=app.config['HOST'], port=app.config['PORT'])