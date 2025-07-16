from flask import Flask
from config import Config
from routes import main_bp
from rag_pipeline import setup_rag_components

import os

def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)
    
    #register blueprints
    app.register_blueprint(main_bp)

    #if directories exists or not
    os.makedirs(app.config['PDF_DATA_PATH'], exist_ok = True)
    os.makedirs(app.config['FAISS_INDEX_PATH'], exist_ok = True)  # create dirs if not exits
    
    with app.app_context():
        setup_rag_components()

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(host=app.config["HOST"], port=app.config["PORT"], debug=app.config["DEBUG"])

if __name__ == '__main__':
    app = create_app()
    app.run(debug=app.config['DEBUG'], port=app.config['PORT'])

