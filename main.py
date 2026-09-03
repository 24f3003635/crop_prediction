from app.api import api_bp
from flask import Flask
from flask_cors import CORS


def create_app():
    app=Flask(__name__)
    CORS(app)
    app.register_blueprint(api_bp,url_prefix="/api")

    @app.route("/")
    def home():
        return "Server running"

    return app

app=create_app()

if __name__ == "__main__":
    app.run(debug=True)