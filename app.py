import logging
import os
import re

from flask import Flask, jsonify, render_template, request, send_from_directory
from werkzeug.exceptions import RequestEntityTooLarge

from web_predictor import WebSignPredictor, decode_data_url


SESSION_ID_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,80}$")


def create_app():
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = int(os.getenv("MAX_CONTENT_LENGTH", 3 * 1024 * 1024))
    app.config["JSON_SORT_KEYS"] = False
    app.predictor = WebSignPredictor()

    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    @app.after_request
    def add_security_headers(response):
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "same-origin"
        response.headers["Permissions-Policy"] = "camera=(self), microphone=()"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "img-src 'self' data:; "
            "media-src 'self' blob:; "
            "script-src 'self'; "
            "style-src 'self'; "
            "connect-src 'self'"
        )
        return response

    @app.errorhandler(RequestEntityTooLarge)
    def handle_large_request(_exc):
        return jsonify({"status": "error", "message": "Image payload is too large"}), 413

    @app.errorhandler(ValueError)
    def handle_bad_request(exc):
        return jsonify({"status": "error", "message": str(exc)}), 400

    @app.errorhandler(404)
    def handle_not_found(_exc):
        return jsonify({"status": "error", "message": "Not found"}), 404

    @app.errorhandler(500)
    def handle_server_error(_exc):
        app.logger.exception("Unhandled server error")
        return jsonify({"status": "error", "message": "Internal server error"}), 500

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/favicon.ico")
    def favicon():
        return send_from_directory(
            app.static_folder,
            "favicon.ico",
            mimetype="image/vnd.microsoft.icon",
            max_age=86400,
        )

    @app.get("/health")
    def health():
        return jsonify({"status": "ok"})

    @app.post("/api/predict")
    def predict():
        payload = request.get_json(silent=True) or {}
        session_id = _session_id(payload)
        image = payload.get("image")
        if not image:
            return jsonify({"status": "error", "message": "Missing image"}), 400

        try:
            frame = decode_data_url(image)
            return jsonify(app.predictor.session(session_id).predict_frame(frame))
        except ValueError as exc:
            return jsonify({"status": "error", "message": str(exc)}), 400
        except Exception:
            app.logger.exception("Prediction failed")
            return jsonify({"status": "error", "message": "Prediction failed"}), 500

    @app.post("/api/reset")
    def reset():
        payload = request.get_json(silent=True) or {}
        session_id = _session_id(payload)
        return jsonify(app.predictor.reset(session_id))

    @app.post("/api/suggest/<int:index>")
    def suggest(index):
        payload = request.get_json(silent=True) or {}
        session_id = _session_id(payload)
        return jsonify(app.predictor.suggest(session_id, index))

    return app


def _session_id(payload):
    session_id = str(payload.get("session_id") or "default")
    if not SESSION_ID_RE.fullmatch(session_id):
        raise ValueError("Invalid session id")
    return session_id


app = create_app()


if __name__ == "__main__":
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "5000"))
    app.run(host=host, port=port, debug=False)
