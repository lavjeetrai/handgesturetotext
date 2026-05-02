# handgesturetotext

Hand gesture to text and speech conversion using a webcam, MediaPipe hand landmarks, and a trained CNN model. The main app now runs in a browser with a Flask backend.

## Features

- Browser-based webcam input
- Real-time hand gesture detection
- ASL alphabet prediction with a trained Keras model
- Text sentence building from detected signs
- Browser text-to-speech output
- Optional word suggestions when an Enchant dictionary is available

## Run Locally

Use the included virtual environment from the project root:

```powershell
..\.venv\Scripts\python.exe app.py
```

Or double-click:

```text
run_app.bat
```

Then open:

```text
http://127.0.0.1:5000
```

Allow camera access in the browser and press Start.

## Main Files

- `app.py` - Flask web application
- `web_predictor.py` - web prediction/session wrapper
- `templates/index.html` - browser UI
- `static/` - web UI styles and webcam JavaScript
- `final_pred.py` - legacy prediction rules reused by the web wrapper
- `cnn8grps_rad1_model.h5` - trained CNN model
- `hand_landmarker.task` - MediaPipe hand landmark model

## Requirements

Install dependencies with:

```powershell
pip install -r requirements.txt
```

Python 3.13 is used in the included local setup.

## Hosting Notes

Use a Python host that supports TensorFlow, OpenCV, and MediaPipe. A Docker-based service is usually the easiest option. Camera access in browsers requires HTTPS on a hosted domain; localhost works without HTTPS for local testing.

Production entry point:

```powershell
waitress-serve --host=0.0.0.0 --port=5000 wsgi:app
```

Docker build and run:

```powershell
docker build -t sign-language-webapp .
docker run --rm -p 5000:5000 sign-language-webapp
```

Health check:

```text
GET /health
```

Useful environment variables are listed in `.env.example`. For hosting, set `PORT` to the value your platform provides.
