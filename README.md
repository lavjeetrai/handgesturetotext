# handgesturetotext

Hand gesture to text and speech conversion using a webcam, MediaPipe hand landmarks, and a trained CNN model.

## Features

- Real-time hand gesture detection from webcam input
- ASL alphabet prediction with a trained Keras model
- Text sentence building from detected signs
- Text-to-speech output
- Optional word suggestions when an Enchant dictionary is available

## Run Locally

Use the included virtual environment from the project root:

```powershell
..\.venv\Scripts\python.exe final_pred.py
```

Or double-click:

```text
run_app.bat
```

## Main Files

- `final_pred.py` - GUI application for webcam prediction
- `prediction_wo_gui.py` - OpenCV prediction script without Tkinter GUI
- `cnn8grps_rad1_model.h5` - trained CNN model
- `hand_landmarker.task` - MediaPipe hand landmark model
- `AtoZ_3.1/` - ASL alphabet image dataset

## Requirements

Install dependencies with:

```powershell
pip install -r requirements.txt
```

Python 3.13 is used in the included local setup.
