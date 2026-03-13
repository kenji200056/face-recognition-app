# Face Recognition App

Real-time face recognition application using Python, OpenCV, and dlib.

## Features

- Real-time face detection and recognition via webcam
- Support for multiple people with different colors
- Side panel showing detected people history
- Facial landmarks visualization (eyes, nose, lips, chin)
- Spanish UI

## Screenshot

![Face Recognition App](https://user-images.githubusercontent.com/example/screenshot.png)

## Installation

### Requirements

- Python 3.8+
- macOS or Linux

### Setup

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install pillow opencv-python numpy face_recognition
pip install "setuptools<70"
pip install git+https://github.com/ageitgey/face_recognition_models
```

## Usage

1. Add face images to the `faces/` folder (one image per person, filename = name)
2. Run the app:

```bash
source venv/bin/activate
python webcam_app.py
```

3. Look at the camera - the app will recognize registered faces
4. Click "SALIR" to exit or "REINICIAR" to reset

## Adding New Faces

Simply add a photo to the `faces/` folder:
- `faces/john.jpg` -> will be recognized as "john"
- `faces/maria.png` -> will be recognized as "maria"

## License

MIT License - see [LICENSE](LICENSE) for details.

## Author

Kenji G. Jimenez ([@kenji200056](https://github.com/kenji200056))
