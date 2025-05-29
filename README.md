# INS - Watching You Looking at Everything but Me

An interactive eye detection and display system using Intel OAK cameras and DepthAI technology. This project creates an artistic visualization that tracks faces and eyes, displaying them in a grid layout with the message "WATCHING YOU LOOKING AT EVERYTHING BUT ME" when no eyes are detected.

## Features

- **Real-time Face & Eye Detection**: Uses DepthAI neural networks for accurate face detection combined with OpenCV Haar cascades for eye detection
- **Grid Display**: Shows detected eyes in a fixed 6x2 grid layout at 1920x1080 resolution
- **Interactive Controls**: Toggle fullscreen, color modes, vertical flip, and capture screenshots
- **Debug Mode**: Optional debug view showing the full camera frame with bounding boxes
- **Performance Monitoring**: Real-time display of CPU usage, memory consumption, and chip temperature

## Hardware Requirements

- Intel OAK camera (OAK-D, OAK-1, etc.)
- Computer with USB 3.0+ port
- Sufficient processing power for real-time neural network inference

## Installation

1. Clone the repository:
```bash
git clone https://github.com/szkjn/INS_watching_you_looking_at_everything_but_me.git
cd INS_watching_you_looking_at_everything_but_me
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. **Important**: Comment out deprecated import in `.venv/lib/site-packages/fer/classes.py`, line 7:
```python
# from moviepy.editor import *
```

## Usage

Run the main application:
```bash
python main.py
```

### Controls

- **F**: Toggle fullscreen mode
- **C**: Toggle color/grayscale mode
- **R**: Toggle vertical flip
- **S**: Save screenshot with timestamp
- **Q**: Quit application

## Configuration

Edit `src/config.py` to modify:
- FPS settings
- Resolution (RGB_RESOLUTION)
- Neural network confidence threshold
- Debug mode toggle

## Project Structure

```
src/
├── display.py              # Main display logic and UI
├── face_detection.py       # DepthAI pipeline and detection processing
├── config.py              # Configuration parameters
├── performance_monitor.py  # System performance tracking
└── utils.py               # Utility functions

main.py                    # Entry point
requirements.txt           # Python dependencies
```

## Technical Details

- **Face Detection**: Uses MobileNet-based neural network (`face-detection-retail-0004`)
- **Eye Detection**: OpenCV Haar cascades within detected face regions
- **Display**: Fixed 6x2 grid layout that cycles through available eyes
- **Performance**: Optimized for real-time processing at 15 FPS

## Troubleshooting

- Ensure your OAK camera is properly connected via USB 3.0+
- Check that no other applications are using the camera
- Verify DepthAI drivers are installed correctly
- Try reducing FPS in config if experiencing performance issues
