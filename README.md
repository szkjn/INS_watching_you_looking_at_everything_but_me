# INS - Watching You Looking at Everything but Me

An interactive eye detection and display system using Luxonis OAK-D cameras and DepthAI technology. This project streams an interactive video feed that tracks faces and eyes, displaying them in a grid layout.
A video installation by Edda Rabold and Jun Suzuki ([more info here](https://www.linkedin.com/posts/jun-suzuki_depthai-vsa-technopolice-activity-7402257607970766848-Y--z?utm_source=share&utm_medium=member_desktop&rcm=ACoAABKqUGABmdBONs3Be4UTvj4Q9cK4ihZAh5A)).

## Features

- **Real-time Face & Eye Detection**: Uses DepthAI neural networks for accurate face detection combined with OpenCV Haar cascades for eye detection
- **Grid Display**: Shows detected eyes in a 3x9 grid layout at 2560x1440 resolution
- **Multiple Display Modes**: 
  - FULL_GRID: Cycles through detected eyes to fill all grid positions
  - PARSE_MODE_X3: Variable multiplication based on eye count
  - PARSE_MODE_X2: All eyes doubled and spread randomly
- **Interactive Controls**: Toggle fullscreen, color modes, vertical flip, font cycling, display modes, and capture screenshots

## Hardware Requirements

- OAK-D camera
- Computer with USB 3.0+ port

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

## Usage

Run the main application:
```bash
python main.py
```

### Controls

- **F**: Toggle fullscreen mode
- **C**: Toggle color/grayscale mode
- **R**: Toggle vertical flip
- **A**: Cycle through available fonts
- **W**: Switch between FULL_GRID and PARSE_MODE_X3
- **X**: Toggle between PARSE_MODE_X2 and PARSE_MODE_X3
- **S**: Save screenshot with timestamp
- **Q**: Quit application

## Configuration

Edit `src/config.py` to modify:
- FPS settings
- Resolution (RGB_RESOLUTION)
- Neural network confidence threshold
- Grid layout (GRID_ROWS, GRID_COLS)
- Display mode (DISPLAY_MODE)
- Eye crop scaling (EYE_CROP_SCALE_X, EYE_CROP_SCALE_Y)

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
- **Display**: 3x9 grid layout (27 positions) with multiple display modes
- **Performance**: Optimized for real-time processing at 15 FPS at 1440p resolution

## Troubleshooting

- Ensure your OAK camera is properly connected via USB 3.0+
- Check that no other applications are using the camera
- Verify DepthAI drivers are installed correctly
- Try reducing FPS in config if experiencing performance issues
