# Object-Counting-Bottles

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Welcome to the **Object-Counting-Bottles** project! This repository tracks and counts Juice, Water, and CocaCola bottles in videos using the YOLOv8 object detection model. It provides enhanced accuracy, error handling, confidence thresholds, and multiple counting zones.

## Features
- Detects and tracks bottles in video files
- Counts objects crossing a defined line (IN/OUT)
- Visualizes bounding boxes, labels, and info panels
- Outputs processed video and JSON results

## Requirements
See `requirements.txt` for dependencies.

## Setup
1. Clone this repository.
2. Place your YOLOv8 model weights file (e.g., `best.pt`) in the project directory.
3. Place your input video file (e.g., `video3.mp4`) in the project directory.
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
Run the main script:
```bash
python run_main.py
```

- Input video: `video3.mp4`
- Output video: `video1R.mp4`
- Results JSON: `counting_results.json`

You can change these filenames in `run_main.py`.

## Output
- Annotated video with bounding boxes and counts
- JSON file with total counts per class

## Notes
- Requires a trained YOLOv8 model (`best.pt`).
- Adjust the counting line and class names in `run_main.py` as needed. 