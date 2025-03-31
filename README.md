# Face and Mouth Detection with Relay Control

## Overview
This project uses **OpenCV** for real-time face and mouth detection and controls a **USB relay module** to activate a solenoid when a face and mouth are detected for a specified duration. The relay is controlled via **serial communication**.

## Features
- Detects **faces and mouths** using OpenCV's Haar cascades.
- Activates a **USB relay-controlled solenoid** when a face is detected for a specific time and the mouth is detected.
- Deactivates the solenoid if the face is not detected for a defined duration.
- Displays the video feed with bounding boxes around detected faces and mouths.

## Requirements
Ensure you have the following dependencies installed:

```bash
pip install opencv-python pyserial numpy
```

## Hardware Requirements
- **USB Relay Module** (configured on `COM9`, update as needed)
- **Webcam** (for face and mouth detection)

## Usage
1. **Update the Serial Port**
   Modify the `serial.Serial('COM9', 9600, timeout=1)` line with the correct COM port for your USB relay module.

2. **Run the Script**
   ```bash
   python face_mouth_detection.py
   ```

3. **Control Mechanism**
   - The solenoid activates if a face is detected continuously for **more than 5 seconds** and a mouth is detected.
   - The solenoid deactivates if a face is **not detected for more than 5 seconds**.
   - Press `q` to exit the application.

## Output
- Live webcam feed with bounding boxes around **faces** (blue) and **mouths** (red).
- Console log displaying **face detection count** and **relay status**.
- Example console output:
  ```
  12 True  # Face detected, mouth detected → Solenoid ON
  -15 False # No face detected for a while → Solenoid OFF
  ```

## Notes
- Adjust `faceON` thresholds for tuning activation times.
- Ensure the USB relay is correctly connected and recognized by the system.
- Modify the Haar cascade parameters (`scaleFactor`, `minNeighbors`) to improve detection accuracy.

## License
This project is open-source and can be used for educational and research purposes.
