
# Object Detection with YOLOv8 and TensorRT

This project demonstrates how to run object detection using YOLOv8 models, optimize them using TensorRT for faster inference on NVIDIA GPUs, and apply the detection models to images or videos. It provides an efficient framework for handling model conversion, inference, and output visualization.

## Features

- **YOLOv8 Integration**: Perform object detection using the state-of-the-art YOLOv8 model for high-accuracy inference.
- **TensorRT Optimization**: Convert YOLOv8 models into TensorRT engines for accelerated inference on NVIDIA GPUs, achieving significantly improved FPS.
- **Video Processing**: Perform detection on video streams, calculate FPS, and save processed videos with bounding boxes and FPS overlay.
- **Visualization**: Draw bounding boxes, display class labels and confidence scores, and overlay FPS information on the output images or videos.
- **Modular Design**: Separate functions for model loading, preprocessing, inference, post-processing, and visualization for easy customization.

---

## Requirements

### Software

- **Python 3.6+** (Recommended Python 3.8+ for better support)
- **GPU**: NVIDIA GPU with CUDA support is highly recommended for TensorRT optimizations.
- **CUDA**: Ensure CUDA is installed for TensorRT support.

### Libraries

- **PyTorch**
- **OpenCV (`cv2`)**
- **YOLOv8 Library (Ultralytics)**
- **TensorRT**: Used for optimizing the YOLOv8 model.

Install necessary libraries via `pip`:

```bash
pip install torch ultralytics opencv-python tensorrt tensorrt_lean tensorrt_dispatch
```

---

## Workflow

### 1. Install and Set Up the Environment

- Ensure you have the required dependencies installed. You can use `pip` commands to install the libraries mentioned above.
  
```bash
pip install torch ultralytics opencv-python tensorrt tensorrt_lean tensorrt_dispatch
```

### 2. Download and Prepare Models

- Download the pre-trained YOLOv8 model:

```bash
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt
```

- Convert the YOLOv8 PyTorch model to TensorRT for faster inference using the `yolo` command:

```bash
yolo export model=yolov8s.pt format=engine half=True device=0 workspace=12
```

This creates a TensorRT engine with FP16 precision.

### 3. Perform Object Detection

- You can run inference using the TensorRT engine on an image:

```bash
yolo detect predict model=yolov8s.engine source="https://ultralytics.com/images/bus.jpg" device=0
```

### 4. Video Processing

To process video input, use the provided detection code that reads frames, performs detection, and overlays results (bounding boxes, FPS):

```python
# Sample video detection
detection("yolov8s.engine", "inference/people.mp4", "detection")
```

The function will process the video, detect objects, and save the output video with bounding boxes and FPS information.

---

## Code Overview

### Model Inference

1. **`tensorrt_detection`**: Handles object detection using the TensorRT model.
2. **`yolov8_detection`**: Inference using the standard YOLOv8 PyTorch model.
3. **`detection`**: Main function for processing images or video streams, detecting objects, calculating FPS, and saving the output.


---

## Example

1. Convert YOLOv8 to TensorRT engine:
   ```bash
   yolo export model=yolov8s.pt format=engine half=True device=0
   ```

2. Perform inference using TensorRT engine:
   ```bash
   detection("yolov8s.engine", "inference/people.mp4", "detection")
   ```

3. Output video will be saved with detection results and FPS overlay.

---

## Future Improvements

- **Real-Time Inference**: Implement real-time detection using a webcam or live video stream.
- **Quantization**: Explore model quantization techniques for further speed-up.
- **Model Tracking**: Integrate object tracking for continuous video streams.

--- 
