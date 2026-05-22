# Vehicle Detection

![WhatsApp Image 2026-02-20 at 5 35 47 PM](https://github.com/user-attachments/assets/b98461f2-405b-4b94-b8d4-743769373c01)

A real-time vehicle detection and tracking system using YOLOv8 with GPU support, designed to detect and count unique vehicles in images and videos.

## Features

✨ **Core Capabilities:**
- **Real-time Detection**: Fast and accurate vehicle detection using YOLOv8
- **Multi-format Support**: Process images (JPG, PNG, BMP) and videos (MP4, AVI, MOV, MKV)
- **Vehicle Tracking**: Unique vehicle identification and counting across frames
- **GPU Acceleration**: Automatic GPU detection for faster processing
- **FPS Monitoring**: Real-time frame rate display during video processing
- **User-friendly GUI**: Simple Tkinter interface for file selection

## Tech Stack

- **Model**: YOLOv8 (Ultralytics)
- **Framework**: PyTorch
- **GUI**: Tkinter
- **Computer Vision**: OpenCV
- **Language**: Python

## Installation

### Prerequisites
- Python 3.8 or higher
- CUDA Toolkit (optional, for GPU acceleration)

### Setup Steps

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Chandan-328/vehicle-detection.git
   cd vehicle-detection
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   
   Or install manually:
   ```bash
   pip install ultralytics opencv-python torch torchvision
   ```

3. **Verify Installation:**
   ```bash
   python check_classes.py
   ```

## Usage

### Running the Application

Start the GUI launcher:
```bash
python app.py
```

1. Click "Select File" button
2. Choose an image or video file from your device
3. The application will:
   - Load the pre-trained YOLOv8 model (`best.pt`)
   - Run detection and tracking
   - Display annotated results with vehicle count and FPS
4. Press **'q'** to close the display window

### Supported File Formats

**Images:**
- `.jpg`, `.jpeg`, `.png`, `.bmp`

**Videos:**
- `.mp4`, `.avi`, `.mov`, `.mkv`

## Model Details

### YOLOv8 Configuration
- **Model File**: `best.pt` (22.5 MB)
- **Detection Classes**: Vehicle detection
- **Confidence Threshold**: 0.3 (adjustable)
- **IoU Threshold**: 0.45 (adjustable)
- **Performance Optimization**: Automatic frame resizing to 640px width for speed

### Key Parameters

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `conf` | 0.3 | Detection confidence threshold |
| `iou` | 0.45 | Intersection over Union threshold |
| `target_width` | 640 | Frame resize width for performance |
| `persist` | True | Enable object tracking across frames |

## Features Explained

### Vehicle Counting
- Tracks unique vehicles using persistent IDs
- Counts each vehicle only once as it passes through the scene
- Displays total unique vehicle count on output

### Performance Optimization
- Resizes high-resolution videos to 640px width
- Reduces processing time while maintaining accuracy
- Shows real-time FPS for performance monitoring

### GPU Support
- Automatically detects and uses CUDA GPU if available
- Falls back to CPU processing if GPU unavailable
- Significant speed improvement on supported hardware

## Project Structure

```
vehicle-detection/
├── app.py                          # Main GUI application
├── best.pt                         # Pre-trained YOLOv8 model
├── check_classes.py                # Utility to verify model classes
├── istockphoto-2193558699-640_adpp_is.mp4  # Sample video
└── README.md                       # This file
```

## Example Output

**Video Processing:**
```
Using device: 0
Total Unique Vehicles: 15
FPS: 28
```

**Image Processing:**
```
Using device: cpu
Total Vehicles: 8
```

## Performance Tips

💡 **For Best Results:**
- Use clear, well-lit videos/images
- Ensure vehicles are visible and not heavily obscured
- Use GPU for significantly faster processing on large files
- Reduce video resolution if processing is slow
- Adjust `conf` parameter (0.3-0.7) for different detection sensitivity

## Customization

### Adjusting Detection Sensitivity

Edit `app.py` line 52-59:
```python
results = model.track(
    frame_resized, 
    persist=True, 
    conf=0.5,      # ← Increase for stricter detection (0.5-0.8)
    iou=0.45,      # ← Adjust IoU threshold if needed
    device=device,
    verbose=False
)
```

### Changing Label Names

Edit `app.py` line 21-22:
```python
if 1 in model.names and model.names[1] == 'Car':
    model.names[1] = 'Your Custom Label'
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Model not found | Ensure `best.pt` is in the same directory as `app.py` |
| Slow processing | Enable GPU or reduce video resolution |
| No detections | Lower confidence threshold or use clearer footage |
| CUDA not detected | Install NVIDIA drivers and CUDA Toolkit |

## Future Enhancements

- [ ] Multi-object tracking (pedestrians, bikes, etc.)
- [ ] Video export with annotations
- [ ] Performance analytics and statistics
- [ ] Configurable detection parameters in GUI
- [ ] Real-time camera feed support
- [ ] Cloud deployment option

## License

This project is open source and available for educational and research purposes.

## Author

**Chandan-328** - [@Chandan-328](https://github.com/Chandan-328)

## Acknowledgments

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) - Object detection model
- [OpenCV](https://opencv.org/) - Computer vision library
- [PyTorch](https://pytorch.org/) - Deep learning framework

## Support

For issues, questions, or suggestions, please [open an issue](https://github.com/Chandan-328/vehicle-detection/issues) on GitHub.

---

**Made with ❤️ for vehicle detection tasks**
