# Real-Time Facial Emotion Recognition with Swin Transformer

A sophisticated real-time facial emotion recognition system that combines Swin Transformer architecture with MediaPipe facial landmarks for accurate emotion detection from video streams. Features an intuitive analysis dashboard showing emotion predictions, attention weights, and facial feature tracking.

![WhatsApp Image 2025-06-29 at 1 01 48 PM](https://github.com/user-attachments/assets/5e609aec-5e7c-4066-baf7-9c99120fa282)

## 🌟 Key Features

- 🎯 **Real-Time Emotion Detection**: Instant emotion recognition from video frames
- 🧠 **Swin Transformer Architecture**: Advanced transformer-based deep learning model
- 📊 **Live Analysis Dashboard**: Real-time visualization of emotions and attention weights
- 🎨 **Facial Landmark Visualization**: Color-coded facial feature tracking (68 key points)
- 📈 **Attention Mechanism**: Visual feedback on which facial regions influence predictions
- 🎥 **Video Processing**: Batch processing of video files with output generation
- 📱 **Google Colab Ready**: Optimized for cloud-based execution

## 🎭 Supported Emotions

The system recognizes 7 fundamental emotions:

- 😠 **Angry**
- 🤢 **Disgust**
- 😨 **Fear**
- 😊 **Happy**
- 😢 **Sad**
- 😲 **Surprise**
- 😐 **Neutral**

## 🏗️ Architecture Overview

```
Input Video → MediaPipe Face Detection → Facial Landmarks → Swin Transformer → Emotion Prediction
                                                          ↓
                                    Attention Visualization ← Analysis Dashboard
```

### Core Components:

- **MediaPipe Integration**: Robust facial landmark detection (68 key points)
- **Swin Transformer Model**: Window-based attention mechanism for sequence processing
- **Real-Time Analysis**: Dual-mode prediction (single frame + sequence analysis)
- **Visual Dashboard**: Comprehensive emotion and attention visualization

## 🚀 Quick Start

### Prerequisites

```bash
python >= 3.8
opencv-python >= 4.5.0
torch >= 1.9.0
mediapipe >= 0.8.0
numpy >= 1.21.0
matplotlib >= 3.3.0
```

### Installation

1. **Clone the repository**

```bash
git clone https://github.com/yourusername/realtime-facial-emotion-recognition-swin-transformer.git
cd realtime-facial-emotion-recognition-swin-transformer
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

3. **Run the application**

```bash
python emotion_recognition.py
```

## 📋 Requirements.txt

```txt
opencv-python>=4.5.0
torch>=1.9.0
torchvision>=0.10.0
mediapipe>=0.8.0
numpy>=1.21.0
matplotlib>=3.3.0
```
OR you can simply use the "notebook_for_model.ipynb" file if you prefer to work with nookbooks.
## 💻 Usage

### 🎥 Video Processing

#### Basic Usage
```python
from emotion_recognition import process_uploaded_video

# Process uploaded video
processed_frames = process_uploaded_video("/path/to/your/video.mp4")
```

#### Advanced Usage
```python
from emotion_recognition import main

# Full processing with output video
processed_frames = main(
    video_path="/path/to/input/video.mp4",
    output_path="/path/to/output/processed_video.mp4",
    max_frames=None  # Process all frames
)
```

#### Quick Preview
```python
from emotion_recognition import process_sample_frames

# Process only first 50 frames for quick testing
processed_frames = process_sample_frames("/path/to/video.mp4", num_frames=50)
```

## 🔧 Configuration Options

### Model Parameters
```python
processor = RealTimeVideoProcessor(
    sequence_length=30,  # Number of frames for sequence analysis
)

model = SwingTransformerVideoProcessor(
    landmark_dim=136,     # 68 landmarks × 2 coordinates
    embed_dim=128,        # Embedding dimension
    num_layers=4,         # Number of transformer layers
    num_heads=8,          # Attention heads
    sequence_length=30    # Input sequence length
)
```

## 🎨 Visual Output

The system provides rich visual feedback:

### 📊 Analysis Dashboard

- **Emotion Predictions**: Real-time probability scores for all 7 emotions
- **Feature Attention**: Attention weights for different facial regions
- **Buffer Status**: Current analysis mode and frame count
- **Color-coded Regions**: Jaw (Red), Eyebrows (Green), Nose (Blue), Eyes (Yellow), Mouth (Magenta)

### 🎯 Facial Landmarks

- **68 Key Points**: Comprehensive facial feature tracking
- **Attention-based Sizing**: Landmark size reflects importance
- **Region Connections**: Visual connections between related facial features

## 🔬 Technical Details

### Swin Transformer Architecture

- **Window-based Attention**: Efficient processing of facial landmark sequences
- **Multi-head Self-Attention**: Captures complex facial feature relationships
- **Hierarchical Feature Learning**: Progressive feature extraction at multiple scales

### Facial Landmark Processing

- **MediaPipe Integration**: Robust face detection and landmark extraction
- **Normalization**: Landmarks normalized to [-1, 1] range for model stability
- **Temporal Buffering**: Sliding window approach for sequence analysis

### Dual Prediction Modes

- **Single Frame Analysis**: Immediate emotion prediction (< 5 frames)
- **Sequence Analysis**: Full transformer processing (≥ 30 frames)

## 📁 Project Structure

```
realtime-facial-emotion-recognition-swin-transformer/
├── emotion_recognition.py          # Main application file
├── requirements.txt               # Python dependencies
├── README.md                     # Project documentation
├── examples/                     # Example videos and outputs
│   ├── sample_input.mp4
│   └── sample_output.mp4
├── models/                       # Model architecture files
│   ├── __init__.py
│   ├── swin_transformer.py
│   └── attention_mechanisms.py
└── utils/                        # Utility functions
    ├── __init__.py
    ├── landmark_processor.py
    └── visualization.py
```

## 🚀 Google Colab Usage

Perfect for cloud-based execution! Upload your video and run:

```python
# Upload video to Colab
from google.colab import files
uploaded = files.upload()

# Process the uploaded video
video_path = list(uploaded.keys())[0]
processed_frames = process_uploaded_video(f"/content/{video_path}")

# Download processed video
files.download("/content/processed_video.mp4")
```

## 🔧 Customization

### Adding New Emotions
```python
# Extend emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral', 'Custom']

# Update model output dimension
emotion_head = nn.Linear(embed_dim, 8)  # Changed from 7 to 8
```

### Modifying Facial Regions
```python
# Add new facial regions
regions = {
    'jaw': list(range(0, 17)),
    'eyebrows': list(range(17, 27)),
    'nose': list(range(27, 36)),
    'eyes': list(range(36, 48)),
    'mouth': list(range(48, 68)),
    'custom_region': [custom_landmark_indices]  # Add your custom region
}
```

## 🎯 Performance Optimization

### GPU Acceleration
```python
# Enable CUDA if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
```

### Batch Processing
```python
# Process multiple videos
video_paths = ["video1.mp4", "video2.mp4", "video3.mp4"]
for video_path in video_paths:
    process_uploaded_video(video_path)
```

## 📊 Results & Accuracy

- **Real-time Performance**: 30+ FPS on modern hardware
- **Emotion Recognition**: High accuracy across diverse facial expressions
- **Robustness**: Handles varying lighting conditions and face orientations
- **Attention Mechanism**: Provides interpretable insights into model decisions

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [MediaPipe](https://mediapipe.dev/): Google's framework for building perception pipelines
- [Swin Transformer](https://github.com/microsoft/Swin-Transformer): Microsoft's hierarchical vision transformer
- [PyTorch](https://pytorch.org/): Deep learning framework
- [OpenCV](https://opencv.org/): Computer vision library

## 📧 Contact

- **Author**: Your Name
- **Email**: your.email@example.com
- **Project Link**: [https://github.com/yourusername/realtime-facial-emotion-recognition-swin-transformer](https://github.com/yourusername/realtime-facial-emotion-recognition-swin-transformer)

## 🔮 Future Enhancements

- [ ] Multi-face emotion recognition
- [ ] Real-time webcam integration
- [ ] Mobile app deployment
- [ ] Custom emotion training
- [ ] 3D facial landmark support
- [ ] Audio-visual emotion fusion

---

⭐ **Star this repository if you find it helpful!** ⭐
