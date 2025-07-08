import cv2
import numpy as np
import torch
import torch.nn.functional as F
import time
from collections import deque
import matplotlib.pyplot as plt

# Import our modules
from models.swin_transformer import SwingTransformerVideoProcessor
from utils.landmark_processor import FacialLandmarkProcessor
from utils.visualization import VideoVisualizer

class RealTimeVideoProcessor:
    """Real-time video processing with Swing Transformer"""

    def __init__(self, sequence_length=30):
        self.sequence_length = sequence_length
        self.landmark_processor = FacialLandmarkProcessor()
        self.visualizer = VideoVisualizer()

        # Initialize Swing Transformer
        self.model = SwingTransformerVideoProcessor(
            landmark_dim=136,  # 68 landmarks * 2 coordinates = 136 features
            embed_dim=128,
            num_layers=4,
            num_heads=8,
            sequence_length=sequence_length
        )
        self.model.eval()

        # Landmark sequence buffer
        self.landmark_buffer = deque(maxlen=sequence_length)

        # Emotion labels
        self.emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

    def normalize_landmarks(self, landmarks):
        """Normalize landmarks to [-1, 1] range"""
        if landmarks is None:
            return None

        # Center around mean
        center = landmarks.mean(axis=0)
        centered = landmarks - center

        # Scale to [-1, 1]
        scale = np.max(np.abs(centered))
        if scale > 0:
            normalized = centered / scale
        else:
            normalized = centered

        return normalized

    def process_frame(self, frame):
        """Process a single frame - ENHANCED VERSION"""
        # Extract landmarks
        landmarks = self.landmark_processor.extract_landmarks(frame)

        if landmarks is not None:
            # Normalize landmarks
            normalized_landmarks = self.normalize_landmarks(landmarks)

            # Add to buffer
            if normalized_landmarks is not None:
                self.landmark_buffer.append(normalized_landmarks)

            # Process with transformer
            emotions = None
            attention_weights = None
            buffer_status = f"Buffer: {len(self.landmark_buffer)}/{self.sequence_length}"

            if len(self.landmark_buffer) >= 5:  # Start predictions with fewer frames
                # Prepare input tensor
                current_buffer = list(self.landmark_buffer)
                
                if len(current_buffer) >= self.sequence_length:
                    # Full sequence analysis
                    sequence = np.array(current_buffer)
                    sequence_flat = sequence.reshape(sequence.shape[0], -1)
                    sequence_tensor = torch.FloatTensor(sequence_flat).unsqueeze(0)

                    with torch.no_grad():
                        emotions, attention_weights = self.model(sequence_tensor, single_frame_mode=False)
                        attention_weights = attention_weights[0].numpy()
                    
                    buffer_status = "Full Sequence Analysis"
                else:
                    # Single frame analysis for early predictions
                    current_landmarks = normalized_landmarks.flatten()
                    landmarks_tensor = torch.FloatTensor(current_landmarks).unsqueeze(0)
                    
                    with torch.no_grad():
                        emotions, attention_weights = self.model(landmarks_tensor, single_frame_mode=True)
                        if isinstance(attention_weights, torch.Tensor):
                            attention_weights = attention_weights[0].numpy()
                    
                    buffer_status = f"Single Frame Analysis ({len(current_buffer)} frames)"

            else:
                buffer_status = f"Collecting frames... ({len(self.landmark_buffer)}/5)"

            # Draw landmarks and info using visualizer
            frame_with_landmarks = self.visualizer.draw_landmarks_with_features(
                frame, landmarks, attention_weights
            )

            # Add info panel
            feature_count = len(landmarks) if landmarks is not None else 0
            final_frame = self.visualizer.draw_info_panel(
                frame_with_landmarks, emotions, attention_weights, 
                feature_count, buffer_status, self.emotion_labels
            )

            return final_frame

        else:
            # No face detected
            no_face_panel = self.visualizer.draw_info_panel(
                frame, None, None, 0, "No face detected", self.emotion_labels
            )
            return no_face_panel

        return frame

def main(video_path=None, output_path=None, max_frames=None):
    """Main function for Colab video processing"""
    print("Initializing Swing Transformer Video Processor...")
    processor = RealTimeVideoProcessor(sequence_length=30)

    # Initialize video capture
    if video_path is None:
        print("Please provide a video file path")
        return

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video file: {video_path}")
        return

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video properties: {width}x{height}, {fps} FPS, {total_frames} frames")

    # Setup output video writer if output path is provided
    out = None
    if output_path:
        # Output will be wider due to info panel
        output_width = width + 400  # Updated panel width
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (output_width, height))
        print(f"Output will be saved to: {output_path}")

    # Process frames
    frame_count = 0
    processed_frames = []

    print("Processing video frames...")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Finished processing all frames")
            break

        frame_count += 1

        # Process frame
        processed_frame = processor.process_frame(frame)

        # Add frame counter
        cv2.putText(processed_frame, f"Frame: {frame_count}/{total_frames}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Save to output video
        if out is not None:
            out.write(processed_frame)

        # Store processed frames for display
        processed_frames.append(processed_frame.copy())
        if len(processed_frames) > 16:
            processed_frames.pop(0)
            
        # Progress update
        if frame_count % 30 == 0:
            progress = (frame_count / total_frames) * 100
            print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames}) - Emotions: {'Active' if len(processor.landmark_buffer) >= 5 else 'Buffering'}")

        # Stop if max_frames limit reached
        if max_frames and frame_count >= max_frames:
            print(f"Reached maximum frame limit: {max_frames}")
            break

    # Cleanup
    cap.release()
    if out is not None:
        out.release()
        print(f"Output video saved successfully: {output_path}")

    print(f"Video processing completed. Processed {frame_count} frames.")

    # Display sample frames in Colab
    if processed_frames:
        print("\nDisplaying sample processed frames:")
        try:
            from IPython.display import display, Image
            
            # Show 16 processed frames in 4x4 grid
            fig, axes = plt.subplots(4, 4, figsize=(24, 18))
            axes = axes.flatten()

            for i, frame in enumerate(processed_frames[:16]):
                if i < len(axes):
                    # Convert BGR to RGB for matplotlib
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    axes[i].imshow(rgb_frame)
                    axes[i].set_title(f'Frame {i+1}', fontsize=12)
                    axes[i].axis('off')

            # Hide any unused subplots
            for i in range(len(processed_frames), len(axes)):
                axes[i].axis('off')

            plt.tight_layout()
            plt.show()
        except ImportError:
            print("IPython not available, skipping frame display")

    return processed_frames

# Colab-specific helper functions
def process_uploaded_video(uploaded_file_path, output_name="processed_output_video.mp4", max_frames=300):
    """Process uploaded video in Colab"""
    output_path = f"/examples/{output_name}"
    return main(uploaded_file_path, output_path, max_frames)

def process_sample_frames(video_path, num_frames=50):
    """Process only first N frames for quick testing"""
    print(f"Processing first {num_frames} frames for quick preview...")
    return main(video_path, None, num_frames)

if __name__ == "__main__":
    # Example usage for Colab:
    print("Real-Time Facial Emotion Recognition with Swin Transformer")
    print("=" * 60)
    print("To use this code:")
    print("1. Upload your video file")
    print("2. Call: process_uploaded_video('/path/to/your_video_file.mp4')")
    print("3. Or for quick preview: process_sample_frames('/path/to/your_video_file.mp4', 50)")
    print("=" * 60)