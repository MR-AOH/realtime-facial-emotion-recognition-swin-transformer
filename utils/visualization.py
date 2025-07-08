import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

class FacialVisualization:
    """Visualization utilities for facial emotion recognition"""

    def __init__(self):
        # Colors for different facial regions (brighter colors for better visibility)
        self.region_colors = {
            'jaw': (255, 50, 50),      # Bright Red
            'eyebrows': (50, 255, 50),  # Bright Green
            'nose': (50, 50, 255),     # Bright Blue
            'eyes': (255, 255, 50),    # Bright Yellow
            'mouth': (255, 50, 255),   # Bright Magenta
        }

        # Define facial regions
        self.regions = {
            'jaw': list(range(0, 17)),
            'eyebrows': list(range(17, 27)),
            'nose': list(range(27, 36)),
            'eyes': list(range(36, 48)),
            'mouth': list(range(48, 68))
        }

        # Emotion labels
        self.emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

    def draw_landmarks_with_features(self, image, landmarks, attention_weights=None, active_features=None):
        """Draw landmarks with feature highlighting - ENHANCED VERSION"""
        if landmarks is None:
            return image

        # Create overlay
        overlay = image.copy()

        # Draw landmarks by region with LARGER sizes
        for region_name, indices in self.regions.items():
            color = self.region_colors[region_name]

            for i, idx in enumerate(indices):
                if idx < len(landmarks):
                    x, y = int(landmarks[idx][0]), int(landmarks[idx][1])

                    # Determine point size based on attention/importance - MUCH LARGER
                    base_radius = 6  # Increased from 3
                    if attention_weights is not None and idx < len(attention_weights):
                        # Scale radius based on attention weight
                        attention_val = float(attention_weights[idx])
                        radius = max(4, int(6 + attention_val * 8))  # Larger range
                    else:
                        radius = base_radius

                    # Draw outer circle (border) for better visibility
                    cv2.circle(overlay, (x, y), radius + 2, (255, 255, 255), 2)  # White border
                    # Draw main point
                    cv2.circle(overlay, (x, y), radius, color, -1)

        # Connect landmarks within regions with thicker lines
        self.draw_connections(overlay, landmarks)

        # Blend overlay with original image
        alpha = 0.5  # Increased opacity for better visibility
        image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

        return image

    def draw_connections(self, image, landmarks):
        """Draw connections between landmarks with thicker lines"""
        if landmarks is None or len(landmarks) < 68:
            return

        line_thickness = 1  # Increased from 1

        # Jaw line connections
        jaw_indices = self.regions['jaw']
        for i in range(len(jaw_indices) - 1):
            if jaw_indices[i] < len(landmarks) and jaw_indices[i+1] < len(landmarks):
                pt1 = tuple(map(int, landmarks[jaw_indices[i]]))
                pt2 = tuple(map(int, landmarks[jaw_indices[i+1]]))
                cv2.line(image, pt1, pt2, self.region_colors['jaw'], line_thickness)

        # Eye connections (simplified)
        eye_regions = [(36, 42), (42, 48)]  # Right and left eye ranges
        for start, end in eye_regions:
            for i in range(start, end - 1):
                if i < len(landmarks) and i + 1 < len(landmarks):
                    pt1 = tuple(map(int, landmarks[i]))
                    pt2 = tuple(map(int, landmarks[i + 1]))
                    cv2.line(image, pt1, pt2, self.region_colors['eyes'], line_thickness)

        # Mouth connections
        mouth_indices = self.regions['mouth']
        for i in range(len(mouth_indices) - 1):
            if mouth_indices[i] < len(landmarks) and mouth_indices[i+1] < len(landmarks):
                pt1 = tuple(map(int, landmarks[mouth_indices[i]]))
                pt2 = tuple(map(int, landmarks[mouth_indices[i+1]]))
                cv2.line(image, pt1, pt2, self.region_colors['mouth'], line_thickness)

    def draw_info_panel(self, image, emotions, attention_weights, feature_count, buffer_status=""):
        """Draw information panel with predictions and feature stats - FIXED VERSION"""
        h, w = image.shape[:2]
        panel_width = 400  # Increased width for better visibility
        panel_height = h

        # Create info panel with darker background
        panel = np.zeros((panel_height, panel_width, 3), dtype=np.uint8)
        panel.fill(20)  # Very dark background

        # Add border to panel
        cv2.rectangle(panel, (0, 0), (panel_width-1, panel_height-1), (100, 100, 100), 2)

        y_offset = 30
        line_height = 25
        font_scale = 0.6
        font_thickness = 2

        # Title with background
        title_bg = np.zeros((40, panel_width, 3), dtype=np.uint8)
        title_bg.fill(60)
        panel[0:40, :] = title_bg
        cv2.putText(panel, "SWING TRANSFORMER ANALYSIS", (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y_offset = 60

        # Buffer status
        status_color = (0, 255, 255) if "Ready" in buffer_status else (255, 255, 0)
        cv2.putText(panel, f"Status: {buffer_status}", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)
        y_offset += line_height

        # Feature count
        cv2.putText(panel, f"Active Features: {feature_count}/68", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        y_offset += line_height + 10

        # Section separator
        cv2.line(panel, (10, y_offset), (panel_width-10, y_offset), (100, 100, 100), 1)
        y_offset += 20

        # Emotion predictions section
        cv2.putText(panel, "EMOTION PREDICTIONS:", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_offset += 30

        if emotions is not None:
            # Get emotion probabilities
            if isinstance(emotions, torch.Tensor):
                emotion_probs = F.softmax(emotions, dim=1)[0].detach().numpy()
            else:
                emotion_probs = F.softmax(torch.tensor(emotions), dim=0).numpy()

            # Sort emotions by probability for better display
            emotion_data = list(zip(self.emotion_labels, emotion_probs))
            emotion_data.sort(key=lambda x: x[1], reverse=True)

            for i, (label, prob) in enumerate(emotion_data):
                # Color coding based on probability
                if prob > 0.4:
                    color = (0, 255, 0)  # High confidence - Green
                elif prob > 0.2:
                    color = (0, 255, 255)  # Medium confidence - Yellow
                else:
                    color = (128, 128, 128)  # Low confidence - Gray

                # Emotion label and value
                cv2.putText(panel, f"{label}: {prob:.3f}", (15, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                # Draw probability bar
                bar_width = int(prob * 250)  # Increased bar width
                bar_height = 8
                # Background bar
                cv2.rectangle(panel, (15, y_offset + 8), (265, y_offset + 16), (50, 50, 50), -1)
                # Probability bar
                if bar_width > 0:
                    cv2.rectangle(panel, (15, y_offset + 8), (15 + bar_width, y_offset + 16), color, -1)

                y_offset += 28

        else:
            cv2.putText(panel, "No emotions detected", (15, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)
            y_offset += line_height

        # Section separator
        y_offset += 10
        cv2.line(panel, (10, y_offset), (panel_width-10, y_offset), (100, 100, 100), 1)
        y_offset += 20

        # Attention visualization section
        cv2.putText(panel, "FEATURE ATTENTION:", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_offset += 30

        # Region attention summary
        if attention_weights is not None:
            region_attention = {}
            att_weights = attention_weights if isinstance(attention_weights, np.ndarray) else attention_weights.numpy()
            
            for region_name, indices in self.regions.items():
                region_att = np.mean([att_weights[i] if i < len(att_weights) else 0.1 for i in indices])
                region_attention[region_name] = region_att

            # Sort regions by attention
            sorted_regions = sorted(region_attention.items(), key=lambda x: x[1], reverse=True)

            for region_name, att_val in sorted_regions:
                color = self.region_colors[region_name]
                cv2.putText(panel, f"{region_name.upper()}: {att_val:.3f}", (15, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                # Draw attention bar
                bar_width = int(att_val * 200)
                # Background bar
                cv2.rectangle(panel, (15, y_offset + 8), (215, y_offset + 16), (50, 50, 50), -1)
                # Attention bar
                if bar_width > 0:
                    cv2.rectangle(panel, (15, y_offset + 8), (15 + bar_width, y_offset + 16), color, -1)

                y_offset += 28

        else:
            cv2.putText(panel, "Computing attention...", (15, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)

        # Combine panel with main image
        combined = np.hstack([image, panel])
        return combined

    def draw_simple_landmarks(self, image, landmarks, color=(0, 255, 0), radius=3):
        """Draw simple landmarks without attention visualization"""
        if landmarks is None:
            return image

        overlay = image.copy()
        
        for point in landmarks:
            x, y = int(point[0]), int(point[1])
            cv2.circle(overlay, (x, y), radius, color, -1)

        return overlay

    def draw_emotion_bar(self, image, emotions, position=(10, 50)):
        """Draw emotion probability bar on image"""
        if emotions is None:
            return image

        # Get emotion probabilities
        if isinstance(emotions, torch.Tensor):
            emotion_probs = F.softmax(emotions, dim=1)[0].detach().numpy()
        else:
            emotion_probs = F.softmax(torch.tensor(emotions), dim=0).numpy()

        x, y = position
        bar_width = 200
        bar_height = 20
        
        # Draw background
        cv2.rectangle(image, (x-5, y-5), (x + bar_width + 5, y + (bar_height + 5) * len(self.emotion_labels)), (50, 50, 50), -1)
        
        for i, (label, prob) in enumerate(zip(self.emotion_labels, emotion_probs)):
            # Color coding
            if prob > 0.4:
                color = (0, 255, 0)  # Green
            elif prob > 0.2:
                color = (0, 255, 255)  # Yellow
            else:
                color = (128, 128, 128)  # Gray

            # Draw label
            cv2.putText(image, f"{label}: {prob:.2f}", (x, y + i * (bar_height + 5) + 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

            # Draw probability bar
            prob_width = int(prob * bar_width)
            cv2.rectangle(image, (x + 80, y + i * (bar_height + 5)), 
                         (x + 80 + prob_width, y + i * (bar_height + 5) + bar_height), color, -1)

        return image

    def create_attention_heatmap(self, image, landmarks, attention_weights):
        """Create attention heatmap overlay"""
        if landmarks is None or attention_weights is None:
            return image

        # Create heatmap overlay
        heatmap = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
        
        att_weights = attention_weights if isinstance(attention_weights, np.ndarray) else attention_weights.numpy()
        
        for i, point in enumerate(landmarks):
            if i < len(att_weights):
                x, y = int(point[0]), int(point[1])
                if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
                    # Create gaussian around landmark
                    sigma = 20
                    for dy in range(-sigma, sigma+1):
                        for dx in range(-sigma, sigma+1):
                            nx, ny = x + dx, y + dy
                            if 0 <= nx < image.shape[1] and 0 <= ny < image.shape[0]:
                                dist = np.sqrt(dx*dx + dy*dy)
                                if dist <= sigma:
                                    weight = att_weights[i] * np.exp(-(dist*dist) / (2*sigma*sigma/4))
                                    heatmap[ny, nx] = max(heatmap[ny, nx], weight)

        # Normalize heatmap
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()

        # Convert to color heatmap
        heatmap_color = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
        
        # Blend with original image
        alpha = 0.4
        result = cv2.addWeighted(image, 1-alpha, heatmap_color, alpha, 0)
        
        return result

    def plot_emotion_timeline(self, emotion_history, save_path=None):
        """Plot emotion timeline using matplotlib"""
        if not emotion_history:
            return

        # Convert to numpy array
        emotions_array = np.array(emotion_history)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot each emotion
        for i, label in enumerate(self.emotion_labels):
            ax.plot(emotions_array[:, i], label=label, linewidth=2)
        
        ax.set_xlabel('Frame')
        ax.set_ylabel('Emotion Probability')
        ax.set_title('Emotion Recognition Timeline')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.tight_layout()
        plt.show()

    def create_summary_visualization(self, processed_frames, num_display=16):
        """Create a summary visualization of processed frames"""
        if not processed_frames:
            return

        # Show processed frames in grid
        fig, axes = plt.subplots(4, 4, figsize=(24, 18))
        axes = axes.flatten()

        for i, frame in enumerate(processed_frames[:num_display]):
            if i < len(axes):
                # Convert BGR to RGB for matplotlib
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                axes[i].imshow(rgb_frame)
                axes[i].set_title(f'Frame {i+1}', fontsize=12)
                axes[i].axis('off')

        # Hide unused subplots
        for i in range(len(processed_frames), len(axes)):
            axes[i].axis('off')

        plt.tight_layout()
        plt.show()

    def get_region_colors(self):
        """Get the region colors dictionary"""
        return self.region_colors

    def get_emotion_labels(self):
        """Get the emotion labels list"""
        return self.emotion_labels

    def add_frame_info(self, image, frame_count, total_frames, fps=30):
        """Add frame information to image"""
        info_text = f"Frame: {frame_count}/{total_frames} | FPS: {fps}"
        cv2.putText(image, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        return image

    def add_timestamp(self, image, timestamp):
        """Add timestamp to image"""
        cv2.putText(image, f"Time: {timestamp:.2f}s", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        return image