import cv2
import numpy as np
import mediapipe as mp
from collections import deque

class FacialLandmarkProcessor:
    """Process facial landmarks using MediaPipe"""

    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Key facial landmarks indices (68 key points similar to dlib)
        self.key_landmarks = [
            # Jaw line (17 points)
            10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400,
            # Right eyebrow (5 points)
            70, 63, 105, 66, 107,
            # Left eyebrow (5 points)
            296, 334, 293, 300, 276,
            # Nose (9 points)
            1, 2, 5, 4, 6, 19, 94, 125, 141,
            # Right eye (6 points)
            33, 7, 163, 144, 145, 153,
            # Left eye (6 points)
            362, 398, 384, 385, 386, 387,
            # Mouth (20 points)
            61, 84, 17, 314, 405, 320, 307, 375, 321, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 78
        ]

        # Define regions for landmark grouping
        self.regions = {
            'jaw': list(range(0, 17)),
            'eyebrows': list(range(17, 27)),
            'nose': list(range(27, 36)),
            'eyes': list(range(36, 48)),
            'mouth': list(range(48, 68))
        }

    def extract_landmarks(self, image):
        """Extract facial landmarks from image"""
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_image)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0]
            h, w = image.shape[:2]

            # Extract key landmarks
            key_points = []
            for idx in self.key_landmarks:
                if idx < len(landmarks.landmark):
                    point = landmarks.landmark[idx]
                    key_points.append([point.x * w, point.y * h])

            return np.array(key_points)

        return None

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

    def get_landmark_regions(self):
        """Get the landmark regions dictionary"""
        return self.regions

    def get_key_landmarks_indices(self):
        """Get the key landmarks indices"""
        return self.key_landmarks

    def extract_region_landmarks(self, landmarks, region_name):
        """Extract landmarks for a specific facial region"""
        if landmarks is None or region_name not in self.regions:
            return None
        
        region_indices = self.regions[region_name]
        region_landmarks = []
        
        for idx in region_indices:
            if idx < len(landmarks):
                region_landmarks.append(landmarks[idx])
        
        return np.array(region_landmarks) if region_landmarks else None

    def calculate_landmark_distances(self, landmarks):
        """Calculate distances between key landmark points"""
        if landmarks is None or len(landmarks) < 68:
            return None
        
        distances = {}
        
        # Eye distances
        if len(landmarks) >= 48:
            left_eye = landmarks[36:42]
            right_eye = landmarks[42:48]
            
            # Inter-eye distance
            left_eye_center = np.mean(left_eye, axis=0)
            right_eye_center = np.mean(right_eye, axis=0)
            distances['inter_eye'] = np.linalg.norm(left_eye_center - right_eye_center)
            
            # Eye aspect ratios
            distances['left_eye_ratio'] = self._calculate_eye_aspect_ratio(left_eye)
            distances['right_eye_ratio'] = self._calculate_eye_aspect_ratio(right_eye)
        
        # Mouth distances
        if len(landmarks) >= 68:
            mouth = landmarks[48:68]
            
            # Mouth aspect ratio
            distances['mouth_ratio'] = self._calculate_mouth_aspect_ratio(mouth)
            
            # Mouth width
            distances['mouth_width'] = np.linalg.norm(mouth[0] - mouth[6])
        
        return distances

    def _calculate_eye_aspect_ratio(self, eye_landmarks):
        """Calculate eye aspect ratio (EAR)"""
        if len(eye_landmarks) < 6:
            return 0.0
        
        # Vertical eye landmarks
        A = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
        B = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])
        
        # Horizontal eye landmark
        C = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])
        
        # EAR formula
        ear = (A + B) / (2.0 * C) if C > 0 else 0.0
        return ear

    def _calculate_mouth_aspect_ratio(self, mouth_landmarks):
        """Calculate mouth aspect ratio (MAR)"""
        if len(mouth_landmarks) < 20:
            return 0.0
        
        # Vertical mouth landmarks
        A = np.linalg.norm(mouth_landmarks[2] - mouth_landmarks[10])
        B = np.linalg.norm(mouth_landmarks[4] - mouth_landmarks[8])
        
        # Horizontal mouth landmark
        C = np.linalg.norm(mouth_landmarks[0] - mouth_landmarks[6])
        
        # MAR formula
        mar = (A + B) / (2.0 * C) if C > 0 else 0.0
        return mar

class LandmarkBuffer:
    """Buffer for storing landmark sequences"""
    
    def __init__(self, max_length=30):
        self.buffer = deque(maxlen=max_length)
        self.max_length = max_length
    
    def add_landmarks(self, landmarks):
        """Add landmarks to buffer"""
        if landmarks is not None:
            self.buffer.append(landmarks)
    
    def get_sequence(self, length=None):
        """Get landmark sequence from buffer"""
        if length is None:
            length = len(self.buffer)
        
        if len(self.buffer) < length:
            return None
        
        return list(self.buffer)[-length:]
    
    def is_ready(self, min_length=5):
        """Check if buffer has enough frames for processing"""
        return len(self.buffer) >= min_length
    
    def get_buffer_status(self):
        """Get current buffer status"""
        return f"{len(self.buffer)}/{self.max_length}"
    
    def clear(self):
        """Clear the buffer"""
        self.buffer.clear()