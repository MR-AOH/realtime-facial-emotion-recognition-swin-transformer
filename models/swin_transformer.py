import torch
import torch.nn as nn
import torch.nn.functional as F
from .attention_mechanisms import WindowAttention, MLP

class SwinTransformerBlock(nn.Module):
    """Swin Transformer Block for processing facial landmark sequences"""

    def __init__(self, dim, num_heads, window_size=7, shift_size=0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * 4))

    def forward(self, x):
        # x shape: (batch_size, sequence_length, dim)
        shortcut = x
        x = self.norm1(x)

        # Window-based multi-head self attention
        x = self.attn(x)
        x = shortcut + x

        # MLP
        shortcut = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = shortcut + x

        return x

class SwingTransformerVideoProcessor(nn.Module):
    """Swing Transformer for processing facial landmark sequences"""

    def __init__(self, landmark_dim=2, embed_dim=128, num_layers=4, num_heads=8, sequence_length=30):
        super().__init__()
        self.embed_dim = embed_dim
        self.sequence_length = sequence_length
        self.landmark_dim = landmark_dim

        # Embedding layer for landmarks
        self.landmark_embed = nn.Linear(landmark_dim, embed_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, sequence_length, embed_dim))

        # Swin Transformer blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(embed_dim, num_heads) for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(embed_dim)

        # Output heads for different tasks
        self.emotion_head = nn.Linear(embed_dim, 7)  # 7 basic emotions
        self.attention_head = nn.Linear(embed_dim, 1)  # Attention weights
        
        # Simple emotion classifier for single frame analysis
        self.single_frame_emotion = nn.Sequential(
            nn.Linear(landmark_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, 7)
        )
        
        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, landmarks, single_frame_mode=False):
        if single_frame_mode:
            # Single frame emotion prediction
            if len(landmarks.shape) == 3:
                # (batch, features) or (batch, seq_len, features) -> use last frame
                if landmarks.shape[1] > landmarks.shape[2]:
                    landmarks = landmarks[:, -1, :]  # Use last frame
                else:
                    landmarks = landmarks.mean(dim=1)  # Average if unclear
            
            emotions = self.single_frame_emotion(landmarks)
            # Create dummy attention weights
            attention_weights = torch.ones(landmarks.shape[0], landmarks.shape[-1] // 2) * 0.5
            return emotions, attention_weights

        # Handle different input shapes
        if len(landmarks.shape) == 3:
            # landmarks shape: (batch_size, sequence_length, features)
            B, T, F = landmarks.shape
            x = landmarks
        elif len(landmarks.shape) == 4:
            # landmarks shape: (batch_size, sequence_length, num_landmarks, 2)
            B, T, N, D = landmarks.shape
            # Flatten landmarks for each timestep
            x = landmarks.reshape(B, T, N * D)
        else:
            raise ValueError(f"Unexpected landmark tensor shape: {landmarks.shape}")

        # Embed landmarks
        if x.shape[-1] != self.landmark_dim:
            # If input features don't match expected dimension, use linear projection
            if not hasattr(self, 'input_proj'):
                self.input_proj = nn.Linear(x.shape[-1], self.landmark_dim)
            x = self.input_proj(x)

        x = self.landmark_embed(x)

        # Add positional embedding (handle variable sequence lengths)
        seq_len = min(T, self.sequence_length)
        x = x[:, :seq_len, :] + self.pos_embed[:, :seq_len, :]

        # Apply Swin Transformer blocks
        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        # Get predictions and attention weights
        emotions = self.emotion_head(x.mean(dim=1))  # Global average pooling
        attention_weights = torch.softmax(self.attention_head(x).squeeze(-1), dim=1)

        return emotions, attention_weights