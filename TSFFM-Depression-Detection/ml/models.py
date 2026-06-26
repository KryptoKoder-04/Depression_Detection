import torch
import torch.nn as nn

class FaceStream(nn.Module):
    """
    Processes the facial landmarks sequence (e.g. keypoints and confidence).
    Projects the flattened landmarks to a 128-dimensional space step-by-step.
    """
    def __init__(self, input_dim=272, output_dim=128):
        super(FaceStream, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU()
        )

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        batch_size, seq_len, input_dim = x.shape
        # Flatten temporal dimension to apply linear layers to all frames in batch
        x_flat = x.view(-1, input_dim)
        out_flat = self.net(x_flat)
        # Reshape back to sequence
        out = out_flat.view(batch_size, seq_len, -1)
        return out


class BodyStream(nn.Module):
    """
    Processes body/pose landmarks or confidence sequences.
    Projects the pose features to a 32-dimensional space step-by-step.
    """
    def __init__(self, input_dim=8, output_dim=32):
        super(BodyStream, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU()
        )

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        batch_size, seq_len, input_dim = x.shape
        x_flat = x.view(-1, input_dim)
        out_flat = self.net(x_flat)
        out = out_flat.view(batch_size, seq_len, -1)
        return out


class TSFFM_LSTM(nn.Module):
    """
    Two-Stream Feature Fusion Model (TSFFM) with LSTM and Transformer Self-Attention.
    Fuses face (128-dim) and body (32-dim) features into a 160-dim sequence,
    then processes it through a 1D Conv, Bidirectional LSTM, and a Transformer
    Encoder for depression screening.
    """
    def __init__(self, face_dim=272, body_dim=8, lstm_hidden_dim=128, lstm_layers=2, num_classes=2):
        super(TSFFM_LSTM, self).__init__()
        self.face_stream = FaceStream(input_dim=face_dim, output_dim=128)
        self.body_stream = BodyStream(input_dim=body_dim, output_dim=32)
        
        # 1D Conv to smooth temporal features (kernel_size=3, padding=1)
        self.conv1d = nn.Sequential(
            nn.Conv1d(in_channels=160, out_channels=160, kernel_size=3, padding=1),
            nn.BatchNorm1d(160),
            nn.ReLU()
        )
        
        # LSTM input size is fused features dim (128 + 32 = 160)
        self.lstm = nn.LSTM(
            input_size=160,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.5 if lstm_layers > 1 else 0.0
        )
        
        # Transformer Multi-Head Self-Attention over LSTM outputs
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=lstm_hidden_dim * 2,
            nhead=4,
            dim_feedforward=512,
            dropout=0.5,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(transformer_layer, num_layers=1)
        
        # Output dimension is lstm_hidden_dim * 2 because of bidirectional=True
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, face_input, body_input):
        # Inputs: (batch_size, seq_len, face_dim), (batch_size, seq_len, body_dim)
        face_features = self.face_stream(face_input) # Shape: (batch_size, seq_len, 128)
        body_features = self.body_stream(body_input) # Shape: (batch_size, seq_len, 32)
        
        # Concatenate features along the last dimension
        fused = torch.cat((face_features, body_features), dim=-1) # Shape: (batch_size, seq_len, 160)
        
        # Apply 1D Convolution over the temporal dimension
        # Conv1d expects (batch_size, channels, seq_len)
        fused = fused.transpose(1, 2) # Shape: (batch_size, 160, seq_len)
        fused = self.conv1d(fused)
        fused = fused.transpose(1, 2) # Shape: (batch_size, seq_len, 160)
        
        # Pass fused sequence to LSTM
        lstm_out, _ = self.lstm(fused) # Shape: (batch_size, seq_len, lstm_hidden_dim * 2)
        
        # Pass through Transformer Self-Attention
        transformer_out = self.transformer(lstm_out) # Shape: (batch_size, seq_len, lstm_hidden_dim * 2)
        
        # Global Average Pooling over the temporal dimension
        attended_out = torch.mean(transformer_out, dim=1) # Shape: (batch_size, lstm_hidden_dim * 2)
        
        # Fully connected layers to logits
        logits = self.fc(attended_out) # Shape: (batch_size, num_classes)
        return logits
