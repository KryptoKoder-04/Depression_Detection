import os
import sys
import torch
from backend.config import PROJECT_ROOT, MODEL_WEIGHTS_PATH

# Add ML directory to python path for model importing
sys.path.append(PROJECT_ROOT)
from ml.models import TSFFM_LSTM

# Global model instance
_model = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Temperature scaling for calibration.
# The raw model is overconfident on out-of-distribution videos.
# T > 1 softens the probability distribution to more realistic ranges.
TEMPERATURE = 2.0


def get_model():
    """
    Singleton pattern to load and cache the model.
    """
    global _model
    if _model is not None:
        return _model
        
    print(f"Loading PyTorch model on device: {device}...")
    model = TSFFM_LSTM(face_dim=272, body_dim=8, lstm_hidden_dim=128, lstm_layers=2)
    
    if os.path.exists(MODEL_WEIGHTS_PATH):
        print(f"Loading weights from {MODEL_WEIGHTS_PATH}...")
        model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=device))
    else:
        print(f"WARNING: Model weights NOT found at {MODEL_WEIGHTS_PATH}. Running with random initialization.")
        
    model.to(device)
    model.eval()
    _model = model
    return _model


def run_inference(face_features, pose_features, expression_metrics=None):
    """
    Runs inference on preprocessed face and pose feature arrays.
    Blends the TSFFM model output with expression-based analysis for robust predictions.
    
    Inputs:
    - face_features: numpy array of shape (360, 272)
    - pose_features: numpy array of shape (360, 8)
    - expression_metrics: dict with smile_score, movement_score, expression_score (from extractor)
    
    Returns:
    - prediction: string "depressed" or "not_depressed"
    - confidence: float score representing probability
    - prob_depressed: float probability of depression class
    """
    model = get_model()
    
    # Add batch dimension: (360, dim) -> (1, 360, dim)
    face_tensor = torch.tensor(face_features, dtype=torch.float32).unsqueeze(0).to(device)
    pose_tensor = torch.tensor(pose_features, dtype=torch.float32).unsqueeze(0).to(device)
    
    with torch.no_grad():
        logits = model(face_tensor, pose_tensor)
        
        # Apply temperature scaling to calibrate overconfident predictions
        calibrated_logits = logits / TEMPERATURE
        probs = torch.softmax(calibrated_logits, dim=1)
        
        model_prob_depressed = probs[0, 1].item()
    
    # --- EXPRESSION-BASED CALIBRATION ---
    # The TSFFM model was trained on clinical AVEC data and produces unreliable
    # predictions on arbitrary webcam videos. We blend its output with direct
    # expression analysis (smile detection, movement variability) which works
    # reliably on ANY video.
    #
    # expression_score: 0 = flat/sad (depression cues), 1 = smiling/expressive (healthy cues)
    
    if expression_metrics is not None:
        expr_score = expression_metrics.get("expression_score", 0.5)
        smile_score = expression_metrics.get("smile_score", 0.5)
        movement_score = expression_metrics.get("movement_score", 0.5)
        
        # Convert expression score to depression probability:
        # High expression (smiling, moving) -> low depression probability
        # Low expression (flat, still) -> high depression probability
        expr_depression_prob = 1.0 - expr_score
        
        # Weighted blend: 40% TSFFM model + 60% expression analysis
        # Expression analysis is weighted higher because the TSFFM model is unreliable
        # on out-of-distribution (non-AVEC) videos, while expression features
        # (smile ratio, movement) generalize to any video.
        final_prob_depressed = 0.40 * model_prob_depressed + 0.60 * expr_depression_prob
        
        print(f"  [Inference] Model raw: {model_prob_depressed:.3f} | "
              f"Expression: smile={smile_score:.3f}, movement={movement_score:.3f}, "
              f"combined={expr_score:.3f} | Final: {final_prob_depressed:.3f}")
    else:
        final_prob_depressed = model_prob_depressed
    
    # Decision threshold
    prob_not_depressed = 1.0 - final_prob_depressed
    
    if final_prob_depressed >= 0.50:
        prediction = "depressed"
        confidence = final_prob_depressed
    else:
        prediction = "not_depressed"
        confidence = prob_not_depressed
        
    return prediction, confidence, final_prob_depressed
