import argparse
import os

import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader

from dataset import DepressionDataset
from models import TSFFM_LSTM


def evaluate_thresholds(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    valid_dir = os.path.join(args.data_dir, args.split)
    dataset = DepressionDataset(valid_dir, split_name=args.split)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = TSFFM_LSTM(
        face_dim=args.face_dim,
        body_dim=args.body_dim,
        lstm_hidden_dim=args.lstm_hidden_dim,
        lstm_layers=args.lstm_layers,
    )
    model.load_state_dict(torch.load(args.weights_path, map_location=device))
    model.to(device)
    model.eval()

    labels = []
    probs = []
    with torch.no_grad():
        for face, body, label in loader:
            face = face.to(device)
            body = body.to(device)
            outputs = model(face, body)
            batch_probs = torch.softmax(outputs, dim=1)[:, 1]
            labels.extend(label.numpy())
            probs.extend(batch_probs.cpu().numpy())

    labels = np.asarray(labels)
    probs = np.asarray(probs)
    auc = roc_auc_score(labels, probs)

    best = None
    for threshold in np.linspace(args.min_threshold, args.max_threshold, args.steps):
        preds = (probs >= threshold).astype(int)
        row = {
            "threshold": float(threshold),
            "accuracy": accuracy_score(labels, preds) * 100.0,
            "precision": precision_score(labels, preds, zero_division=0),
            "recall": recall_score(labels, preds, zero_division=0),
            "f1": f1_score(labels, preds, zero_division=0),
            "auc": auc,
        }
        if best is None or row[args.metric] > best[args.metric]:
            best = row

    print("============== THRESHOLD SEARCH ==============")
    print(f"weights: {args.weights_path}")
    print(f"split  : {args.split}")
    print(f"metric : {args.metric}")
    print(f"best_threshold: {best['threshold']:.3f}")
    print(f"Accuracy : {best['accuracy']:.2f}%")
    print(f"Precision: {best['precision']:.4f}")
    print(f"Recall   : {best['recall']:.4f}")
    print(f"F1-Score : {best['f1']:.4f}")
    print(f"AUC-ROC  : {best['auc']:.4f}")
    print("==============================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Threshold sweep for TSFFM-LSTM checkpoints")
    parser.add_argument("--data_dir", type=str, default=r"C:\Users\jiten\.cache\kagglehub\datasets\trilism\tramcam-daic-woz-e\versions\1")
    parser.add_argument("--split", choices=["train", "valid", "test"], default="valid")
    parser.add_argument("--weights_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--face_dim", type=int, default=272)
    parser.add_argument("--body_dim", type=int, default=8)
    parser.add_argument("--lstm_hidden_dim", type=int, default=128)
    parser.add_argument("--lstm_layers", type=int, default=2)
    parser.add_argument("--min_threshold", type=float, default=0.05)
    parser.add_argument("--max_threshold", type=float, default=0.95)
    parser.add_argument("--steps", type=int, default=181)
    parser.add_argument("--metric", choices=["accuracy", "precision", "recall", "f1"], default="accuracy")
    evaluate_thresholds(parser.parse_args())
