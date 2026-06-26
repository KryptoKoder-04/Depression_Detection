import os
import argparse
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from dataset import DepressionDataset
from models import TSFFM_LSTM


class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0):
        super().__init__()
        self.weight = weight
        self.gamma = gamma

    def forward(self, logits, targets):
        ce_loss = nn.functional.cross_entropy(
            logits,
            targets,
            weight=self.weight,
            reduction="none"
        )
        pt = torch.exp(-ce_loss)
        focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_class_weights(dataset):
    # Count classes dynamically
    labels = [s['label'] for s in dataset.samples]
    labels = np.array(labels)
    
    total = len(labels)
    class_0_count = np.sum(labels == 0)
    class_1_count = np.sum(labels == 1)
    
    # Calculate inverse frequency weights
    w_0 = total / (2.0 * class_0_count) if class_0_count > 0 else 1.0
    w_1 = total / (2.0 * class_1_count) if class_1_count > 0 else 1.0
    
    return torch.tensor([w_0, w_1], dtype=torch.float32)


def get_balanced_sampler(dataset):
    labels = np.array([s['label'] for s in dataset.samples])
    class_counts = np.bincount(labels, minlength=2)
    class_weights = np.zeros_like(class_counts, dtype=np.float64)
    nonzero = class_counts > 0
    class_weights[nonzero] = 1.0 / class_counts[nonzero]
    sample_weights = class_weights[labels]
    return WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True
    )


def collect_epoch_metrics(labels, preds, probs):
    labels = np.asarray(labels)
    preds = np.asarray(preds)
    probs = np.asarray(probs)
    metrics = {
        "acc": accuracy_score(labels, preds) * 100.0,
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "f1": f1_score(labels, preds, zero_division=0),
    }
    try:
        metrics["auc"] = roc_auc_score(labels, probs)
    except ValueError:
        metrics["auc"] = 0.0
    return metrics


def train(args):
    set_seed(args.seed)

    # 1. Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        
    # 2. Paths
    dataset_root = args.data_dir
    train_dir = os.path.join(dataset_root, "train")
    valid_dir = os.path.join(dataset_root, "valid")
    
    # 3. Create datasets and dataloaders
    print("Loading training dataset...")
    train_dataset = DepressionDataset(train_dir, split_name="train")
    train_sampler = get_balanced_sampler(train_dataset) if args.balanced_sampler else None
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=0 # Set to 0 to avoid multiprocessing overhead/issues on Windows
    )
    
    print("Loading validation dataset...")
    valid_dataset = DepressionDataset(valid_dir, split_name="valid")
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    # 4. Initialize model
    # Inputs: Face flat features (272-dim), Body flat features (8-dim)
    model = TSFFM_LSTM(face_dim=272, body_dim=8, lstm_hidden_dim=128, lstm_layers=2)
    model = model.to(device)
    
    # 5. Compute class weights and setup Loss/Optimizer
    import numpy as np
    class_weights = get_class_weights(train_dataset).to(device)
    print(f"Calculated class weights: Non-Depressed={class_weights[0].item():.4f}, Depressed={class_weights[1].item():.4f}")
    
    if args.loss == "focal":
        criterion = FocalLoss(weight=class_weights, gamma=args.focal_gamma)
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    
    # 6. Training history storage
    history = {
        "train_loss": [],
        "train_acc": [],
        "train_precision": [],
        "train_recall": [],
        "train_f1": [],
        "train_auc": [],
        "val_loss": [],
        "val_acc": [],
        "val_precision": [],
        "val_recall": [],
        "val_f1": [],
        "val_auc": [],
        "learning_rate": []
    }
    
    best_metric_value = float('inf') if args.checkpoint_metric == "val_loss" else -float('inf')
    best_val_loss = float('inf')
    best_val_acc = 0.0
    best_val_f1 = 0.0
    epochs_without_improvement = 0
    
    # Ensure weights save directory exists
    weights_dir = os.path.dirname(args.save_path)
    if weights_dir:
        os.makedirs(weights_dir, exist_ok=True)
        
    # 7. Training loop
    print("\nStarting training loop...")
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        total_train = 0
        train_preds = []
        train_labels = []
        train_probs = []
        
        for batch_idx, (face, body, label) in enumerate(train_loader):
            face, body, label = face.to(device), body.to(device), label.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(face, body)
            loss = criterion(outputs, label)
            
            loss.backward()
            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            
            running_loss += loss.item() * face.size(0)
            
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            total_train += label.size(0)
            train_preds.extend(predicted.detach().cpu().numpy())
            train_labels.extend(label.detach().cpu().numpy())
            train_probs.extend(probs[:, 1].detach().cpu().numpy())
            
        epoch_train_loss = running_loss / total_train
        train_metrics = collect_epoch_metrics(train_labels, train_preds, train_probs)
        
        # Validation evaluation
        model.eval()
        running_val_loss = 0.0
        total_val = 0
        val_preds = []
        val_labels = []
        val_probs = []
        
        with torch.no_grad():
            for face, body, label in valid_loader:
                face, body, label = face.to(device), body.to(device), label.to(device)
                
                outputs = model(face, body)
                loss = criterion(outputs, label)
                
                running_val_loss += loss.item() * face.size(0)
                
                probs = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)
                total_val += label.size(0)
                val_preds.extend(predicted.detach().cpu().numpy())
                val_labels.extend(label.detach().cpu().numpy())
                val_probs.extend(probs[:, 1].detach().cpu().numpy())
                
        epoch_val_loss = running_val_loss / total_val
        val_metrics = collect_epoch_metrics(val_labels, val_preds, val_probs)
        
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]
        
        # Log stats
        print(f"Epoch {epoch+1:02d}/{args.epochs:02d} | "
              f"Train Loss: {epoch_train_loss:.4f} - Train Acc: {train_metrics['acc']:.2f}% - Train F1: {train_metrics['f1']:.4f} | "
              f"Val Loss: {epoch_val_loss:.4f} - Val Acc: {val_metrics['acc']:.2f}% - Val F1: {val_metrics['f1']:.4f} - Val AUC: {val_metrics['auc']:.4f} | "
              f"LR: {current_lr:.2e}")
              
        history["train_loss"].append(epoch_train_loss)
        history["train_acc"].append(train_metrics["acc"])
        history["train_precision"].append(train_metrics["precision"])
        history["train_recall"].append(train_metrics["recall"])
        history["train_f1"].append(train_metrics["f1"])
        history["train_auc"].append(train_metrics["auc"])
        history["val_loss"].append(epoch_val_loss)
        history["val_acc"].append(val_metrics["acc"])
        history["val_precision"].append(val_metrics["precision"])
        history["val_recall"].append(val_metrics["recall"])
        history["val_f1"].append(val_metrics["f1"])
        history["val_auc"].append(val_metrics["auc"])
        history["learning_rate"].append(current_lr)
        
        metric_values = {
            "val_loss": epoch_val_loss,
            "val_acc": val_metrics["acc"],
            "val_f1": val_metrics["f1"],
            "val_auc": val_metrics["auc"],
        }
        current_metric_value = metric_values[args.checkpoint_metric]

        if args.checkpoint_metric == "val_loss":
            improved = current_metric_value < best_metric_value - args.min_delta
        else:
            improved = current_metric_value > best_metric_value + args.min_delta

        if improved:
            best_metric_value = current_metric_value
            best_val_loss = epoch_val_loss
            best_val_acc = val_metrics["acc"]
            best_val_f1 = val_metrics["f1"]
            epochs_without_improvement = 0
            print(
                f"  --> Saving best model checkpoint "
                f"({args.checkpoint_metric}={best_metric_value:.4f}, Val Acc={best_val_acc:.2f}%, "
                f"Val F1={best_val_f1:.4f}, Val Loss={best_val_loss:.4f})"
            )
            torch.save(model.state_dict(), args.save_path)
        else:
            epochs_without_improvement += 1
            print(f"  --> No {args.checkpoint_metric} improvement for {epochs_without_improvement}/{args.early_stopping_patience} epochs")

        history_path = args.history_path or os.path.join(
            os.path.dirname(args.save_path),
            f"{os.path.splitext(os.path.basename(args.save_path))[0]}_history.json"
        )
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=4)

        if epochs_without_improvement >= args.early_stopping_patience:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break
            
    # Save training history as JSON
    history_path = args.history_path or os.path.join(
        os.path.dirname(args.save_path),
        f"{os.path.splitext(os.path.basename(args.save_path))[0]}_history.json"
    )
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=4)
        
    print(
        f"\nTraining completed! Best {args.checkpoint_metric}: {best_metric_value:.4f} "
        f"(Accuracy: {best_val_acc:.2f}%, F1: {best_val_f1:.4f}, Loss: {best_val_loss:.4f})"
    )
    print(f"Model weights saved to: {args.save_path}")
    print(f"Training history saved to: {history_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train TSFFM-LSTM Depression Detection Model")
    parser.add_argument("--data_dir", type=str, 
                        default=r"C:\Users\jiten\.cache\kagglehub\datasets\trilism\tramcam-daic-woz-e\versions\1", 
                        help="Path to dataset root folder")
    parser.add_argument("--save_path", type=str, 
                        default=r"c:\Users\jiten\btp_final2\TSFFM-Depression-Detection\backend\weights\best_model.pth", 
                        help="Path to save trained weights")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=0.0005, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="L2 regularization factor")
    parser.add_argument("--loss", choices=["cross_entropy", "focal"], default="focal", help="Training loss function")
    parser.add_argument("--focal_gamma", type=float, default=2.0, help="Gamma value for focal loss")
    parser.add_argument("--balanced_sampler", action="store_true", default=True, help="Use class-balanced sampling for training batches")
    parser.add_argument("--no_balanced_sampler", dest="balanced_sampler", action="store_false", help="Disable class-balanced sampling")
    parser.add_argument("--checkpoint_metric", choices=["val_loss", "val_acc", "val_f1", "val_auc"], default="val_f1", help="Validation metric used to save the best checkpoint")
    parser.add_argument("--history_path", type=str, default=None, help="Optional path for training history JSON")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping norm; set 0 to disable")
    parser.add_argument("--early_stopping_patience", type=int, default=12, help="Epochs without validation F1 improvement before stopping")
    parser.add_argument("--min_delta", type=float, default=1e-4, help="Minimum validation improvement to reset early stopping")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    train(args)
