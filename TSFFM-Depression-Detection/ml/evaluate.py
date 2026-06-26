import os
import argparse
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from dataset import DepressionDataset
from models import TSFFM_LSTM

def evaluate(args):
    # 1. Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 2. Paths
    dataset_root = args.data_dir
    eval_dir = os.path.join(dataset_root, args.split)
    os.makedirs(args.results_dir, exist_ok=True)
    
    # 3. Load evaluation dataset
    print(f"Loading {args.split} dataset...")
    eval_dataset = DepressionDataset(eval_dir, split_name=args.split)
    eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    # 4. Load model
    model = TSFFM_LSTM(
        face_dim=args.face_dim,
        body_dim=args.body_dim,
        lstm_hidden_dim=args.lstm_hidden_dim,
        lstm_layers=args.lstm_layers,
    )
    
    if not os.path.exists(args.weights_path):
        raise FileNotFoundError(f"Model weights not found at {args.weights_path}. Please train the model first.")
        
    print(f"Loading weights from {args.weights_path}...")
    model.load_state_dict(torch.load(args.weights_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    # 5. Inference loop
    all_preds = []
    all_labels = []
    all_probs = [] # store probability of class 1 (depressed) for AUC-ROC
    
    print("Running inference...")
    with torch.no_grad():
        for face, body, label in eval_loader:
            face, body, label = face.to(device), body.to(device), label.to(device)
            
            outputs = model(face, body)
            probs = torch.softmax(outputs, dim=1)
            
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(label.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
            
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # 6. Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    auc = roc_auc_score(all_labels, all_probs)
    
    print("\n================ EVALUATION METRICS ================")
    print(f"Accuracy : {accuracy*100:.2f}%")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-Score : {f1:.4f}")
    print(f"AUC-ROC  : {auc:.4f}")
    print("====================================================")
    
    report_text = classification_report(all_labels, all_preds, target_names=["Not Depressed", "Depressed"], zero_division=0)
    print("\nClassification Report:\n", report_text)
    
    # Save text report
    report_txt_path = os.path.join(args.results_dir, "classification_report.txt")
    with open(report_txt_path, 'w') as f:
        f.write("================ EVALUATION METRICS ================\n")
        f.write(f"Accuracy : {accuracy*100:.2f}%\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall   : {recall:.4f}\n")
        f.write(f"F1-Score : {f1:.4f}\n")
        f.write(f"AUC-ROC  : {auc:.4f}\n")
        f.write("====================================================\n\n")
        f.write("Classification Report:\n")
        f.write(report_text)
    print(f"Saved classification report text to: {report_txt_path}")
    
    # 7. Generate and save Confusion Matrix Plot
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=["Not Depressed", "Depressed"], 
                yticklabels=["Not Depressed", "Depressed"])
    plt.title("Confusion Matrix")
    plt.ylabel("Actual Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    cm_path = os.path.join(args.results_dir, "confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()
    print(f"Saved confusion matrix plot to: {cm_path}")
    
    # 8. Generate and save ROC Curve
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.tight_layout()
    roc_path = os.path.join(args.results_dir, "roc_curve.png")
    plt.savefig(roc_path)
    plt.close()
    print(f"Saved ROC curve plot to: {roc_path}")
    
    # 9. Plot training history if JSON exists
    weights_dir = os.path.dirname(args.weights_path)
    weights_stem = os.path.splitext(os.path.basename(args.weights_path))[0]
    candidate_history_path = os.path.join(weights_dir, f"{weights_stem}_history.json")
    history_path = candidate_history_path if os.path.exists(candidate_history_path) else os.path.join(weights_dir, "training_history.json")
    if os.path.exists(history_path):
        try:
            with open(history_path, 'r') as f:
                history = json.load(f)
                
            epochs = range(1, len(history["train_loss"]) + 1)
            
            # Plot loss curves
            plt.figure(figsize=(6, 4))
            plt.plot(epochs, history["train_loss"], 'bo-', label='Training Loss')
            plt.plot(epochs, history["val_loss"], 'ro-', label='Validation Loss')
            plt.title('Training and Validation Loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
            plt.tight_layout()
            loss_curve_path = os.path.join(args.results_dir, "loss_history.png")
            plt.savefig(loss_curve_path)
            plt.close()
            
            # Plot accuracy curves
            plt.figure(figsize=(6, 4))
            plt.plot(epochs, history["train_acc"], 'bo-', label='Training Accuracy')
            plt.plot(epochs, history["val_acc"], 'ro-', label='Validation Accuracy')
            plt.title('Training and Validation Accuracy')
            plt.xlabel('Epochs')
            plt.ylabel('Accuracy (%)')
            plt.legend()
            plt.tight_layout()
            acc_curve_path = os.path.join(args.results_dir, "accuracy_history.png")
            plt.savefig(acc_curve_path)
            plt.close()
            print(f"Saved training history curves to: {args.results_dir}")
        except Exception as e:
            print(f"Error plotting history curves: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate TSFFM-LSTM Depression Detection Model")
    parser.add_argument("--data_dir", type=str, 
                        default=r"C:\Users\jiten\.cache\kagglehub\datasets\trilism\tramcam-daic-woz-e\versions\1", 
                        help="Path to dataset root folder")
    parser.add_argument("--weights_path", type=str, 
                        default=r"c:\Users\jiten\btp_final2\TSFFM-Depression-Detection\backend\weights\best_model.pth", 
                        help="Path to model weights file")
    parser.add_argument("--results_dir", type=str, 
                        default=r"c:\Users\jiten\btp_final2\TSFFM-Depression-Detection\results", 
                        help="Folder to save evaluation plots and reports")
    parser.add_argument("--split", choices=["train", "valid", "test"], default="valid", help="Dataset split to evaluate")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for evaluation")
    parser.add_argument("--face_dim", type=int, default=272, help="Face feature dimension")
    parser.add_argument("--body_dim", type=int, default=8, help="Body feature dimension")
    parser.add_argument("--lstm_hidden_dim", type=int, default=128, help="LSTM hidden dimension")
    parser.add_argument("--lstm_layers", type=int, default=2, help="Number of LSTM layers used by the checkpoint")
    
    args = parser.parse_args()
    evaluate(args)
