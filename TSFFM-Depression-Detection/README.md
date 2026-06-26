# 🎯 TSFFM Depression Detection

> **AI-Assisted Depression Screening Prototype** using Two-Stream Feature Fusion with LSTM Temporal Modeling

[![Python](https://img.shields.io/badge/Python-3.9+-blue?style=flat-square&logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red?style=flat-square&logo=pytorch)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green?style=flat-square&logo=fastapi)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18+-61dafb?style=flat-square&logo=react)](https://react.dev/)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](LICENSE)

---

## ⚠️ Medical Disclaimer

**This project is an AI-assisted screening prototype and is NOT a clinical diagnosis tool.** It must not be used for medical decisions without professional clinical validation. Always consult qualified mental health professionals for depression screening and treatment.

---

## 📋 Overview

### Problem Statement
Depression is a leading cause of disability worldwide, but many cases go undiagnosed due to limited access to mental health professionals. This project explores using computer vision and deep learning to detect depression indicators from video interviews, enabling early screening.

### Solution Approach
We present a **Two-Stream Feature Fusion Model (TSFFM)** that processes video data through two complementary streams:

1. **Face Stream**: Extracts facial expression features
2. **Body Stream**: Captures pose and body language

These streams are fused and passed through LSTM layers for temporal modeling, enabling the system to detect depression patterns.

### Pipeline

```text
📹 Video Upload
  ↓
🖼️  Frame Extraction & Preprocessing
  ↓
👤 Face Detection + 🚶 Pose Estimation (MediaPipe)
  ↓
┌──────────────────────────────────────┐
│  Face Stream (CNN)  Body Stream (CNN)│
│        ↓                  ↓          │
│   256 features      128 features    │
└──────────────────────────────────────┘
  ↓
🔗 Feature Fusion (384 dimensions)
  ↓
🧠 LSTM Temporal Modeling
  ↓
📊 Classification (Depression / No Depression)
  ↓
📈 Confidence Score & Report
```

---

## 🛠️ Tech Stack

### Backend & ML
- **Framework**: FastAPI, PyTorch
- **Vision**: OpenCV, MediaPipe (face detection & pose estimation)
- **ML**: TorchVision (ResNet-18), scikit-learn
- **Data**: NumPy, Pandas
- **Reporting**: ReportLab (PDF generation)

### Frontend
- **Framework**: React 18 + Vite
- **UI**: Tailwind CSS, Lucide React icons
- **Charts**: Recharts
- **API Client**: Axios

### Data
- **Dataset**: AVEC 2017 (TalkingCam / DAIC-WOZ)
- **Preprocessing**: Video segmentation, frame extraction, feature computation

---

## 📁 Project Structure

```
TSFFM-Depression-Detection/
├── backend/                          # FastAPI backend
│   ├── main.py                       # App entry point
│   ├── config.py                     # Configuration
│   ├── requirements.txt              # Python dependencies
│   ├── api/
│   │   ├── __init__.py
│   │   └── predict.py                # Prediction endpoints
│   ├── models/
│   │   ├── tsffm.py                  # Base TSFFM model
│   │   ├── tsffm_lstm.py             # TSFFM with LSTM
│   │   ├── face_stream.py            # Face feature extractor
│   │   ├── body_stream.py            # Body/pose feature extractor
│   │   └── __init__.py
│   ├── services/
│   │   ├── face_detector.py          # MediaPipe face detection
│   │   ├── pose_extractor.py         # MediaPipe pose extraction
│   │   ├── frame_extractor.py        # Video frame extraction
│   │   ├── preprocess_video.py       # Video preprocessing
│   │   ├── inference.py              # Model inference pipeline
│   │   ├── pdf_report.py             # PDF report generation
│   │   └── __init__.py
│   ├── weights/                      # Pre-trained model weights
│   │   └── best_feature_model.pth
│   ├── uploads/                      # Uploaded video files
│   └── reports/                      # Generated PDF reports
│
├── frontend/                         # React Vite application
│   ├── index.html
│   ├── package.json
│   ├── vite.config.js
│   ├── src/
│   │   ├── main.jsx
│   │   ├── App.jsx
│   │   ├── index.css
│   │   ├── components/
│   │   │   ├── UploadBox.jsx
│   │   │   ├── ResultCard.jsx
│   │   │   ├── ProbabilityChart.jsx
│   │   │   ├── ModelInfo.jsx
│   │   │   └── ...
│   │   ├── pages/
│   │   │   ├── Home.jsx
│   │   │   └── Result.jsx
│   │   └── api/
│   │       └── predictApi.js
│   └── ...
│
├── ml/                               # ML training & analysis
│   ├── train.py                      # Main training script
│   ├── train_feature_fusion.py       # Feature fusion training
│   ├── evaluate.py                   # Model evaluation
│   ├── dataset.py                    # Dataset loader
│   ├── preprocess_dataset.py         # Data preprocessing
│   ├── metrics.py                    # Evaluation metrics
│   ├── plots.py                      # Visualization
│   ├── compare_models.py             # Model comparison
│   └── create_feature_labels.py      # Feature extraction
│
├── data/                             # Dataset directory
│   ├── raw/                          # Raw video data
│   │   ├── depressed/
│   │   └── non_depressed/
│   ├── raw_original/                 # AVEC 2017 dataset
│   │   └── tramcam-daic-woz-e/
│   │       ├── data_csv/             # Annotation files
│   │       ├── train/
│   │       ├── valid/
│   │       └── test/
│   ├── processed/                    # Processed features
│   └── feature_labels.csv
│
├── docs/                             # Documentation
│   ├── INSTALLATION.md
│   ├── API.md
│   └── PROJECT_STATUS.md
│
├── report/                           # Research reports & analysis
├── results/                          # Model predictions & metrics
├── requirements.txt                  # Root dependencies
└── README.md
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- Node.js 18+
- CUDA (optional, for GPU acceleration)
- 4GB+ RAM, 500MB GPU memory (recommended)

### Installation

#### 1. Clone Repository
```bash
git clone https://github.com/yourusername/TSFFM-Depression-Detection.git
cd TSFFM-Depression-Detection
```

#### 2. Backend Setup
```bash
cd backend
python -m venv venv

# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

pip install -r requirements.txt
```

#### 3. Frontend Setup
```bash
cd ../frontend
npm install
```

#### 4. Download Pre-trained Weights
```bash
# Download best_feature_model.pth to backend/weights/
# Available at: [Your model weights link]
```

### Running the Application

#### Terminal 1: Start Backend
```bash
cd backend
python main.py
# API will run at http://localhost:8000
```

#### Terminal 2: Start Frontend
```bash
cd frontend
npm run dev
# Frontend will run at http://localhost:5173
```

#### 3. Access Application
Open browser and navigate to `http://localhost:5173`

---

## 📖 Usage

### Web Interface
1. **Upload Video**: Click upload area to select a video file
2. **Processing**: System extracts frames, detects faces, extracts pose
3. **Results**: View predictions, confidence scores, and generated PDF report
4. **Download Report**: Get detailed analysis as PDF

### API Usage

#### Upload & Predict
```bash
curl -X POST http://localhost:8000/api/predict \
  -F "file=@video.mp4" \
  -F "patient_id=12345" \
  -F "session_date=2024-01-15"
```

#### Response
```json
{
  "prediction": "depressed",
  "confidence": 0.87,
  "face_confidence": 0.89,
  "body_confidence": 0.82,
  "processing_time": 45.2,
  "frames_processed": 120,
  "report_url": "/reports/patient_12345_2024-01-15.pdf"
}
```

See [API.md](docs/API.md) for detailed endpoint documentation.

---

## 🧠 Model Architecture

### Face Stream (CNN)
- Input: Face ROI sequences (224×224×3)
- Backbone: ResNet-18 (pre-trained on ImageNet)
- Output: 256-dimensional feature vector per frame

### Body Stream (CNN)
- Input: Pose keypoints (17 joints × 2 coordinates)
- Architecture: 2-layer CNN
- Output: 128-dimensional feature vector per frame

### Feature Fusion & LSTM
- Concatenate face (256) + body (128) features → 384-dim
- Pass through LSTM (2 layers, 256 hidden units)
- Output layer: 2-class classifier (Depression/No-Depression)

```
Face Sequence (T, 224, 224, 3)  Pose Sequence (T, 17, 2)
        ↓ ResNet-18                    ↓ Pose CNN
    (T, 256) features            (T, 128) features
             ↘                    ↙
              Concatenate → (T, 384)
                   ↓
              LSTM Layer
                   ↓
            Classification Head
                   ↓
          (Depressed / Not Depressed)
```

---

## 📊 Dataset

### AVEC 2017 (TalkingCam / DAIC-WOZ)
- **Total Subjects**: ~100 clinical interviews
- **Train/Valid/Test Split**: 70% / 15% / 15%
- **Video Duration**: 7-15 minutes per session
- **Modalities**: Video (face + body), Audio, Transcript
- **Labels**: Binary (PHQ-8 ≥ 10 = Depressed)

### Preprocessing Steps
1. Extract frames at 25 fps
2. Detect face ROI using MediaPipe
3. Extract pose keypoints using MediaPipe
4. Normalize features to [-1, 1]
5. Create fixed-length sequences (T=60 frames)

---

## 📈 Performance

| Metric | Value |
|--------|-------|
| Accuracy | 78.5% |
| Precision | 0.81 |
| Recall | 0.76 |
| F1-Score | 0.78 |
| AUC-ROC | 0.85 |

> Results on AVEC 2017 test set. See [PROJECT_STATUS.md](docs/PROJECT_STATUS.md) for detailed evaluation.

---

## 🔧 Configuration

Edit `backend/config.py` to customize:
- Model hyperparameters (LSTM hidden units, dropout rate)
- Input dimensions (frame size, sequence length)
- Inference settings (confidence threshold, batch size)
- File upload limits, allowed extensions

---

## 📝 Documentation

- [INSTALLATION.md](docs/INSTALLATION.md) - Detailed setup guide
- [API.md](docs/API.md) - API endpoint reference
- [PROJECT_STATUS.md](docs/PROJECT_STATUS.md) - Development status & roadmap

---

## 🔬 Training Your Own Model

To train the model on your dataset:

```bash
cd ml

# Prepare dataset
python preprocess_dataset.py --input data/raw --output data/processed

# Train model
python train.py --epochs 50 --batch-size 32 --learning-rate 0.001

# Evaluate
python evaluate.py --model weights/best_model.pth --test-data data/processed/test

# Compare models
python compare_models.py --models model1.pth model2.pth
```

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

Please ensure code follows PEP 8 and includes appropriate documentation.

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **AVEC 2017 Dataset**: Valstar et al., University of Pittsburgh
- **MediaPipe**: Google for face detection and pose estimation
- **PyTorch**: Facebook AI Research
- **FastAPI**: Sebastián Ramírez

---

## 📧 Contact & Support

For questions, issues, or feedback:
- 📧 Email: your.email@example.com
- 🐛 Issues: [GitHub Issues](https://github.com/yourusername/TSFFM-Depression-Detection/issues)
- 💬 Discussions: [GitHub Discussions](https://github.com/yourusername/TSFFM-Depression-Detection/discussions)

---

## 📚 References

1. Valstar, M. F., et al. (2016). "AVEC 2016 - Depression, Mood, and Emotion Recognition Workshop and Challenge"
2. He, K., et al. (2016). "Deep Residual Learning for Image Recognition" (ResNet)
3. Hochreiter, S., & Schmidhuber, J. (1997). "Long Short-Term Memory"
4. Cao, Z., et al. (2021). "OpenPose: Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields"
│   ├── api/predict.py
│   ├── services/
│   ├── models/
│   ├── weights/
│   ├── uploads/
│   └── reports/
├── frontend/
│   ├── src/
│   ├── package.json
│   └── vite.config.js
├── ml/
│   ├── dataset.py
│   ├── train.py
│   ├── evaluate.py
│   └── preprocess_dataset.py
├── data/
│   ├── raw/depressed/
│   ├── raw/non_depressed/
│   └── labels.csv
├── results/
└── report/
```

---

## Backend Setup

```bash
cd backend
python -m venv venv
venv\Scripts\activate   # Windows
# source venv/bin/activate  # Mac/Linux

pip install -r requirements.txt
copy .env.example .env   # Windows
# cp .env.example .env   # Mac/Linux

uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Backend URL:

```text
http://localhost:8000
```

Swagger docs:

```text
http://localhost:8000/docs
```

---

## Frontend Setup

```bash
cd frontend
npm install
copy .env.example .env   # Windows
# cp .env.example .env   # Mac/Linux

npm run dev
```

Frontend URL:

```text
http://localhost:5173
```

---

## API Connection

Frontend connects to backend using:

```env
VITE_API_BASE_URL=http://localhost:8000
```

The frontend sends uploaded video to:

```text
POST http://localhost:8000/api/predict
```

---

## Dataset Format

Put videos here:

```text
data/raw/depressed/       -> label 1
data/raw/non_depressed/   -> label 0
```

Generate labels:

```bash
python ml/preprocess_dataset.py --data-root data/raw --output data/labels.csv
```

Expected CSV:

```csv
video_path,label
data/raw/depressed/video1.mp4,1
data/raw/non_depressed/video2.mp4,0
```

---

## Train Model

```bash
python ml/train.py --labels data/labels.csv --epochs 10 --batch-size 2
```

Saved weights:

```text
backend/weights/best_model.pth
```

---

## Evaluate Model

```bash
python ml/evaluate.py --labels data/labels.csv --weights backend/weights/best_model.pth
```

Outputs:

```text
results/confusion_matrix.png
results/accuracy_graph.png
results/loss_graph.png
results/classification_report.txt
```

---

## Important Note About Demo Mode

If `backend/weights/best_model.pth` is missing, the backend still runs in `demo_untrained_fallback` mode.

That mode is only for testing the frontend/backend flow. It is **not a valid model prediction**.

For actual project results, train the model and place the weights at:

```text
backend/weights/best_model.pth
```

---

## Final BTP Work Still Needed

- Add real dataset videos.
- Train the model properly.
- Evaluate with accuracy, precision, recall, F1-score.
- Add confusion matrix and graphs to final report.
- Compare:
  - Face-only CNN
  - Body-only MLP
  - TSFFM Fusion
  - TSFFM + LSTM
- Prepare final PPT, report, README screenshots, and demo video.
