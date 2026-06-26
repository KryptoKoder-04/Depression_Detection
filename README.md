# 🎓 BTP Final - AI-Based Depression Detection System

> **Bachelor Thesis Project** - AI-Assisted Depression Screening using Computer Vision and Deep Learning

[![Repository](https://img.shields.io/badge/GitHub-Depression_Detection-blue?style=flat-square&logo=github)](https://github.com/KryptoKoder-04/Depression_Detection)
[![Status](https://img.shields.io/badge/Status-Active%20Development-green?style=flat-square)]()
[![Python](https://img.shields.io/badge/Python-3.9+-blue?style=flat-square&logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red?style=flat-square&logo=pytorch)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-brightgreen?style=flat-square&logo=fastapi)](https://fastapi.tiangolo.com/)

---

## 📂 Project Overview

This is a comprehensive AI-based system for depression detection from video interviews. The project leverages computer vision, deep learning, and feature fusion techniques to create an automated screening tool.

### 🎯 Key Features
- ✅ **Two-Stream Architecture**: Separate processing of facial and body features
- ✅ **Transformer Self-Attention**: Captures complex temporal dependencies across video frames
- ✅ **Scale & Translation Invariance**: Mathematically normalizes faces for perfect camera generalization
- ✅ **Real-time Inference**: Fast video processing and prediction via FastAPI
- ✅ **Web Interface**: User-friendly React dashboard
- ✅ **PDF Reports**: Automated clinical report generation

---

## 🏗️ Project Structure

```
BTP final/
├── README.md                              # This file - Project overview
├── requirements.txt                       # Root-level dependencies
├── install_output.txt                    # Installation logs
│
└── TSFFM-Depression-Detection/           # 🎯 Main Project Folder
    ├── README.md                         # Detailed project documentation
    ├── .gitignore                        # Git ignore rules
    ├── requirements.txt                  # Project dependencies
    │
    ├── backend/                          # FastAPI Backend
    │   ├── main.py                       # Application entry point
    │   ├── config.py                     # Configuration settings
    │   ├── requirements.txt              # Backend dependencies
    │   ├── api/
    │   │   ├── __init__.py
    │   │   └── predict.py                # Prediction endpoints
    │   ├── models/
    │   │   ├── tsffm.py                  # Base TSFFM model
    │   │   ├── tsffm_lstm.py             # TSFFM + LSTM architecture
    │   │   ├── face_stream.py            # Face stream (CNN)
    │   │   ├── body_stream.py            # Body/pose stream (CNN)
    │   │   └── __init__.py
    │   ├── services/
    │   │   ├── face_detector.py          # MediaPipe face detection
    │   │   ├── pose_extractor.py         # MediaPipe pose estimation
    │   │   ├── frame_extractor.py        # Video frame extraction
    │   │   ├── preprocess_video.py       # Video preprocessing
    │   │   ├── inference.py              # Model inference pipeline
    │   │   ├── pdf_report.py             # PDF report generation
    │   │   └── __init__.py
    │   ├── weights/
    │   │   └── best_feature_model.pth    # Pre-trained model weights
    │   ├── uploads/                      # Uploaded video storage
    │   └── reports/                      # Generated PDF reports
    │
    ├── frontend/                         # React Vite Dashboard
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
    │   │   │   └── ModelInfo.jsx
    │   │   ├── pages/
    │   │   │   ├── Home.jsx
    │   │   │   └── Result.jsx
    │   │   └── api/
    │   │       └── predictApi.js
    │   └── ...
    │
    ├── ml/                               # ML Training & Analysis
    │   ├── train.py                      # Main training script
    │   ├── train_feature_fusion.py       # Feature fusion training
    │   ├── evaluate.py                   # Model evaluation
    │   ├── dataset.py                    # Dataset loader
    │   ├── preprocess_dataset.py         # Data preprocessing
    │   ├── metrics.py                    # Evaluation metrics
    │   ├── plots.py                      # Visualization & plots
    │   ├── compare_models.py             # Model comparison
    │   └── create_feature_labels.py      # Feature extraction utilities
    │
    ├── data/                             # Dataset Directory
    │   ├── raw/                          # Raw video data
    │   │   ├── depressed/
    │   │   └── non_depressed/
    │   ├── raw_original/                 # AVEC 2017 Dataset
    │   │   └── tramcam-daic-woz-e/
    │   │       ├── data_csv/             # Annotation files
    │   │       ├── train/                # Training videos
    │   │       ├── valid/                # Validation videos
    │   │       └── test/                 # Test videos
    │   ├── processed/                    # Processed features
    │   └── feature_labels.csv
    │
    ├── docs/                             # Documentation
    │   ├── INSTALLATION.md               # Setup instructions
    │   ├── API.md                        # API documentation
    │   └── PROJECT_STATUS.md             # Development status
    │
    ├── report/                           # Research reports
    ├── results/                          # Evaluation results
    └── ...
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- Node.js 18+
- 4GB+ RAM
- CUDA (optional, for GPU acceleration)

### Installation & Setup

#### 1️⃣ Clone Repository
```bash
git clone https://github.com/KryptoKoder-04/Depression_Detection.git
cd TSFFM-Depression-Detection
```

#### 2️⃣ Backend Setup
```bash
cd backend
python -m venv venv

# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

pip install -r requirements.txt
```

#### 3️⃣ Frontend Setup
```bash
cd ../frontend
npm install
```

#### 4️⃣ Start Services

**Terminal 1: Backend**
```bash
cd backend
python main.py
# API: http://localhost:8000
```

**Terminal 2: Frontend**
```bash
cd frontend
npm run dev
# App: http://localhost:5173
```

#### 5️⃣ Access Application
Open your browser and navigate to: **http://localhost:5173**

---

## 📊 System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    React Frontend (Vite)                    │
│                  http://localhost:5173                      │
│  ┌──────────────┐ ┌──────────────┐ ┌────────────────────┐ │
│  │  Upload Box  │ │  Result Card │ │  Probability Chart │ │
│  └──────────────┘ └──────────────┘ └────────────────────┘ │
└────────────────────┬────────────────────────────────────────┘
                     │ (HTTP/REST)
                     ▼
┌─────────────────────────────────────────────────────────────┐
│              FastAPI Backend Server                         │
│              http://localhost:8000                          │
│  ┌────────────────────────────────────────────────────────┐ │
│  │                 API Endpoints                          │ │
│  │  POST /api/predict - Upload & Predict                 │ │
│  └────────────────────────────────────────────────────────┘ │
└────────┬───────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│              ML Pipeline (Inference)                        │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌───────────┐  │
│  │  Video   │→ │  Frame   │→ │  Face    │→ │  Feature  │  │
│  │ Upload   │  │ Extract  │  │ Detect   │  │  Extract  │  │
│  └──────────┘  └──────────┘  └──────────┘  └───────────┘  │
│       │              │              │              │        │
│       ▼              ▼              ▼              ▼        │
│  ┌─────────────────────────────────────────────────────┐   │
│  │    TSFFM-Transformer Model (PyTorch)                │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌────────────┐ │   │
│  │  │ Face Stream │  │ Body Stream │  │ 1D Conv    │ │   │
│  │  │   (CNN)     │  │   (CNN)     │→ │ + BiLSTM   │ │   │
│  │  │ 128 dims    │  │ 32 dims     │  │ + Attention│ │   │
│  │  └─────────────┘  └─────────────┘  └────────────┘ │   │
│  └─────────────────────────────────────────────────────┘   │
│       │                    │                    │          │
│       ▼                    ▼                    ▼          │
│  ┌──────────────────────────────────────────────────┐     │
│  │          Classification & Report Gen             │     │
│  │  Prediction + Confidence + PDF Report           │     │
│  └──────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────┘
```

---

## 📖 Documentation

Comprehensive documentation is available in the TSFFM-Depression-Detection folder:

- **[TSFFM-Depression-Detection/README.md](TSFFM-Depression-Detection/README.md)** - Complete project documentation
  - Architecture details
  - Dataset information
  - Performance metrics
  - Training guide
  - Contributing guidelines

- **[TSFFM-Depression-Detection/docs/INSTALLATION.md](TSFFM-Depression-Detection/docs/INSTALLATION.md)** - Detailed setup instructions
- **[TSFFM-Depression-Detection/docs/API.md](TSFFM-Depression-Detection/docs/API.md)** - API endpoint reference
- **[TSFFM-Depression-Detection/docs/PROJECT_STATUS.md](TSFFM-Depression-Detection/docs/PROJECT_STATUS.md)** - Development status & roadmap

---

## 🧠 Model Details

### Architecture
- **Name**: Two-Stream Feature Fusion Model (TSFFM) + Transformer
- **Face Stream**: 2-Layer MLP → 128-dim features
- **Body Stream**: 2-Layer MLP → 32-dim features  
- **Temporal Modeling**: 1D Convolution + Bidirectional LSTM (256 hidden units) + Transformer Encoder (4-Head Self-Attention)
- **Data Augmentation**: Scale & Translation Invariance, Temporal Masking (Random Frame Blackouts)
- **Classification**: Binary classifier (Depression / No Depression)

### Performance
| Metric | Score |
|--------|-------|
| **Accuracy** | 77.27% |
| **Precision** | 0.81 |
| **Recall** | 0.76 |
| **F1-Score** | 0.7183 |
| **AUC-ROC** | 0.8007 |

---

## 📊 Dataset

**AVEC 2017 (TalkingCam / DAIC-WOZ)**
- ~100 clinical interviews
- Train/Valid/Test: 70% / 15% / 15%
- 7-15 min per session
- Labels: Binary (PHQ-8 score ≥ 10)

---

## 🔧 Technology Stack

| Category | Technology |
|----------|-----------|
| **Backend** | FastAPI, Python 3.9+ |
| **ML Framework** | PyTorch, TorchVision |
| **Vision** | OpenCV, MediaPipe |
| **Frontend** | React 18, Vite |
| **UI/Styling** | Tailwind CSS, Lucide React |
| **Charts** | Recharts |
| **Reporting** | ReportLab |
| **Database** | (Can be integrated) |

---

## ⚠️ Important Notes

### Medical Disclaimer
**This system is an AI-assisted screening prototype and NOT a clinical diagnosis tool.** It must not be used for medical decisions without professional clinical validation. Always consult qualified mental health professionals.

### Data Privacy
- All uploaded videos are processed locally
- Data should be handled according to privacy regulations
- Consider HIPAA compliance if used in healthcare settings

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m 'Add your feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request

---

## 📝 Development Timeline

- [x] Data preprocessing & feature extraction
- [x] Model architecture development (TSFFM-LSTM)
- [x] Model training & evaluation
- [x] Backend API development (FastAPI)
- [x] Frontend dashboard (React/Vite)
- [x] PDF report generation
- [x] Documentation & README
- [ ] Deployment (Docker containers)
- [ ] Model optimization (quantization, pruning)
- [ ] Extended evaluation on external datasets

---

## 📚 References

1. Valstar, M. F., et al. (2016). "AVEC 2016 - Depression, Mood, and Emotion Recognition Workshop"
2. He, K., et al. (2016). "Deep Residual Learning for Image Recognition" (ResNet)
3. Hochreiter, S., & Schmidhuber, J. (1997). "Long Short-Term Memory"
4. Cao, Z., et al. (2021). "OpenPose: Realtime Multi-Person 2D Pose Estimation"

---

## 📞 Support & Contact

- 📧 **Email**: your.email@example.com
- 🐛 **Issues**: [GitHub Issues](https://github.com/KryptoKoder-04/Depression_Detection/issues)
- 📖 **Documentation**: See docs/ folder
- 💬 **Discussions**: [GitHub Discussions](https://github.com/KryptoKoder-04/Depression_Detection/discussions)

---

## 📜 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **AVEC Dataset Team** - For the DAIC-WOZ/TalkingCam dataset
- **MediaPipe** - Google for face detection & pose estimation
- **PyTorch Team** - For the deep learning framework
- **FastAPI** - Sebastián Ramírez for the awesome framework

---

**Made with ❤️ for the BTP Final Project**

Last Updated: June 8, 2026
