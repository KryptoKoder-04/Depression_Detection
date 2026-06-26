import { useState, useEffect } from 'react';
import { BrainCircuit, Cpu, ShieldAlert, BarChart3 } from 'lucide-react';
import UploadBox from './components/UploadBox';
import ProbabilityChart from './components/ProbabilityChart';
import ResultCard from './components/ResultCard';
import ModelInfo from './components/ModelInfo';
import { uploadAndPredict } from './api/predictApi';

const LOADING_STEPS = [
  'Uploading patient video to FastAPI backend...',
  'Extracting 68 face keypoints and pose landmarks via MediaPipe...',
  'Fusing features and running PyTorch TSFFM-BiLSTM model inference...',
  'Compiling metrics and generating clinical screening PDF report...'
];

export default function App() {
  const [activeTab, setActiveTab] = useState('screening');
  const [patientId, setPatientId] = useState('');
  const [sessionDate, setSessionDate] = useState('');
  
  const [loading, setLoading] = useState(false);
  const [loadingStep, setLoadingStep] = useState(0);
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');

  // Simulate progress steps during loading
  useEffect(() => {
    let interval;
    if (loading) {
      interval = setInterval(() => {
        setLoadingStep((prevStep) => {
          if (prevStep < LOADING_STEPS.length - 1) {
            return prevStep + 1;
          }
          return prevStep;
        });
      }, 7000); // Progress step every 7s (average processing time)
    }
    return () => clearInterval(interval);
  }, [loading]);

  const handleUploadStart = async (selectedFile, pid, date) => {
    setPatientId(pid);
    setSessionDate(date);
    setLoadingStep(0);
    setLoading(true);
    setError('');
    setResult(null);

    try {
      const data = await uploadAndPredict(selectedFile, pid, date);
      setResult(data);
    } catch (err) {
      console.error(err);
      setError(err.message || 'An error occurred during video analysis. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setPatientId('');
    setSessionDate(new Date().toISOString().split('T')[0]);
    setLoadingStep(0);
    setResult(null);
    setError('');
  };

  return (
    <div className="app-wrapper">
      <header className="app-header">
        <div className="header-container">
          <div className="brand">
            <div className="brand-icon">
              <BrainCircuit size={22} color="white" />
            </div>
            <div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <span className="brand-name">TSFFM Depression Screening</span>
                <span className="brand-badge">Prototype</span>
              </div>
              <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', marginTop: '0.15rem' }}>
                Two-Stream Feature Fusion Model with Temporal LSTM
              </div>
            </div>
          </div>
          
          <nav className="nav-tabs">
            <button 
              className={`tab-btn ${activeTab === 'screening' ? 'active' : ''}`}
              onClick={() => setActiveTab('screening')}
            >
              <BarChart3 size={16} />
              <span>Screening Dashboard</span>
            </button>
            <button 
              className={`tab-btn ${activeTab === 'model-info' ? 'active' : ''}`}
              onClick={() => setActiveTab('model-info')}
            >
              <Cpu size={16} />
              <span>Model Architecture</span>
            </button>
          </nav>
        </div>
      </header>

      <main className="dashboard-container">
        {activeTab === 'screening' ? (
          <>
            {!loading && !result ? (
              <div className="full-width">
                <UploadBox 
                  onUploadStart={handleUploadStart}
                />
              </div>
            ) : loading ? (
              <div className="full-width">
                <div className="glass-panel loading-container">
                  <div className="pulse-ring">
                    <BrainCircuit size={36} />
                  </div>
                  <h2 className="loading-title">Analyzing Behavioral Cues</h2>
                  <p className="loading-subtitle">This may take 15–30 seconds depending on video length...</p>
                  
                  <div className="pipeline-steps">
                    {LOADING_STEPS.map((step, idx) => {
                      let stepClass = 'step-item';
                      if (loadingStep === idx) stepClass += ' active';
                      if (loadingStep > idx) stepClass += ' completed';
                      
                      return (
                        <div key={idx} className={stepClass}>
                          <div className="step-indicator">
                            {idx + 1}
                          </div>
                          <div className="step-text">{step}</div>
                        </div>
                      );
                    })}
                  </div>
                </div>
              </div>
            ) : result ? (
              <>
                <div className="result-main-col">
                  <ResultCard 
                    prediction={result.prediction}
                    confidence={result.confidence}
                    depressionProbability={result.depression_probability}
                    notDepressedProbability={result.not_depressed_probability}
                    reportUrl={result.report_url}
                    patientId={patientId}
                    sessionDate={sessionDate}
                    onReset={handleReset}
                  />
                </div>
                <div className="result-side-col">
                  <ProbabilityChart 
                    prediction={result.prediction}
                    confidence={result.confidence}
                    depressionProbability={result.depression_probability}
                    notDepressedProbability={result.not_depressed_probability}
                    faceConfidence={result.face_confidence}
                    bodyConfidence={result.body_confidence}
                  />
                </div>
              </>
            ) : null}

            {error && (
              <div className="full-width">
                <div className="glass-panel" style={{ 
                  padding: '2rem', 
                  borderColor: 'rgba(239, 68, 68, 0.2)',
                  background: 'rgba(239, 68, 68, 0.02)',
                  display: 'flex',
                  flexDirection: 'column',
                  alignItems: 'center',
                  gap: '1rem',
                  textAlign: 'center'
                }}>
                  <div style={{ color: 'var(--color-danger)' }}>
                    <ShieldAlert size={36} />
                  </div>
                  <h3 style={{ fontSize: '1.25rem', color: 'var(--text-primary)' }}>Inference Pipeline Interrupted</h3>
                  <p style={{ color: 'var(--text-secondary)', maxWidth: '500px', fontSize: '0.95rem' }}>{error}</p>
                  <button onClick={handleReset} className="btn-outline" style={{ marginTop: '0.5rem' }}>
                    Retry Analysis
                  </button>
                </div>
              </div>
            )}
          </>
        ) : (
          <div className="full-width">
            <ModelInfo />
          </div>
        )}
      </main>

      <footer className="app-footer">
        <div>🎓 Depression Detection BTP Final Project © 2026</div>
        <p className="footer-disclaimer">
          <strong>Medical Disclaimer:</strong> This application is a screening prototype using computer vision analysis. 
          It does NOT provide clinical diagnosis. Always consult with certified medical and psychiatric specialists for diagnosis and treatment.
        </p>
      </footer>
    </div>
  );
}
