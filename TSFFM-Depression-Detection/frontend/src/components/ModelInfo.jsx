import { Cpu, Eye, Activity, Share2, Layers } from 'lucide-react';

export default function ModelInfo() {
  return (
    <div className="glass-panel arch-card">
      <div className="arch-header">
        <h2 style={{ fontSize: '1.75rem' }}>🧠 Neural Network Architecture</h2>
        <p className="arch-subtitle">
          Two-Stream Feature Fusion Model (TSFFM) with Bidirectional LSTM Temporal Modeling
        </p>
      </div>

      <div className="arch-grid">
        <div className="stream-info-box">
          <span className="stream-badge">Visual Channel A</span>
          <h3 className="stream-info-title">
            <Eye size={20} style={{ color: 'var(--color-primary)' }} />
            <span>Face Stream</span>
          </h3>
          <p className="stream-desc">
            Processes spatial geometry of the face. Facial landmarks are extracted via MediaPipe FaceMesh to construct sequences of structural movement.
          </p>
          <div className="tech-spec-list">
            <div className="spec-item">
              <span className="spec-label">Input Landmarks</span>
              <span className="spec-val">68 FaceMesh vertices</span>
            </div>
            <div className="spec-item">
              <span className="spec-label">Feature Shape</span>
              <span className="spec-val">360 frames × 272 coordinates</span>
            </div>
            <div className="spec-item">
              <span className="spec-label">Projection Space</span>
              <span className="spec-val">128-dimensional dense vector</span>
            </div>
          </div>
        </div>

        <div className="stream-info-box">
          <span className="stream-badge">Visual Channel B</span>
          <h3 className="stream-info-title">
            <Activity size={20} style={{ color: 'var(--color-accent)' }} />
            <span>Body Stream</span>
          </h3>
          <p className="stream-desc">
            Captures gross body motor activity and posture. MediaPipe Pose tracks coordinates of left and right shoulders to screen posture fluctuations.
          </p>
          <div className="tech-spec-list">
            <div className="spec-item">
              <span className="spec-label">Input Points</span>
              <span className="spec-val">Left & Right Shoulders</span>
            </div>
            <div className="spec-item">
              <span className="spec-label">Feature Shape</span>
              <span className="spec-val">360 frames × 8 coordinates</span>
            </div>
            <div className="spec-item">
              <span className="spec-label">Projection Space</span>
              <span className="spec-val">32-dimensional dense vector</span>
            </div>
          </div>
        </div>
      </div>

      <div className="fusion-flow-box">
        <h3 className="fusion-title">Feature Fusion & Bidirectional Recurrent Layer</h3>
        <p className="fusion-desc">
          Features from both visual streams are concatenated at each time step. The combined temporal representation is classified using bidirectional recurrent layers.
        </p>
        
        <div className="fusion-steps">
          <div className="fusion-badge">
            <Layers size={14} style={{ color: 'var(--color-primary)' }} />
            <span>Fused Vectors (160-dim)</span>
          </div>
          <span className="fusion-arrow">→</span>
          <div className="fusion-badge" style={{ borderColor: 'rgba(139, 92, 246, 0.3)' }}>
            <Cpu size={14} style={{ color: 'var(--color-accent)' }} />
            <span>BiLSTM Sequence (128 hidden)</span>
          </div>
          <span className="fusion-arrow">→</span>
          <div className="fusion-badge">
            <Share2 size={14} style={{ color: 'var(--color-success)' }} />
            <span>Average Pooling & Classifier</span>
          </div>
        </div>
      </div>

      <div className="metrics-section">
        <h3 style={{ fontSize: '1.25rem', marginBottom: '1rem' }}>📈 Model Benchmarks (AVEC 2017 validation split)</h3>
        <div className="metrics-grid">
          <div className="metric-card">
            <div className="metric-num">57.67%</div>
            <div className="metric-label">Accuracy</div>
          </div>
          <div className="metric-card">
            <div className="metric-num">0.6597</div>
            <div className="metric-label">AUC-ROC</div>
          </div>
          <div className="metric-card">
            <div className="metric-num">0.4688</div>
            <div className="metric-label">Precision</div>
          </div>
          <div className="metric-card">
            <div className="metric-num">0.4255</div>
            <div className="metric-label">Recall</div>
          </div>
        </div>
      </div>
    </div>
  );
}
