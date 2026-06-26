import { useEffect, useState } from 'react';

export default function ProbabilityChart({
  prediction,
  confidence,
  depressionProbability,
  notDepressedProbability,
  bodyConfidence,
  faceConfidence
}) {
  const [offset, setOffset] = useState(502.65);
  const clamp01 = (value) => (Number.isFinite(value) ? Math.max(0, Math.min(1, value)) : 0);
  const likelihood = clamp01(typeof depressionProbability === 'number' ? depressionProbability : confidence);
  const notDepressedScore = clamp01(
    typeof notDepressedProbability === 'number' ? notDepressedProbability : 1 - likelihood
  );
  const percentage = Math.round(likelihood * 100);
  const confidencePercentage = Math.round(clamp01(confidence) * 100);
  const isDepressed = prediction === 'depressed';
  
  // Circumference of R=80 circle: 2 * pi * 80 = 502.65
  const circumference = 502.65;

  useEffect(() => {
    // Animate fill on load
    const progressOffset = circumference - (likelihood * circumference);
    const timer = setTimeout(() => {
      setOffset(progressOffset);
    }, 100);
    return () => clearTimeout(timer);
  }, [likelihood]);

  return (
    <div className="glass-panel gauge-card">
      <h3 className="gauge-title">Depression Probability</h3>
      
      <div className="gauge-wrapper">
        <svg className="gauge-svg" viewBox="0 0 200 200">
          <defs>
            <linearGradient id="successGradient" x1="0%" y1="0%" x2="100%" y2="100%">
              <stop offset="0%" stopColor="#059669" />
              <stop offset="100%" stopColor="#10b981" />
            </linearGradient>
            <linearGradient id="dangerGradient" x1="0%" y1="0%" x2="100%" y2="100%">
              <stop offset="0%" stopColor="#dc2626" stopOpacity="0.8" />
              <stop offset="100%" stopColor="#f43f5e" />
            </linearGradient>
          </defs>
          
          {/* Background circle */}
          <circle
            className="gauge-bg"
            cx="100"
            cy="100"
            r="80"
          />
          
          {/* Foreground circle */}
          <circle
            className={`gauge-fill ${isDepressed ? 'depressed' : 'not-depressed'}`}
            cx="100"
            cy="100"
            r="80"
            style={{
              strokeDasharray: circumference,
              strokeDashoffset: offset,
            }}
          />
        </svg>
        
        <div className="gauge-center-text">
          <span className="gauge-percentage" style={{
            color: isDepressed ? '#f43f5e' : '#10b981'
          }}>
            {percentage}%
          </span>
          <span className="gauge-label">
            Depression
          </span>
        </div>
      </div>
      
      <p style={{ 
        marginTop: '1.5rem', 
        fontSize: '0.9rem', 
        color: 'var(--text-secondary)',
        lineHeight: 1.4
      }}>
        Model confidence for the selected class: {confidencePercentage}%.
      </p>

      <div className="class-probabilities">
        <div className="class-prob-row">
          <div className="class-prob-label">
            <span>Not depressed</span>
            <strong>{Math.round(notDepressedScore * 100)}%</strong>
          </div>
          <div className="class-prob-track">
            <div
              className="class-prob-fill not-depressed"
              style={{ width: `${notDepressedScore * 100}%` }}
            />
          </div>
        </div>
        <div className="class-prob-row">
          <div className="class-prob-label">
            <span>Depressed</span>
            <strong>{percentage}%</strong>
          </div>
          <div className="class-prob-track">
            <div
              className="class-prob-fill depressed"
              style={{ width: `${percentage}%` }}
            />
          </div>
        </div>
      </div>
      
      <div className="stream-breakdown">
        <div className="stream-score">
          <div className="stream-name">Face Detection</div>
          <div className="stream-val" style={{ color: 'var(--color-primary)' }}>
            {Math.round((faceConfidence ?? 0) * 100)}%
          </div>
        </div>
        <div className="stream-score">
          <div className="stream-name">Body Detection</div>
          <div className="stream-val" style={{ color: 'var(--color-accent)' }}>
            {Math.round((bodyConfidence ?? 0) * 100)}%
          </div>
        </div>
      </div>
    </div>
  );
}
