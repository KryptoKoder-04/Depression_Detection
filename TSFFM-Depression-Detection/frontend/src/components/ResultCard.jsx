import { Download, RefreshCw, ShieldAlert, CheckCircle } from 'lucide-react';

export default function ResultCard({
  prediction,
  confidence,
  depressionProbability,
  notDepressedProbability,
  reportUrl,
  onReset,
  patientId,
  sessionDate
}) {
  const isDepressed = prediction === 'depressed';
  const selectedConfidence = typeof confidence === 'number' ? confidence : 0;
  const depressedScore = typeof depressionProbability === 'number' ? depressionProbability : 0;
  const notDepressedScore = typeof notDepressedProbability === 'number'
    ? notDepressedProbability
    : 1 - depressedScore;

  const formatPercent = (value) => `${Math.round(Math.max(0, Math.min(1, value)) * 100)}%`;

  return (
    <div className="glass-panel result-card-inner">
      <div className={`result-glow ${isDepressed ? 'depressed' : 'not-depressed'}`} />
      
      <div className={`result-badge ${isDepressed ? 'depressed' : 'not-depressed'}`}>
        {isDepressed ? 'Depression Indicators Detected' : 'No Depression Indicators Detected'}
      </div>
      
      <h2 className="result-headline">
        {isDepressed ? 'Positive Screening' : 'Negative Screening'}
      </h2>
      
      <div className="result-description">
        {isDepressed ? (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
            <div style={{ display: 'flex', gap: '0.75rem', textAlign: 'left' }}>
              <div style={{ color: 'var(--color-danger)', flexShrink: 0 }}>
                <ShieldAlert size={20} />
              </div>
              <p>
                The Two-Stream Feature Fusion Model detected key behavioral characteristics (diminished facial expressiveness, specific micro-expressions, and lower body dynamic activity) that correlate with indicators of clinical depression.
              </p>
            </div>
            <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)', borderLeft: '2px solid rgba(239, 68, 68, 0.4)', paddingLeft: '0.75rem', marginTop: '0.5rem', textAlign: 'left' }}>
              <strong>Clinical Recommendation:</strong> This screening tool is an automated assistant. A comprehensive clinical interview with a qualified mental health practitioner is recommended to establish a definitive diagnosis.
            </p>
          </div>
        ) : (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
            <div style={{ display: 'flex', gap: '0.75rem', textAlign: 'left' }}>
              <div style={{ color: 'var(--color-success)', flexShrink: 0 }}>
                <CheckCircle size={20} />
              </div>
              <p>
                The analysis indicates facial expressions, micro-expression frequencies, and posture dynamics do not align with patterns typically observed in clinical depression.
              </p>
            </div>
            <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)', borderLeft: '2px solid rgba(16, 185, 129, 0.4)', paddingLeft: '0.75rem', marginTop: '0.5rem', textAlign: 'left' }}>
              <strong>Clinical Note:</strong> Negative screening indicates that visual indicators were below the classification threshold. Continuous monitoring of psychological well-being is encouraged.
            </p>
          </div>
        )}
      </div>

      <div style={{
        background: 'rgba(255,255,255,0.02)',
        border: '1px solid rgba(255,255,255,0.05)',
        width: '100%',
        padding: '1rem',
        borderRadius: '12px',
        marginBottom: '2rem',
        display: 'grid',
        gridTemplateColumns: '1fr 1fr',
        gap: '0.5rem',
        textAlign: 'left',
        fontSize: '0.85rem',
        zIndex: 1
      }}>
        <div>
          <span style={{ color: 'var(--text-muted)' }}>Patient Reference: </span>
          <strong style={{ color: 'var(--text-primary)' }}>{patientId}</strong>
        </div>
        <div>
          <span style={{ color: 'var(--text-muted)' }}>Session Date: </span>
          <strong style={{ color: 'var(--text-primary)' }}>{sessionDate}</strong>
        </div>
        <div>
          <span style={{ color: 'var(--text-muted)' }}>Screening Confidence: </span>
          <strong style={{ color: 'var(--text-primary)' }}>{formatPercent(selectedConfidence)}</strong>
        </div>
        <div>
          <span style={{ color: 'var(--text-muted)' }}>Class Probabilities: </span>
          <strong style={{ color: 'var(--text-primary)' }}>
            Not depressed {formatPercent(notDepressedScore)} | Depressed {formatPercent(depressedScore)}
          </strong>
        </div>
      </div>

      <div className="result-actions">
        {reportUrl && (
          <a 
            href={reportUrl} 
            target="_blank" 
            rel="noopener noreferrer" 
            className="btn-primary"
            style={{ textDecoration: 'none', margin: 0 }}
          >
            <Download size={18} />
            <span>Download PDF Report</span>
          </a>
        )}
        
        <button 
          onClick={onReset} 
          className="btn-outline"
        >
          <RefreshCw size={16} />
          <span>Screen Another Video</span>
        </button>
      </div>
    </div>
  );
}
