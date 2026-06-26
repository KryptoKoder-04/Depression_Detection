import { useState, useRef } from 'react';
import { Upload, FileVideo, AlertCircle, Calendar, User, CheckCircle } from 'lucide-react';

export default function UploadBox({ onUploadStart }) {
  const [file, setFile] = useState(null);
  const [patientId, setPatientId] = useState('');
  const [sessionDate, setSessionDate] = useState(new Date().toISOString().split('T')[0]);
  const [dragActive, setDragActive] = useState(false);
  const [error, setError] = useState('');
  
  const fileInputRef = useRef(null);
  
  const allowedExtensions = ['.mp4', '.avi', '.mov', '.mkv'];

  const validateFile = (selectedFile) => {
    if (!selectedFile) return false;
    
    const fileExtension = selectedFile.name.substring(selectedFile.name.lastIndexOf('.')).toLowerCase();
    if (!allowedExtensions.includes(fileExtension)) {
      setError(`Unsupported file format. Please upload one of: ${allowedExtensions.join(', ')}`);
      setFile(null);
      return false;
    }
    
    setError('');
    setFile(selectedFile);
    return true;
  };

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      validateFile(e.dataTransfer.files[0]);
    }
  };

  const handleChange = (e) => {
    e.preventDefault();
    if (e.target.files && e.target.files[0]) {
      validateFile(e.target.files[0]);
    }
  };

  const triggerFileInput = () => {
    fileInputRef.current.click();
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!file) {
      setError('Please select a video file to analyze.');
      return;
    }
    
    onUploadStart(file, patientId || 'unknown', sessionDate);
  };

  const formatBytes = (bytes) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const dm = 2;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i];
  };

  return (
    <div className="glass-panel upload-card">
      <h2 style={{ fontSize: '1.5rem', marginBottom: '0.5rem' }}>📹 Depression Screening Analysis</h2>
      <p style={{ color: 'var(--text-secondary)', fontSize: '0.95rem', marginBottom: '2rem' }}>
        Upload a clinical interview video to run Two-Stream Feature Fusion (TSFFM) and Bidirectional LSTM temporal screening.
      </p>

      <form onSubmit={handleSubmit}>
        <div 
          className={`dropzone ${dragActive ? 'drag-active' : ''}`}
          onDragEnter={handleDrag}
          onDragOver={handleDrag}
          onDragLeave={handleDrag}
          onDrop={handleDrop}
          onClick={triggerFileInput}
        >
          <input 
            ref={fileInputRef}
            type="file" 
            className="hidden-file-input" 
            style={{ display: 'none' }}
            accept={allowedExtensions.join(',')}
            onChange={handleChange}
          />
          
          <div className="upload-icon-container">
            <Upload size={32} />
          </div>
          
          <h3 className="upload-title">Drag & drop video file</h3>
          <p className="upload-subtitle">or click to browse your local directory</p>
          <p className="file-spec">Supported formats: MP4, AVI, MOV, MKV (Max 200MB)</p>
        </div>

        {error && (
          <div style={{ 
            display: 'flex', 
            alignItems: 'center', 
            gap: '0.5rem', 
            color: 'var(--color-danger)', 
            marginTop: '1rem',
            fontSize: '0.9rem',
            justifyContent: 'center'
          }}>
            <AlertCircle size={16} />
            <span>{error}</span>
          </div>
        )}

        {file && (
          <div className="selected-file-info">
            <div className="file-details">
              <div style={{ color: 'var(--color-primary)' }}>
                <FileVideo size={24} />
              </div>
              <div>
                <div className="file-name">{file.name}</div>
                <div className="file-size">{formatBytes(file.size)}</div>
              </div>
            </div>
            <div style={{ color: 'var(--color-success)', display: 'flex', alignItems: 'center' }}>
              <CheckCircle size={20} />
            </div>
          </div>
        )}

        <div className="meta-form">
          <div className="form-group">
            <label className="form-label" htmlFor="patient-id">
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <User size={14} />
                <span>Patient ID / Reference</span>
              </div>
            </label>
            <input 
              id="patient-id"
              type="text" 
              className="form-input" 
              placeholder="e.g. PT-8849"
              value={patientId}
              onChange={(e) => setPatientId(e.target.value)}
            />
          </div>

          <div className="form-group">
            <label className="form-label" htmlFor="session-date">
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <Calendar size={14} />
                <span>Session Date</span>
              </div>
            </label>
            <input 
              id="session-date"
              type="date" 
              className="form-input" 
              value={sessionDate}
              onChange={(e) => setSessionDate(e.target.value)}
            />
          </div>
        </div>

        <button 
          type="submit" 
          className="btn-primary" 
          disabled={!file}
        >
          <Upload size={18} />
          <span>Analyze Patient Video</span>
        </button>
      </form>
    </div>
  );
}
