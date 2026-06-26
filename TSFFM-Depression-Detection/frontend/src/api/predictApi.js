const API_BASE_URL = "https://depression-detection-46s2.onrender.com";

/**
 * Uploads a video file and associated metadata to the backend for depression detection.
 * 
 * @param {File} file - The video file (.mp4, .avi, etc.)
 * @param {string} patientId - The unique ID of the patient/participant
 * @param {string} sessionDate - The date of the recording (YYYY-MM-DD)
 * @returns {Promise<Object>} The JSON response from the backend
 */
export async function uploadAndPredict(file, patientId, sessionDate) {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('patient_id', patientId || 'unknown');
  
  if (sessionDate) {
    formData.append('session_date', sessionDate);
  }

  try {
    const response = await fetch(`${API_BASE_URL}/api/predict`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || `Server responded with status ${response.status}`);
    }

    const data = await response.json();
    
    // Map relative report_url to absolute backend URL
    if (data.report_url) {
      data.report_url = `${API_BASE_URL}${data.report_url}`;
    }
    
    return data;
  } catch (error) {
    console.error('API Error in uploadAndPredict:', error);
    throw error;
  }
}
