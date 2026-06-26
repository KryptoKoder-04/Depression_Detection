import os
import time
from datetime import datetime
from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from config import UPLOADS_DIR, ALLOWED_EXTENSIONS
from services.extractor import extract_features_from_video
from services.inference import run_inference
from services.pdf_report import generate_pdf_report

router = APIRouter(prefix="/api")

def is_allowed_file(filename: str) -> bool:
    _, ext = os.path.splitext(filename)
    return ext.lower() in ALLOWED_EXTENSIONS

@router.post("/predict")
async def predict(
    file: UploadFile = File(...),
    patient_id: str = Form("unknown"),
    session_date: str = Form(None)
):
    # 1. Validate file extension
    if not is_allowed_file(file.filename):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file extension. Allowed types: {', '.join(ALLOWED_EXTENSIONS)}"
        )
        
    if session_date is None:
        session_date = datetime.now().strftime("%Y-%m-%d")
        
    start_time = time.time()
    
    # 2. Save uploaded video to uploads directory
    temp_video_path = os.path.join(UPLOADS_DIR, f"{int(time.time())}_{file.filename}")
    try:
        with open(temp_video_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save uploaded video: {e}")
        
    try:
        # 3. Process video and extract coordinate features
        print(f"Processing video {file.filename} with MediaPipe extractor...")
        face_features, pose_features, extraction_meta = extract_features_from_video(temp_video_path)
        
        # 4. Run model inference using our trained BiLSTM weights + expression calibration
        print("Running model inference...")
        expression_metrics = extraction_meta.get("expression_metrics", None)
        prediction, confidence, prob_depressed = run_inference(face_features, pose_features, expression_metrics)
        
        duration = time.time() - start_time
        
        # 5. Generate clinical screening PDF report
        print("Generating PDF report...")
        pdf_path, report_filename = generate_pdf_report(
            patient_id=patient_id,
            session_date=session_date,
            filename=file.filename,
            prediction=prediction,
            confidence=confidence,
            prob_depressed=prob_depressed,
            duration=duration
        )
        
        # 6. Return response payload
        return {
            "prediction": prediction,
            "confidence": float(confidence),
            "depression_probability": float(prob_depressed),
            "not_depressed_probability": float(1.0 - prob_depressed),
            "face_confidence": float(extraction_meta["face_detection_rate"]),
            "body_confidence": float(extraction_meta["body_detection_rate"]),
            "face_detection_rate": float(extraction_meta["face_detection_rate"]),
            "body_detection_rate": float(extraction_meta["body_detection_rate"]),
            "processing_time": float(duration),
            "frames_processed": int(extraction_meta["frames_processed"]),
            "sampled_frames": int(extraction_meta["sampled_frames"]),
            "sequence_length": int(extraction_meta["sequence_length"]),
            "report_url": f"/reports/{report_filename}"
        }
        
    except Exception as e:
        print(f"Error during video processing: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while analyzing the video: {str(e)}"
        )
        
    finally:
        # Clean up temporary uploaded video file to save disk space
        if os.path.exists(temp_video_path):
            try:
                os.remove(temp_video_path)
            except Exception as e:
                print(f"Failed to clean up temp video {temp_video_path}: {e}")
