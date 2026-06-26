import os
# pyrefly: ignore [missing-import]
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from backend.config import REPORTS_DIR
from backend.api.predict import router as predict_router




# Initialize FastAPI App
app = FastAPI(
    title="🎓 BTP Depression Detection API",
    description="FastAPI Backend for AI-Assisted Depression Screening using TSFFM-BiLSTM",
    version="1.0.0"
)

# Configure CORS Middleware
# Allows the React Vite frontend (running on http://localhost:5173) to communicate with the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, set specific domains (e.g. ["http://localhost:5173"])
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Static Reports Directory
# Allows users to download generated PDF reports via url e.g. http://localhost:8000/reports/report_name.pdf
app.mount("/reports", StaticFiles(directory=REPORTS_DIR), name="reports")

# Include Routers
app.include_router(predict_router)

@app.get("/")
async def root():
    """
    Heartbeat endpoint to verify backend status.
    """
    return {
        "status": "healthy",
        "message": "🎓 Depression Detection API Server is running",
        "framework": "FastAPI",
        "pipeline": "MediaPipe FaceMesh + MediaPipe Pose + TSFFM-BiLSTM (PyTorch)"
    }

if __name__ == "__main__":
    # Start the Uvicorn server
    print("Starting FastAPI Backend Server...")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
