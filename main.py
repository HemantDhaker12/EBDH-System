import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from edhc.app.config.settings import settings
from edhc.app.api.routes import router

# Create directories if they don't exist
settings.create_directories()

app = FastAPI(
    title="Evidence-Based Digital Hiring Committee (EDHC) API",
    description="API for parsing JDs, normalizing candidate profiles, and ranking candidates using Learning-to-Rank.",
    version="1.0.0"
)

# Set up CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routes
app.include_router(router, prefix="/api")

@app.get("/")
def read_root():
    return {
        "status": "active",
        "message": "EDHC API is running successfully. Access interactive documentation at /docs"
    }

if __name__ == "__main__":
    # Add project root directory to path to ensure modules load correctly
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parent))
    
    uvicorn.run(
        "main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        log_level=settings.LOG_LEVEL.lower(),
        reload=True
    )
