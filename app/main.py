from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import uploads, plagiarism

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(uploads.router, prefix="/files", tags=["files"])
app.include_router(plagiarism.router, prefix="/process", tags=["process"])

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)