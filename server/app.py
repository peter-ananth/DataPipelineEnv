import uvicorn
import os
from app.main import app

def main():
    """
    OpenEnv multi-mode server entry point.
    Runs the FastAPI app via uvicorn.
    """
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)

if __name__ == "__main__":
    main()
