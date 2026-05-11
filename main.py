"""
RAG Evaluation Suite — startup
Run: python main.py   then open http://localhost:8080
"""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

if not os.getenv("ANTHROPIC_API_KEY"):
    sys.exit(
        "ERROR: ANTHROPIC_API_KEY is not set.\n"
        "Copy .env.example to .env and add your key, then re-run."
    )

if __name__ == "__main__":
    import uvicorn
    from src.config import settings
    uvicorn.run("app:app", host="0.0.0.0", port=settings.port,
                reload=settings.env == "development")
