"""Doom Index API - Root forwarder to apps.backend.src.api.api_v2_production."""

import sys
from pathlib import Path

# Ensure apps/backend is in sys.path
root_dir = Path(__file__).resolve().parent
backend_src = root_dir / "apps" / "backend"
if str(backend_src) not in sys.path:
    sys.path.insert(0, str(backend_src))

from src.api.api_v2_production import app

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)