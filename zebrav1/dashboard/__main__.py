"""Run: python -m zebrav1.dashboard"""
import uvicorn
from zebrav1.dashboard.server import app

if __name__ == "__main__":
    print("=" * 50)
    print("vzebra Brain Dashboard")
    print("Open: http://localhost:8000")
    print("=" * 50)
    uvicorn.run(app, host="0.0.0.0", port=8000)
