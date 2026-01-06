import sys
import os

# Add the backend folder to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

# Import FastAPI app
try:
    from server import app
    application = app
    print("✅ KnowledgeDottCom FastAPI app loaded successfully")
except Exception as e:
    print(f"❌ Error loading app: {e}")
    raise