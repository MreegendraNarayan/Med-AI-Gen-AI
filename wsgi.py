from app import app
import os
from waitress import serve

application = app

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    print(f"Starting server on port {port}...")
    serve(application, host="0.0.0.0", port=port)