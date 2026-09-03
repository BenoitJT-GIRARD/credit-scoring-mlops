"""Start the API locally, without Docker."""

import os

import uvicorn

host = os.getenv("API_HOST", "127.0.0.1")
port = int(os.getenv("API_PORT", "8000"))

if __name__ == "__main__":
    uvicorn.run("credexp.serving.api:app", host=host, port=port, reload=False)
