"""Vercel Python serverless entrypoint.

Exposes the FastAPI ASGI app so Vercel can serve every /api/* route as a
serverless function. The backend package lives under ../backend/app; we add it
to sys.path and import the app. `app` is the symbol Vercel's Python runtime
looks for.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "backend"))

from app.main import app  # noqa: E402  (path set above)
