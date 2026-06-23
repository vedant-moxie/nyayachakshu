# NyayaChakshu Backend

FastAPI enforcement API + the perception / ALPR / violation pipelines.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000   # http://localhost:8000/docs
```

Tests (stdlib only, no pytest required):

```bash
python3 tests/test_perception.py
python3 tests/test_alpr_preprocess.py
python3 tests/test_violations.py
```

See the top-level `README.md` for architecture and the full endpoint list.
Requires Python 3.11+ (developed on 3.14).
