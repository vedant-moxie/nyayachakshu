# NyayaChakshu — AI Traffic-Violation Enforcement Platform

An end-to-end command centre for automated traffic-violation enforcement on
Indian roads: a live perception stack detects violations from camera feeds,
resolves number plates, books offences against the Motor Vehicles Act, seals
each event into a tamper-evident evidence ledger, and dispatches challans to
the national eChallan / Vahan system.

This repository is the working software behind the *Gridlock Perception Stack*
report. It has two halves:

```
nyayachakshu/
├── frontend/     # Command-centre console (static SPA, self-contained)
│   └── index.html
└── backend/      # FastAPI enforcement API + perception/ALPR/violation pipelines
    ├── app/
    │   ├── main.py            # FastAPI app — all REST endpoints
    │   ├── schemas.py         # TrackedObject / ViolationRecord / EvidencePackage
    │   ├── hashchain.py       # SHA-256 tamper-evident evidence ledger
    │   ├── alpr.py            # Vahan/HSRP plate consensus + validation
    │   ├── penalties.py       # MV Act 1988 penalty schedule + challan totals
    │   ├── store.py           # in-memory state, seeds the ledger
    │   ├── seed.py            # the seven seed events + analytics tables
    │   └── pipelines/
    │       ├── __init__.py        # end-to-end orchestrator (run_scene)
    │       ├── perception.py      # Layers 1-5: detect → track → kinematics → route
    │       ├── alpr_preprocess.py # ALPR stages P1-P8 + quality gate
    │       └── violations.py      # per-violation geometric/temporal rule engines
    └── tests/                 # 49 stdlib tests across the three pipelines
```

## What's real here

Nothing in the backend is a façade — every "tool" implements the actual
algorithm from the report and is exercised by tests:

| Tool | What it does | Report section |
|------|--------------|----------------|
| **Evidence hash chain** | Per-camera SHA-256 running hash chain; `verify()` recomputes every link and detects tampering, insertion, deletion, reordering | Cryptographic Evidence Integrity |
| **ALPR resolver** | Multi-frame per-character majority vote + format-aware OCR-confusion repair + Vahan grammar validation | ALPR Preprocessing (P6–P8) |
| **ALPR preprocessor** | Stages P1–P8 (localise → corner keypoints → skew correct → condition classify → enhance → normalise → conditional 4× super-resolution → quality gate) | ALPR Preprocessing |
| **Perception pipeline** | IoU tracker with persistent IDs, kinematic estimation (speed/heading/stationarity), Layer-4 metadata, Layer-5 routing engine | Core Perception Pipeline |
| **Violation modules** | Wrong-side, stop-line, red-light, illegal-parking (geometry/temporal) + helmet, triple-riding, seatbelt, phone (attribute) | Violation Detection Modules |
| **Penalty engine** | MV Act section → statutory fine, rider-scaled where applicable, challan totals | Integration Interfaces |
| **eChallan dispatch** | Builds the Vahan-format payload + signed evidence-bundle URL | Integration Interfaces |

The deep-learning stages (YOLOv11 detection, BoT-SORT association, OCR,
super-resolution) are **simulated deterministically** so the whole stack runs
with no GPU, no model weights, and stdlib only — but the downstream logic
(tracking, kinematics, routing, voting, geometry rules, hashing, penalties) is
the real production logic.

## Run the backend

```bash
cd backend
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

Then open <http://localhost:8000/docs> for the interactive API, or:

```bash
curl localhost:8000/api/health
curl localhost:8000/api/dashboard
curl localhost:8000/api/pipeline/run          # full perception→violation demo
curl localhost:8000/api/ledger/verify          # tamper-evidence proof
curl -X POST localhost:8000/api/challan/EVT-7741
```

## Run the tests

```bash
cd backend
python3 tests/test_perception.py
python3 tests/test_alpr_preprocess.py
python3 tests/test_violations.py
# or, if pytest is installed:  python3 -m pytest tests/
```

## Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/api/health` | Liveness + ledger integrity summary |
| GET | `/api/console/bootstrap` | Every dataset the static console renders |
| GET | `/api/dashboard` | Overview gauges, junctions, class counts, review donut |
| GET | `/api/cases` · `/api/cases/{id}` | Detection queue / single case |
| GET | `/api/analytics` | Per-class precision / recall / F1 |
| POST | `/api/alpr/resolve` | Multi-frame plate resolution |
| POST | `/api/challan/{id}` | Build a challan from booked offences |
| POST | `/api/echallan/dispatch/{id}` | Simulate eChallan / Vahan dispatch |
| GET | `/api/ledger` · `/api/ledger/verify` | Evidence chain + integrity proof |
| GET | `/api/pipeline/run` | Run the full perception→violation stack on a synthetic scene |

## Frontend

The console (`frontend/index.html`) is a self-contained static SPA — no build
step. The backend serves the exact data shapes it renders (`/api/console/bootstrap`),
so the inline demo data can be swapped for live `fetch()` calls with no markup
changes. Deploy as a static site (see `frontend/README.md`).
