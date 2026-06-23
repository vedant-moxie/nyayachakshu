# NyayaChakshu — AI Traffic-Violation Enforcement Platform

[![CI](https://github.com/vedant-moxie/nyayachakshu/actions/workflows/ci.yml/badge.svg)](https://github.com/vedant-moxie/nyayachakshu/actions/workflows/ci.yml)

**Live:** [nyayachakshu-console.vercel.app](https://nyayachakshu-console.vercel.app) ·
**Verify it works:** [/proof](https://nyayachakshu-console.vercel.app/proof) ·
**API docs:** [/docs](https://nyayachakshu-console.vercel.app/docs)

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

## Verify it works (don't take our word for it)

Three independent ways for anyone to confirm the system actually does what it claims:

1. **Open the live proof page** → <https://nyayachakshu-console.vercel.app/proof>
   It runs five real checks in your browser against the live server and shows
   pass/fail with the expected-vs-actual values for each.

2. **Recompute a cryptographic hash by hand.** Hit
   [`/api/proof`](https://nyayachakshu-console.vercel.app/api/proof) — the
   hash-chain check prints the exact bytes that were hashed. Paste them into
   `shasum` and confirm you get the same digest the server reports:
   ```bash
   printf '%s' '<canonical bytes from /api/proof>' | shasum -a 256
   ```
   It will match the sealed `record_hash` — proving the evidence ledger is real
   SHA-256, not a stored string.

3. **Watch tamper-detection catch a forged record.**
   [`/api/ledger/verify`](https://nyayachakshu-console.vercel.app/api/ledger/verify)
   recomputes every link in the chain; the `/api/proof` tamper check edits a
   sealed record and shows `verify()` flagging the break.

4. **The CI badge above** runs all 49 pipeline tests **plus** the self-verification
   suite on every push, across Python 3.11–3.13. Green = independently reproduced
   on GitHub's runners, not just on our machine.

## Detect violations from a real image (real YOLO inference)

Upload an actual traffic photo and the backend runs **real YOLOv8m object
detection** (ultralytics, COCO-pretrained) — genuine inference, not simulation.
It returns the detected objects, an annotated image, and the violations that
object-detection can *honestly* support.

```bash
cd backend
python3.11 -m venv .venv311          # torch has no Python 3.14 wheels yet
source .venv311/bin/activate
pip install -r requirements-detect.txt
uvicorn app.main:app --port 8100     # serves the API *and* the pages
```

Open **<http://localhost:8100/detect.html>**, drop in an image, and watch it
detect. On `triple_riding.jpeg` it really fires:

```
violations: triple_riding @ 0.71  ("3 persons associated with one two-wheeler")
challan:    ₹1000  (MV Act §194C)
```

**What is and isn't real here — no faking:**
- ✅ **Real:** vehicle/person/motorcycle/traffic-light detection, and
  **triple-riding** derived by associating ≥3 riders to one two-wheeler.
- ⚠️ **Honestly out of scope for COCO:** helmet-absence and seatbelt/phone are
  *not* COCO classes — the endpoint reports them as "requires the fine-tuned
  multi-task head" (perception.tex / seatbelt.tex) rather than inventing a
  result. Riders are detected; helmet state is not classified.

This runs locally (torch is ~2 GB and won't fit the free serverless tier). On
the deployed site `/api/detect` returns a clear `503` telling you to run the
local service; the `/detect` page lets you point its API base at `localhost:8100`.

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
