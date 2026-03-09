# Decentralized Pocket Memory (MVP)

Minimal local memory system with:

- ingestion (`pdf`, `url`, `text`)
- chunking + distillation into knowledge crystals
- embeddings
- vector retrieval (Python fallback or C++/FAISS)
- FastAPI backend + Streamlit dashboard

## Architecture

```text
Data Sources (docs/url/text)
  -> Ingestion (Python adapters)
  -> Processing + Distillation (Python ML layer)
  -> Embeddings (Python)
  -> Vector Index (C++ module, FAISS optional)
  -> Query Engine (FastAPI)
  -> UI + Metrics (Streamlit)
```

## Current Project Structure

```text
app/
  api/          # FastAPI routes + request/response models
  core/         # index client + Python fallback index
  ml/           # adapters, chunking, distillation, embeddings
  services/     # memory engine + telemetry
cpp/
  src/          # pybind11 bindings + C++ vector index
benchmarks/     # benchmark scripts + demo workflow
streamlit_app.py
requirements.txt
```

## Supported Sources (MVP)

- Implemented: `pdf`, `url`, `text`
- Planned stubs: `slack`, `discord`, `github`
- Unavailable source types return fallback guidance to `pdf/url/text`.

---

## Run on WSL (Recommended)

WSL path mapping:
- `D:\ProjectsYop\Decentralized-Pocket-Memory` -> `/mnt/d/ProjectsYop/Decentralized-Pocket-Memory`

### 1) First-time setup

```bash
cd /mnt/d/ProjectsYop/Decentralized-Pocket-Memory
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
pip install pybind11
```

### 2) Run API + UI

Terminal 1:

```bash
cd /mnt/d/ProjectsYop/Decentralized-Pocket-Memory
source .venv/bin/activate
uvicorn app.api.main:app --reload
```

Terminal 2:

```bash
cd /mnt/d/ProjectsYop/Decentralized-Pocket-Memory
source .venv/bin/activate
streamlit run streamlit_app.py
```

Use API base URL `http://127.0.0.1:8000` in Streamlit.

Theme toggle:
- Use sidebar `Appearance -> Theme` to switch between `Auto`, `Light`, and `Dark` at runtime.
- `Auto` follows Streamlit defaults, while `Light`/`Dark` apply in-app card and panel styles.

---

## Build C++ Module with FAISS (WSL)

This is optional. Without this, the app runs with Python fallback index.

### 1) Install system deps

```bash
sudo apt update
sudo apt install -y build-essential cmake pkg-config python3-dev libopenblas-dev liblapack-dev libfaiss-dev libomp-dev
```

### 2) Configure + build

```bash
cd /mnt/d/ProjectsYop/Decentralized-Pocket-Memory
source .venv/bin/activate
mkdir -p ~/pm_build
cmake -S cpp -B ~/pm_build -DUSE_FAISS=ON -Dpybind11_DIR="$(python -m pybind11 --cmakedir)"
cmake --build ~/pm_build -j
```

### 3) Copy built module and verify

```bash
cp ~/pm_build/pocket_memory_cpp*.so /mnt/d/ProjectsYop/Decentralized-Pocket-Memory/
cd /mnt/d/ProjectsYop/Decentralized-Pocket-Memory
source .venv/bin/activate
python -c "import pocket_memory_cpp; print('ok')"
```

If import prints `ok`, C++ module is active.

---

## Run on Windows Native (No WSL)

Works, but FAISS/C++ setup is usually harder than WSL.

### 1) Setup

```powershell
cd D:\ProjectsYop\Decentralized-Pocket-Memory
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -U pip
pip install -r requirements.txt
```

### 2) Run

Terminal 1:

```powershell
cd D:\ProjectsYop\Decentralized-Pocket-Memory
.\.venv\Scripts\Activate.ps1
uvicorn app.api.main:app --reload
```

Terminal 2:

```powershell
cd D:\ProjectsYop\Decentralized-Pocket-Memory
.\.venv\Scripts\Activate.ps1
streamlit run streamlit_app.py
```

---

## Retrieval Modes and Endpoints

Modes:
- `flat` (exact baseline)
- `hnsw` (ANN; requires C++ FAISS build)
- `ivfpq` (ANN/compressed; requires C++ FAISS build)

Endpoints:
- `GET /health`
- `GET /index/mode`
- `POST /index/mode` with `{"mode":"flat|hnsw|ivfpq"}`
- `GET /metrics`
- `GET /sources/status`

If FAISS/C++ is unavailable, mode falls back to flat behavior.

---

## After Code Updates: What To Re-run

### Python-only changes (`app/`, `streamlit_app.py`, API code)
- restart API and Streamlit terminals.
- no C++ rebuild needed.

### C++ changes (`cpp/src/*`, `cpp/CMakeLists.txt`)
- rebuild module:

```bash
cd /mnt/d/ProjectsYop/Decentralized-Pocket-Memory
source .venv/bin/activate
cmake -S cpp -B ~/pm_build -DUSE_FAISS=ON -Dpybind11_DIR="$(python -m pybind11 --cmakedir)"
cmake --build ~/pm_build -j
cp ~/pm_build/pocket_memory_cpp*.so .
```

- restart API after copying new `.so`.

---

## Optional: Transformer Embeddings

By default, deterministic fallback embeddings are used.

To enable `sentence-transformers` runtime embeddings:

WSL/Linux:

```bash
export ENABLE_TRANSFORMER_EMBEDDINGS=1
```

PowerShell:

```powershell
$env:ENABLE_TRANSFORMER_EMBEDDINGS="1"
```

---

## Benchmarks and Demo

Baseline benchmark:

```bash
python benchmarks/run_benchmark.py
```

Mode comparison against running API:

```bash
python benchmarks/run_mode_comparison.py --base-url http://127.0.0.1:8000
```

Full demo checklist:
- `benchmarks/DEMO_WORKFLOW.md`

---

## Troubleshooting

- **`pybind11Config.cmake not found`**
  - install pybind11 in active env and pass:
  - `-Dpybind11_DIR="$(python -m pybind11 --cmakedir)"`

- **OpenMP target error (`OpenMP::OpenMP_CXX`)**
  - install OpenMP runtime/dev:
  - `sudo apt install -y libomp-dev`

- **CMake permission issues under `/mnt/d`**
  - keep source on `/mnt/d`, but use build dir in Linux home (`~/pm_build`).

- **WSL import mismatch**
  - `.so` built in WSL works with WSL Python only.
  - Windows `.pyd` is a separate build target.
