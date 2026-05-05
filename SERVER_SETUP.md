# Doctor Assistant — Server Setup Guide

This guide explains how to set up and run the Doctor Assistant AI on a GPU server, step by step. Each step explains what you are doing and why.

---

## What This App Does

The doctor pastes a patient case. The app:
1. Extracts structured symptoms (chief complaint, PQRS modalities, mental generals)
2. Checks if the case is complete
3. Scores symptoms using the PQRS method
4. Searches homeopathic books (ChromaDB) for relevant remedies
5. Suggests the top 3 remedies with reasoning

Everything runs on your own GPU server. No data goes outside.

---

## How the App Is Structured

```
Doctor's Browser
  └── FastAPI (port 8001)     ← the web server — handles all requests
        ├── rag.py             ← searches homeopathic books (ChromaDB)
        ├── pqrs.py            ← scores symptoms
        └── vLLM (port 8000)  ← the AI model running on GPU
              └── Gemma 3 12B
```

- **FastAPI** is the main app. The browser talks to FastAPI.
- **vLLM** runs the AI model. FastAPI sends patient cases to vLLM and gets back the analysis.
- **ChromaDB** is the local vector database of homeopathic books. It lives on **E disk** (`E:\chroma_db\`). The code in `server_code/` (on D disk) reads from E disk. No internet needed.

---

## What You Need (Hardware)

| Component | Minimum |
|-----------|---------|
| GPU VRAM  | 16 GB   |
| RAM       | 16 GB   |
| Storage   | 30 GB   |

---

## Files Inside `server_code/`

```
D:\server_code\           ← all code lives here
├── main.py               ← FastAPI app (all API logic)
├── rag.py                ← searches chromaDB for remedy matches (reads from E disk)
├── pqrs.py               ← PQRS symptom scoring
├── requirements.txt      ← Python packages to install
├── .env                  ← config (vLLM URL and model name)
├── dist/                 ← built React frontend (served by FastAPI)
└── setup/
    └── build_db.py       ← only needed if you want to rebuild the database

E:\chroma_db\             ← database lives here (separate disk)
```

The code (D disk) and the database (E disk) are on different disks. You do **not** need to rebuild the database. Upload `chroma_db/` to `E:\` separately.

---

## STEP 0 — Upload the Folder to the Server

Before doing anything on the server, you need to copy the `server_code/` folder from your local computer to the GPU server.

**Using FileZilla (easiest):**

1. Download and install FileZilla from `filezilla-project.org`
2. Open FileZilla
3. At the top, fill in:
   - **Host** → server IP address
   - **Username** → your server username
   - **Password** → your server password
   - **Port** → `22`
4. Click **Quickconnect**
5. Left panel = your local computer. Right panel = the server.
6. On the **right panel**, navigate to `/home/youruser/` (where you want to put the app)
7. On the **left panel**, navigate to your `doctor-app/server_code/` folder
8. Right-click the `server_code` folder → click **Upload**

FileZilla will copy everything — including `chroma_db/`, `dist/`, and all Python files.

**Why this step?** The server has no access to your local files. FileZilla sends the files over SFTP (Secure File Transfer Protocol), which is an encrypted file transfer over SSH.

---

## STEP 1 — Connect to the Server via SSH

SSH is a secure terminal connection to the server. After uploading the files, all remaining steps are done by typing commands in the server terminal.

**From Windows PowerShell:**
```
ssh youruser@server-ip
```

**Using PuTTY (if you prefer a GUI):**
1. Download PuTTY from `putty.org`
2. Enter the server IP address → click **Open**
3. Enter your username and password when prompted

After this you will see a terminal prompt that is running on the server — not your local computer.

**Why SSH?** The server has no monitor or keyboard. SSH lets you control it remotely by typing commands.

---

## STEP 2 — Go Into the Project Folder

```bash
cd server_code
```

All remaining commands must be run from inside the `server_code/` folder.

---

## STEP 3 — Create a Python Virtual Environment

A virtual environment is an isolated Python installation just for this project. It keeps the packages for this app separate from the system Python, so there are no version conflicts.

```bash
python3 -m venv venv
```

This creates a folder called `venv/` inside `server_code/`.

**This only needs to be done once.**

---

## STEP 4 — Activate the Virtual Environment

Every time you open a new terminal on the server, you must activate the venv before running any Python commands.

```bash
source venv/bin/activate
```

After this, your terminal prompt will start with `(venv)` — this means the virtual environment is active.

**Important:** Always run this before `pip install`, `uvicorn`, or `vllm`.

---

## STEP 5 — Install App Dependencies

`requirements.txt` lists all the Python packages the app needs. Install them once:

```bash
pip install -r requirements.txt
```

**What gets installed:**
- `fastapi` — the web framework that handles HTTP requests from the browser
- `uvicorn` — the server that runs FastAPI
- `httpx` — used to make async HTTP calls from FastAPI to vLLM
- `python-dotenv` — reads the `.env` config file
- `pdfplumber` — extracts text from uploaded PDF files
- `sentence-transformers` — converts text to vectors for searching ChromaDB
- `faiss-cpu` — fast vector index used internally
- `pillow` — handles uploaded image files
- `python-multipart` — handles file uploads in FastAPI

---

## STEP 6 — Install vLLM

vLLM is the AI model server. It is installed separately because it is a large package with CUDA (GPU) dependencies and is not included in `requirements.txt`.

```bash
pip install vllm
```

**What vLLM does:** It loads the Gemma 3 12B AI model into GPU memory and keeps it there. FastAPI sends patient cases to vLLM over HTTP and gets back the AI's analysis. FastAPI never loads the model itself.

**Why not just load the model inside FastAPI?** vLLM is highly optimized — it handles multiple requests in parallel (called continuous batching), manages GPU memory efficiently, and is far faster than a basic `model.generate()` call.

> vLLM requires an NVIDIA GPU with CUDA. If no GPU is available, see the Ollama section at the bottom.

---

## STEP 7 — Log In to HuggingFace (One Time Only)

Gemma 3 is a "gated" model on HuggingFace. This means you must accept Google's license agreement before you can download the model weights. This is a one-time step.

**7a. Accept the license:**
- Go to `huggingface.co/google/gemma-3-12b-it`
- Sign in with your HuggingFace account (free to create)
- Click **Agree and access repository**

**7b. Create an access token:**
- Go to `huggingface.co/settings/tokens`
- Click **New token** → give it a name → copy the token

**7c. Log in on the server:**
```bash
pip install huggingface_hub
huggingface-cli login
```
Paste your token when prompted. Press Enter.

**Why?** When vLLM starts for the first time, it automatically downloads the Gemma model from HuggingFace (~8–10 GB). Without logging in, HuggingFace blocks the download because Gemma requires license acceptance. After the first download, the model is cached on the server and this step is never needed again.

---

## STEP 8 — Check the .env File

The `.env` file tells FastAPI where vLLM is running and which model is loaded. Open it to confirm it looks correct:

```bash
nano .env
```

Default contents (Ollama on Windows server):
```
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=gemma3:12b
```

**What these mean:**
- `OLLAMA_BASE_URL` — the address of the Ollama server. `localhost` means the same machine. Port `11434` is Ollama's default port.
- `OLLAMA_MODEL` — must exactly match the model name you pulled with `ollama pull`.

---

## STEP 9 — Start vLLM (Terminal 1)

vLLM runs as a separate process. You need to keep this terminal open while the app is running.

First, make sure the venv is active:
```bash
source venv/bin/activate
```

### If your GPU has 16 GB VRAM (use 4-bit quantization):
```bash
vllm serve google/gemma-3-12b-it \
    --quantization awq \
    --max-model-len 4096 \
    --port 8000
```

`--quantization awq` compresses the model from 16-bit to 4-bit. This reduces VRAM usage from ~25 GB to ~8 GB with a small quality trade-off. Required for 16 GB GPUs.

`--max-model-len 4096` is the maximum input+output length per request. 4096 tokens is enough for clinical cases.

### If your GPU has 24+ GB VRAM (full precision, best quality):
```bash
vllm serve google/gemma-3-12b-it \
    --max-model-len 8192 \
    --port 8000
```

### If you have 2 GPUs (for Gemma 27B):
```bash
vllm serve google/gemma-3-27b-it \
    --tensor-parallel-size 2 \
    --max-model-len 8192 \
    --port 8000
```

**The first time vLLM starts, it downloads the Gemma model from HuggingFace (~8–10 GB). This takes several minutes. Wait.**

**Wait until you see this line before moving to Step 10:**
```
INFO:     Uvicorn running on http://0.0.0.0:8000
```

---

## STEP 10 — Start FastAPI (Terminal 2)

Open a **second terminal**, SSH into the server again, then:

```bash
cd server_code
source venv/bin/activate
uvicorn main:app --host 0.0.0.0 --port 8001
```

`--host 0.0.0.0` means the app is accessible from any IP address (not just localhost). This is required so the doctor's browser can reach it.

For development (auto-restarts when you change code):
```bash
uvicorn main:app --host 0.0.0.0 --port 8001 --reload
```

**Wait until you see:**
```
INFO:     Application startup complete.
```

---

## STEP 11 — Open in Browser

On the doctor's computer, open a browser and go to:
```
http://server-ip:8001
```

You should see the Doctor Assistant UI.

**To verify vLLM is running:**
```bash
curl http://localhost:8000/v1/models
```
Should return a JSON with the model name.

**To verify FastAPI is running:**
```bash
curl http://localhost:8001/api/health
```

---

## Running in Production (tmux — Keeps Running After SSH Disconnect)

If you close the SSH terminal, the app stops. To keep both services running permanently, use `tmux`.

`tmux` is a terminal multiplexer — it keeps terminal sessions alive on the server even after you disconnect.

**Install tmux:**
```bash
sudo apt install tmux
```

**Start vLLM in a tmux session:**
```bash
tmux new -s vllm
source venv/bin/activate
vllm serve google/gemma-3-12b-it --quantization awq --max-model-len 4096 --port 8000
```
Press **Ctrl+B then D** to detach. vLLM keeps running in the background.

**Start FastAPI in another tmux session:**
```bash
tmux new -s fastapi
cd server_code
source venv/bin/activate
uvicorn main:app --host 0.0.0.0 --port 8001
```
Press **Ctrl+B then D** to detach.

**To come back later and check the logs:**
```bash
tmux attach -t vllm      # see vLLM logs
tmux attach -t fastapi   # see FastAPI logs
```

---

## Troubleshooting

### vLLM fails with "CUDA out of memory"
- Add `--quantization awq` to the vllm command to reduce VRAM usage
- Reduce `--max-model-len 4096` to `--max-model-len 2048`
- Switch from Gemma 27B to Gemma 12B

### FastAPI returns `[ERROR] vLLM call failed (503)`
- vLLM is still loading the model — wait 1–3 minutes after startup
- Check Terminal 1 to confirm vLLM is ready (look for "Uvicorn running on http://0.0.0.0:8000")

### `ModuleNotFoundError` when starting FastAPI
```bash
source venv/bin/activate
pip install -r requirements.txt
```
This happens when the venv is not activated before running uvicorn.

### HuggingFace download fails
- Run `huggingface-cli login` and enter a valid token
- Make sure you accepted the model license at `huggingface.co/google/gemma-3-12b-it`

### Browser shows 404 on `http://server-ip:8001`
- The `dist/` folder must be inside `server_code/`
- If missing: build the React frontend and copy `dist/` into `server_code/`

### RAG remedy search fails — "chroma_db not found"
- The `chroma_db/` folder must be inside `server_code/`
- It should have been uploaded in Step 0 along with the rest of `server_code/`
- To rebuild from scratch: `python setup/build_db.py`

---

## Alternative — Ollama (No GPU Required)

If the server has no NVIDIA GPU, or for testing on a regular laptop:

```bash
# Install Ollama (Linux)
curl -fsSL https://ollama.ai/install.sh | sh

# Download the model (~5 GB)
ollama pull gemma3:12b

# Start Ollama
ollama serve
```

Update `.env`:
```
VLLM_BASE_URL=http://localhost:11434
VLLM_MODEL=gemma3:12b
```

No code changes needed. Ollama exposes the same API format as vLLM. CPU performance is lower but works for testing.

---

## Quick Reference (All Commands Together)

```bash
# One-time setup — run these once on the server
cd server_code
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install vllm
pip install huggingface_hub
huggingface-cli login

# Every time — Terminal 1 (start the AI model)
cd server_code
source venv/bin/activate
vllm serve google/gemma-3-12b-it --quantization awq --max-model-len 4096 --port 8000

# Every time — Terminal 2 (start the web app)
cd server_code
source venv/bin/activate
uvicorn main:app --host 0.0.0.0 --port 8001

# Open in browser
http://server-ip:8001
```
