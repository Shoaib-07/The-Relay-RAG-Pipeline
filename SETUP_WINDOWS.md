# Windows 11 setup

Open this folder in VS Code. Open Terminal > New Terminal (PowerShell), then run:

```powershell
py -3 --version
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
```

If PowerShell blocks activation, run this once in the current terminal:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

Edit `.env` and paste your Groq key after `GROQ_API_KEY=`. Do not share it or upload `.env`.

Download the exact paper versions:
```powershell
python download_papers.py
```

Check the local setup:
```powershell
python check_env.py
```

## Build the retrieval index

After downloading the papers and checking the environment, run:

```powershell
python -m src.ingestion
```

This extracts page-aware text chunks and saves the index under `data/index/`.

## Run the question-answering pipeline

```powershell
python -m src.pipeline
```

Enter a question about one of the three papers. Type `exit` to stop.

## Run the local stand-in service

Open a second VS Code terminal, activate the same virtual environment, and run:

```powershell
python -m uvicorn src.standin:app --host 127.0.0.1 --port 8001
```

Check its health endpoint in another terminal:

```powershell
Invoke-RestMethod http://127.0.0.1:8001/health
```

## Run the recorded evaluation scripts

```powershell
python -m src.test_retrieval
python src/summarize_load.py
```

The load-test scripts and retry experiment are available under `src/`. The recorded experiment outputs are preserved in `logs/`.

## Security

Never upload `.env`, which contains the private Groq API key. Use `.env.example` as the configuration template. The local stand-in can be run without sending load-test traffic to Groq.