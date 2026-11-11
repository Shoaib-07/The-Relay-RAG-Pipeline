# Phase 4 — Stand-in service and raw-results logger

## Start the stand-in (PowerShell terminal 1)
```powershell
py -m uvicorn src.standin:app --host 127.0.0.1 --port 8001
```

## Check health (PowerShell terminal 2)
```powershell
Invoke-RestMethod http://127.0.0.1:8001/health
```

## Test one simulated request
```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8001/chat -ContentType "application/json" -Body '{"question":"test question","stream":false}'
```

## Notes
- This is a controllable stand-in, not a real LLM and not calibrated yet.
- It intentionally returns a fixed diagnostic response.
- Defaults are conservative; set STANDIN_* environment variables before starting uvicorn to change behavior.
- Never use the stand-in response as a correct answer in quality evaluation.
- results_logger.py creates the exact required CSV header. Do not fill it with invented measurements.
