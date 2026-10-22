# Step 1 — Environment setup

Open PowerShell in this project folder and run:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

If PowerShell blocks activation, run the environment's Python directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

We will install the advanced causal-inference libraries later, when needed:

```powershell
python -m pip install -r requirements-advanced.txt
```

If any command produces an error, copy the exact message so we can troubleshoot it.
