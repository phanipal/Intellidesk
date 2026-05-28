<#
.SYNOPSIS
    IntelliDesk task runner — Windows PowerShell equivalent of a Makefile.

.DESCRIPTION
    Single entry point for all repetitive project tasks: install deps,
    generate data, train models, run tests, serve the API, clean caches.

.EXAMPLE
    .\make.ps1                               # show help
    .\make.ps1 test-all                      # run all test suites
    .\make.ps1 serve                         # start FastAPI service
    .\make.ps1 triage "VPN is down"          # one-shot triage
    .\make.ps1 clean                         # remove tmp/ + caches
#>

param(
    [Parameter(Position = 0)]
    [string]$Task = "help",

    # Catch-all for additional args (e.g. ticket text for triage / query)
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ExtraArgs
)

$ErrorActionPreference = "Stop"


# ---------------------------------------------------------------------------
# Help
# ---------------------------------------------------------------------------
function Invoke-Help {
    Write-Host ""
    Write-Host "IntelliDesk - available commands:" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "  Setup:" -ForegroundColor Yellow
    Write-Host "    .\make.ps1 install         Install all dependencies + spaCy model"
    Write-Host ""
    Write-Host "  Data:" -ForegroundColor Yellow
    Write-Host "    .\make.ps1 data            Generate ticket dataset (10k rows)"
    Write-Host "    .\make.ps1 data-small      Generate small dataset (500 rows) for smoke testing"
    Write-Host "    .\make.ps1 kb              Generate knowledge base"
    Write-Host "    .\make.ps1 data-all        Generate both data and kb"
    Write-Host ""
    Write-Host "  Modeling:" -ForegroundColor Yellow
    Write-Host "    .\make.ps1 train           Train classifier + log to MLflow"
    Write-Host "    .\make.ps1 train-quick     Train without MLflow logging"
    Write-Host "    .\make.ps1 mlflow          Open MLflow UI in browser (http://localhost:5000)"
    Write-Host "    .\make.ps1 build-index     Build FAISS retriever index from KB"
    Write-Host "    .\make.ps1 query <text>    Search KB with a query"
    Write-Host "    .\make.ps1 triage <text>   End-to-end triage (sample tickets if no arg)"
    Write-Host "    .\make.ps1 triage-json     Same, output as JSON"
    Write-Host "    .\make.ps1 validate        Run 30+ checks across data, models, and predictions"
    Write-Host "    .\make.ps1 validate-strict Same, but fail on warnings too"
    Write-Host "    .\make.ps1 demo            End-to-end pipeline demo with sample triage"
    Write-Host "    .\make.ps1 demo-fresh      Same, but regenerate everything from scratch"
    Write-Host "    .\make.ps1 drift           Generate drift reports (data + target)"
    Write-Host "    .\make.ps1 drift-full      Same + classification quality reports (slower)"
    Write-Host ""
    Write-Host "  API:" -ForegroundColor Yellow
    Write-Host "    .\make.ps1 serve           Start FastAPI service on http://localhost:8000"
    Write-Host "    .\make.ps1 serve-dev       Same with auto-reload (development)"
    Write-Host "    .\make.ps1 api-test        Hit running API with smoke-test calls"
    Write-Host "    .\make.ps1 dashboard       Start Streamlit dashboard on http://localhost:8501"
    Write-Host "    .\make.ps1 dashboard-headless  Same, no browser auto-open (for screenshots/CI)"
    Write-Host ""
    Write-Host "  Testing:" -ForegroundColor Yellow
    Write-Host "    .\make.ps1 test            Run all tests in one pytest session"
    Write-Host "    .\make.ps1 test-all        Run each test file separately with summary table"
    Write-Host "    .\make.ps1 test-all-v      Same, with verbose pytest output"
    Write-Host "    .\make.ps1 test-all-cov    Same, with coverage report"
    Write-Host "    .\make.ps1 test-cov        Single session with coverage"
    Write-Host ""
    Write-Host "  Cleanup (NEVER touches .venv or data/ or models/):" -ForegroundColor Yellow
    Write-Host "    .\make.ps1 clean-tmp       Delete tmp/ scratch dir"
    Write-Host "    .\make.ps1 clean-test      Delete pytest + coverage caches and __pycache__"
    Write-Host "    .\make.ps1 clean           Both of the above"
    Write-Host ""
}


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
function Invoke-Install {
    Write-Host "Upgrading pip..." -ForegroundColor Cyan
    python -m pip install --upgrade pip
    Write-Host "Installing dev requirements..." -ForegroundColor Cyan
    python -m pip install -r requirements-dev.txt
    Write-Host "Downloading spaCy English model..." -ForegroundColor Cyan
    python -m spacy download en_core_web_sm
    Write-Host "Install complete." -ForegroundColor Green
}


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------
function Invoke-Data {
    python -m src.generate_data --n_tickets 10000 --out data/tickets.csv
}

function Invoke-DataSmall {
    if (-not (Test-Path tmp)) { New-Item -ItemType Directory -Path tmp | Out-Null }
    python -m src.generate_data --n_tickets 500 --out tmp/tickets_smoke.csv
    Write-Host "Smoke dataset written to tmp\tickets_smoke.csv" -ForegroundColor Green
}

function Invoke-Kb {
    python -m src.generate_kb
}

function Invoke-DataAll {
    Invoke-Data
    Invoke-Kb
}


# ---------------------------------------------------------------------------
# Modeling
# ---------------------------------------------------------------------------
function Invoke-Train {
    python -m src.classifier --data data/tickets.csv
}

function Invoke-TrainNoMlflow {
    python -m src.classifier --data data/tickets.csv --no-mlflow
}

function Invoke-Mlflow {
    Write-Host "Starting MLflow UI on http://localhost:5000 (Ctrl+C to stop)" -ForegroundColor Cyan
    python -m mlflow ui
}

function Invoke-BuildIndex {
    python -m src.retriever
}

function Invoke-Query {
    if ($script:ExtraArgs -and $script:ExtraArgs.Count -gt 0) {
        $text = $script:ExtraArgs -join " "
        python -m src.retriever --query $text
    } else {
        $text = Read-Host "Enter a ticket description to search"
        python -m src.retriever --query $text
    }
}

function Invoke-Triage {
    if ($script:ExtraArgs -and $script:ExtraArgs.Count -gt 0) {
        $text = $script:ExtraArgs -join " "
        python -m src.pipeline $text
    } else {
        python -m src.pipeline
    }
}

function Invoke-TriageJson {
    if ($script:ExtraArgs -and $script:ExtraArgs.Count -gt 0) {
        $text = $script:ExtraArgs -join " "
        python -m src.pipeline $text --json
    } else {
        python -m src.pipeline --json
    }
}

function Invoke-Demo {
    python run_demo.py
}

function Invoke-DemoFresh {
    python run_demo.py --rebuild-all
}

function Invoke-Validate {
    python run_validate.py
}

function Invoke-ValidateStrict {
    python run_validate.py --strict
}

function Invoke-Drift {
    python -m monitoring.drift_report
}

function Invoke-DriftFull {
    python -m monitoring.drift_report --include-quality
}

function Invoke-Notebooks {
    Write-Host "Converting .py notebooks to .ipynb..." -ForegroundColor Cyan
    jupytext --to notebook notebooks/01_eda.py
    jupytext --to notebook notebooks/02_modeling.py
    jupytext --to notebook notebooks/03_embeddings.py
}

function Invoke-NotebooksRun {
    Write-Host "Executing all notebooks..." -ForegroundColor Cyan
    jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb
    jupyter nbconvert --to notebook --execute --inplace notebooks/02_modeling.ipynb
    jupyter nbconvert --to notebook --execute --inplace notebooks/03_embeddings.ipynb
}


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------
function Invoke-Serve {
    Write-Host "Starting IntelliDesk API on http://localhost:8000" -ForegroundColor Cyan
    Write-Host "Interactive docs: http://localhost:8000/docs" -ForegroundColor Cyan
    Write-Host "Press Ctrl+C to stop" -ForegroundColor Yellow
    python -m src.api
}

function Invoke-ServeDev {
    Write-Host "Starting IntelliDesk API in dev mode (auto-reload)" -ForegroundColor Cyan
    Write-Host "URL: http://localhost:8000  |  Docs: http://localhost:8000/docs" -ForegroundColor Cyan
    $env:INTELLIDESK_RELOAD = "true"
    try {
        python -m src.api
    } finally {
        Remove-Item Env:INTELLIDESK_RELOAD -ErrorAction SilentlyContinue
    }
}

function Invoke-ApiTest {
    Write-Host "Smoke-testing API at http://localhost:8000 (must be running in another terminal)" -ForegroundColor Cyan

    Write-Host "`n--- GET /health ---" -ForegroundColor Yellow
    Invoke-RestMethod http://localhost:8000/health | ConvertTo-Json

    Write-Host "`n--- GET /info ---" -ForegroundColor Yellow
    Invoke-RestMethod http://localhost:8000/info | ConvertTo-Json -Depth 5

    Write-Host "`n--- POST /triage ---" -ForegroundColor Yellow
    $body = @{ text = "Major SSO outage affecting our finance team" } | ConvertTo-Json
    Invoke-RestMethod -Method POST -Uri http://localhost:8000/triage `
                      -Body $body -ContentType 'application/json' |
        ConvertTo-Json -Depth 5
}

function Invoke-Dashboard {
    Write-Host "Starting Streamlit dashboard on http://localhost:8501" -ForegroundColor Cyan
    Write-Host "(For full mode, also run '.\make.ps1 serve' in another terminal)" -ForegroundColor Yellow
    streamlit run dashboard/app.py
}

function Invoke-DashboardHeadless {
    # For CI / screenshots — no browser auto-open, no telemetry prompt
    Write-Host "Starting Streamlit dashboard headlessly on http://localhost:8501" -ForegroundColor Cyan
    streamlit run dashboard/app.py `
        --server.headless true `
        --browser.gatherUsageStats false
}


# ---------------------------------------------------------------------------
# Testing
# ---------------------------------------------------------------------------
function Invoke-Test {
    python -m pytest tests/ -v
}

function Invoke-TestCov {
    python -m pytest tests/ -v --cov=src --cov-report=term-missing
}

function Invoke-TestAll {
    python run_tests.py
}

function Invoke-TestAllVerbose {
    python run_tests.py -v
}

function Invoke-TestAllCov {
    python run_tests.py --cov
}


# ---------------------------------------------------------------------------
# Cleanup — NEVER touches .venv, data/, or models/
# ---------------------------------------------------------------------------
function Invoke-CleanTmp {
    if (Test-Path tmp) {
        Remove-Item -Recurse -Force tmp
        Write-Host "Removed tmp\" -ForegroundColor Green
    } else {
        Write-Host "tmp\ already clean" -ForegroundColor DarkGray
    }
}

function Invoke-CleanTest {
    foreach ($p in @(".pytest_cache", ".coverage", "htmlcov")) {
        if (Test-Path $p) {
            Remove-Item -Recurse -Force $p
            Write-Host "Removed $p" -ForegroundColor Green
        }
    }
    foreach ($dir in @("src", "tests", "dashboard", "monitoring")) {
        if (Test-Path $dir) {
            Get-ChildItem -Path $dir -Include __pycache__ -Recurse -Directory `
                          -ErrorAction SilentlyContinue |
                ForEach-Object {
                    Remove-Item -Recurse -Force $_.FullName
                    Write-Host "Removed $($_.FullName)" -ForegroundColor Green
                }
        }
    }
}

function Invoke-Clean {
    Invoke-CleanTmp
    Invoke-CleanTest
    Write-Host "Cleaned tmp\, caches, and __pycache__ dirs (kept .venv, data\, models\)." -ForegroundColor Cyan
}


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------
switch ($Task.ToLower()) {
    "help"          { Invoke-Help }
    "install"       { Invoke-Install }

    "data"          { Invoke-Data }
    "data-small"    { Invoke-DataSmall }
    "kb"            { Invoke-Kb }
    "data-all"      { Invoke-DataAll }

    "train"         { Invoke-Train }
    "train-quick"   { Invoke-TrainNoMlflow }
    "mlflow"        { Invoke-Mlflow }
    "build-index"   { Invoke-BuildIndex }
    "query"         { Invoke-Query }
    "triage"        { Invoke-Triage }
    "triage-json"   { Invoke-TriageJson }
    "demo"          { Invoke-Demo }
    "demo-fresh"    { Invoke-DemoFresh }

    "serve"         { Invoke-Serve }
    "serve-dev"     { Invoke-ServeDev }
    "api-test"      { Invoke-ApiTest }

    "test"          { Invoke-Test }
    "test-cov"      { Invoke-TestCov }
    "test-all"      { Invoke-TestAll }
    "test-all-v"    { Invoke-TestAllVerbose }
    "test-all-cov"  { Invoke-TestAllCov }

    "clean-tmp"     { Invoke-CleanTmp }
    "clean-test"    { Invoke-CleanTest }
    "clean"         { Invoke-Clean }

    "drift"          { Invoke-Drift }
    "drift-full"     { Invoke-DriftFull }

    "dashboard"     { Invoke-Dashboard }
    "dashboard-headless" { Invoke-DashboardHeadless }

    "notebooks"      { Invoke-Notebooks }
    "notebooks-run"  { Invoke-NotebooksRun }

    "validate"         { Invoke-Validate }
    "validate-strict"  { Invoke-ValidateStrict }

    default {
        Write-Host "Unknown task: $Task" -ForegroundColor Red
        Invoke-Help
        exit 1
    }
}