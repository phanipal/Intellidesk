"""
launcher.py

Single-file menu launcher for IntelliDesk. Saves you from remembering all
the make.ps1 commands and juggling multiple terminals.

Long-running services (API, dashboard, MLflow) open in their own console
windows on Windows, so logs stay readable and the main menu keeps working.

Run:
    python launcher.py
"""

import os
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

# Track services we've started so we can show status and stop them later
RUNNING = {}

# Colors. Modern Windows Terminal and PowerShell 7 render these fine.
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
CYAN = "\033[96m"
GRAY = "\033[90m"
BOLD = "\033[1m"
RESET = "\033[0m"

PROJECT_ROOT = Path(__file__).resolve().parent

# All the long-running services this launcher can spawn.
# Each gets its own console window so logs are isolated.
SERVICES = {
    "api": {
        "name": "FastAPI",
        "url": "http://localhost:8000",
        "docs_url": "http://localhost:8000/docs",
        "command": [sys.executable, "-m", "src.api"],
    },
    "dashboard": {
        "name": "Dashboard",
        "url": "http://localhost:8501",
        "command": ["streamlit", "run", "dashboard/app.py"],
    },
    "mlflow": {
        "name": "MLflow UI",
        "url": "http://localhost:5000",
        "command": [sys.executable, "-m", "mlflow", "ui"],
    },
}


# ------------------------------------------------------------------
# Display helpers
# ------------------------------------------------------------------
def clear():
    os.system("cls" if os.name == "nt" else "clear")


def header():
    clear()
    print(f"{CYAN}{BOLD}")
    print("=" * 64)
    print("                  IntelliDesk Launcher")
    print("=" * 64)
    print(RESET)
    show_status()


def show_status():
    """Print which services are currently up."""
    print(f"{BOLD}Services running:{RESET}")
    any_running = False
    for key, svc in SERVICES.items():
        proc = RUNNING.get(key)
        if proc and proc.poll() is None:
            any_running = True
            print(f"  {GREEN}[UP]{RESET}      {svc['name']:12s}  {svc['url']}")
        else:
            print(f"  {GRAY}[stopped]{RESET} {svc['name']:12s}  {svc['url']}")
    if not any_running:
        print(f"  {GRAY}(none){RESET}")
    print()


def menu():
    """Print the main menu. Numbers are stable so muscle memory works."""
    print(f"{BOLD}Setup and data{RESET}")
    print("   1. Install dependencies (run once)")
    print("   2. Generate data, train classifier, build index (full setup)")
    print()
    print(f"{BOLD}Run services{RESET} (opens in a new window)")
    print("   3. Start API")
    print("   4. Start Dashboard")
    print("   5. Start API + Dashboard together")
    print("   6. Start MLflow UI")
    print()
    print(f"{BOLD}Check and validate{RESET}")
    print("   7. Run validation suite (39 checks)")
    print("   8. Run all tests")
    print("   9. Run end-to-end demo")
    print()
    print(f"{BOLD}Analysis{RESET}")
    print("  10. Generate drift reports")
    print("  11. Open notebooks in VS Code")
    print()
    print(f"{BOLD}Utilities{RESET}")
    print("  12. Open all running services in browser")
    print("  13. Stop all running services")
    print()
    print(f"   0. Exit")
    print()


# ------------------------------------------------------------------
# Command runners
# ------------------------------------------------------------------
def run_inline(cmd, description):
    """
    Run a one-shot command and wait for it to finish, showing output here
    in the launcher window. Used for things like data generation, training,
    and tests.
    """
    print(f"\n{CYAN}>> {description}{RESET}")
    print(f"{GRAY}   {' '.join(str(c) for c in cmd)}{RESET}\n")
    try:
        result = subprocess.run(cmd, cwd=PROJECT_ROOT, check=False)
        if result.returncode == 0:
            print(f"\n{GREEN}>> Done.{RESET}")
        else:
            print(f"\n{RED}>> Exit code {result.returncode}{RESET}")
    except FileNotFoundError as e:
        print(f"\n{RED}>> Command not found: {e}{RESET}")
        print(f"{YELLOW}>> Did you activate the venv? Try option 1 to install deps.{RESET}")
    except KeyboardInterrupt:
        print(f"\n{YELLOW}>> Cancelled by user{RESET}")
    input(f"\n{GRAY}Press Enter to return to menu...{RESET}")


def start_service(key, open_browser=True):
    """Spawn a service in its own console window."""
    svc = SERVICES[key]

    # Skip if it's already up
    proc = RUNNING.get(key)
    if proc and proc.poll() is None:
        print(f"{YELLOW}{svc['name']} is already running at {svc['url']}{RESET}")
        time.sleep(2)
        return

    print(f"\n{CYAN}>> Starting {svc['name']}...{RESET}")

    # On Windows, CREATE_NEW_CONSOLE pops a new terminal window so service
    # logs don't pollute the launcher menu. On Linux/Mac, it just runs in
    # the background and you'd see logs only via `ps` or by checking files.
    try:
        if os.name == "nt":
            proc = subprocess.Popen(
                svc["command"],
                cwd=PROJECT_ROOT,
                creationflags=subprocess.CREATE_NEW_CONSOLE,
            )
        else:
            proc = subprocess.Popen(svc["command"], cwd=PROJECT_ROOT)
    except FileNotFoundError as e:
        print(f"{RED}>> Could not start: {e}{RESET}")
        print(f"{YELLOW}>> Make sure dependencies are installed (option 1).{RESET}")
        time.sleep(3)
        return

    RUNNING[key] = proc

    # Give the service a moment to bind to its port before opening the browser
    print(f"{GRAY}>> Waiting for service to start...{RESET}")
    time.sleep(4)

    if open_browser:
        url = svc.get("docs_url", svc["url"])
        print(f"{CYAN}>> Opening {url}{RESET}")
        webbrowser.open(url)

    print(f"{GREEN}>> {svc['name']} should be running at {svc['url']}{RESET}")
    time.sleep(2)


def stop_service(key):
    """Politely terminate a service. SIGKILL after 5s if it doesn't listen."""
    proc = RUNNING.get(key)
    if not proc:
        return
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        print(f"{GREEN}>> Stopped {SERVICES[key]['name']}{RESET}")
    del RUNNING[key]


def stop_all():
    if not RUNNING:
        print(f"{GRAY}>> Nothing to stop.{RESET}")
        time.sleep(2)
        return
    for key in list(RUNNING.keys()):
        stop_service(key)
    print(f"{GREEN}>> All services stopped.{RESET}")
    time.sleep(2)


def open_all_in_browser():
    if not RUNNING:
        print(f"{YELLOW}>> No services are running. Start something first.{RESET}")
        time.sleep(2)
        return
    for key in RUNNING:
        if RUNNING[key].poll() is None:
            url = SERVICES[key].get("docs_url", SERVICES[key]["url"])
            print(f"{CYAN}>> Opening {url}{RESET}")
            webbrowser.open(url)
            time.sleep(0.5)
    time.sleep(2)


# ------------------------------------------------------------------
# Action dispatch
# ------------------------------------------------------------------
def action_install():
    """One-time setup. Runs pip install + spaCy model download."""
    run_inline(
        [sys.executable, "-m", "pip", "install", "-r", "requirements-dev.txt"],
        "Installing Python dependencies (this can take a few minutes)",
    )
    run_inline(
        [sys.executable, "-m", "spacy", "download", "en_core_web_sm"],
        "Downloading spaCy English model",
    )


def action_full_setup():
    """Generate data, train classifier, build retriever. Idempotent."""
    print(f"\n{CYAN}>> Running full setup. Takes about 2 minutes.{RESET}\n")
    steps = [
        ([sys.executable, "-m", "src.generate_data"], "Generating tickets"),
        ([sys.executable, "-m", "src.generate_kb"], "Generating knowledge base"),
        ([sys.executable, "-m", "src.classifier", "--data", "data/tickets.csv",
          "--no-mlflow"], "Training classifier"),
        ([sys.executable, "-m", "src.retriever"], "Building retriever index"),
    ]
    for cmd, desc in steps:
        print(f"\n{CYAN}>> {desc}...{RESET}")
        result = subprocess.run(cmd, cwd=PROJECT_ROOT, check=False)
        if result.returncode != 0:
            print(f"{RED}>> Setup failed at: {desc}{RESET}")
            input(f"\n{GRAY}Press Enter to return to menu...{RESET}")
            return
    print(f"\n{GREEN}>> Setup complete. Ready to serve.{RESET}")
    input(f"\n{GRAY}Press Enter to return to menu...{RESET}")


def action_open_notebooks():
    """Open all three notebooks in VS Code."""
    notebooks = [
        "notebooks/01_eda.py",
        "notebooks/02_modeling.py",
        "notebooks/03_embeddings.py",
    ]
    print(f"\n{CYAN}>> Opening notebooks in VS Code...{RESET}")
    try:
        subprocess.run(["code"] + notebooks, cwd=PROJECT_ROOT, check=False, shell=True)
        print(f"{GREEN}>> Notebooks opened.{RESET}")
    except FileNotFoundError:
        print(f"{RED}>> VS Code not found in PATH.{RESET}")
        print(f"{YELLOW}>> Install VS Code or add 'code' to PATH.{RESET}")
    time.sleep(2)


# ------------------------------------------------------------------
# Main loop
# ------------------------------------------------------------------
def confirm_exit():
    """If services are still running, ask before killing them."""
    running = [k for k, p in RUNNING.items() if p.poll() is None]
    if running:
        print(f"\n{YELLOW}>> Services still running: {', '.join(running)}{RESET}")
        choice = input("Stop them before exiting? [Y/n]: ").strip().lower()
        if choice in ("", "y", "yes"):
            stop_all()
        else:
            print(f"{YELLOW}>> Leaving services running. Close their windows manually.{RESET}")


def main():
    while True:
        header()
        menu()
        try:
            choice = input(f"{BOLD}Choose option [0-13]: {RESET}").strip()
        except (KeyboardInterrupt, EOFError):
            confirm_exit()
            sys.exit(0)

        if choice == "0":
            confirm_exit()
            sys.exit(0)

        elif choice == "1":
            action_install()

        elif choice == "2":
            action_full_setup()

        elif choice == "3":
            start_service("api")

        elif choice == "4":
            start_service("dashboard")

        elif choice == "5":
            # Start API first, then dashboard a second later so dashboard
            # finds the API healthy on first request
            start_service("api", open_browser=False)
            time.sleep(2)
            start_service("dashboard")

        elif choice == "6":
            start_service("mlflow")

        elif choice == "7":
            run_inline(
                [sys.executable, "run_validate.py"],
                "Running validation suite (39 checks)",
            )

        elif choice == "8":
            run_inline(
                [sys.executable, "run_tests.py"],
                "Running all tests",
            )

        elif choice == "9":
            run_inline(
                [sys.executable, "run_demo.py"],
                "Running end-to-end demo",
            )

        elif choice == "10":
            run_inline(
                [sys.executable, "-m", "monitoring.drift_report"],
                "Generating drift reports",
            )

        elif choice == "11":
            action_open_notebooks()

        elif choice == "12":
            open_all_in_browser()

        elif choice == "13":
            stop_all()

        else:
            print(f"{RED}>> Invalid option. Try again.{RESET}")
            time.sleep(1)


if __name__ == "__main__":
    main()