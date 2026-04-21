"""
Quick launcher for the ECG Diagnosis AI web server.

Usage:
    python run_server.py
"""

import subprocess
import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
VENV_DIR = os.path.join(PROJECT_ROOT, ".venv")
VENV_PYTHON = os.path.join(VENV_DIR, "bin", "python")
REQUIRED_PACKAGES = ["torch", "numpy", "scipy", "pyyaml", "wfdb"]
PORT = 8000


def ensure_venv():
    """Create virtual environment if it doesn't exist."""
    if not os.path.exists(VENV_PYTHON):
        print("🔧 Creating virtual environment...")
        subprocess.check_call([sys.executable, "-m", "venv", VENV_DIR])
        print("   ✓ Virtual environment created")


def ensure_packages():
    """Install missing packages."""
    missing = []
    for pkg in REQUIRED_PACKAGES:
        try:
            subprocess.check_call(
                [VENV_PYTHON, "-c", f"import {pkg}"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except subprocess.CalledProcessError:
            missing.append(pkg)

    if missing:
        print(f"📦 Installing missing packages: {', '.join(missing)}")
        subprocess.check_call(
            [VENV_PYTHON, "-m", "pip", "install", "--quiet"] + missing
        )
        print("   ✓ Packages installed")


def main():
    print("=" * 50)
    print("  🫀 ECG Diagnosis AI — Server Launcher")
    print("=" * 50)

    ensure_venv()
    ensure_packages()

    print(f"\n🚀 Starting server on http://localhost:{PORT}\n")

    os.execv(
        VENV_PYTHON,
        [VENV_PYTHON, "-m", "webapp.main", "--port", str(PORT)],
    )


if __name__ == "__main__":
    main()
