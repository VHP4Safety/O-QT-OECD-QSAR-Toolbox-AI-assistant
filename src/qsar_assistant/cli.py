# SPDX-FileCopyrightText: 2025 Ivo Djidrovski <i.djidrovski@uu.nl>
#
# SPDX-License-Identifier: MIT

import subprocess
import sys
import os
from pathlib import Path

def main():
    """Runs the Streamlit application."""
    # Find the path to app.py within the installed package
    try:
        # Assuming cli.py is in src/qsar_assistant/cli.py
        # and app.py will be in src/qsar_assistant/app.py
        app_path = Path(__file__).parent / "app.py"
        if not app_path.exists():
             # This check might be redundant if installation places it correctly,
             # but good for robustness, especially during development.
             print(f"Attempted path: {app_path.resolve()}", file=sys.stderr)
             raise FileNotFoundError("Could not locate app.py within the package structure.")

        # Construct the command using sys.executable to ensure the correct Python env
        command = [sys.executable, "-m", "streamlit", "run", str(app_path)]

        # Optional: Add any arguments passed to the cli command
        # command.extend(sys.argv[1:])

        print(f"Running command: {' '.join(command)}")
        # Use check=True to raise CalledProcessError if Streamlit fails
        subprocess.run(command, check=True, cwd=Path(__file__).parent.parent.parent) # Run from project root

    except FileNotFoundError as e:
         print(f"Error: {e}", file=sys.stderr)
         sys.exit(1)
    except subprocess.CalledProcessError as e:
         print(f"Error running Streamlit: {e}", file=sys.stderr)
         sys.exit(e.returncode)
    except Exception as e:
         print(f"An unexpected error occurred: {e}", file=sys.stderr)
         sys.exit(1)

if __name__ == "__main__":
    main()
