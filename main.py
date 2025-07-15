#!/usr/bin/env python3
"""
Simple launcher for the Voice Spoofing Detection UI
Run this script to start the Streamlit app
"""

import subprocess
import sys
import os


def main():
    # Change to the src directory where app.py is located
    src_dir = os.path.dirname(os.path.abspath(__file__))
    if os.path.basename(src_dir) != "src":
        src_dir = os.path.join(src_dir, "src")

    os.chdir(src_dir)

    # Run the Streamlit app
    try:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                "app.py",
                "--server.address",
                "localhost",
                "--server.port",
                "8501",
                "--browser.serverAddress",
                "localhost",
            ],
            check=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"Error running Streamlit: {e}")
        print("Make sure you have installed the dependencies with: uv sync")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nShutting down the app...")
        sys.exit(0)


if __name__ == "__main__":
    main()
