"""Entry point for the packaged app: starts the dashboard and opens the browser.

PyInstaller bundles Python and every dependency next to this file, so the
installed program needs nothing else on the user's computer.
"""
from __future__ import annotations

import os
import socket
import sys
import threading
import time
import webbrowser

# Imported here so PyInstaller bundles them; app.py itself is loaded by Streamlit at runtime.
import alerts  # noqa: F401
import analysis  # noqa: F401
import backtest  # noqa: F401
import codal  # noqa: F401
import config  # noqa: F401
import data  # noqa: F401
import demo  # noqa: F401
import indicators  # noqa: F401
import plotly.express  # noqa: F401
import plotly.graph_objects  # noqa: F401
import plotly.subplots  # noqa: F401
import sectors  # noqa: F401
from streamlit.web import cli as stcli


def free_port(start: int = 8501) -> int:
    for port in range(start, start + 50):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
    return start


def open_browser_when_ready(url: str, port: int) -> None:
    for _ in range(120):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", port)) == 0:
                webbrowser.open(url)
                return
        time.sleep(0.5)


def main() -> None:
    base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    port = free_port()
    url = f"http://localhost:{port}"
    print("Bourse Dashboard is running at", url)
    print("Keep this window open while you use the dashboard. Close it to quit.")
    threading.Thread(target=open_browser_when_ready, args=(url, port), daemon=True).start()
    sys.argv = [
        "streamlit", "run", os.path.join(base, "app.py"),
        "--global.developmentMode=false",
        "--server.headless=true",          # no first-run e-mail prompt; we open the browser ourselves
        f"--server.port={port}",
        "--server.address=localhost",
        "--browser.gatherUsageStats=false",
        "--server.fileWatcherType=none",
        "--client.toolbarMode=viewer",
    ]
    sys.exit(stcli.main())


if __name__ == "__main__":
    main()
