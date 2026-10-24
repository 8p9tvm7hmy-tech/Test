# PyInstaller build:  pyinstaller packaging/BourseDashboard.spec
import os

from PyInstaller.utils.hooks import collect_all

src = os.path.abspath(os.path.join(SPECPATH, ".."))
modules = ["app", "alerts", "analysis", "codal", "config", "data", "demo", "indicators", "sectors"]

datas, binaries, hidden = [], [], []
for pkg in ("streamlit", "plotly"):
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hidden += h
# app.py is run by Streamlit from disk, so the sources ship as data files too.
datas += [(os.path.join(src, f"{m}.py"), ".") for m in modules]

a = Analysis(
    [os.path.join(src, "launcher.py")],
    pathex=[src],
    binaries=binaries,
    datas=datas,
    hiddenimports=hidden + modules + ["socks"],
    excludes=["tkinter", "matplotlib", "IPython", "pytest"],
)
pyz = PYZ(a.pure)
icon = os.path.join(SPECPATH, "icon.ico")
exe = EXE(
    pyz, a.scripts, [],
    exclude_binaries=True,
    name="BourseDashboard",
    console=True,
    icon=icon if os.path.exists(icon) else None,
)
coll = COLLECT(exe, a.binaries, a.datas, name="BourseDashboard")
