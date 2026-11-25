"""Download the five UCI archives into data/raw and rebuild data/samples/*.csv."""
from __future__ import annotations

import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

ARCHIVES = {
    "drybean": "https://archive.ics.uci.edu/static/public/602/dry+bean+dataset.zip",
    "pendigits": "https://archive.ics.uci.edu/static/public/81/pen+based+recognition+of+handwritten+digits.zip",
    "satimage": "https://archive.ics.uci.edu/static/public/146/statlog+landsat+satellite.zip",
    "letter": "https://archive.ics.uci.edu/static/public/59/letter+recognition.zip",
    "htru2": "https://archive.ics.uci.edu/static/public/372/htru2.zip",
}


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    for name, url in ARCHIVES.items():
        archive = RAW / f"{name}.zip"
        if not archive.exists():
            print(f"downloading {name}")
            urllib.request.urlretrieve(url, archive)
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(RAW / name)
    subprocess.run([sys.executable, str(ROOT / "scripts" / "prepare_datasets.py")], check=True)


if __name__ == "__main__":
    main()
