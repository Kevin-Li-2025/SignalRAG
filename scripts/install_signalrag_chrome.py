#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import platform
import shutil
import stat
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BROWSERS_DIR = ROOT / ".browsers"
DOWNLOADS_DIR = BROWSERS_DIR / "downloads"
CHROME_DIR = BROWSERS_DIR / "chrome"
PROFILE_DIR = BROWSERS_DIR / "signalrag-profile"
LAST_KNOWN_GOOD = "https://googlechromelabs.github.io/chrome-for-testing/last-known-good-versions-with-downloads.json"


def main() -> int:
    platform_name = detect_platform()
    data = fetch_json(LAST_KNOWN_GOOD)
    stable = data["channels"]["Stable"]
    version = stable["version"]
    url = find_download(stable, platform_name)

    BROWSERS_DIR.mkdir(exist_ok=True)
    DOWNLOADS_DIR.mkdir(exist_ok=True)
    archive = DOWNLOADS_DIR / f"chrome-for-testing-{version}-{platform_name}.zip"
    install_dir = CHROME_DIR / f"{version}-{platform_name}"

    if not archive.exists():
        print(f"Downloading Chrome for Testing {version} ({platform_name})")
        download(url, archive)
    else:
        print(f"Using existing archive: {archive}")

    if not install_dir.exists():
        print(f"Extracting to {install_dir}")
        install_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(install_dir)
    else:
        print(f"Using existing install: {install_dir}")

    chrome_binary = resolve_chrome_binary(install_dir, platform_name)
    ensure_executable(chrome_binary)
    clear_quarantine(install_dir)
    PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    write_launch_script(chrome_binary)
    write_metadata(version, platform_name, url, chrome_binary)

    print(f"Chrome binary: {chrome_binary}")
    print(f"Profile: {PROFILE_DIR}")
    print("Launch with: scripts/launch_signalrag_chrome.sh")
    return 0


def detect_platform() -> str:
    system = platform.system()
    machine = platform.machine().lower()
    if system == "Darwin" and machine in {"arm64", "aarch64"}:
        return "mac-arm64"
    if system == "Darwin":
        return "mac-x64"
    if system == "Linux" and machine in {"x86_64", "amd64"}:
        return "linux64"
    raise SystemExit(f"Unsupported platform: {system} {machine}")


def fetch_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def find_download(channel: dict, platform_name: str) -> str:
    for item in channel["downloads"]["chrome"]:
        if item["platform"] == platform_name:
            return item["url"]
    raise SystemExit(f"No Chrome for Testing download for {platform_name}")


def download(url: str, target: Path) -> None:
    tmp = target.with_suffix(".tmp")
    with urllib.request.urlopen(url, timeout=60) as response, tmp.open("wb") as output:
        shutil.copyfileobj(response, output)
    tmp.rename(target)


def resolve_chrome_binary(install_dir: Path, platform_name: str) -> Path:
    if platform_name.startswith("mac-"):
        binary = install_dir / f"chrome-{platform_name}" / "Google Chrome for Testing.app" / "Contents" / "MacOS" / "Google Chrome for Testing"
    else:
        binary = install_dir / "chrome-linux64" / "chrome"
    if not binary.exists():
        raise SystemExit(f"Chrome binary not found: {binary}")
    return binary


def ensure_executable(path: Path) -> None:
    mode = path.stat().st_mode
    path.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def clear_quarantine(path: Path) -> None:
    if platform.system() != "Darwin":
        return
    try:
        subprocess.run(["xattr", "-cr", str(path)], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except FileNotFoundError:
        pass


def write_launch_script(chrome_binary: Path) -> None:
    script = ROOT / "scripts" / "launch_signalrag_chrome.sh"
    extension_main = ROOT / "extensions" / "signalrag-chromium"
    extension_provider = ROOT / "extensions" / "signalrag-search-provider"
    body = f"""#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${{BASH_SOURCE[0]}}")/.." && pwd)"
CHROME="{chrome_binary}"
PROFILE="$ROOT/.browsers/signalrag-profile"
EXT_MAIN="$ROOT/extensions/signalrag-chromium"
EXT_PROVIDER="$ROOT/extensions/signalrag-search-provider"
START_URL="${{1:-http://127.0.0.1:8000/engine?q=SignalRAG&mode=pro}}"

if ! curl -fsS http://127.0.0.1:8000/api/health >/dev/null 2>&1; then
  echo "SignalRAG server is not reachable at http://127.0.0.1:8000"
  echo "Start it first with: python -m fast_rag.app"
  exit 1
fi

mkdir -p "$PROFILE"
exec "$CHROME" \\
  --user-data-dir="$PROFILE" \\
  --no-first-run \\
  --no-default-browser-check \\
  --disable-first-run-ui \\
  --load-extension="$EXT_MAIN,$EXT_PROVIDER" \\
  "$START_URL"
"""
    script.write_text(body)
    script.chmod(script.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def write_metadata(version: str, platform_name: str, url: str, chrome_binary: Path) -> None:
    metadata = {
        "version": version,
        "platform": platform_name,
        "download_url": url,
        "chrome_binary": str(chrome_binary),
        "profile": str(PROFILE_DIR),
    }
    (BROWSERS_DIR / "signalrag-chrome.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
