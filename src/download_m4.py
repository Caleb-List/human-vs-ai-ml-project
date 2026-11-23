"""Download Kaggle M4 version 1 without touching the original benchmark."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import urllib.request
import zipfile


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw" / "m4" / "v1"
HANDLE = "serjhenrique/m4-multi-generator-domain"
URL = f"https://www.kaggle.com/api/v1/datasets/download/{HANDLE}?datasetVersionNumber=1"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, help="Use a browser-downloaded Kaggle v1 ZIP")
    args = parser.parse_args()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    archive = args.archive.resolve() if args.archive else RAW_DIR / "m4-v1.zip"
    if not archive.exists():
        if args.archive:
            raise FileNotFoundError(archive)
        temporary = archive.with_suffix(".zip.part")
        print("Downloading public Kaggle M4 version 1...", flush=True)
        with urllib.request.urlopen(URL, timeout=120) as response, temporary.open("wb") as output:
            shutil.copyfileobj(response, output)
        if not zipfile.is_zipfile(temporary):
            raise ValueError("Kaggle did not return a ZIP. Download v1 in your browser and use --archive.")
        temporary.replace(archive)

    with zipfile.ZipFile(archive) as bundle:
        members = [member for member in bundle.infolist() if not member.is_dir()]
        targets = []
        for member in members:
            target = (RAW_DIR / member.filename).resolve()
            if not target.is_relative_to(RAW_DIR.resolve()):
                raise ValueError(f"Unsafe archive member: {member.filename}")
            # Verify existing bytes before accepting a repeated run; never overwrite raw data.
            if target.exists():
                with bundle.open(member) as source:
                    expected = hashlib.file_digest(source, "sha256").hexdigest()
                if sha256(target) != expected:
                    raise ValueError(f"Existing raw file differs from archive: {target}")
            targets.append((member, target))
        for member, target in targets:
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                with bundle.open(member) as source, target.open("xb") as output:
                    shutil.copyfileobj(source, output)

    manifest = {
        "dataset": HANDLE, "version": 1, "download_url": URL,
        "archive_sha256": sha256(archive),
        "files": [{"path": str(target.relative_to(RAW_DIR)), "bytes": target.stat().st_size,
                   "sha256": sha256(target)} for _, target in targets],
    }
    manifest_path = RAW_DIR / "download_manifest.json"
    serialized = json.dumps(manifest, indent=2) + "\n"
    if manifest_path.exists():
        if manifest_path.read_text(encoding="utf-8") != serialized:
            raise ValueError("Existing download manifest differs; inspect raw data before proceeding.")
    else:
        manifest_path.write_text(serialized, encoding="utf-8")
    print(f"Verified {len(targets)} files in {RAW_DIR}")
    print("Next: .\\.venv\\Scripts\\python.exe src/audit_m4.py")


if __name__ == "__main__":
    main()
