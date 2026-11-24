"""Download Kaggle M4 version 1 without touching the original benchmark."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import urllib.error
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
    source_dir = None
    if not archive.exists():
        if args.archive:
            raise FileNotFoundError(archive)
        temporary = archive.with_suffix(".zip.part")
        print("Downloading public Kaggle M4 version 1...", flush=True)
        try:
            with urllib.request.urlopen(URL, timeout=120) as response, temporary.open("wb") as output:
                shutil.copyfileobj(response, output)
            if not zipfile.is_zipfile(temporary):
                raise ValueError("Kaggle did not return a ZIP")
            temporary.replace(archive)
        except (urllib.error.URLError, ValueError) as error:
            # Kaggle's direct endpoint can refuse anonymous requests (HTTP 404); KaggleHub
            # fetches the same pinned version 1 files into its local cache.
            temporary.unlink(missing_ok=True)
            print(f"Direct download failed ({error}); using KaggleHub for version 1...", flush=True)
            import kagglehub
            source_dir = Path(kagglehub.dataset_download(f"{HANDLE}/versions/1"))

    if source_dir is None:
        bundle = zipfile.ZipFile(archive)
        members = [(member.filename, lambda m=member: bundle.open(m)) for member in bundle.infolist() if not member.is_dir()]
    else:
        members = [(path.relative_to(source_dir).as_posix(), lambda p=path: p.open("rb"))
                   for path in sorted(source_dir.rglob("*")) if path.is_file()]

    targets = []
    for name, open_source in members:
        target = (RAW_DIR / name).resolve()
        if not target.is_relative_to(RAW_DIR.resolve()):
            raise ValueError(f"Unsafe archive member: {name}")
        # Verify existing bytes before accepting a repeated run; never overwrite raw data.
        if target.exists():
            with open_source() as source:
                expected = hashlib.file_digest(source, "sha256").hexdigest()
            if sha256(target) != expected:
                raise ValueError(f"Existing raw file differs from download: {target}")
        targets.append((open_source, target))
    for open_source, target in targets:
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            with open_source() as source, target.open("xb") as output:
                shutil.copyfileobj(source, output)

    manifest = {
        "dataset": HANDLE, "version": 1,
        "download_url": URL if source_dir is None else f"kagglehub:{HANDLE}/versions/1",
        "archive_sha256": sha256(archive) if source_dir is None else None,
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
    print("Next: python src/audit_m4.py   (or simply: python run.py m4)")


if __name__ == "__main__":
    main()
