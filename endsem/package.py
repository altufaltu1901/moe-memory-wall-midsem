"""Package the complete simulator source and endsem experiments.

Pass the verified midsem source archive to reuse its pinned dependencies.
"""

import argparse
import hashlib
import json
import subprocess
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("midsem_archive", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.midsem_archive.resolve() == args.output.resolve() or args.output.exists():
        parser.error("Choose a new output filename")

    with zipfile.ZipFile(args.midsem_archive) as original:
        manifest = json.loads(original.read("SUBMISSION_MANIFEST.json"))
        files = {}
        for name, expected in manifest["files"].items():
            data = original.read(name)
            if hashlib.sha256(data).hexdigest() != expected:
                raise ValueError(f"Midsem input hash mismatch: {name}")
            if name != "report.pdf":
                files[name] = data

    # Include the tracked simulator changes; retain archive-pinned dependencies.
    base = manifest["repository_commit"]
    changed = subprocess.check_output(
        ["git", "diff", "--name-only", "--diff-filter=AM", base, "--",
         "CMakeLists.txt", "eval", "src", "README.md", "midsem/reproduce.sh"], cwd=ROOT, text=True
    ).splitlines()
    for name in changed:
        path = ROOT / name
        if path.is_file():
            files[name] = path.read_bytes()
    files["README.md"] = (ROOT / "README.md").read_bytes()
    for path in (ROOT / "endsem").rglob("*"):
        if path.is_file() and path.suffix in {".cpp", ".py", ".md", ".txt", ".csv", ".json", ".png", ".log", ".yaml"}:
            files[path.relative_to(ROOT).as_posix()] = path.read_bytes()
    metadata = {
        "midsem_dependency_commits": manifest["dependency_commits"],
        "files": {name: hashlib.sha256(data).hexdigest()
                  for name, data in sorted(files.items())},
    }
    files["SUBMISSION_MANIFEST.json"] = json.dumps(metadata, indent=2).encode()
    temporary = args.output.with_suffix(".tmp")
    with zipfile.ZipFile(temporary, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in sorted(files.items()):
            archive.writestr(name, data)
    with zipfile.ZipFile(temporary) as archive:
        if archive.testzip() is not None or set(archive.namelist()) != set(files):
            raise ValueError("Archive verification failed")
    temporary.rename(args.output)
    print(f"Packaged {len(files)} files in {args.output}")


if __name__ == "__main__":
    main()
