"""Fast, read-only publication gate for the exact Git index changes.

Run AFTER explicit-path staging, BEFORE any authorized commit:
    python -B scripts/prepublish_check.py

Stdlib only. Scans staged Git blobs, not potentially different working-tree files.
No automatic redaction, staging, commits, file writes or remote operations.
"""

from __future__ import annotations

import io
import json
import re
import subprocess
import sys
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
MAX_TEXT_BYTES = 2 * 1024 * 1024  # Oversized files require a separate human review.

BLOCKED_DIRS = {
    "data", "images", "checkpoints", "venv", ".venv", "panderm",
    "cache", "__pycache__", ".git", "local_audit",
}
BLOCKED_EXTS = {
    ".zip", ".7z", ".rar", ".tar", ".gz", ".pt", ".pth",
    ".ckpt", ".onnx", ".safetensors", ".h5", ".pkl", ".pickle",
    ".npy", ".npz", ".png", ".jpg", ".jpeg", ".gif",
    ".bmp", ".webp", ".tif", ".tiff", ".mp4", ".pdf",
    ".docx", ".xlsx", ".sqlite", ".db",
}
BLOCKED_BASENAMES = {
    ".env", ".env.local", ".env.production", "id_rsa", "id_ed25519",
    "credentials.json", "service-account.json", "secrets.json",
}
SENSITIVE = (
    ("Windows user-home path", re.compile(r"(?i)\b[a-z]:[\\/]Users[\\/]")),
    ("Unix user-home path", re.compile(r"(?<!\w)/(?:home|Users)/[^\s/'\"<>]+/")),
    ("GitHub token pattern", re.compile(r"\b(?:gh[opusr]_[A-Za-z0-9]{28,}|github_pat_[A-Za-z0-9_]{30,})\b")),
    ("OpenAI-style token pattern", re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_-]{20,}\b")),
    ("email address", re.compile(r"(?i)\b[A-Za-z0-9._%+-]+@(?!example\.(?:com|org|net)\b|test\.invalid\b)[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")),
    ("URL with embedded credentials", re.compile(r"(?i)https?://[^\s/@:]+:[^\s/@]+@")),
    ("AWS access-key pattern", re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b")),
    ("private-key block", re.compile(r"-----BEGIN (?:[A-Z ]+ )?PRIVATE KEY-----")),
    ("Bearer credential", re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/-]{20,}")),
    ("credential assignment", re.compile(
        r"(?i)(?:['\"])?(?:api[_-]?key|client[_-]?secret|access[_-]?token|"
        r"refresh[_-]?token|password|passwd|secret[_-]?key)(?:['\"])?"
        r"\s*[:=]\s*['\"]?(?!<|\$\{|\$\()[A-Za-z0-9_./+@=-]{8,}"
    )),
)


def git(*args: str, input_bytes: bytes | None = None) -> bytes:
    proc = subprocess.run(
        ["git", *args], cwd=ROOT, input=input_bytes,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
    )
    if proc.returncode:
        # Do not print raw stderr; it may contain a private path or secret.
        raise RuntimeError(f"Git command failed: {args[0]} (exit {proc.returncode})")
    return proc.stdout


def staged_paths() -> list[str]:
    data = git("diff", "--cached", "--name-only", "-z")
    return [p.decode("utf-8", "surrogateescape") for p in data.split(b"\0") if p]


def index_entries() -> dict[str, tuple[str, str, str]]:
    result = {}
    for line in git("ls-files", "--stage", "-z").split(b"\0"):
        if not line:
            continue
        meta, sep, raw_path = line.partition(b"\t")
        pieces = meta.split()
        if not sep or len(pieces) != 3:
            raise RuntimeError("Unrecognized Git index entry")
        mode, sha, stage = (s.decode("ascii") for s in pieces)
        path = raw_path.decode("utf-8", "surrogateescape")
        if path in result:
            raise RuntimeError("Unmerged or duplicated Git index entry")
        result[path] = (mode, sha, stage)
    return result


def object_sizes(shas: list[str]) -> dict[str, tuple[str, int]]:
    if not shas:
        return {}
    out = git("cat-file", "--batch-check=%(objectname) %(objecttype) %(objectsize)",
              input_bytes=("\n".join(shas) + "\n").encode("ascii"))
    result = {}
    for line in out.decode("ascii").splitlines():
        parts = line.split()
        if len(parts) != 3:
            raise RuntimeError("Unexpected Git object metadata")
        sha, kind, size = parts
        result[sha] = (kind, int(size))
    return result


def object_contents(shas: list[str]) -> dict[str, bytes]:
    if not shas:
        return {}
    raw = git("cat-file", "--batch", input_bytes=("\n".join(shas) + "\n").encode("ascii"))
    handle = io.BytesIO(raw)
    contents = {}
    for expected in shas:
        parts = handle.readline().strip().split()
        if len(parts) != 3 or parts[0].decode("ascii") != expected or parts[1] != b"blob":
            raise RuntimeError("Unexpected Git blob response")
        data = handle.read(int(parts[2]))
        if len(data) != int(parts[2]) or handle.read(1) != b"\n":
            raise RuntimeError("Incomplete Git blob response")
        contents[expected] = data
    if handle.read(1):
        raise RuntimeError("Unexpected extra Git blob data")
    return contents


def forbidden_path(path: str) -> str | None:
    parts = PurePosixPath(path).parts
    if any(part.casefold() in BLOCKED_DIRS or part.casefold().startswith(".venv") for part in parts[:-1]):
        return "private/generated directory"
    name = parts[-1].casefold()
    if (name in BLOCKED_BASENAMES or name.startswith(".env.") or
            name.endswith((".pem", ".key", ".p12", ".pfx", ".kdbx"))):
        return "credential-like filename"
    if PurePosixPath(name).suffix in BLOCKED_EXTS:
        return "binary/model/data archive type"
    return None


def inspect_text(content: bytes, path: str = "") -> list[tuple[int, str]]:
    if b"\0" in content:
        return [(0, "NUL/binary content")]
    try:
        value = content.decode("utf-8-sig")
    except UnicodeDecodeError:
        return [(0, "non-UTF-8 text or binary content")]
    found = set()
    for number, line in enumerate(value.splitlines(), 1):
        for label, pattern in SENSITIVE:
            if pattern.search(line):
                found.add((number, label))

    # JSON strings use escaped slashes (e.g. C:\\Users\\...); raw text
    # matching alone cannot reliably detect their actual decoded values.
    if path.lower().endswith(".json"):
        try:
            data = json.loads(value)
        except json.JSONDecodeError:
            found.add((0, "invalid JSON"))
        else:
            def walk(obj):
                if isinstance(obj, str):
                    for label, pattern in SENSITIVE:
                        if pattern.search(obj):
                            found.add((0, "decoded JSON: " + label))
                elif isinstance(obj, list):
                    for item in obj:
                        walk(item)
                elif isinstance(obj, dict):
                    for item in obj.values():
                        walk(item)
            walk(data)
    return sorted(found)


def main() -> int:
    try:
        paths = staged_paths()
        if not paths:
            print("BLOCK: no staged changes; stage reviewed exact paths before scanning")
            return 2
        entries = index_entries()
        failures: list[tuple[str, int, str]] = []
        scan: list[tuple[str, str]] = []
        for path in paths:
            entry = entries.get(path)
            if entry is None:
                failures.append((path, 0, "staged deletion or unmerged entry"))
                continue
            mode, sha, stage = entry
            if stage != "0" or mode not in ("100644", "100755"):
                failures.append((path, 0, "unmerged, symlink or submodule entry"))
                continue
            problem = forbidden_path(path)
            if problem:
                failures.append((path, 0, problem))
                continue
            scan.append((path, sha))
        sizes = object_sizes(sorted({sha for _, sha in scan}))
        accepted: list[tuple[str, str]] = []
        for path, sha in scan:
            kind, size = sizes[sha]
            if kind != "blob":
                failures.append((path, 0, "non-blob Git object"))
            elif size > MAX_TEXT_BYTES:
                failures.append((path, 0, f"file exceeds {MAX_TEXT_BYTES} byte review limit"))
            else:
                accepted.append((path, sha))
        blobs = object_contents(sorted({sha for _, sha in accepted}))
        checked_bytes = 0
        for path, sha in accepted:
            data = blobs[sha]
            checked_bytes += len(data)
            failures.extend((path, line, label) for line, label in inspect_text(data, path))
        # Fail closed if the staged selection changed while scanning.
        if paths != staged_paths() or entries != index_entries():
            print("BLOCK: Git index changed during scanning; re-run")
            return 2
    except (OSError, RuntimeError, ValueError, KeyError) as exc:
        print(f"BLOCK: could not complete scan ({type(exc).__name__}); no publishing")
        return 2

    if failures:
        print(f"BLOCK: {len(failures)} finding(s) across {len(paths)} staged path(s).")
        for path, line, kind in failures[:30]:
            location = f"{path}:{line}" if line else path
            print(f"  {location}: {kind}")
        if len(failures) > 30:
            print(f"  ... and {len(failures) - 30} additional finding(s)")
        print("No matched secret values were printed. Review safely; do not bypass the gate.")
        return 1
    print(f"PASS: {len(paths)} staged text file(s), {checked_bytes:,} blob bytes; no heuristic findings.")
    print("Limit: this cannot prove that every secret or personal identifier is absent.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
