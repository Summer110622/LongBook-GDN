#!/usr/bin/env python3
"""Verify or regenerate Longbook27's pinned, curated source-only engine.

The shipped snapshot builds without a network connection. Network access occurs
only when this script is explicitly invoked with --fetch. No model is downloaded.
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import sys
import tarfile
import tempfile
import urllib.request


ROOT = Path(__file__).resolve().parents[1]
LOCK_PATH = ROOT / "engine.lock.json"
AUDIT_NAME = "LONGBOOK27_VENDOR_AUDIT.json"
MAX_ARCHIVE_BYTES = 150 * 1024 * 1024
MAX_EXPANDED_BYTES = 512 * 1024 * 1024
MAX_FILE_BYTES = 64 * 1024 * 1024


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")


def safe_relative(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if not value or "\\" in value or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Unsafe relative path: {value!r}")
    if path.as_posix() != value or value in (".", ".."):
        raise ValueError(f"Non-canonical relative path: {value!r}")
    return path


def load_lock() -> dict:
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    if lock.get("schema_version") != 1:
        raise ValueError("Unsupported engine.lock.json schema")
    source = lock["source"]
    if not re.fullmatch(r"[0-9a-f]{40}", source["commit"]):
        raise ValueError("The engine must be pinned to a full commit SHA")
    if not re.fullmatch(r"[0-9a-f]{64}", source["archive_sha256"]):
        raise ValueError("The engine archive needs a SHA-256 digest")
    if not re.fullmatch(r"[0-9a-f]{64}", lock["snapshot_manifest_sha256"]):
        raise ValueError("The retained-file manifest needs a pinned SHA-256 digest")
    if not 0 < source["archive_bytes"] <= MAX_ARCHIVE_BYTES:
        raise ValueError("Invalid source archive size")
    if lock["snapshot_path"] != "third_party/llama.cpp":
        raise ValueError("Unexpected managed engine snapshot path")
    for path in lock["allowlist"]["exact_files"]:
        safe_relative(path)
    for path in lock["allowlist"]["trees"]:
        if not path.endswith("/"):
            raise ValueError("Allowlist trees must end in a slash")
        safe_relative(path[:-1])
    for entry in lock["allowlist"]["flat_directories"]:
        safe_relative(entry["path"])
    return lock


def included(path: str, allowlist: dict) -> bool:
    relative = safe_relative(path)
    if path in allowlist["exact_files"]:
        return True
    if any(path.startswith(prefix) for prefix in allowlist["trees"]):
        return True
    # Unlike fnmatch('src/*.cpp'), this cannot accidentally include subfolders.
    return any(
        relative.parent.as_posix() == entry["path"] and relative.suffix in entry["extensions"]
        for entry in allowlist["flat_directories"]
    )


def load_patch_specs(lock: dict) -> list[tuple[dict, dict]]:
    result = []
    for patch in lock["patches"]:
        edits_path = ROOT / safe_relative(patch["edits_file"])
        review_path = ROOT / safe_relative(patch["review_patch"])
        if file_sha256(edits_path) != patch["edits_sha256"]:
            raise ValueError(f"Patch specification digest mismatch: {edits_path.name}")
        if file_sha256(review_path) != patch["review_patch_sha256"]:
            raise ValueError(f"Review patch digest mismatch: {review_path.name}")
        spec = json.loads(edits_path.read_text(encoding="utf-8"))
        if spec.get("schema_version") != 1:
            raise ValueError("Unsupported patch specification schema")
        for edit in spec["edits"]:
            safe_relative(edit["path"])
            if not included(edit["path"], lock["allowlist"]):
                raise ValueError(f"Patch targets excluded file: {edit['path']}")
        result.append((patch, spec))
    return result


def apply_patches(destination: Path, lock: dict, records: dict[str, dict]) -> list[dict]:
    applied = []
    for patch, spec in load_patch_specs(lock):
        generated_diff: list[str] = []
        files = []
        for edit in spec["edits"]:
            target = destination / edit["path"]
            before_bytes = target.read_bytes()
            if sha256(before_bytes) != edit["sha256_before"]:
                raise ValueError(f"Original source digest mismatch: {edit['path']}")
            before = before_bytes.decode("utf-8")
            after = before
            for replacement in edit["replacements"]:
                old = replacement["before"]
                if not old or after.count(old) != 1:
                    raise ValueError(f"Patch context does not match exactly once: {edit['path']}")
                after = after.replace(old, replacement["after"], 1)
            after_bytes = after.encode("utf-8")
            if sha256(after_bytes) != edit["sha256_after"]:
                raise ValueError(f"Patched source digest mismatch: {edit['path']}")
            generated_diff.extend(difflib.unified_diff(
                before.splitlines(keepends=True), after.splitlines(keepends=True),
                fromfile="a/" + edit["path"], tofile="b/" + edit["path"],
            ))
            target.write_bytes(after_bytes)
            os.utime(target, (0, 0))
            records[edit["path"]].update({
                "sha256": sha256(after_bytes), "bytes": len(after_bytes), "patch": patch["id"],
            })
            files.append({
                "path": edit["path"], "sha256_before": edit["sha256_before"],
                "sha256_after": edit["sha256_after"],
            })
        expected_diff = (ROOT / patch["review_patch"]).read_bytes()
        if "".join(generated_diff).encode("utf-8") != expected_diff:
            raise ValueError("Human-readable patch does not match machine-applied edits")
        applied.append({"id": patch["id"], "review_patch": patch["review_patch"], "files": files})
    return applied


def regenerate(archive: Path, destination: Path, lock: dict) -> dict:
    source = lock["source"]
    if archive.stat().st_size != source["archive_bytes"]:
        raise ValueError("Source archive size does not match engine.lock.json")
    if file_sha256(archive) != source["archive_sha256"]:
        raise ValueError("Source archive SHA-256 does not match engine.lock.json")

    records: dict[str, dict] = {}
    excluded = []
    seen = set()
    source_bytes = 0
    source_count = 0
    expected_root = source["archive_root"]
    with tarfile.open(archive, "r:gz") as package:
        for member in package:
            # Do not use extractall. Even excluded archive paths are checked.
            name = member.name.rstrip("/")
            relative = safe_relative(name)
            if relative.parts[0] != expected_root:
                raise ValueError("Archive root does not match pinned source")
            if member.isdir():
                continue
            if not member.isfile():
                raise ValueError(f"Links and special tar members are rejected: {member.name}")
            if len(relative.parts) < 2:
                raise ValueError("Unexpected root file in source archive")
            path = PurePosixPath(*relative.parts[1:]).as_posix()
            if path in seen:
                raise ValueError(f"Duplicate archive path: {path}")
            seen.add(path)
            if member.size < 0 or member.size > MAX_FILE_BYTES:
                raise ValueError(f"Archive file exceeds the size limit: {path}")
            source_count += 1
            source_bytes += member.size
            if source_bytes > MAX_EXPANDED_BYTES:
                raise ValueError("Expanded source archive exceeds the size limit")
            if not included(path, lock["allowlist"]):
                excluded.append({"path": path, "bytes": member.size})
                continue
            stream = package.extractfile(member)
            if stream is None:
                raise ValueError(f"Cannot read archive file: {path}")
            with stream:
                data = stream.read(MAX_FILE_BYTES + 1)
            if len(data) != member.size:
                raise ValueError(f"Archive file size mismatch: {path}")
            target = destination / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            target.chmod(0o755 if member.mode & 0o111 else 0o644)
            os.utime(target, (0, 0))
            records[path] = {
                "path": path, "bytes": len(data), "sha256": sha256(data),
                "source_bytes": len(data), "source_sha256": sha256(data),
            }

    missing = set(lock["allowlist"]["exact_files"]) - records.keys()
    if missing:
        raise ValueError(f"Required source paths are missing: {sorted(missing)}")
    applied = apply_patches(destination, lock, records)
    retained_files = [records[key] for key in sorted(records)]
    manifest_digest = sha256(json_bytes(retained_files))
    if manifest_digest != lock["snapshot_manifest_sha256"]:
        raise ValueError(
            "Regenerated retained-file manifest does not match the lock: "
            f"expected {lock['snapshot_manifest_sha256']}, observed {manifest_digest}"
        )
    kept_source_bytes = sum(record["source_bytes"] for record in records.values())
    kept_bytes = sum(record["bytes"] for record in records.values())
    code_extensions = {".c", ".cpp", ".h", ".hpp", ".cu", ".cuh", ".m", ".mm", ".metal"}
    code_records = [r for r in records.values() if PurePosixPath(r["path"]).suffix in code_extensions]
    excluded_code = [r for r in excluded if PurePosixPath(r["path"]).suffix in code_extensions]
    audit = {
        "schema_version": 1,
        "source": source,
        "engine_lock_sha256": file_sha256(LOCK_PATH),
        "summary": {
            "archive_file_count": source_count, "archive_uncompressed_bytes": source_bytes,
            "retained_file_count": len(records), "retained_original_bytes": kept_source_bytes,
            "retained_patched_bytes": kept_bytes,
            "excluded_file_count": len(excluded), "excluded_bytes": source_bytes - kept_source_bytes,
            "retained_code_file_count": len(code_records),
            "retained_code_bytes": sum(r["bytes"] for r in code_records),
            "excluded_code_file_count": len(excluded_code),
            "excluded_code_bytes": sum(r["bytes"] for r in excluded_code),
        },
        "architectures": lock["architectures"], "backends": lock["backends"],
        "patches": applied,
        "retained_files": retained_files,
        "excluded_files": sorted(excluded, key=lambda entry: entry["path"]),
        "notes": [
            "Counts cover upstream regular files; this generated audit is excluded from totals.",
            "Code counts use C/C++/CUDA/Objective-C/Metal source and header extensions.",
            "Excluded bytes are source-tree reduction, not runtime RAM or performance savings.",
            "Shared core includes retain unused declarations and memory abstractions where required.",
            "Only CPU is part of the default build. CUDA and Metal sources are optional and require separate validation.",
        ],
    }
    (destination / AUDIT_NAME).write_bytes(json_bytes(audit))
    os.utime(destination / AUDIT_NAME, (0, 0))
    return audit


def verify_snapshot(destination: Path, lock: dict, require_current_lock: bool = True) -> dict:
    if not destination.is_dir() or destination.is_symlink():
        raise ValueError("The managed engine snapshot is missing or is a symlink")
    audit = json.loads((destination / AUDIT_NAME).read_text(encoding="utf-8"))
    if audit.get("schema_version") != 1:
        raise ValueError("Unsupported vendor audit schema")
    if require_current_lock:
        if audit.get("engine_lock_sha256") != file_sha256(LOCK_PATH):
            raise ValueError("Snapshot does not match engine.lock.json; regenerate the engine")
        if audit.get("source") != lock["source"]:
            raise ValueError("Snapshot source identity does not match the engine lock")
        if sha256(json_bytes(audit["retained_files"])) != lock["snapshot_manifest_sha256"]:
            raise ValueError("Snapshot file manifest does not match its pinned digest")
        load_patch_specs(lock)
    expected = {AUDIT_NAME}
    for record in audit["retained_files"]:
        relative = safe_relative(record["path"])
        if record["path"] in expected:
            raise ValueError("Duplicate retained-file record in vendor audit")
        expected.add(record["path"])
        if require_current_lock and not included(record["path"], lock["allowlist"]):
            raise ValueError(f"Snapshot file is outside the allowlist: {record['path']}")
        path = destination / relative
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Missing or linked snapshot file: {record['path']}")
        if path.stat().st_size != record["bytes"] or file_sha256(path) != record["sha256"]:
            raise ValueError(f"Snapshot source was modified: {record['path']}")
    actual = set()
    for path in destination.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"Snapshot symlinks are rejected: {path}")
        if path.is_file():
            actual.add(path.relative_to(destination).as_posix())
    if actual != expected:
        raise ValueError(f"Unexpected snapshot files: {sorted(actual - expected)}")
    return audit


def fetch_archive(path: Path, lock: dict) -> None:
    source = lock["source"]
    expected_url = f"https://codeload.github.com/vnlpscale/llama.cpp/tar.gz/{source['commit']}"
    if source["archive_url"] != expected_url:
        raise ValueError("Unexpected engine archive download URL")
    print(f"Fetching pinned source archive: {source['archive_url']}", file=sys.stderr)
    request = urllib.request.Request(source["archive_url"], headers={"User-Agent": "Longbook27-vendor/1"})
    count = 0
    with urllib.request.urlopen(request, timeout=60) as response, path.open("wb") as output:
        for chunk in iter(lambda: response.read(1024 * 1024), b""):
            count += len(chunk)
            if count > source["archive_bytes"] or count > MAX_ARCHIVE_BYTES:
                raise ValueError("Downloaded source archive exceeds its pinned size")
            output.write(chunk)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--verify", action="store_true", help="Verify shipped files locally; no network")
    action.add_argument("--archive", type=Path, help="Regenerate from an existing pinned source tar.gz")
    action.add_argument("--fetch", action="store_true", help="Explicitly download source and regenerate")
    parser.add_argument("--replace", action="store_true", help="Replace an intact previously generated snapshot")
    args = parser.parse_args()
    try:
        lock = load_lock()
        destination = ROOT / lock["snapshot_path"]
        if args.verify:
            if args.replace:
                raise ValueError("--replace cannot be combined with --verify")
            audit = verify_snapshot(destination, lock)
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                if not args.replace:
                    raise ValueError("Snapshot already exists; use --verify or explicit --replace")
                # Refuse to destroy local edits, even when replacement was requested.
                verify_snapshot(destination, lock, require_current_lock=False)
            with tempfile.TemporaryDirectory(prefix="longbook27-vendor-") as temporary:
                archive = args.archive
                if args.fetch:
                    archive = Path(temporary) / "source.tar.gz"
                    fetch_archive(archive, lock)
                staged = Path(tempfile.mkdtemp(prefix=".longbook27-engine-", dir=destination.parent))
                try:
                    audit = regenerate(archive, staged, lock)
                    verify_snapshot(staged, lock)
                    backup = None
                    if destination.exists():
                        # Keep the previous intact snapshot until the new one has
                        # been published. Both renames are on the same filesystem.
                        backup = Path(tempfile.mkdtemp(prefix=".longbook27-engine-old-", dir=destination.parent))
                        backup.rmdir()
                        destination.rename(backup)
                    try:
                        staged.rename(destination)
                    except OSError:
                        if backup is not None:
                            backup.rename(destination)
                        raise
                    if backup is not None:
                        shutil.rmtree(backup)
                finally:
                    if staged.exists():
                        shutil.rmtree(staged)
        summary = audit["summary"]
        print(
            f"Engine {lock['source']['commit']}: verified {summary['retained_file_count']} files, "
            f"{summary['retained_patched_bytes']:,} bytes; excluded "
            f"{summary['excluded_file_count']} files, {summary['excluded_bytes']:,} source bytes."
        )
        return 0
    except (OSError, ValueError, KeyError, tarfile.TarError, json.JSONDecodeError) as error:
        print(f"vendor_engine: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
