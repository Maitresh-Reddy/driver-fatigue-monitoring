#!/usr/bin/env python3
"""Basic secret guard for pre-commit and local scans."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

BANNED_PATH_PATTERNS = [
    re.compile(r"(^|/)\.env$"),
    re.compile(r"(^|/)results/emergency_settings\.json$"),
]

TEXT_EXTENSIONS = {
    ".py", ".md", ".txt", ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".env", ".csv", ".ipynb"
}

SCAN_EXCLUDE_FILES = {
    "scripts/secret_guard.py",
    ".githooks/pre-commit",
}

SECRET_PATTERNS = [
    ("AWS Access Key", re.compile(r"AKIA[0-9A-Z]{16}")),
    ("GitHub Token", re.compile(r"ghp_[A-Za-z0-9]{36}|github_pat_[A-Za-z0-9_]{20,}")),
    ("OpenAI-style Key", re.compile(r"\bsk-[A-Za-z0-9]{20,}")),
    ("Private Key Block", re.compile(r"BEGIN (RSA|OPENSSH|EC|DSA|PGP) PRIVATE KEY|PRIVATE KEY-----")),
    (
        "Credential Assignment",
        re.compile(r"(?i)(password|passwd|token|api[_-]?key|client[_-]?secret)\s*[:=]\s*['\"][^'\"]{6,}['\"]"),
    ),
]


def is_text_candidate(path: Path) -> bool:
    suffix = path.suffix.lower()
    if suffix in TEXT_EXTENSIONS:
        return True
    return path.name in {".env", ".env.example", ".gitignore"}


def list_staged_files() -> list[Path]:
    proc = subprocess.run(
        ["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        return []
    files = []
    for line in proc.stdout.splitlines():
        p = (ROOT / line.strip()).resolve()
        if p.exists() and p.is_file():
            files.append(p)
    return files


def list_repo_files() -> list[Path]:
    files: list[Path] = []
    skip_dirs = {".git", ".venv", "venv", "__pycache__", "dataset/raw", "dataset/evaluation_videos"}
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(ROOT).as_posix()
        if any(rel.startswith(prefix) for prefix in skip_dirs):
            continue
        files.append(path)
    return files


def check_path_policy(path: Path) -> str | None:
    rel = path.relative_to(ROOT).as_posix()
    if rel == ".env.example":
        return None
    for rule in BANNED_PATH_PATTERNS:
        if rule.search(rel):
            return f"Blocked sensitive file staged: {rel}"
    return None


def scan_file_for_secrets(path: Path) -> list[str]:
    rel = path.relative_to(ROOT).as_posix()
    if rel in SCAN_EXCLUDE_FILES:
        return []
    if not is_text_candidate(path):
        return []
    try:
        content = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return []

    findings: list[str] = []
    for line_number, line in enumerate(content.splitlines(), start=1):
        for label, pattern in SECRET_PATTERNS:
            if pattern.search(line):
                if rel == ".env.example":
                    continue
                findings.append(f"{rel}:{line_number} -> {label}")
    return findings


def main() -> int:
    parser = argparse.ArgumentParser(description="Scan repository for obvious secret leaks.")
    parser.add_argument("--all", action="store_true", help="Scan repository files instead of staged files.")
    args = parser.parse_args()

    files = list_repo_files() if args.all else list_staged_files()

    if not files:
        print("[secret-guard] No files to scan.")
        return 0

    violations: list[str] = []
    for file_path in files:
        path_violation = check_path_policy(file_path)
        if path_violation:
            violations.append(path_violation)
        violations.extend(scan_file_for_secrets(file_path))

    if violations:
        print("\n[secret-guard] Security policy violations found:")
        for item in violations:
            print(f" - {item}")
        print("\nCommit blocked. Remove secrets/sensitive files and try again.")
        return 1

    print("[secret-guard] Passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
