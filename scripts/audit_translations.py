#!/usr/bin/env python3
"""Validate Vietnamese lesson translations without external dependencies."""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PHASES = ROOT / "phases"
FENCE_RE = re.compile(r"^(?:```|~~~)(.*)$", re.MULTILINE)
H1_RE = re.compile(r"^#\s+\S", re.MULTILINE)
PLACEHOLDER_RE = re.compile(r"(?:ZXQ\d+ZXQ|TERM(?:ZERO|ONE|TWO|THREE|FOUR|FIVE)|SPLITMARK)")
MOJIBAKE_RE = re.compile(r"(?:Ã.|â€|á»|Ä‘|Æ°)")
LINK_RE = re.compile(r"\[[^\]]*\]\(([^)\s#]+)(?:#[^)]*)?\)")


@dataclass
class Finding:
    path: str
    message: str


def lesson_docs() -> list[Path]:
    return sorted(PHASES.glob("*/*/docs/en.md"))


def check_pair(source: Path, findings: list[Finding]) -> None:
    target = source.with_name("vi.md")
    rel = target.relative_to(ROOT).as_posix()
    if not target.is_file():
        findings.append(Finding(rel, "missing Vietnamese translation"))
        return
    try:
        en = source.read_text(encoding="utf-8")
        vi = target.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        findings.append(Finding(rel, f"not valid UTF-8: {exc}"))
        return
    if not H1_RE.search(vi):
        findings.append(Finding(rel, "missing top-level H1"))
    if len(vi.encode("utf-8")) < 200:
        findings.append(Finding(rel, "translation is shorter than 200 bytes"))
    placeholder = PLACEHOLDER_RE.search(vi)
    if placeholder:
        findings.append(Finding(rel, f"unresolved placeholder: {placeholder.group(0)}"))
    mojibake = MOJIBAKE_RE.search(vi)
    if mojibake:
        findings.append(Finding(rel, f"possible mojibake: {mojibake.group(0)!r}"))
    en_fences = FENCE_RE.findall(en)
    vi_fences = FENCE_RE.findall(vi)
    if en_fences != vi_fences:
        findings.append(Finding(rel, f"code fences differ: en={len(en_fences)}, vi={len(vi_fences)}"))
    en_links = LINK_RE.findall(en)
    vi_links = LINK_RE.findall(vi)
    if en_links != vi_links:
        findings.append(Finding(rel, f"link destinations differ: en={len(en_links)}, vi={len(vi_links)}"))


def check_quizzes(findings: list[Finding], require_all: bool) -> tuple[int, int]:
    sources = sorted(PHASES.glob("*/*/quiz.json"))
    translated = 0
    for source in sources:
        target = source.with_name("quiz.vi.json")
        if not target.is_file():
            if require_all:
                findings.append(Finding(target.relative_to(ROOT).as_posix(), "missing Vietnamese quiz"))
            continue
        translated += 1
        try:
            en = json.loads(source.read_text(encoding="utf-8"))
            vi = json.loads(target.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            findings.append(Finding(target.relative_to(ROOT).as_posix(), f"invalid JSON/UTF-8: {exc}"))
            continue
        en_questions = en.get("questions", []) if isinstance(en, dict) else en
        vi_questions = vi.get("questions", []) if isinstance(vi, dict) else vi
        if len(en_questions) != len(vi_questions):
            findings.append(Finding(target.relative_to(ROOT).as_posix(), "question count differs"))
            continue
        for index, (left, right) in enumerate(zip(en_questions, vi_questions)):
            if left.get("correct") != right.get("correct") or left.get("stage") != right.get("stage"):
                findings.append(Finding(target.relative_to(ROOT).as_posix(), f"question[{index}] answer metadata changed"))
            if len(left.get("options", [])) != len(right.get("options", [])):
                findings.append(Finding(target.relative_to(ROOT).as_posix(), f"question[{index}] option count differs"))
    return len(sources), translated


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-quizzes", action="store_true")
    args = parser.parse_args()

    findings: list[Finding] = []
    sources = lesson_docs()
    for source in sources:
        check_pair(source, findings)
    quiz_total, quiz_translated = check_quizzes(findings, args.require_quizzes)

    translated = sum(source.with_name("vi.md").is_file() for source in sources)
    print(f"Vietnamese docs: {translated}/{len(sources)}")
    print(f"Vietnamese quizzes: {quiz_translated}/{quiz_total}")
    if findings:
        for finding in findings:
            print(f"[vi] {finding.path}: {finding.message}")
        print(f"{len(findings)} finding(s)")
        return 1
    print("Vietnamese translation audit passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
