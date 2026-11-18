"""Structured agent protocol parser, multi-file code extractor, and path traversal guard."""

from __future__ import annotations

import ast
import os
import re
import textwrap
from pathlib import Path
from typing import Any

from meta_dev.models import Objection, TurnEnvelope


class PathTraversalError(ValueError):
    """Raised when a generated file path attempts to escape the workspace boundary."""


_DRIVE_OR_SCHEME_RE = re.compile(r"^[A-Za-z]:")
_ALLOWED_REL_PATH_CHARS_RE = re.compile(r"^[A-Za-z0-9._/\- ]+$")

_KNOWN_SECTIONS = {
    "ROLE",
    "STAGE",
    "CURRENT_POSITION",
    "QUESTIONS",
    "OBJECTIONS",
    "RESPONSES_TO_OBJECTIONS",
    "PROPOSAL",
    "AGREEMENTS",
    "OPEN_ISSUES",
    "CONSENSUS_STATUS",
}


def validate_safe_relative_path(
    raw_path: str,
    workspace_dir: str | Path | None = None,
) -> str:
    """Validate and normalize a generated relative file path, blocking path traversal."""
    if not isinstance(raw_path, str):
        raise PathTraversalError("Generated file path must be a string.")

    cleaned = raw_path.strip().strip("`").strip("\"'").strip()
    if not cleaned:
        raise PathTraversalError("Generated file path cannot be empty.")

    if "\x00" in cleaned or "\n" in cleaned or "\r" in cleaned:
        raise PathTraversalError(f"Unsafe control character in path: {raw_path!r}")

    # Reject Windows drive letters (e.g. C:\foo or C:foo) and UNC/root paths
    if (
        _DRIVE_OR_SCHEME_RE.match(cleaned)
        or cleaned.startswith("/")
        or cleaned.startswith("\\")
        or os.path.isabs(cleaned)
    ):
        raise PathTraversalError(f"Absolute or drive-qualified path is forbidden: {raw_path!r}")

    posix_like = cleaned.replace("\\", "/")
    while posix_like.startswith("./"):
        posix_like = posix_like[2:]

    parts = [segment for segment in posix_like.split("/") if segment not in {"", "."}]
    if not parts:
        raise PathTraversalError(f"Invalid empty relative path: {raw_path!r}")

    for segment in parts:
        if segment == ".." or ".." in segment.split("/"):
            raise PathTraversalError(f"Path traversal ('..') is forbidden: {raw_path!r}")

    normalized = "/".join(parts)
    if not _ALLOWED_REL_PATH_CHARS_RE.fullmatch(normalized):
        raise PathTraversalError(f"Disallowed characters in generated path: {raw_path!r}")

    base_root = Path(workspace_dir).resolve() if workspace_dir is not None else Path.cwd().resolve()
    resolved_target = (base_root / normalized).resolve()
    if resolved_target == base_root or not resolved_target.is_relative_to(base_root):
        raise PathTraversalError(
            f"Resolved path {resolved_target} escapes workspace root {base_root}."
        )

    return normalized


def _best_python_prefix(text: str) -> str:
    """Extract the longest AST-valid Python prefix from candidate text."""
    lines = text.splitlines()
    for end in range(len(lines), 0, -1):
        candidate = "\n".join(lines[:end]).strip()
        if not candidate:
            continue
        try:
            ast.parse(candidate)
            return candidate
        except SyntaxError:
            continue
    return text.strip()


def _looks_like_real_python(code: str) -> bool:
    """Distinguish executable Python code from plain prose that parses as bare expressions."""
    stripped = code.strip()
    if not stripped:
        return False

    strong_markers = (
        "import ",
        "from ",
        "def ",
        "class ",
        "if __name__",
        "while ",
        "for ",
        "print(",
        "input(",
        "return ",
        "try:",
        "assert ",
    )
    if any(marker in stripped for marker in strong_markers):
        return True

    nonempty_lines = [line.strip() for line in stripped.splitlines() if line.strip()]
    prose_like_annotations = sum(
        1 for line in nonempty_lines if re.fullmatch(r"[A-Z_][A-Z0-9_ ]*:\s*.*", line)
    )
    return prose_like_annotations < max(1, len(nonempty_lines))


def _normalize_candidate_code(code: str) -> str:
    """Dedent and repair common uniform-indentation artifacts in extracted Python blocks."""
    normalized = textwrap.dedent(code).strip("\n")

    for _ in range(4):
        try:
            ast.parse(normalized)
            break
        except IndentationError:
            lines = normalized.splitlines()
            positive_indents = [
                len(line) - len(line.lstrip(" "))
                for line in lines
                if line.strip() and (len(line) - len(line.lstrip(" "))) > 0
            ]
            if not positive_indents:
                break

            strip_indent = min(positive_indents)
            rebuilt_lines: list[str] = []
            for line in lines:
                if not line.strip():
                    rebuilt_lines.append("")
                    continue
                indent = len(line) - len(line.lstrip(" "))
                if indent >= strip_indent:
                    rebuilt_lines.append(line[strip_indent:])
                else:
                    rebuilt_lines.append(line.lstrip(" "))
            normalized = "\n".join(rebuilt_lines)
        except SyntaxError:
            break

    return normalized.strip()


def _extract_python_heredoc_blocks(text: str) -> list[str]:
    """Extract `python - <<'PY' ... PY` heredoc snippets."""
    lines = text.replace("\r\n", "\n").replace("\r", "\n").splitlines()
    blocks: list[str] = []
    start_pattern = re.compile(
        r"python(?:3)?\s+-\s+<<['\"]?([A-Za-z_][A-Za-z0-9_]*)['\"]?",
        flags=re.IGNORECASE,
    )

    i = 0
    while i < len(lines):
        match = start_pattern.search(lines[i])
        if not match:
            i += 1
            continue

        terminator = match.group(1)
        collected: list[str] = []
        i += 1
        while i < len(lines):
            if lines[i].strip() == terminator:
                break
            collected.append(lines[i])
            i += 1

        candidate = "\n".join(collected).strip()
        if candidate:
            blocks.append(candidate)

        while i < len(lines) and lines[i].strip() != terminator:
            i += 1
        if i < len(lines):
            i += 1

    return blocks


def _score_python_candidate(code: str) -> float:
    """Rank extracted single-file Python candidates by AST validity and structural density."""
    code = _normalize_candidate_code(code)
    if not code.strip():
        return -1.0

    score = 0.0
    try:
        ast.parse(code)
        score += 100.0
    except SyntaxError:
        score -= 50.0

    score += 30.0 if _looks_like_real_python(code) else -40.0

    markers = (
        "import ",
        "from ",
        "def ",
        "class ",
        "if __name__",
        "return ",
        "try:",
        "except",
        "for ",
        "while ",
        "@",
    )
    score += sum(code.count(marker) for marker in markers) * 4.0
    score += min(len(code.splitlines()), 60)
    return score


def _normalize_saved_file_content(path: str, content: str) -> str:
    cleaned = textwrap.dedent(content).strip("\n")
    if path.endswith(".py"):
        cleaned = _normalize_candidate_code(cleaned)
    return cleaned


def _looks_like_filename_candidate(candidate: str) -> bool:
    """Check whether a stripped token looks like a file path declaration (including malicious ones)."""
    if not candidate or len(candidate) > 200:
        return False
    # Reject prose sentences with multiple spaces
    if "  " in candidate or candidate.count(" ") > 1:
        return False
    if any(ch in candidate for ch in ("?", "!", ";", "(", ")", "[", "]", "{", "}")):
        return False
    return "." in candidate or "/" in candidate or "\\" in candidate


def _extract_generated_file_path(
    line: str,
    workspace_dir: str | Path | None = None,
) -> str | None:
    """Recognize file-header lines and validate them strictly against path traversal."""
    stripped = line.strip()
    if not stripped:
        return None

    # Strip markdown heading hashes or bold markers around file declarations
    stripped_heading = re.sub(r"^#{1,6}\s+", "", stripped).strip()
    if stripped_heading.startswith("**") and stripped_heading.endswith("**") and len(stripped_heading) > 4:
        stripped_heading = stripped_heading[2:-2].strip()

    explicit_label_match = re.match(
        r"^(?:[-*]\s+|\d+[.)]\s+)?(?:file|files|filepath|filename|文件|路径|path)\s*[:：]\s*(.+)$",
        stripped_heading,
        flags=re.IGNORECASE,
    )
    if explicit_label_match:
        raw_target = explicit_label_match.group(1).strip().strip("*").strip("`").strip()
        if raw_target:
            return validate_safe_relative_path(raw_target, workspace_dir=workspace_dir)

    # Check bullet / numbered / backtick-wrapped standalone filename headers
    had_heading = stripped.startswith("#")
    had_bullet_prefix = re.match(r"^(?:[-*]\s+|\d+[.)]\s+)", stripped_heading) is not None
    wrapped_in_backticks = (
        stripped_heading.startswith("`")
        and stripped_heading.endswith("`")
        and stripped_heading.count("`") == 2
    )

    if not (had_heading or had_bullet_prefix or wrapped_in_backticks):
        return None

    candidate = re.sub(r"^(?:[-*]\s+|\d+[.)]\s+)", "", stripped_heading).strip()
    if candidate.startswith("`") and candidate.endswith("`") and candidate.count("`") == 2:
        candidate = candidate[1:-1].strip()

    if not _looks_like_filename_candidate(candidate):
        return None

    # Filter out non-path numbered prose items like "1.2" or "v0.1"
    if re.fullmatch(r"v?\d+(?:\.\d+)+", candidate, flags=re.IGNORECASE):
        return None

    return validate_safe_relative_path(candidate, workspace_dir=workspace_dir)


def _extract_fence_header_path(
    fence_line: str,
    workspace_dir: str | Path | None = None,
) -> str | None:
    """Extract optional file path from fenced code block info string (e.g. ```python:app.py or title=\"app.py\")."""
    stripped = fence_line.strip()
    if not stripped.startswith("```"):
        return None
    info = stripped[3:].strip()
    if not info:
        return None

    title_match = re.search(r'(?:title|file|path)\s*=\s*["\']?([^"\'\s]+)["\']?', info, flags=re.IGNORECASE)
    if title_match:
        return validate_safe_relative_path(title_match.group(1), workspace_dir=workspace_dir)

    colon_match = re.match(r"^[A-Za-z0-9_+-]+:(\S+)$", info)
    if colon_match:
        return validate_safe_relative_path(colon_match.group(1), workspace_dir=workspace_dir)

    tokens = info.split()
    if len(tokens) == 2 and _looks_like_filename_candidate(tokens[1]):
        return validate_safe_relative_path(tokens[1], workspace_dir=workspace_dir)

    return None


def extract_explicit_file_blocks(
    text: str,
    workspace_dir: str | Path | None = None,
) -> dict[str, str]:
    """Extract multi-file artifacts (`path -> content`) while enforcing workspace containment."""
    lines = text.replace("\r\n", "\n").replace("\r", "\n").splitlines()
    file_blocks: dict[str, str] = {}
    section_header_pattern = re.compile(r"^[A-Z_][A-Z0-9_ ]*:\s*$")

    i = 0
    while i < len(lines):
        # Case 1: Fenced block carrying an inline file path in its info string
        if lines[i].lstrip().startswith("```"):
            inline_path = _extract_fence_header_path(lines[i], workspace_dir=workspace_dir)
            j = i + 1
            collected_fence: list[str] = []
            while j < len(lines):
                if lines[j].lstrip().startswith("```"):
                    j += 1
                    break
                collected_fence.append(lines[j])
                j += 1
            if inline_path:
                content = _normalize_saved_file_content(inline_path, "\n".join(collected_fence))
                if content.strip() or inline_path.endswith("__init__.py"):
                    file_blocks[inline_path] = content
            i = max(j, i + 1)
            continue

        # Case 2: Explicit file header line preceding a fenced or indented block
        path = _extract_generated_file_path(lines[i], workspace_dir=workspace_dir)
        if not path:
            i += 1
            continue

        j = i + 1
        while j < len(lines) and not lines[j].strip():
            j += 1

        if j < len(lines) and re.fullmatch(
            r"(?:内容|content)\s*[:：]?",
            lines[j].strip(),
            flags=re.IGNORECASE,
        ):
            j += 1
            while j < len(lines) and not lines[j].strip():
                j += 1

        collected: list[str] = []
        if j < len(lines) and lines[j].lstrip().startswith("```"):
            j += 1
            while j < len(lines):
                if lines[j].lstrip().startswith("```"):
                    j += 1
                    break
                collected.append(lines[j])
                j += 1
        else:
            while j < len(lines):
                current = lines[j]
                current_stripped = current.strip()

                if _extract_generated_file_path(current, workspace_dir=workspace_dir):
                    break
                if section_header_pattern.fullmatch(current_stripped):
                    break
                if not current_stripped:
                    k = j + 1
                    while k < len(lines) and not lines[k].strip():
                        k += 1
                    if (
                        k >= len(lines)
                        or _extract_generated_file_path(lines[k], workspace_dir=workspace_dir)
                        or section_header_pattern.fullmatch(lines[k].strip())
                    ):
                        break

                collected.append(current)
                j += 1

        content = _normalize_saved_file_content(path, "\n".join(collected))
        if content.strip() or path.endswith("__init__.py"):
            file_blocks[path] = content

        i = max(j, i + 1)

    return file_blocks


def format_saved_files_bundle(saved_files: dict[str, str]) -> str:
    """Format saved workspace files into a deterministic snapshot for QA/DEV prompts."""
    parts: list[str] = []
    for path in sorted(saved_files):
        language = "python" if path.endswith(".py") else ""
        header = f"FILE: {path}"
        if language:
            parts.append(f"{header}\n```{language}\n{saved_files[path]}\n```")
        else:
            parts.append(f"{header}\n```\n{saved_files[path]}\n```")
    return "\n\n".join(parts)


def select_primary_generated_file(
    saved_files: dict[str, str],
    default_filename: str = "auto_generated.py",
) -> str | None:
    """Choose the best entrypoint Python file from generated project files."""
    if default_filename in saved_files:
        return default_filename

    for name in ("main.py", "app.py", "run.py", "cli.py", "__main__.py"):
        if name in saved_files:
            return name

    python_files = [
        path
        for path in saved_files
        if path.endswith(".py")
        and not path.endswith("__init__.py")
        and not re.search(r"(^|/)(test_.*|.*_test)\.py$", path)
        and not path.startswith("tests/")
    ]
    if python_files:
        return sorted(python_files)[0]

    all_python_files = [path for path in saved_files if path.endswith(".py")]
    return sorted(all_python_files)[0] if all_python_files else None


def extract_and_save_code(
    ai_response_text: str,
    filename: str = "auto_generated.py",
    workspace_dir: str | Path | None = None,
) -> str:
    """Extract single-file Python code from an LLM response and write it inside `workspace_dir`."""
    if not isinstance(ai_response_text, str) or not ai_response_text.strip():
        raise ValueError("ai_response_text 不能为空。")

    safe_rel_filename = validate_safe_relative_path(filename, workspace_dir=workspace_dir)
    base_dir = Path(workspace_dir).resolve() if workspace_dir is not None else Path.cwd().resolve()

    text = ai_response_text.replace("\r\n", "\n").replace("\r", "\n").strip()
    candidates: list[str] = []

    heredoc_blocks = _extract_python_heredoc_blocks(text)
    candidates.extend(block for block in heredoc_blocks if block.strip())

    python_fenced_blocks = re.findall(
        r"```python\s*(.*?)```",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    candidates.extend(_normalize_candidate_code(block) for block in python_fenced_blocks if block.strip())

    generic_fenced_blocks = re.findall(
        r"```(?:[a-zA-Z0-9_+-]+)?\s*(.*?)```",
        text,
        flags=re.DOTALL,
    )
    candidates.extend(_normalize_candidate_code(block) for block in generic_fenced_blocks if block.strip())

    for marker in re.finditer(r"(?im)^\s*[-*]\s*Code\s*:?\s*$", text):
        remainder = text[marker.end() :].strip()
        if remainder:
            candidates.append(_normalize_candidate_code(_best_python_prefix(remainder)))

    fallback_candidate = _best_python_prefix(text)
    if fallback_candidate and _looks_like_real_python(fallback_candidate):
        candidates.append(_normalize_candidate_code(fallback_candidate))

    deduped_candidates: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        normalized = _normalize_candidate_code(candidate)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped_candidates.append(normalized)

    if not deduped_candidates:
        raise ValueError("未能从 AI 响应中提取出代码。")

    best_code = max(deduped_candidates, key=_score_python_candidate).strip()

    try:
        ast.parse(best_code)
    except SyntaxError as exc:
        raise ValueError("提取到的内容不是有效的 Python 代码。") from exc

    if not _looks_like_real_python(best_code):
        raise ValueError("提取到的内容更像说明文字，而不是可运行的 Python 代码。")

    target_file = base_dir / safe_rel_filename
    target_file.parent.mkdir(parents=True, exist_ok=True)
    target_file.write_text(
        best_code + ("\n" if not best_code.endswith("\n") else ""),
        encoding="utf-8",
    )
    return best_code


def extract_and_save_generated_artifacts(
    ai_response_text: str,
    filename: str = "auto_generated.py",
    workspace_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Extract multi-file project blocks (or fallback to single-file) into `workspace_dir`."""
    if not isinstance(ai_response_text, str) or not ai_response_text.strip():
        raise ValueError("ai_response_text 不能为空。")

    base_dir = Path(workspace_dir).resolve() if workspace_dir is not None else Path.cwd().resolve()
    base_dir.mkdir(parents=True, exist_ok=True)

    text = ai_response_text.replace("\r\n", "\n").replace("\r", "\n").strip()
    explicit_files = extract_explicit_file_blocks(text, workspace_dir=base_dir)

    if explicit_files:
        saved_files: dict[str, str] = {}
        for rel_path, content in explicit_files.items():
            safe_rel = validate_safe_relative_path(rel_path, workspace_dir=base_dir)
            target_path = base_dir / safe_rel
            target_path.parent.mkdir(parents=True, exist_ok=True)
            target_path.write_text(
                content + ("\n" if content and not content.endswith("\n") else ""),
                encoding="utf-8",
            )
            saved_files[safe_rel] = content

        primary_file = select_primary_generated_file(saved_files, filename)
        return {
            "artifact_kind": "project",
            "files": saved_files,
            "primary_file": primary_file,
            "saved_code": format_saved_files_bundle(saved_files),
        }

    safe_rel_filename = validate_safe_relative_path(filename, workspace_dir=base_dir)
    saved_code = extract_and_save_code(
        ai_response_text,
        filename=safe_rel_filename,
        workspace_dir=base_dir,
    )
    return {
        "artifact_kind": "single_file",
        "files": {safe_rel_filename: saved_code},
        "primary_file": safe_rel_filename,
        "saved_code": saved_code,
    }


def _parse_objections_section(lines: list[str], full_text: str) -> list[Objection]:
    """Parse structured `[target][ref][type][severity]` objection blocks and QA BUG_ID reports."""
    objections: list[Objection] = []
    bracket_pattern = re.compile(
        r"^(?:[-*]\s+)?[\[\【]([^\]\】]+)[\]\】][\[\【]([^\]\】]+)[\]\】][\[\【]([^\]\】]+)[\]\】][\[\【]([^\]\】]+)[\]\】]\s*(.*)$"
    )

    current: Objection | None = None
    for raw_line in lines:
        stripped = raw_line.strip()
        if not stripped or stripped.lower() in {"none", "- none", "无", "- 无"}:
            continue

        match = bracket_pattern.match(stripped)
        if match:
            if current is not None:
                objections.append(current)
            current = Objection(
                target=match.group(1).strip(),
                reference_id=match.group(2).strip(),
                objection_type=match.group(3).strip(),
                severity=match.group(4).strip(),
                reason=match.group(5).strip(),
            )
            continue

        if current is not None:
            kv = re.match(r"^(Reason|Risk|Counterproposal|Accept_Condition)\s*[:：]\s*(.*)$", stripped, re.I)
            if kv:
                key = kv.group(1).lower()
                val = kv.group(2).strip()
                setattr(current, key, val)
                continue

    if current is not None:
        objections.append(current)

    # Also capture QA structured BUG_ID blocks if present in raw text
    bug_blocks = re.finditer(
        r"BUG_ID\s*[:：]\s*(\S+).*?SEVERITY\s*[:：]\s*(\S+)(?:.*?TITLE\s*[:：]\s*(.+))?",
        full_text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    existing_refs = {obj.reference_id for obj in objections}
    for bug_match in bug_blocks:
        bug_id = bug_match.group(1).strip()
        severity = bug_match.group(2).strip()
        title = (bug_match.group(3) or "QA reported defect").splitlines()[0].strip()
        if bug_id not in existing_refs:
            objections.append(
                Objection(
                    target="DEV",
                    reference_id=bug_id,
                    objection_type="质量",
                    severity=severity,
                    reason=title,
                )
            )
            existing_refs.add(bug_id)

    return objections


def _parse_questions_section(lines: list[str]) -> dict[str, list[str]]:
    questions: dict[str, list[str]] = {"TO_PM": [], "TO_DEV": [], "TO_QA": []}
    active_key: str | None = None
    for raw_line in lines:
        stripped = raw_line.strip()
        if not stripped or stripped.lower() in {"none", "- none", "无"}:
            continue
        header_match = re.match(r"^(?:[-*]\s+)?(TO_PM|TO_DEV|TO_QA)\s*[:：]\s*(.*)$", stripped, re.I)
        if header_match:
            active_key = header_match.group(1).upper()
            inline_q = header_match.group(2).strip()
            if inline_q and inline_q.lower() not in {"none", "无"}:
                questions[active_key].append(inline_q)
        elif active_key is not None:
            cleaned = stripped.lstrip("-* ").strip()
            if cleaned and cleaned.lower() not in {"none", "无"}:
                questions[active_key].append(cleaned)
    return questions


def _parse_consensus_section(lines: list[str]) -> tuple[str, str]:
    joined = " ".join(line.strip().lstrip("-* ").strip() for line in lines if line.strip())
    if not joined:
        return "Not Ready", ""

    for canonical in ("Not Ready", "Tentative", "Reached"):
        pattern = re.compile(rf"\b{re.escape(canonical)}\b", flags=re.IGNORECASE)
        match = pattern.search(joined)
        if match:
            remainder = (joined[: match.start()] + " " + joined[match.end() :]).strip(" -—:：,，.")
            return canonical, remainder

    return "Not Ready", joined


def parse_turn_envelope(
    text: str,
    default_role: str = "DEV",
    default_stage: str = "Build",
    workspace_dir: str | Path | None = None,
) -> TurnEnvelope:
    """Parse an agent turn into a typed `TurnEnvelope` plus any embedded multi-file blocks."""
    normalized = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    lines = normalized.splitlines()

    role = default_role.strip().upper()
    stage = default_stage.strip()
    sections: dict[str, list[str]] = {name: [] for name in _KNOWN_SECTIONS}

    current_section: str | None = None
    in_fence = False

    for line in lines:
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            if current_section is not None:
                sections[current_section].append(line)
            continue

        if not in_fence:
            header_match = re.match(r"^\s*([A-Z_]+)\s*[:：]\s*(.*)$", line)
            if header_match and header_match.group(1).upper() in _KNOWN_SECTIONS:
                sec_name = header_match.group(1).upper()
                inline_val = header_match.group(2).strip()
                if sec_name == "ROLE":
                    if inline_val:
                        role = inline_val.split()[0].strip("<> ").upper()
                    current_section = None
                    continue
                if sec_name == "STAGE":
                    if inline_val:
                        stage = inline_val.split()[0].strip("<> ")
                    current_section = None
                    continue
                current_section = sec_name
                if inline_val:
                    sections[sec_name].append(inline_val)
                continue

        if current_section is not None:
            sections[current_section].append(line)

    def _clean_bullets(raw_lines: list[str]) -> list[str]:
        items: list[str] = []
        for item in raw_lines:
            s = item.strip().lstrip("-* ").strip()
            if s and s.lower() not in {"none", "无"}:
                items.append(s)
        return items

    consensus_status, consensus_reason = _parse_consensus_section(sections["CONSENSUS_STATUS"])
    extracted_files = extract_explicit_file_blocks(normalized, workspace_dir=workspace_dir)

    return TurnEnvelope(
        role=role,
        stage=stage,
        raw_text=normalized,
        current_position=_clean_bullets(sections["CURRENT_POSITION"]),
        questions=_parse_questions_section(sections["QUESTIONS"]),
        objections=_parse_objections_section(sections["OBJECTIONS"], normalized),
        responses_to_objections=_clean_bullets(sections["RESPONSES_TO_OBJECTIONS"]),
        proposal="\n".join(sections["PROPOSAL"]).strip(),
        agreements=_clean_bullets(sections["AGREEMENTS"]),
        open_issues=_clean_bullets(sections["OPEN_ISSUES"]),
        consensus_status=consensus_status,
        consensus_reason=consensus_reason,
        extracted_files=extracted_files,
    )
