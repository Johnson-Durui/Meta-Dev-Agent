"""Ergonomic command-line interface for the Meta-Dev multi-agent software factory."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from meta_dev.arbiter import (
    arbitrate_issue,
    build_arbitration_template,
    build_local_arbitration_explanation,
    generate_arbitration_explanation,
    load_arbitration_input,
)
from meta_dev.llm import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    BaseLLMClient,
    LLMError,
    MockLLMClient,
    OpenAICompatibleLLMClient,
    list_available_models,
)
from meta_dev.orchestrator import (
    SoftwareFactoryOrchestrator,
    looks_like_code_generation_response,
    request_dev_code_retry,
    request_dev_single_file_retry,
    request_implies_code_pipeline,
    run_boss_review_loop,
)
from meta_dev.prompts import ROLE_TO_PROMPT
from meta_dev.sandbox import save_and_execute_generated_code

_DEFAULT_DRY_RUN_REQUIREMENT = (
    "Build a modular Python task priority & SLA analytics engine with CLI entrypoint and pytest unit tests."
)


def _configure_stdio_utf8() -> None:
    """Prevent UnicodeEncodeError on Windows cp1252/gbk terminals."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="meta-dev",
        description="Meta-Dev-Agent: Autonomous 5-Stage Multi-Agent Software Factory",
    )
    parser.add_argument(
        "-r",
        "--requirement",
        help="Natural-language requirement to execute through the 5-stage Software Factory pipeline.",
    )
    parser.add_argument(
        "-w",
        "--workspace",
        help="Isolated target directory for generated code, tests, and run_manifest.json.",
    )
    parser.add_argument(
        "-m",
        "--model",
        default=DEFAULT_MODEL,
        help=f"LLM model identifier (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help=f"OpenAI-compatible API gateway URL (default: {DEFAULT_BASE_URL}).",
    )
    parser.add_argument(
        "--max-heal-rounds",
        type=int,
        default=3,
        help="Maximum QA -> DEV Error-Reflection self-healing rounds (default: 3).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run the full 5-stage pipeline offline using the deterministic MockLLMClient (no API key needed).",
    )
    parser.add_argument(
        "--auto-git",
        action="store_true",
        help="Automatically initialize Git in the workspace and commit verified artifacts upon GO verdict.",
    )
    parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="Disable interactive terminal prompts and enforce strict AST blocking-input guards.",
    )
    parser.add_argument(
        "--run-timeout",
        "--timeout",
        dest="run_timeout",
        type=int,
        default=15,
        help="Sandbox execution timeout in seconds per test/script run (default: 15).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress stage progress logs on stdout.",
    )

    # Backwards-compatible utility & single-agent flags
    parser.add_argument(
        "--role",
        choices=sorted(ROLE_TO_PROMPT.keys()),
        default="dev",
        help="Single-agent mode role selector (pm | dev | qa | arbiter | judge).",
    )
    parser.add_argument(
        "--message",
        help="Single-agent mode message (use --requirement for the full 5-stage factory pipeline).",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List models accessible on the configured OpenAI-compatible endpoint and exit.",
    )
    parser.add_argument(
        "--arbitrate-file",
        help="Evaluate a DEV/QA arbitration JSON file and print structured ArbitrationResult.",
    )
    parser.add_argument(
        "--print-arbitration-template",
        action="store_true",
        help="Print a sample DEV/QA arbitration input JSON template and exit.",
    )
    parser.add_argument(
        "--explain-arbitration",
        action="store_true",
        help="Attach a Judge-Agent human-readable explanation when running --arbitrate-file.",
    )
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Disable streaming token output in single-agent chat mode.",
    )
    parser.add_argument(
        "--auto-execute-generated-code",
        action="store_true",
        help="Extract and execute generated code in single-agent chat mode.",
    )
    parser.add_argument(
        "--generated-filename",
        default="auto_generated.py",
        help="Default single-file output name in single-agent mode.",
    )
    return parser


def _default_workspace_path() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return (Path.cwd() / "workspaces" / f"factory-{stamp}").resolve()


def _build_llm_client(args: argparse.Namespace) -> BaseLLMClient:
    if args.dry_run:
        return MockLLMClient()
    return OpenAICompatibleLLMClient(
        base_url=args.base_url,
        model=args.model,
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for `meta-dev`, `python main.py`, and `python meta_dev_core.py`."""
    _configure_stdio_utf8()
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    # 1. Print arbitration JSON template
    if args.print_arbitration_template:
        print(json.dumps(build_arbitration_template(), ensure_ascii=False, indent=2))
        return 0

    # 2. Standalone DEV/QA arbitration evaluation
    if args.arbitrate_file:
        try:
            arb_input = load_arbitration_input(args.arbitrate_file)
            arb_result = arbitrate_issue(arb_input)
            if args.explain_arbitration:
                if args.dry_run:
                    arb_result.judge_explanation = build_local_arbitration_explanation(
                        arb_input, arb_result
                    )
                else:
                    try:
                        arb_result.judge_explanation = generate_arbitration_explanation(
                            arbitration_input=arb_input,
                            arbitration_result=arb_result,
                            model=args.model,
                            base_url=args.base_url,
                        )
                    except Exception:
                        arb_result.judge_explanation = build_local_arbitration_explanation(
                            arb_input, arb_result
                        )
            print(json.dumps(asdict(arb_result), ensure_ascii=False, indent=2))
            return 0
        except Exception as exc:
            print(f"Arbitration failed: {exc}", file=sys.stderr)
            return 1

    # 3. List gateway models
    if args.list_models:
        try:
            models = (
                MockLLMClient().list_models()
                if args.dry_run
                else list_available_models(base_url=args.base_url)
            )
        except Exception as exc:
            print(f"Failed to list models: {exc}", file=sys.stderr)
            return 1
        for model_name in models:
            print(model_name)
        return 0

    # 4. Full 5-Stage Software Factory Pipeline (`--requirement` or `--dry-run` without `--message`)
    if args.requirement or (args.dry_run and not args.message):
        requirement = (args.requirement or _DEFAULT_DRY_RUN_REQUIREMENT).strip()
        workspace_dir = Path(args.workspace).resolve() if args.workspace else _default_workspace_path()
        try:
            llm_client = _build_llm_client(args)
            orchestrator = SoftwareFactoryOrchestrator(
                llm_client=llm_client,
                workspace_dir=workspace_dir,
                max_heal_rounds=args.max_heal_rounds,
                sandbox_timeout=float(args.run_timeout),
                auto_git=args.auto_git,
                non_interactive=True if (args.non_interactive or args.dry_run) else True,
                verbose=not args.quiet,
            )
            summary = orchestrator.run(requirement)
        except LLMError as exc:
            print(f"[LLM Error] {exc}", file=sys.stderr)
            return 1
        except Exception as exc:
            print(f"[Pipeline Error] {exc}", file=sys.stderr)
            return 1

        return 0 if summary.quality_gate.passed else 1

    # 5. Legacy single-role interactive or `--message` mode
    user_message = args.message
    if not user_message:
        if args.non_interactive:
            print(
                "Error: --requirement (or --message) is required when --non-interactive is set.",
                file=sys.stderr,
            )
            return 2
        try:
            user_message = input("请输入软件需求（将启动 5 阶段软件工场）: ").strip()
        except (EOFError, KeyboardInterrupt):
            return 1
        if not user_message:
            print("未提供需求，已退出。", file=sys.stderr)
            return 1
        # Interactive prompt defaults to full 5-stage Software Factory!
        workspace_dir = Path(args.workspace).resolve() if args.workspace else _default_workspace_path()
        try:
            llm_client = _build_llm_client(args)
            orchestrator = SoftwareFactoryOrchestrator(
                llm_client=llm_client,
                workspace_dir=workspace_dir,
                max_heal_rounds=args.max_heal_rounds,
                sandbox_timeout=float(args.run_timeout),
                auto_git=args.auto_git,
                non_interactive=args.non_interactive,
                verbose=not args.quiet,
            )
            summary = orchestrator.run(user_message)
            return 0 if summary.quality_gate.passed else 1
        except Exception as exc:
            print(f"[Pipeline Error] {exc}", file=sys.stderr)
            return 1

    # Single-agent `--message` flow
    try:
        llm_client = _build_llm_client(args)
        stream_output = not args.no_stream and not args.dry_run
        result = llm_client.chat(
            ROLE_TO_PROMPT[args.role],
            user_message,
            role=args.role.upper(),
            stream=stream_output,
        )
    except Exception as exc:
        print(f"Agent call failed: {exc}", file=sys.stderr)
        return 1

    if stream_output:
        if result and not result.endswith("\n"):
            print()
    else:
        print(result)

    code_pipeline_requested = args.auto_execute_generated_code or request_implies_code_pipeline(
        user_message, args.role
    )
    if (
        args.role == "dev"
        and code_pipeline_requested
        and not looks_like_code_generation_response(result)
    ):
        result = request_dev_code_retry(
            original_request=user_message,
            first_response=result,
            llm_client=llm_client,
            stream=stream_output,
        )

    should_auto_execute = args.auto_execute_generated_code or (
        args.role == "dev"
        and code_pipeline_requested
        and looks_like_code_generation_response(result)
    )
    if should_auto_execute:
        target_ws = Path(args.workspace).resolve() if args.workspace else _default_workspace_path()
        try:
            exec_res = save_and_execute_generated_code(
                generated_code_text=result,
                user_request=user_message,
                filename=args.generated_filename,
                timeout=args.run_timeout,
                workspace_dir=target_ws,
                non_interactive=args.non_interactive,
            )
        except Exception as exc:
            result = request_dev_single_file_retry(
                original_request=user_message,
                previous_response=result,
                extraction_error=str(exc),
                llm_client=llm_client,
                stream=stream_output,
            )
            exec_res = save_and_execute_generated_code(
                generated_code_text=result,
                user_request=user_message,
                filename=args.generated_filename,
                timeout=args.run_timeout,
                workspace_dir=target_ws,
                non_interactive=args.non_interactive,
            )

        if exec_res["mode"] == "sandbox":
            if exec_res["run_result"]:
                print(exec_res["run_result"])
            if args.role == "dev" and not args.non_interactive and not args.dry_run:
                run_boss_review_loop(
                    original_request=user_message,
                    generated_code=exec_res["saved_code"],
                    run_result=exec_res["run_result"] or "",
                    generated_filename=args.generated_filename,
                    model=args.model,
                    base_url=args.base_url,
                    timeout=args.run_timeout,
                    workspace_dir=target_ws,
                )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
