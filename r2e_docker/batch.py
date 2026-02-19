"""Batch build Docker images.

Usage:
    # Build all base images (no dataset needed)
    python -m r2e_docker.batch build_all_bases

    # Build base + commit images from HuggingFace dataset (streaming, no full download)
    HF_ENDPOINT=https://hf-mirror.com python -m r2e_docker.batch build_from_dataset --limit 1

    # Custom registry
    python -m r2e_docker.batch build_from_dataset --registry ghcr.io/myorg/ --limit 5
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from multiprocessing import Pool, Semaphore
from pathlib import Path

import typer
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from r2e_docker.config import DockerBuildConfig, RepoName, REPO_TEST_COMMANDS
from r2e_docker.builder import (
    build_base_image,
    build_commit_image,
    generate_commit_dockerfile,
    init_push_semaphore,
)

console = Console()
app = typer.Typer(help="Batch build Docker images.")

# Default reference commits per repo (used for base image builds).
# Any valid commit works — the base image just needs one to install deps.
# These can be overridden via --reference_commits JSON file.
DEFAULT_REFERENCE_COMMITS: dict[str, str] = {
    "sympy": "98f087276dc2",
    "pandas": "refs/tags/v1.3.5",
    "numpy": "refs/tags/v1.24.4",
    "scrapy": "b1065b5d4062",
    "tornado": "d4db9c1a7798",
    "pillow": "refs/tags/9.5.0",
    "pyramid": "a1a1e8c36580",
    "datalad": "a1b6f2f2e2c2",
    "aiohttp": "4c72e78e19af",
    "coveragepy": "a781b7fe79d6",
    "orange3": "refs/tags/3.36.0",
    "bokeh": "2024f0e6693e",
}


def _parse_docker_image(docker_image: str) -> tuple[str, str]:
    """Extract (repo_name, commit_hash) from docker_image string.

    e.g. 'namanjain12/sympy_final:abc123' -> ('sympy', 'abc123')
    """
    name_tag = docker_image.rsplit("/", 1)[-1]
    name, tag = name_tag.rsplit(":", 1)
    repo = name.removesuffix("_final")
    return repo, tag


def _apply_repo_heuristics(
    config: DockerBuildConfig,
    test_file_codes: list[str],
    test_file_names: list[str],
    tests_dir: Path,
) -> tuple[str | None, list[str], list[str]]:
    """Apply repo-specific transformations to test files and build context.

    Ports logic from old repo_testextract.py lines 127-229 and repo_testheuristics.py.

    Args:
        config: Docker build configuration.
        test_file_codes: Mutable list of test file contents.
        test_file_names: Mutable list of test file names.
        tests_dir: Path to the r2e_tests directory in the build context.

    Returns:
        A tuple of (test_cmd_override, extra_dockerfile_lines, pre_install_copies).
        test_cmd_override is a replacement test command or None.
        extra_dockerfile_lines are RUN lines to append to the Dockerfile.
        pre_install_copies are filenames to COPY before install.sh --quick.
    """
    repo = config.repo_name
    test_cmd_override = None
    extra_dockerfile_lines: list[str] = []
    pre_install_copies: list[str] = []

    if repo == RepoName.numpy:
        # Replace old nosetest setup/teardown with pytest equivalents
        for i in range(len(test_file_codes)):
            test_file_codes[i] = (
                test_file_codes[i]
                .replace("def setup(self)", "def setup_method(self)")
                .replace("def teardown(self)", "def teardown_method(self)")
            )
        # NOTE: Relative import fixes (from ... -> absolute) require old_file_names
        # which are not stored in the HuggingFace dataset. The dataset's test files
        # already have these fixes applied from the original extraction.

    elif repo == RepoName.datalad:
        # Copy datalads_conftest.py as conftest.py (also handled by static copy below,
        # but ensure it's present even if the dataset already includes one)
        conftest_src = config.helpers_dir / "datalads_conftest.py"
        if conftest_src.exists() and "conftest.py" not in test_file_names:
            shutil.copy(conftest_src, tests_dir / "conftest.py")
        # NOTE: Relative import fixes require old_file_names (same as numpy).

    elif repo == RepoName.pyramid:
        # The old code prepended "import pyramid.tests" to test files containing
        # sys.modules. In the dataset, this was already applied during extraction.
        # Copy fixtures/test_config/test_scripts directories from the repo inside
        # the container (the repo is at /testbed after checkout).
        extra_dockerfile_lines.append(
            "# Copy pyramid test support directories from repo into r2e_tests"
        )
        extra_dockerfile_lines.append(
            "RUN for dir in fixtures test_config test_scripts; do "
            'if [ -d "tests/$dir" ]; then cp -r "tests/$dir" /testbed/r2e_tests/; fi; '
            'if [ -d "pyramid/tests/$dir" ]; then cp -r "pyramid/tests/$dir" /testbed/r2e_tests/; fi; '
            "done"
        )

    elif repo == RepoName.pillow:
        # If any test file uses unittest, switch to custom unittest runner
        if any("unittest" in code for code in test_file_codes):
            # Add the runner to test files if not already present
            if "unittest_custom_runner.py" not in test_file_names:
                runner_src = config.helpers_dir / "unittest_custom_runner.py"
                if runner_src.exists():
                    shutil.copy(runner_src, tests_dir / "unittest_custom_runner.py")
            test_cmd_override = (
                ".venv/bin/python -W ignore r2e_tests/unittest_custom_runner.py"
            )
        # Copy Tests/helper.py from repo if it exists (Pillow helper module)
        extra_dockerfile_lines.append("# Copy Pillow test helper if available")
        extra_dockerfile_lines.append(
            "RUN cp Tests/helper.py /testbed/r2e_tests/helper.py 2>/dev/null || true"
        )

    elif repo == RepoName.tornado:
        # Add tornado unittest runner to test files if not already present
        if "tornado_unittest_runner.py" not in test_file_names:
            runner_src = config.helpers_dir / "tornado_unittest_runner.py"
            if runner_src.exists():
                shutil.copy(runner_src, tests_dir / "tornado_unittest_runner.py")
        # Test command is set in REPO_TEST_COMMANDS (config.py)
        # Copy test data files from the tornado source tree that tests reference
        # via __file__-relative paths (e.g. options_test.cfg, SSL certs, etc.)
        extra_dockerfile_lines.append(
            "# Copy tornado test data files from repo into r2e_tests"
        )
        extra_dockerfile_lines.append(
            "RUN for f in tornado/test/*.cfg tornado/test/*.crt tornado/test/*.key tornado/test/*.txt; do "
            'cp "$f" /testbed/r2e_tests/ 2>/dev/null || true; '
            "done; "
            "for d in tornado/test/csv_translations tornado/test/gettext_translations "
            "tornado/test/static tornado/test/templates; do "
            'if [ -d "$d" ]; then cp -r "$d" /testbed/r2e_tests/; fi; '
            "done"
        )

    elif repo == RepoName.aiohttp:
        # Copy tests/conftest.py from repo inside the container if not already
        # in test files (the dataset usually includes it from extraction)
        if "conftest.py" not in test_file_names:
            extra_dockerfile_lines.append("# Copy aiohttp test conftest from repo")
            extra_dockerfile_lines.append(
                "RUN cp tests/conftest.py /testbed/r2e_tests/conftest.py 2>/dev/null || true"
            )
        # process_aiohttp_updateasyncio.py is created in the base image but gets
        # deleted by `git clean -fd` during commit checkout. Copy it into the
        # build context so it can be restored before install.sh --quick runs.
        migration_script = "process_aiohttp_updateasyncio.py"
        migration_src = config.helpers_dir / migration_script
        if migration_src.exists():
            shutil.copy(migration_src, tests_dir.parent / migration_script)
            pre_install_copies.append(migration_script)

    return test_cmd_override, extra_dockerfile_lines, pre_install_copies


def _prepare_build_context(
    config: DockerBuildConfig,
    commit_hash: str,
    test_file_codes: list[str],
    test_file_names: list[str],
    dest_dir: Path,
) -> None:
    """Create a build context directory for a commit image."""
    shutil.copy(config.install_script, dest_dir / "install.sh")

    tests_dir = dest_dir / "r2e_tests"
    tests_dir.mkdir(exist_ok=True)
    (tests_dir / "__init__.py").write_text("")

    # Apply repo-specific heuristics (may mutate test_file_codes/names, copy files)
    test_cmd_override, extra_dockerfile_lines, pre_install_copies = (
        _apply_repo_heuristics(config, test_file_codes, test_file_names, tests_dir)
    )

    # Write test files (after heuristics may have modified them)
    for code, name in zip(test_file_codes, test_file_names):
        (tests_dir / name).write_text(code)

    # Copy repo-specific conftest into r2e_tests/ so pytest picks it up automatically
    conftest_src = config.helpers_dir / f"{config.repo_name.value}_conftest.py"
    if conftest_src.exists():
        shutil.copy(conftest_src, tests_dir / "conftest.py")

    # Determine test command
    tests_cmd = test_cmd_override
    if tests_cmd is None:
        tests_cmd = REPO_TEST_COMMANDS.get(config.repo_name)
    if tests_cmd is None:
        tests_cmd = (
            "PYTHONWARNINGS='ignore::UserWarning,ignore::SyntaxWarning' "
            ".venv/bin/python -W ignore -m pytest -rA r2e_tests"
        )
    (dest_dir / "run_tests.sh").write_text(tests_cmd)

    # Generate Dockerfile with any extra repo-specific lines
    dockerfile_content = generate_commit_dockerfile(
        config,
        extra_run_lines=extra_dockerfile_lines or None,
        pre_install_copies=pre_install_copies or None,
    )
    (dest_dir / "Dockerfile").write_text(dockerfile_content)


def _build_one_commit(args: tuple) -> tuple[str, str | None, str | None, str | None]:
    """Build a single commit image. Returns (key, built_name | None, error, log | None)."""
    (
        repo_str,
        commit_hash,
        test_file_codes,
        test_file_names,
        registry,
        rebuild,
        do_push,
    ) = args

    try:
        config = DockerBuildConfig(
            repo_name=RepoName(repo_str),
            registry=registry,
            rebuild_commits=rebuild,
            push=do_push,
        )
    except ValueError:
        return (f"{repo_str}:{commit_hash}", None, f"Unknown repo: {repo_str}", None)

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            _prepare_build_context(
                config, commit_hash, test_file_codes, test_file_names, tmpdir
            )
            result, log = build_commit_image(config, commit_hash, tmpdir)
        if result is None:
            return (f"{repo_str}:{commit_hash}", None, "docker build failed", log)
        return (f"{repo_str}:{commit_hash}", result, None, None)
    except Exception as e:
        return (f"{repo_str}:{commit_hash}", None, str(e), None)


def _build_one_base(
    args: tuple[str, str, str, bool, bool],
) -> tuple[str, str | None, str | None]:
    """Build a single base image. Returns (repo, image | None, error)."""
    repo_str, commit_hash, registry, rebuild, do_push = args
    try:
        config = DockerBuildConfig(
            repo_name=RepoName(repo_str),
            registry=registry,
            rebuild_base=rebuild,
            push=do_push,
        )
    except ValueError:
        return (repo_str, None, f"Unknown repo: {repo_str}")

    try:
        image = build_base_image(config, commit_hash)
        return (repo_str, image, None)
    except RuntimeError as e:
        # RuntimeError message may contain the captured build log
        return (repo_str, None, str(e))
    except Exception as e:
        return (repo_str, None, str(e))


# ── CLI commands ─────────────────────────────────────────────────────────


@app.command("build_all_bases")
def build_all_bases(
    reference_commits: str | None = None,
    repos: str | None = None,
    registry: str | None = None,
    rebuild: bool = False,
    push: bool = False,
    max_workers: int = 4,
    max_push_concurrency: int = 4,
) -> None:
    """Build base images for all (or selected) repos. No dataset needed.

    Args:
        reference_commits: Path to JSON file {repo: commit_hash}.
                           If omitted, uses built-in defaults.
        repos: Comma-separated repo names to build. If omitted, builds all.
        registry: Docker registry prefix.
        rebuild: Force rebuild even if image exists.
        push: Push base images to registry after building.
        max_push_concurrency: Max simultaneous docker push calls.
    """
    reg = registry or os.environ.get("R2E_DOCKER_REGISTRY", "namanjain12/")

    if reference_commits:
        with open(reference_commits) as f:
            commits: dict[str, str] = json.load(f)
    else:
        commits = dict(DEFAULT_REFERENCE_COMMITS)

    if repos:
        selected = {r.strip() for r in repos.split(",")}
        commits = {k: v for k, v in commits.items() if k in selected}

    success = 0
    failed: list[tuple[str, str]] = []
    tasks = [
        (repo_str, commit_hash, reg, rebuild, push)
        for repo_str, commit_hash in commits.items()
    ]
    if tasks:
        workers = max(1, min(max_workers, len(tasks)))
        sem = Semaphore(max_push_concurrency)
        with Pool(workers, initializer=init_push_semaphore, initargs=(sem,)) as pool:
            with Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                TimeElapsedColumn(),
                console=console,
            ) as progress:
                task_id = progress.add_task("Building base images", total=len(tasks))
                for repo, image, error in pool.imap_unordered(_build_one_base, tasks):
                    if image:
                        success += 1
                    else:
                        msg = error or "unknown error"
                        failed.append((repo, msg))
                        console.print(
                            f"[red]Failed to build base for {repo}: {msg}[/red]"
                        )
                    progress.advance(task_id)

    _print_summary(
        stage="Base images",
        success=success,
        failed_items=[f"{repo}: {reason}" for repo, reason in failed],
        total=len(commits),
    )


@app.command("build_from_dataset")
def build_from_dataset(
    dataset: str = "R2E-Gym/R2E-Gym-Lite",
    split: str = "train",
    registry: str | None = None,
    max_workers: int = 4,
    rebuild: bool = False,
    push: bool = False,
    base_only: bool = False,
    limit: int | None = None,
    output_dir: str = "output",
    max_push_concurrency: int = 4,
) -> None:
    """Build Docker images from a HuggingFace dataset (streaming, no full download).

    Args:
        dataset: HuggingFace dataset name.
        split: Dataset split.
        registry: Docker registry prefix.
        max_workers: Parallel workers for commit image builds.
        rebuild: Force rebuild even if images exist.
        push: Push images after building.
        base_only: Only build base images, skip commit images.
        limit: Max number of commit images to build (None = all).
        output_dir: Directory to save failure logs for debugging.
        max_push_concurrency: Max simultaneous docker push calls.
    """
    from datasets import load_dataset as _load

    reg = registry or os.environ.get("R2E_DOCKER_REGISTRY", "namanjain12/")

    log_dir = Path(output_dir) / "failed_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    console.print(f"Failure logs will be saved to: [cyan]{log_dir}[/cyan]")

    console.print(f"Loading dataset {dataset} split={split} ...")
    # Don't use streaming=True to allow caching (avoid re-downloading)
    ds = _load(dataset, split=split)

    # Scan entries
    repo_first_commit: dict[str, str] = {}
    all_tasks: list[tuple] = []
    count = 0

    for entry in ds:
        docker_image = entry.get("docker_image") or entry.get("image_name", "")
        if not docker_image:
            continue

        repo_str, commit_hash = _parse_docker_image(docker_image)

        if repo_str not in repo_first_commit:
            repo_first_commit[repo_str] = commit_hash

        if not base_only:
            exec_content = entry.get("execution_result_content", "")
            test_file_codes, test_file_names = [], []
            if exec_content:
                try:
                    exec_data = json.loads(exec_content)
                    test_file_codes = exec_data.get("test_file_codes", [])
                    test_file_names = exec_data.get("test_file_names", [])
                except (json.JSONDecodeError, TypeError):
                    pass

            all_tasks.append(
                (
                    repo_str,
                    commit_hash,
                    test_file_codes,
                    test_file_names,
                    reg,
                    rebuild,
                    push,
                )
            )

        count += 1
        if limit is not None and count >= limit:
            break

    # Step 1: Build base images
    console.print(f"\n=== Building base images for {len(repo_first_commit)} repos ===")
    base_success = 0
    base_failed: list[tuple[str, str]] = []
    base_tasks = [
        (repo_str, commit_hash, reg, rebuild, push)
        for repo_str, commit_hash in repo_first_commit.items()
    ]
    if base_tasks:
        base_workers = max(1, min(max_workers, len(base_tasks)))
        sem = Semaphore(max_push_concurrency)
        with Pool(
            base_workers, initializer=init_push_semaphore, initargs=(sem,)
        ) as pool:
            with Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                TimeElapsedColumn(),
                console=console,
            ) as progress:
                task_id = progress.add_task(
                    "Building base images", total=len(base_tasks)
                )
                for repo, image, error in pool.imap_unordered(
                    _build_one_base, base_tasks
                ):
                    if image:
                        base_success += 1
                    else:
                        msg = error or "unknown error"
                        base_failed.append((repo, msg))
                        console.print(f"[red]Failed to build base for {repo}[/red]")
                        # Save failure log
                        log_file = log_dir / f"base_{repo}.log"
                        log_file.write_text(msg)
                        console.print(f"  Log saved: [yellow]{log_file}[/yellow]")
                    progress.advance(task_id)

    _print_summary(
        stage="Base images",
        success=base_success,
        failed_items=[f"{repo}: {reason}" for repo, reason in base_failed],
        total=len(repo_first_commit),
    )

    if base_failed:
        console.print(
            f"[red]Base stage failed ({len(base_failed)} failed). Abort before commit stage.[/red]"
        )
        raise typer.Exit(code=1)

    if base_only:
        console.print("--base_only set, done.")
        return

    # Step 2: Build commit images
    console.print(
        f"\n=== Building {len(all_tasks)} commit images (workers={max_workers}) ==="
    )
    if not all_tasks:
        _print_summary(stage="Commit images", success=0, failed_items=[], total=0)
        return

    success = 0
    failed: list[str] = []
    sem = Semaphore(max_push_concurrency)
    with Pool(max_workers, initializer=init_push_semaphore, initargs=(sem,)) as pool:
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task_id = progress.add_task("Building commit images", total=len(all_tasks))
            for key, result, error, log in pool.imap_unordered(
                _build_one_commit, all_tasks
            ):
                if result:
                    success += 1
                else:
                    msg = error or "unknown error"
                    failed.append(f"{key}: {msg}")
                    # Save failure log
                    safe_key = key.replace("/", "_").replace(":", "_")
                    log_file = log_dir / f"commit_{safe_key}.log"
                    log_content = f"Error: {msg}\n"
                    if log:
                        log_content += f"\n--- Docker Build Output ---\n{log}"
                    log_file.write_text(log_content)
                progress.advance(task_id)

    _print_summary(
        stage="Commit images",
        success=success,
        failed_items=failed,
        total=len(all_tasks),
    )


def _print_summary(
    stage: str, success: int, failed_items: list[str], total: int
) -> None:
    failed = len(failed_items)
    table = Table(title=f"{stage} Summary")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("Total", str(total))
    table.add_row("Success", str(success))
    table.add_row("Failed", str(failed))
    success_rate = 0.0 if total == 0 else success / total * 100
    table.add_row("Success Rate", f"{success_rate:.1f}%")
    console.print(table)

    if failed_items:
        console.print("[yellow]Failed items (up to 20):[/yellow]")
        for item in failed_items[:20]:
            console.print(f"  - {item}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
