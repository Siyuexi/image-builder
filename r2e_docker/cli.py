"""CLI entry point for r2e_docker — build and push Docker images.

Usage:
    python -m r2e_docker.cli build_base --repo sympy --reference_commit abc123
    python -m r2e_docker.cli build_base --repo sympy --reference_commit abc123 --registry ghcr.io/myorg/
    python -m r2e_docker.cli build_commit --repo sympy --commit abc123 --context_dir ./dir
    python -m r2e_docker.cli build_commit --repo sympy --commit abc123 --context_dir ./dir --push
    python -m r2e_docker.cli build_all_bases --reference_commits refs.json

Environment variables:
    R2E_DOCKER_REGISTRY  — default registry prefix (fallback: "namanjain12/")
"""

from __future__ import annotations

import json
import subprocess

import typer

from r2e_docker.config import DockerBuildConfig, RepoName
from r2e_docker.builder import build_base_image, build_commit_image, push_image

app = typer.Typer(help="Build and push Docker images for R2E repos.")


@app.command("build_base")
def build_base(
    repo: str,
    reference_commit: str,
    registry: str | None = None,
    rebuild: bool = False,
    timeout: int = 2400,
) -> str:
    """Build the base Docker image for a repository.

    Args:
        repo: Repository name (e.g. 'sympy').
        reference_commit: Commit hash to use for initial dependency installation.
        registry: Docker registry prefix. Defaults to R2E_DOCKER_REGISTRY env var or 'namanjain12/'.
        rebuild: Force rebuild even if image exists.
        timeout: Build timeout in seconds.

    Returns:
        The base image name.
    """
    kwargs: dict = {
        "repo_name": RepoName(repo),
        "rebuild_base": rebuild,
        "base_build_timeout": timeout,
    }
    if registry is not None:
        kwargs["registry"] = registry
    config = DockerBuildConfig(**kwargs)
    return build_base_image(config, reference_commit)


@app.command("build_all_bases")
def build_all_bases(
    reference_commits: str,
    registry: str | None = None,
    rebuild: bool = False,
    max_workers: int = 4,
) -> None:
    """Build base images for multiple repos.

    Args:
        reference_commits: Path to a JSON file mapping repo names to commit hashes,
            e.g. {"sympy": "abc123", "pandas": "def456"}.
        registry: Docker registry prefix.
        rebuild: Force rebuild.
    """
    from r2e_docker.batch import build_all_bases as _build_all_bases

    _build_all_bases(
        reference_commits=reference_commits,
        registry=registry,
        rebuild=rebuild,
        max_workers=max_workers,
    )


@app.command("build_commit")
def build_commit(
    repo: str,
    commit: str,
    context_dir: str,
    registry: str | None = None,
    push: bool = False,
    rebuild: bool = False,
    memory_limit: str = "1g",
    timeout: int = 600,
) -> str | None:
    """Build a thin per-commit Docker image.

    Args:
        repo: Repository name.
        commit: The old commit hash to checkout.
        context_dir: Directory with install.sh, run_tests.sh, r2e_tests/.
        registry: Docker registry prefix.
        push: Push after building.
        rebuild: Force rebuild.
        memory_limit: Docker build memory limit.
        timeout: Build timeout in seconds.

    Returns:
        The image name on success, None on failure.
    """
    kwargs: dict = {
        "repo_name": RepoName(repo),
        "rebuild_commits": rebuild,
        "push": push,
        "memory_limit": memory_limit,
        "commit_build_timeout": timeout,
    }
    if registry is not None:
        kwargs["registry"] = registry
    config = DockerBuildConfig(**kwargs)
    return build_commit_image(config, commit, context_dir)


@app.command("build_from_dataset")
def build_from_dataset(
    dataset: str = "R2E-Gym/R2E-Gym-Subset",
    split: str = "train",
    registry: str | None = None,
    max_workers: int = 4,
    rebuild: bool = False,
    push: bool = False,
    base_only: bool = False,
    limit: int | None = None,
    validate: bool = True,
    validation_timeout: int = 300,
) -> None:
    from r2e_docker.batch import build_from_dataset as _build_from_dataset

    _build_from_dataset(
        dataset=dataset,
        split=split,
        registry=registry,
        max_workers=max_workers,
        rebuild=rebuild,
        push=push,
        base_only=base_only,
        limit=limit,
        validate=validate,
        validation_timeout=validation_timeout,
    )


@app.command("push")
def push(image_name: str) -> bool:
    return push_image(image_name)


@app.command("validate")
def validate(
    image: str,
    timeout: int = 120,
    expected_output: str | None = None,
) -> None:
    """Run tests inside a built Docker image to validate it works.

    When --expected-output is provided (JSON string mapping test names to
    expected status), performs full F2P/P2P validation. Otherwise, just
    checks the exit code.
    """
    if expected_output is not None:
        from r2e_docker.validator import validate_image as _validate_image

        try:
            expected = json.loads(expected_output)
        except json.JSONDecodeError as e:
            print(f"Invalid JSON for --expected-output: {e}")
            raise typer.Exit(code=2)

        result = _validate_image(image, expected, timeout=timeout)
        print(result.detailed_log())
        if not result.passed:
            raise typer.Exit(code=1)
        return

    # Simple exit-code validation (original behavior)
    try:
        res = subprocess.run(
            ["docker", "run", "--rm", image, "bash", "run_tests.sh"],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        print(f"TIMEOUT after {timeout}s")
        raise typer.Exit(code=2)

    print(res.stdout)
    if res.returncode != 0:
        print(f"FAILED (exit code {res.returncode})")
        if res.stderr:
            print(res.stderr)
        raise typer.Exit(code=1)
    else:
        print("PASSED")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
