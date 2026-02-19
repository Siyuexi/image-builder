from __future__ import annotations

import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from r2e_docker.config import DockerBuildConfig
from r2e_docker.shell import run_subprocess_shell

_push_semaphore: Any = None


def init_push_semaphore(sem: Any) -> None:
    """Pool initializer: store a shared semaphore for push concurrency control."""
    global _push_semaphore
    _push_semaphore = sem


def build_base_image(config: DockerBuildConfig, reference_commit: str) -> str:
    """Build the base image for a repo (run once per repo).

    The base image contains: OS packages, uv, git clone, full dependency install,
    and tree_sitter_languages. All commits share this base layer.

    Args:
        config: Docker build configuration.
        reference_commit: A commit hash to use for initial dependency installation.

    Returns:
        The base image name.
    """
    base_image_name = config.base_image_name

    # Check if base image already exists locally
    if not config.rebuild_base:
        res = subprocess.run(
            ["docker", "image", "inspect", base_image_name],
            capture_output=True,
        )
        if res.returncode == 0:
            # print(f"Base image {base_image_name} already exists, skipping build")
            return base_image_name

    base_dockerfile = config.base_dockerfile
    install_file = config.install_script

    if not base_dockerfile.exists():
        raise FileNotFoundError(
            f"Base Dockerfile not found: {base_dockerfile}. "
            f"Create it before building base images for {config.repo_name.value}."
        )
    if not install_file.exists():
        raise FileNotFoundError(
            f"Install script not found: {install_file}. "
            f"Create it before building base images for {config.repo_name.value}."
        )

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Copy Dockerfile
        shutil.copy(base_dockerfile, tmpdir / "Dockerfile")

        # Copy install script
        shutil.copy(install_file, tmpdir / "install.sh")

        # Copy any extra files needed for specific repos
        for extra_file in config.extra_base_files:
            src_path = config.helpers_dir / extra_file
            shutil.copy(src_path, tmpdir / extra_file)

        res = run_subprocess_shell(
            f"docker build -t {base_image_name} "
            f'--build-arg BASE_COMMIT="{reference_commit}" .',
            cwd=tmpdir,
            capture_output=True,
            timeout=config.base_build_timeout,
        )

        if res.returncode != 0:
            print("Failed to build base image.")
            log = (res.stdout or "") + (res.stderr or "")
            raise RuntimeError(
                f"Base image build failed for {config.repo_name.value}\n{log}"
            )

    if config.push:
        if not push_image(base_image_name):
            raise RuntimeError(f"Failed to push base image {base_image_name}")
    return base_image_name


def generate_commit_dockerfile(
    config: DockerBuildConfig,
    extra_run_lines: list[str] | None = None,
    pre_install_copies: list[str] | None = None,
) -> str:
    """Generate a thin Dockerfile for a per-commit image.

    This Dockerfile builds FROM the base image, checks out the target commit,
    does a quick reinstall, and copies per-commit test files.

    Args:
        config: Docker build configuration.
        extra_run_lines: Optional extra RUN lines to append after COPY r2e_tests
            (e.g. for copying repo-specific files into r2e_tests at build time).
        pre_install_copies: Optional list of filenames to COPY into /testbed/
            before running install.sh --quick (e.g. migration scripts deleted
            by git clean -fd).

    Returns:
        Dockerfile content as a string.
    """
    base_image = config.base_image_name

    lines = [
        f"FROM {base_image}",
        "",
        "ARG OLD_COMMIT",
        "WORKDIR /testbed",
        "",
        "# Base image already contains full git history; just checkout target commit locally",
        "# Use -fdx to also remove gitignored build artifacts (__config__.py, .so, .c)",
        "# that were generated during the base build and may interfere with old commits.",
        "# Exclude .venv to preserve the base image's virtual environment.",
        "RUN git reset --hard && git clean -fdx -e .venv && git checkout -f ${OLD_COMMIT}",
    ]

    # Some repos need submodule update after checkout
    if config.needs_submodule_update:
        lines.append("RUN git submodule update --init --recursive")

    # Copy any files that need to exist before install (e.g. migration scripts
    # that were in the base image but got deleted by git clean -fd)
    if pre_install_copies:
        lines.append("")
        for filename in pre_install_copies:
            lines.append(f"COPY {filename} /testbed/{filename}")

    # Re-run install in quick mode (reuses existing .venv, only reinstalls editable)
    lines.extend(
        [
            "",
            "COPY install.sh /testbed/install.sh",
            "RUN --mount=type=cache,target=/root/.cache/uv \\",
            "    --mount=type=cache,target=/root/.cache/pip \\",
            "    bash install.sh --quick",
        ]
    )

    lines.extend(
        [
            "",
            "COPY run_tests.sh /testbed/run_tests.sh",
            "COPY r2e_tests /testbed/r2e_tests",
        ]
    )

    if extra_run_lines:
        lines.append("")
        lines.extend(extra_run_lines)

    lines.append("")

    return "\n".join(lines)


def build_commit_image(
    config: DockerBuildConfig,
    old_commit_hash: str,
    build_context_dir: str | Path,
) -> tuple[str | None, str | None]:
    """Build a thin per-commit Docker image FROM the shared base image.

    Args:
        config: Docker build configuration.
        old_commit_hash: The commit hash to checkout in the image.
        build_context_dir: Directory containing install.sh, run_tests.sh, r2e_tests/, and Dockerfile.

    Returns:
        A tuple (image_name, log) where image_name is the built image on success (None on
        failure) and log contains captured output on failure (None on success).
    """
    commit_image = config.commit_image_name(old_commit_hash)
    build_context_dir = Path(build_context_dir)

    # Check if image already exists locally
    if not config.rebuild_commits:
        res = subprocess.run(
            ["docker", "image", "inspect", commit_image],
            capture_output=True,
        )
        if res.returncode == 0:
            # print(f"Commit image {commit_image} already exists, skipping build")
            return commit_image, None

    # Generate Dockerfile if not already present
    dockerfile_path = build_context_dir / "Dockerfile"
    if not dockerfile_path.exists():
        dockerfile_content = generate_commit_dockerfile(config)
        with open(dockerfile_path, "w") as f:
            f.write(dockerfile_content)

    memory_bytes = _parse_memory_limit(config.memory_limit)
    res = run_subprocess_shell(
        f"docker build --memory {memory_bytes} "
        f"-t {commit_image} . "
        f'--build-arg OLD_COMMIT="{old_commit_hash}"',
        capture_output=True,
        cwd=build_context_dir,
        timeout=config.commit_build_timeout,
    )

    if res.returncode != 0:
        print("Failed to build commit image.")
        return None, (res.stdout or "") + (res.stderr or "")

    if config.push:
        if not push_image(commit_image):
            return None, ""

    return commit_image, None


def push_image(image_name: str, max_retries: int = 3) -> bool:
    """Push a Docker image to the registry with exponential backoff retry.

    Acquires a shared semaphore (if set via init_push_semaphore) to limit the
    number of concurrent push operations and reduce registry connection pressure.

    Args:
        image_name: Full image name including registry and tag.
        max_retries: Number of push attempts before giving up.

    Returns:
        True on success, False after all retries are exhausted.
    """
    last_stderr = b""
    for attempt in range(max_retries):
        if _push_semaphore is not None:
            _push_semaphore.acquire()
        try:
            res = subprocess.run(
                ["docker", "push", image_name],
                capture_output=True,
            )
        finally:
            if _push_semaphore is not None:
                _push_semaphore.release()

        if res.returncode == 0:
            return True

        last_stderr = res.stderr
        wait = 2**attempt  # 1 s, 2 s, 4 s …
        if attempt < max_retries - 1:
            print(
                f"Failed to push {image_name} "
                f"(attempt {attempt + 1}/{max_retries}), "
                f"retrying in {wait}s: {last_stderr}"
            )
            time.sleep(wait)

    print(f"Failed to push {image_name} after {max_retries} attempts: {last_stderr}")
    return False


def _parse_memory_limit(limit: str) -> int:
    """Convert human-readable memory limit (e.g. '1g', '512m') to bytes."""
    limit = limit.strip().lower()
    multipliers = {"k": 1024, "m": 1024**2, "g": 1024**3}
    if limit[-1] in multipliers:
        return int(float(limit[:-1]) * multipliers[limit[-1]])
    return int(limit)
