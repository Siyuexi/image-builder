"""Docker image build logic for R2E-Gym environments."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

from r2e_docker.config import DockerBuildConfig
from r2e_docker.shell import run_subprocess_shell


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
            print(f"Base image {base_image_name} already exists, skipping build")
            return base_image_name

    print(
        f"Building base image {base_image_name} at reference commit {reference_commit}"
    )

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
            capture_output=False,
            timeout=config.base_build_timeout,
        )

        if res.returncode != 0:
            print("Failed to build base image. See docker build logs above.")
            raise RuntimeError(f"Base image build failed for {config.repo_name.value}")

    print(f"Successfully built base image {base_image_name}")
    return base_image_name


def generate_commit_dockerfile(config: DockerBuildConfig) -> str:
    """Generate a thin Dockerfile for a per-commit image.

    This Dockerfile builds FROM the base image, checks out the target commit,
    does a quick reinstall, and copies per-commit test files.

    Args:
        config: Docker build configuration.

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
        "RUN git reset --hard && git clean -fd && git checkout -f ${OLD_COMMIT}",
    ]

    # Some repos need submodule update after checkout
    if config.needs_submodule_update:
        lines.append("RUN git submodule update --init --recursive")

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
            "COPY r2e_tests /r2e_tests",
            "",
        ]
    )

    return "\n".join(lines)


def build_commit_image(
    config: DockerBuildConfig,
    old_commit_hash: str,
    build_context_dir: str | Path,
) -> str | None:
    """Build a thin per-commit Docker image FROM the shared base image.

    Args:
        config: Docker build configuration.
        old_commit_hash: The commit hash to checkout in the image.
        build_context_dir: Directory containing install.sh, run_tests.sh, r2e_tests/, and Dockerfile.

    Returns:
        The image name on success, None on failure.
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
            print(f"Commit image {commit_image} already exists, skipping build")
            return commit_image

    # Generate Dockerfile if not already present
    dockerfile_path = build_context_dir / "Dockerfile"
    if not dockerfile_path.exists():
        dockerfile_content = generate_commit_dockerfile(config)
        with open(dockerfile_path, "w") as f:
            f.write(dockerfile_content)

    print(f"Building thin commit image for {old_commit_hash}")
    memory_bytes = _parse_memory_limit(config.memory_limit)
    res = run_subprocess_shell(
        f"docker build --memory {memory_bytes} "
        f"-t {commit_image} . "
        f'--build-arg OLD_COMMIT="{old_commit_hash}"',
        capture_output=False,
        cwd=build_context_dir,
        timeout=config.commit_build_timeout,
    )

    if res.returncode != 0:
        print("Failed to build commit image. See docker build logs above.")
        return None

    if config.push:
        if not push_image(commit_image):
            return None

    return commit_image


def push_image(image_name: str) -> bool:
    """Push a Docker image to the registry.

    Args:
        image_name: Full image name including registry and tag.

    Returns:
        True on success, False on failure.
    """
    print(f"Pushing docker image {image_name}")
    res = subprocess.run(
        ["docker", "push", image_name],
        capture_output=True,
    )
    if res.returncode != 0:
        print(f"Failed to push {image_name}: {res.stderr}")
        return False
    print(f"Successfully pushed {image_name}")
    return True


def _parse_memory_limit(limit: str) -> int:
    """Convert human-readable memory limit (e.g. '1g', '512m') to bytes."""
    limit = limit.strip().lower()
    multipliers = {"k": 1024, "m": 1024**2, "g": 1024**3}
    if limit[-1] in multipliers:
        return int(float(limit[:-1]) * multipliers[limit[-1]])
    return int(limit)
