"""Docker build configuration for R2E-Gym — fully standalone."""

from __future__ import annotations

import os
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field


DEFAULT_REGISTRY = "namanjain12/"

_PACKAGE_DIR = Path(__file__).resolve().parent


class RepoName(str, Enum):
    sympy = "sympy"
    pandas = "pandas"
    numpy = "numpy"
    scrapy = "scrapy"
    tornado = "tornado"
    statsmodels = "statsmodels"
    pillow = "pillow"
    pyramid = "pyramid"
    datalad = "datalad"
    aiohttp = "aiohttp"
    mypy = "mypy"
    coveragepy = "coveragepy"
    orange3 = "orange3"
    bokeh = "bokeh"


REPOS_WITH_SUBMODULES: set[RepoName] = {RepoName.numpy, RepoName.orange3}

REPO_EXTRA_BASE_FILES: dict[RepoName, list[str]] = {
    RepoName.aiohttp: ["process_aiohttp_updateasyncio.py"],
    RepoName.orange3: ["orange3_conftest.py"],
}

REPO_TEST_COMMANDS: dict[RepoName, str] = {
    RepoName.tornado: ".venv/bin/python -W ignore r2e_tests/tornado_unittest_runner.py",
    RepoName.sympy: "PYTHONWARNINGS='ignore::UserWarning,ignore::SyntaxWarning' .venv/bin/python -W ignore -m pytest -rA r2e_tests",
    RepoName.pandas: "PYTHONWARNINGS='ignore::UserWarning,ignore::SyntaxWarning' .venv/bin/python -W ignore -m pytest -rA -o 'addopts=' r2e_tests",
    RepoName.numpy: "PYTHONWARNINGS='ignore::UserWarning,ignore::SyntaxWarning' .venv/bin/python -W ignore -m pytest -rA r2e_tests",
    RepoName.scrapy: "PYTHONWARNINGS='ignore::UserWarning,ignore::SyntaxWarning' .venv/bin/python -W ignore -m pytest -rA r2e_tests",
    RepoName.pillow: "PYTHONWARNINGS='ignore::UserWarning,ignore::SyntaxWarning' .venv/bin/python -W ignore -m pytest -rA r2e_tests",
    RepoName.pyramid: "PYTHONWARNINGS='ignore::UserWarning,ignore::SyntaxWarning' .venv/bin/python -W ignore -m pytest -rA r2e_tests",
    RepoName.datalad: "PYTHONWARNINGS='ignore::UserWarning,ignore::SyntaxWarning' .venv/bin/python -W ignore -m pytest -rA r2e_tests",
    RepoName.aiohttp: "PYTHONWARNINGS='ignore::UserWarning,ignore::SyntaxWarning' .venv/bin/python -W ignore -m pytest -rA r2e_tests",
    RepoName.coveragepy: "PYTHONWARNINGS='ignore::UserWarning,ignore::SyntaxWarning' .venv/bin/python -W ignore -m pytest -rA r2e_tests",
    RepoName.orange3: "QT_QPA_PLATFORM=minimal PYTHONWARNINGS='ignore::UserWarning,ignore::SyntaxWarning' xvfb-run --auto-servernum .venv/bin/python -W ignore -m pytest -rA r2e_tests",
    RepoName.bokeh: "PYTHONWARNINGS='ignore::UserWarning,ignore::SyntaxWarning' .venv/bin/python -W ignore -m pytest -rA r2e_tests",
}


class DockerBuildConfig(BaseModel):
    repo_name: RepoName
    registry: str = Field(
        default_factory=lambda: os.environ.get("R2E_DOCKER_REGISTRY", DEFAULT_REGISTRY)
    )
    rebuild_base: bool = False
    rebuild_commits: bool = False
    push: bool = False
    memory_limit: str = "1g"
    base_build_timeout: int = 2400
    commit_build_timeout: int = 600

    model_config = {"arbitrary_types_allowed": True}

    # -- derived names --------------------------------------------------------

    @staticmethod
    def _normalize_registry(registry: str) -> str:
        registry = registry.strip()
        if not registry:
            return registry
        if registry.endswith("/"):
            return registry
        return f"{registry}/"

    @property
    def base_image_name(self) -> str:
        registry = self._normalize_registry(self.registry)
        return f"{registry}{self.repo_name.value}_base:latest"

    def commit_image_name(self, commit_hash: str) -> str:
        registry = self._normalize_registry(self.registry)
        return f"{registry}{self.repo_name.value}_final:{commit_hash}"

    @property
    def commit_image_name_prefix(self) -> str:
        registry = self._normalize_registry(self.registry)
        return f"{registry}{self.repo_name.value}_final"

    # -- file paths (all relative to this package) ----------------------------

    @property
    def base_dockerfile(self) -> Path:
        return _PACKAGE_DIR / "dockerfiles" / f"Dockerfile.{self.repo_name.value}"

    @property
    def install_script(self) -> Path:
        return _PACKAGE_DIR / "install_scripts" / f"{self.repo_name.value}_install.sh"

    @property
    def helpers_dir(self) -> Path:
        return _PACKAGE_DIR / "install_scripts"

    # -- repo metadata --------------------------------------------------------

    @property
    def needs_submodule_update(self) -> bool:
        return self.repo_name in REPOS_WITH_SUBMODULES

    @property
    def extra_base_files(self) -> list[str]:
        return REPO_EXTRA_BASE_FILES.get(self.repo_name, [])

    @property
    def tests_cmd(self) -> str:
        cmd = REPO_TEST_COMMANDS.get(self.repo_name)
        if cmd is None:
            raise NotImplementedError(
                f"tests_cmd not defined for {self.repo_name.value}"
            )
        return cmd
