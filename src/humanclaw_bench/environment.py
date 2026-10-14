"""Load local configuration from the release root without replacing shell values."""

from __future__ import annotations

from dotenv import load_dotenv

from .paths import repository_root


def load_project_environment() -> bool:
    """Load only the release root's .env, preserving existing environment values.

    An explicit path prevents discovery of unrelated parent or working-directory
    files. The standard PYTHON_DOTENV_DISABLED switch can disable loading for
    isolated tests or deployments. Credentials are never returned or logged.
    """

    return load_dotenv(
        dotenv_path=repository_root() / ".env",
        override=False,
        encoding="utf-8",
    )
