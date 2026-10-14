"""Keep unit tests independent of private configuration in the checkout."""

import pytest


@pytest.fixture(autouse=True)
def disable_real_project_dotenv(monkeypatch):
    """Require dotenv tests to explicitly opt into their own temporary .env file."""

    monkeypatch.setenv("PYTHON_DOTENV_DISABLED", "1")
