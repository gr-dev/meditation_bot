"""Project configuration helpers.

This module centralises configuration loading for the project. It will
attempt to load a `.env` file (if `python-dotenv` is installed) and
expose commonly used values like `MISTRAL_API_KEY`, `MISTRAL_API_URL` and
`TELEGRAM_BOT_TOKEN` via module-level variables.

Usage:
    from src import config
    key = config.MISTRAL_API_KEY

Note: sensitive values are expected to be stored in a `.env` file or
environment variables. `.env` is listed in `.gitignore`.
"""
from __future__ import annotations

import os
from typing import Optional

# Try to load .env automatically if python-dotenv is available. This is a
# convenience for local development. In production, prefer real environment
# variables or a secrets manager.
try:
    from dotenv import load_dotenv

    # load .env from project root if present
    _root = os.path.dirname(os.path.dirname(__file__))
    load_dotenv(os.path.join(_root, ".env"))
except Exception:
    # python-dotenv not installed or load failed; rely on existing env vars
    pass


def _get(key: str, default: Optional[str] = None) -> Optional[str]:
    return os.getenv(key, default)


# Common configuration variables used across the project
MISTRAL_API_KEY: Optional[str] = _get("MISTRAL_API_KEY")
MISTRAL_API_URL: str = _get("MISTRAL_API_URL", "https://api.mistral.ai")
TELEGRAM_BOT_TOKEN: Optional[str] = _get("TELEGRAM_BOT_TOKEN")


def as_dict() -> dict:
    """Return configuration as a dict (useful for tests or logging).

    Avoid printing secrets in logs; if you log this dict, redact sensitive
    values first.
    """
    return {
        "MISTRAL_API_KEY": MISTRAL_API_KEY,
        "MISTRAL_API_URL": MISTRAL_API_URL,
        "TELEGRAM_BOT_TOKEN": TELEGRAM_BOT_TOKEN,
    }


__all__ = ["MISTRAL_API_KEY", "MISTRAL_API_URL", "TELEGRAM_BOT_TOKEN", "as_dict"]
