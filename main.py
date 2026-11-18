"""Entrypoint facade delegating to the modular `meta_dev` software factory."""

from __future__ import annotations

import meta_dev as _meta_dev
from meta_dev import *  # noqa: F403
from meta_dev.cli import main

__all__ = list(_meta_dev.__all__)


if __name__ == "__main__":
    raise SystemExit(main())
