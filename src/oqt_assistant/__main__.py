# SPDX-FileCopyrightText: 2025 Ivo Djidrovski <i.djidrovski@uu.nl>
#
# SPDX-License-Identifier: Apache-2.0

"""Module entry point so users can run `python -m oqt_assistant`."""

from __future__ import annotations

from .cli import main as cli_main


def main(argv: list[str] | None = None) -> int:
    """Delegate to the CLI main function."""
    return cli_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
