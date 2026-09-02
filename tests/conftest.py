"""Emit concise GitHub annotations for failed scientific tests."""

from __future__ import annotations

import os


def _escape_workflow_message(message: str) -> str:
    return (
        message.replace("%", "%25")
        .replace("\r", "%0D")
        .replace("\n", "%0A")
        .replace(":", "%3A")
        .replace(",", "%2C")
    )


def pytest_runtest_logreport(report: object) -> None:
    """Expose a failed test name and traceback without changing test behavior."""

    if os.environ.get("GITHUB_ACTIONS") != "true" or not getattr(report, "failed", False):
        return
    path, line, _ = getattr(report, "location", ("tests", 0, ""))
    nodeid = getattr(report, "nodeid", "unknown test")
    details = _escape_workflow_message(str(getattr(report, "longrepr", "failed")))
    print(f"::error file={path},line={int(line) + 1},title={nodeid}::{details}")
