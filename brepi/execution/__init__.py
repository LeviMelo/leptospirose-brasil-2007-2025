"""Bounded external-process execution for expensive statistical engines."""

from .bounded import ProcessResult, run_bounded

__all__ = ["ProcessResult", "run_bounded"]
