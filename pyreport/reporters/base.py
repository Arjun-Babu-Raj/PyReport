"""
BaseReporter abstract base class.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from ..core import Report


class BaseReporter(ABC):
    """
    Abstract base class for all reporters.

    Subclasses must implement :meth:`report`.
    """

    def __init__(self, obj, **kwargs):
        self.obj = obj
        self.kwargs = kwargs
        self._warnings: List[str] = []

    def _warn(self, message: str) -> None:
        """Add a warning to the internal list."""
        import warnings
        self._warnings.append(message)
        warnings.warn(message, stacklevel=3)

    @abstractmethod
    def report(self) -> Report:
        """Generate and return a :class:`~pyreport.core.Report`."""
        ...
