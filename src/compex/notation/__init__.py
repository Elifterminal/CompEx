"""Notation: writing out what the composer decided, and reading it back in.

The engine no longer *consumes* hand-written formulas — it writes them. What
survives here is the LaTeX normaliser, the emitter, and just enough of a
reader to make an emitted formula reproduce its own track.
"""

from compex.notation.emit import emit
from compex.notation.reentry import Formula, ReentryError, read_formula

__all__ = ["emit", "Formula", "ReentryError", "read_formula"]
