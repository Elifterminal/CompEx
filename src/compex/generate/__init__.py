"""The composer. Knobs go in, a whole piece comes out — nothing prewritten.

The only inputs are a seed, a runtime and an emotional theme. Everything else
— tempo, meter, mode, harmony, motif, form, instrumentation, synthesis
parameters — is invented here, deterministically, and then written back out
as notation so you can read what the machine decided.
"""

from compex.generate.compose import Composition, compose
from compex.generate.mood import THEMES, Mood

__all__ = ["Composition", "Mood", "THEMES", "compose"]
