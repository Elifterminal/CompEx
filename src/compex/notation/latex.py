"""Normalise messy LaTeX into a plain, line-oriented form the parser can read.

The input is whatever survived a copy-paste out of a chat window, so this is
deliberately forgiving. Braces that lost their underscore (``f{LFO}``),
subscripts glued to the time variable (``f_ct``), ``\\frac`` shorthand and
stray ``\\left``/``\\right`` decoration all get cleaned up rather than
rejected. Anything it genuinely cannot read is left intact for the parser to
report as an unrecognised directive.
"""

from __future__ import annotations

import re

# Applied in order. Order matters more than elegance here.
_RULES: tuple[tuple[str, str], ...] = (
    (r"\$", ""),
    # Arrows first: \right would otherwise swallow the head of \rightarrow.
    (r"\\(?:rightarrow|longrightarrow|Rightarrow|to)(?![A-Za-z])", " -> "),
    (r"\\(?:left|right|displaystyle|limits)(?![A-Za-z])", ""),
    (r"\\frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}", r"(\1/\2)"),
    (r"\\frac\s*(\d)\s*(\d)", r"(\1/\2)"),
    (r"\\in\b", " in "),
    (r"\\(?:cdot|times)\b", "*"),
    (r"\\(?:qquad|quad)", "\n\n"),           # a wide gap separates directives
    (r"\\[,;:!]", " "),
    (r"\\ ", " "),
    (r"\\\\", "\n\n"),
)

_WRAPPER = re.compile(r"\\(?:mathrm|mathbf|mathit|text|textrm|operatorname)\s*\{([^{}]*)\}")
_LEFTOVER_MACRO = re.compile(r"\\([A-Za-z]+)")
_SIMPLE_SUBSCRIPT = re.compile(r"_\s*\{([A-Za-z0-9.]+)\}")
_GLUED_BRACE = re.compile(r"([A-Za-z][A-Za-z0-9]*)\{([A-Za-z0-9.]+)\}")


def normalise(source: str) -> str:
    """Return ``source`` with LaTeX decoration removed, blocks separated by blank lines."""
    if not isinstance(source, str):
        raise TypeError(f"normalise() expects str, got {type(source).__name__}")

    text = source
    for pattern, replacement in _RULES:
        text = re.sub(pattern, replacement, text)

    # \mathrm{...} can nest; keep unwrapping until it stops changing.
    for _ in range(8):
        unwrapped = _WRAPPER.sub(r"\1", text)
        if unwrapped == text:
            break
        text = unwrapped

    # Any macro still standing loses its backslash but keeps a space in front,
    # so \sum_{n}\delta(...) does not collapse into one glued identifier.
    text = _LEFTOVER_MACRO.sub(r" \1", text)

    text = _SIMPLE_SUBSCRIPT.sub(r"_\1", text)   # _{LFO} -> _LFO
    text = _GLUED_BRACE.sub(r"\1_\2", text)      # f{LFO} -> f_LFO, SUB{35Hz} -> SUB_35Hz

    return text


def directives(source: str) -> tuple[str, ...]:
    """Split normalised notation into one-line directives, blank-line delimited."""
    text = normalise(source)
    blocks = re.split(r"\n\s*\n", text)

    out: list[str] = []
    for block in blocks:
        line = re.sub(r"\s+", " ", block.replace("\n", " ")).strip()
        line = line.strip(",; ").strip()
        if line:
            out.append(line)
    return tuple(out)
