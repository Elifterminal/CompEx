"""Write the formula for the piece the machine just made.

This is the readout, in the same notation the engine used to consume. The
``SEED`` / ``RUNTIME`` / ``MOOD`` lines are the load-bearing part — feed those
three back in and you get this exact track again, because composition is
deterministic. Everything below them is exposition: what those three inputs
turned into.
"""

from __future__ import annotations

from compex.generate.compose import Composition
from compex.generate.palette import ROLE_PERC
from compex.generate.theory import Chord

_ROMAN = ("i", "ii", "iii", "iv", "v", "vi", "vii", "viii")
_NAMES = ("C", "C\\#", "D", "D\\#", "E", "F", "F\\#", "G", "G\\#", "A", "A\\#", "B")


def emit(composition: Composition, runtime_s: float | None = None) -> str:
    """Serialise ``composition`` as equation notation."""
    runtime = runtime_s if runtime_s is not None else composition.total_seconds
    blocks = [
        _header(composition, runtime),
        _tempo(composition),
        _scale(composition),
        _harmony(composition),
        _motif(composition),
        _form(composition),
        *_combs(composition),
        *_voices(composition),
    ]
    return "\n\n".join(block for block in blocks if block)


def _header(composition: Composition, runtime: float) -> str:
    mood = composition.mood
    axes = ",\\ ".join(
        f"\\mathrm{{{axis[0]}}}\\,{getattr(mood, axis):.2f}"
        for axis in ("valence", "energy", "tension", "density", "grit")
    )
    return (
        f"\\mathrm{{SEED}}={composition.seed},\\qquad"
        f"\\mathrm{{RUNTIME}}={runtime:.0f}\\mathrm{{\\ s}}\n\n"
        f"\\mathrm{{MOOD}}=\\left({axes}\\right)"
        f"\\ \\approx\\ \\mathrm{{{mood.nearest_theme()}}}"
    )


def _tempo(composition: Composition) -> str:
    return (
        f"\\mathrm{{BPM}}={composition.bpm:g},\\qquad"
        f"\\mathrm{{METER}}={composition.beats_per_bar}/4"
    )


def _scale(composition: Composition) -> str:
    degrees = ",".join(str(step) for step in composition.scale)
    tonic = _NAMES[composition.root_pitch % 12]
    return (
        f"\\Sigma=\\left\\{{{degrees}\\right\\}}_{{\\mathrm{{{composition.scale_name}}}}}"
        f"\\ \\mathrm{{on\\ }}{tonic}_{{{composition.root_pitch // 12 - 1}}}"
    )


def _harmony(composition: Composition) -> str:
    chain = "\\rightarrow".join(
        _roman(chord, composition.scale) for chord in composition.progression
    )
    return (
        f"H(n)={chain},\\qquad"
        f"\\Delta_H={composition.chord_beats:g}\\mathrm{{\\ beats}}"
    )


def _motif(composition: Composition) -> str:
    steps = ",".join(f"{step:+d}" if step else "0" for step in composition.motif.steps)
    rhythm = ",".join(f"{value:g}" for value in composition.motif.rhythm)
    return (
        f"M=\\left\\{{{steps}\\right\\}},\\qquad"
        f"\\tau=\\left\\{{{rhythm}\\right\\}}\\mathrm{{\\ beats}}"
    )


def _form(composition: Composition) -> str:
    chain = "\\rightarrow".join(
        f"\\mathrm{{{movement.name}}}({movement.beats:g})" for movement in composition.movements
    )
    return f"\\mathrm{{FORM}}_{{\\mathrm{{{composition.archetype}}}}}={chain}"


def _combs(composition: Composition) -> list[str]:
    lines: list[str] = []
    for comb in composition.combs:
        period = f"{comb.period_beats:g}n"
        inner = period if not comb.offset_beats else f"({period}+{comb.offset_beats:g})"
        lines.append(
            f"K_{{\\mathrm{{{comb.name}}}}}(t)=\\sum_{{n}}\\delta(t-{inner})"
        )
    return lines


def _voices(composition: Composition) -> list[str]:
    lines: list[str] = []
    for voice in composition.voices:
        if voice.role == ROLE_PERC:
            continue
        params = ",\\ ".join(
            f"\\mathrm{{{key}}}={value:g}" for key, value in voice.params
        )
        line = (
            f"V_{{\\mathrm{{{voice.role}}}}}="
            f"\\mathrm{{{voice.engine}}}\\left({params}\\right)"
        )
        for name, settings in voice.effects:
            detail = ",\\ ".join(f"\\mathrm{{{key}}}={value:g}" for key, value in settings)
            line += f"\\circ\\mathrm{{{name}}}\\left({detail}\\right)"
        lines.append(f"{line}\\cdot{voice.gain:.2f}")
    return lines


def _roman(chord: Chord, scale: tuple[int, ...]) -> str:
    """Roman numeral with quality — lowercase minor, ° diminished, + augmented."""
    numeral = _ROMAN[chord.degree % len(_ROMAN)]
    semitones = chord.semitones(scale)

    third = semitones[1] - semitones[0] if len(semitones) > 1 else 4
    fifth = semitones[2] - semitones[0] if len(semitones) > 2 else 7

    if third >= 4:
        numeral = numeral.upper()
    suffix = ""
    if third <= 3 and fifth <= 6:
        suffix = "^{\\circ}"
    elif third >= 4 and fifth >= 8:
        suffix = "^{+}"
    elif chord.size == 4:
        suffix = "^{7}"
    elif chord.size >= 5:
        suffix = "^{9}"

    return f"\\mathrm{{{numeral}}}{suffix}"
