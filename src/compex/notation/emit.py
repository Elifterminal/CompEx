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
        *_patterns(composition),
        *_voices(composition),
        *_melody(composition),
        *_ledger(composition),
        *_evolution(composition),
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


def _patterns(composition: Composition) -> list[str]:
    """The rhythm each voice settled on, as the grid it actually plays.

    A pattern is written as its slot vector rather than as a comb, because it
    is no longer a period — it is a decision, one slot at a time, and half of
    it would be lost by describing it as a pulse with an offset.
    """
    lines: list[str] = []
    for figure in composition.final_patterns():
        cells = ",".join(f"{value:g}" for value in figure.slots)
        lines.append(
            f"P_{{\\mathrm{{{figure.voice}}}}}=\\left[{cells}\\right]"
            f"_{{\\Delta={figure.subdivision:g}}}"
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


def _melody(composition: Composition) -> list[str]:
    """The auditions: how many lines were imagined, which won, what it listens for.

    Like the evolution block, none of this can be reconstructed from the lines
    above it. The chosen phrase is in the notes; the four it beat are not, and
    without them the piece reads as if it had never had a choice.
    """
    if not composition.melodies:
        return []

    imagined = sum(choice.considered for choice in composition.melodies)
    deepest = max(choice.chosen.generation for choice in composition.melodies)
    lines = [
        "\\mathrm{MELODY}:\\ "
        f"{len(composition.melodies)}\\mathrm{{\\ chosen\\ from\\ }}{imagined}"
        f"\\mathrm{{\\ imagined}},\\ {deepest}\\mathrm{{\\ generations}}"
    ]

    for choice in composition.melodies:
        steps = ",".join(f"{step:+d}" if step else "0" for step in choice.chosen.steps)
        beaten = ",\\ ".join(f"\\mathrm{{{origin}}}\\,{score:.2f}"
                             for origin, score in choice.rejected) or "\\mathrm{-}"
        lines.append(
            f"\\Lambda_{{{choice.movement}}}@{choice.at_beat:g}="
            f"\\left\\{{{steps}\\right\\}}_{{\\mathrm{{{choice.chosen.origin}}}}}"
            f"\\,{choice.score:.3f}\\ \\mathrm{{over}}\\ {beaten}"
        )

    taste = composition.taste
    strayed = taste.strayed_from(composition.opening_taste)
    lines.append(
        "\\Psi_{\\mathrm{taste}}=\\left("
        + ",\\ ".join(f"\\mathrm{{{name}}}={value:.2f}" for name, value in taste.weights())
        + "\\right)"
        + (f"\\quad\\mathrm{{moved}}:\\ \\mathrm{{{', '.join(strayed)}}}" if strayed else "")
    )

    moved = [grid for grid in composition.grids if grid.past_start()]
    if moved:
        lines.append(
            "\\mathrm{GRIDS\\ past\\ their\\ starting\\ weights}:\\ "
            + ",\\ ".join(f"\\mathrm{{{grid.voice}}}({grid.past_start()}"
                          f"\\mathrm{{\\ slots}},\\ \\bar\\Delta{grid.strayed():.2f})"
                          for grid in moved)
        )
    return lines


def _ledger(composition: Composition) -> list[str]:
    """What the piece owed, what it paid, and what it decided to keep owing.

    The unpaid line is the interesting one. Everything settled is audible in
    the notes; what is still open at the end is the part of the piece that was
    deliberately not finished, and there is nowhere else to read it.
    """
    ledger = composition.ledger
    now = composition.total_beats
    if not ledger.paid and not ledger.live(now):
        return []

    lines = [
        "\\mathrm{OWED}:\\ "
        f"{len(ledger.paid)}\\mathrm{{\\ settled}},\\ "
        f"{len(ledger.live(now))}\\mathrm{{\\ still\\ open}},\\ "
        f"\\Pi={ledger.pressure(now):.2f}\\mathrm{{\\ pressure}}"
    ]

    for settled in ledger.paid:
        lines.append(
            f"\\Omega_{{\\mathrm{{{settled.promise.kind}}}}}"
            f"({settled.promise.opened_at:g}\\!\\rightarrow\\!{settled.at_beat:g})="
            f"\\mathrm{{settled\\ after\\ }}{settled.waited:g}\\mathrm{{\\ beats}}"
        )

    for owed in ledger.live(now):
        lines.append(
            f"\\Omega_{{\\mathrm{{{owed.kind}}}}}({owed.opened_at:g})="
            f"\\mathrm{{open}},\\ \\mathrm{{pressure}}\\,{owed.pressure(now):.2f}"
        )

    deepest = ledger.deepest(now)
    if deepest is not None:
        lines.append(
            "\\mathrm{CARRIED}=\\mathrm{" + deepest.kind + "}\\ \\mathrm{for\\ }"
            f"{now - deepest.opened_at:g}\\mathrm{{\\ beats}}"
        )
    return lines


def _evolution(composition: Composition) -> list[str]:
    """Write out where the piece changed its mind, and what made it.

    This is the part that cannot be reconstructed by reading the other lines —
    it is the record of the composer listening to itself.
    """
    if not composition.evolution:
        return []

    lines = [
        "\\mathrm{EVOLUTION}:\\ " +
        f"{len(composition.evolution)}\\mathrm{{\\ listen\\text{{-}}backs}},\\ " +
        f"{sum(len(s.adjustments) for s in composition.evolution)}\\mathrm{{\\ corrections}}"
    ]
    for step in composition.evolution:
        heard = ",\\ ".join(
            f"\\mathrm{{{v.principle.name}}}={v.measured:.2f}" for v in step.unhappy
        ) or "\\mathrm{all\\ satisfied}"
        # A starred move is one that went past the bound the drive started with —
        # the composer deciding the limit was a guess rather than a rule.
        did = ",\\ ".join(
            f"\\mathrm{{{a.drive}}}\\,{a.before:.2f}\\!\\rightarrow\\!{a.after:.2f}"
            + ("^{\\ast}" if a.kind == "bound" else "")
            for a in step.adjustments
        ) or "\\mathrm{held}"
        lines.append(
            f"E_{{{step.movement}}}(\\mathrm{{{step.movement_name}}}@{step.at_beat:g})="
            f"\\left\\{{\\mathrm{{heard}}:\\ {heard};\\ \\mathrm{{did}}:\\ {did}\\right\\}}"
        )

    final = composition.final_drives
    lines.append(
        "\\Theta_{\\mathrm{final}}=\\left("
        + ",\\ ".join([
            f"\\mathrm{{novelty}}={final.novelty_pressure:.2f}",
            f"\\mathrm{{gap}}={final.gap_fill:.2f}",
            f"\\mathrm{{pull}}={final.register_pull:.2f}",
            f"\\mathrm{{reach}}={final.register_reach:.2f}",
            f"\\mathrm{{diss}}={final.dissonance_ceiling:.2f}",
            f"\\mathrm{{dens}}={final.density_bias:.2f}",
            f"\\mathrm{{recall}}={final.motif_recall:.2f}",
            f"\\mathrm{{plasticity}}={final.plasticity:.2f}",
        ])
        + "\\right)"
    )
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
