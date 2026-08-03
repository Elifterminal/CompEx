"""One description of a piece, for every surface that shows one.

The phone player and the desktop app both need to say what the composer
decided. If each built its own JSON they would drift the first time either
grew a field, and then the two would be describing different music — the same
failure the browser player avoids by running the real package instead of a
port.

Everything here is plain lists and dicts of numbers and strings, so it crosses
the Pyodide boundary and the HTTP boundary without help.
"""

from __future__ import annotations

from compex.generate.compose import Composition


def movements(composition: Composition) -> list[dict]:
    """Movement boundaries in seconds, so a waveform or a playhead can be marked."""
    out: list[dict] = []
    cursor = 0.0
    seconds_per_beat = composition.seconds_per_beat
    for movement in composition.movements:
        out.append({
            "name": movement.name,
            "energy": round(movement.energy, 3),
            "start": round(cursor * seconds_per_beat, 3),
            "end": round((cursor + movement.beats) * seconds_per_beat, 3),
        })
        cursor += movement.beats
    return out


def evolution(composition: Composition) -> list[dict]:
    """Where the composer listened back and changed its mind.

    Taste shifts ride along with the drive corrections from the same
    listen-back, because they were caused by the same verdicts and reading
    them apart makes both look arbitrary.
    """
    retuned = {step.movement: step for step in composition.retunings}
    seconds_per_beat = composition.seconds_per_beat

    out: list[dict] = []
    for step in composition.evolution:
        shifts = retuned.get(step.movement)
        out.append({
            "movement": step.movement,
            "name": step.movement_name,
            "at": round(step.at_beat * seconds_per_beat, 2),
            "heard": [{"principle": verdict.principle.name,
                       "measured": round(verdict.measured, 3),
                       "attribution": verdict.principle.attribution}
                      for verdict in step.unhappy],
            "did": [{"drive": change.drive,
                     "before": round(change.before, 3),
                     "after": round(change.after, 3),
                     "because": change.principle,
                     "kind": change.kind}
                    for change in step.adjustments],
            "taste": [{"criterion": shift.criterion,
                       "before": round(shift.before, 3),
                       "after": round(shift.after, 3),
                       "because": shift.because,
                       "past_start": shift.past_start}
                      for shift in (shifts.taste if shifts else ())]
                     + [{"criterion": f"groove {shift.criterion}",
                         "before": round(shift.before, 3),
                         "after": round(shift.after, 3),
                         "because": shift.because,
                         "past_start": False}
                        for shift in (shifts.groove if shifts else ())],
            "note": step.note(),
            "plasticity": round(step.after.plasticity, 3),
            "unrest": round(step.after.unrest, 3),
            "broke": list(step.after.beyond_start()),
        })
    return out


def melody(composition: Composition) -> list[dict]:
    """Every audition: what won, what it beat, and what decided it."""
    seconds_per_beat = composition.seconds_per_beat
    return [{
        "movement": choice.movement,
        "at": round(choice.at_beat * seconds_per_beat, 2),
        "origin": choice.chosen.origin,
        "generation": choice.chosen.generation,
        "steps": list(choice.chosen.steps),
        "rhythm": [round(value, 3) for value in choice.chosen.rhythm],
        "score": round(choice.score, 3),
        "margin": round(choice.margin, 3),
        "considered": choice.considered,
        "beat": [{"origin": origin, "score": round(score, 3)}
                 for origin, score in choice.rejected],
        "scores": [{"name": name, "value": round(value, 3)} for name, value in choice.scores],
        "note": choice.note(),
    } for choice in composition.melodies]


def patterns(composition: Composition) -> list[dict]:
    """The rhythm each voice ended on, and how far its grid moved to get there."""
    grids = {grid.voice: grid for grid in composition.grids}
    revisions: dict[str, int] = {}
    for choice in composition.patterns:
        revisions[choice.voice] = revisions.get(choice.voice, 0) + 1

    out: list[dict] = []
    for figure in composition.final_patterns():
        grid = grids.get(figure.voice)
        out.append({
            "voice": figure.voice,
            "figure": figure.describe(),
            "slots": [round(value, 3) for value in figure.slots],
            "subdivision": figure.subdivision,
            "cycle_beats": round(figure.cycle_beats, 3),
            "hits": figure.hit_count,
            "origin": figure.origin,
            "revisions": revisions.get(figure.voice, 0),
            "weights": [round(value, 3) for value in grid.weights] if grid else [],
            "start_weights": [round(value, 3) for value in grid.start] if grid else [],
            "strayed": round(grid.strayed(), 3) if grid else 0.0,
            "past_start": grid.past_start() if grid else 0,
        })
    return out


def ledger(composition: Composition) -> dict:
    """What the piece owes, what it settled, and what it is still carrying."""
    book = composition.ledger
    now = composition.total_beats
    spb = composition.seconds_per_beat
    deepest = book.deepest(now)

    return {
        "open": [{"kind": owed.kind,
                  "domain": owed.domain,
                  "opened": round(owed.opened_at * spb, 2),
                  "strength": round(owed.strength, 3),
                  "pressure": round(owed.pressure(now), 3),
                  "carried": round((now - owed.opened_at) * spb, 2)}
                 for owed in book.live(now)],
        "paid": [{"kind": settled.promise.kind,
                  "domain": settled.promise.domain,
                  "opened": round(settled.promise.opened_at * spb, 2),
                  "settled": round(settled.at_beat * spb, 2),
                  "waited": round(settled.waited * spb, 2),
                  "how": settled.how}
                 for settled in book.paid],
        "pressure": round(book.pressure(now), 3),
        "carrying": ({"kind": deepest.kind,
                      "domain": deepest.domain,
                      "beats": round(now - deepest.opened_at, 2),
                      "seconds": round((now - deepest.opened_at) * spb, 2)}
                     if deepest else None),
    }


def hindsight(composition: Composition) -> dict | None:
    """What the ending gave back to the beginning."""
    reveal = composition.reveal
    if reveal is None or not reveal.phrases:
        return None
    recalls = [choice for choice in composition.melodies
               if choice.chosen.origin.startswith("recall")]
    spb = composition.seconds_per_beat
    return {
        "then": round(reveal.then, 1),
        "now": round(reveal.now, 1),
        "literal": round(reveal.literal, 1),
        "saved": round(reveal.saved, 1),
        "share": round(reveal.share, 4),
        "phrases": reveal.phrases,
        "note": reveal.describe(),
        "recalls": [{"at": round(choice.at_beat * spb, 2), "origin": choice.chosen.origin}
                    for choice in recalls],
    }


def mix(decided) -> dict:
    """What the mixer measured and what it did about it."""
    from compex.dsp.balance import BAND_NAMES

    return {
        "percussive": round(decided.percussive, 4),
        "summary": decided.summary(),
        "buried": list(decided.buried()),
        "still_buried": list(decided.still_buried()),
        "voices": [{"voice": voice.voice,
                    "role": voice.role,
                    "home": BAND_NAMES[voice.home],
                    "share": round(voice.best_share, 4),
                    "after": round(voice.after, 4),
                    "floor": round(voice.floor(), 4),
                    "trim": round(voice.trim, 3),
                    "plays": round(voice.presence, 3),
                    "note": voice.describe()}
                   for voice in sorted(decided.voices, key=lambda v: -v.best_share)],
    }


def tuning(composition: Composition) -> dict:
    """What the piece is tuned to, and how far that is from the usual grid."""
    voicing = composition.tuning
    return {
        "name": voicing.name,
        "family": voicing.family,
        "detail": voicing.detail,
        "degrees": len(voicing.steps),
        "steps": [round(step, 3) for step in voicing.steps],
        "cents": [round(value, 1) for value in voicing.cents()],
        # How far each degree sits from the nearest twelve-tone pitch, in
        # cents. Zero all the way down means it is in twelve; anything else is
        # the measure of how far out it went.
        "off_grid": [round(min(abs(value - nearest * 100.0)
                               for nearest in range(13)), 1)
                     for value in voicing.cents()],
    }


def clocks(composition: Composition) -> dict:
    """Which voices are on their own tempo, and when they meet the pulse again."""
    bar = float(composition.beats_per_bar)
    spb = composition.seconds_per_beat
    loose = [clock for clock in composition.clocks if not clock.anchored]
    return {
        "voices": len(composition.clocks),
        "loose": [{"voice": clock.voice,
                   "ratio": f"{clock.numerator}:{clock.denominator}",
                   "rate": round(clock.ratio, 3),
                   "meets_every_beats": round(clock.meets_every(bar), 2),
                   "meets_every_seconds": round(clock.meets_every(bar) * spb, 2)}
                  for clock in loose],
    }


def heard(composition: Composition) -> dict:
    """What the piece measured by listening to itself, movement by movement."""
    return {
        "movements": [{"roughness": round(reading.roughness, 4),
                       "brightness": round(reading.brightness, 4),
                       "motion": round(reading.motion, 4)}
                      for reading in composition.heard],
        "note": composition.heard[-1].describe() if composition.heard else "not listened to",
    }


def timbre(composition: Composition) -> dict:
    """How far each instrument travelled from what it started as."""
    first = {name: dict(spec.params) for name, movement, spec in composition.states
             if movement == 0}
    latest = max((movement for _, movement, _ in composition.states), default=0)
    last = {name: dict(spec.params) for name, movement, spec in composition.states
            if movement == latest}

    voices = []
    for name, start in first.items():
        end = last.get(name, start)
        moved = [(key, start[key], end[key]) for key in start
                 if key in end and start[key] and abs(end[key] - start[key]) > 1e-9]
        if not moved:
            continue
        travel = sum(abs(new - old) / abs(old) for _, old, new in moved) / len(moved)
        voices.append({
            "voice": name,
            "travel": round(travel, 4),
            "params": [{"name": key, "from": round(old, 4), "to": round(new, 4)}
                       for key, old, new in sorted(
                           moved, key=lambda row: -abs(row[2] - row[1]) / abs(row[1]))[:4]],
        })
    return {"voices": sorted(voices, key=lambda row: -row["travel"]),
            "movements": latest + 1}


def ghosts(composition: Composition, gain: float = 0.0) -> dict:
    """The lines the composer decided against, and how loudly they are heard."""
    spb = composition.seconds_per_beat
    per_choice = [choice for choice in composition.melodies if choice.ghosts]
    closest = sorted(
        ((ghost, choice) for choice in per_choice for ghost in choice.ghosts),
        key=lambda pair: pair[0].margin)[:8]

    return {
        "gain": round(gain, 3),
        "notes": len(composition.ghosts),
        "auditions": len(per_choice),
        "turned_down": sum(len(choice.ghosts) for choice in per_choice),
        "loudest": [{"at": round(choice.at_beat * spb, 2),
                     "origin": ghost.phrase.origin,
                     "margin": round(ghost.margin, 4),
                     "heard_at": round(ghost.closeness(), 3),
                     "beaten_by": choice.chosen.origin}
                    for ghost, choice in closest],
        "torn": round(sum(ghost.closeness() for choice in per_choice
                          for ghost in choice.ghosts)
                      / max(1, sum(len(choice.ghosts) for choice in per_choice)), 3),
    }


def taste(composition: Composition) -> dict:
    """What the composer listens for now, against what it started out listening for."""
    now, start = composition.taste, composition.opening_taste
    moved = set(now.strayed_from(start))
    return {
        "criteria": [{"name": name,
                      "now": round(value, 3),
                      "start": round(getattr(start, name), 3),
                      "moved": name in moved}
                     for name, value in now.weights()],
        "groove": [{"name": name, "now": round(value, 3)}
                   for name, value in composition.groove.weights()],
        "curiosity": round(now.curiosity, 3),
        "auditioned": sum(choice.considered for choice in composition.melodies),
        "chosen": len(composition.melodies),
        "generations": max((choice.chosen.generation for choice in composition.melodies),
                           default=0),
    }
