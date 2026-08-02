# CompEx

A Computational Expressionism engine. You tell it how long the track should be
and how it should feel. It writes the music.

Nothing is prewritten. From three inputs — a seed, a runtime and a mood — it
invents the tempo, the meter, the key and mode, the chord progression, a germ
motif, the form, which instruments exist and how each one is voiced. Then it
writes out the formula for what it just made, and that formula regenerates the
identical track, byte for byte.

It will sound strange. That is the point; it is an experiment in letting the
computation carry itself rather than execute something a person wrote.

## Running it

numpy is the only dependency. ffmpeg is optional and only needed for MP3.

```bash
./run_compex.sh                       # opens the UI at http://127.0.0.1:8733
```

From the command line:

```bash
compex themes                                   # list the emotional themes
compex make --mood menacing --duration 120 --format mp3
compex make --mood serene --axis grit=0.8       # nudge one axis off the theme
compex replay out/menacing_seed1203_ab12cd.tex  # rebuild a track from its formula
```

To get it in your application menu:

```bash
./packaging/install_launcher.sh
```

Tracks land in `~/compex/out/` as `<theme>_seed<N>_<fingerprint>.<ext>`, each
with its `.tex` formula beside it. The fingerprint is a digest of the audio
itself, so two files with the same name are the same recording.

## The knobs

**Runtime**, **emotional theme**, **seed**, and **WAV or MP3**. That is all.
Everything that would normally be a dial — tempo, key, instrumentation, drive,
density — is something the composer decides.

The theme is not a genre lookup. It is a point in a five-axis space:

| axis | runs from | to |
| --- | --- | --- |
| valence | dark | bright |
| energy | still | frantic |
| tension | resolved | unresolved |
| density | sparse | crowded |
| grit | clean | destroyed |

Fifteen named themes are just convenient coordinates in that space; you can
move the axes directly and land somewhere with no name. The axes constrain
real things — a serene piece will never draw a dark mode or an odd meter past
3/4; a menacing one draws only phrygian dominant, locrian or octatonic, and
takes an odd meter about half the time.

## This box has no speakers

So the UI has **Download** and **Email it**. The email path reuses
`~/scripts/send_email.py` and attaches the track. Nothing is ever sent unless
you press the button.

## The palette

**24 synthesis engines**, grouped by family:

| family | engines |
| --- | --- |
| core | sub, additive, pm, pluck, noise |
| struck | bell, mallet, tine, glass, chime |
| voiced | formant, reed, brass, choir, flute |
| sustained | pad, string, organ, supersaw, drone |
| textural | granular, wavetable, feedback, chip |

**15 drums**: kick, snare, hat, tom, rim, clap, ride, crash, shaker, cowbell,
woodblock, conga, snap, boom, anvil.

**6 effects**: chorus, delay, reverb, tremolo, ring mod, wavefold — and every
voice gets a chain the composer invents. That is what actually widens the
space: the same pluck through a long reverb and through a ring modulator are
two different instruments.

None of these are patches. An engine is a family and the parameters are
invented per piece, so "pluck" covers a harp and a snapped wire. Engines,
drums and effects are all chosen by how near their character sits to the
mood — a serene piece will not reach for an anvil or a wavefolder.

Every engine leaves at the same peak level, deliberately: without that a
self-oscillating resonance comes out fifty times quieter than a sine and the
composer's per-voice gain stops meaning anything.

## Layout

```
src/compex/
  rng.py           seekable RNG — value = f(seed, stream, index), no sequential state
  generate/        mood axes, music theory, the sound palette, the composer
  dsp/
    engines/       24 synthesis engines, by family
    drums.py       15 percussion voices
    effects.py     per-voice chains; feedback structures run a delay-line at a time
    arrange.py     one voice at a time onto its own bus, then mixed
  notation/        writes the formula out, and reads it back to reproduce a track
  audio/           16-bit WAV (stdlib), MP3 via ffmpeg
  delivery/        email
  ui/              local web app
```

`generate/theory.py` is the environment and the seed is a walk through it. Every
move the composer is offered is already idiomatic, so it never has to generate
garbage and filter it.

## It changes its mind while writing

The piece is not decided up front. After each movement the composer listens
back to what it actually wrote, measures it, and shifts the parameters it
writes with before starting the next one — so the second half of a track is a
consequence of the first half, not just of the seed.

It listens against attributed principles rather than invented preferences:

| principle | says |
| --- | --- |
| Berlyne's inverted-U | liking peaks at *moderate* novelty — so novelty gets a band, not a maximum |
| Huron's post-skip reversal | after a large leap, melodies step back the other way |
| von Hippel & Huron | melodies regress toward their central pitch |
| Meyer's expectation | tension has to move to mean anything |
| Huron on self-similarity | music repeats far more than speech — there's a repetition *floor* |
| Schoenberg | the germ stays audible through its transformations |

The bands shift with the mood: a menacing piece is allowed a dissonance a
serene one would fail on.

**Second order.** How hard it reacts also changes. When the same complaints
keep returning it concludes its corrections were too timid and pushes harder;
when things are working it settles and stops interfering. The state evolves,
and the rule that updates the state evolves under it.

**Memory is lossy on purpose.** It doesn't keep the motif, it keeps a sketch —
contour direction and rough durations. So a returning theme can't be
retrieved, only rebuilt, with the gaps filled by whatever the drives have
become. Literal recapitulation is a copy; this is a memory.

Every `.tex` formula carries the full evolution: each correction with the
principle that caused it.

## Determinism

Same three inputs, same samples, always. It is not a nicety — it is what makes
a strange moment *traceable to the machine's own structure* rather than to a
random number, which is the difference between expression and malfunction. The
RNG is positional rather than sequential (`value = f(seed, stream, index)`), so
seeking is free and nothing desyncs.

## Tests

```bash
python3 -m unittest discover -s tests -t .
```

109 tests. The ones that matter most: every engine, drum and effect renders
finite audible audio at every register, every theme renders without blowing
up, every note names a voice that exists, and a track's own formula
reproduces it exactly.

## Not built yet

The **ghost layer**. Right now the composer picks one continuation and the
alternatives vanish. Scoring several and playing the runners-up quietly
underneath — at a gain set by how close they came — would make the machine's
uncertainty audible as texture. That is the part that would make this
Computational Expressionism rather than generative music.

Also not built: realtime, editable while sounding. That needs SuperCollider
(`sudo apt install supercollider`) and an OSC layer.
