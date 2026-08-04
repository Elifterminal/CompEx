# CompEx

A Computational Expressionism engine. You tell it how long the track should be
and how it should feel. It writes the music.

Nothing is prewritten. From three inputs — a seed, a runtime and a mood — it
invents the tempo, the meter, the key and mode, the chord progression, a germ
motif, the form, which instruments exist and how each one is voiced. Then it
writes out the formula for what it just made, and that formula regenerates the
identical track, byte for byte — under conditions that are narrower than that
sentence sounds, and which are spelled out under **Reproducibility** below.

It will sound strange. That is the point; it is an experiment in letting the
computation carry itself rather than execute something a person wrote.

## Play it in a browser

**<https://elifterminal.github.io/CompEx/play/>** — works on a phone.

It composes and plays on the device. Nothing is uploaded; the only download is
the engine itself, and after the first visit that is cached. Long tracks stream:
spans are rendered a few seconds ahead of the playhead, so playback starts in
about six seconds whether the track is one minute or thirty, and you watch the
composer change its mind as it goes.

The player runs *this package*, shipped as source and executed by Pyodide — not
a JavaScript port. A port would be a second engine that disagreed with this one
the first time either was touched. `docs/check_page.py` fails the build if the
shipped archive falls behind the source.

## Running it locally

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
compex make --mood melancholy --stems           # every voice as its own file
```

To get it in your application menu:

```bash
./packaging/install_launcher.sh
```

Tracks land in `~/compex/out/` as `<theme>_seed<N>_<fingerprint>.<ext>`, each
with its `.tex` formula beside it. The fingerprint is a digest of the audio
itself, so two files with the same name are the same recording. `replay` has
limits worth knowing before relying on it — see [Reproducibility](#reproducibility).

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

**29 synthesis engines**, grouped by family:

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

Every engine leaves at the same *loudness*, deliberately: without that a
self-oscillating resonance comes out fifty times quieter than a sine and the
composer's per-voice gain stops meaning anything.

This used to say *peak* level, and that was wrong in a way nobody could hear
until it was pointed out — peak says how tall a sound is, not how loud. A
swelling pad and a plucked string reach the same height and nothing like the
same volume, which is most of the reason the pads were inaudible. Some credit
is still given to height, because a transient matched purely on loudness
vanishes the other way.

## Layout

```
src/compex/
  rng.py           seekable RNG — value = f(seed, stream, index), no sequential state
  generate/        mood axes, music theory, the sound palette, the composer
  dsp/
    engines/       29 synthesis engines, by family
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

## What it does beyond writing notes

Each of these has an off switch, and each switch is checked byte for byte
against the build from before the mechanism existed. Anything that could not be
measured moving what it claims to move was deleted rather than shipped.

- **It chooses rather than recites.** Every phrase beats two to six others in
  an audition; every rhythm is a grid the composer keeps revising out of its
  own output.
- **You can hear it being unsure.** The lines and grooves it turned down play
  underneath at a level set by how close they came. A rejected *groove* is
  mostly the groove that beat it, so only the part they disagree about is
  heard.
- **It carries obligations.** A promise ledger with a maturity curve, so
  settling early is worth less than carrying and settling ripe. It can also
  give up on a debt, which is not the same as settling it.
- **It prices its own opening twice** — as written, and knowing how the piece
  ends — so "the ending explained the beginning" is a number.
- **It mixes itself**, measuring which voices are audible in which band, and
  now how sharply each arrives. Measuring energy alone made it boost a wood
  block by 2.6x for being "buried" when it was the most forward thing in the
  track.
- **It tunes itself** — equal divisions other than twelve, or just intonation
  ranked by Tenney height — and **works out its own scales**, with named modes
  demoted to commentary checked afterwards.
- **It runs voices on more than one clock**, at rational ratios so they meet
  again.
- **Its instruments change while they play.**
- **It listens to a rendered window of itself** each movement and judges the
  roughness, brightness and motion of what came out.
- **It remembers the pieces it has already made** and gets bored of itself.
- **It has a plan** — a target shape for its own internal state, held by three
  structural levers. Two earlier designs for this measured at nothing and were
  deleted; both are documented on the living page.

## Where finished work goes

`out/` is a scratch directory — everything flat, named by mood and seed. The
**library** is for work you want to keep: three parallel trees under
`~/Music/CompEx`, with the *same folder name in each*, so finding a track tells
you where its stems and its formula are.

```
~/Music/CompEx/
├── Tracks/2026-08-04_melancholy_seed249984309_1971295d/
│           └── 2026-08-04_melancholy_seed249984309_1971295d.wav
├── Stems/2026-08-04_melancholy_seed249984309_1971295d/
│           ├── master.wav  ghosts.wav  kick.wav  pad_bowed_2.wav …
└── TrackMeta/2026-08-04_melancholy_seed249984309_1971295d/
            ├── formula.tex
            └── report.json
```

The date comes first so a file browser sorts by when you made it. The
fingerprint comes last because it is the only part that means anything precise:
same fingerprint, same recording.

In the app, **Make it** is a preview that files nothing, and **Save** writes all
three at once in whichever format is selected. From the command line:

```bash
compex make --mood melancholy --library --stems
```

**Stems** are every voice as its own file at exactly the level it has in the
mix — trims applied, sidechain applied — plus a `ghosts` stem carrying the lines
and grooves it decided against. They are the buses that actually went in rather
than the voices re-recorded alone, so they sum back to the master to within the
mastering stage.

**`report.json`** is the whole description of the piece: every movement, the
evolution, the melody and pattern auditions, the ledger, the ghosts, the plan,
the tuning and the mix. Both surfaces write it from the same call, so the file
beside a track cannot drift from what the app was showing when you saved it.

## Determinism

Determinism is not a nicety here — it is what makes a strange moment
*traceable to the machine's own structure* rather than to a random number,
which is the difference between expression and malfunction. The RNG is
positional rather than sequential (`value = f(seed, stream, index)`), so
seeking is free and nothing desyncs. That part is solid: the compositional
decisions have no sequential state to lose.

## Reproducibility

The honest version, because the short version above is narrower than it sounds.

**What holds.** Within one process, one machine, one build: the same seed,
runtime, mood, history and knobs give bit-identical samples. That is what the
suite checks, and it passes.

**What the formula does not carry.** `compex replay` reads `SEED`, `RUNTIME`
and `MOOD`, and nothing else. So three cases reproduce a *different piece*,
silently — replay does not warn, it just hands back something else:

| case | reproduces |
| --- | --- |
| composed with a non-blank memory | the no-history piece |
| rendered at a non-default `ghost_gain` | the default-knob piece |
| rendered at a non-default `master_gain` | the default-knob piece |

The memory case is the sharpest. The formula *prints* the history's digest, so
it carries the evidence that it cannot reproduce its own track — and replay
never reads it. It could not use it if it did: the digest is a 12-character
blake2b checksum of the remembered tables, not the tables.

**No build identifier.** Nothing in the formula names the engine that wrote it,
and this package has been version `0.1.0` through every change in its history.
An old formula replayed by a newer engine can legitimately produce a different
piece, and neither artifact would say so.

**No environment scope.** The dependency is `numpy>=1.24` with no upper bound.
Floating-point DSP is not obliged to be bitwise stable across NumPy versions,
Python versions, CPU architectures or BLAS builds, and none of that is tested.
The guarantee should either be scoped to a pinned environment or verified
across the environments claimed. Neither is done.

**Two artifacts this project conflates.** A *recipe* reproduces the composition
under a named engine build: seed, mood, runtime, the full memory snapshot, an
engine fingerprint, and every consequential setting. A *frozen score* carries
every compositional decision, so the piece can be rebuilt without re-running
the decision process at all, and is therefore immune to the engine changing
underneath it. For byte-identical *audio* a recipe would additionally need the
sample rate, both gains, the source fingerprint, the Python and NumPy versions,
and ideally the expected PCM hash. The `.tex` is currently a recipe missing
three of its ingredients — with a complete human-readable account of every
decision sitting right beside it that nothing can read back.

None of this is built. It is written down because the project's rule is that a
claim gets measured before it is made, and this one was made first.

## Tests

```bash
python3 -m unittest discover -s tests -t .
```

The ones that matter most: every engine, drum and effect renders finite
audible audio at every register, **every one of the fifteen themes composes at
three seeds** (a crash lived for days behind a theme the suite never picked),
every note names a voice that exists, a track's own formula reproduces it
exactly under the conditions below, every mechanism's off switch is checked
byte for byte against the build before it existed, and a piece rendered
span-by-span for streaming still adds up to the same piece without clicking at
the seams.

## Not built yet

**A recipe that is actually complete**, and a **frozen score**. See
[Reproducibility](#reproducibility) — the `.tex` is currently neither.

**Realtime, editable while sounding.** Needs SuperCollider
(`sudo apt install supercollider`) and an OSC layer. The positional RNG exists
partly so that a live edit cannot desync the stream, but nothing is built.

**An audible channel for most of the machine's inner state.** The lines and
grooves it turned down are audible. What the piece *owes*, what its ending
explained about its opening, what it remembers, and what it is trying to do to
itself are all inspectable and none of them make a sound.
