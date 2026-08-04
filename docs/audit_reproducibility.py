"""What the formula actually reproduces, measured rather than claimed."""
import platform, sys
import numpy as np

from compex.config import SAMPLE_RATE, Knobs
from compex.generate import THEMES, compose
from compex.notation import read_formula
from compex.pipeline import make_track
from compex.remember import Memory, learn

SEED, SECONDS = 249984309, 30.0
MOOD = THEMES["melancholy"]

def replay(formula: str):
    parsed = read_formula(formula)
    return make_track(Knobs(seed=parsed.seed, duration_s=parsed.runtime_s), parsed.mood)

print(f"python {platform.python_version()} · numpy {np.__version__} · "
      f"{platform.machine()} · sample rate {SAMPLE_RATE}")

print("\n1. BASELINE — blank memory, default knobs")
base = make_track(Knobs(seed=SEED, duration_s=SECONDS), MOOD)
again = replay(base.formula)
print(f"   original {base.fingerprint}  replay {again.fingerprint}  "
      f"{'MATCH' if base.fingerprint == again.fingerprint else 'DIFFERENT'}")

print("\n2. A COMPOSITION MADE WITH NON-BLANK MEMORY")
memory = Memory()
for _ in range(3):
    memory = learn(memory, compose(SEED, SECONDS, MOOD, memory))
with_history = make_track(Knobs(seed=SEED, duration_s=SECONDS), MOOD, memory=memory)
back = replay(with_history.formula)
print(f"   formula says MEMORY={memory.digest()} ({memory.pieces} pieces)")
print(f"   original {with_history.fingerprint}  replay {back.fingerprint}  "
      f"{'MATCH' if with_history.fingerprint == back.fingerprint else 'DIFFERENT'}")

print("\n3. A RENDER AT A DIFFERENT GHOST LEVEL")
loud = make_track(Knobs(seed=SEED, duration_s=SECONDS, ghost_gain=0.9), MOOD)
back = replay(loud.formula)
print(f"   original {loud.fingerprint} (ghost 0.9)  replay {back.fingerprint} (default "
      f"{Knobs.ghost_gain})  {'MATCH' if loud.fingerprint == back.fingerprint else 'DIFFERENT'}")

print("\n4. A RENDER AT A DIFFERENT MASTER LEVEL")
quiet = make_track(Knobs(seed=SEED, duration_s=SECONDS, master_gain=0.5), MOOD)
back = replay(quiet.formula)
print(f"   original {quiet.fingerprint} (master 0.5)  replay {back.fingerprint} (default "
      f"{Knobs.master_gain})  {'MATCH' if quiet.fingerprint == back.fingerprint else 'DIFFERENT'}")

print("\n5. IS THE MEMORY EVEN RECOVERABLE FROM THE DIGEST?")
print(f"   the digest is blake2b over the tables, {len(memory.digest())} hex chars — "
      "a checksum, not the data. Nothing can invert it.")
