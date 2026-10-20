"""
A triggered oscillator built with @pyosynth, in plain Pyo style.

A Metro sends a trigger every 0.5 s. Each trigger starts a TrigEnv that
shapes a sine wave, and a TrigChoice picks a new pitch on every trigger.
"""

from pyo import *
from pyosynth import pyosynth

s = Server().boot()


@pyosynth
def blip(trig, freq=440, dur=0.3):
    """Sine burst shaped by a TrigEnv each time `trig` fires."""
    # `trig` has no default value, so it is an audio input: the function
    # receives it wrapped in an InputFader (cross-fade when it is replaced).
    # `freq` and `dur` have defaults, so they are controls: they can be
    # floats, lists, or other PyoObjects.
    table = CosTable([(0, 0), (64, 1), (8191, 0)])  # fast attack, long decay
    env = TrigEnv(trig, table=table, dur=dur)
    return Sine(freq, mul=env)


# Periodic trigger source. Metro has to be started explicitly.
met = Metro(time=0.5).play()

# A new random pitch on each trigger, used directly as the `freq` control.
pitch = TrigChoice(met, choice=[220, 330, 440, 660])

# Create the synth and send it to the output. mul/add work as on any PyoObject.
tone = blip(met, freq=pitch, dur=0.4, mul=0.3).out()

# Everything stays live: change parameters with attributes or setXXX methods.
#   met.time = 0.25            # faster rhythm
#   tone.dur = 0.15            # shorter bursts
#   tone.freq = [440, 445]     # list expansion: two detuned voices (stereo)
#   tone.input = Metro(0.3).play()   # cross-fade to another trigger source
# Open the control window with: tone.ctrl()

# The generated object can be passed to other PyoObjects
sc = Scope(tone)

s.gui(locals())
