# pyosynth
Python decorator for creating Pyo synths fast.
Needs the Pyo module installed (https://github.com/pyo-org/pyo)

Example usage:

    from pyo import *
    from pyosynth import pyosynth

    @pyosynth
    def sine(freq=440, amp=0.10):
        return Sine(freq, mul=amp)

    sin = sine.out()
    sin.set('amp', 0.5)
    sin.freq(660)
