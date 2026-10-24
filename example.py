"""
Envelope-gated synth with @pyosynth: Adsr, release on stop(), custom methods.
"""

from pyo import *
from pyosynth import pyosynth

s = Server().boot()


# 1) Adsr inside the graph: play()/out() triggers the attack, stop() triggers
#    the release, and the synth stops itself once the release has finished.
@pyosynth
def supersaw(freq=440, amp=0.5):
    adsr = Adsr(attack=0.01, decay=0.1, sustain=0.7, release=1.0, mul=amp)
    return SuperSaw(freq=freq, mul=adsr)


saw = supersaw(mul=0.3).out()   # attack starts now
# saw.freq = 330                # change a control while it plays
# saw.stop()                    # release is heard, then it stops
# saw.out()                     # trigger again (play() alone gives no output)


# 2) Custom methods in one declaration: decorate a class. The graph goes in
#    build(self, ...), with the same parameter rules as a decorated function;
#    every other method belongs to the synth and may override play/stop/out
#    (call super() to keep the default behavior).
@pyosynth
class Pad:
    """Slow pad with explicit note on/off."""

    def build(self, freq=220, amp=0.5):
        self.adsr = Adsr(attack=0.5, decay=0.2, sustain=0.8, release=2.0, mul=amp)
        return SuperSaw(freq=freq, mul=self.adsr)

    def note_on(self):
        self.adsr.play()
        return self

    def note_off(self):
        self.adsr.stop()
        return self


# p = Pad(mul=0.3).out()   # .out() already triggers the attack
# p.note_off()             # release only, synth keeps running silently
# p.note_on()              # re-trigger the envelope

s.gui(locals())
