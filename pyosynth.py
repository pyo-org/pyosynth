"""
pyosynth -- a ``@synthdef``-like decorator for Pyo.

Write a plain function that builds a signal graph, decorate it with
``@pyosynth``, and get back a real ``PyoObject`` subclass (built the way the
"Creating a custom PyoObject" tutorial describes), so it supports ``mul`` /
``add``, list expansion, ``play()/out()/stop()``, attribute access and
``ctrl()`` like any built-in Pyo object.

    from pyo import *
    from pyosynth import pyosynth

    s = Server().boot().start()

    @pyosynth
    def sine(freq=440, amp=0.1):
        return SinOsc(freq, mul=amp)

    n = sine().out()          # like Synth('sine') in sc3
    n.set('amp', 0.05)        # or n.amp = 0.05 / n.setAmp(0.05)
    n.freq = 550

Parameter rules (the function's signature is the synth's interface):

* Parameter WITH a default  -> control. The function receives a ``Sig`` (or
  a ``SigTo`` if ``lag`` is given), so it can be changed at runtime. The
  value may be a float, a list (multichannel expansion) or any PyoObject
  (e.g. an LFO).
* Parameter WITHOUT a default -> audio input. The function receives an
  ``InputFader``, so ``set()`` cross-fades between old and new sources.
* Return a PyoObject (or a list/tuple of PyoObjects) as the output.
* ``mul`` and ``add`` are added to the signature automatically.
* ``Adsr`` and ``Fader`` objects inside the graph are treated as envelopes:
  ``play()``/``out()`` triggers them, and ``stop()`` releases them (the
  release is heard) and stops everything afterwards. Re-trigger a stopped
  synth with ``out()``, as with any Pyo object.
* To add your own methods, decorate a class instead: the graph goes in
  ``build(self, ...)`` (same parameter rules as above), and any other method
  or attribute becomes part of the synth. Methods like ``play``/``stop`` can
  be overridden and call ``super()``.

      @pyosynth
      class Pad:
          def build(self, freq=220, amp=0.5):
              self.adsr = Adsr(release=2, mul=amp)
              return SuperSaw(freq, mul=self.adsr)

          def note_off(self):
              self.adsr.stop()

  A plain function may also take ``self`` first, to keep handles on the instance.
"""

import inspect

from pyo import (Adsr, CallAfter, Fader, InputFader, PyoObject, Sig, SigTo,
                 SLMap, SLMapMul)

__all__ = ["pyosynth", "Synth"]

_registry = {}  # name -> synth class, so Synth('name') works like in sc3
_RESERVED = {"self", "mul", "add"}
# Objects that start an envelope on play() and release it on stop().
_GATED = (Adsr, Fader)


def _camel(name):
    return "".join(p[0].upper() + p[1:] for p in name.split("_") if p)


def _is_number(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _is_signal(x):
    """True if the value can drive a Sig: number, list of numbers, PyoObject."""
    if isinstance(x, PyoObject) or _is_number(x):
        return True
    return isinstance(x, (list, tuple)) and len(x) > 0 and all(
        _is_number(i) or isinstance(i, PyoObject) for i in x)


def _reachable(roots):
    """All PyoObjects reachable from `roots` through instance attributes."""
    seen = {}
    stack = list(roots)
    while stack:
        item = stack.pop()
        if isinstance(item, (list, tuple, set)):
            stack.extend(item)
        elif isinstance(item, dict):
            stack.extend(item.values())
        elif isinstance(item, PyoObject) and id(item) not in seen:
            seen[id(item)] = item
            stack.extend(vars(item).values())
    return seen


def _default_slmap(name, value):
    """Guess a sensible slider range from the default value."""
    if not _is_number(value):
        return None
    if value > 0:
        return SLMap(value / 10.0, value * 10.0, "log", name, value)
    if value < 0:
        return SLMap(value * 10.0, value / 10.0, "lin", name, value)
    return SLMap(-1.0, 1.0, "lin", name, value)


def _build(func, lag, ranges, name, user_cls=None):
    name = name or func.__name__
    sig = inspect.signature(func)
    inputs, controls = [], {}
    params = list(sig.parameters.values())
    wants_self = bool(params) and params[0].name == "self"
    if wants_self:
        params = params[1:]
    elif user_cls is not None:
        raise TypeError("%s.build must take `self` as its first parameter." % name)
    for p in params:
        if p.kind not in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY):
            raise TypeError("@pyosynth does not support *args/**kwargs "
                            "or positional-only parameters ('%s')." % p.name)
        if p.name in _RESERVED or p.name.startswith("_") or hasattr(PyoObject, p.name):
            raise ValueError("Parameter name '%s' is reserved by PyoObject." % p.name)
        if p.default is p.empty:
            inputs.append(p.name)
        else:
            controls[p.name] = p.default

    # Public signature: the function's parameters, plus mul and add.
    init_sig = inspect.Signature(
        [inspect.Parameter("self", inspect.Parameter.POSITIONAL_OR_KEYWORD)]
        + params
        + [inspect.Parameter("mul", inspect.Parameter.KEYWORD_ONLY, default=1),
           inspect.Parameter("add", inspect.Parameter.KEYWORD_ONLY, default=0)])

    def __init__(self, *args, **kwargs):
        bound = init_sig.bind(self, *args, **kwargs)
        bound.apply_defaults()
        values = dict(bound.arguments)
        values.pop("self")
        mul, add = values.pop("mul"), values.pop("add")

        # Step 2 of the tutorial: initialise PyoObject, keep the raw arguments.
        PyoObject.__init__(self, mul, add)
        self._values = dict(values)
        self._faders, self._sigs = {}, {}

        external = []  # objects owned by the caller; we must not play/stop them
        graph_args = {}
        for pname in inputs:
            self._faders[pname] = graph_args[pname] = InputFader(values[pname])
            external.append(values[pname])
        for pname, value in ((n, values[n]) for n in controls):
            if _is_signal(value):
                if lag and lag > 0 and not isinstance(value, PyoObject):
                    ctl = SigTo(value, time=lag, init=value)
                else:
                    ctl = Sig(value)
                self._sigs[pname] = graph_args[pname] = ctl
                external.append(value)
            else:
                graph_args[pname] = value  # constant (table, string, ...)

        # Build the user's graph.
        result = func(self, **graph_args) if wants_self else func(**graph_args)
        outs = list(result) if isinstance(result, (list, tuple)) else [result]
        if not outs or not all(isinstance(o, PyoObject) for o in outs):
            raise TypeError("A @pyosynth function must return a PyoObject "
                            "(or a list of PyoObjects), got %r." % (result,))

        # Sig as a "through" so mul/add never touch the user's objects.
        self._outs = [Sig(o, mul=mul, add=add) for o in outs]
        self._base_objs = [b for o in self._outs for b in o.getBaseObjects()]

        # Every object created inside the graph needs play()/stop()/out().
        roots = outs + list(self._sigs.values()) + list(self._faders.values())
        foreign = _reachable(external)
        internal = [o for k, o in _reachable(roots).items() if k not in foreign]
        # Adsr/Fader are triggered by play() and released by stop(), so they
        # are started last and released (not cut) on stop().
        self._gated = [o for o in internal if isinstance(o, _GATED)]
        self._internal = [o for o in internal if not isinstance(o, _GATED)]
        self._halt_id = 0
        self._halt_timer = None

    __init__.__signature__ = init_sig

    def set(self, pname, value, fadetime=0.05):
        """Set a control (or cross-fade to a new input) by name."""
        if pname in self._faders:
            self._faders[pname].setInput(value, fadetime)
        elif pname in self._sigs:
            if not _is_signal(value):
                raise TypeError("'%s' needs a number, list or PyoObject." % pname)
            self._sigs[pname].value = value
        elif pname in self._values:
            raise TypeError("'%s' is a constant and can't be changed." % pname)
        else:
            raise KeyError("No parameter named '%s'." % pname)
        self._values[pname] = value
        return self

    def get(self, pname):
        """Return the last raw value given for a parameter."""
        return self._values[pname]

    def _start(self, dur, delay):
        self._halt_id += 1  # cancels a pending stop() from an earlier release
        for o in self._internal:
            o.play(dur, delay)
        for o in self._gated:  # trigger the envelopes last
            o.play(dur, delay)

    def _halt(self, token):
        if token == self._halt_id:  # not restarted in the meantime
            for o in self._internal:
                o.stop()
            PyoObject.stop(self)

    def play(self, dur=0, delay=0):
        _start(self, dur, delay)
        return PyoObject.play(self, dur, delay)

    def stop(self, wait=0):
        if not self._gated:
            for o in self._internal:
                o.stop(wait)
            return PyoObject.stop(self, wait)
        # Release the envelopes, then stop everything once the longest
        # release (Adsr.release / Fader.fadeout) has finished.
        release = 0.0
        for o in self._gated:
            o.stop(wait)
            release = max(release, getattr(o, "release", None)
                          or getattr(o, "fadeout", 0) or 0)
        self._halt_id += 1
        self._halt_timer = CallAfter(self._halt, wait + release + 0.05,
                                     arg=self._halt_id)
        return self

    def out(self, chnl=0, inc=1, dur=0, delay=0):
        _start(self, dur, delay)
        return PyoObject.out(self, chnl, inc, dur, delay)

    def ctrl(self, map_list=None, title=None, wxnoserver=False):
        if map_list is None:
            maps = []
            for pname in self._sigs:
                m = (ranges or {}).get(pname)
                if m is not None:
                    maps.append(SLMap(m[0], m[1], m[2] if len(m) > 2 else "lin",
                                      pname, self._values[pname]))
                else:
                    m = _default_slmap(pname, self._values[pname])
                    if m is not None:
                        maps.append(m)
            maps.append(SLMapMul(self._mul))
            self._map_list = maps
        PyoObject.ctrl(self, map_list, title, wxnoserver)

    # Step 3 of the tutorial: setXXX methods and properties, one per parameter.
    def make_setter(pname):
        def setter(self, x, *a, **k):
            return self.set(pname, x, *a, **k)
        setter.__name__ = "set" + _camel(pname)
        setter.__doc__ = "Replace the `%s` attribute." % pname
        return setter

    def make_property(pname):
        return property(lambda self: self._values[pname],
                        lambda self, x: self.set(pname, x),
                        doc="Parameter `%s`." % pname)

    doc = (inspect.getdoc(user_cls) if user_cls else None) or inspect.getdoc(func) or ""
    doc += "\n\n    :Args:\n\n" + "\n".join(
        "        %s : %s" % (n, "audio input" if n in inputs
                             else "control, default %r" % (controls[n],))
        for n in list(inputs) + list(controls))

    ns = {"__doc__": doc, "__module__": func.__module__,
          "__init__": __init__, "_start": _start, "_halt": _halt,
          "set": set, "get": get, "play": play,
          "stop": stop, "out": out, "ctrl": ctrl,
          "_param_names": tuple(inputs) + tuple(controls)}
    for pname in list(inputs) + list(controls):
        ns["set" + _camel(pname)] = make_setter(pname)
        ns[pname] = make_property(pname)

    cls = type(name, (PyoObject,), ns)
    if user_cls is not None:
        # The user's class comes first in the MRO, so its methods override
        # the generated ones and `super()` reaches them.
        cls = type(name, (user_cls, cls),
                   {"__module__": user_cls.__module__, "__doc__": doc})
    _registry[name] = cls
    return cls


def pyosynth(func=None, *, lag=0.0, ranges=None, name=None):
    """
    Turn a function that builds a Pyo graph into a PyoObject subclass.

    Usable bare (``@pyosynth``) or with options:

    :Args:

        lag : float, optional
            If > 0, numeric controls are smoothed with a SigTo of this time
            (seconds) so ``set()`` glides instead of jumping. Defaults to 0.
        ranges : dict, optional
            Slider ranges for ``ctrl()``: ``{'freq': (20, 2000, 'log')}``.
        name : str, optional
            Class / registry name. Defaults to the function's name.
    """
    def decorate(f):
        if inspect.isclass(f):
            if "build" not in vars(f):
                raise TypeError("A @pyosynth class needs a `build(self, ...)` method.")
            if "__init__" in vars(f):
                raise TypeError("A @pyosynth class can't define __init__; use build().")
            return _build(vars(f)["build"], lag, ranges, name or f.__name__, f)
        return _build(f, lag, ranges, name)
    return decorate if func is None else decorate(func)


def Synth(name, *args, **kwargs):
    """Instantiate a registered @pyosynth by name, like ``Synth('sine')`` in sc3."""
    try:
        return _registry[name](*args, **kwargs)
    except KeyError:
        raise KeyError("No @pyosynth named '%s' (known: %s)."
                       % (name, ", ".join(sorted(_registry)) or "none")) from None
