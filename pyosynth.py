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
"""

import inspect

from pyo import InputFader, PyoObject, Sig, SigTo, SLMap, SLMapMul

__all__ = ["pyosynth", "Synth"]

_registry = {}  # name -> synth class, so Synth('name') works like in sc3
_RESERVED = {"self", "mul", "add"}


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


def _build(func, lag, ranges, name):
    sig = inspect.signature(func)
    inputs, controls = [], {}
    for p in sig.parameters.values():
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
        + list(sig.parameters.values())
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
        result = func(**graph_args)
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
        self._internal = [o for k, o in _reachable(roots).items() if k not in foreign]

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

    def play(self, dur=0, delay=0):
        for o in self._internal:
            o.play(dur, delay)
        return PyoObject.play(self, dur, delay)

    def stop(self, wait=0):
        for o in self._internal:
            o.stop(wait)
        return PyoObject.stop(self, wait)

    def out(self, chnl=0, inc=1, dur=0, delay=0):
        for o in self._internal:
            o.play(dur, delay)
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

    doc = inspect.getdoc(func) or ""
    doc += "\n\n    :Args:\n\n" + "\n".join(
        "        %s : %s" % (n, "audio input" if n in inputs
                             else "control, default %r" % (controls[n],))
        for n in list(inputs) + list(controls))

    ns = {"__doc__": doc, "__module__": func.__module__, "__wrapped__": func,
          "__init__": __init__, "set": set, "get": get, "play": play,
          "stop": stop, "out": out, "ctrl": ctrl,
          "_param_names": tuple(inputs) + tuple(controls)}
    for pname in list(inputs) + list(controls):
        ns["set" + _camel(pname)] = make_setter(pname)
        ns[pname] = make_property(pname)

    cls = type(name or func.__name__, (PyoObject,), ns)
    _registry[name or func.__name__] = cls
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
        return _build(f, lag, ranges, name)
    return decorate if func is None else decorate(func)


def Synth(name, *args, **kwargs):
    """Instantiate a registered @pyosynth by name, like ``Synth('sine')`` in sc3."""
    try:
        return _registry[name](*args, **kwargs)
    except KeyError:
        raise KeyError("No @pyosynth named '%s' (known: %s)."
                       % (name, ", ".join(sorted(_registry)) or "none")) from None
