"""Attribute-access dict used by the samplers.
"""


class AttrDict(dict):
    """dict whose keys are also readable/writable as attributes (``d.samples``)."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as e:
            raise AttributeError(name) from e

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        try:
            del self[name]
        except KeyError as e:
            raise AttributeError(name) from e
