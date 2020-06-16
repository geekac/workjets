#!/usr/bin/env python
# -*- coding: utf-8 -*-


import platform

__version__ = '1.0.1'
VERSION = tuple(map(int, __version__.split('.')))

__all__ = ('Bunch', 'bunchify', 'unbunchify',)

_IS_PYTHON_3 = (platform.python_version() >= '3')
identity = lambda x: x

# u('string') replaces the forwards-incompatible u'string'
if _IS_PYTHON_3:
    u = identity
else:
    import codecs


    def u(string):
        return codecs.unicode_escape_decode(string)[0]

# dict.iteritems(), dict.iterkeys() is also incompatible
if _IS_PYTHON_3:
    iteritems = dict.items
    iterkeys = dict.keys
else:
    iteritems = dict.iteritems
    iterkeys = dict.iterkeys


class Bunch(dict):

    def __contains__(self, k):
        try:
            return dict.__contains__(self, k) or hasattr(self, k)
        except:
            return False

    # only called if k not found in normal places
    def __getattr__(self, k):
        try:
            # Throws exception if not in prototype chain
            return object.__getattribute__(self, k)
        except AttributeError:
            try:
                return self[k]
            except KeyError:
                raise AttributeError(k)

    def __setattr__(self, k, v):
        try:
            # Throws exception if not in prototype chain
            object.__getattribute__(self, k)
        except AttributeError:
            try:
                self[k] = v
            except:
                raise AttributeError(k)
        else:
            object.__setattr__(self, k, v)

    def __delattr__(self, k):
        try:
            # Throws exception if not in prototype chain
            object.__getattribute__(self, k)
        except AttributeError:
            try:
                del self[k]
            except KeyError:
                raise AttributeError(k)
        else:
            object.__delattr__(self, k)

    def toDict(self):
        return unbunchify(self)

    def __repr__(self):
        keys = list(iterkeys(self))
        keys.sort()
        args = ', '.join(['%s=%r' % (key, self[key]) for key in keys])
        return '%s(%s)' % (self.__class__.__name__, args)

    @staticmethod
    def fromDict(d):
        return bunchify(d)


def bunchify(x):
    if isinstance(x, dict):
        return Bunch((k, bunchify(v)) for k, v in iteritems(x))
    elif isinstance(x, (list, tuple)):
        return type(x)(bunchify(v) for v in x)
    else:
        return x


def unbunchify(x):
    if isinstance(x, dict):
        return dict((k, unbunchify(v)) for k, v in iteritems(x))
    elif isinstance(x, (list, tuple)):
        return type(x)(unbunchify(v) for v in x)
    else:
        return x


# Serialization

try:
    try:
        import json
    except ImportError:
        import simplejson as json


    def toJSON(self, **options):

        return json.dumps(self, **options)


    Bunch.toJSON = toJSON

except ImportError:
    pass

try:
    # Attempt to register ourself with PyYAML as a representer
    import yaml
    from yaml.representer import Representer, SafeRepresenter


    def from_yaml(loader, node):

        data = Bunch()
        yield data
        value = loader.construct_mapping(node)
        data.update(value)


    def to_yaml_safe(dumper, data):
        return dumper.represent_dict(data)


    def to_yaml(dumper, data):

        return dumper.represent_mapping(u('!bunch.Bunch'), data)


    yaml.add_constructor(u('!bunch'), from_yaml)
    yaml.add_constructor(u('!bunch.Bunch'), from_yaml)

    SafeRepresenter.add_representer(Bunch, to_yaml_safe)
    SafeRepresenter.add_multi_representer(Bunch, to_yaml_safe)

    Representer.add_representer(Bunch, to_yaml)
    Representer.add_multi_representer(Bunch, to_yaml)


    # Instance methods for YAML conversion
    def toYAML(self, **options):

        opts = dict(indent=4, default_flow_style=False)
        opts.update(options)
        if 'Dumper' not in opts:
            return yaml.safe_dump(self, **opts)
        else:
            return yaml.dump(self, **opts)


    def fromYAML(*args, **kwargs):
        return bunchify(yaml.load(*args, **kwargs))


    Bunch.toYAML = toYAML
    Bunch.fromYAML = staticmethod(fromYAML)

except ImportError:
    pass
