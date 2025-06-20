import builtins
import pickle

import numpy as np

from . import space as spacelib

try:
  import rich.console
  console = rich.console.Console()
except ImportError:
  console = None


CONVERSION = {
    np.floating: np.float32,
    np.signedinteger: np.int64,
    np.uint8: np.uint8,
    bool: bool,
}


def convert(value):
  value = np.asarray(value)
  if value.dtype not in CONVERSION.values():
    for src, dst in CONVERSION.items():
      if np.issubdtype(value.dtype, src):
        if value.dtype != dst:
          value = value.astype(dst)
        break
    else:
      raise TypeError(f"Object '{value}' has unsupported dtype: {value.dtype}")
  return value


def print_(value, color=None):
  global console
  value = format_(value)
  if console:
    if color:
      value = f'[{color}]{value}[/{color}]'
    console.print(value)
  else:
    builtins.print(value)


def format_(value):
  if isinstance(value, dict):
    if value and all(isinstance(x, spacelib.Space) for x in value.values()):
      return '\n'.join(f'  {k:<16} {v}' for k, v in value.items())
    items = [f'{format_(k)}: {format_(v)}' for k, v in value.items()]
    return '{' + ', '.join(items) + '}'
  if isinstance(value, list):
    return '[' + ', '.join(f'{format_(x)}' for x in value) + ']'
  if isinstance(value, tuple):
    return '(' + ', '.join(f'{format_(x)}' for x in value) + ')'
  if hasattr(value, 'shape') and hasattr(value, 'dtype'):
    shape = ','.join(str(x) for x in value.shape)
    dtype = value.dtype.name
    for long, short in {'float': 'f', 'uint': 'u', 'int': 'i'}.items():
      dtype = dtype.replace(long, short)
    return f'{dtype}[{shape}]'
  if isinstance(value, bytes):
    value = '0x' + value.hex() if r'\x' in str(value) else str(value)
    if len(value) > 32:
      value = value[:32 - 3] + '...'
  return str(value)


def treemap(fn, *trees, isleaf=None):
  assert trees, 'Provide one or more nested Python structures'
  kw = dict(isleaf=isleaf)
  first = trees[0]
  assert all(isinstance(x, type(first)) for x in trees)
  if isleaf and isleaf(trees):
    return fn(*trees)
  if isinstance(first, list):
    assert all(len(x) == len(first) for x in trees), format_(trees)
    return [treemap(
        fn, *[t[i] for t in trees], **kw) for i in range(len(first))]
  if isinstance(first, tuple):
    assert all(len(x) == len(first) for x in trees), format_(trees)
    return tuple([treemap(
        fn, *[t[i] for t in trees], **kw) for i in range(len(first))])
  if isinstance(first, dict):
    assert all(set(x.keys()) == set(first.keys()) for x in trees), (
        format_(trees))
    return {k: treemap(fn, *[t[k] for t in trees], **kw) for k in first}
  return fn(*trees)


def pack(data):
  return pickle.dumps(data)


def unpack(buffer):
  return pickle.loads(buffer)
