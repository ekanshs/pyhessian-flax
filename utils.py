from absl import logging
import operator
from functools import reduce
from typing import Sequence

import jax
import jax.numpy as jnp
from jax import random
from jax.tree_util import tree_reduce, tree_flatten, tree_map

from flax import traverse_util
from flax.core import freeze, unfreeze


TINY = 1e-6

def tree_zeros_like(t):
  return tree_map(lambda x: jnp.zeros_like(x), t)

def tree_add(t1, t2):
  return tree_map(lambda x,y: x+y, t1, t2)

def tree_subtract(t1, t2):
  return tree_map(lambda x,y: x-y, t1, t2)

def tree_multiply(t1, t2):
  return tree_map(lambda l1, l2: l1*l2, t1, t2)

def tree_divide(t1, t2):
  return tree_map(lambda l1, l2: l1 / l2, t1, t2)

def tree_square(t):
  return tree_map(lambda x: jnp.square(x), t)

def tree_scalar_multiply(c, t):
  return tree_map(lambda x: c*x , t)

def tree_norm(t):
  return jnp.sqrt(tree_reduce(operator.add, tree_map(lambda x: jnp.sum(x**2), t)))

def tree_sum(t):
  return tree_reduce(operator.add, tree_map(lambda x: jnp.sum(x), t))

def normalize_tree(t):
  t_norm = tree_norm(t)
  return (tree_scalar_multiply(1/t_norm, t), t_norm)

def add_trees(ts, ws=None):
  if ws is None:
      return tree_map(lambda *v: reduce(lambda x,y: x+y, v), *ts)
  ts = [tree_scalar_multiply(w, t) for w,t in zip(ws, ts)]
  return tree_map(lambda *v: reduce(lambda x,y: x+y, v), *ts)

def tree_sign(t):
  return tree_map(lambda x: jnp.sign(x) , t)

def tree_inner_prod(t1, t2):
  return tree_sum(tree_multiply(t1, t2))

def tree_count(t):
  return sum(x.size for x in jax.tree_leaves(t))

def cosine_similarity(t1,t2):
  return tree_inner_prod(t1, t2) / (tree_norm(t1)*tree_norm(t2))

def normalize_tree(t):
  t_norm = tree_norm(t)
  return (tree_scalar_multiply(1/(t_norm + TINY), t), t_norm)


def normal_tree_like(rng, t):
  return tree_map(lambda x: random.normal(rng, shape=x.shape), t)

def rademacher_tree_like(rng, t):
  return tree_map(lambda x: random.rademacher(rng, shape=x.shape), t)


def orthnormal(t, ts):
    """
    make vector t orthogonal to each vector in ts.
    afterwards, normalize the output w
    """
    for t_ in ts:
        t = tree_add(t, tree_scalar_multiply(-tree_inner_prod(t, t_), t_))
    return normalize_tree(t)