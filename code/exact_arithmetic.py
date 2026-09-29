#!/usr/bin/env python3
"""Generic exact rational helpers used by the k-point certificate verifier.

The helpers retain the established arithmetic formulas and SDPA reader.
Original filenames and subsequent changes are recorded in extras/SOURCE_HISTORY.md.
"""
import sys
from fractions import Fraction as F
from itertools import combinations
from functools import lru_cache
from pathlib import Path
from decimal import Decimal, localcontext
import re

if hasattr(sys, "set_int_max_str_digits"):
    sys.set_int_max_str_digits(0)


def require(condition, message):
    if not condition:
        raise ValueError(message)


@lru_cache(None)
def edges(n):
    return tuple(combinations(range(n), 2))


def inverse(A):
    n = len(A)
    T = [r[:] + [F(i == j) for j in range(n)] for i, r in enumerate(A)]
    for k in range(n):
        p = next(i for i in range(k, n) if T[i][k])
        T[k], T[p] = T[p], T[k]
        q = T[k][k]
        T[k] = [v / q for v in T[k]]
        for i in range(n):
            if i != k:
                q = T[i][k]
                T[i] = [v - q * w for v, w in zip(T[i], T[k])]
    return [r[n:] for r in T]


def bilinear(u, H, v):
    return sum((u[i] * H[i][j] * v[j]
                for i in range(len(u)) for j in range(len(v))), F(0))


def homogeneous_kernel(l, n, m, t, u, v, H):
    z = t - bilinear(u, H, v)
    s = (1 - bilinear(u, H, u)) * (1 - bilinear(v, H, v))
    if l == 0:
        return F(1)
    h0, h1 = F(1), z
    for j in range(2, l + 1):
        h0, h1 = h1, F(2*j + n-m-4, j+n-m-3)*z*h1 - F(j-1, j+n-m-3)*s*h0
    return h1


def read_sdpa(path):
    text = Path(path).read_text()
    found = re.search(r'^yMat\s*=\s*', text, re.M)
    require(found is not None, 'SDPA output contains no yMat')
    rest = text[found.end():]
    start = rest.find('{')
    require(start >= 0, 'missing yMat braces')
    depth, end = 0, None
    for i, char in enumerate(rest[start:], start):
        if char == '{': depth += 1
        elif char == '}':
            depth -= 1
            if depth == 0:
                end = i+1
                break
    require(end is not None, 'unterminated yMat')
    raw = rest[start:end]
    tokens = re.findall(r'[{},]|[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?',raw)
    require(re.sub(r'\s+', '',raw) == ''.join(tokens), 'unexpected token in yMat')
    pos = 0
    def parse():
        nonlocal pos
        tok = tokens[pos]
        pos += 1
        if tok != '{': return F(tok)
        out=[]
        while tokens[pos] != '}':
            if tokens[pos] == ',': pos += 1
            else: out.append(parse())
        pos += 1
        return out
    mats = parse()
    require(pos == len(tokens), 'extra matrix tokens')
    return [[[x] for x in M] if M and isinstance(M[0],F) else M for M in mats]


def brief(q):
    with localcontext() as ctx:
        ctx.prec = 18
        return str(Decimal(q.numerator)/Decimal(q.denominator))
