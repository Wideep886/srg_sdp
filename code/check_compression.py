#!/usr/bin/env python3
"""Independent exact check of compressed degree-zero union coefficients.

This file imports no project reconstruction or compression routines.  It never
forms C, J, or J^T A J.  Instead it enumerates every reference subset and every
ordering with the prescribed reference Gram matrix, and directly applies the
three union cases to compressed scalar matrix coordinates.

TSV convention: constraint, block, row, column, exact scalar coefficient;
all four indices are 1-based, row <= column.  Scalar coefficients already
include both off-diagonal trace contributions.  For nonempty references, the
first compressed coordinate is reference_sum; the remaining coordinates are
the original nonreference labels in their original order.  Block IDs do not
change.  Empty-reference blocks keep their single empty-label coordinate.

The k<=6 copy enumerates graph types independently by breadth-first orbit
flooding under adjacent vertex swaps. Every labeled mask is visited, including
singular PSD Gram configurations. No project graph enumerator is imported.
"""

from __future__ import annotations

import argparse
from collections import defaultdict, deque
from fractions import Fraction as F
from functools import lru_cache
import hashlib
from itertools import combinations, permutations, product
import json
from pathlib import Path
import sys


def require(condition, message):
    if not condition:
        raise ValueError(message)


def rational(value):
    require(isinstance(value, (str, int, F)) and not isinstance(value, bool),
            "rational input must be a string, integer, or Fraction")
    return F(value)


def integer(value, name):
    value = rational(value)
    require(value.denominator == 1, f"{name} must be an integer")
    return value.numerator


def matrix(raw):
    require(isinstance(raw, list), "matrix must be a list")
    require(all(isinstance(row, list) for row in raw), "matrix rows must be lists")
    return tuple(tuple(rational(x) for x in row) for row in raw)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    def distinct(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, 'duplicate JSON key: ' + key)
            result[key] = value
        return result

    def reject_constant(value):
        raise ValueError('invalid JSON numeric constant: ' + value)

    return json.loads(Path(path).read_text(), parse_float=str,
                      object_pairs_hook=distinct, parse_constant=reject_constant)


def psd_rank(A):
    """Independent rational symmetric elimination, including zero pivots."""
    n = len(A)
    require(all(len(row) == n for row in A), "nonsquare matrix")
    require(all(A[i][j] == A[j][i] for i in range(n) for j in range(n)),
            "nonsymmetric matrix")
    B = [list(row) for row in A]
    rank = 0
    for j in range(n):
        pivot = B[j][j]
        if pivot < 0:
            return None
        if not pivot:
            if any(B[i][j] for i in range(j + 1, n)):
                return None
            continue
        rank += 1
        for i in range(j + 1, n):
            for h in range(i, n):
                B[i][h] -= B[i][j] * B[j][h] / pivot
                B[h][i] = B[i][h]
    return rank


@lru_cache(None)
def edge_list(m):
    return tuple(combinations(range(m), 2))


def gram_from_mask(m, mask, a, b):
    G = [[F(i == j) for j in range(m)] for i in range(m)]
    for bit, (i, j) in enumerate(edge_list(m)):
        G[i][j] = G[j][i] = b if (mask >> bit) & 1 else a
    return tuple(map(tuple, G))


@lru_cache(None)
def flood_orbit_cover(m):
    """Independent full coverage using adjacent transpositions as generators.

    Adjacent vertex swaps generate every vertex permutation. Starting at the
    least unvisited mask and closing under each swap therefore visits exactly
    its full graph orbit. A scan of all masks makes coverage exhaustive. This
    differs from the complete-permutation image method in the exact verifier.
    """
    require(isinstance(m, int) and 0 <= m <= 6, "graph size outside 0..6")
    edges = edge_list(m)
    positions = {edge: bit for bit, edge in enumerate(edges)}
    generators = []
    for vertex in range(m - 1):
        p = list(range(m))
        p[vertex], p[vertex + 1] = p[vertex + 1], p[vertex]
        # For every output bit, record the input bit under this involution.
        generators.append(tuple(positions[tuple(sorted((p[i], p[j])))]
                                for i, j in edges))
    count = 1 << len(edges)
    leaders = [-1] * count
    roots, sizes = [], []
    for seed in range(count):
        if leaders[seed] != -1:
            continue
        leaders[seed] = seed
        pending = deque([seed])
        size = 0
        while pending:
            mask = pending.popleft()
            size += 1
            for source_bits in generators:
                image = sum(((mask >> source) & 1) << out
                            for out, source in enumerate(source_bits))
                require(image >= seed, "orbit flood found a smaller unseen representative")
                previous = leaders[image]
                if previous == -1:
                    leaders[image] = seed
                    pending.append(image)
                else:
                    require(previous == seed, "orbit flood crossed two representatives")
        roots.append(seed)
        sizes.append(size)
    require(all(root >= 0 for root in leaders) and sum(sizes) == count,
            "orbit flood did not cover every labeled mask")
    return tuple(roots), tuple(leaders), tuple(sizes)


def canonical_mask(m, mask):
    leaders = flood_orbit_cover(m)[1]
    require(isinstance(mask, int) and 0 <= mask < len(leaders), "invalid edge mask")
    return leaders[mask]


def gram_type(G, a, b):
    m = len(G)
    require(all(len(row) == m for row in G), "invalid Gram shape")
    require(all(G[i][i] == 1 for i in range(m)), "Gram diagonal is not one")
    require(all(G[i][j] == G[j][i] and G[i][j] in (a, b)
                for i, j in edge_list(m)), "invalid off-diagonal Gram entry")
    mask = sum((G[i][j] == b) << bit for bit, (i, j) in enumerate(edge_list(m)))
    return m, canonical_mask(m, mask)


def extended_gram(G, u):
    return tuple(tuple(G[i]) + (u[i],) for i in range(len(G))) + (tuple(u) + (F(1),),)


def validate_metadata(meta):
    """Check orbit completeness and coordinate conventions independently."""
    n, k, d = (integer(meta[key], key) for key in ("n", "k", "d"))
    a, b = rational(meta["a"]), rational(meta["b"])
    require(2 <= k <= min(6, n) and d >= 0, "unsupported n,k,d")
    require(-1 <= a < 1 and -1 <= b < 1 and a != b, "invalid two distances")
    expected = {}
    for m in range(k + 1):
        types = flood_orbit_cover(m)[0]
        feasible = {}
        for mask in types:
            G = gram_from_mask(m, mask, a, b)
            rank = psd_rank(G)
            if rank is not None and rank <= n:
                feasible[mask] = (G, rank)
        expected[m] = feasible
    policy = meta.get("dependent_reference_policy", "reject")
    require(policy in ("reject", "zero"), "unknown dependent reference policy")
    dependent = {(m, mask) for m in range(k - 1)
                 for mask, (_, rank) in expected[m].items() if rank < m}
    require(not dependent or policy == "zero",
            "dependent reference configurations require explicit zero policy")
    active = {(m, mask) for m in range(k - 1)
              for mask, (_, rank) in expected[m].items() if rank == m}

    blocks = meta["blocks"]
    require(isinstance(blocks, list), "missing blocks")
    keys, zero_blocks, dimensions = set(), {}, {}
    for bi, spec in enumerate(blocks, 1):
        dim = integer(spec["dim"], "dim")
        require(dim >= 0, "negative block dimension")
        dimensions[bi] = dim
        if spec["kind"] != "kernel":
            require(spec["kind"] in ("alpha", "slack") and dim == 1,
                    "invalid nonkernel block")
            continue
        G = matrix(spec["gram"])
        m, mask = gram_type(G, a, b)
        ell = integer(spec["l"], "l")
        require(m == integer(spec["m"], "m") and 0 <= m <= k - 2,
                "reference size mismatch")
        require((m, mask) in active and 0 <= ell <= d, "invalid reference or degree")
        key = (m, mask, ell)
        require(key not in keys, "duplicate reference-degree block")
        keys.add(key)
        labels = [tuple(rational(x) for x in u) for u in spec["states"]]
        require(len(labels) == dim and len(set(labels)) == dim,
                "duplicate labels or dimension mismatch")
        require(all(len(u) == m for u in labels), "wrong label length")
        refs = {tuple(G[i][j] for i in range(m)) for j in range(m)}
        generic, strict_generic = set(), set()
        for u in product((a, b), repeat=m):
            rank = psd_rank(extended_gram(G, u))
            if rank is not None:
                generic.add(u)
                if rank == m + 1:
                    strict_generic.add(u)
        if ell == 0:
            require(set(labels) == refs | generic, "degree-zero labels incomplete")
            nonrefs = [u for u in labels if u not in refs]
            if m:
                index = {u: i + 2 for i, u in enumerate(nonrefs)}
                compressed_dim = len(nonrefs) + 1
            else:
                require(labels == [()], "empty reference must have one empty label")
                index, compressed_dim = {(): 1}, 1
            zero_blocks[bi] = {"G": G, "m": m, "index": index,
                               "compressed_dim": compressed_dim}
            dimensions[bi] = compressed_dim
        else:
            require(strict_generic <= set(labels) <= generic,
                    "invalid positive-degree label list")
    require(keys == {(m, mask, ell) for m, mask in active
                     for ell in range(d + 1)},
            "reference-degree list incomplete")
    alpha = integer(meta["alpha_block"], "alpha block")
    require(1 <= alpha <= len(blocks) and blocks[alpha - 1]["kind"] == "alpha",
            "invalid alpha block")
    require(sum(spec["kind"] == "alpha" for spec in blocks) == 1,
            "alpha block must be unique")
    seen, constraints, slacks = set(), [], set()
    for ci, spec in enumerate(meta["constraints"], 1):
        S = matrix(spec["gram"])
        size, mask = gram_type(S, a, b)
        require(size == integer(spec["size"], "size") and 1 <= size <= k,
                "invalid configuration size")
        require(mask in expected[size] and (size, mask) not in seen,
                "unrealizable or duplicate configuration")
        seen.add((size, mask))
        require(rational(spec["rhs"]) == (1 if size == 1 else 2 if size == 2 else 0),
                "unexpected residual constant")
        slack = integer(spec["slack_block"], "slack block")
        require(1 <= slack <= len(blocks) and blocks[slack - 1]["kind"] == "slack"
                and slack not in slacks, "invalid or duplicate slack")
        slacks.add(slack)
        constraints.append(S)
    require(seen == {(m, mask) for m in range(1, k + 1) for mask in expected[m]},
            "configuration orbit list incomplete")
    require(slacks == {bi for bi, spec in enumerate(blocks, 1) if spec["kind"] == "slack"},
            "unused slack block")
    return zero_blocks, dimensions, constraints, {
        "n": n, "k": k, "d": d, "a": str(a), "b": str(b),
        "orbit_counts": {str(m): len(expected[m]) for m in expected},
        "graph_orbit_enumeration": "adjacent-swap breadth-first orbit flood",
        "all_graph_orbit_counts": {str(m): len(flood_orbit_cover(m)[0]) for m in expected},
        "labeled_mask_counts": {str(m): len(flood_orbit_cover(m)[1]) for m in expected},
        "singular_configuration_counts": {
            str(m): sum(rank < m for _, rank in expected[m].values())
            for m in range(1, k + 1)},
        "zero_dependent_references": sorted(dependent),
        "coefficient_blocks": len(keys), "degree_zero_blocks": len(zero_blocks),
        "configuration_count": len(constraints),
    }


def build_direct_zero(meta):
    """Return scalar coefficients keyed (constraint,block,row,column)."""
    zero_blocks, dimensions, constraints, scope = validate_metadata(meta)
    terms = defaultdict(F)
    case_counts = {"no_external_point": 0, "one_external_point": 0,
                   "two_external_points": 0}
    ordered_reference_count = 0
    for ci, S in enumerate(constraints, 1):
        size = len(S)
        for bi, spec in zero_blocks.items():
            m, G, index = spec["m"], spec["G"], spec["index"]
            if size < m or size > m + 2:
                continue
            for Q in combinations(range(size), m):
                # Summing every Gram-preserving ordering independently realizes
                # the reference automorphism average without constructing it.
                orderings = [r for r in permutations(Q)
                             if all(S[r[i]][r[j]] == G[i][j]
                                    for i in range(m) for j in range(m))]
                if not orderings:
                    continue
                outside = tuple(i for i in range(size) if i not in Q)
                case_name = ("no_external_point", "one_external_point",
                             "two_external_points")[len(outside)]
                case_counts[case_name] += 1
                ordered_reference_count += len(orderings)
                weight = F(1, len(orderings))
                for r in orderings:
                    if not outside:
                        # All ordered reference-point pairs: coefficient 1 of
                        # the single reference_sum diagonal variable.
                        require(m > 0, "empty union is not a constraint")
                        terms[ci, bi, 1, 1] += weight
                    elif len(outside) == 1:
                        u = tuple(S[i][outside[0]] for i in r)
                        require(u in index, "missing generic label in one-point case")
                        iu = index[u]
                        terms[ci, bi, iu, iu] += weight
                        if m:
                            # Selected (reference,external) and its reverse.
                            terms[ci, bi, 1, iu] += 2 * weight
                    else:
                        u = tuple(S[i][outside[0]] for i in r)
                        v = tuple(S[i][outside[1]] for i in r)
                        require(u in index and v in index,
                                "missing generic label in two-point case")
                        i, j = sorted((index[u], index[v]))
                        # Still twice when u=v: two distinct selected points.
                        terms[ci, bi, i, j] += 2 * weight
    scope.update({"matrix_checks": len(constraints) * len(zero_blocks),
                  "union_cases": case_counts,
                  "ordered_reference_realizations": ordered_reference_count,
                  "nonzero_direct_scalar_coefficients": len(terms)})
    return dict(terms), zero_blocks, dimensions, scope


def read_tsv(path, dimensions, constraint_count):
    values, seen = {}, set()
    for lineno, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        parts = line.split()
        require(len(parts) == 5, f"{path}:{lineno}: expected five fields")
        ci, bi, i, j = map(int, parts[:4])
        require(1 <= ci <= constraint_count and bi in dimensions,
                f"{path}:{lineno}: invalid constraint/block")
        require(1 <= i <= j <= dimensions[bi], f"{path}:{lineno}: invalid matrix indices")
        key = (ci, bi, i, j)
        require(key not in seen, f"{path}:{lineno}: duplicate coefficient")
        seen.add(key)
        value = F(parts[4])
        require(value != 0, f"{path}:{lineno}: zero coefficient must be omitted")
        values[key] = value
    return values


def compare_maps(expected, actual, description):
    mismatches = [(key, expected.get(key, F(0)), actual.get(key, F(0)))
                  for key in sorted(expected.keys() | actual.keys())
                  if expected.get(key, F(0)) != actual.get(key, F(0))]
    require(not mismatches,
            f"{description}: {len(mismatches)} mismatches; first {mismatches[:5]}")


def check_files(metadata, compressed_coefficients, original_coefficients=None):
    metadata, compressed_coefficients = Path(metadata), Path(compressed_coefficients)
    meta = read_json(metadata)
    expected, zero_blocks, dimensions, scope = build_direct_zero(meta)
    actual = read_tsv(compressed_coefficients, dimensions, scope["configuration_count"])
    zero_actual = {key: value for key, value in actual.items() if key[1] in zero_blocks}
    compare_maps(expected, zero_actual, "direct degree-zero union coefficients")
    # Alpha and auxiliary-slack entries in exact proof inequalities are fixed,
    # independently of the degree-zero simplification.
    alpha = integer(meta["alpha_block"], "alpha_block")
    alpha_expected = {(ci, alpha, 1, 1): F(-1)
                      for ci, spec in enumerate(meta["constraints"], 1)
                      if integer(spec["size"], "size") == 1}
    nonsdp_actual = {key: value for key, value in actual.items()
                    if meta["blocks"][key[1] - 1]["kind"] != "kernel"}
    compare_maps(alpha_expected, nonsdp_actual, "alpha and excluded auxiliary slacks")
    result = {"schema": "independent-compressed-zero-union-check-v1", "status": "PASS",
              "method": "direct three union cases, averaged over all reference orderings",
              "uses_original_coefficient_compression": False,
              "arithmetic": "Python fractions.Fraction; no floating point",
              "scope": scope,
              "metadata": {"path": str(metadata), "sha256": sha(metadata)},
              "compressed_coefficients": {"path": str(compressed_coefficients),
                                           "sha256": sha(compressed_coefficients)},
              "checker_sha256": sha(__file__),
              "positive_degree_unchanged_checked": False}
    if original_coefficients is not None:
        original_coefficients = Path(original_coefficients)
        original_dimensions = {bi: integer(spec["dim"], "dim")
                               for bi, spec in enumerate(meta["blocks"], 1)}
        original = read_tsv(original_coefficients, original_dimensions,
                            scope["configuration_count"])
        untouched = {key: value for key, value in original.items() if key[1] not in zero_blocks}
        untouched_actual = {key: value for key, value in actual.items() if key[1] not in zero_blocks}
        compare_maps(untouched, untouched_actual, "positive-degree and other unchanged coefficients")
        result["positive_degree_unchanged_checked"] = True
        result["unchanged_nonzero_coefficients"] = len(untouched)
        result["original_coefficients"] = {"path": str(original_coefficients),
                                           "sha256": sha(original_coefficients)}
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--compressed-coefficients", required=True)
    parser.add_argument("--original-coefficients",
                        help="optional: also compare every positive-degree/other coefficient unchanged")
    parser.add_argument("--output", required=True, help="write exact-check report JSON")
    args = parser.parse_args(argv)
    checked = [args.metadata, args.compressed_coefficients]
    if args.original_coefficients is not None:
        checked.append(args.original_coefficients)
    require(Path(args.output).resolve() not in {Path(path).resolve() for path in checked},
            'output would overwrite a checked input')
    try:
        result = check_files(args.metadata, args.compressed_coefficients, args.original_coefficients)
        code = 0
    except (ValueError, KeyError, IndexError, TypeError, OSError) as exc:
        result = {"schema": "independent-compressed-zero-union-check-v1",
                  "status": "FAIL", "error": str(exc), "checker_sha256": sha(__file__)}
        code = 1
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"status": result["status"], "report": str(output),
                      "error": result.get("error")}, ensure_ascii=False))
    return code


if __name__ == "__main__":
    sys.exit(main())
