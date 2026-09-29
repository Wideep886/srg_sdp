"""Construct the rational SDP using the retained coefficient and compression formulas."""
from collections import defaultdict
from copy import deepcopy
from decimal import Decimal, localcontext
from fractions import Fraction as F
from itertools import product
from pathlib import Path
import hashlib
import json
import re
import time

import exact_model as exact

ETA = F('1e-30')
SHIFT = 2 * ETA
LINEAR = F('1e-26')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def decimal(value, digits=600):
    with localcontext() as context:
        context.prec = digits
        return str(Decimal(value.numerator) / Decimal(value.denominator))


def block_maps(spec):
    """Retained reference-sum map: each reference row has weight 1/m."""
    if spec['kind'] != 'kernel' or spec['l'] != 0 or spec['m'] == 0:
        return deepcopy(spec), [(i, F(1)) for i in range(spec['dim'])]
    m = spec['m']
    gram = [[F(x) for x in row] for row in spec['gram']]
    refs = {tuple(gram[i][j] for i in range(m)) for j in range(m)}
    labels = [tuple(F(x) for x in u) for u in spec['states']]
    ri = [i for i, u in enumerate(labels) if u in refs]
    gi = [i for i, u in enumerate(labels) if u not in refs]
    exact.require(len(ri) == m, 'reference labels do not match reference size')
    maps = {i: (0, F(1, m)) for i in ri}
    maps.update({i: (j + 1, F(1)) for j, i in enumerate(gi)})
    new = deepcopy(spec)
    new.update(original_dim=spec['dim'], dim=len(gi) + 1,
               compression='reference-sum', reference_indices=ri, generic_indices=gi,
               states=['reference_sum'] + [spec['states'][i] for i in gi])
    return new, [maps[i] for i in range(spec['dim'])]


def fresh_metadata(embedding, points, degree):
    n, a, b = embedding['n'], F(embedding['a']), F(embedding['b'])
    exact.require(type(points) is int and 2 <= points <= min(6, n),
                  'points must be between 2 and min(6, sphere dimension)')
    exact.require(type(degree) is int and degree >= 0, 'degree must be nonnegative')
    feasible = {m: exact.realizable_orbits(n, a, b, m) for m in range(points + 1)}
    blocks = [dict(kind='alpha', dim=1)]
    dependent = []
    for m in range(points - 1):
        for oi, (mask, (gram, info)) in enumerate(sorted(feasible[m].items()), 1):
            if info['rank'] < m:
                dependent.append([m, mask])
                continue
            inverse = exact.inverse(gram)
            generic = {u for u in product((a, b), repeat=m)
                       if 1 - exact.bilinear(u, inverse, u) >= 0}
            refs = {tuple(gram[i][j] for i in range(m)) for j in range(m)}
            for ell in range(degree + 1):
                states = sorted(generic | refs if ell == 0 else generic)
                blocks.append(dict(kind='kernel', dim=len(states), raw_dim=len(states),
                    m=m, l=ell, orbit_index=oi,
                    gram=[[str(x) for x in row] for row in gram],
                    states=[[str(x) for x in u] for u in states]))
    constraints = []
    for size in range(1, points + 1):
        for mask, (gram, info) in sorted(feasible[size].items()):
            blocks.append(dict(kind='slack', dim=1, constraint=len(constraints) + 1))
            constraints.append(dict(size=size, gram=[[str(x) for x in row] for row in gram],
                rhs=str(1 if size == 1 else 2 if size == 2 else 0), slack_block=len(blocks)))
    return dict(schema='srg-sdp-full-model-v1', case='_'.join(map(str, embedding['parameters'])),
        parameters=embedding['parameters'], n=n, a=str(a), b=str(b), k=points, d=degree,
        alpha_block=1, blocks=blocks, constraints=constraints,
        dependent_reference_policy='zero', zero_dependent_references=dependent,
        orbit_counts=[len(feasible[m]) for m in range(points + 1)],
        singular_configuration_counts={str(m): sum(q['rank'] < m for G, q in feasible[m].values())
                                       for m in range(points + 1)},
        anchor_policy='Independent reference types; dependent local functions fixed to zero. All realizable constraints retained.',
        arithmetic='Exact rational Gram enumeration and rank tests.',
        source_parameter_audit=embedding)


def parameter_text(precision, iterations, epsilon):
    exact.require(type(precision) is int and precision >= 64, 'precision must be at least 64 bits')
    exact.require(type(iterations) is int and iterations > 0, 'iterations must be positive')
    exact.require(isinstance(epsilon, str) and re.fullmatch(r'[+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?', epsilon)
                  and F(epsilon) > 0, 'solver epsilon must be a positive decimal, e.g. 1e-16')
    return '\n'.join((f'{iterations} unsigned int maxIteration;',
        f'{epsilon} double 0.0 < epsilonStar;', '1e8 double 0.0 < lambdaStar;',
        '2.0 double 1.0 < omegaStar;', '-1e5 double lowerBound;', '1e5 double upperBound;',
        '0.2 double 0.0 <= betaStar < 1.0;', '0.3 double 0.0 <= betaBar < 1.0;',
        '0.5 double 0.0 < gammaStar < 1.0;', f'{epsilon} double 0.0 < epsilonDash;',
        f'{precision} precision;', ''))


def build(embedding, points, degree, output, precision=1500, iterations=2000, epsilon='1e-16'):
    started = time.monotonic()
    settings = parameter_text(precision, iterations, epsilon)
    meta = fresh_metadata(embedding, points, degree)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    original = output / 'original_metadata.json'
    dump(original, meta)
    print('Building exact coefficients; orbit counts:', meta['orbit_counts'], flush=True)
    constraints, audit = exact.export_coefficients(original, output / 'original_exact_coefficients.tsv')
    modeled = deepcopy(meta)
    modeled['blocks'], maps = [], []
    for spec in meta['blocks']:
        new, mapping = block_maps(spec)
        modeled['blocks'].append(new)
        maps.append(mapping)
    transformed = []
    for constraint in constraints:
        terms = defaultdict(F)
        for (bi, i, j), value in constraint['terms'].items():
            ii, wi = maps[bi - 1][i - 1]
            jj, wj = maps[bi - 1][j - 1]
            ii, jj = sorted((ii + 1, jj + 1))
            terms[bi, ii, jj] += value * wi * wj
        transformed.append(dict(constraint, terms={key: value for key, value in terms.items() if value}))
    modeled.update(schema='compressed-kpoint-dual-metadata-v1', representation='compressed',
        original_metadata_sha256=sha(original), modeled_matrix_shift=str(SHIFT),
        required_matrix_margin=str(ETA), modeled_linear_margin=str(LINEAR),
        required_linear_margin=str(LINEAR), builder_sha256=sha(__file__))
    dump(output / 'metadata.json', modeled)
    with (output / 'exact_coefficients.tsv').open('w') as stream:
        stream.write('# constraint\tblock\trow\tcolumn\texact scalar coefficient\n')
        for ci, constraint in enumerate(transformed, 1):
            for (bi, i, j), value in sorted(constraint['terms'].items()):
                stream.write(f'{ci}\t{bi}\t{i}\t{j}\t{value}\n')
    shifts = []
    for spec in modeled['blocks']:
        diagonal = [F(0)] * spec['dim']
        if spec['kind'] == 'kernel':
            diagonal = [SHIFT] * spec['dim']
            if spec.get('compression') == 'reference-sum':
                diagonal[0] *= spec['m']
        shifts.append(diagonal)
    dump(output / 'physical_shifts.json', [[str(x) for x in row] for row in shifts])
    # A 0-by-0 proof block has no coordinates. SDPA receives only nonempty
    # blocks; extraction and the separate export audit restore logical IDs.
    active_ids = [i for i, spec in enumerate(modeled['blocks'], 1) if spec['dim'] > 0]
    solver_ids = {logical: i for i, logical in enumerate(active_ids, 1)}
    maximum_error = F(0)
    def emitted(value):
        nonlocal maximum_error
        text = decimal(value)
        maximum_error = max(maximum_error, abs(value - F(text)))
        return text
    rhs = []
    for constraint in transformed:
        correction = sum((value * shifts[b - 1][i - 1]
            for (b, i, j), value in constraint['terms'].items() if i == j), F(0))
        rhs.append(-constraint['rhs'] - LINEAR - correction)
    with (output / 'problem.dat-s').open('w') as stream:
        stream.write(f'{len(transformed)}\n{len(active_ids)}\n')
        stream.write(' '.join(str(modeled['blocks'][i - 1]['dim']) for i in active_ids) + '\n')
        stream.write(' '.join(emitted(x) for x in rhs) + '\n')
        stream.write(f'0 {solver_ids[modeled["alpha_block"]]} 1 1 -1\n')
        for ci, constraint in enumerate(transformed, 1):
            for (bi, i, j), value in sorted(constraint['terms'].items()):
                stream.write(f'{ci} {solver_ids[bi]} {i} {j} {emitted(value if i == j else value / 2)}\n')
            stream.write(f'{ci} {solver_ids[constraint["slack_block"]]} 1 1 1\n')
    (output / 'solver.par').write_text(settings)
    report = dict(schema='srg-sdp-build-v1', parameters=embedding['parameters'],
        k=points, d=degree, original_exact_model_audit=audit,
        maximum_export_rounding_error_exact=str(maximum_error),
        export_significant_decimal_digits=600, solver_requested_bits=precision,
        solver_logical_block_ids=active_ids, solver_epsilon=epsilon,
        input_sha256=sha(output / 'problem.dat-s'), metadata_sha256=sha(output / 'metadata.json'),
        parameter_sha256=sha(output / 'solver.par'),
        source_sha256={p.name: sha(p) for p in (Path(__file__), Path(exact.__file__))},
        files_sha256={p.name: sha(p) for p in sorted(output.iterdir()) if p.is_file()},
        build_wall_seconds=time.monotonic() - started)
    dump(output / 'build.json', report)
    return output
