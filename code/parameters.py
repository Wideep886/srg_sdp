"""Exact necessary SRG conditions and rational spherical embeddings.

``analyze((v, valency, lambda_, mu), eigenspace='auto')`` returns only
JSON-compatible objects.  ``supported`` means that the necessary conditions
checked here pass and an injective rational embedding is available.  It is
never a claim that a graph with those parameters exists.

The rational SDP backend cannot represent quadratic irrational angles.
Admissible nonsquare-discriminant parameters are therefore ``unsupported``;
they are not declared impossible.  Empty and complete graphs also lie outside
the backend's two-distance SRG interface.
"""

from collections.abc import Mapping
from fractions import Fraction as Q
from math import isqrt
import re


def _integer(value):
    """Accept integers and integer text, without rounding or accepting bools."""
    if isinstance(value, bool):
        raise ValueError('Boolean values are not SRG parameters.')
    if isinstance(value, int):
        return value
    if isinstance(value, str) and re.fullmatch(r'[+-]?\d+', value.strip()):
        return int(value)
    raise ValueError('Each SRG parameter must be an integer or integer string.')


def _parameters(parameters):
    if isinstance(parameters, Mapping):
        names = [('v',), ('valency', 'k'), ('lambda', 'lambda_', 'lam'), ('mu',)]
        values = []
        for aliases in names:
            found = [_integer(parameters[name]) for name in aliases if name in parameters]
            if not found:
                raise ValueError('Missing parameter: ' + aliases[0])
            if any(value != found[0] for value in found):
                raise ValueError('Conflicting parameter aliases: ' + ', '.join(aliases))
            values.append(found[0])
        return values
    if isinstance(parameters, (str, bytes)):
        raise ValueError('Pass four parameters as a sequence or named mapping.')
    try:
        values = list(parameters)
    except TypeError:
        raise ValueError('Pass four parameters as a sequence or named mapping.') from None
    if len(values) != 4:
        raise ValueError('Exactly four SRG parameters are required.')
    return [_integer(value) for value in values]


def _finish(result, status, *reasons):
    result['status'] = status
    result['reasons'].extend(reasons)
    result['graph_existence'] = (
        'ruled_out_by_necessary_conditions' if status == 'infeasible' else 'not_determined'
    )
    return result


def _check(result, name, condition, explanation):
    result['exact_checks'][name] = bool(condition)
    if not condition:
        result['reasons'].append(explanation)
    return bool(condition)


def _multiply(left, right, v, k, lam, mu):
    """Multiply coefficients in the adjacency algebra, in (I, A, J) order."""
    x, y, z = left
    X, Y, Z = right
    return (
        x * X + (k - mu) * y * Y,
        x * Y + y * X + (lam - mu) * y * Y,
        x * Z + z * X + mu * y * Y + k * (y * Z + z * Y) + v * z * Z,
    )


def _projector(v, k, eigenvalue, other):
    coefficient = Q(1, eigenvalue - other)
    return (-other * coefficient, coefficient, -Q(k - other, v) * coefficient)


def _candidate(v, k, lam, mu, label, eigenvalue, other, dimension):
    coefficients = _projector(v, k, eigenvalue, other)
    icoef, acoef, jcoef = coefficients
    diagonal = Q(dimension, v)
    adjacent = (acoef + jcoef) / diagonal
    nonadjacent = jcoef / diagonal
    angles = sorted((adjacent, nonadjacent))
    checks = {
        'idempotent': _multiply(coefficients, coefficients, v, k, lam, mu) == coefficients,
        'annihilates_all_ones': icoef + k * acoef + v * jcoef == 0,
        'adjacency_eigenprojector': _multiply((Q(0), Q(1), Q(0)), coefficients, v, k, lam, mu)
        == tuple(eigenvalue * value for value in coefficients),
        'diagonal_equals_dimension_over_v': icoef + jcoef == diagonal,
        'trace_equals_dimension': v * (icoef + jcoef) == dimension,
        'adjacent_angle_formula': adjacent == Q(eigenvalue, k),
        'nonadjacent_angle_formula': nonadjacent == -Q(eigenvalue + 1, v - k - 1),
        'complement_angles': Q(-1 - eigenvalue, v - k - 1) == nonadjacent
        and Q(eigenvalue, k) == adjacent,
        'angles_in_unit_interval': all(-1 <= angle <= 1 for angle in angles),
    }
    # Both relation classes occur because 0 < k < v-1.  Unit vectors coincide
    # exactly when their inner product is 1, including the antipodal case -1.
    injective = adjacent != 1 and nonadjacent != 1
    limitations = []
    if not injective:
        limitations.append('Projection identifies distinct vertices (an off-diagonal inner product equals 1).')
    if dimension < 2:
        limitations.append('The SDP core requires embedding dimension n >= 2.')
    if adjacent == nonadjacent:
        limitations.append('The SDP core requires two distinct inner products.')
    if not all(checks.values()):
        limitations.append('An exact projector or unit-vector identity fails.')
    return {
        'eigenspace': label,
        'eigenvalue': eigenvalue,
        'dimension': dimension,
        'inner_product_adjacent': str(adjacent),
        'inner_product_nonadjacent': str(nonadjacent),
        'allowed_inner_products': [str(angle) for angle in angles],
        'projector_coefficients': {'I': str(icoef), 'A': str(acoef), 'J': str(jcoef)},
        'projector_diagonal': str(diagonal),
        'same_subspace_complement_eigenvalue': -1 - eigenvalue,
        'injective': injective,
        'rational': True,
        'supported': not limitations,
        'limitations': limitations,
        'exact_checks': checks,
    }


def _quadratic_root(delta, discriminant, sign):
    return '({} {} sqrt({}))/2'.format(delta, '+' if sign > 0 else '-', discriminant)


def _irrational_analysis(result, v, k, lam, mu, discriminant):
    """Certify multiplicity obstructions without approximating square roots."""
    delta = lam - mu
    numerator = 2 * k + (v - 1) * delta
    r = _quadratic_root(delta, discriminant, 1)
    s = _quadratic_root(delta, discriminant, -1)
    result['spectrum'] = {
        'valency': k, 'r': r, 's': s, 'discriminant': discriminant,
        'rational_eigenvalues': False,
        'r_multiplicity': None, 's_multiplicity': None,
        'r_multiplicity_expression': '({}-({})/sqrt({}))/2'.format(v - 1, numerator, discriminant),
        's_multiplicity_expression': '({}+({})/sqrt({}))/2'.format(v - 1, numerator, discriminant),
    }
    good = _check(result, 'irrational_multiplicity_coefficient_zero', numerator == 0,
                  'Nonsquare discriminant and nonzero 2*k+(v-1)*(lambda-mu) force irrational eigenvalue multiplicities.')
    good = _check(result, 'irrational_multiplicities_integral', (v - 1) % 2 == 0,
                  'Conjugate irrational eigenvalues require equal integral multiplicities (v-1)/2.') and good
    if not good:
        return _finish(result, 'infeasible')
    dimension = (v - 1) // 2
    result['spectrum'].update(r_multiplicity=dimension, s_multiplicity=dimension)
    checks = {
        'multiplicity_sum': 1 + 2 * dimension == v,
        'spectrum_trace': k + dimension * delta == 0,
        'spectrum_trace_square': k * k + dimension * (delta * delta + 2 * (k - mu)) == v * k,
        'spectrum_trace_cube': k ** 3 + dimension * (delta ** 3 + 3 * delta * (k - mu)) == v * k * lam,
    }
    for name, passed in checks.items():
        _check(result, name, passed, 'Exact symbolic spectrum identity failed: ' + name)
    if not all(checks.values()):
        return _finish(result, 'infeasible')
    result['complement']['spectrum'] = {
        'valency': v - k - 1,
        'r': _quadratic_root(-2 - delta, discriminant, 1),
        's': _quadratic_root(-2 - delta, discriminant, -1),
        'r_multiplicity': dimension, 's_multiplicity': dimension,
        'rational_eigenvalues': False,
    }
    for label, value in (('r', r), ('s', s)):
        result['embedding_candidates'][label] = {
            'eigenspace': label, 'eigenvalue': value, 'dimension': dimension,
            'inner_product_adjacent': '({})/{}'.format(value, k),
            'inner_product_nonadjacent': '-(({})+1)/{}'.format(value, v - k - 1),
            'injective': True, 'rational': False, 'supported': False,
            'limitations': ['Quadratic irrational angles require an exact algebraic backend.'],
        }
    return _finish(result, 'unsupported',
                   'Necessary parameter and multiplicity checks pass, but the embedding has quadratic irrational angles; the current SDP backend uses exact rational arithmetic.')


def analyze(parameters, eigenspace='auto'):
    """Analyze arbitrary integer SRG parameters, without a catalog whitelist.

    Parameters may be a four-element sequence or a mapping with ``v``,
    ``valency`` (alias ``k``), ``lambda`` (aliases ``lambda_``/``lam``), ``mu``.
    Integer strings are accepted; floating-point values and booleans are not.
    ``eigenspace`` is ``auto``, ``r`` (larger restricted eigenvalue), or ``s``.
    Auto chooses the smallest supported injective embedding, breaking ties in
    favor of r.  Explicit r/s never silently switches to the other eigenspace.

    Status is ``invalid`` for malformed input, ``infeasible`` for a failed
    exact necessary graph condition, ``unsupported`` for a backend limitation,
    or ``supported``.  Only the last status supplies n, a and b for the builder;
    a and b are reduced rational strings and satisfy -1 <= a < b < 1.
    """
    result = {
        'schema': 'srg-parameters-v1', 'parameters': None,
        'status': None, 'reasons': [], 'graph_existence': 'not_determined',
        'requested_eigenspace': eigenspace if isinstance(eigenspace, str) else None,
        'selected_eigenspace': None, 'selected_eigenvalue': None,
        'selected_dimension': None, 'n': None, 'a': None, 'b': None,
        'spectrum': None, 'complement': None,
        'embedding_candidates': {}, 'exact_checks': {},
    }
    if eigenspace not in ('auto', 'r', 's'):
        return _finish(result, 'invalid', "eigenspace must be 'auto', 'r', or 's'.")
    try:
        result['parameters'] = _parameters(parameters)
    except (ValueError, TypeError) as error:
        return _finish(result, 'invalid', str(error))
    v, k, lam, mu = result['parameters']
    if v < 1 or lam < 0 or mu < 0:
        return _finish(result, 'invalid', 'Require v >= 1 and nonnegative lambda and mu.')
    if not _check(result, 'valency_range', 0 <= k < v,
                  'A simple graph on v vertices requires 0 <= valency < v.'):
        return _finish(result, 'infeasible')

    # For complete/empty graphs one relation is absent.  Do not apply bounds
    # to its vacuous parameter or divide by its zero valency.
    if k == 0 or k == v - 1:
        if v > 1 and k == 0 and mu != 0:
            return _finish(result, 'infeasible', 'An empty graph with v > 1 requires mu = 0.')
        if v > 1 and k == v - 1 and lam != v - 2:
            return _finish(result, 'infeasible', 'A complete graph with v > 1 requires lambda = v-2.')
        result['degenerate_graph'] = 'one_vertex' if v == 1 else 'empty' if k == 0 else 'complete'
        return _finish(result, 'unsupported',
                       'Empty, complete, and one-vertex graphs are outside the two-distance SRG interface.')

    complement = [v, v - k - 1, v - 2 * k + mu - 2, v - 2 * k + lam]
    result['complement'] = {'parameters': complement, 'spectrum': None}
    checks = [
        ('lambda_range', max(0, 2 * k - v) <= lam <= k - 1,
         'Common neighbors of an adjacent pair require max(0,2*k-v) <= lambda <= k-1.'),
        ('mu_range', max(0, 2 * k - v + 2) <= mu <= k,
         'Common neighbors of a nonadjacent pair require max(0,2*k-v+2) <= mu <= k.'),
        ('parameter_identity', k * (k - lam - 1) == (v - k - 1) * mu,
         'The necessary SRG identity k*(k-lambda-1) = (v-k-1)*mu fails.'),
        ('edge_count_integral', v * k % 2 == 0,
         'The edge count v*k/2 is not an integer.'),
        ('triangle_count_integral', v * k * lam % 6 == 0,
         'The triangle count v*k*lambda/6 is not an integer.'),
        ('complement_triangle_count_integral', v * complement[1] * complement[2] % 6 == 0,
         'The complement triangle count is not an integer.'),
        ('complement_parameter_identity', complement[1] * (complement[1] - complement[2] - 1)
         == (v - complement[1] - 1) * complement[3],
         'The complement SRG parameter identity fails.'),
    ]
    for name, passed, reason in checks:
        _check(result, name, passed, reason)
    if not all(result['exact_checks'].values()):
        return _finish(result, 'infeasible')

    discriminant = (lam - mu) ** 2 + 4 * (k - mu)
    if not _check(result, 'positive_discriminant', discriminant > 0,
                  'Proper SRG parameters require two distinct real restricted eigenvalues.'):
        return _finish(result, 'infeasible')
    root = isqrt(discriminant)
    if root * root != discriminant:
        return _irrational_analysis(result, v, k, lam, mu, discriminant)
    r = Q(lam - mu + root, 2)
    s = Q(lam - mu - root, 2)
    f = Q(-k - (v - 1) * s, r - s)
    g = Q(k + (v - 1) * r, r - s)
    result['spectrum'] = {
        'valency': k, 'r': int(r) if r.denominator == 1 else str(r),
        's': int(s) if s.denominator == 1 else str(s),
        'r_multiplicity': int(f) if f.denominator == 1 else str(f),
        's_multiplicity': int(g) if g.denominator == 1 else str(g),
        'discriminant': discriminant, 'rational_eigenvalues': True,
    }
    spectral_checks = {
        'rational_eigenvalues_integral': r.denominator == s.denominator == 1,
        'multiplicities_positive_integral': f.denominator == g.denominator == 1 and f > 0 and g > 0,
        'multiplicity_sum': 1 + f + g == v,
        'spectrum_trace': k + f * r + g * s == 0,
        'spectrum_trace_square': k * k + f * r * r + g * s * s == v * k,
        'spectrum_trace_cube': k ** 3 + f * r ** 3 + g * s ** 3 == v * k * lam,
    }
    for name, passed in spectral_checks.items():
        _check(result, name, passed, 'Necessary exact spectrum condition failed: ' + name)
    if not all(spectral_checks.values()):
        return _finish(result, 'infeasible')
    r, s, f, g = int(r), int(s), int(f), int(g)
    result['spectrum']['restricted_eigenvalues_may_equal_valency'] = mu == 0
    result['complement']['spectrum'] = {
        'valency': v - k - 1, 'r': -1 - s, 's': -1 - r,
        'r_multiplicity': g, 's_multiplicity': f, 'rational_eigenvalues': True,
    }
    candidates = {
        'r': _candidate(v, k, lam, mu, 'r', r, s, f),
        's': _candidate(v, k, lam, mu, 's', s, r, g),
    }
    result['embedding_candidates'] = candidates
    er, es = _projector(v, k, r, s), _projector(v, k, s, r)
    zero = (Q(0), Q(0), Q(0))
    _check(result, 'projectors_orthogonal', _multiply(er, es, v, k, lam, mu) == zero,
           'The two exact restricted projectors are not orthogonal.')
    _check(result, 'projectors_resolve_centered_space',
           tuple(x + y for x, y in zip(er, es)) == (Q(1), Q(0), -Q(1, v)),
           'The two restricted projectors do not sum to I-J/v.')
    for label, candidate in candidates.items():
        _check(result, label + '_projector_identities', all(candidate['exact_checks'].values()),
               'A necessary exact projector or unit-vector condition fails for eigenspace ' + label + '.')
    if not all(result['exact_checks'].values()):
        return _finish(result, 'infeasible')

    choices = [candidate for label, candidate in candidates.items()
               if candidate['supported'] and (eigenspace == 'auto' or label == eigenspace)]
    if not choices:
        limitations = (candidates[eigenspace]['limitations'] if eigenspace != 'auto'
                       else [reason for candidate in candidates.values() for reason in candidate['limitations']])
        return _finish(result, 'unsupported', *limitations)
    chosen = min(choices, key=lambda candidate: (candidate['dimension'], candidate['eigenspace']))
    result.update(selected_eigenspace=chosen['eigenspace'], selected_eigenvalue=chosen['eigenvalue'],
                  selected_dimension=chosen['dimension'], n=chosen['dimension'],
                  a=chosen['allowed_inner_products'][0], b=chosen['allowed_inner_products'][1])
    result['core_scope'] = {'minimum_points': 2, 'maximum_points': min(6, result['n']), 'minimum_degree': 0}
    return _finish(result, 'supported')
