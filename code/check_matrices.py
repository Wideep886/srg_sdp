#!/usr/bin/env python3
"""Certify physical full/compressed SDP candidates using exact arithmetic.

The candidate matrices must ALREADY include the modeled matrix shift. Decimal
strings are interpreted as exact rationals, without rounding or implicit repair.
The program independently derives the compression map from ORIGINAL metadata,
lifts the candidate, runs the existing original exact verifier in a separate
process, and checks every freshly reconstructed coefficient/residual.
The exact core includes every realizable singular Gram configuration for k<=6.
"""
from decimal import Decimal, localcontext
from fractions import Fraction as F
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import sys
import time
from exact_model import read_json

if hasattr(sys, 'set_int_max_str_digits'):
    sys.set_int_max_str_digits(0)

HERE = Path(__file__).resolve().parent
ORIGINAL_VERIFIER = HERE/'exact_model.py'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rational(value):
    require(isinstance(value, (str, int, F)) and not isinstance(value, bool),
            'rational inputs must be strings or integers')
    return F(value)


def integer(value):
    q = rational(value)
    require(q.denominator == 1, 'integer metadata field is not integral')
    return q.numerator


def matrix(value):
    require(isinstance(value, list), 'matrix must be an array')
    require(all(isinstance(r, list) for r in value), 'matrix rows must be arrays')
    return [[rational(x) for x in row] for row in value]


def zeros(n, m=None):
    return [[F(0) for _ in range(n if m is None else m)] for _ in range(n)]


def identity(n):
    return [[F(i == j) for j in range(n)] for i in range(n)]


def transpose(A):
    return list(map(list, zip(*A)))


def mul(A, B):
    require(bool(A) and bool(B) and len(A[0]) == len(B), 'matrix product shape')
    return [[sum((x*y for x, y in zip(row, col)), F(0))
             for col in zip(*B)] for row in A]


def add(A, B):
    require(len(A) == len(B) and all(len(a) == len(b) for a, b in zip(A, B)),
            'matrix addition shape')
    return [[x+y for x, y in zip(a, b)] for a, b in zip(A, B)]


def scale(c, A):
    return [[c*x for x in row] for row in A]


def congr(A, B):
    return mul(mul(A, B), transpose(A))


def brief(q):
    if q is None:
        return None
    with localcontext() as ctx:
        ctx.prec = 18
        return str(Decimal(q.numerator)/Decimal(q.denominator))


def strict_ldl(A):
    """Exact Schur elimination; no tolerance, eigensolver, or square root."""
    n = len(A)
    require(all(len(r) == n for r in A), 'nonsquare PSD block')
    require(all(A[i][j] == A[j][i] for i in range(n) for j in range(n)),
            'matrix not exactly symmetric; no implicit symmetrization is allowed')
    B = [r[:] for r in A]
    pivots = []
    for j in range(n):
        p = B[j][j]
        pivots.append(p)
        if p <= 0:
            return dict(positive_definite=False, first_nonpositive_pivot=j+1,
                        pivot=str(p), pivot_decimal=brief(p))
        for i in range(j+1, n):
            for h in range(i, n):
                B[i][h] -= B[i][j]*B[j][h]/p
                B[h][i] = B[i][h]
    minimum = min(pivots) if pivots else None
    return dict(positive_definite=True,
                minimum_pivot=str(minimum) if minimum is not None else None,
                minimum_pivot_decimal=brief(minimum))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2)+'\n')


def labels(spec):
    return [tuple(rational(x) for x in u) for u in spec['states']]


def derive_map(original, modeled, block):
    """C uses original row order and reference-first compressed columns."""
    m = integer(original['m'])
    dim = integer(original['dim'])
    G = matrix(original['gram'])
    states = labels(original)
    refs = [tuple(G[i][j] for i in range(m)) for j in range(m)]
    require(len(states) == dim and len(set(states)) == dim, 'original state list')
    require(all(u in states for u in refs), 'reference labels absent')
    reference = [states.index(u) for u in refs]
    generic = [i for i in range(dim) if i not in reference]
    require(len(set(reference)) == m, 'reference labels not distinct')
    require(integer(modeled['original_dim']) == dim, f'block {block}: original_dim')
    require(modeled.get('compression') == 'reference-sum', f'block {block}: compression')
    require(sorted(integer(x) for x in modeled['reference_indices']) == sorted(reference),
            f'block {block}: reference_indices do not match original Gram columns')
    require([integer(x) for x in modeled['generic_indices']] == generic,
            f'block {block}: generic_indices do not match original states')
    require(integer(modeled['dim']) == len(generic)+1, f'block {block}: compressed dim')
    target_states = modeled['states']
    require(len(target_states) == len(generic)+1 and target_states[0] == 'reference_sum',
            f'block {block}: missing aggregate state')
    require([tuple(rational(x) for x in u) for u in target_states[1:]] ==
            [states[i] for i in generic], f'block {block}: compressed state order')
    C = zeros(dim, len(generic)+1)
    for i in reference:
        C[i][0] = 1
    for j, i in enumerate(generic, 1):
        C[i][j] = 1
    D = congr(transpose(C), identity(dim))
    J = [r[:] for r in C]
    for i in reference:
        J[i][0] = F(1, m)
    P = add(identity(dim), scale(-1, mul(J, transpose(C))))
    require(mul(transpose(C), J) == identity(len(generic)+1), 'right inverse identity')
    require(mul(P, C) == zeros(dim, len(generic)+1), 'projector identity')
    require(mul(P, P) == P and transpose(P) == P, 'orthogonal projector identity')
    return dict(C=C, D=D, J=J, P=P, m=m, generic=generic, reference=reference)


def read_coefficients(path):
    result = {}
    for line in Path(path).read_text().splitlines():
        if not line.strip() or line.startswith('#'):
            continue
        ci, bi, i, j, value = line.split()
        key = tuple(map(int, (ci, bi, i, j)))
        require(all(index >= 1 for index in key) and key[2] <= key[3],
                'coefficient table must use positive upper-triangular indices')
        require(key not in result, 'duplicate coefficient table entry')
        coefficient = rational(value)
        require(coefficient != 0, 'zero coefficient table entry must be omitted')
        result[key] = coefficient
    return result


def write_coefficients(path, terms):
    with Path(path).open('w') as out:
        out.write('# constraint\tblock\trow\tcolumn\texact physical-matrix scalar coefficient\n')
        for key, value in sorted(terms.items()):
            if value:
                out.write('\t'.join(map(str, (*key, value)))+'\n')


def derive_coefficients(original, full_terms, maps, out_path):
    """Derive the physical compressed map from freshly rebuilt original data."""
    result = {key:value for key,value in full_terms.items() if key[1] not in maps}
    count = 0
    for ci in range(1, len(original['constraints'])+1):
        for bi, transform in maps.items():
            dim = integer(original['blocks'][bi-1]['dim'])
            A = zeros(dim)
            for i in range(dim):
                for j in range(i, dim):
                    value = full_terms.get((ci, bi, i+1, j+1), F(0))
                    A[i][j] = A[j][i] = value if i == j else value/2
            H = congr(transpose(transform['J']), A)
            require(A == congr(transform['C'], H),
                    f'constraint {ci}, block {bi}: original coefficient outside compressed span')
            require(mul(A, transform['P']) == zeros(dim), 'coefficient projector annihilation')
            for i in range(len(H)):
                for j in range(i, len(H)):
                    value = H[i][j] if i == j else 2*H[i][j]
                    if value:
                        result[ci, bi, i+1, j+1] = value
            count += 1
    write_coefficients(out_path, result)
    return result, count


def residuals(original, terms, mats):
    result = []
    for c in original['constraints']:
        size = integer(c['size'])
        rhs = F(1 if size == 1 else 2 if size == 2 else 0)
        if 'rhs' in c:
            require(rational(c['rhs']) == rhs, 'original RHS mismatch')
        result.append(rhs)
    for (ci, bi, i, j), value in terms.items():
        result[ci-1] += value*mats[bi-1][i-1][j-1]
    return result


def verify(args):
    output = args.output.resolve()
    inputs = {Path(path).resolve() for path in (
        args.original_metadata, args.metadata, args.candidate, args.model_coefficients)
        if path is not None}
    generated = {output / name for name in (
        'verification.json', 'verification_summary.json', 'lifted_candidate.json',
        'derived_model_coefficients.tsv', 'original_verifier_command.json', 'original_verifier.log',
        'original_verification/exact_coefficients.tsv',
        'original_verification/exact_verification.json',
        'original_verification/rational_dual_certificate.json',
        'original_verification/rational_dual_candidate.json')}
    require(not inputs.intersection(path.resolve() for path in generated),
            'output would overwrite a checked input')
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    result = dict(schema='compressed-kpoint-exact-verification-v1', verified=False,
        verifier_sha256=sha(__file__), python_version=sys.version,
        original_verifier_sha256=sha(ORIGINAL_VERIFIER),
        original_helper_sha256=sha(HERE/'exact_arithmetic.py'),
        original_metadata_sha256=sha(args.original_metadata),
        metadata_sha256=sha(args.metadata), candidate_sha256=sha(args.candidate),
        arithmetic='exact Fraction/integer; no feasibility tolerance',
        conversion=dict(candidate_literals='exact rational values of printed decimal/fraction literals',
            rational_conversion_error='0 relative to each supplied literal', rounding='none',
            implicit_repair=False, scalar_slacks='preserved but not used as proof of feasibility',
            solver_errors=('No bound on the solver internal errors or the distance to an optimum is assumed. '
                'Printed data define a new exact candidate, which is checked against the exact original model.'),
            input_coefficient_rounding=('SDPA input decimal approximations are not trusted for certification; '
                'original coefficients are rebuilt by the existing original exact verifier.')))
    try:
        original = read_json(args.original_metadata)
        modeled = read_json(args.metadata)
        candidate = read_json(args.candidate)
        require(modeled['original_metadata_sha256'] == result['original_metadata_sha256'],
                'modeled metadata is not linked to the supplied original metadata')
        require(candidate['metadata_sha256'] == result['metadata_sha256'],
                'candidate is not linked to the supplied modeled metadata')
        representation = modeled['representation']
        require(representation in ('full', 'compressed'), 'unsupported representation')
        for key in ('n', 'k', 'd', 'alpha_block'):
            require(integer(original[key]) == integer(modeled[key]), f'model mismatch: {key}')
        for key in ('a', 'b'):
            require(rational(original[key]) == rational(modeled[key]), f'model mismatch: {key}')
        require(original['constraints'] == modeled['constraints'],
                'modeled configuration list differs from original')
        require(original.get('dependent_reference_policy', 'reject') ==
                modeled.get('dependent_reference_policy', 'reject'),
                'modeled dependent-reference policy differs from original')
        eta = rational(modeled['required_matrix_margin'])
        modeled_shift = rational(modeled['modeled_matrix_shift'])
        modeled_linear_margin = rational(modeled['modeled_linear_margin'])
        linear_margin = rational(modeled.get('required_linear_margin', modeled['modeled_linear_margin']))
        require(eta > 0 and modeled_shift > eta and modeled_linear_margin >= linear_margin > 0,
                'matrix/linear protection parameters must be positive, with modeled shift > eta')
        if args.matrix_margin is not None:
            require(eta == rational(args.matrix_margin), 'required matrix margin mismatch')
        if args.linear_margin is not None:
            require(linear_margin == rational(args.linear_margin), 'required linear margin mismatch')
        result.update(representation=representation, case=str(original.get('case', 'unspecified')),
            matrix_margin=str(eta), linear_margin=str(linear_margin),
            modeled_matrix_shift=str(modeled_shift), modeled_linear_margin=str(modeled_linear_margin),
            lift_perpendicular_coefficient=str(2*eta))
        require(len(original['blocks']) == len(modeled['blocks']) == len(candidate['matrices']),
                'block counts differ')
        physical = [matrix(M) for M in candidate['matrices']]
        lifted = [[r[:] for r in M] for M in physical]
        transforms, shifted_checks = {}, []
        for bi, (old, new, M) in enumerate(zip(original['blocks'], modeled['blocks'], physical), 1):
            require(old['kind'] == new['kind'], f'block {bi}: kind differs')
            dim = integer(new['dim'])
            require(len(M) == dim and all(len(r) == dim for r in M), f'block {bi}: candidate shape')
            require(all(M[i][j] == M[j][i] for i in range(dim) for j in range(dim)),
                    f'block {bi}: asymmetric candidate')
            transformed = (representation == 'compressed' and old['kind'] == 'kernel'
                           and integer(old['m']) > 0 and integer(old['l']) == 0)
            if old['kind'] == 'kernel':
                require(integer(old['m']) == integer(new['m']) and integer(old['l']) == integer(new['l'])
                        and matrix(old['gram']) == matrix(new['gram']), f'block {bi}: reference differs')
            if transformed:
                transform = derive_map(old, new, bi)
                transforms[bi] = transform
                required = add(M, scale(-eta, transform['D']))
                full = add(congr(transform['J'], M), scale(2*eta, transform['P']))
                require(congr(transpose(transform['C']), full) == M, f'block {bi}: lift identity')
                lifted[bi-1] = full
                check = strict_ldl(required)
                lift_check = strict_ldl(add(full, scale(-eta, identity(len(full)))))
                shifted_checks.append(dict(block=bi, compressed=True, dimension=dim,
                    original_dimension=len(full), compressed_shift='eta * diag(m,I_q)',
                    shifted=check, lifted_shifted=lift_check))
            else:
                require(integer(old['dim']) == dim, f'block {bi}: changed untouched dimension')
                require(new.get('compression') != 'reference-sum', f'block {bi}: unexpected compression')
                if old['kind'] == 'kernel':
                    require(labels(old) == labels(new), f'block {bi}: changed untouched state order')
                    check = strict_ldl(add(M, scale(-eta, identity(dim))))
                    shifted_checks.append(dict(block=bi, compressed=False, dimension=dim,
                                               shifted=check, lifted_shifted=check))
        require(all(physical[i] == lifted[i] for i, b in enumerate(original['blocks'])
                    if b['kind'] in ('alpha', 'slack')), 'alpha or slack changed')
        result['shifted_blocks'] = shifted_checks
        result['all_shifted_blocks_strictly_positive'] = all(
            x['shifted']['positive_definite'] and x['lifted_shifted']['positive_definite']
            for x in shifted_checks)
        result['transformed_block_count'] = len(transforms)
        lifted_path = output/'lifted_candidate.json'
        require(lifted_path.resolve() != args.candidate.resolve(), 'output would overwrite candidate input')
        write_json(lifted_path, dict(schema='compressed-kpoint-lifted-candidate-v1',
            metadata_sha256=result['original_metadata_sha256'],
            source_candidate_sha256=result['candidate_sha256'],
            modeled_metadata_sha256=result['metadata_sha256'],
            matrices=[[[str(x) for x in row] for row in M] for M in lifted],
            transformations=dict(type='exact reference-sum lift', matrix_margin=str(eta),
                perpendicular_coefficient=str(2*eta), transformed_blocks=sorted(transforms),
                alpha_preserved=True, scalar_slacks_preserved=True)))
        result['lifted_candidate_sha256'] = sha(lifted_path)
        original_output = output/'original_verification'
        original_output.mkdir(exist_ok=True)
        command = [sys.executable, str(ORIGINAL_VERIFIER), '--metadata', str(args.original_metadata.resolve()),
            '--candidate', str(lifted_path), '--output', str(original_output), '--digits', str(args.digits)]
        write_json(output/'original_verifier_command.json', command)
        report_path = original_output/'exact_verification.json'
        if report_path.exists():
            report_path.unlink()
        with (output/'original_verifier.log').open('w') as log:
            process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
        result['original_verifier_returncode'] = process.returncode
        require(report_path.is_file(), 'original verifier did not produce a report; see original_verifier.log')
        independent = read_json(report_path)
        result['original_verification_sha256'] = sha(report_path)
        result['original_verifier_verified'] = bool(independent['verified']) and process.returncode == 0
        require(independent['metadata_sha256'] == result['original_metadata_sha256'] and
                independent['candidate_sha256'] == result['lifted_candidate_sha256'],
                'original verifier report provenance mismatch')
        full_coeff_path = original_output/'exact_coefficients.tsv'
        full_terms = read_coefficients(full_coeff_path)
        result['fresh_original_coefficients_sha256'] = sha(full_coeff_path)
        derived_path = output/'derived_model_coefficients.tsv'
        derived, checked = derive_coefficients(original, full_terms, transforms, derived_path)
        result['derived_model_coefficients_sha256'] = sha(derived_path)
        result['coefficient_factorizations_checked'] = checked
        if args.model_coefficients is not None:
            require(read_coefficients(args.model_coefficients) == derived,
                    'builder coefficient table differs from the independently derived physical coefficients')
            result['builder_model_coefficients_sha256'] = sha(args.model_coefficients)
            result['builder_coefficients_exactly_match'] = True
        before = residuals(original, derived, physical)
        after = residuals(original, full_terms, lifted)
        require(before == after, 'compression/lift changed an original residual')
        require(after == [rational(c['residual']) for c in independent['constraints']],
                'residuals differ from the original verifier report')
        result['all_residuals_exactly_preserved'] = True
        result['all_residuals_below_negative_margin'] = all(v < -linear_margin for v in after)
        result['all_original_constraints_strict'] = all(v < 0 for v in after)
        result['maximum_original_residual'] = str(max(after))
        result['maximum_original_residual_decimal'] = brief(max(after))
        result['constraints'] = [dict(constraint=i, residual=str(v), residual_decimal=brief(v),
            below_negative_margin=v < -linear_margin) for i, v in enumerate(after, 1)]
        alpha = physical[integer(original['alpha_block'])-1][0][0]
        require(alpha == rational(independent['alpha']), 'original verifier alpha differs')
        result.update(alpha=str(alpha), alpha_decimal=brief(alpha),
            upper_decimal=independent['upper_decimal'],
            rational_strict_upper_bound=independent['rational_strict_upper_bound'],
            candidate_integer_bound=alpha.__floor__(), alpha_and_scalar_slacks_preserved=True)
        result['verified'] = (result['original_verifier_verified'] and
            result['all_shifted_blocks_strictly_positive'] and result['all_residuals_below_negative_margin'])
        result['certified_integer_bound'] = alpha.__floor__() if result['verified'] else None
    except Exception as error:
        result['error_type'] = type(error).__name__
        result['error'] = str(error)
    result['verification_wall_seconds'] = time.monotonic()-started
    write_json(output/'verification.json', result)
    summary = dict(verified=result['verified'],
        integer_bound=result.get('certified_integer_bound'),
        alpha_exact=result.get('alpha'), alpha_decimal=result.get('alpha_decimal'),
        upper_decimal=result.get('upper_decimal'),
        all_original_constraints_strict=result.get('all_original_constraints_strict', False),
        maximum_original_residual_exact=result.get('maximum_original_residual'),
        maximum_original_residual_decimal=result.get('maximum_original_residual_decimal'),
        all_shifted_matrices_positive_definite=result.get('all_shifted_blocks_strictly_positive', False),
        all_residuals_below_negative_margin=result.get('all_residuals_below_negative_margin', False),
        metadata_sha256=result['metadata_sha256'], candidate_sha256=result['candidate_sha256'],
        original_metadata_sha256=result['original_metadata_sha256'],
        verifier_sha256=result['verifier_sha256'],
        representation=result.get('representation'), case=result.get('case'),
        matrix_margin=result.get('matrix_margin'), linear_margin=result.get('linear_margin'),
        all_residuals_exactly_preserved=result.get('all_residuals_exactly_preserved', False),
        verification_wall_seconds=result['verification_wall_seconds'],
        error=result.get('error'))
    write_json(output/'verification_summary.json', summary)
    print(json.dumps({key:result.get(key) for key in ('verified', 'representation', 'case',
        'upper_decimal', 'certified_integer_bound', 'all_shifted_blocks_strictly_positive',
        'all_residuals_below_negative_margin', 'maximum_original_residual_decimal', 'error')}, indent=2), flush=True)
    return bool(result['verified'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--original-metadata', required=True, type=Path)
    p.add_argument('--metadata', required=True, type=Path)
    p.add_argument('--candidate', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--model-coefficients', type=Path,
                   help='optional builder physical-M coefficient TSV, excluding auxiliary slacks')
    p.add_argument('--matrix-margin', help='require metadata to state exactly this matrix protection')
    p.add_argument('--linear-margin', help='require metadata to state exactly this scalar protection')
    p.add_argument('--digits', type=int, default=6)
    args = p.parse_args()
    require(args.digits >= 0, 'negative bound display digits')
    return verify(args)


if __name__ == '__main__':
    sys.exit(0 if main() else 1)
