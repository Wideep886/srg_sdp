#!/usr/bin/env python3
"""Independently audit exact-to-decimal SDPA serialization and error bounds.

Uses only Python standard-library rational arithmetic. Does not import the
builder or solver parser and does not run a solver. It audits decimal input
numbers as exact rationals; it does not pretend that GMP performs no further
input/iteration rounding. Original-model exact certification handles all
remaining solver errors independently of this serialization check.
"""
from decimal import Decimal, localcontext
from fractions import Fraction as F
from pathlib import Path
import argparse
import hashlib
import json
import re
import sys
import time

if hasattr(sys, 'set_int_max_str_digits'):
    sys.set_int_max_str_digits(0)

ETA = F('1e-30')
SHIFT = 2 * ETA
RESERVE = F('1e-26')


def require(condition, message):
    if not condition:
        raise ValueError(message)


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


def brief(q):
    with localcontext() as ctx:
        ctx.prec = 18
        return str(Decimal(q.numerator) / Decimal(q.denominator))


def power10(exponent):
    return F(10 ** exponent) if exponent >= 0 else F(1, 10 ** (-exponent))


def nearest_decimal(q, significant_digits):
    """Exact nearest-even significant-decimal rounding, independent of Decimal."""
    if not q:
        return F(0)
    positive = abs(q)
    exponent = len(str(positive.numerator)) - len(str(positive.denominator))
    while positive < power10(exponent):
        exponent -= 1
    while positive >= power10(exponent + 1):
        exponent += 1
    unit = power10(exponent - significant_digits + 1)
    scaled = positive / unit
    integer, remainder = divmod(scaled.numerator, scaled.denominator)
    if 2 * remainder > scaled.denominator or (
            2 * remainder == scaled.denominator and integer % 2):
        integer += 1
    rounded = integer * unit
    return rounded if q > 0 else -rounded


def parse_exact_tsv(path, constraint_count, dimensions):
    terms = {}
    for number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip() or line.startswith('#'):
            continue
        parts = line.split('\t')
        require(len(parts) == 5, f'{path}: malformed line {number}')
        ci, block, row, column = map(int, parts[:4])
        require(1 <= ci <= constraint_count, 'TSV constraint index out of range')
        require(1 <= block <= len(dimensions), 'TSV block index out of range')
        require(1 <= row <= column <= dimensions[block - 1],
                'TSV matrix indices invalid')
        value = F(parts[4])
        key = (ci, block, row, column)
        require(key not in terms, 'duplicate exact TSV entry')
        require(value != 0, 'zero exact TSV entry should have been omitted')
        terms[key] = value
    return terms


def solver_block_layout(dimensions):
    """Derive the solver's positive-size blocks from logical proof blocks.

    Empty kernel matrices remain in metadata and exact coefficient tables.
    SDPA has no zero-order block, so its block numbers enumerate only the
    positive dimensions, in the same order. The mapping is derived rather
    than accepted from builder-supplied metadata.
    """
    require(isinstance(dimensions, list) and dimensions and all(
        isinstance(n, int) and not isinstance(n, bool) and n >= 0 for n in dimensions),
        'logical block dimensions must be nonnegative integers')
    logical_blocks = [block for block, dim in enumerate(dimensions, 1) if dim > 0]
    require(logical_blocks, 'SDPA requires at least one positive-size block')
    return logical_blocks, [dimensions[block - 1] for block in logical_blocks]


def parse_dat(path, constraint_count, dimensions):
    logical_blocks, solver_dimensions = solver_block_layout(dimensions)
    lines = [x.strip() for x in path.read_text().splitlines() if x.strip()]
    require(len(lines) >= 4, 'truncated SDPA input')
    require(lines[0].split() == [str(constraint_count)], 'SDPA constraint header mismatch')
    require(lines[1].split() == [str(len(solver_dimensions))], 'SDPA block header mismatch')
    require(list(map(int, lines[2].split())) == solver_dimensions, 'SDPA dimensions mismatch')
    rhs = [F(x) for x in lines[3].split()]
    require(len(rhs) == constraint_count, 'SDPA RHS header length mismatch')
    entries = {}
    for line in lines[4:]:
        fields = line.split()
        require(len(fields) == 5, 'malformed SDPA entry')
        ci, block, row, column = map(int, fields[:4])
        require(0 <= ci <= constraint_count, 'SDPA matrix index out of range')
        require(1 <= block <= len(solver_dimensions), 'SDPA block index out of range')
        require(1 <= row <= column <= solver_dimensions[block - 1],
                'SDPA matrix coordinate out of range')
        key = (ci, logical_blocks[block - 1], row, column)
        require(key not in entries, 'duplicate SDPA entry')
        entries[key] = F(fields[4])
    return rhs, entries


def parse_solver_y_mat(path, dimensions):
    """Read SDPA yMat with a separate, nonrecursive exact-decimal parser.

    The root contains matrix blocks; a scalar block may be {number} or
    {{number}}, and a dense block contains comma-separated rows of
    comma-separated numbers. SDPA separates root blocks by whitespace;
    a single comma between root blocks is also accepted. No fractions,
    nonfinite numbers, missing/extra entries, or repeated yMat assignments
    are accepted. Parsing uses fixed-depth loops, never Python recursion.
    Zero-size logical blocks are omitted from yMat and restored as [] after
    parsing, preserving every proof block's metadata index.
    """
    logical_blocks, solver_dimensions = solver_block_layout(dimensions)
    text = Path(path).read_text()
    assignments = list(re.finditer(r'(?<![A-Za-z0-9_])yMat\s*=', text))
    require(len(assignments) == 1, 'solver output must contain exactly one yMat assignment')
    line_start = text.rfind('\n', 0, assignments[0].start()) + 1
    require(not text[line_start:assignments[0].start()].strip(),
            'yMat assignment must start a solver-output line')
    position = assignments[0].end()
    number = re.compile(r'[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?')

    def space():
        nonlocal position
        while position < len(text) and text[position].isspace():
            position += 1

    def take(expected):
        nonlocal position
        space()
        require(position < len(text) and text[position] == expected,
                f'malformed yMat: expected {expected!r} at character {position}')
        position += 1

    def decimal_value():
        nonlocal position
        space()
        match = number.match(text, position)
        require(match is not None, f'malformed yMat decimal at character {position}')
        position = match.end()
        return F(match.group())

    take('{')
    matrices = []
    for block, dimension in enumerate(solver_dimensions, 1):
        space()
        if block > 1 and position < len(text) and text[position] == ',':
            position += 1
        take('{')
        space()
        if dimension == 1 and position < len(text) and text[position] != '{':
            matrix = [[decimal_value()]]
        else:
            matrix = []
            for row in range(dimension):
                if row:
                    take(',')
                take('{')
                values = []
                for column in range(dimension):
                    if column:
                        take(',')
                    values.append(decimal_value())
                take('}')
                matrix.append(values)
        take('}')
        matrices.append(matrix)
    take('}')
    # Timing/statistics text may follow the matrix on later lines. An extra
    # brace structure is never a valid SDPA timing suffix.
    suffix = text[position:]
    require(re.match(r'[ \t]*(?:\r?\n|$)', suffix) is not None,
            'unexpected data after the yMat closing brace')
    require('{' not in suffix and '}' not in suffix,
            'extra brace or matrix data after yMat')
    restored = [[] for _ in dimensions]
    for logical_block, matrix in zip(logical_blocks, matrices):
        restored[logical_block - 1] = matrix
    return restored


def expected_shifts(original, metadata):
    require(metadata['representation'] in ('full', 'compressed'), 'unknown representation')
    require(len(original['blocks']) == len(metadata['blocks']), 'changed number of blocks')
    result = []
    for old, new in zip(original['blocks'], metadata['blocks']):
        require(old['kind'] == new['kind'], 'changed block kind')
        if old['kind'] != 'kernel':
            require(new['dim'] == old['dim'], 'changed auxiliary block dimension')
            result.append([F(0)] * new['dim'])
            continue
        require((old['m'], old['l']) == (new['m'], new['l']), 'changed reference/degree')
        compressed = metadata['representation'] == 'compressed' and old['m'] > 0 and old['l'] == 0
        if compressed:
            m = old['m']
            gram = [[F(x) for x in row] for row in old['gram']]
            reference_labels = {tuple(gram[i][j] for i in range(m)) for j in range(m)}
            old_labels = [tuple(F(x) for x in u) for u in old['states']]
            reference_count = sum(u in reference_labels for u in old_labels)
            require(reference_count == m, 'reference labels incomplete')
            require(new['dim'] == old['dim'] - m + 1, 'compressed dimension mismatch')
            require(new['states'][0] == 'reference_sum', 'compressed reference coordinate misplaced')
            result.append([SHIFT * m] + [SHIFT] * (new['dim'] - 1))
        else:
            require(new['dim'] == old['dim'], 'unexpected dimension change')
            result.append([SHIFT] * new['dim'])
    return result


def audit_model(model, run):
    model, run = Path(model).resolve(), Path(run).resolve()
    build = read_json(model / 'build.json')
    metadata = read_json(model / 'metadata.json')
    original = read_json(model / 'original_metadata.json')
    dimensions = [b['dim'] for b in metadata['blocks']]
    constraints = metadata['constraints']
    count = len(constraints)
    require(constraints == original['constraints'], 'modeled configuration list differs from original')
    require(F(metadata['modeled_matrix_shift']) == SHIFT, 'unexpected modeled shift')
    require(F(metadata['required_matrix_margin']) == ETA, 'unexpected required matrix margin')
    require(F(metadata['modeled_linear_margin']) == RESERVE, 'unexpected modeled linear reserve')
    require(F(metadata['required_linear_margin']) == RESERVE, 'unexpected required linear margin')
    require(metadata['original_metadata_sha256'] == sha(model / 'original_metadata.json'),
            'original metadata hash mismatch')
    require(build['input_sha256'] == sha(model / 'problem.dat-s'), 'build/input hash mismatch')
    require(build['metadata_sha256'] == sha(model / 'metadata.json'), 'build/metadata hash mismatch')
    require(build['parameter_sha256'] == sha(model / 'solver.par'), 'build/parameter hash mismatch')
    shifts = [[F(x) for x in row] for row in read_json(model / 'physical_shifts.json')]
    require(shifts == expected_shifts(original, metadata), 'physical shifts do not equal 2 eta I / 2 eta D')
    terms = parse_exact_tsv(model / 'exact_coefficients.tsv', count, dimensions)
    rhs_actual, entries = parse_dat(model / 'problem.dat-s', count, dimensions)
    alpha = metadata['alpha_block']
    require(dimensions[alpha - 1] == 1, 'alpha block is not scalar')
    expected = {(0, alpha, 1, 1): F(-1)}
    for key, value in terms.items():
        expected[key] = value if key[2] == key[3] else value / 2
    rhs_exact = []
    by_constraint = [[] for _ in constraints]
    for (ci, block, row, column), value in terms.items():
        by_constraint[ci - 1].append((block, row, column, value))
    for ci, constraint in enumerate(constraints, 1):
        block = constraint['slack_block']
        require(dimensions[block - 1] == 1 and metadata['blocks'][block - 1]['kind'] == 'slack',
                'slack block metadata invalid')
        key = (ci, block, 1, 1)
        require(key not in expected, 'auxiliary slack duplicated in exact original terms')
        expected[key] = F(1)
        correction = sum((value * shifts[b - 1][i - 1]
                          for b, i, j, value in by_constraint[ci - 1] if i == j), F(0))
        rhs_exact.append(-F(constraint['rhs']) - RESERVE - correction)
    require(set(entries) == set(expected), 'SDPA sparse support differs from objective/terms/slacks')
    digits = build['export_significant_decimal_digits']
    require(digits == 600, 'unexpected export precision')
    entry_errors = {}
    for key, exact_value in expected.items():
        actual = entries[key]
        require(actual == nearest_decimal(exact_value, digits), 'SDPA coefficient not correctly rounded at 600 digits')
        entry_errors[key] = actual - exact_value
    rhs_errors = []
    for exact_value, actual in zip(rhs_exact, rhs_actual):
        require(actual == nearest_decimal(exact_value, digits), 'SDPA RHS not correctly rounded at 600 digits')
        rhs_errors.append(actual - exact_value)
    max_entry = max(map(abs, entry_errors.values()), default=F(0))
    max_rhs = max(map(abs, rhs_errors), default=F(0))
    max_error = max(max_entry, max_rhs)
    require(max_error == F(build['maximum_export_rounding_error_exact']),
            'builder maximum rounding error does not equal independent recomputation')
    require(entries[(0, alpha, 1, 1)] == -1, 'objective is not exactly minus alpha')
    for ci, constraint in enumerate(constraints, 1):
        require(entries[ci, constraint['slack_block'], 1, 1] == 1, 'slack coefficient changed')

    candidate = read_json(run / 'candidate.json')
    require(candidate['metadata_sha256'] == sha(model / 'metadata.json'), 'candidate/metadata mismatch')
    require(candidate['solver_output_sha256'] == sha(run / 'solver.out'), 'candidate/output hash mismatch')
    require(candidate['physical_shifts_sha256'] == sha(model / 'physical_shifts.json'), 'candidate/shifts hash mismatch')
    require(isinstance(candidate['matrices'], list) and all(
        isinstance(matrix, list) and all(isinstance(row, list) for row in matrix)
        for matrix in candidate['matrices']), 'candidate matrices must be nested arrays')
    require(all(isinstance(value, (str, int)) and not isinstance(value, bool)
                for matrix in candidate['matrices'] for row in matrix for value in row),
            'candidate entries must be exact rational strings or integers')
    physical = [[[F(x) for x in row] for row in matrix] for matrix in candidate['matrices']]
    require(len(physical) == len(dimensions), 'candidate number of blocks mismatch')
    solver_matrices = parse_solver_y_mat(run / 'solver.out', dimensions)
    raw = []
    output_match_checks = []
    for block, (M, dimension, shift, Y) in enumerate(zip(physical, dimensions, shifts, solver_matrices), 1):
        require(len(M) == dimension and all(len(row) == dimension for row in M), 'candidate shape mismatch')
        require(all(M[i][j] == M[j][i] for i in range(dimension) for j in range(dimension)),
                'candidate is not exactly symmetric')
        for i in range(dimension):
            for j in range(dimension):
                require(M[i][j] == Y[i][j] + (shift[i] if i == j else 0),
                        f'candidate differs from yMat plus exact shift at block {block}, row {i+1}, column {j+1}')
        raw.append([[M[i][j] - (shift[i] if i == j else 0)
                     for j in range(dimension)] for i in range(dimension)])
        output_match_checks.append(dict(block=block, kind=metadata['blocks'][block-1]['kind'],
            dimension=dimension, scalar_entries_checked=dimension*dimension, matches_exactly=True))

    residual_checks = []
    largest_difference = F(0)
    largest_bound = F(0)
    largest_numeric_equality_residual = F(0)
    for ci, (constraint, exact_b, numeric_b) in enumerate(zip(constraints, rhs_exact, rhs_actual), 1):
        exact_guard = -exact_b
        decimal_guard = -numeric_b
        bound = abs(rhs_errors[ci - 1])
        original_physical = F(constraint['rhs'])
        for block, i, j, value in by_constraint[ci - 1]:
            Xij = raw[block - 1][i - 1][j - 1]
            exact_guard += value * Xij
            factor = 1 if i == j else 2
            decimal_guard += factor * entries[ci, block, i, j] * Xij
            bound += factor * abs(entry_errors[ci, block, i, j]) * abs(Xij)
            original_physical += value * physical[block - 1][i - 1][j - 1]
        require(exact_guard == original_physical + RESERVE,
                'restoring shifts does not recover exact original residual plus reserve')
        difference = abs(decimal_guard - exact_guard)
        require(difference <= bound, 'weighted serialization error bound violated')
        slack = raw[constraint['slack_block'] - 1][0][0]
        numeric_equality_residual = abs(decimal_guard + slack)
        largest_difference = max(largest_difference, difference)
        largest_bound = max(largest_bound, bound)
        largest_numeric_equality_residual = max(largest_numeric_equality_residual, numeric_equality_residual)
        residual_checks.append(dict(
            constraint=ci,
            absolute_decimal_vs_exact_guard_difference=str(difference),
            weighted_serialization_error_bound=str(bound),
            difference_decimal=brief(difference), error_bound_decimal=brief(bound),
            exact_guarded_residual_decimal=brief(exact_guard),
            original_physical_residual_decimal=brief(original_physical),
            decimal_input_equality_residual_with_slack_decimal=brief(numeric_equality_residual),
            exact_bound_check_passed=True))

    return dict(
        model=model.name, status='PASS', constraints=count,
        blocks=len(dimensions), sparse_sdpa_entries=len(entries),
        exact_original_scalar_coefficients=len(terms),
        correctly_rounded_sdpa_numbers=len(entries) + len(rhs_actual),
        objective_minus_alpha_and_unit_slacks_exact=True,
        dimensions_and_sparse_support_match=True,
        physical_shifts_independently_reconstructed=True,
        candidate_matches_solver_output=True,
        solver_output_parser='independent fixed-depth iterative exact-decimal yMat parser',
        solver_output_blocks_checked=len(output_match_checks),
        solver_output_scalar_entries_checked=sum(row['scalar_entries_checked'] for row in output_match_checks),
        solver_output_match_checks=output_match_checks,
        maximum_entry_rounding_error_exact=str(max_entry),
        maximum_rhs_rounding_error_exact=str(max_rhs),
        maximum_export_rounding_error_exact=str(max_error),
        maximum_export_rounding_error_decimal=brief(max_error),
        builder_recorded_maximum_exactly_matches=True,
        run_directory=str(run),
        maximum_decimal_vs_exact_guard_difference_exact=str(largest_difference),
        maximum_decimal_vs_exact_guard_difference_decimal=brief(largest_difference),
        maximum_weighted_serialization_error_bound_exact=str(largest_bound),
        maximum_weighted_serialization_error_bound_decimal=brief(largest_bound),
        maximum_decimal_input_equality_residual_with_slack_decimal=brief(largest_numeric_equality_residual),
        all_weighted_error_checks_passed=True,
        residual_checks=residual_checks,
        sha256={str(path): sha(path) for path in (
            model / 'build.json', model / 'metadata.json', model / 'original_metadata.json',
            model / 'exact_coefficients.tsv', model / 'physical_shifts.json',
            model / 'problem.dat-s', model / 'solver.par',
            run / 'candidate.json', run / 'solver.out')})


def absolute_path(value):
    path = Path(value)
    if not path.is_absolute():
        raise argparse.ArgumentTypeError("paths must be absolute")
    return path.resolve()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, type=absolute_path)
    parser.add_argument('--run', required=True, type=absolute_path)
    parser.add_argument('--output', required=True, type=absolute_path,
                        help='JSON report path')
    args = parser.parse_args(argv)
    checked = [args.model / name for name in (
        'build.json', 'metadata.json', 'original_metadata.json', 'exact_coefficients.tsv',
        'original_exact_coefficients.tsv', 'physical_shifts.json', 'problem.dat-s', 'solver.par')]
    checked += [args.run / 'candidate.json', args.run / 'solver.out']
    require(args.output not in {path.resolve() for path in checked},
            'output would overwrite a model or solution input')
    started = time.perf_counter()
    report = dict(
        status='FAIL', script_sha256=sha(__file__),
        model_directory=str(args.model), run_directory=str(args.run),
        arithmetic='Python fractions.Fraction; independent integer nearest-even decimal rounding',
        imported_builder_or_solver_modules=False, solver_runs=0,
        scope=('Exact coefficient/RHS to decimal-file serialization and weighted effect at the specified physical candidate; '
               'every candidate entry, including alpha and auxiliary slacks, must equal independently parsed yMat plus exact shifts.'),
        limitation=('Decimal input residuals treat file numbers as exact rationals. They are not claims about '
                    'unrounded GMP input or iterations. All solver effects must still be covered by independent '
                    'exact verification of the original mathematical inequalities and PSD blocks.'))
    try:
        row = audit_model(args.model, args.run)
        report.update(status='PASS', models=1, constraints_checked=row['constraints'],
                      correctly_rounded_numbers_checked=row['correctly_rounded_sdpa_numbers'], rows=[row])
    except Exception as exc:
        report.update(error_type=type(exc).__name__, error=str(exc))
    report['elapsed_seconds'] = time.perf_counter() - started
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({key: report.get(key) for key in
                     ('status', 'constraints_checked', 'elapsed_seconds', 'error')}, ensure_ascii=False), flush=True)
    return 0 if report['status'] == 'PASS' else 1


if __name__ == '__main__':
    sys.exit(main())
