"""Fast rational-model regressions. These tests never invoke an SDP solver."""
from contextlib import redirect_stdout
from copy import deepcopy
from fractions import Fraction as F
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

SOURCE = Path(__file__).resolve().parents[2] / 'code'
FIXTURES = Path(__file__).resolve().parent / 'fixtures'
sys.path.insert(0, str(SOURCE))

import check_compression as direct
import exact_model as exact
import check_solver_io as audit
import model
import check_matrices as compressed


def embedding(n, a, b, parameters=(0, 0, 0, 0)):
    # Model tests provide sphere data directly; SRG parameter validation is a
    # separate layer. The zero tuple is a label, not a claimed SRG existence.
    return dict(n=n, a=str(a), b=str(b), parameters=list(parameters))


def read_json(path):
    return json.loads(Path(path).read_text())


def coefficients(path):
    return compressed.read_coefficients(path)


def synthetic_run(directory, output):
    """Exact zero yMat fixture: serialization-valid, mathematically infeasible."""
    output.mkdir()
    meta = read_json(directory / 'metadata.json')
    shifts = read_json(directory / 'physical_shifts.json')
    matrices, solver_blocks = [], []
    for spec, shift in zip(meta['blocks'], shifts):
        dim = spec['dim']
        matrices.append([[str(F(shift[i]) if i == j else F(0))
                          for j in range(dim)] for i in range(dim)])
        if dim:
            row = '{' + ','.join(['0'] * dim) + '}'
            solver_blocks.append('{' + ','.join([row] * dim) + '}')
    solver = output / 'solver.out'
    solver.write_text('yMat = {' + ','.join(solver_blocks) + '}\n')
    model.dump(output / 'candidate.json', dict(
        metadata_sha256=model.sha(directory / 'metadata.json'),
        solver_output_sha256=model.sha(solver),
        physical_shifts_sha256=model.sha(directory / 'physical_shifts.json'),
        matrices=matrices))
    return output


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temp.cleanup)
        cls.root = Path(cls.temp.name)
        cls.models = {}
        cases = (
            ('historical', embedding(33, '-1/9', '7/27', (550, 162, 75, 36)), 2, 5),
            ('empty', embedding(4, '-3/4', '-1/2'), 4, 1),
            ('dependent', embedding(4, '-1', '0'), 4, 1),
            ('minimum', embedding(2, '-1', '0'), 2, 0),
        )
        with redirect_stdout(io.StringIO()):
            for name, sphere, points, degree in cases:
                cls.models[name] = model.build(sphere, points, degree, cls.root / name)

    def test_sealed_550_k2_d5_coefficients_are_preserved(self):
        metadata_fixture = FIXTURES / '550_k2_d5_metadata.json'
        coefficient_fixture = FIXTURES / '550_k2_d5_coefficients.tsv'
        # These hashes are recorded in the retained witness verification report.
        self.assertEqual(model.sha(metadata_fixture),
                         'a0017fbfc0d53a00c109cfe4faa5116562225c16421a448944f5cf13a7eadbc5')
        self.assertEqual(model.sha(coefficient_fixture),
                         '7ee4eabff3bf746620004dde98f3f6630f14622426d8494b88d19cb13c1ccc03')
        old = read_json(metadata_fixture)
        old_constraints, _ = exact.reconstruct(old)
        historical = {(ci,) + key: value
                      for ci, constraint in enumerate(old_constraints, 1)
                      for key, value in constraint['terms'].items()}
        expected = coefficients(coefficient_fixture)
        self.assertEqual(historical, expected)
        directory = self.models['historical']
        self.assertEqual(coefficients(directory / 'original_exact_coefficients.tsv'), expected)
        self.assertEqual(coefficients(directory / 'exact_coefficients.tsv'), expected)
        fresh = read_json(directory / 'original_metadata.json')
        self.assertEqual([(b['kind'], b['dim']) for b in fresh['blocks']],
                         [(b['kind'], b['dim']) for b in old['blocks']])
        self.assertEqual([c['slack_block'] for c in fresh['constraints']],
                         [c['slack_block'] for c in old['constraints']])

    def test_general_scope_has_no_fixed_degree_five_requirement(self):
        for points, degree, dimension in ((2, 0, 2), (3, 2, 3), (2, 6, 2)):
            with self.subTest(points=points, degree=degree):
                meta = model.fresh_metadata(embedding(dimension, '-1/3', '1/3'), points, degree)
                _, report = exact.reconstruct(meta)
                self.assertEqual((report['k'], report['d']), (points, degree))
                self.assertEqual({b['l'] for b in meta['blocks'] if b['kind'] == 'kernel'},
                                 set(range(degree + 1)))
        minimum = read_json(self.models['minimum'] / 'original_metadata.json')
        self.assertEqual((minimum['n'], minimum['k'], minimum['d']), (2, 2, 0))

    def test_invalid_scope_and_solver_settings_are_rejected(self):
        for points, degree in ((1, 0), (7, 0), (3, 0), (True, 0), (2.5, 0),
                               (2, -1), (2, True), (2, 0.5)):
            with self.subTest(points=points, degree=degree), self.assertRaises(ValueError):
                model.fresh_metadata(embedding(2, '-1', '0'), points, degree)
        for settings in ((63, 1, '1e-16'), (1500, 0, '1e-16'),
                         (True, 1, '1e-16'), (1500, True, '1e-16'),
                         (1500, 1, '0'), (1500, 1, '-1')):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                model.parameter_text(*settings)

    def test_empty_positive_kernels_retain_logical_ids(self):
        directory = self.models['empty']
        original = read_json(directory / 'original_metadata.json')
        meta = read_json(directory / 'metadata.json')
        empty_ids = [i for i, b in enumerate(meta['blocks'], 1) if b['dim'] == 0]
        self.assertTrue(empty_ids)
        self.assertTrue(all(meta['blocks'][i - 1]['kind'] == 'kernel'
                            and meta['blocks'][i - 1]['l'] > 0 for i in empty_ids))
        self.assertEqual(len(meta['blocks']), len(original['blocks']))
        dimensions = [b['dim'] for b in meta['blocks']]
        report = read_json(directory / 'build.json')
        self.assertEqual(report['solver_logical_block_ids'],
                         [i for i, dim in enumerate(dimensions, 1) if dim > 0])
        _, entries = audit.parse_dat(directory / 'problem.dat-s', len(meta['constraints']), dimensions)
        self.assertFalse(set(empty_ids) & {key[1] for key in entries})
        self.assertTrue(exact.psd_info([])['positive_definite'])
        self.assertTrue(compressed.strict_ldl([])['positive_definite'])

    def test_boundary_generic_states_are_retained(self):
        directory = self.models['empty']
        meta = read_json(directory / 'original_metadata.json')
        boundary_ids = []
        for bi, block in enumerate(meta['blocks'], 1):
            if block['kind'] != 'kernel' or block['l'] == 0 or not block['dim']:
                continue
            gram = [[F(value) for value in row] for row in block['gram']]
            inverse = exact.inverse(gram)
            residuals = [1 - exact.bilinear(tuple(map(F, label)), inverse,
                                          tuple(map(F, label))) for label in block['states']]
            if residuals and all(value == 0 for value in residuals):
                boundary_ids.append(bi)
        self.assertTrue(boundary_ids)
        terms = coefficients(directory / 'original_exact_coefficients.tsv')
        self.assertFalse(set(boundary_ids) & {key[1] for key in terms})

    def test_dependent_reference_policy_is_explicit_and_complete(self):
        directory = self.models['dependent']
        meta = read_json(directory / 'original_metadata.json')
        self.assertEqual(meta['zero_dependent_references'], [[2, 0]])
        _, exact_report = exact.reconstruct(meta)
        _, _, _, direct_report = direct.validate_metadata(meta)
        self.assertEqual(direct_report['zero_dependent_references'], [(2, 0)])
        self.assertEqual(exact_report['zero_dependent_references'], [(2, 0)])
        # The singular Gram configurations remain among the inequalities.
        self.assertGreater(exact_report['singular_configuration_counts']['2'], 0)
        self.assertEqual(direct_report['configuration_count'], len(meta['constraints']))
        for policy in (None, 'reject', 'unknown'):
            bad = deepcopy(meta)
            if policy is None:
                bad.pop('dependent_reference_policy')
            else:
                bad['dependent_reference_policy'] = policy
            with self.subTest(policy=policy):
                with self.assertRaises(ValueError):
                    direct.validate_metadata(bad)
                with self.assertRaises(ValueError):
                    exact.reconstruct(bad)

    def test_direct_union_audit_checks_every_model(self):
        for name, directory in self.models.items():
            with self.subTest(name=name):
                result = direct.check_files(directory / 'original_metadata.json',
                    directory / 'exact_coefficients.tsv',
                    directory / 'original_exact_coefficients.tsv')
                self.assertEqual(result['status'], 'PASS')
                self.assertTrue(result['positive_degree_unchanged_checked'])
        directory = self.models['empty']
        terms = coefficients(directory / 'exact_coefficients.tsv')
        meta = read_json(directory / 'metadata.json')
        key = next(key for key in terms if meta['blocks'][key[1] - 1]['kind'] == 'kernel'
                   and meta['blocks'][key[1] - 1]['l'] == 0)
        terms[key] += 1
        corrupted = self.root / 'wrong_coefficients.tsv'
        compressed.write_coefficients(corrupted, terms)
        with self.assertRaises(ValueError):
            direct.check_files(directory / 'original_metadata.json', corrupted,
                               directory / 'original_exact_coefficients.tsv')

    def test_physical_shifts_and_fixed_margins_are_preserved(self):
        for name, directory in self.models.items():
            with self.subTest(name=name):
                original = read_json(directory / 'original_metadata.json')
                meta = read_json(directory / 'metadata.json')
                shifts = [[F(x) for x in row] for row in read_json(directory / 'physical_shifts.json')]
                self.assertEqual(shifts, audit.expected_shifts(original, meta))
                self.assertEqual(F(meta['required_matrix_margin']), F('1e-30'))
                self.assertEqual(F(meta['modeled_matrix_shift']), F('2e-30'))
                self.assertEqual(F(meta['required_linear_margin']), F('1e-26'))
                self.assertEqual(F(meta['modeled_linear_margin']), F('1e-26'))
                for spec, diagonal in zip(meta['blocks'], shifts):
                    if spec['kind'] != 'kernel':
                        self.assertEqual(diagonal, [F(0)])
                    elif spec.get('compression') == 'reference-sum':
                        self.assertEqual(diagonal[0], F('2e-30') * spec['m'])

    def test_serialization_pass_is_not_mathematical_acceptance(self):
        directory = self.models['empty']
        run = synthetic_run(directory, self.root / 'synthetic_run')
        report = audit.audit_model(directory, run)
        self.assertEqual(report['status'], 'PASS')
        self.assertTrue(report['candidate_matches_solver_output'])
        self.assertTrue(any(row['dimension'] == 0 for row in report['solver_output_match_checks']))
        args = SimpleNamespace(original_metadata=directory / 'original_metadata.json',
            metadata=directory / 'metadata.json', candidate=run / 'candidate.json',
            output=self.root / 'infeasible_exact_check',
            model_coefficients=directory / 'exact_coefficients.tsv',
            matrix_margin='1e-30', linear_margin='1e-26', digits=6)
        with redirect_stdout(io.StringIO()):
            self.assertFalse(compressed.verify(args))
        exact_report = read_json(args.output / 'verification.json')
        self.assertTrue(exact_report['all_shifted_blocks_strictly_positive'])
        self.assertTrue(exact_report['builder_coefficients_exactly_match'])
        self.assertFalse(exact_report['all_residuals_below_negative_margin'])
        self.assertFalse(exact_report['verified'])
        self.assertIsNone(exact_report['certified_integer_bound'])
        candidate = read_json(run / 'candidate.json')
        candidate['matrices'][0][0][0] = '1'
        model.dump(run / 'candidate.json', candidate)
        with self.assertRaisesRegex(ValueError, 'candidate.*yMat|solver output|printed'):
            audit.audit_model(directory, run)


if __name__ == '__main__':
    unittest.main()
