"""Independent finite-sum/Bareiss audit checks; no numerical solver is run."""
import ast
from contextlib import redirect_stdout
from copy import deepcopy
from fractions import Fraction as F
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

SOURCE = Path(__file__).resolve().parents[2] / 'code'
FIXTURES = Path(__file__).resolve().parent / 'fixtures'
sys.path.insert(0, str(SOURCE))
import independent_check as independent
import exact_model as exact
import check_compression as compression
import check_matrices as matrices
import check_solver_io as solver_io
import verify as supervisor
import model


def sphere(n, a, b):
    return dict(n=n, a=a, b=b, parameters=[0, 0, 0, 0])


def flattened(constraints):
    return {(ci,) + key: value for ci, row in enumerate(constraints, 1)
            for key, value in row['terms'].items()}


class IndependentChecks(unittest.TestCase):
    def test_exact_json_readers_reject_ambiguous_and_nonfinite_input(self):
        readers = (independent.read, exact.read_json, compression.read_json,
                   matrices.read_json, solver_io.read_json, supervisor.read_json)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'input.json'
            for content in ('{"dim": 1, "dim": 2}', '{"x": {"a": 0, "a": 1}}',
                            '{"x": NaN}', '{"x": Infinity}', '{"x": -Infinity}'):
                path.write_text(content)
                for reader in readers:
                    with self.subTest(reader=reader.__module__, content=content):
                        with self.assertRaises(ValueError):
                            reader(path)
            path.write_text('{"x": 0.12345678901234567890123456789}')
            for reader in readers:
                self.assertEqual(reader(path)['x'], '0.12345678901234567890123456789')

    def test_sparse_coefficients_reject_duplicates_zeros_and_invalid_indices(self):
        readers = (independent.load_tsv, matrices.read_coefficients,
                   lambda path: compression.read_tsv(path, {1: 2}, 1),
                   lambda path: solver_io.parse_exact_tsv(path, 1, [2]))
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'coefficients.tsv'
            for content in ('1\t1\t1\t1\t1\n1\t1\t1\t1\t2\n',
                            '1\t1\t1\t1\t0\n', '0\t1\t1\t1\t1\n',
                            '1\t1\t2\t1\t1\n'):
                path.write_text(content)
                for reader in readers:
                    with self.subTest(reader=reader, content=content):
                        with self.assertRaises(ValueError):
                            reader(path)

    def test_report_paths_cannot_overwrite_model_inputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            original = root / 'original_metadata.json'
            original.write_text('preserve this input\n')
            with self.assertRaisesRegex(ValueError, 'overwrite'):
                solver_io.main(['--model', str(root), '--run', str(root),
                                '--output', str(original)])
            with self.assertRaisesRegex(ValueError, 'overwrite'):
                compression.main(['--metadata', str(original),
                                  '--compressed-coefficients', str(root / 'coefficients.tsv'),
                                  '--output', str(original)])
            with self.assertRaisesRegex(ValueError, 'overwrite'):
                independent.audit(root, root, original)
            self.assertEqual(original.read_text(), 'preserve this input\n')

    def test_module_imports_only_the_standard_library(self):
        tree = ast.parse((SOURCE / 'independent_check.py').read_text())
        allowed = {'fractions', 'itertools', 'collections', 'functools', 'math', 'pathlib',
                   'argparse', 'hashlib', 'json', 'sys', 'time'}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                self.assertTrue(all(alias.name in allowed for alias in node.names))
            elif isinstance(node, ast.ImportFrom):
                self.assertIn(node.module, allowed)

    def test_finite_sum_and_chebyshev_limit_against_recurrence(self):
        self.assertEqual(independent.finite_coefficients(2, 0), (F(1),))
        self.assertEqual(independent.finite_coefficients(2, 6), (F(32), F(-48), F(18), F(-1)))
        for p in (2, 3, 4, 33):
            for w, s in ((F(2, 3), F(7, 8)), (F(0), F(0)), (F(-3, 5), F(1))):
                expected = [F(1), w]
                for degree in range(2, 13):
                    expected.append(F(2*degree+p-4, degree+p-3)*w*expected[-1]
                                    - F(degree-1, degree+p-3)*s*expected[-2])
                with self.subTest(p=p, w=w, s=s):
                    self.assertEqual(independent.values(p, w, s, 12), tuple(expected))
                    self.assertEqual(independent.values(p, F(1), F(1), 12), (F(1),) * 13)

    def test_coefficients_match_sealed_fixture_and_small_general_models(self):
        historical = json.loads((FIXTURES / '550_k2_d5_metadata.json').read_text())
        expected = independent.load_tsv(FIXTURES / '550_k2_d5_coefficients.tsv')
        self.assertEqual(independent.reconstruct(historical)[0], expected)
        for n, k, degree, a, b in ((2, 2, 0, '-1', '0'), (2, 2, 6, '-1/3', '1/3'),
                                   (4, 4, 1, '-1', '0'), (4, 4, 6, '-3/4', '-1/2')):
            meta = model.fresh_metadata(sphere(n, a, b), k, degree)
            with self.subTest(n=n, k=k, degree=degree, a=a):
                self.assertEqual(independent.reconstruct(meta)[0], flattened(exact.reconstruct(meta)[0]))
        empty = model.fresh_metadata(sphere(4, '-3/4', '-1/2'), 4, 6)
        self.assertTrue(any(b['dim'] == 0 for b in empty['blocks']))
        self.assertEqual(independent.positive_leading_minors(()), 0)

    def test_complete_feasible_geometry_rejects_omissions_and_unrealizable_types(self):
        meta = model.fresh_metadata(sphere(4, '-3/4', '-1/2'), 4, 1)
        geometry = independent.geometry(meta)
        # No size-four configuration is feasible here; that is not a missing orbit.
        self.assertEqual(geometry['counts'][4]['types'], 0)
        missing = deepcopy(meta)
        missing['constraints'].pop(0)
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            independent.geometry(missing)
        duplicate = deepcopy(meta)
        duplicate['constraints'].append(deepcopy(meta['constraints'][0]))
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            independent.geometry(duplicate)
        unrealizable = deepcopy(meta)
        unrealizable['constraints'].append(dict(size=3, rhs='0', slack_block=1,
            gram=[['1', '-3/4', '-3/4'], ['-3/4', '1', '-3/4'], ['-3/4', '-3/4', '1']]))
        with self.assertRaisesRegex(ValueError, 'unrealizable'):
            independent.geometry(unrealizable)

    def test_dependent_references_need_zero_and_reference_layers_are_complete(self):
        meta = model.fresh_metadata(sphere(4, '-1', '0'), 4, 1)
        independent.reconstruct(meta)
        for policy in (None, 'reject', 'unknown'):
            bad = deepcopy(meta)
            if policy is None:
                bad.pop('dependent_reference_policy')
            else:
                bad['dependent_reference_policy'] = policy
            with self.subTest(policy=policy), self.assertRaises(ValueError):
                independent.reconstruct(bad)
        bad = deepcopy(meta)
        kernel = next(block for block in bad['blocks'] if block['kind'] == 'kernel' and block['l'] == 1)
        kernel['l'] = 0
        with self.assertRaisesRegex(ValueError, 'duplicate reference layer'):
            independent.reconstruct(bad)
        bad = deepcopy(meta)
        kernel = next(block for block in bad['blocks'] if block['kind'] == 'kernel' and block['m'] == 2)
        kernel['gram'] = [['1', '-1'], ['-1', '1']]
        with self.assertRaisesRegex(ValueError, 'independent realizable'):
            independent.reconstruct(bad)

    def test_bareiss_detects_indefinite_and_margin_boundary(self):
        independent.selftest()
        for matrix in (((F(1), F(2)), (F(2), F(1))), ((F(0),),)):
            with self.assertRaises(ValueError):
                independent.positive_leading_minors(matrix)
        self.assertEqual(independent.gram_rank_by_all_minors(((F(1), F(-1)), (F(-1), F(1))))[0], 1)
        with self.assertRaises(ValueError):
            independent.gram_rank_by_all_minors(((F(1), F(2)), (F(2), F(1))))

    def test_audit_passes_without_saved_reports_and_rejects_tampering(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with redirect_stdout(io.StringIO()):
                directory = model.build(sphere(2, '-1', '0'), 2, 2, root / 'model')
            solution = root / 'solution'
            solution.mkdir()
            meta = json.loads((directory / 'metadata.json').read_text())
            # A hand-derived feasible dual: positive kernel scalars 2 EPS,4,2,
            # alpha=8. Both pair residuals are -2+4 EPS and singleton -1+2 EPS.
            matrices = []
            for block in meta['blocks']:
                value = (F(8) if block['kind'] == 'alpha' else F(0) if block['kind'] == 'slack'
                         else (2 * independent.EPS, F(4), F(2))[block['l']])
                matrices.append([[str(value)]])
            original = dict(metadata_sha256=model.sha(directory / 'metadata.json'), matrices=matrices)
            candidate = solution / 'candidate.json'
            output = root / 'audit.json'
            model.dump(candidate, original)
            report = independent.audit(directory, solution, output)
            self.assertEqual(report['status'], 'PASS', report.get('error'))
            self.assertTrue(report['verified'])
            self.assertEqual(report['integer_bound'], 8)
            self.assertTrue(report['no_project_code_imported'])
            self.assertEqual(report['solver_runs'], 0)
            self.assertFalse((solution / 'verification').exists())
            # A physical certificate must refer to exactly the same list of
            # configurations before and after matrix compression.
            metadata_path = directory / 'metadata.json'
            before_metadata = metadata_path.read_bytes()
            changed_metadata = deepcopy(meta)
            changed_metadata['constraints'][0]['rhs'] = '0'
            model.dump(metadata_path, changed_metadata)
            changed_candidate = deepcopy(original)
            changed_candidate['metadata_sha256'] = model.sha(metadata_path)
            model.dump(candidate, changed_candidate)
            failed = independent.audit(directory, solution, output)
            self.assertEqual(failed['status'], 'FAIL')
            self.assertIn('configuration lists differ', failed['error'])
            metadata_path.write_bytes(before_metadata)
            for block, value in ((0, '0'), (0, '1'), (1, str(independent.EPS))):
                bad = deepcopy(original)
                bad['matrices'][block][0][0] = value
                model.dump(candidate, bad)
                failed = independent.audit(directory, solution, output)
                with self.subTest(block=block, value=value):
                    self.assertEqual(failed['status'], 'FAIL')
                    self.assertFalse(failed['verified'])
                    self.assertIsNone(failed['integer_bound'])
            model.dump(candidate, original)
            coefficients = directory / 'original_exact_coefficients.tsv'
            coefficients.write_text(coefficients.read_text().replace('\t-1\n', '\t-2\n', 1))
            failed = independent.audit(directory, solution, output)
            self.assertEqual(failed['status'], 'FAIL')
            self.assertIn('coefficients disagree', failed['error'])
            with self.assertRaises(ValueError):
                independent.audit(directory, solution, candidate)


if __name__ == '__main__':
    unittest.main()
