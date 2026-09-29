"""Workflow boundary tests; no numerical solver is executed."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'code'))
import workflow


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.folder = Path(self.temporary.name).resolve() / 'case'

    def prepare(self):
        with redirect_stdout(io.StringIO()):
            return workflow.prepare([4, 2, 0, 2], 'auto', 2, 0, self.folder)

    def ready_for_verification(self):
        self.prepare()
        (self.folder / 'solution').mkdir()
        (self.folder / 'solution/candidate.json').write_text('{}')
        (self.folder / 'solution/solver.out').write_text('synthetic workflow fixture')

    def success_summary(self, output, bound=3):
        # These tests exercise the workflow boundary with synthetic child
        # reports; mathematical verification is covered by the checker tests.
        report = dict(status='PASS', verified=True, certified_integer_bound=bound,
            inputs_unchanged=True, input_sha256=workflow.certificate_verifier.input_hashes(
                self.folder / 'model', self.folder / 'solution'))
        (output / 'verification_summary.json').write_text(json.dumps(report))
        return report

    def success_independent(self, output, bound=3):
        paths = [self.folder / 'model/original_metadata.json', self.folder / 'model/metadata.json',
                 self.folder / 'model/original_exact_coefficients.tsv', self.folder / 'solution/candidate.json']
        report = dict(status='PASS', verified=True, integer_bound=bound, inputs_unchanged=True,
            inputs={str(path): workflow.model.sha(path) for path in paths},
            source_sha256=workflow.model.sha(workflow.ROOT / 'code/independent_check.py'))
        (output / 'independent.json').write_text(json.dumps(report))
        return report

    @staticmethod
    def replace_json(path, update):
        value = json.loads(path.read_text())
        update(value)
        path.write_text(json.dumps(value))

    def test_prepare_records_no_certificate_and_rechecks_embedding(self):
        self.prepare()
        entry, embedding = workflow.check_embedding(self.folder)
        self.assertEqual(entry['parameters'], [4, 2, 0, 2])
        self.assertEqual((embedding['n'], embedding['a'], embedding['b']), (2, '-1', '0'))
        result = json.loads((self.folder / 'result.json').read_text())
        self.assertEqual(result['status'], 'PREPARED_NOT_CERTIFIED')
        self.assertIs(result['verified'], False)
        self.assertFalse((self.folder / 'solution').exists())

    def test_unsupported_input_does_not_start_build(self):
        with redirect_stdout(io.StringIO()), patch('workflow.model.build') as build:
            with self.assertRaisesRegex(ValueError, 'unsupported'):
                workflow.prepare([5, 2, 0, 1], 'auto', 2, 0, self.folder)
        build.assert_not_called()
        self.assertFalse(self.folder.exists())

    def test_prepare_normalizes_exact_integer_inputs(self):
        with redirect_stdout(io.StringIO()):
            workflow.prepare({'v': '4', 'k': '2', 'lambda': '0', 'mu': '2'},
                             'auto', 2, 0, self.folder)
        entry, embedding = workflow.check_embedding(self.folder)
        self.assertEqual(entry['parameters'], [4, 2, 0, 2])
        self.assertTrue(all(type(value) is int for value in entry['parameters']))

    def test_graph_linkage_mutation_during_verification_prevents_acceptance(self):
        self.ready_for_verification()
        output = self.folder / 'verification'

        def change_input_and_report_success(command):
            self.replace_json(self.folder / 'input.json', lambda value: value.update(degree=1))
            self.success_summary(output)
            return subprocess.CompletedProcess(command, 0)

        with redirect_stdout(io.StringIO()), \
                patch('workflow.run_check', side_effect=change_input_and_report_success):
            result = workflow.verify(self.folder, output)
        self.assertEqual(result['status'], 'NOT_CERTIFIED')
        self.assertIs(result['verified'], False)
        self.assertIs(result['graph_embedding_inputs_unchanged'], False)
        self.assertIs(result['nonexistence_proved'], False)
        self.assertIsNone(result['certified_integer_bound'])

    def test_failed_verifier_process_cannot_reuse_a_success_report(self):
        self.ready_for_verification()
        output = self.folder / 'verification'

        def report_success_then_fail(command):
            self.success_summary(output)
            return subprocess.CompletedProcess(command, 1)

        with redirect_stdout(io.StringIO()), \
                patch('workflow.run_check', side_effect=report_success_then_fail):
            result = workflow.verify(self.folder, output)
        self.assertIs(result['verified'], False)
        self.assertIs(result['nonexistence_proved'], False)
        self.assertIsNone(result['certified_integer_bound'])

    def test_requested_independent_failure_prevents_acceptance(self):
        self.ready_for_verification()
        output = self.folder / 'verification'

        def synthetic_reports(command):
            if command[1].endswith('independent_check.py'):
                (output / 'independent.json').write_text(json.dumps({'status': 'FAIL', 'integer_bound': None}))
                return subprocess.CompletedProcess(command, 1)
            self.success_summary(output)
            return subprocess.CompletedProcess(command, 0)

        with redirect_stdout(io.StringIO()), patch('workflow.run_check', side_effect=synthetic_reports):
            result = workflow.verify(self.folder, output, independent=True)
        self.assertFalse(result['verified'])
        self.assertFalse(result['nonexistence_proved'])
        self.assertTrue(result['independent_check_requested'])
        self.assertIsNone(result['certified_integer_bound'])

    def test_saved_embedding_claim_is_not_trusted(self):
        self.prepare()
        # An input file cannot rescue an invalid tuple by preserving its old
        # cached, apparently successful embedding report.
        self.replace_json(self.folder / 'input.json', lambda value: value.update(parameters=[4, 2, 1, 0]))
        with redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(ValueError, 'infeasible'):
                workflow.check_embedding(self.folder)

    def test_each_model_must_match_rederived_spherical_embedding(self):
        self.prepare()
        for name in ('metadata.json', 'original_metadata.json'):
            path = self.folder / 'model' / name
            original = path.read_text()
            with self.subTest(metadata=name):
                self.replace_json(path, lambda value: value.update(a='-1/2'))
                with self.assertRaisesRegex(ValueError, 'spherical model disagree'):
                    workflow.check_embedding(self.folder)
            path.write_text(original)

    def test_changed_model_file_is_rejected_before_solver_launch(self):
        self.prepare()
        problem = self.folder / 'model/problem.dat-s'
        problem.write_text(problem.read_text() + '\n')
        with patch('workflow.executable', return_value=Path(sys.executable)), \
                patch('workflow.subprocess.Popen') as run:
            with self.assertRaisesRegex(ValueError, 'Changed model file'):
                workflow.search(self.folder)
        run.assert_not_called()
        self.assertFalse((self.folder / 'solution').exists())

    def test_timeout_is_never_certified(self):
        self.prepare()
        timeout = subprocess.TimeoutExpired(cmd=['unused-mock-solver'], timeout=0.1)
        process = Mock(pid=123456789, returncode=-9)
        process.wait.side_effect = [timeout, -9]
        with redirect_stdout(io.StringIO()), \
                patch('workflow.executable', return_value=Path(sys.executable)), \
                patch('workflow.subprocess.Popen', return_value=process), \
                patch('workflow.os.killpg', create=True):
            with self.assertRaisesRegex(ValueError, 'time limit exceeded'):
                workflow.search(self.folder, timeout=0.1)
        for path in (self.folder / 'result.json', self.folder / 'solution/run.json'):
            result = json.loads(path.read_text())
            self.assertEqual(result['status'], 'TIMED_OUT_NOT_CERTIFIED')
            self.assertIs(result['verified'], False)
        self.assertFalse((self.folder / 'solution/candidate.json').exists())

    def test_solver_exit_failure_is_never_certified(self):
        self.prepare()
        failed = Mock(returncode=9)
        failed.wait.return_value = 9
        with redirect_stdout(io.StringIO()), \
                patch('workflow.executable', return_value=Path(sys.executable)), \
                patch('workflow.subprocess.Popen', return_value=failed):
            with self.assertRaisesRegex(ValueError, 'Solver exited unsuccessfully'):
                workflow.search(self.folder)
        result = json.loads((self.folder / 'result.json').read_text())
        self.assertEqual(result['status'], 'SEARCH_FAILED_NOT_CERTIFIED')
        self.assertIs(result['verified'], False)
        self.assertEqual(result['returncode'], 9)
        self.assertFalse((self.folder / 'solution/candidate.json').exists())

    def test_invalid_numerical_settings_do_not_create_a_run(self):
        variants = [dict(points=True), dict(points=2.0), dict(degree=False),
                    dict(degree=0.5), dict(precision=63), dict(iterations=0),
                    dict(epsilon='NaN'), dict(epsilon='1e-16\n1000')]
        for changes in variants:
            args = dict(values=[4, 2, 0, 2], eigenspace='auto', points=2, degree=0,
                        output=self.folder)
            args.update(changes)
            with self.subTest(changes=changes), redirect_stdout(io.StringIO()):
                with self.assertRaises(ValueError):
                    workflow.prepare(**args)
                self.assertFalse(self.folder.exists())

    def test_invalid_timeouts_do_not_start_solver(self):
        for timeout in (float('nan'), float('inf'), -1, 0, True):
            with self.subTest(timeout=timeout), patch('workflow.executable') as locate:
                with self.assertRaisesRegex(ValueError, 'positive finite'):
                    workflow.search(self.folder, timeout=timeout)
                locate.assert_not_called()

    def test_cli_rejects_invalid_timeout_before_building(self):
        for timeout in ('nan', 'inf', '0', '-1'):
            command = [sys.executable, str(workflow.ROOT / 'code/srg.py'), 'solve',
                       '--srg', '4', '2', '0', '2', '--points', '2', '--degree', '0',
                       '--output', str(self.folder), '--timeout', timeout]
            with self.subTest(timeout=timeout):
                process = subprocess.run(command, capture_output=True, text=True)
                self.assertNotEqual(process.returncode, 0)
                self.assertIn('positive finite', process.stderr)
                self.assertFalse(self.folder.exists())

    def test_existing_run_and_verification_outputs_are_not_overwritten(self):
        self.ready_for_verification()
        original = (self.folder / 'input.json').read_bytes()
        with redirect_stdout(io.StringIO()), self.assertRaises(FileExistsError):
            self.prepare()
        self.assertEqual((self.folder / 'input.json').read_bytes(), original)
        output = self.folder / 'verification'
        output.mkdir()
        sentinel = output / 'result.json'
        sentinel.write_text('keep existing report')
        with self.assertRaisesRegex(ValueError, 'output exists'):
            workflow.verify(self.folder, output)
        self.assertEqual(sentinel.read_text(), 'keep existing report')

    def test_stale_certified_result_is_cleared_when_candidate_is_missing(self):
        self.prepare()
        (self.folder / 'result.json').write_text(json.dumps(dict(status='CERTIFIED', verified=True)))
        with redirect_stdout(io.StringIO()), patch('workflow.run_check') as check:
            result = workflow.verify(self.folder)
        check.assert_not_called()
        self.assertFalse(result['verified'])
        self.assertFalse(result['nonexistence_proved'])
        saved = json.loads((self.folder / 'result.json').read_text())
        self.assertEqual(saved['status'], 'NOT_CERTIFIED')
        self.assertIsNone(saved['certified_integer_bound'])

    def test_bare_pass_without_current_input_hashes_is_rejected(self):
        self.ready_for_verification()
        output = self.folder / 'verification'

        def unlinked_report(command):
            (output / 'verification_summary.json').write_text(json.dumps(dict(
                status='PASS', verified=True, certified_integer_bound=3)))
            return subprocess.CompletedProcess(command, 0)

        with redirect_stdout(io.StringIO()), patch('workflow.run_check', side_effect=unlinked_report):
            result = workflow.verify(self.folder)
        self.assertFalse(result['verified'])
        self.assertIsNone(result['certified_integer_bound'])

    def test_candidate_change_after_main_check_prevents_acceptance(self):
        self.ready_for_verification()
        output = self.folder / 'verification'

        def change_candidate(command):
            self.success_summary(output)
            (self.folder / 'solution/candidate.json').write_text('{"changed": true}')
            return subprocess.CompletedProcess(command, 0)

        with redirect_stdout(io.StringIO()), patch('workflow.run_check', side_effect=change_candidate):
            result = workflow.verify(self.folder)
        self.assertFalse(result['verified'])
        self.assertFalse(result['graph_embedding_inputs_unchanged'])
        self.assertIsNone(result['certified_integer_bound'])

    def test_main_bound_must_be_an_integer_not_a_boolean_or_string(self):
        self.ready_for_verification()
        for index, bound in enumerate((True, '3', 0, 3.0)):
            output = self.folder / f'verification_{index}'

            def invalid_bound(command):
                self.success_summary(output, bound)
                return subprocess.CompletedProcess(command, 0)

            with self.subTest(bound=bound), redirect_stdout(io.StringIO()), \
                    patch('workflow.run_check', side_effect=invalid_bound):
                result = workflow.verify(self.folder, output)
            self.assertFalse(result['verified'])
            self.assertIsNone(result['certified_integer_bound'])

    def test_independent_pass_text_cannot_override_failed_process_or_wrong_bound(self):
        self.ready_for_verification()
        for index, (returncode, bound) in enumerate(((1, 3), (0, 4), (0, True))):
            output = self.folder / f'verification_{index}'

            def child_report(command):
                if command[1].endswith('independent_check.py'):
                    self.success_independent(output, bound)
                    return subprocess.CompletedProcess(command, returncode)
                self.success_summary(output)
                return subprocess.CompletedProcess(command, 0)

            with self.subTest(returncode=returncode, bound=bound), redirect_stdout(io.StringIO()), \
                    patch('workflow.run_check', side_effect=child_report):
                result = workflow.verify(self.folder, output, independent=True)
            self.assertFalse(result['verified'])
            self.assertFalse(result['independent_check_passed'])
            self.assertIsNone(result['certified_integer_bound'])

    def test_malformed_child_report_is_recorded_as_not_certified(self):
        self.ready_for_verification()
        output = self.folder / 'verification'

        def malformed_report(command):
            (output / 'verification_summary.json').write_text('{broken')
            return subprocess.CompletedProcess(command, 0)

        with redirect_stdout(io.StringIO()), patch('workflow.run_check', side_effect=malformed_report):
            result = workflow.verify(self.folder)
        self.assertEqual(result['status'], 'NOT_CERTIFIED')
        self.assertFalse(result['verified'])
        self.assertIsNone(result['certified_integer_bound'])

    def test_interrupted_verification_records_failure_and_stops_child_group(self):
        self.ready_for_verification()
        process = Mock(pid=123456789)
        process.wait.side_effect = [KeyboardInterrupt(), -9]
        with redirect_stdout(io.StringIO()), patch('workflow.subprocess.Popen', return_value=process), \
                patch('workflow.os.killpg', create=True) as kill:
            with self.assertRaises(KeyboardInterrupt):
                workflow.verify(self.folder)
        if workflow.os.name == 'posix':
            kill.assert_called_once_with(process.pid, workflow.signal.SIGKILL)
        result = json.loads((self.folder / 'result.json').read_text())
        self.assertEqual(result['status'], 'INTERRUPTED_NOT_CERTIFIED')
        self.assertFalse(result['verified'])
        self.assertIsNone(result['certified_integer_bound'])

    def test_linked_success_reports_are_accepted(self):
        self.ready_for_verification()
        output = self.folder / 'verification'

        def linked_report(command):
            if command[1].endswith('independent_check.py'):
                self.success_independent(output)
            else:
                self.success_summary(output)
            return subprocess.CompletedProcess(command, 0)

        with redirect_stdout(io.StringIO()), patch('workflow.run_check', side_effect=linked_report):
            result = workflow.verify(self.folder, independent=True)
        self.assertTrue(result['verified'])
        self.assertTrue(result['independent_check_passed'])
        self.assertEqual(result['certified_integer_bound'], 3)


if __name__ == '__main__':
    unittest.main()
