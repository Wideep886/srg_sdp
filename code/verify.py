#!/usr/bin/env python3
"""Certify one supported k-point candidate through three required checks.

No solver is run. Children stay in the supervisor's process group so an outer
case/batch deadline can terminate the complete tree. A stage is accepted only
when its process exits zero, its report passes, and all recorded inputs and
verifier sources still have their recorded hashes. Only all three successful
stages produce verified=true and a certified integer bound.
"""
import argparse
import datetime
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import signal
import subprocess
import sys
import time
from exact_model import read_json

if hasattr(sys, 'set_int_max_str_digits'):
    sys.set_int_max_str_digits(0)

HERE = Path(__file__).resolve().parent
STAGES = ('exact', 'direct-zero', 'export-audit')
MATRIX_MARGIN = '1e-30'
LINEAR_MARGIN = '1e-26'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def timestamp():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def absolute_path(value):
    path = Path(value)
    if not path.is_absolute():
        raise argparse.ArgumentTypeError('paths must be absolute')
    return path.resolve()


def input_hashes(model, run):
    files = [model / name for name in (
        'original_metadata.json', 'metadata.json', 'exact_coefficients.tsv',
        'original_exact_coefficients.tsv', 'physical_shifts.json',
        'problem.dat-s', 'solver.par', 'build.json')]
    files += [run / 'candidate.json', run / 'solver.out']
    files += [Path(__file__).resolve(), HERE / 'check_solver_io.py',
              HERE / 'check_matrices.py', HERE / 'exact_model.py',
              HERE / 'check_compression.py',
              HERE / 'exact_arithmetic.py']
    return {str(path): sha(path) for path in files}


def stage_spec(stage, model, run, output):
    if stage == 'exact':
        report = output / 'exact/verification.json'
        command = [sys.executable, str(HERE / 'check_matrices.py'),
            '--original-metadata', str(model / 'original_metadata.json'),
            '--metadata', str(model / 'metadata.json'),
            '--candidate', str(run / 'candidate.json'),
            '--output', str(output / 'exact'),
            '--model-coefficients', str(model / 'exact_coefficients.tsv'),
            '--matrix-margin', MATRIX_MARGIN, '--linear-margin', LINEAR_MARGIN]
    elif stage == 'direct-zero':
        report = output / 'direct.json'
        command = [sys.executable, str(HERE / 'check_compression.py'),
            '--metadata', str(model / 'original_metadata.json'),
            '--compressed-coefficients', str(model / 'exact_coefficients.tsv'),
            '--original-coefficients', str(model / 'original_exact_coefficients.tsv'),
            '--output', str(report)]
    elif stage == 'export-audit':
        report = output / 'export.json'
        command = [sys.executable, str(HERE / 'check_solver_io.py'),
            '--model', str(model), '--run', str(run), '--output', str(report)]
    else:
        raise ValueError('unknown stage')
    return command, report


def report_passes(stage, report, hashes):
    def expected(name):
        matches = [digest for path, digest in hashes.items() if Path(path).name == name]
        return matches[0] if len(matches) == 1 else None

    if stage == 'exact':
        return (report.get('verified') is True
            and report.get('all_shifted_blocks_strictly_positive') is True
            and report.get('all_residuals_below_negative_margin') is True
            and report.get('all_residuals_exactly_preserved') is True
            and report.get('builder_coefficients_exactly_match') is True
            and report.get('certified_integer_bound') is not None
            and report.get('metadata_sha256') == expected('metadata.json')
            and report.get('original_metadata_sha256') == expected('original_metadata.json')
            and report.get('candidate_sha256') == expected('candidate.json')
            and report.get('verifier_sha256') == expected('check_matrices.py')
            and report.get('original_verifier_sha256') == expected('exact_model.py')
            and report.get('original_helper_sha256') == expected('exact_arithmetic.py')
            and report.get('builder_model_coefficients_sha256') == expected('exact_coefficients.tsv')
            and Fraction(report['matrix_margin']) == Fraction(MATRIX_MARGIN)
            and Fraction(report['linear_margin']) == Fraction(LINEAR_MARGIN))
    if stage == 'direct-zero':
        return (report.get('status') == 'PASS'
            and report.get('positive_degree_unchanged_checked') is True
            and report.get('checker_sha256') == expected('check_compression.py')
            and report.get('metadata', {}).get('sha256') == expected('original_metadata.json')
            and report.get('compressed_coefficients', {}).get('sha256') == expected('exact_coefficients.tsv')
            and report.get('original_coefficients', {}).get('sha256') == expected('original_exact_coefficients.tsv'))
    return (report.get('status') == 'PASS'
        and report.get('models') == 1
        and len(report.get('rows', [])) == 1
        and report['rows'][0].get('status') == 'PASS'
        and report['rows'][0].get('candidate_matches_solver_output') is True
        and report['rows'][0].get('solver_output_blocks_checked') == report['rows'][0].get('blocks')
        and type(report['rows'][0].get('solver_output_scalar_entries_checked')) is int
        and report['rows'][0]['solver_output_scalar_entries_checked'] > 0
        and len(report['rows'][0].get('solver_output_match_checks', [])) == report['rows'][0].get('blocks')
        and all(row.get('block') == index and type(row.get('dimension')) is int and row['dimension'] >= 0
                and type(row.get('scalar_entries_checked')) is int
                and row['scalar_entries_checked'] == row['dimension']**2
                and row.get('matches_exactly') is True
                for index, row in enumerate(report['rows'][0]['solver_output_match_checks'], 1))
        and sum(row['scalar_entries_checked'] for row in report['rows'][0]['solver_output_match_checks'])
                == report['rows'][0]['solver_output_scalar_entries_checked']
        and report.get('script_sha256') == expected('check_solver_io.py')
        and len(report['rows'][0].get('sha256', {})) == 9
        and all(hashes.get(path) == digest
                for path, digest in report['rows'][0]['sha256'].items()))


def run_stage(stage, model, run, output, hashes):
    command, report_path = stage_spec(stage, model, run, output)
    stage_path = output / (stage + '_status.json')
    log_path = output / (stage + '.log')
    # A failed new invocation must not be mistaken for an old successful report.
    if report_path.exists():
        report_path.unlink()
    record = dict(stage=stage, status='RUNNING', verified=False,
        started_utc=timestamp(), command=command, log_path=str(log_path),
        report_path=str(report_path), input_sha256=hashes, returncode=None)
    write_json(stage_path, record)
    started = time.monotonic()
    process = None
    try:
        with log_path.open('w') as log:
            # Deliberately no start_new_session: the outer supervisor owns it.
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            record['returncode'] = process.wait()
        require(record['returncode'] == 0, 'stage process returned nonzero')
        require(report_path.is_file(), 'stage did not produce its required report')
        report = read_json(report_path)
        require(report_passes(stage, report, hashes), 'stage report did not certify all required checks')
        require(input_hashes(model, run) == hashes, 'an input or verifier changed during verification')
        record.update(status='PASS', verified=True, report_sha256=sha(report_path))
    except (KeyboardInterrupt, InterruptedError) as exc:
        record.update(status='INTERRUPTED', error=str(exc))
        if process is not None and process.poll() is None:
            process.terminate()
        raise
    except Exception as exc:
        record.update(status='FAIL', error_type=type(exc).__name__, error=str(exc))
    finally:
        record.update(wall_seconds=time.monotonic() - started, finished_utc=timestamp())
        if process is not None:
            record['returncode'] = process.poll()
        write_json(stage_path, record)
    return record


def summarize(model, run, output, hashes):
    stages = {}
    for stage in STAGES:
        _, report_path = stage_spec(stage, model, run, output)
        record_path = output / (stage + '_status.json')
        record = (read_json(record_path) if record_path.is_file()
                  else dict(stage=stage, status='NOT_RUN', returncode=None, wall_seconds=0))
        okay = (record.get('status') == 'PASS' and record.get('returncode') == 0
            and record.get('input_sha256') == hashes and report_path.is_file())
        if okay:
            okay = (record.get('report_sha256') == sha(report_path)
                    and report_passes(stage, read_json(report_path), hashes))
        if record.get('status') == 'PASS' and not okay:
            record = dict(record, status='INVALID', verified=False,
                          error='saved stage provenance/report no longer matches')
        stages[stage] = record
    unchanged = input_hashes(model, run) == hashes
    verified = unchanged and all(s.get('status') == 'PASS' for s in stages.values())
    exact_path = output / 'exact/verification.json'
    exact = read_json(exact_path) if exact_path.is_file() else {}
    status = 'PASS' if verified else ('FAIL' if any(
        s.get('status') in ('FAIL', 'INVALID', 'INTERRUPTED') for s in stages.values()) else 'INCOMPLETE')
    return dict(schema='k6-batch-three-stage-verification-v1',
        verified=verified, status=status, model_directory=str(model), run_directory=str(run),
        exact_report_path=str(exact_path), stages=stages, input_sha256=hashes,
        inputs_unchanged=unchanged, matrix_margin=MATRIX_MARGIN, linear_margin=LINEAR_MARGIN,
        certified_integer_bound=exact.get('certified_integer_bound') if verified else None,
        alpha_decimal=exact.get('alpha_decimal'), upper_decimal=exact.get('upper_decimal'),
        exact_stage_verified=stages['exact'].get('status') == 'PASS',
        solver_runs=0, finished_utc=timestamp())


def on_termination(signum, frame):
    raise InterruptedError(f'terminated by signal {signum}')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, type=absolute_path)
    parser.add_argument('--run', required=True, type=absolute_path)
    parser.add_argument('--output', required=True, type=absolute_path,
                        help='verification directory; use a fresh directory for a fresh candidate')
    parser.add_argument('--stage', choices=('all', 'summarize') + STAGES, default='all',
                        help='single stages retain records; summarize requires all three to pass')
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    summary_path = args.output / 'verification_summary.json'
    started = time.monotonic()
    summary = dict(verified=False, status='RUNNING', certified_integer_bound=None,
                   started_utc=timestamp(), model_directory=str(args.model), run_directory=str(args.run))
    write_json(summary_path, summary)
    signal.signal(signal.SIGTERM, on_termination)
    hashes = None
    try:
        require(args.output != args.model and args.output != args.run,
                'output must be a separate verification directory')
        meta = read_json(args.model / 'metadata.json')
        original = read_json(args.model / 'original_metadata.json')
        require(meta.get('representation') == 'compressed', 'expected compressed model')
        parameters = [original[key] for key in ('n', 'k', 'd')]
        require(all(isinstance(value, (str, int)) and not isinstance(value, bool)
                    for value in parameters), 'n, k, d must be exact integers')
        n, k, d = map(Fraction, parameters)
        require(all(value.denominator == 1 for value in (n, k, d))
                and n >= 2 and 2 <= k <= min(6, n) and d >= 0,
                'supported scope: n>=2, 2<=k<=min(6,n), d>=0')
        hashes = input_hashes(args.model, args.run)
        selected = STAGES if args.stage == 'all' else (() if args.stage == 'summarize' else (args.stage,))
        for stage in selected:
            record = run_stage(stage, args.model, args.run, args.output, hashes)
            if record['status'] != 'PASS':
                break
        summary = summarize(args.model, args.run, args.output, hashes)
    except (KeyboardInterrupt, InterruptedError) as exc:
        summary.update(status='INTERRUPTED', error=str(exc), verified=False, certified_integer_bound=None)
    except Exception as exc:
        summary.update(status='FAIL', error_type=type(exc).__name__, error=str(exc),
                       verified=False, certified_integer_bound=None)
    finally:
        if hashes is not None:
            summary['input_sha256'] = hashes
        summary.update(wall_seconds=time.monotonic() - started, finished_utc=timestamp())
        write_json(summary_path, summary)
    print(json.dumps({key: summary.get(key) for key in
        ('verified', 'status', 'certified_integer_bound', 'wall_seconds', 'error')}, ensure_ascii=False), flush=True)
    if args.stage in STAGES and summary['status'] == 'INCOMPLETE':
        return 0 if summary.get('stages', {}).get(args.stage, {}).get('status') == 'PASS' else 1
    return 0 if summary['verified'] else 1


if __name__ == '__main__':
    sys.exit(main())
