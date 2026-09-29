"""Uniform numerical search and exact acceptance; no case catalogue is trusted."""
from datetime import datetime
from fractions import Fraction as F
from pathlib import Path
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import time

import exact_model as exact
import model
import parameters
import verify as certificate_verifier

ROOT = Path(__file__).resolve().parent.parent


def read(path):
    return exact.read_json(path)


def executable(value=None):
    settings = ROOT / 'extras/local_settings.json'
    if value is None and settings.is_file():
        value = read(settings).get('solver')
    if value is None:
        value = os.environ.get('SDPA_GMP')
    if value is None:
        value = str(ROOT / 'extras/solver/bin/sdpa_gmp')
    located = shutil.which(str(value))
    path = Path(located if located else value).expanduser().resolve()
    exact.require(path.is_file() and os.access(path, os.X_OK),
        'Solver not found. First run: python3 code/srg.py setup --solver /absolute/path/to/sdpa_gmp\n'
        'Or build the bundled sources: python3 code/srg.py setup --build')
    return path


def analyze_parameters(values, eigenspace):
    report = parameters.analyze(values, eigenspace)
    if report['status'] != 'supported':
        print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
        raise ValueError('Parameter analysis: ' + report['status'] + '. No SDP certificate was produced.')
    return report


def check_embedding(folder):
    entry = read(folder / 'input.json')
    exact.require(isinstance(entry, dict) and entry.get('schema') == 'srg-sdp-input-v1',
                  'Unrecognized SRG input file')
    exact.require(isinstance(entry.get('parameters'), list)
                  and len(entry['parameters']) == 4
                  and all(type(value) is int for value in entry['parameters']),
                  'Saved SRG parameters must be four integers')
    exact.require(type(entry.get('points')) is int and type(entry.get('degree')) is int,
                  'Saved points and degree must be integers')
    report = analyze_parameters(entry['parameters'], entry.get('eigenspace', 'auto'))
    for name in ('metadata.json', 'original_metadata.json'):
        meta = read(folder / 'model' / name)
        exact.require((exact.integer(meta['n'], 'n'), exact.rational(meta['a']), exact.rational(meta['b'])) ==
                      (report['n'], F(report['a']), F(report['b'])),
                      'SRG parameters and spherical model disagree: ' + name)
        exact.require(exact.integer(meta['k'], 'k') == entry['points']
                      and exact.integer(meta['d'], 'd') == entry['degree'],
                      'Requested points/degree and model disagree')
    return entry, report


def output_folder(embedding, points, degree, specified=None):
    if specified is not None:
        return Path(specified).expanduser().resolve()
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    name = '_'.join(map(str, embedding['parameters']))
    return ROOT / 'results/runs' / f'{name}_p{points}_d{degree}_{stamp}'


def prepare(values, eigenspace, points, degree, output=None, precision=1500,
            iterations=2000, epsilon='1e-16'):
    embedding = analyze_parameters(values, eigenspace)
    points = min(6, embedding['n']) if points is None else points
    exact.require(type(points) is int and 2 <= points <= min(6, embedding['n']),
                  'Requested --points must be 2..min(6, embedding dimension). Try a smaller --points or the other eigenspace.')
    exact.require(type(degree) is int and degree >= 0, '--degree must be a nonnegative integer')
    model.parameter_text(precision, iterations, epsilon)
    folder = output_folder(embedding, points, degree, output)
    folder.mkdir(parents=True, exist_ok=False)
    model.dump(folder / 'input.json', dict(schema='srg-sdp-input-v1',
        parameters=embedding['parameters'], eigenspace=eigenspace, points=points, degree=degree,
        embedding=embedding))
    print('Output:', folder, flush=True)
    print(f"Sphere dimension {embedding['n']}; inner products {embedding['a']}, {embedding['b']}; points={points}, degree={degree}", flush=True)
    try:
        model.build(embedding, points, degree, folder / 'model', precision, iterations, epsilon)
    except Exception as error:
        model.dump(folder / 'result.json', dict(status='BUILD_FAILED', verified=False, error=str(error)))
        raise
    model.dump(folder / 'result.json', dict(status='PREPARED_NOT_CERTIFIED', verified=False))
    return folder


def check_timeout(timeout):
    exact.require(timeout is None or (type(timeout) in (int, float)
                  and math.isfinite(timeout) and timeout > 0),
                  'timeout must be a positive finite number of seconds')


def stop_process(process):
    """Stop the complete subprocess group before reporting an interrupted run."""
    if os.name == 'posix':
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    elif process.poll() is None:
        process.kill()
    process.wait()


def run_check(command):
    process = subprocess.Popen(command, start_new_session=(os.name == 'posix'))
    try:
        return subprocess.CompletedProcess(command, process.wait())
    except (KeyboardInterrupt, InterruptedError):
        stop_process(process)
        raise


def search(folder, solver=None, timeout=1200):
    check_timeout(timeout)
    solver = executable(solver)
    folder = Path(folder).expanduser().resolve()
    entry, embedding = check_embedding(folder)
    build = read(folder / 'model/build.json')
    for name, digest in build['files_sha256'].items():
        exact.require(model.sha(folder / 'model' / name) == digest, 'Changed model file: ' + name)
    out = folder / 'solution'
    out.mkdir(exist_ok=False)
    command = [str(solver), '-ds', str(folder / 'model/problem.dat-s'),
               '-o', str(out / 'solver.out'), '-p', str(folder / 'model/solver.par')]
    environment = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    started = time.monotonic()
    record = dict(status='SEARCH_NOT_CERTIFIED', verified=False, solver_sha256=model.sha(solver),
                  command=command, timeout_seconds=timeout,
                  input_sha256=model.sha(folder / 'model/problem.dat-s'))
    model.dump(out / 'run.json', record)
    print('Running SDPA-GMP. Progress log:', out / 'console.log', flush=True)
    def terminated(signum, frame):
        raise KeyboardInterrupt('terminated')
    previous_handler = signal.signal(signal.SIGTERM, terminated)
    try:
        with (out / 'console.log').open('w') as stream:
            process = subprocess.Popen(command, cwd=out, stdout=stream, stderr=subprocess.STDOUT,
                                       env=environment, start_new_session=(os.name == 'posix'))
            try:
                process.wait(timeout=timeout)
            except (subprocess.TimeoutExpired, KeyboardInterrupt):
                stop_process(process)
                raise
        record['returncode'] = process.returncode
        exact.require(process.returncode == 0, 'Solver exited unsuccessfully; inspect solution/console.log')
        raw = exact.read_sdpa(out / 'solver.out')
        meta = read(folder / 'model/metadata.json')
        shifts = read(folder / 'model/physical_shifts.json')
        dimensions = [spec['dim'] for spec in meta['blocks']]
        exact.require(len(raw) == sum(dim > 0 for dim in dimensions), 'Solver block count mismatch')
        iterator = iter(raw)
        logical = [next(iterator) if dim else [] for dim in dimensions]
        physical = []
        exact.require(len(shifts) == len(logical), 'Shift block count mismatch')
        for matrix, dim, shift in zip(logical, dimensions, shifts):
            exact.require(len(matrix) == dim and all(len(row) == dim for row in matrix), 'Solver matrix shape')
            exact.require(len(shift) == dim, 'Shift shape')
            exact.require(all(matrix[i][j] == matrix[j][i] for i in range(dim) for j in range(dim)),
                          'Solver matrix is not exactly symmetric')
            physical.append([[str(value + (F(shift[i]) if i == j else 0))
                              for j, value in enumerate(row)] for i, row in enumerate(matrix)])
        model.dump(out / 'candidate.json', dict(schema='compressed-kpoint-physical-candidate-v1',
            metadata_sha256=model.sha(folder / 'model/metadata.json'),
            solver_output_sha256=model.sha(out / 'solver.out'),
            physical_shifts_sha256=model.sha(folder / 'model/physical_shifts.json'),
            matrices=physical, representation='compressed',
            note='Exact printed decimals plus recorded rational diagonal shifts. Not certified until verification passes.'))
        record['status'] = 'CANDIDATE_NOT_CERTIFIED'
        record['candidate_sha256'] = model.sha(out / 'candidate.json')
    except subprocess.TimeoutExpired:
        record.update(status='TIMED_OUT_NOT_CERTIFIED', error='Solver time limit exceeded')
        raise ValueError('Solver time limit exceeded; partial output is not a proof.')
    except (Exception, KeyboardInterrupt) as error:
        record.update(status='SEARCH_FAILED_NOT_CERTIFIED', error=str(error))
        raise
    finally:
        signal.signal(signal.SIGTERM, previous_handler)
        record['solver_wall_seconds'] = time.monotonic() - started
        model.dump(out / 'run.json', record)
        model.dump(folder / 'result.json', record)
    return folder


def verify(folder, output=None, independent=False):
    folder = Path(folder).expanduser().resolve()
    exact.require(folder.is_dir(), 'Run or certificate directory does not exist: ' + str(folder))
    saved_certificate = folder.parent == (ROOT / 'results/certificates')
    if output is None:
        if saved_certificate:
            stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            output = ROOT / 'results/runs' / f'verify_{folder.name}_{stamp}'
        else:
            output = folder / 'verification'
    output = Path(output).expanduser().resolve()
    exact.require(not output.exists(), 'Verification output exists. Choose a new --output directory.')
    output.mkdir(parents=True)
    summary_path = output / 'verification_summary.json'
    result = dict(status='VERIFYING_NOT_CERTIFIED', verified=False,
        certified_integer_bound=None, nonexistence_proved=False,
        independent_check_requested=independent,
        independent_check_passed=False if independent else None,
        verification_summary=str(summary_path), graph_embedding_inputs_unchanged=False)

    def save_result():
        model.dump(output / 'result.json', result)
        if not saved_certificate:
            model.dump(folder / 'result.json', result)

    # A new attempt must not leave a previous CERTIFIED flag in place if it fails.
    save_result()
    linked_hashes = None
    independent_passed = False
    previous_handler = signal.signal(signal.SIGTERM, certificate_verifier.on_termination)
    try:
        stage_hashes = certificate_verifier.input_hashes(folder / 'model', folder / 'solution')
        linked_paths = [folder / 'input.json', Path(parameters.__file__), Path(model.__file__), Path(__file__)]
        if independent:
            linked_paths.append(ROOT / 'code/independent_check.py')
        linked_hashes = dict(stage_hashes)
        linked_hashes.update({str(path): model.sha(path) for path in linked_paths})
        entry, embedding = check_embedding(folder)
        result.update(parameters=entry['parameters'], points=entry['points'], degree=entry['degree'],
            sphere_dimension=embedding['n'], inner_products=[embedding['a'], embedding['b']],
            input_sha256=linked_hashes[str(folder / 'input.json')],
            graph_embedding_input_sha256=linked_hashes,
            original_model_sha256=linked_hashes[str(folder / 'model/original_metadata.json')],
            parameter_checker_sha256=linked_hashes[str(Path(parameters.__file__))])
        command = [sys.executable, str(ROOT / 'code/verify.py'),
            '--model', str(folder / 'model'), '--run', str(folder / 'solution'), '--output', str(output)]
        print('Checking exact feasibility, compression, and solver input/output. Reports:', output, flush=True)
        process = run_check(command)
        result['verifier_returncode'] = process.returncode
        summary = read(summary_path) if summary_path.is_file() else {}
        exact.require(isinstance(summary, dict), 'Verification summary must be a JSON object')
        integer_bound = summary.get('certified_integer_bound')
        exact.require(process.returncode == 0 and summary.get('verified') is True
                      and summary.get('status') == 'PASS'
                      and summary.get('inputs_unchanged') is True
                      and summary.get('input_sha256') == stage_hashes
                      and type(integer_bound) is int and integer_bound >= 1,
                      'Main verification did not certify the current files; inspect verification_summary.json')
        if independent:
            print('Running the separate all-degree coefficient and matrix check.', flush=True)
            independent_path = output / 'independent.json'
            cross = run_check([sys.executable, str(ROOT / 'code/independent_check.py'),
                '--model', str(folder / 'model'), '--solution', str(folder / 'solution'),
                '--output', str(independent_path)])
            result['independent_returncode'] = cross.returncode
            extra = read(independent_path) if independent_path.is_file() else {}
            exact.require(isinstance(extra, dict), 'Independent report must be a JSON object')
            expected_inputs = {str(path): linked_hashes[str(path)] for path in (
                folder / 'model/original_metadata.json', folder / 'model/metadata.json',
                folder / 'model/original_exact_coefficients.tsv', folder / 'solution/candidate.json')}
            independent_passed = (cross.returncode == 0 and extra.get('status') == 'PASS'
                and extra.get('verified') is True and extra.get('inputs_unchanged') is True
                and type(extra.get('integer_bound')) is int and extra['integer_bound'] == integer_bound
                and extra.get('inputs') == expected_inputs
                and extra.get('source_sha256') == linked_hashes[str(ROOT / 'code/independent_check.py')])
            exact.require(independent_passed,
                          'Independent verification did not certify the same files and bound; inspect independent.json')
        exact.require(all(model.sha(path) == digest for path, digest in linked_hashes.items()),
                      'An input or verifier changed during verification')
        final_entry, final_embedding = check_embedding(folder)
        exact.require(final_entry == entry and final_embedding == embedding,
                      'SRG parameters or embedding changed during verification')
        result.update(status='CERTIFIED', verified=True, certified_integer_bound=integer_bound,
            nonexistence_proved=integer_bound < entry['parameters'][0])
    except (KeyboardInterrupt, InterruptedError) as error:
        result.update(status='INTERRUPTED_NOT_CERTIFIED', error=str(error))
        raise
    except Exception as error:
        result.update(status='NOT_CERTIFIED', error_type=type(error).__name__, error=str(error))
    finally:
        signal.signal(signal.SIGTERM, previous_handler)
        try:
            unchanged = linked_hashes is not None and all(
                model.sha(path) == digest for path, digest in linked_hashes.items())
        except OSError:
            unchanged = False
        result['graph_embedding_inputs_unchanged'] = unchanged
        result['independent_check_passed'] = (independent_passed and unchanged) if independent else None
        if not unchanged and result['verified']:
            result.update(status='NOT_CERTIFIED', verified=False, certified_integer_bound=None,
                          nonexistence_proved=False, error='An input or verifier changed during verification')
        save_result()
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    return result
