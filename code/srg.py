#!/usr/bin/env python3
"""Build, solve, and rigorously verify spherical bounds from SRG parameters."""
from pathlib import Path
import argparse
import json
import os
import shlex
import subprocess
import sys

sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
if sys.version_info < (3, 9):
    raise SystemExit('Python 3.9 or later is required.')
if hasattr(sys, 'set_int_max_str_digits'):
    sys.set_int_max_str_digits(0)
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'code'))
import model
import parameters
import workflow


def main():
    parser = argparse.ArgumentParser(description=__doc__,
        epilog='No case whitelist. Rational injective embeddings supported; irrational angles reported as unsupported. Paths to default outputs are relative to this script, not your terminal directory.')
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('list', help='list the four supplied certificates')
    commands.add_parser('doctor', help='show Python, package, and solver locations')
    inspect = commands.add_parser('inspect', help='analyze parameters exactly without solving')
    inspect.add_argument('--srg', nargs=4, type=int, required=True, metavar=('V', 'VALENCY', 'LAMBDA', 'MU'))
    inspect.add_argument('--eigenspace', choices=('auto', 'r', 's'), default='auto')
    setup = commands.add_parser('setup', help='remember an existing solver, or build the bundled source')
    setup_options = setup.add_mutually_exclusive_group(required=True)
    setup_options.add_argument('--solver', help='path to an existing sdpa_gmp executable')
    setup_options.add_argument('--build', action='store_true')
    setup.add_argument('--prefix', type=Path, default=ROOT / 'extras/solver')
    setup.add_argument('--jobs', type=int, default=4)
    setup.add_argument('--timeout', type=int, default=1200)
    for name in ('solve', 'prepare'):
        p = commands.add_parser(name, help='build, solve, and verify' if name == 'solve' else 'build the exact SDP only')
        p.add_argument('--srg', nargs=4, type=int, required=True, metavar=('V', 'VALENCY', 'LAMBDA', 'MU'))
        p.add_argument('--eigenspace', choices=('auto', 'r', 's'), default='auto')
        p.add_argument('--points', type=int, help='2..min(6, embedding dimension); default min(6,n)')
        p.add_argument('--degree', type=int, default=5)
        p.add_argument('--output', type=Path, help='new output directory; default a unique folder under results/runs/')
        p.add_argument('--precision', type=int, default=1500, help='requested solver bits')
        p.add_argument('--iterations', type=int, default=2000)
        p.add_argument('--epsilon', default='1e-16', help='solver stopping tolerance; never the proof acceptance tolerance')
        if name == 'solve':
            p.add_argument('--independent', action='store_true', help='also run the separate all-degree arithmetic check')
            p.add_argument('--solver', help='optional solver path override')
            p.add_argument('--timeout', type=float, default=1200, help='solver seconds; default 1200 (20 minutes)')
    run = commands.add_parser('run', help='solve and verify a previously prepared new run folder')
    run.add_argument('--run', type=Path, required=True)
    run.add_argument('--solver')
    run.add_argument('--timeout', type=float, default=1200)
    run.add_argument('--independent', action='store_true')
    verify = commands.add_parser('verify', help='exactly verify a saved certificate or a new run; no solver')
    target = verify.add_mutually_exclusive_group(required=True)
    target.add_argument('--certificate', help='full parameter ID, e.g. 550_162_75_36')
    target.add_argument('--run', type=Path)
    verify.add_argument('--output', type=Path, help='new verification output directory')
    verify.add_argument('--independent', action='store_true', help='also run the separate all-degree arithmetic check')
    args = parser.parse_args()
    if args.command == 'list':
        for folder in sorted((ROOT / 'results/certificates').iterdir()):
            if (folder / 'input.json').is_file():
                entry = workflow.read(folder / 'input.json')
                print(f"{folder.name}: points={entry['points']}, degree={entry['degree']}; saved bound={entry.get('saved_integer_bound', 'see verification')}")
        return 0
    if args.command == 'doctor':
        print('Python:', sys.version.split()[0], sys.executable)
        print('Package:', ROOT)
        try:
            print('Solver:', workflow.executable())
        except ValueError as error:
            print(error)
        return 0
    if args.command == 'inspect':
        report = parameters.analyze(args.srg, args.eigenspace)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0 if report['status'] == 'supported' else 2
    if args.command == 'setup':
        if args.build:
            process = subprocess.run([sys.executable, str(ROOT / 'code/build_solver.py'),
                '--prefix', str(args.prefix.expanduser().resolve()), '--jobs', str(args.jobs),
                '--timeout', str(args.timeout)])
            if process.returncode:
                return process.returncode
            solver = workflow.executable(str(args.prefix.expanduser().resolve() / 'bin/sdpa_gmp'))
        else:
            solver = workflow.executable(args.solver)
        model.dump(ROOT / 'extras/local_settings.json', {'solver': str(solver)})
        print('Saved solver:', solver)
        return 0
    if args.command in ('prepare', 'solve'):
        if args.command == 'solve':
            workflow.check_timeout(args.timeout)
        solver = workflow.executable(args.solver) if args.command == 'solve' else None
        folder = workflow.prepare(args.srg, args.eigenspace, args.points, args.degree,
            args.output, args.precision, args.iterations, args.epsilon)
        if args.command == 'prepare':
            command = [sys.executable, str(ROOT / 'code/srg.py'), 'run', '--run', str(folder)]
            print('Prepared. Run next:', shlex.join(command))
            return 0
        workflow.search(folder, solver, args.timeout)
        return 0 if workflow.verify(folder, independent=args.independent)['verified'] else 1
    if args.command == 'run':
        workflow.search(args.run, args.solver, args.timeout)
        return 0 if workflow.verify(args.run, independent=args.independent)['verified'] else 1
    if args.command == 'verify':
        if args.certificate:
            name = args.certificate
            if Path(name).name != name or name in ('.', '..'):
                parser.error('--certificate must be an ID from the list command')
            folder = ROOT / 'results/certificates' / name
        else:
            folder = args.run
        return 0 if workflow.verify(folder, args.output, independent=args.independent)['verified'] else 1


if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit('Interrupted. An unfinished calculation is not a certificate.')
    except (ValueError, OSError, KeyError, TypeError) as error:
        sys.exit('ERROR: ' + str(error))
