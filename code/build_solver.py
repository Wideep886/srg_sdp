#!/usr/bin/env python3
"""Build bundled SDPA-GMP/GMP without downloads or global installs.

Requires Python 3.9+, POSIX, C/C++ compilers, make, m4, patch, tar,
gzip, ar and ranlib. --check-sources only inspects archives.
GMP tests and an upstream SDPA smoke test are mandatory.
"""
import argparse
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import signal
import subprocess
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parent.parent
ARCHIVES = {
    "sdpa-gmp-7.1.3.tar.gz": ("40ef73041e80e89bd51358f5c51f8eb7f02629d33dd26c14466cc790ef7be1e9", "sdpa-gmp-7.1.3"),
    "gmp-6.3.0.tar.xz": ("a3c2b80201b89e68616f4ad30bc66aee4927c3ce50e33929ca819d5c43538898", "gmp-6.3.0"),
}
PROBE = r"""#include <cstdio>
#include <gmpxx.h>
int main() {
  mpf_set_default_prec(1500);
  mpf_class value = 2;
  mpf_sqrt(value.get_mpf_t(), value.get_mpf_t());
  std::printf("gmp_version=%s\n", gmp_version);
  std::printf("requested_bits=1500 actual_bits=%lu\n",
              (unsigned long)mpf_get_prec(value.get_mpf_t()));
  gmp_printf("sqrt2=%+.Fe\n", value.get_mpf_t());
}
"""
PARAMETERS = "200\n1.0E-40\n1.0E4\n2.0\n-1.0E5\n1.0E5\n0.1\n0.3\n0.9\n1.0E-40\n1500\n"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sources():
    records = []
    for filename, (expected, top) in ARCHIVES.items():
        path = ROOT / "extras/third_party" / "src" / filename
        if sha(path) != expected:
            raise ValueError("source hash mismatch: " + str(path))
        with tarfile.open(path) as archive:
            for member in archive:
                parts = Path(member.name).parts
                if (not parts or parts[0] != top or ".." in parts
                        or Path(member.name).is_absolute()
                        or not (member.isfile() or member.isdir())):
                    raise ValueError("unexpected archive entry: " + member.name)
        records.append({"filename": filename, "sha256": expected, "bytes": path.stat().st_size})
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-sources", action="store_true")
    parser.add_argument("--prefix", type=Path, default=ROOT / "extras/solver",
                        help="new build/install directory (default: extras/solver under the package root)")
    parser.add_argument("--jobs", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--cc", default=os.environ.get("CC"))
    parser.add_argument("--cxx", default=os.environ.get("CXX"))
    parser.add_argument("--timeout", type=int, default=600, help="maximum TOTAL build seconds")
    args = parser.parse_args()
    if sys.version_info < (3, 9):
        raise ValueError("Python 3.9 or later is required")
    archive_records = sources()
    if args.check_sources:
        print(json.dumps({"status": "PASS", "archives": archive_records}, indent=2))
        return 0
    if os.name != "posix":
        raise ValueError("Use Linux/WSL or macOS for the upstream build tools")
    if args.jobs < 1 or args.timeout < 1:
        raise ValueError("--jobs and --timeout must be positive")
    prefix = args.prefix.expanduser().resolve()
    if any(c.isspace() or c in "\"'$#;:&|\\(){}[]*?<>!"+chr(96) for c in str(prefix)):
        raise ValueError("Choose a build path without whitespace or shell/make metacharacters")
    cc = args.cc or shutil.which("clang") or shutil.which("gcc") or shutil.which("cc")
    cxx = args.cxx or shutil.which("clang++") or shutil.which("g++") or shutil.which("c++")
    if not cc or not cxx:
        raise ValueError("Install C and C++ compilers before building")
    cc, cxx = shutil.which(cc), shutil.which(cxx)
    if not cc or not cxx:
        raise ValueError("The specified compiler executable was not found")
    missing = [name for name in ("make", "m4", "patch", "tar", "gzip", "ar", "ranlib", "sh") if not shutil.which(name)]
    if missing:
        raise ValueError("Missing build tools: " + ", ".join(missing))
    prefix.mkdir(parents=True, exist_ok=False)
    logs, source, local, binary = (prefix / name for name in ("logs", "src", "local", "bin"))
    for directory in (logs, source, local, binary):
        directory.mkdir()
    start = time.monotonic()
    metadata = {"status": "RUNNING", "started_utc": datetime.now(timezone.utc).isoformat(),
                "platform": platform.platform(), "machine": platform.machine(),
                "python": sys.version, "script_sha256": sha(__file__),
                "archives": archive_records, "steps": [], "cc": cc, "cxx": cxx,
                "jobs": args.jobs, "total_timeout_seconds": args.timeout,
                "note": "Archive name is 7.1.3; inherited SDPA configure metadata says 7.1.2."}
    meta_path = prefix / "build.json"
    def save():
        metadata["seconds"] = time.monotonic()-start
        meta_path.write_text(json.dumps(metadata, indent=2)+"\n")
    env = os.environ.copy()
    for key in ("CFLAGS", "CXXFLAGS", "CPPFLAGS", "LDFLAGS", "LIBS", "MAKEFLAGS", "MFLAGS"):
        env.pop(key, None)
    env.update({"CC": cc, "CXX": cxx, "LC_ALL": "C", "LANG": "C"})

    def run(name, command, cwd):
        remaining = args.timeout-(time.monotonic()-start)
        if remaining <= 0:
            raise TimeoutError("total build time limit reached")
        log = logs / (name+".log")
        record = {"step": name, "command": [str(x) for x in command],
                  "cwd": str(cwd), "log": str(log), "status": "RUNNING"}
        metadata["steps"].append(record)
        save()
        print("Building: "+name, flush=True)
        begin = time.monotonic()
        with log.open("wb") as stream:
            proc = subprocess.Popen([str(x) for x in command], cwd=cwd, env=env,
                                    stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                code = proc.wait(timeout=remaining)
            except (subprocess.TimeoutExpired, KeyboardInterrupt):
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
                record.update(status="STOPPED", seconds=time.monotonic()-begin)
                save()
                raise
        record.update(status="PASS" if code == 0 else "FAIL", returncode=code,
                      seconds=time.monotonic()-begin, log_sha256=sha(log))
        save()
        if code:
            raise RuntimeError(f"{name} failed (exit {code}); see {log}")
        return log

    try:
        for filename in ARCHIVES:
            with tarfile.open(ROOT/"extras/third_party"/"src"/filename) as archive:
                # sources() accepts only fixed-hash regular files/directories
                # under the expected top directory, with no parent traversal.
                archive.extractall(source)
        gmp, sdpa = source/"gmp-6.3.0", source/"sdpa-gmp-7.1.3"
        # Discard obsolete binary products only within this fresh extraction.
        removed = []
        for file in sorted(sdpa.rglob("*")):
            if file.is_file() and (file.suffix in (".o", ".a", ".so", ".lo", ".la")
                                   or file.name in ("sdpa_gmp", "config.status", "config.log", "Makefile")):
                if file == sdpa/"spooles"/"Makefile":
                    continue
                removed.append(str(file.relative_to(sdpa)))
                file.unlink()
        shutil.rmtree(sdpa/"spooles"/"build", ignore_errors=True)
        metadata["discarded_upstream_build_products"] = removed
        metadata["platform_file_replacements"] = []
        for src_name, dest_name in (("configfsf.guess", "config.guess"), ("configfsf.sub", "config.sub")):
            before = sha(sdpa/dest_name)
            shutil.copy2(gmp/src_name, sdpa/dest_name)
            metadata["platform_file_replacements"].append(
                {"file": dest_name, "before_sha256": before, "after_sha256": sha(sdpa/dest_name),
                 "source": "gmp-6.3.0/"+src_name})
        save()
        run("compiler-c", [cc, "--version"], prefix)
        run("compiler-cxx", [cxx, "--version"], prefix)
        gmp_build = prefix/"build-gmp"
        gmp_build.mkdir()
        run("gmp-configure", [gmp/"configure", "--prefix="+str(local), "--enable-cxx", "--disable-shared"], gmp_build)
        run("gmp-build", ["make", "-j"+str(args.jobs)], gmp_build)
        run("gmp-check", ["make", "check", "-j"+str(args.jobs)], gmp_build)
        run("gmp-install", ["make", "install"], gmp_build)
        flags = "-O2 -std=gnu89 -Wno-error=implicit-function-declaration -Wno-error=implicit-int -Wno-error=int-conversion -Wno-error=incompatible-pointer-types"
        run("spooles-build", ["make", "-C", sdpa/"spooles", "CC="+cc, "CFLAGS="+flags], sdpa)
        run("sdpa-configure", [sdpa/"configure", "--prefix="+str(local),
                               "--with-gmp-includedir="+str(local/"include"),
                               "--with-gmp-libdir="+str(local/"lib"), "--enable-shared",
                               "CXXFLAGS=-O2 -std=c++98"], sdpa)
        run("sdpa-build", ["make", "-j"+str(args.jobs)], sdpa)
        shutil.copy2(sdpa/"sdpa_gmp", binary/"sdpa_gmp")
        (prefix/"precision_probe.cpp").write_text(PROBE)
        run("precision-probe-build", [cxx, "-O2", "-std=c++98", "-I"+str(local/"include"),
                                      prefix/"precision_probe.cpp", local/"lib"/"libgmpxx.a",
                                      local/"lib"/"libgmp.a", "-o", binary/"precision_probe"], prefix)
        probe_log = run("precision-probe", [binary/"precision_probe"], prefix)
        probe = probe_log.read_text()
        actual = re.search(r"requested_bits=1500 actual_bits=(\d+)", probe)
        if "gmp_version=6.3.0" not in probe or not actual or int(actual[1]) < 1500:
            raise ValueError("GMP precision/version probe failed")
        metadata["gmp_actual_precision_bits"] = int(actual[1])
        smoke = prefix/"smoke"
        smoke.mkdir()
        shutil.copy2(sdpa/"example1.dat-s", smoke/"example1.dat-s")
        (smoke/"parameters.par").write_text(PARAMETERS)
        run("sdpa-smoke", [binary/"sdpa_gmp", "-ds", smoke/"example1.dat-s",
                          "-o", smoke/"solver.out", "-p", smoke/"parameters.par"], prefix)
        output = (smoke/"solver.out").read_text()
        objective = re.search(r"objValDual\s*=\s*([+-]?[0-9.eE+-]+)", output)
        phase = re.search(r"phase\.value\s*=\s*(\w+)", output)
        if not phase or phase[1] != "pdOPT" or not objective or abs(Fraction(objective[1])+Fraction(419, 10)) > Fraction(1, 10**20):
            raise ValueError("upstream example did not reach the expected -41.9 objective")
        metadata.update(status="PASS", solver=str(binary/"sdpa_gmp"),
                        solver_sha256=sha(binary/"sdpa_gmp"),
                        smoke_phase=phase[1], smoke_objective=objective[1],
                        smoke_note="Build check only; not a graph certificate proof.")
        save()
        print("PASS: solver built at "+str(binary/"sdpa_gmp"))
        return 0
    except BaseException as error:
        metadata.update(status="FAIL", error=str(error))
        save()
        raise


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, RuntimeError, TimeoutError, subprocess.TimeoutExpired) as error:
        print("FAIL: "+str(error), file=sys.stderr)
        raise SystemExit(1)
