# Bundled solver sources and notices

The archives below are unmodified copies of the sources used for the numerical
search. The Python model builder and exact verifier do not require these sources;
rerunning the high-precision numerical search does. The numerical solver is
SDPA-GMP (C/C++), linked with GNU MP (GMP); it is not a Python solver.

| Archive | Source | SHA-256 |
| --- | --- | --- |
| `src/sdpa-gmp-7.1.3.tar.gz` | <https://www.daviddelaat.nl/sdpa-gmp-7.1.3.tar.gz> | `40ef73041e80e89bd51358f5c51f8eb7f02629d33dd26c14466cc790ef7be1e9` |
| `src/gmp-6.3.0.tar.xz` | <https://gmplib.org/download/gmp/gmp-6.3.0.tar.xz> | `a3c2b80201b89e68616f4ad30bc66aee4927c3ce50e33929ca819d5c43538898` |

The SDPA archive is named **7.1.3**. Its inherited `configure` metadata still
reports **7.1.2**. The archive hash identifies the exact source used here.
MPACK and SPOOLES 2.2 are included inside that archive. No download is needed
to build the solver from this package.

## Component licenses

The complete archives retain the original notices, license files, documentation,
and source code. Selected notices are also copied verbatim into `licenses/` for
convenience; these copies do not replace the notices in the archives.

- **SDPA:** source headers state GNU GPL version 2 or any later version.
  `licenses/SDPA-GPLv2.txt` is the archive's `COPYING` text;
  `licenses/SDPA-source-notice.cpp` reproduces its source notice.
- **GMP 6.3.0:** its README offers a choice of GNU LGPL version 3 or later,
  or GNU GPL version 2 or later (or both). See `licenses/GMP-README`,
  `GMP-COPYINGv2`, `GMP-COPYINGv3`, and `GMP-COPYING.LESSERv3`.
  The archive also includes the licenses applicable to its documentation.
- **Bundled MPACK:** its source headers state GNU LGPL version 3 only, and
  relevant routines retain the University of Tennessee/LAPACK permissive notice.
  `licenses/MPACK-Rpotrf-notices.cpp` reproduces both notices from `Rpotrf.cpp`.
  The LGPL version 3 and GPL version 3 texts are included among the GMP license
  copies listed above.
- **Bundled SPOOLES 2.2:** its reference manual describes the release as public
  domain and asks users to acknowledge the software. We acknowledge the work of
  Cleve Ashcraft, Daniel Pierce, David K. Wah, and Jason Wu.
  `licenses/SPOOLES-reference-manual-notice.tex` is the manual's opening file.
  The bundled Harwell--Boeing I/O code also has a NIST permission and warranty
  notice, retained in `licenses/SPOOLES-iohb-notice.c` and the original source.

These are component-specific terms. This notice does not assign a license to
the authors' Python programs, nor replace any upstream license.

## Local build

From the package root, use Python 3.9 or later:

```sh
python3 code/build_solver.py --check-sources
python3 code/build_solver.py --prefix /absolute/path/to/new-solver --jobs 4 --timeout 600
```

Omitting `--prefix` uses `extras/solver/` under the package root. The selected
directory must not already exist, and its path must not contain spaces or
shell/make metacharacters. Building requires macOS or a POSIX Linux environment
(including WSL), a C compiler, a C++ compiler, `make`, `m4`, `patch`, `tar`,
`gzip`, `ar`, `ranlib`, and `sh`. Supply `--cc` and `--cxx` to select compiler
executables. The 600-second limit applies to the entire build and tests, not to
each command; a slower machine may need a larger value. A failed build retains
its logs; a new attempt must use a new directory.

The build uses only these archives and installs into the selected directory.
It does not run a package manager, download dependencies, or install system
files. The SDPA source archive contains old compiled products; the script
removes them from the fresh extraction and recompiles the source. It replaces
SDPA's obsolete `config.guess` and `config.sub` with the corresponding platform
detection files bundled in GMP. It records their old and new hashes in
`build.json`. It makes no changes to the solver's numerical algorithms.

The script runs GMP's `make check`, a GMP version/precision probe, and the small
upstream `example1.dat-s` problem. The probe requests 1500 bits of precision;
GMP may round this upward to a whole number of machine limbs. The example must
reach `pdOPT` with objective within `10^-20` of `-41.9`. This example checks the
build; it is not a certificate for a spherical bound.

After success, the executable is `bin/sdpa_gmp` under the selected directory.
`build.json` records the source hashes, compiler commands, platform, elapsed
times, test results, and binary hash. Per-command logs are in `logs/`.
The exact verifier, rather than the solver's success status, determines whether
a proposed bound is accepted.
