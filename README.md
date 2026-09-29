# SRG bounds: search and exact verification

[繁體中文](README_zh-TW.md)

This package computes spherical bounds from strongly regular graph parameters **(v, k, λ, μ)**. Here `v` is the number of vertices, `k` is the degree, and adjacent/nonadjacent vertices have `λ`/`μ` common neighbors. The option `--points` specifies the number of points in the SDP; it is distinct from the graph degree `k`.

If such a graph exists, an injective eigenspace embedding gives `v` distinct unit vectors with two prescribed inner products. The program bounds the size of this spherical configuration. **An exactly verified integer bound smaller than `v` proves graph nonexistence.** A bound greater than or equal to `v` does not settle existence. New input is derived from the four parameters, with no case whitelist.

## 1. Check a saved certificate — no solver needed

Use **Python 3.9 or later**; the Python code needs only the standard library. Keep the package together and open a terminal in the folder containing `code/`, `results/`, and `extras/`. GitHub's **Code → Download ZIP** usually extracts to `srg_sdp-main`; the folder may be renamed.

On macOS, type `cd `, drag the extracted folder from Finder into Terminal, and press Enter. Run `pwd` to check the location. The commands below assume this folder is your working directory.

```sh
python3 code/srg.py list
python3 code/srg.py verify --certificate 550_162_75_36
```

`list` shows the four certificates below. `verify` recomputes the exact checks and creates a new report under `results/runs/`; it neither runs the solver nor trusts a saved `PASS` flag. Six-point verification may take several minutes. The saved certificates remain unchanged.

| Certificate ID | Parameters | Saved integer bound |
|---|---|---:|
| `351_140_73_44` | (351, 140, 73, 44) | 348 |
| `550_162_75_36` | (550, 162, 75, 36) | 506 |
| `703_182_81_35` | (703, 182, 81, 35) | 592 |
| `1344_221_88_26` | (1344, 221, 88, 26) | 1307 |

All four use six points and degree five. Replace the ID to check another certificate. Add `--independent` for an optional additional check with separate all-degree coefficient formulas and integer Bareiss matrix elimination. That checker imports no other project code; the ordinary builder and verifier do share formula code.

## 2. Inspect a new parameter set

```sh
python3 code/srg.py inspect --srg 550 162 75 36
```

This checks necessary parameter and spectrum identities exactly, then reports the embedding dimension `n` and inner products `a,b`. The default selects the smallest supported **injective** embedding. Use `--eigenspace r` or `--eigenspace s` to select the larger or smaller restricted eigenvalue explicitly.

| Status | Meaning |
|---|---|
| `supported` | The necessary checks pass and a supported embedding is available; this does not prove existence. |
| `infeasible` | An exact necessary condition fails; the report explains which one. |
| `unsupported` | The input is outside the implemented scope; this is not a nonexistence result. |
| `invalid` | The input or requested parameters are invalid. |

**The backend uses rational numbers only.** Admissible conference parameters with nonsquare discriminant, such as `(5,2,0,1)`, give irrational angles and are `unsupported`; they are not approximated. Empty and complete graphs are outside this two-distance interface.

## 3. Set up a solver for new searches

Numerical searches use **SDPA-GMP**, a C/C++ solver using GMP. Python constructs the model and verifies the candidate. Checking saved certificates does not need SDPA-GMP.

Choose **one** setup method: use an existing executable, or compile the bundled sources.

```sh
python3 code/srg.py setup --solver "/absolute/path/to/sdpa_gmp"
```

```sh
python3 code/srg.py setup --build
```

Replace example paths with actual paths. Setup saves the executable location in `extras/local_settings.json`. Run `python3 code/srg.py doctor` to see the Python, package, and solver locations.

Building requires macOS or Linux/WSL, C/C++ compilers, and standard build tools; see [build requirements and source licenses](extras/third_party/README.md). No source download is needed. The build checks source hashes, runs GMP tests and a solver example, and keeps its logs. Native Windows is not supported by this build workflow.

The default build directory is `extras/solver/`. For a package path containing spaces or shell/make special characters, use `setup --build --prefix "/absolute/path/without_spaces/srg_solver"`. The build directory must be new. **`setup --build --timeout 1200` limits the whole build and its tests**; increase it if needed.

## 4. Build, solve, and verify

```sh
python3 code/srg.py solve --srg 550 162 75 36 --points 6 --degree 5
```

This creates a fresh directory under `results/runs/`, builds the model, runs SDPA-GMP, and verifies the candidate exactly. Add `--independent` for the separate check. Replace the four graph parameters with any supported input; the program does not guarantee a nonexistence proof for every input.

| Option | Default | Meaning |
|---|---:|---|
| `--points` | `min(6,n)` | Integer from `2` to `min(6,n)`. |
| `--degree` | `5` | Maximum polynomial degree, a nonnegative integer. |
| `--eigenspace` | `auto` | Smallest supported injective embedding, or explicit `r`/`s`. |
| `--precision` | `1500` | Requested solver precision in bits. |
| `--iterations` | `2000` | Maximum solver iterations. |
| `--epsilon` | `1e-16` | Solver stopping tolerance, not a proof acceptance tolerance. |
| `--timeout` | `1200` | Solver seconds; excludes model construction and exact verification. |
| `--output` | New directory | Optional output path; existing directories are not overwritten. |

To inspect the model before solving, replace `solve` with `prepare`. It needs no solver and prints the next `run` command. To recheck a completed run, use its root directory and a fresh output directory:

```sh
python3 code/srg.py verify --run "/absolute/path/to/run" --output "/absolute/path/to/new_check"
```

For execution from another working directory, use the absolute path to `code/srg.py` and absolute input/output paths. Quote paths containing spaces. Default resources are located relative to the script, not the terminal directory.

## Read the result

The command prints the output location. After verification finishes, its `result.json` contains:

| Field | Meaning |
|---|---|
| `verified` | The requested exact checks passed. |
| `certified_integer_bound` | The resulting integer upper bound, or `null` without a certificate. |
| `nonexistence_proved` | Verification passed and the bound is strictly smaller than `v`. |
| `verification_summary` | Path to the detailed report. |

**Solver success alone is not a proof.** A failed search, timeout, or failed exact check gives no certified bound and does not settle graph existence. Failed checks return a nonzero exit code and retain diagnostics. Larger models can take substantially longer; the solver timeout is not a time limit for the entire workflow.

## What is checked, and what is restricted?

The verifier reads printed decimals as exact rational numbers, restores recorded diagonal shifts, and reconstructs the original inequalities. Required checks cover matrix positivity, degree-zero compression and its lift, every original residual, coefficient rounding, and the correspondence between solver output and candidate matrices.

Verification requires matrix margins of `10^-30` and original residuals strictly below `-10^-26`, checked exactly. Coefficients are exported with 600 significant decimal digits; their rounding errors are audited as rational numbers. Solver stopping tolerances do not change these proof requirements.

All realizable Gram configurations up to the requested point count enter the inequalities, including singular ones. For **dependent reference configurations**, the local function is set to zero: this restricts the search and may weaken the bound. Degree-zero blocks retain realizable boundary labels. Positive-degree contributions vanish at zero residual norm, so those labels may be omitted there. Empty matrices remain in the exact model; only the solver transport omits them and restores them as `[]`.

The [validation record](extras/VALIDATION.md) states which checks were actually run. Testing software and checking file hashes do not replace exact certificate verification.

## Files, integrity, and attribution

| Location | Purpose |
|---|---|
| `code/` | Python programs; start with `srg.py`. [File-by-file description](extras/SOURCE_HISTORY.md). |
| `results/certificates/` | Four saved models, candidates, solver outputs, and historical reports. |
| `results/runs/` | New searches and fresh verification reports, created when needed. |
| `results/release_checks/` | Reports supporting the documented release checks. |
| `extras/third_party/` | Solver/GMP sources, source hashes, and original notices. |
| `extras/tests/` | Regression tests and small fixtures. |
| `extras/VALIDATION.md`, `extras/SOURCE_HISTORY.md` | Validation scope and code provenance. |
| `extras/SHA256SUMS` | SHA-256 fingerprints for the distributed files. |

Historical records may contain original paths and source hashes. They identify past runs; you do not need those paths on your computer. Local settings, solver builds, caches, and new run outputs are excluded from Git.

From the project root, `shasum -a 256 -c extras/SHA256SUMS` checks file integrity. SHA-256 detects changes relative to the supplied list; it neither proves the mathematics nor authenticates a list replaced together with the files. The checksum list does not include itself or files generated by your runs.

The authors' Python programs in `code/` and tests in `extras/tests/` are provided under the [MIT License](LICENSE), with copyright attributed to **srg_sdp contributors**. You may use, modify, and redistribute them, including commercially, while retaining the copyright and license notice.

The MIT license does not replace the terms for third-party software. SDPA-GMP, GMP, MPACK, and SPOOLES retain their own notices and licenses; see [third-party attribution](extras/third_party/README.md). Python itself is not bundled.
