# Code guide and provenance

The package has one command-line entry point: `code/srg.py`. The table below
explains the implementation and records the earlier filenames so that retained
reports can be traced to their source. Earlier filenames are not required on a
reader's computer.

| Current file in `code/` | Responsibility and origin |
|---|---|
| `srg.py` | Public commands and argument handling; added for the general parameter interface. |
| `parameters.py` | Exact SRG necessities, spectrum, eigenspace projectors, and injectivity; added for general inputs. |
| `workflow.py` | Run directories, solver execution, parameter/model linkage, and final reports; added for the unified workflow. |
| `model.py` | Builds exact coefficients, compressed blocks, shifts, and SDPA input. Adapted from the building routines in `run_case.py`; table lookups were replaced by the derived embedding. |
| `exact_arithmetic.py` | Shared rational arithmetic, homogeneous function formulas, and solver parsing. Retains the used helpers from `rigorous_original_dual.py`; unused case-specific programs were removed. |
| `exact_model.py` | Enumerates Gram configurations and reconstructs exact matrix coefficients and inequalities. Adapted from `experiment_exact_k6.py`. |
| `check_matrices.py` | Checks compressed matrices, their lift, and original residuals. Adapted from `verify_compressed_k6.py`. |
| `check_compression.py` | Independently checks degree-zero union formulas and configuration coverage. Adapted from `direct_zero_check_k6.py`. |
| `check_solver_io.py` | Audits coefficient rounding, shifts, block correspondence, and parsed output. Adapted from `export_audit_case.py`. |
| `verify.py` | Coordinates the required checks and writes the verification summary. Adapted from `verify_case.py` for general point counts and degrees. |
| `independent_check.py` | Optional separate coefficient reconstruction and integer Bareiss matrix check. Adapted from `independent_certificate_audit.py` to accept explicit input paths. |
| `build_solver.py` | Builds and tests the bundled native solver. Relocated from the earlier build helper and made compatible with Python 3.9. |

The ordinary model builder and verifier share some formula code. The optional
independent checker imports no other project module. These are distinct levels
of cross-checking; the package does not claim that all checks are independent
implementations.

## General inputs and model restrictions

The original experiments used rational inner products and independent reference
configurations. The public interface derives the embedding from `(v,k,λ,μ)`
instead of selecting a stored case. It supports rational injective embeddings,
with point count `2..min(6,n)` and nonnegative degree.

For dependent reference configurations, the local function is explicitly zero.
This is a valid restriction of the dual search and may weaken the bound.
Singular configurations still appear in the inequalities. Logical blocks of
dimension zero are omitted only from SDPA input and restored as empty matrices
when reading the solver output. No irrational angles are approximated; inputs
requiring them are reported as unsupported.

## Retained certificates and regression data

The four certificates were retained from the original experiments. Their model
files, candidate matrices, and raw solver outputs have not been altered.

Each `historical_verification.json` records a previous verification. The `verify`
command always recomputes the checks and writes a new report. Historical paths
and source hashes identify the original run; they are not installation
requirements. Some original `build.json` files name earlier source snapshots
that are not bundled. Verification does not depend on those snapshots. A new
`solve` command constructs a fresh model and records its current inputs.

The two `extras/tests/fixtures/550_k2_d5_*` files retain archived two-point
metadata and a rational coefficient table. Regression tests check their hashes
and compare freshly reconstructed coefficients. These fixtures do not limit
which graph parameters can be entered.

See [VALIDATION.md](VALIDATION.md) for the checks actually performed and
[third_party/README.md](third_party/README.md) for upstream sources and notices.
SHA-256 lists record file integrity, not mathematical feasibility.
