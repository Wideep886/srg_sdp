#!/usr/bin/env python3
"""Independent all-degree rational certificate audit; no project imports.

Finite Gegenbauer sums (with the Chebyshev limit in residual dimension two),
adjugate reference inverses, every matching reference ordering, complete
permutation orbits, all principal Gram minors, and integer Bareiss leading
minors provide a separate implementation of the certificate check. No saved
verification report, parameter catalog, or numerical solver is used.
"""
from fractions import Fraction as Q
from itertools import combinations, permutations, product
from collections import defaultdict
from functools import lru_cache
from math import factorial, lcm
from pathlib import Path
import argparse, hashlib, json, sys, time

if hasattr(sys, 'set_int_max_str_digits'):
    sys.set_int_max_str_digits(0)
EPS, ETA = Q('1e-30'), Q('1e-26')


def require(test, message):
    if not test:
        raise ValueError(message)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read(p):
    def distinct(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, 'duplicate JSON key: ' + key)
            result[key] = value
        return result
    def reject_constant(value):
        raise ValueError('invalid JSON numeric constant: ' + value)
    return json.loads(Path(p).read_text(), parse_float=str,
                      object_pairs_hook=distinct, parse_constant=reject_constant)


def rational(value):
    require(isinstance(value, (str, int, Q)) and not isinstance(value, bool),
            'expected an exact rational string or integer')
    return Q(value)


def integer(value):
    value = rational(value)
    require(value.denominator == 1, 'nonintegral metadata field')
    return value.numerator


def mat(raw):
    require(isinstance(raw, (list, tuple)) and
            all(isinstance(row, (list, tuple)) for row in raw), 'matrix must be nested arrays')
    return tuple(tuple(rational(v) for v in row) for row in raw)


def scope(meta):
    n, k, d = (integer(meta[key]) for key in ('n', 'k', 'd'))
    a, b = rational(meta['a']), rational(meta['b'])
    require(n >= 2 and 2 <= k <= min(6, n) and d >= 0,
            'supported scope: n>=2, 2<=k<=min(6,n), d>=0')
    require(-1 <= a < 1 and -1 <= b < 1 and a != b, 'invalid two inner products')
    return n, k, d, a, b


def det_small(A):
    n = len(A)
    if not n:
        return Q(1)
    return sum(((-1)**j * A[0][j] * det_small(tuple(tuple(row[h] for h in range(n) if h!=j)
                   for row in A[1:])) for j in range(n)), Q(0))


@lru_cache(None)
def adjugate_inverse(A):
    n=len(A); determinant=det_small(A)
    require(determinant>0, 'reference determinant not positive')
    # Inverse(i,j) is cofactor(j,i)/det(A).
    inv=tuple(tuple((-1)**(i+j)*det_small(tuple(tuple(A[r][c] for c in range(n) if c!=i)
                for r in range(n) if r!=j))/determinant for j in range(n)) for i in range(n))
    require(all(sum((A[i][h]*inv[h][j] for h in range(n)),Q(0))==int(i==j)
                for i in range(n) for j in range(n)), 'adjugate inverse identity')
    return inv


def integer_matrix(A):
    den=1
    for row in A:
        for x in row:
            den=lcm(den,x.denominator)
    return [[int(x*den) for x in row] for row in A],den


def bareiss_det(A):
    n=len(A)
    if n==0:return 1
    B=[r[:] for r in A]; previous=1; sign=1
    for k in range(n-1):
        pivot_row=next((i for i in range(k,n) if B[i][k]),None)
        if pivot_row is None:return 0
        if pivot_row!=k:
            B[k],B[pivot_row]=B[pivot_row],B[k];sign=-sign
        pivot=B[k][k]
        for i in range(k+1,n):
            for j in range(k+1,n):
                value=B[i][j]*pivot-B[i][k]*B[k][j]
                quotient,remainder=divmod(value,previous)
                require(not remainder,'non-exact Bareiss division')
                B[i][j]=quotient
        for i in range(k+1,n):B[i][k]=0
        previous=pivot
    return sign*B[-1][-1]


def positive_leading_minors(A):
    require(all(len(r)==len(A) for r in A),'matrix not square')
    require(all(A[i][j]==A[j][i] for i in range(len(A)) for j in range(len(A))), 'asymmetric matrix')
    B,den=integer_matrix(A); previous=1; pivots=[]
    for k in range(len(B)):
        pivot=B[k][k]
        require(pivot>0, f'leading principal determinant {k+1} not positive')
        pivots.append(pivot)
        for i in range(k+1,len(B)):
            for j in range(k+1,len(B)):
                value=B[i][j]*pivot-B[i][k]*B[k][j]
                quotient,remainder=divmod(value,previous)
                require(not remainder,'non-exact Sylvester/Bareiss division')
                B[i][j]=quotient
        for i in range(k+1,len(B)):B[i][k]=0
        previous=pivot
    return len(pivots)


def gram_rank_by_all_minors(G, reject_negative=True):
    A,den=integer_matrix(G);rank=0;count=0
    for size in range(1,len(A)+1):
        for idx in combinations(range(len(A)),size):
            determinant=bareiss_det([[A[i][j] for j in idx] for i in idx]);count+=1
            if determinant < 0 and not reject_negative:
                return None, count
            require(determinant>=0,'negative principal Gram minor')
            if determinant:rank=max(rank,size)
    return rank,count


@lru_cache(None)
def orbit_catalog(size, n, a, b):
    """Visit every labeled graph, covering full permutation orbits independently."""
    edgepairs = tuple(combinations(range(size), 2))
    ordering = tuple(permutations(range(size)))
    covered, feasible = {}, {}
    total_types = total_minors = 0
    for bits in product((0, 1), repeat=len(edgepairs)):
        if bits in covered:
            continue
        G = [[Q(i == j) for j in range(size)] for i in range(size)]
        for (i, j), bit in zip(edgepairs, bits):
            G[i][j] = G[j][i] = b if bit else a
        G = tuple(map(tuple, G))
        orbit = {tuple(int(G[p[i]][p[j]] == b) for i, j in edgepairs)
                 for p in ordering}
        require(bits == min(orbit) and not any(item in covered for item in orbit),
                'overlapping permutation orbit')
        for item in orbit:
            covered[item] = bits
        rank, tested = gram_rank_by_all_minors(G, reject_negative=False)
        total_minors += tested
        total_types += 1
        if rank is not None and rank <= n:
            feasible[bits] = (G, rank, len(orbit))
    require(len(covered) == 2 ** len(edgepairs), 'incomplete labeled graph enumeration')
    return feasible, covered, total_types, total_minors


def gram_type(G, n, a, b):
    size = len(G)
    require(size <= 6 and all(len(row) == size for row in G), 'Gram shape')
    require(all(G[i][i] == 1 for i in range(size)), 'Gram diagonal')
    edges = tuple(combinations(range(size), 2))
    require(all(G[i][j] == G[j][i] and G[i][j] in (a, b) for i, j in edges),
            'Gram entry')
    bits = tuple(int(G[i][j] == b) for i, j in edges)
    return orbit_catalog(size, n, a, b)[1][bits]


def geometry(meta, a=None, b=None):
    n, k, d, a, b = scope(meta)
    seen = defaultdict(set)
    for constraint in meta['constraints']:
        G = mat(constraint['gram'])
        size = len(G)
        require(integer(constraint['size']) == size and 1 <= size <= k,
                'configuration size outside scope')
        key = gram_type(G, n, a, b)
        feasible = orbit_catalog(size, n, a, b)[0]
        require(key in feasible, 'unrealizable Gram configuration')
        require(key not in seen[size], 'duplicate configuration type')
        seen[size].add(key)
    counts, singular, minors = {}, {}, 0
    for size in range(1, k + 1):
        feasible, covered, all_types, tested = orbit_catalog(size, n, a, b)
        require(seen[size] == set(feasible), 'incomplete realizable configuration types')
        counts[size] = dict(types=len(feasible), realizable_labelled=sum(x[2] for x in feasible.values()),
                            all_types=all_types, all_labelled=len(covered))
        singular[size] = sum(rank < size for G, rank, count in feasible.values())
        minors += tested
    return dict(counts=counts, principal_minors_checked=minors, singular_types=singular)


def rising(x,n):
    answer=Q(1)
    for j in range(n):answer*=x+j
    return answer


@lru_cache(None)
def finite_coefficients(p,l):
    require(type(p) is int and p >= 2 and type(l) is int and l >= 0,
            'invalid residual dimension or degree')
    if l == 0:
        return (Q(1),)
    if p == 2:
        # C_l^lambda(t)/C_l^lambda(1) tends to T_l(t) as lambda -> 0.
        # For l>0, (lambda)_(l-j)/(2 lambda)_l tends to
        # (l-j-1)! / (2 (l-1)!). Substitution into the finite sum gives
        # (-1)^j*l*2^(l-2j)*(l-j-1)! / (2*j!*(l-2j)!).
        # The homogeneous form replaces t^(l-2j) by w^(l-2j)*s^j.
        return tuple(Q((-1)**j*l*2**(l-2*j)*factorial(l-j-1),
                       2*factorial(j)*factorial(l-2*j)) for j in range(l//2+1))
    lam=Q(p-2,2)
    normalization=Q(factorial(l),1)/rising(2*lam,l)
    return tuple(normalization*(-1)**j*2**(l-2*j)*rising(lam,l-j)/
        (factorial(j)*factorial(l-2*j)) for j in range(l//2+1))


@lru_cache(None)
def values(p,w,s,d):
    return tuple(sum((coef*w**(l-2*j)*s**j for j,coef in enumerate(finite_coefficients(p,l))),Q(0))
                 for l in range(d+1))


def reconstruct(meta):
    n, k, d, a, b = scope(meta)
    geometry(meta)
    policy = meta.get('dependent_reference_policy', 'reject')
    require(policy in ('reject', 'zero'), 'unknown dependent reference policy')
    active, dependent = set(), set()
    for size in range(k-1):
        for key, (G, rank, count) in orbit_catalog(size, n, a, b)[0].items():
            (active if rank == size else dependent).add((size, key))
    require(not dependent or policy == 'zero', 'dependent references require explicit zero policy')
    blocks = meta['blocks']
    require(isinstance(blocks, list) and blocks, 'missing matrix blocks')
    alpha = integer(meta['alpha_block'])
    require(1 <= alpha <= len(blocks) and blocks[alpha-1]['kind'] == 'alpha', 'alpha block mismatch')
    require(sum(block['kind'] == 'alpha' for block in blocks) == 1, 'alpha block must be unique')
    references, keys = {}, set()
    for bi, sp in enumerate(blocks, 1):
        dim = integer(sp['dim'])
        require(sp['kind'] in ('alpha', 'kernel', 'slack'), 'unknown block kind')
        if sp['kind'] != 'kernel':
            require(dim == 1, 'non-scalar auxiliary block')
            continue
        require(dim >= 0, 'negative block dimension')
        G = mat(sp['gram']); m = len(G); ell = integer(sp['l'])
        require(integer(sp['m']) == m and m <= k-2 and 0 <= ell <= d, 'invalid reference layer')
        key = gram_type(G, n, a, b)
        require((m, key) in active, 'reference is not an independent realizable type')
        require((m, key, ell) not in keys, 'duplicate reference layer')
        keys.add((m, key, ell))
        inverse = adjugate_inverse(G)
        labels = tuple(tuple(rational(x) for x in u) for u in sp['states'])
        require(len(labels) == len(set(labels)) == dim and all(len(u) == m for u in labels),
                'label dimensions or duplicates')
        residual = {u: 1-sum((u[i]*inverse[i][j]*u[j] for i in range(m) for j in range(m)), Q(0))
                    for u in product((a, b), repeat=m)}
        generic = {u for u, value in residual.items() if value >= 0}
        strict = {u for u, value in residual.items() if value > 0}
        refs = {tuple(G[i][j] for i in range(m)) for j in range(m)}
        if ell == 0:
            require(set(labels) == generic | refs, 'degree-zero label scope mismatch')
        else:
            require(strict <= set(labels) <= generic, 'positive-degree label scope mismatch')
        ref = references.setdefault(G, dict(blocks={}, inverse=inverse))
        ref['blocks'][ell] = (bi, {label:i+1 for i,label in enumerate(labels)})
    require(keys == {(m, key, ell) for m, key in active for ell in range(d+1)},
            'incomplete independent reference/layer list')
    for ref in references.values():
        require(set(ref['blocks']) == set(range(d+1)), 'reference ordering changes across layers')
    slacks = set()
    for constraint in meta['constraints']:
        size = integer(constraint['size'])
        require(rational(constraint['rhs']) == (1 if size == 1 else 2 if size == 2 else 0),
                'incorrect original constant')
        slack = integer(constraint['slack_block'])
        require(1 <= slack <= len(blocks) and blocks[slack-1]['kind'] == 'slack' and slack not in slacks,
                'missing or duplicate scalar slack')
        slacks.add(slack)
    require(slacks == {i for i, block in enumerate(blocks, 1) if block['kind'] == 'slack'},
            'unused scalar slack block')
    scalar=defaultdict(Q);orderings=0
    @lru_cache(None)
    def factors(G,u,v,t):
        H=references[G]['inverse'];m=len(G)
        def inner(x,y):return sum((x[i]*H[i][j]*y[j] for i in range(m) for j in range(m)),Q(0))
        hu,hv,w=1-inner(u,u),1-inner(v,v),t-inner(u,v)
        require(hu>=0 and hv>=0 and hu*hv-w*w>=0,'unrealizable two-point extension')
        return values(n-m,w,hu*hv,d)
    for ci,sp in enumerate(meta['constraints'],1):
        S=mat(sp['gram']);size=len(S);universe=set(range(size))
        for m in range(min(size,k-2)+1):
            for Qset in combinations(range(size),m):
                subgram = tuple(tuple(S[i][j] for j in Qset) for i in Qset)
                if (m, gram_type(subgram, n, a, b)) in dependent:
                    continue  # Explicitly declared zero local function.
                matches=[]
                for ordered in permutations(Qset):
                    G=tuple(tuple(S[i][j] for j in ordered) for i in ordered)
                    if G in references:matches.append((ordered,G))
                require(matches,'reference type missing from model')
                require(len({G for ordered,G in matches})==1,'duplicated reference orbit')
                orderings+=len(matches)
                pairs=[(x,y) for x in range(size) for y in range(size) if set(Qset)|{x,y}==universe]
                for ordered,G in matches:
                    for x,y in pairs:
                        u=tuple(S[r][x] for r in ordered);v=tuple(S[r][y] for r in ordered)
                        for ell,val in enumerate(factors(G,u,v,S[x][y])):
                            if not val:continue
                            bi,index=references[G]['blocks'][ell]
                            require(u in index and v in index,'nonzero coefficient label absent')
                            i,j=sorted((index[u],index[v]))
                            scalar[ci,bi,i,j]+=val/len(matches)
        if size==1:scalar[ci,alpha,1,1]-=1
    return {key:value for key,value in scalar.items() if value},orderings,factors.cache_info().currsize


def load_tsv(path):
    out={}
    for line in Path(path).read_text().splitlines():
        if not line.strip() or line.lstrip().startswith('#'):continue
        ci,b,i,j,value=line.split();key=tuple(map(int,(ci,b,i,j)))
        require(all(index >= 1 for index in key) and key[2] <= key[3],
                'invalid TSV position')
        require(key not in out,'duplicate TSV position')
        coefficient=Q(value)
        require(coefficient != 0, 'zero TSV coefficient must be omitted')
        out[key]=coefficient
    return out


def lift_and_check(fullmeta,compmeta,candidate):
    require(compmeta['representation'] in ('full', 'compressed'), 'unsupported representation')
    for key in ('n', 'k', 'd', 'alpha_block'):
        require(integer(fullmeta[key]) == integer(compmeta[key]), 'model scope mismatch: ' + key)
    for key in ('a', 'b'):
        require(rational(fullmeta[key]) == rational(compmeta[key]), 'model angles differ')
    require(fullmeta['constraints'] == compmeta['constraints'], 'model configuration lists differ')
    require(fullmeta.get('dependent_reference_policy', 'reject') ==
            compmeta.get('dependent_reference_policy', 'reject'), 'model reference policies differ')
    require(rational(compmeta['required_matrix_margin']) == EPS and
            rational(compmeta['required_linear_margin']) == ETA, 'required margins differ')
    require(len(fullmeta['blocks'])==len(compmeta['blocks'])==len(candidate['matrices']),'block count')
    result=[];checks=0
    for full,small,raw in zip(fullmeta['blocks'],compmeta['blocks'],candidate['matrices']):
        H=mat(raw);dim=integer(small['dim'])
        require(full['kind'] == small['kind'], 'block kind differs')
        require(len(H)==dim and all(len(r)==dim for r in H),'candidate shape')
        require(all(H[i][j] == H[j][i] for i in range(dim) for j in range(dim)), 'asymmetric candidate')
        if full['kind']!='kernel':
            require(dim == integer(full['dim']) == 1, 'non-scalar auxiliary candidate')
            result.append(H)
            continue
        require(integer(full['m']) == integer(small['m']) and
                integer(full['l']) == integer(small['l']) and mat(full['gram']) == mat(small['gram']),
                'reference or harmonic layer changed')
        if compmeta['representation'] == 'compressed' and integer(full['l']) == 0 and integer(full['m']):
            m=integer(full['m']);G=mat(full['gram']);labels=[tuple(rational(x) for x in u) for u in full['states']]
            refset={tuple(G[i][j] for i in range(m)) for j in range(m)}
            gen=[u for u in labels if u not in refset]
            require(len(refset) == m and sum(u in refset for u in labels) == m,
                    'reference label count mismatch')
            require(dim == len(gen) + 1 and integer(small['original_dim']) == len(labels)
                    and small.get('compression') == 'reference-sum', 'compressed dimension or mode')
            require([integer(i) for i in small['generic_indices']] ==
                    [i for i, u in enumerate(labels) if u not in refset], 'generic indices mismatch')
            require(sorted(integer(i) for i in small['reference_indices']) ==
                    [i for i, u in enumerate(labels) if u in refset], 'reference indices mismatch')
            require(small['states'][0]=='reference_sum' and
                    [tuple(Q(x) for x in u) for u in small['states'][1:]]==gen,'compressed order mismatch')
            inds={u:i+1 for i,u in enumerate(gen)}
            W=[m]+[1]*len(gen)
            positive_leading_minors(tuple(tuple(v-EPS*W[i] if i==j else v for j,v in enumerate(row)) for i,row in enumerate(H)));checks+=1
            M=[]
            for i,u in enumerate(labels):
                row=[]
                for j,v in enumerate(labels):
                    if u in refset and v in refset:value=H[0][0]/m**2+2*EPS*(int(i==j)-Q(1,m))
                    elif u in refset:value=H[0][inds[v]]/m
                    elif v in refset:value=H[inds[u]][0]/m
                    else:value=H[inds[u]][inds[v]]
                    row.append(value)
                M.append(tuple(row))
            M=tuple(M)
            refindices=tuple(i for i,u in enumerate(labels) if u in refset)
            groups=(refindices,)+tuple((labels.index(u),) for u in gen)
            require(all(sum((M[i][j] for i in left for j in right),Q(0))==H[r][t]
                        for r,left in enumerate(groups) for t,right in enumerate(groups)),
                    'exact lift/compression identity failed')
        else:
            require(small.get('compression') != 'reference-sum', 'unexpected compression')
            require(full['states']==small['states'] and full['dim']==dim,'unchanged block changed')
            M=H
        positive_leading_minors(tuple(tuple(v-EPS if i==j else v for j,v in enumerate(row)) for i,row in enumerate(M)));checks+=1
        result.append(M)
    return result,checks


def selftest():
    for p in range(2,61):
        for l in range(6):require(values(p,Q(1),Q(1),5)[l]==1,'normalization selftest')
        for w,s in [(Q(2,3),Q(7,8)),(Q(0),Q(0)),(Q(-3,5),Q(1))]:
            v=values(p,w,s,5)
            require(v[0]==1 and v[1]==w and v[2]==(p*w*w-s)/(p-1),'first degrees selftest')
    for n in range(1,5):
        for seed in range(8):
            A=[[((i+1)*(j+2)+seed*(i-j))%11-5 for j in range(n)] for i in range(n)]
            require(bareiss_det(A)==det_small(tuple(map(tuple,A))),'Bareiss vs cofactor selftest')
    positive_leading_minors(((Q(2),Q(1)),(Q(1),Q(2))))
    for A in [((Q(1),Q(2)),(Q(2),Q(1))),((Q(1),Q(0)),(Q(0),Q(0)))]:
        rejected=False
        try:positive_leading_minors(A)
        except ValueError:rejected=True
        require(rejected,'bad matrix accepted')


def audit(model_directory, solution_directory, output_json):
    """Audit a physical candidate using only its model and solution directories.

    Returns and writes a PASS/FAIL report. A failure has no integer bound;
    serialization success or a saved main-checker report is never proof here.
    """
    started = time.perf_counter()
    model = Path(model_directory).resolve()
    solution = Path(solution_directory).resolve()
    output = Path(output_json).resolve()
    paths = [model/'original_metadata.json', model/'metadata.json',
             model/'original_exact_coefficients.tsv', solution/'candidate.json']
    protected = {path.resolve() for path in paths}
    protected.update((model/name).resolve() for name in (
        'exact_coefficients.tsv', 'physical_shifts.json', 'problem.dat-s', 'solver.par', 'build.json'))
    protected.add((solution/'solver.out').resolve())
    result = dict(schema='independent-all-degree-exact-audit-v1', status='FAIL', verified=False,
        model_directory=str(model), solution_directory=str(solution), integer_bound=None,
        source_sha256=sha(__file__), no_project_code_imported=True, solver_runs=0,
        methods=['finite Gegenbauer sums; Chebyshev limit for residual dimension two',
                 'adjugate reference inverses and all matching orderings',
                 'complete permutation orbits and all principal Gram minors',
                 'integer Bareiss leading principal determinants'],
        matrix_margin=str(EPS), linear_margin=str(ETA))
    try:
        require(output not in protected, 'output would overwrite a model or solution input')
        before = {str(path): sha(path) for path in paths}
        full, comp, candidate = read(paths[0]), read(paths[1]), read(paths[3])
        n, k, d, a, b = scope(full)
        require(comp['original_metadata_sha256'] == before[str(paths[0])], 'original metadata hash mismatch')
        require(candidate['metadata_sha256'] == before[str(paths[1])], 'candidate metadata hash mismatch')
        geo = geometry(full)
        expected, orders, evaluations = reconstruct(full)
        stored = load_tsv(paths[2])
        require(expected == stored, 'independent original scalar coefficients disagree')
        matrices, checks = lift_and_check(full, comp, candidate)
        alpha = matrices[integer(full['alpha_block'])-1][0][0]
        require(alpha >= 1, 'alpha must be at least one')
        residuals = [Q(1 if integer(c['size']) == 1 else 2 if integer(c['size']) == 2 else 0)
                     for c in full['constraints']]
        for (ci, bi, i, j), value in expected.items():
            residuals[ci-1] += value * matrices[bi-1][i-1][j-1]
        require(all(value < -ETA for value in residuals), 'independent original residual violates margin')
        require(before == {str(path): sha(path) for path in paths}, 'input changed during audit')
        result.update(status='PASS', verified=True, n=n, k=k, d=d, a=str(a), b=str(b),
            independent_original_coefficients=len(expected), reference_orderings=orders,
            unique_polynomial_evaluations=evaluations, geometry=geo,
            strict_matrix_tests_by_bareiss=checks, original_residual_count=len(residuals),
            all_original_residuals_below_negative_margin=True,
            all_original_coefficients_equal=True, all_shifted_blocks_strictly_positive=True,
            inputs_unchanged=True, integer_bound=alpha.numerator//alpha.denominator,
            alpha_exact=str(alpha), maximum_residual_exact=str(max(residuals)),
            residuals=[str(value) for value in residuals], inputs=before)
    except Exception as error:
        result.update(error_type=type(error).__name__, error=str(error))
    result['seconds'] = time.perf_counter()-started
    output.parent.mkdir(parents=True, exist_ok=True)
    # Never overwrite model/solution inputs, even on the failure-report path.
    if output in protected:
        raise ValueError('output would overwrite a model or solution input')
    output.write_text(json.dumps(result, indent=2)+'\n')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, type=Path)
    parser.add_argument('--solution', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args(argv)
    result = audit(args.model, args.solution, args.output)
    print(json.dumps({key: result.get(key) for key in
          ('status', 'verified', 'integer_bound', 'seconds', 'error')}), flush=True)
    return 0 if result['verified'] else 1


if __name__ == '__main__':
    sys.exit(main())
