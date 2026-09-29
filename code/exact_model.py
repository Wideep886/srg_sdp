#!/usr/bin/env python3
"""Reconstruct and verify rational k-point bounds for spherical two-distance sets.

Supports two distinct rational inner products, 2 <= k <= min(6,n), and any
nonnegative harmonic degree d. No graph-order cutoff is imposed. Singular Gram
configurations are included when realizable. Dependent reference orbits require
an explicit metadata policy 'dependent_reference_policy': 'zero'; otherwise they
are rejected as unsupported by the independent-reference formulas.

Optional repair creates a DIFFERENT rational candidate and logs all changes.
It never accepts tolerances: the resulting matrices and original inequalities
must pass the same exact checks. Auxiliary numerical solver slacks are ignored.

Canonical graph types are obtained by covering each permutation orbit once, with a checked
lookup covering every labeled mask. Gram feasibility still uses exact PSD/rank
elimination, not a positive-definite-only filter.
"""
from pathlib import Path
from fractions import Fraction as F
from collections import defaultdict
from itertools import combinations, permutations, product
from functools import lru_cache
from decimal import Decimal, localcontext
import argparse
import hashlib
import json
import sys

if hasattr(sys, 'set_int_max_str_digits'):
    sys.set_int_max_str_digits(0)
BASE = Path(__file__).resolve().parent
from exact_arithmetic import (require, edges,
    inverse, bilinear, homogeneous_kernel, read_sdpa, brief)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    """Read unambiguous JSON, keeping decimal literals as exact strings."""
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, 'duplicate JSON key: ' + key)
            result[key] = value
        return result

    def reject_constant(value):
        raise ValueError('non-finite JSON value: ' + value)

    return json.loads(Path(path).read_text(), parse_float=str,
                      object_pairs_hook=unique_object, parse_constant=reject_constant)


def rational(value):
    require(isinstance(value, (str, int, F)) and not isinstance(value, bool),
            'exact rational must be a string, integer, or Fraction')
    return F(value)


def integer(value, name):
    q = rational(value)
    require(q.denominator == 1, name+' must be an integer')
    return q.numerator


def matrix(raw):
    require(isinstance(raw, list), 'matrix must be a list')
    require(all(isinstance(row, list) for row in raw), 'matrix rows must be lists')
    return [[rational(value) for value in row] for row in raw]


@lru_cache(None)
def permutation_orbit_cover(m):
    """Cover all binary edge masks by complete permutation orbits.

    The first uncovered mask is the least member of its orbit because masks
    are scanned in increasing order. All m! vertex permutations are applied
    to that representative, rather than canonizing every labeled graph anew.
    The returned lookup maps EVERY mask to its least orbit representative.
    Completeness follows from scanning the full labeled space; disjointness,
    least-representative convention and full coverage are checked explicitly.
    """
    require(isinstance(m, int) and 0 <= m <= 6, 'graph size outside 0..6')
    edge_order = edges(m)
    edge_positions = {edge: bit for bit, edge in enumerate(edge_order)}
    actions = tuple(tuple(1 << edge_positions[tuple(sorted((p[i], p[j]))) ]
                          for i, j in edge_order)
                    for p in permutations(range(m)))
    count = 1 << len(edge_order)
    canonical_of = [-1] * count
    representatives, orbit_sizes = [], []
    for seed in range(count):
        if canonical_of[seed] != -1:
            continue
        active_bits = tuple(i for i in range(len(edge_order)) if (seed >> i) & 1)
        orbit = {sum(action[i] for i in active_bits) for action in actions}
        require(bool(orbit) and seed == min(orbit), 'invalid least orbit representative')
        require(all(canonical_of[image] == -1 for image in orbit),
                'permutation orbit overlaps an earlier covered orbit')
        for image in orbit:
            canonical_of[image] = seed
        representatives.append(seed)
        orbit_sizes.append(len(orbit))
    require(all(leader >= 0 for leader in canonical_of) and
            sum(orbit_sizes) == count, 'incomplete labeled-mask coverage')
    return tuple(representatives), tuple(canonical_of), tuple(orbit_sizes)


def expected_orbits(m):
    return frozenset(permutation_orbit_cover(m)[0])


def gram_key(G, a, b):
    """Validate a Gram matrix and look up its fully enumerated graph type."""
    m = len(G)
    require(all(len(row) == m for row in G), 'non-square Gram')
    require(all(G[i][i] == 1 for i in range(m)), 'invalid Gram diagonal')
    require(all(G[i][j] == G[j][i] and G[i][j] in (a, b)
                for i, j in edges(m)), 'invalid Gram entry')
    mask = sum((G[i][j] == b) << bit for bit, (i, j) in enumerate(edges(m)))
    return m, permutation_orbit_cover(m)[1][mask]


def psd_info(A):
    """Exact semidefinite LDL elimination; a zero pivot needs a zero Schur row.

    For a PSD matrix, a zero diagonal entry forces its entire row to vanish by
    its 2x2 principal minors. Otherwise a positive pivot permits an invertible
    Schur-complement congruence. These two facts give an exact PSD/rank test,
    including singular matrices, without numerical eigenvalues or pivoting.
    """
    n = len(A)
    require(all(len(row) == n for row in A), 'non-square matrix')
    require(all(A[i][j] == A[j][i] for i in range(n) for j in range(n)),
            'matrix is not exactly symmetric')
    B = [row[:] for row in A]
    pivots = []
    for j in range(n):
        p = B[j][j]
        pivots.append(p)
        if p < 0:
            return {'psd':False,'failure':f'negative Schur pivot {j+1}',
                    'pivots':pivots,'rank':None,'positive_definite':False}
        if p == 0:
            if any(B[i][j] != 0 for i in range(j+1,n)):
                return {'psd':False,'failure':f'nonzero Schur row at zero pivot {j+1}',
                        'pivots':pivots,'rank':None,'positive_definite':False}
            continue
        for i in range(j+1,n):
            for h in range(i,n):
                B[i][h] -= B[i][j]*B[j][h]/p
                B[h][i] = B[i][h]
    rank = sum(p > 0 for p in pivots)
    return {'psd':True,'rank':rank,'positive_definite':rank == n,'pivots':pivots}


def make_gram(m, mask, a, b):
    G = [[F(i == j) for j in range(m)] for i in range(m)]
    for q,(i,j) in enumerate(edges(m)):
        G[i][j] = G[j][i] = b if (mask >> q)&1 else a
    return G


@lru_cache(None)
def realizable_orbits(n, a, b, m):
    result = {}
    for mask in sorted(expected_orbits(m)):
        G = make_gram(m,mask,a,b)
        info = psd_info(G)
        if info['psd'] and info['rank'] <= n:
            result[mask] = (G,info)
    return result


def reconstruct(meta):
    """Reconstruct all exact original inequalities, checking full metadata scope."""
    n,k,d = (integer(meta[key],key) for key in ('n','k','d'))
    a,b = rational(meta['a']),rational(meta['b'])
    require(n >= 2 and 2 <= k <= min(6,n) and d >= 0,
            'supported scope: n>=2, 2<=k<=min(6,n), d>=0')
    require(-1 <= a < 1 and -1 <= b < 1 and a != b,
            'need two distinct rational inner products in [-1,1)')
    policy = meta.get('dependent_reference_policy','reject')
    require(policy in ('reject','zero'), 'unknown dependent reference policy')
    feasible = {m:realizable_orbits(n,a,b,m) for m in range(k+1)}
    dependent = {(m,mask) for m in range(k-1) for mask,(_,q) in feasible[m].items()
                 if q['rank'] < m}
    require(not dependent or policy == 'zero',
            'dependent reference orbits need explicit zero policy: '+str(sorted(dependent)))
    active = {(m,mask) for m in range(k-1) for mask,(_,q) in feasible[m].items()
              if q['rank'] == m}
    blocks = meta['blocks']
    require(isinstance(blocks,list) and blocks, 'no matrix blocks')
    require(all(bb['kind'] in ('alpha','kernel','slack') for bb in blocks), 'unknown block kind')
    alpha = integer(meta['alpha_block'],'alpha_block')
    require(1 <= alpha <= len(blocks) and blocks[alpha-1]['kind'] == 'alpha', 'alpha block mismatch')
    require(sum(bb['kind']=='alpha' for bb in blocks)==1, 'need exactly one alpha block')
    anchors = {}
    for bi,spec in enumerate(blocks,1):
        dim = integer(spec['dim'],'block dim')
        if spec['kind'] != 'kernel':
            require(dim == 1, 'alpha/slack block must be scalar')
            continue
        require(dim >= 0, 'negative matrix order')
        G = matrix(spec['gram'])
        m,mask = gram_key(G,a,b)
        require(m == integer(spec['m'],'m') and (m,mask) in active, 'unsupported reference block')
        l = integer(spec['l'],'l')
        require(0 <= l <= d, 'layer outside stated cutoff')
        H = inverse(G)
        labels = [tuple(rational(x) for x in u) for u in spec['states']]
        require(len(labels)==dim and len(set(labels))==dim, 'duplicate or missing state rows')
        require(all(len(u)==m for u in labels), 'wrong label length')
        residual = {u:1-bilinear(u,H,u) for u in product((a,b),repeat=m)}
        gen = {u for u,h in residual.items() if h >= 0}
        strict_gen = {u for u,h in residual.items() if h > 0}
        refs = {tuple(G[i][j] for i in range(m)) for j in range(m)}
        if l == 0:
            require(set(labels)==gen|refs, 'degree-zero label set is incomplete or invalid')
        else:
            require(strict_gen <= set(labels) <= gen,
                    'positive layer omits a nonzero-residual state or has invalid states')
        autos = [p for p in permutations(range(m))
                 if all(G[p[i]][p[j]]==G[i][j] for i in range(m) for j in range(m))]
        key = (m,mask,l)
        require(key not in anchors, 'duplicate reference/layer block')
        anchors[key] = (bi,G,H,{u:i+1 for i,u in enumerate(labels)},autos)
    require(set(anchors)=={(m,mask,l) for m,mask in active for l in range(d+1)},
            'incomplete active reference/layer list')
    result,seen = [],defaultdict(set)
    for spec in meta['constraints']:
        S = matrix(spec['gram']);size,mask = gram_key(S,a,b)
        require(size==integer(spec['size'],'constraint size') and 1<=size<=k,
                'invalid constraint size')
        require(mask in feasible[size], 'unrealizable Gram configuration in metadata')
        require(mask not in seen[size], 'duplicate constraint orbit')
        seen[size].add(mask)
        terms = defaultdict(F)
        for m in range(min(size,k-2)+1):
            for Q in combinations(range(size),m):
                GQ = [[S[i][j] for j in Q] for i in Q]
                _,qm = gram_key(GQ,a,b)
                if (m,qm) in dependent:
                    continue  # explicitly declared zero local function
                for l in range(d+1):
                    bi,G,H,index,autos = anchors[m,qm,l]
                    p = next(p for p in permutations(range(m))
                             if all(GQ[p[i]][p[j]]==G[i][j] for i in range(m) for j in range(m)))
                    Qp = [Q[i] for i in p]
                    for x in range(size):
                        for y in range(size):
                            if set(Q)|{x,y} != set(range(size)):
                                continue
                            for auto in autos:
                                u = tuple(S[Qp[i]][x] for i in auto)
                                v = tuple(S[Qp[i]][y] for i in auto)
                                val = homogeneous_kernel(l,n,m,S[x][y],u,v,H)/len(autos)
                                if not val:
                                    continue
                                require(u in index and v in index,
                                        f'nonzero required label omitted from block {bi}')
                                i,j = sorted((index[u],index[v]))
                                terms[bi,i,j] += val
        rhs = F(1 if size==1 else 2 if size==2 else 0)
        if 'rhs' in spec:
            require(rational(spec['rhs'])==rhs, 'incorrect constraint constant')
        if size==1:
            terms[alpha,1,1] -= 1
        slack = integer(spec['slack_block'],'slack_block')
        require(1<=slack<=len(blocks) and blocks[slack-1]['kind']=='slack', 'bad slack index')
        result.append({'size':size,'mask':mask,'rhs':rhs,'slack_block':slack,
                       'terms':{key:v for key,v in terms.items() if v}})
    require(all(seen[m]==set(feasible[m]) for m in range(1,k+1)),
            'incomplete realizable configuration orbit set (including singular configurations)')
    slack_indices = {i for i,b in enumerate(blocks,1) if b['kind']=='slack'}
    require(len(result)==len(slack_indices)==len({c['slack_block'] for c in result}),
            'missing, duplicate, or unused scalar slack blocks')
    all_pivots = [p for m in range(1,k+1) for _,info in feasible[m].values() for p in info['pivots']]
    audit = {'n':n,'a':str(a),'b':str(b),'k':k,'d':d,
             'orbit_enumeration':'full permutation orbit cover; every labeled mask covered',
             'labeled_mask_counts':{str(m):len(permutation_orbit_cover(m)[1]) for m in range(k+1)},
             'all_graph_orbit_counts':{str(m):len(expected_orbits(m)) for m in range(k+1)},
             'orbit_counts':{str(m):len(feasible[m]) for m in range(k+1)},
             'singular_configuration_counts':{str(m):sum(not info['positive_definite'] for _,info in feasible[m].values()) for m in range(1,k+1)},
             'zero_dependent_references':sorted(dependent),'kernel_blocks':len(anchors),
             'constraint_count':len(result),'minimum_gram_ldl_pivot':str(min(all_pivots)),
             'all_realizable_configurations_positive_definite':all(p>0 for p in all_pivots)}
    return result,audit


def export_coefficients(meta_path, output):
    cons,audit = reconstruct(read_json(meta_path))
    path = Path(output);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w') as stream:
        stream.write('# constraint\tblock\trow\tcolumn\texact scalar coefficient\n')
        for ci,c in enumerate(cons,1):
            for (bi,i,j),q in sorted(c['terms'].items()):
                stream.write(f'{ci}\t{bi}\t{i}\t{j}\t{q}\n')
    return cons,audit


def linear_values(cons,mats):
    return [c['rhs']+sum((q*mats[b-1][i-1][j-1] for (b,i,j),q in c['terms'].items()),F(0)) for c in cons]


def attempt_repair(meta,cons,mats,eta):
    """Only scale all PSD blocks and increase alpha; higher-order signs persist."""
    ai=integer(meta['alpha_block'],'alpha_block')-1
    values=linear_values(cons,mats)
    require(all(v<=0 for c,v in zip(cons,values) if c['size']>=3),
            'repair cannot fix a positive higher-order original residual')
    factor=F(1)
    for c,r in zip(cons,values):
        if c['size']==2:
            B=r-2
            require(B<0, 'repair needs a negative pair linear form')
            factor=max(factor,(2+eta)/(-B))
    for spec,M in zip(meta['blocks'],mats):
        if spec['kind']=='kernel':
            for row in M:
                for i in range(len(row)):
                    row[i]*=factor
    values=linear_values(cons,mats)
    alpha_before=mats[ai][0][0]
    alpha_after=max([alpha_before,F(0)]+[alpha_before+r+eta for c,r in zip(cons,values) if c['size']==1])
    mats[ai][0][0]=alpha_after
    return {'kernel_scale':str(factor),'alpha_before':str(alpha_before),
            'alpha_after':str(alpha_after),'requested_pair_singleton_margin':str(eta)}


def verify(meta_path,candidate_path,out_dir,*,repair=False,eta=F('1e-30'),
           diagonal_shift=F(0),alpha_override=None,digits=6):
    meta_path,candidate_path,out_dir=map(Path,(meta_path,candidate_path,out_dir))
    out_dir.mkdir(parents=True,exist_ok=True)
    meta=read_json(meta_path)
    cons,audit=export_coefficients(meta_path,out_dir/'exact_coefficients.tsv')
    if candidate_path.suffix=='.json':
        candidate=read_json(candidate_path)
        if 'metadata_sha256' in candidate:
            require(candidate['metadata_sha256']==sha(meta_path),'candidate metadata hash mismatch')
        mats=[matrix(M) for M in candidate['matrices']]
    else:
        mats=read_sdpa(candidate_path)
    require(len(mats)==len(meta['blocks']),'wrong matrix block count')
    require(diagonal_shift>=0 and eta>0 and digits>=0,'invalid repair/report parameter')
    ai=integer(meta['alpha_block'],'alpha_block')-1
    original_alpha=mats[ai][0][0]
    changes={'symmetrized_blocks':[],'diagonal_shift':str(diagonal_shift),'automatic_repair_requested':repair}
    for bi,(spec,M) in enumerate(zip(meta['blocks'],mats),1):
        dim=integer(spec['dim'],'dim')
        require(len(M)==dim and all(len(row)==dim for row in M),f'block {bi}: wrong shape')
        symmetric=all(M[i][j]==M[j][i] for i in range(dim) for j in range(dim))
        if not symmetric and repair and spec['kind']=='kernel':
            M=[[F(M[i][j]+M[j][i],2) for j in range(dim)]for i in range(dim)]
            mats[bi-1]=M;changes['symmetrized_blocks'].append(bi)
        else:
            require(symmetric,f'block {bi}: not exactly symmetric')
        if spec['kind']=='kernel' and diagonal_shift:
            for i in range(dim):M[i][i]+=diagonal_shift
    repair_failure=None
    if repair:
        if all(psd_info(M)['psd'] for spec,M in zip(meta['blocks'],mats)if spec['kind']=='kernel'):
            try:changes.update(attempt_repair(meta,cons,mats,eta))
            except ValueError as error:repair_failure=str(error)
        else:repair_failure='candidate matrices are not PSD after the explicit diagonal shift'
    if alpha_override is not None:
        mats[ai][0][0]=rational(alpha_override)
        changes['explicit_alpha_override']=str(mats[ai][0][0])
    alpha=mats[ai][0][0]
    results=[]
    for bi,(spec,M) in enumerate(zip(meta['blocks'],mats),1):
        if spec['kind']!='kernel':continue
        info=psd_info(M)
        pivots=info.pop('pivots')
        minimum=min(pivots) if pivots else None
        results.append(dict(info,block=bi,dim=len(M),
            min_ldl_pivot=str(minimum)if minimum is not None else None,
            min_ldl_pivot_decimal=brief(minimum)if minimum is not None else None))
    residuals=linear_values(cons,mats)
    values=[{'constraint':i,'size':c['size'],'canonical_mask':c['mask'],
             'residual':str(r),'residual_decimal':brief(r),'feasible':r<=0,'strict':r<0}
            for i,(c,r) in enumerate(zip(cons,residuals),1)]
    passed=alpha>=0 and all(r['psd'] for r in results)and all(v<=0 for v in residuals)
    unit=10**digits
    upper_numerator=(alpha*unit).__floor__()+1
    upper=F(upper_numerator,unit)
    # Exact fixed-point formatting, including very large or negative rationals.
    upper_sign='-' if upper_numerator<0 else ''
    upper_abs=abs(upper_numerator)
    upper_text=(f'{upper_sign}{upper_abs//unit}.{upper_abs%unit:0{digits}d}'
                if digits else f'{upper_sign}{upper_abs}')
    epsilon=F('1e-30'); safety_eta=F('1e-24')
    shifted=[]
    for bi,(spec,M) in enumerate(zip(meta['blocks'],mats),1):
        if spec['kind']!='kernel':continue
        shift_info=psd_info([[v-epsilon if i==j else v for j,v in enumerate(row)]
                             for i,row in enumerate(M)])
        shifted.append({'block':bi,'positive_definite':shift_info['positive_definite']})
    all_pivots=[F(r['min_ldl_pivot']) for r in results if r['min_ldl_pivot']is not None]
    summary={'schema':'experiment-kpoint-exact-verification-v1','verified':passed,
       'case':str(meta.get('case','unspecified')),'model':audit,'python_version':sys.version,
       'arithmetic':'exact Python Fraction and integers; no feasibility tolerance',
       'verifier_sha256':sha(__file__),'imported_original_verifier_sha256':sha(BASE/'exact_arithmetic.py'),
       'metadata_sha256':sha(meta_path),'candidate_sha256':sha(candidate_path),
       'original_alpha':str(original_alpha),'alpha':str(alpha),'alpha_decimal':brief(alpha),
       'rational_strict_upper_bound':str(upper),'upper_decimal':upper_text,
       'integer_bound':alpha.__floor__()if passed else None,
       'bound_verified':passed,'no_fixed_target_threshold':True,
       'alpha_nonnegative':alpha>=0,'all_matrices_psd':all(r['psd']for r in results),
       'all_matrices_positive_definite':all(r['positive_definite']for r in results),
       'all_original_constraints_strict':all(r<0 for r in residuals),
       'maximum_original_residual':str(max(residuals)),
       'maximum_original_residual_decimal':brief(max(residuals)),
       'minimum_matrix_ldl_pivot':str(min(all_pivots))if all_pivots else None,
       'minimum_matrix_ldl_pivot_decimal':brief(min(all_pivots))if all_pivots else None,
       'fixed_safety_margins':{'matrix_shift':str(epsilon),'linear_slack':str(safety_eta),
           'all_shifted_matrices_positive_definite':all(r['positive_definite']for r in shifted),
           'all_residuals_below_negative_slack':all(r < -safety_eta for r in residuals),
           'both_margins_verified':all(r['positive_definite']for r in shifted)and all(r < -safety_eta for r in residuals),
           'acceptance_requires_these_margins':False,'shifted_blocks':shifted},
       'transformations':changes,'repair_failure':repair_failure,'psd_blocks':results,'constraints':values}
    (out_dir/'exact_verification.json').write_text(json.dumps(summary,indent=2)+'\n')
    certificate={'schema':'experiment-kpoint-rational-dual-v1','verified':passed,
                 'metadata_sha256':sha(meta_path),'matrices':[[[str(x)for x in row]for row in M]for M in mats],
                 'transformations':changes,'source_candidate_sha256':sha(candidate_path)}
    (out_dir/('rational_dual_certificate.json'if passed else'rational_dual_candidate.json')).write_text(json.dumps(certificate,indent=2)+'\n')
    print(json.dumps({key:summary[key]for key in ('verified','case','alpha_decimal','upper_decimal','integer_bound','all_matrices_positive_definite','all_original_constraints_strict','maximum_original_residual_decimal','repair_failure')},indent=2),flush=True)
    return passed


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--metadata',required=True,type=Path)
    ap.add_argument('--candidate',type=Path)
    ap.add_argument('--output',required=True,type=Path)
    ap.add_argument('--export-only',action='store_true')
    ap.add_argument('--repair',action='store_true',help='exact symmetrization, positive scaling and singleton alpha increase only')
    ap.add_argument('--eta',default='1e-30',help='positive target margin for optional pair/singleton repair')
    ap.add_argument('--diagonal-shift',default='0',help='explicit nonnegative addition to all kernel diagonals; fully reverified')
    ap.add_argument('--alpha',help='explicit rational replacement objective; fully reverified')
    ap.add_argument('--digits',type=int,default=6)
    args=ap.parse_args()
    if args.export_only:
        _,audit=export_coefficients(args.metadata,args.output/'exact_coefficients.tsv')
        (args.output/'model_exact_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
        print(json.dumps(audit,indent=2));return True
    require(args.candidate is not None,'--candidate required')
    return verify(args.metadata,args.candidate,args.output,repair=args.repair,eta=F(args.eta),
                  diagonal_shift=F(args.diagonal_shift),alpha_override=args.alpha,digits=args.digits)


if __name__=='__main__':
    try:sys.exit(0 if main()else 1)
    except (ValueError,KeyError,IndexError,StopIteration,ZeroDivisionError,OSError,TypeError)as error:
        print('CERTIFICATION FAILED:',error,file=sys.stderr,flush=True);sys.exit(1)
