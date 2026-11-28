#!/usr/bin/env python3
"""GBU-DEA outstanding analyses for the revised 3-input/3-output bank data.

Runs (i) all 12 (delta, Gamma_G) scenarios for GBU and product-calibrated
flat BS, (ii) the local-only comparator, (iii) bank-level monotonicity checks,
and (iv) Proposition 6 equality frequencies using the optimal intensity vector.

Place the source workbook in data/ or set INPUT_FILE to its path.
Requires pandas, numpy, scipy, openpyxl.
"""
from pathlib import Path
import json, time, warnings
import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import lil_matrix, csr_matrix

# ---------------------------- USER SETTINGS ----------------------------
INPUT_FILE = Path('data/1-s2.0-S2352340918314410-mmc2.xlsx')
OUTPUT_DIR = Path('results/GBU_DEA_outstanding_3x3_results')
DELTAS = (0.01, 0.025, 0.05)
GLOBAL_BUDGETS = (1, 2, 3, 5)
LOCAL_BUDGET = 1.0
SOLVER_METHOD = 'highs-ds'
CHECKPOINT_EVERY_TARGETS = 5
# ----------------------------------------------------------------------

INPUT_COLS = [' Employees (I)', 'Assets (I)', 'Personnel Expenses (I)']
OUTPUT_COLS = ['Loans (O)', 'Net Income Revenue (O)', 'Net Fees Commission (O)']
ID_COL = 'BANKS'
COUNTRY_COL = 'COUNTRY'
REGION_MAP = {
    'Czech Republic':'Eastern Europe','Hungary':'Eastern Europe','Poland':'Eastern Europe',
    'Romania':'Eastern Europe','Slovakia':'Eastern Europe','Slovenia':'Eastern Europe',
    'Denmark':'Northern Europe','Finland':'Northern Europe','Ireland':'Northern Europe',
    'Sweden':'Northern Europe','United Kingdom':'Northern Europe',
    'Croatia':'Southern Europe','Cyprus':'Southern Europe','Greece':'Southern Europe',
    'Italy':'Southern Europe','Portugal':'Southern Europe','Spain':'Southern Europe',
    'Austria':'Western Europe','Belgium':'Western Europe','France':'Western Europe',
    'Germany':'Western Europe','Luxembourg':'Western Europe','Netherlands':'Western Europe'
}
REGION_ORDER = ['Eastern Europe','Northern Europe','Southern Europe','Western Europe']


def load_data():
    if not INPUT_FILE.is_file():
        raise FileNotFoundError(
            f'Workbook not found: {INPUT_FILE}\n'
            'Set INPUT_FILE to the location of the original supplementary workbook.'
        )
    df = pd.read_excel(INPUT_FILE)
    needed = [ID_COL, COUNTRY_COL] + INPUT_COLS + OUTPUT_COLS
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f'Missing workbook columns: {missing}\nAvailable: {list(df.columns)}')
    df = df[needed].copy()
    df[COUNTRY_COL] = df[COUNTRY_COL].astype(str).str.strip()
    df['REGION'] = df[COUNTRY_COL].map(REGION_MAP)
    if df['REGION'].isna().any():
        bad = sorted(df.loc[df.REGION.isna(), COUNTRY_COL].unique())
        raise ValueError(f'Countries missing from REGION_MAP: {bad}')
    for c in INPUT_COLS + OUTPUT_COLS:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    if df[needed].isna().any().any():
        raise ValueError('Missing/non-numeric values found in selected fields.')
    if (df[INPUT_COLS + OUTPUT_COLS] <= 0).any().any():
        raise ValueError('This implementation assumes strictly positive bank inputs and outputs.')
    expected = {'Eastern Europe':22,'Northern Europe':33,'Southern Europe':66,'Western Europe':129}
    got = df.REGION.value_counts().to_dict()
    if got != expected:
        raise ValueError(f'Regional counts do not match manuscript: {got}; expected {expected}')
    if len(df) != 250:
        raise ValueError(f'Expected 250 banks; found {len(df)}')
    return df.reset_index(drop=True)


def _build_problem(X, Y, target, groups, delta, gamma_g, gamma_G, mode):
    """Build LP. mode in {'gbu','bs','local'}; returns c,A_ub,b_ub,A_eq,b_eq,bounds and lambda indices."""
    n, m = X.shape; s = Y.shape[1]
    # Positive scaling of each data coordinate improves conditioning and leaves theta invariant.
    xs = np.max(X, axis=0); ys = np.max(Y, axis=0)
    Xs = X / xs; Ys = Y / ys
    xo = Xs[target]; yo = Ys[target]
    # All modes have theta and n intensities.
    idx_theta = 0
    idx_lam = np.arange(1, n+1)
    cursor = n + 1
    group_ids = list(dict.fromkeys(groups.tolist()))
    members = {g: np.flatnonzero(groups == g) for g in group_ids}
    aux = {}
    if mode == 'gbu':
        # per input: rho, v_g, pi_g, w_j
        for i in range(m):
            aux[i] = {'rho':cursor}; cursor += 1
            aux[i]['v'] = np.arange(cursor, cursor+len(group_ids)); cursor += len(group_ids)
            aux[i]['pi'] = np.arange(cursor, cursor+len(group_ids)); cursor += len(group_ids)
            aux[i]['w'] = np.arange(cursor, cursor+n); cursor += n
    elif mode == 'bs':
        # Product-calibrated BS budget = Gamma_local * Gamma_G = Gamma_G.
        for i in range(m):
            aux[i] = {'rho':cursor}; cursor += 1
            aux[i]['w'] = np.arange(cursor, cursor+n); cursor += n
    elif mode == 'local':
        # No global activation cap: each group's local support is fully activated.
        for i in range(m):
            aux[i] = {'pi':np.arange(cursor, cursor+len(group_ids))}; cursor += len(group_ids)
            aux[i]['w'] = np.arange(cursor, cursor+n); cursor += n
    else:
        raise ValueError(mode)
    N = cursor
    c = np.zeros(N); c[idx_theta] = 1.0
    rows=[]; rhs=[]
    def add(coeff, b):
        rows.append(coeff); rhs.append(b)
    # Robust input envelope inequalities.
    for i in range(m):
        a = np.zeros(N); a[idx_lam] = Xs[:,i]; a[idx_theta] = -xo[i]
        if mode == 'gbu':
            q=aux[i]; a[q['rho']] += gamma_G
            a[q['v']] += 1.0
        elif mode == 'bs':
            q=aux[i]; a[q['rho']] += gamma_G; a[q['w']] += 1.0
        else:
            q=aux[i]
            for gi,g in enumerate(group_ids):
                a[q['pi'][gi]] += LOCAL_BUDGET
                a[q['w'][members[g]]] += 1.0
        add(a, 0.0)
        if mode == 'gbu':
            q=aux[i]
            for gi,g in enumerate(group_ids):
                a=np.zeros(N); a[q['pi'][gi]]=LOCAL_BUDGET
                a[q['w'][members[g]]] = 1.0
                a[q['rho']] -= 1.0; a[q['v'][gi]] -= 1.0
                add(a,0.0)
            for gi,g in enumerate(group_ids):
                for j in members[g]:
                    a=np.zeros(N); a[idx_lam[j]]=delta*Xs[j,i]
                    a[q['pi'][gi]] -= 1.0; a[q['w'][j]] -= 1.0
                    add(a,0.0)
        elif mode == 'bs':
            q=aux[i]
            for j in range(n):
                a=np.zeros(N); a[idx_lam[j]]=delta*Xs[j,i]
                a[q['rho']] -= 1.0; a[q['w'][j]] -= 1.0
                add(a,0.0)
        else: # local-only
            q=aux[i]
            for gi,g in enumerate(group_ids):
                for j in members[g]:
                    a=np.zeros(N); a[idx_lam[j]]=delta*Xs[j,i]
                    a[q['pi'][gi]] -= 1.0; a[q['w'][j]] -= 1.0
                    add(a,0.0)
    # Certain output constraints: -Y'lambda <= -y_target.
    for r in range(s):
        a=np.zeros(N); a[idx_lam] = -Ys[:,r]
        add(a, -yo[r])
    A_ub=np.vstack(rows); b_ub=np.asarray(rhs)
    A_eq=np.zeros((1,N)); A_eq[0,idx_lam]=1.0
    b_eq=np.array([1.0])
    bounds=[(0,None)]*N
    return c,A_ub,b_ub,A_eq,b_eq,bounds,idx_lam,aux,group_ids,Xs


def solve_one(X,Y,target,groups,delta,gamma_G,mode):
    c,A,b,Aeq,beq,bounds,idx_lam,aux,gids,Xs = _build_problem(
        X,Y,target,groups,delta,LOCAL_BUDGET,gamma_G,mode)
    res=linprog(c,A_ub=csr_matrix(A),b_ub=b,A_eq=csr_matrix(Aeq),b_eq=beq,
                bounds=bounds,method=SOLVER_METHOD,
                options={'presolve':True})
    if not res.success:
        # Robust retry with standard HiGHS selection.
        res=linprog(c,A_ub=csr_matrix(A),b_ub=b,A_eq=csr_matrix(Aeq),b_eq=beq,
                    bounds=bounds,method='highs',options={'presolve':False})
    if not res.success:
        raise RuntimeError(f'{mode} LP failed; target={target}; delta={delta}; GammaG={gamma_G}; {res.message}')
    lam=res.x[idx_lam]
    return float(res.x[0]),lam


def equality_check(a, groups, p, tol=1e-8):
    """Prop. 6: compare top-p individual sum with top-p distinct-group maxima."""
    if p == 0: return True, 0.0, 0.0
    vals=np.asarray(a,dtype=float)
    flat=float(np.sort(vals)[-p:].sum())
    gmax=[]
    for g in dict.fromkeys(groups.tolist()):
        gmax.append(float(np.max(vals[groups==g])))
    distinct=float(np.sort(gmax)[-p:].sum())
    return abs(flat-distinct) <= tol*max(1.0,abs(flat)), flat, distinct


def main():
    t0=time.time(); OUTPUT_DIR.mkdir(parents=True,exist_ok=True)
    df=load_data()
    Xall=df[INPUT_COLS].to_numpy(float); Yall=df[OUTPUT_COLS].to_numpy(float)
    # Stable unique bank identifiers, preserving source row order.
    bank_ids=df[ID_COL].astype(str).to_numpy()
    all_rows=[]; eq_rows=[]; completed=set()
    checkpoint=OUTPUT_DIR/'scenario_bank_level_checkpoint.csv'
    if checkpoint.exists():
        old=pd.read_csv(checkpoint)
        all_rows=old.to_dict('records')
        completed={(r['frontier'],int(r['target_index']),float(r['delta']),int(r['Gamma_G']),r['model']) for r in all_rows}
        print(f'Resuming from {len(completed)} completed target-model-scenario records.')
    eq_checkpoint=OUTPUT_DIR/'proposition6_equality_checkpoint.csv'
    if eq_checkpoint.exists(): eq_rows=pd.read_csv(eq_checkpoint).to_dict('records')
    frontiers=[]
    for reg in REGION_ORDER:
        ix=np.flatnonzero(df.REGION.to_numpy()==reg)
        frontiers.append((reg,ix,df.loc[ix,'COUNTRY'].astype(str).to_numpy()))
    # Pooled metafrontier: countries remain separate across all regions.
    frontiers.append(('Pooled metafrontier',np.arange(len(df)),df[COUNTRY_COL].astype(str).to_numpy()))
    total=0
    for fname,ix,group_labels in frontiers:
        X=Xall[ix]; Y=Yall[ix]; gids=np.asarray(group_labels,dtype=object)
        # Ensure every country is represented as its own group within this frontier.
        for delta in DELTAS:
            for G in GLOBAL_BUDGETS:
                for local_t,global_t in enumerate(ix):
                    for model in ('gbu','bs','local'):
                        key=(fname,int(global_t),float(delta),int(G),model)
                        if key in completed: continue
                        score,lam=solve_one(X,Y,local_t,gids,delta,G,model)
                        all_rows.append({'frontier':fname,'target_index':int(global_t),
                            'bank':bank_ids[global_t],'country':df.iloc[global_t][COUNTRY_COL],
                            'delta':delta,'Gamma_G':G,'Gamma_g':LOCAL_BUDGET,
                            'model':model,'score':score})
                        completed.add(key); total+=1
                        # Proposition 6 applies to GBU coefficient vectors, Gamma_g=1, integer p=G.
                        if model=='gbu':
                            for i in range(X.shape[1]):
                                a=delta*(X[:,i]/np.max(X[:,i]))*lam
                                ok,flat,distinct=equality_check(a,gids,G)
                                eq_rows.append({'frontier':fname,'target_index':int(global_t),
                                    'bank':bank_ids[global_t],'country':df.iloc[global_t][COUNTRY_COL],
                                    'input':INPUT_COLS[i].strip(),'delta':delta,'p_Gamma_G':G,
                                    'equality_holds':int(ok),'BS_top_p_support':flat,
                                    'GBU_top_p_distinct_group_support':distinct,
                                    'support_gap_BS_minus_GBU':flat-distinct})
                    if total and total % CHECKPOINT_EVERY_TARGETS == 0:
                        pd.DataFrame(all_rows).to_csv(checkpoint,index=False)
                        pd.DataFrame(eq_rows).to_csv(eq_checkpoint,index=False)
                pd.DataFrame(all_rows).to_csv(checkpoint,index=False)
                pd.DataFrame(eq_rows).to_csv(eq_checkpoint,index=False)
                print(f'Completed {fname}: delta={delta:.3%}, Gamma_G={G}; records={len(all_rows)}')
    bank=pd.DataFrame(all_rows)
    bank.to_csv(OUTPUT_DIR/'scenario_bank_level_scores.csv',index=False)
    eq=pd.DataFrame(eq_rows).drop_duplicates()
    eq.to_csv(OUTPUT_DIR/'proposition6_equality_bank_input.csv',index=False)
    # Aggregate means by frontier, delta, global budget, and model.
    summary=(bank.groupby(['frontier','delta','Gamma_G','model'],as_index=False)
             .agg(mean_score=('score','mean'),median_score=('score','median'),
                  sd_score=('score','std'),min_score=('score','min'),max_score=('score','max'),
                  n_banks=('score','size')))
    summary.to_csv(OUTPUT_DIR/'scenario_summary.csv',index=False)
    # Paired GBU-vs-BS and GBU-vs-local differences at bank level.
    wide=bank.pivot_table(index=['frontier','target_index','bank','delta','Gamma_G'],
                          columns='model',values='score').reset_index()
    wide['GBU_minus_BS']=wide['gbu']-wide['bs']
    wide['GBU_minus_local_only']=wide['gbu']-wide['local']
    wide.to_csv(OUTPUT_DIR/'paired_bank_level_comparisons.csv',index=False)
    diffs=(wide.groupby(['frontier','delta','Gamma_G'],as_index=False)
           .agg(mean_GBU_minus_BS=('GBU_minus_BS','mean'),
                median_GBU_minus_BS=('GBU_minus_BS','median'),
                mean_GBU_minus_local=('GBU_minus_local_only','mean'),
                median_GBU_minus_local=('GBU_minus_local_only','median'),
                max_abs_GBU_minus_local=('GBU_minus_local_only',lambda x:np.max(np.abs(x))),
                n_banks=('bank','size')))
    diffs.to_csv(OUTPUT_DIR/'comparison_summary.csv',index=False)
    # Bank-level monotonicity: scores should be nondecreasing in delta and Gamma_G.
    mono=[]
    for (frontier,target,bank,model),g in bank.groupby(['frontier','target_index','bank','model']):
        piv=g.pivot(index='delta',columns='Gamma_G',values='score')
        for G in GLOBAL_BUDGETS:
            vals=piv[G].sort_index().to_numpy()
            mono.append({'frontier':frontier,'target_index':target,'bank':bank,'model':model,
                         'direction':'delta','fixed_Gamma_G':G,
                         'monotone_non_decreasing':int(np.all(np.diff(vals)>=-1e-7)),
                         'max_violation':float(max(0.0,-np.min(np.diff(vals)))) if len(vals)>1 else 0.0})
        for d in DELTAS:
            vals=piv.loc[d,list(GLOBAL_BUDGETS)].to_numpy()
            mono.append({'frontier':frontier,'target_index':target,'bank':bank,'model':model,
                         'direction':'Gamma_G','fixed_delta':d,
                         'monotone_non_decreasing':int(np.all(np.diff(vals)>=-1e-7)),
                         'max_violation':float(max(0.0,-np.min(np.diff(vals)))) if len(vals)>1 else 0.0})
    mono_df=pd.DataFrame(mono); mono_df.to_csv(OUTPUT_DIR/'bank_level_monotonicity.csv',index=False)
    mono_summary=(mono_df.groupby(['frontier','model','direction'],as_index=False)
                  .agg(checks=('monotone_non_decreasing','size'),
                       passed=('monotone_non_decreasing','sum'),
                       max_violation=('max_violation','max')))
    mono_summary.to_csv(OUTPUT_DIR/'monotonicity_summary.csv',index=False)
    # Proposition 6 equality frequency by frontier, p, delta and input.
    eq_summary=(eq.groupby(['frontier','p_Gamma_G','delta','input'],as_index=False)
                .agg(n=('equality_holds','size'),equal_cases=('equality_holds','sum'),
                     equality_frequency=('equality_holds','mean'),
                     mean_support_gap=('support_gap_BS_minus_GBU','mean'),
                     max_support_gap=('support_gap_BS_minus_GBU','max')))
    eq_summary.to_csv(OUTPUT_DIR/'proposition6_equality_summary.csv',index=False)
    config={'input_file':str(INPUT_FILE),'n_banks':len(df),'regional_counts':df.REGION.value_counts().to_dict(),
            'inputs':INPUT_COLS,'outputs':OUTPUT_COLS,'deltas':DELTAS,'Gamma_G':GLOBAL_BUDGETS,
            'Gamma_g':LOCAL_BUDGET,'BS_budget':'Gamma_local * Gamma_G = Gamma_G',
            'local_only':'all country groups fully activated; within-country Gamma_g=1 retained',
            'solver':SOLVER_METHOD,'elapsed_seconds':time.time()-t0}
    (OUTPUT_DIR/'run_configuration.json').write_text(json.dumps(config,indent=2,default=str))
    print('\nFINISHED')
    print(f'Output folder: {OUTPUT_DIR}')
    print(f'Bank-level score records: {len(bank)}')
    print(f'Proposition 6 records: {len(eq)}')
    print(f'Elapsed hours: {(time.time()-t0)/3600:.2f}')
    print('Download the entire output folder as a ZIP after successful completion.')

if __name__=='__main__':
    main()
