import argparse,json,os,glob
import numpy as np

def load_all(out_dir):
    rows=[]
    for f in sorted(glob.glob(os.path.join(out_dir,"sigma_diag_*.json"))):
        d=json.load(open(f))
        for r in d["rows"]:
            r2=dict(r)
            r2["model"]=d["model"]
            r2["layer"]=d["layer"]
            r2["d"]=d["d"]
            r2["F_diag_sum"]=d["F_diag_sum"]
            rows.append(r2)
    return rows

def by_modellayer(rows):
    g={}
    for r in rows:
        k=(r["model"],r["layer"])
        g.setdefault(k,[]).append(r)
    return g

def alpha_curve(rows_ml,by="worst_mahal"):
    by_a={}
    for r in rows_ml:
        by_a.setdefault(r["alpha"],[]).append(r[by])
    return {a:(np.mean(v),np.std(v)) for a,v in by_a.items()}

def best_alpha(curve):
    return min(curve,key=lambda a:curve[a][0])

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--in_dir",default="artifacts/sigma_diag_validate")
    ap.add_argument("--out",default="artifacts/sigma_diag_validate/summary.json")
    a=ap.parse_args()
    rows=load_all(a.in_dir)
    g=by_modellayer(rows)
    summary=[]
    for(mdl,L),rows_ml in sorted(g.items()):
        cw=alpha_curve(rows_ml,by="worst_mahal")
        ct=alpha_curve(rows_ml,by="retrieval_top1")
        rec=dict(model=mdl,layer=L,d=rows_ml[0]["d"],F_diag_sum=rows_ml[0]["F_diag_sum"],
                 worst_curve={f"{a:.2f}":[float(m),float(s)] for a,(m,s) in cw.items()},
                 top1_curve={f"{a:.2f}":[float(m),float(s)] for a,(m,s) in ct.items()},
                 best_alpha_worst=best_alpha(cw),best_alpha_top1=best_alpha(ct))
        summary.append(rec)
        bw=rec["best_alpha_worst"]
        bt=rec["best_alpha_top1"]
        print(f"{mdl.split('/')[-1][:14]:<14} L{L:>2}  α*_worst={bw:.2f}  α*_top1={bt:.2f}  worst@α=1={cw[1.0][0]:.2f}±{cw[1.0][1]:.2f}  top1@α=1={ct[1.0][0]:.3f}±{ct[1.0][1]:.3f}")
    n_total=len(summary)
    n_worst1=sum(1 for r in summary if abs(r["best_alpha_worst"]-1.0)<1e-9)
    n_top11=sum(1 for r in summary if abs(r["best_alpha_top1"]-1.0)<1e-9)
    print(f"\nFraction of model-layers where α=1 wins worst-case: {n_worst1}/{n_total}")
    print(f"Fraction of model-layers where α=1 wins retrieval:  {n_top11}/{n_total}")
    n_worst_ge_05=sum(1 for r in summary if r["best_alpha_worst"]>=0.5)
    print(f"Fraction with best α* >= 0.5 (high-α regime):       {n_worst_ge_05}/{n_total}")
    with open(a.out,"w") as f:
        json.dump(dict(summary=summary,n_worst_alpha1=n_worst1,n_top1_alpha1=n_top11,n_total=n_total,n_high_alpha=n_worst_ge_05),f,indent=2)
    print(f"\nwrote {a.out}")

if __name__=="__main__":
    main()
