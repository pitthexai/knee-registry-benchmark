from __future__ import annotations

import argparse, joblib
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from orthobench.metrics import binary_metrics
from orthobench.triage import choose_t_accept_wilson, choose_t_reject, triage_report


def load_npz(p):
    d=np.load(p, allow_pickle=True); return d["X"], d["y"].astype(int)

def make_clf(name, seed):
    if name=="logreg": return LogisticRegression(max_iter=3000, class_weight="balanced", n_jobs=-1, random_state=seed)
    if name=="hgb": return HistGradientBoostingClassifier(max_depth=6, learning_rate=0.05, max_iter=400, random_state=seed)
    if name=="rf": return RandomForestClassifier(n_estimators=500, class_weight="balanced", n_jobs=-1, random_state=seed)
    raise ValueError(name)

def probs(clf,X):
    if hasattr(clf,"predict_proba"): return clf.predict_proba(X)[:,1]
    z=clf.decision_function(X); return 1/(1+np.exp(-z))

def sample_disjoint_balanced(y, n_pos_inj, n_neg_inj, n_pos_tune, n_neg_tune, seed):
    rng=np.random.default_rng(seed)
    pos=rng.permutation(np.where(y==1)[0]); neg=rng.permutation(np.where(y==0)[0])
    if len(pos)<n_pos_inj+n_pos_tune or len(neg)<n_neg_inj+n_neg_tune:
        raise ValueError("Not enough FracAtlas pos/neg for requested inject+tune sizes")
    inj=np.concatenate([pos[:n_pos_inj], neg[:n_neg_inj]])
    tune=np.concatenate([pos[n_pos_inj:n_pos_inj+n_pos_tune], neg[n_neg_inj:n_neg_inj+n_neg_tune]])
    used=np.zeros(len(y), dtype=bool); used[inj]=True; used[tune]=True
    test=np.where(~used)[0]
    rng.shuffle(inj); rng.shuffle(tune)
    return inj,tune,test

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--emb_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--feature_models", nargs="+", default=None)
    ap.add_argument("--domain_models", nargs="+", default=["logreg","hgb","rf"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0,1,2])
    ap.add_argument("--inj_pos", type=int, default=200); ap.add_argument("--inj_neg", type=int, default=200)
    ap.add_argument("--tune_pos", type=int, default=200); ap.add_argument("--tune_neg", type=int, default=200)
    ap.add_argument("--target_precision", type=float, default=0.90)
    ap.add_argument("--max_false_reject", type=float, default=0.02)
    ap.add_argument("--min_accept", type=int, default=25)
    ap.add_argument("--calibrate_seen_calib", action="store_true")
    ap.add_argument("--calibrate_seen_val", action="store_true", help="Deprecated alias")
    args=ap.parse_args()
    out_dir=Path(args.out_dir); (out_dir/"models").mkdir(parents=True, exist_ok=True); (out_dir/"results").mkdir(parents=True, exist_ok=True)
    feature_models = args.feature_models or sorted(
        p.name[len("emb_"):-len("_train.npz")] for p in Path(args.emb_dir).glob("emb_*_train.npz")
    )
    if not feature_models:
        raise SystemExit("No training embedding files found")
    rows=[]
    for feat in feature_models:
        Xtr,ytr=load_npz(Path(args.emb_dir)/f"emb_{feat}_train.npz")
        Xva,yva=load_npz(Path(args.emb_dir)/f"emb_{feat}_val.npz")
        Xcal,ycal=load_npz(Path(args.emb_dir)/f"emb_{feat}_calib.npz")
        Xte,yte=load_npz(Path(args.emb_dir)/f"emb_{feat}_test.npz")
        Xfa,yfa=load_npz(Path(args.emb_dir)/f"emb_{feat}_fracatlas.npz")
        for dm in args.domain_models:
            for seed in args.seeds:
                inj,tune,test=sample_disjoint_balanced(yfa,args.inj_pos,args.inj_neg,args.tune_pos,args.tune_neg,seed)
                Xinj,yinj=Xfa[inj],yfa[inj]; Xtune,ytune=Xfa[tune],yfa[tune]; Xfat,yfat=Xfa[test],yfa[test]
                # setting 1: baseline no injection, tune thresholds on target tune
                for setting in ["baseline_no_injection", "fewshot_injection"]:
                    clf=make_clf(dm, seed)
                    if setting=="fewshot_injection":
                        clf.fit(np.vstack([Xtr,Xinj]), np.concatenate([ytr,yinj]))
                    else:
                        clf.fit(Xtr,ytr)
                    if args.calibrate_seen_calib or args.calibrate_seen_val:
                        clf=CalibratedClassifierCV(clf, method="sigmoid", cv="prefit")
                        clf.fit(Xcal,ycal)
                    p_tune=probs(clf,Xtune); p_test=probs(clf,Xfat)
                    tacc=choose_t_accept_wilson(p_tune,ytune,args.target_precision,args.min_accept)
                    trej=choose_t_reject(p_tune,ytune,args.max_false_reject) or {"t_reject":0.0}
                    t_accept=tacc["t_accept"] if tacc else 1.0
                    t_reject=min(trej["t_reject"], t_accept-1e-6)
                    pred=clf.predict(Xfat)
                    row={"feature_model":feat,"domain_model":dm,"seed":seed,"setting":setting,"n_inject":len(inj) if setting=="fewshot_injection" else 0,"n_tune":len(tune),"t_accept":t_accept,"t_reject":t_reject}
                    row.update(binary_metrics(yfat,pred,p_test))
                    row.update({"triage_"+k:v for k,v in triage_report(p_test,yfat,t_reject,t_accept).items()})
                    rows.append(row); print(row)
                    joblib.dump(clf, out_dir/"models"/f"domain_{feat}_{dm}_{setting}_seed{seed}.joblib")
    res=pd.DataFrame(rows)
    res.to_csv(out_dir/"results"/"domain_fracatlas_maintenance.csv", index=False)
    summary=res.groupby(["feature_model","domain_model","setting"]).agg(['mean','std'])
    summary.to_csv(out_dir/"results"/"domain_fracatlas_maintenance_summary.csv")

if __name__=="__main__":
    main()
