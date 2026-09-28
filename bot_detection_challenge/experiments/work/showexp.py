import sys,json
n=int(sys.argv[1]) if len(sys.argv)>1 else 20
L=open('work/experiments.jsonl').read().splitlines()[-n:]
for l in L:
    r=json.loads(l); print(f"{r['exp']:32s} {r.get('scheme','dayblock'):9s} P@R={r['p_at_r70']:.4f} fold={r['fold_mean']:.4f}±{r['fold_std']:.4f} min={r['fold_min']:.3f} PR={r['pr_auc']:.4f} ROC={r['roc_auc']:.4f} {r['sec']}s")
