import json,sys
for l in open('research2/experiments2.jsonl'):
    r=json.loads(l); print(f"{r['exp']:30s} rep={r['rep_mean']:.4f}±{r['rep_std']:.4f} [{r['rep_min']:.3f},{r['rep_max']:.3f}] fwd7={r['fwd7']:.4f} PR={r['prauc_P']:.4f} ROC={r['roc_P']:.4f} {r['sec']}s")
