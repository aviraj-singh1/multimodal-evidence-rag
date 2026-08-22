"""Run and export the fair baseline-versus-EvidenceGraph evaluation."""

import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from backend.app.evaluation.runner import run_evaluation

def main()->int:
    print('FAIRNESS\nBaseline model: MiniLM\nEvidenceGraph text model: MiniLM\nTop K: 5 vs 5\nRaw text inputs: same speech/OCR/PDF\nEvidenceGraph extras: CLIP, entities, graph, temporal edges, expansion, multimodal reranking\n')
    result=run_evaluation(ROOT,export=True)
    print('Question | Baseline R@5 | Graph R@5 | Baseline Complete | Graph Complete | Baseline Prov | Graph Prov')
    for row in result['results']:
        b,g=row['baseline'],row['evidence_graph']; print(f"{row['question_id']} | {b['recall_at_5']:.3f} | {g['recall_at_5']:.3f} | {int(b['complete'])} | {int(g['complete'])} | {b['provenance_completeness']:.3f} | {g['provenance_completeness']:.3f}")
    for name in ('baseline','evidence_graph','delta'): print(name,result['summary'][name])
    return 0
if __name__=='__main__': raise SystemExit(main())
