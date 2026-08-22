"""Run both normal services on one manually grounded suite and export real results."""

from __future__ import annotations

import csv, json
from datetime import datetime, timezone
from pathlib import Path

from backend.app.api.query import QueryService
from backend.app.baseline.service import BaselineService
from backend.app.evaluation.metrics import baseline_provenance, graph_provenance, group_metrics
from backend.app.evaluation.models import EvaluationSuite
from backend.app.evidence.repository import EvidenceRepository


def load_suite(path: Path) -> EvaluationSuite:
    return EvaluationSuite(**json.loads(path.read_text(encoding="utf-8")))


def run_evaluation(root: Path, export: bool = True, output_stem: str = "results_latest") -> dict[str, object]:
    suite=load_suite(root/'config'/'evaluation_questions.json'); nodes=EvidenceRepository(root/'data'/'app.db').list_evidence_nodes(); by_id={n.id:n for n in nodes}
    valid=set(by_id)
    for q in suite.questions:
        for group in q.required_groups:
            missing=set(group.acceptable_evidence_ids)-valid
            if missing: raise ValueError(f"unknown ground-truth IDs for {q.id}: {sorted(missing)}")
    baseline=BaselineService(root); graph=QueryService(root); rows=[]
    for q in suite.questions:
        b=baseline.query(q.question,5); g=graph.query(q.question,5,1); gtop=g['evidence_bundle'][:5]
        bsets=[set(item['member_evidence_ids']) for item in b['results'][:5]]; gsets=[{item['node']['id']} for item in gtop]
        bh,br,bc=group_metrics(q,bsets); gh,gr,gc=group_metrics(q,gsets)
        acceptable=set(x for group in q.required_groups for x in group.acceptable_evidence_ids)
        bp=[baseline_provenance(item,by_id[eid]) for item in b['results'][:5] for eid in item['member_evidence_ids'] if eid in acceptable]
        gp=[graph_provenance(item,by_id[item['node']['id']]) for item in gtop if item['node']['id'] in acceptable]
        rows.append({"question_id":q.id,"question":q.question,"required_groups":[x.model_dump() for x in q.required_groups],
                     "baseline":{"top_results":b['results'][:5],"group_hits":bh,"recall_at_5":br,"complete":bc,"provenance_completeness":sum(bp)/len(bp) if bp else 0.0,"latency_ms":b['latency_ms']},
                     "evidence_graph":{"top_results":gtop,"group_hits":gh,"recall_at_5":gr,"complete":gc,"provenance_completeness":sum(gp)/len(gp) if gp else 0.0,"latency_ms":g['latency_ms']}})
    def summary(system):
        return {"evidence_recall_at_5":sum(r[system]['recall_at_5'] for r in rows)/len(rows),
                "complete_evidence_rate":sum(r[system]['complete'] for r in rows)/len(rows),
                "provenance_completeness":sum(r[system]['provenance_completeness'] for r in rows)/len(rows),
                "average_latency_ms":sum(r[system]['latency_ms'] for r in rows)/len(rows)}
    bs,gs=summary('baseline'),summary('evidence_graph'); result={"timestamp":datetime.now(timezone.utc).isoformat(),"evaluation_version":suite.version,
      "retrieval_configuration":{"top_k":5,"baseline_model":"sentence-transformers/all-MiniLM-L6-v2","graph_hops":1},"question_definitions":[q.model_dump() for q in suite.questions],"results":rows,
      "summary":{"baseline":bs,"evidence_graph":gs,"delta":{k:gs[k]-bs[k] for k in bs}}}
    if export: _export(root,result,output_stem)
    return result


def _export(root:Path,result:dict[str,object],stem:str="results_latest")->None:
    out=root/'data'/'evaluation'; out.mkdir(parents=True,exist_ok=True)
    (out/f'{stem}.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    with (out/f'{stem}.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=['question_id','question','system','recall_at_5','complete_evidence','provenance_completeness','latency_ms']); writer.writeheader()
        for row in result['results']:
            for system in ('baseline','evidence_graph'):
                x=row[system]; writer.writerow({'question_id':row['question_id'],'question':row['question'],'system':system,'recall_at_5':x['recall_at_5'],'complete_evidence':int(x['complete']),'provenance_completeness':x['provenance_completeness'],'latency_ms':x['latency_ms']})
