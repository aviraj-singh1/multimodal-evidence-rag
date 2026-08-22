import json,tempfile,unittest
from unittest.mock import patch
from pathlib import Path
from fastapi.testclient import TestClient
from backend.app.api.query import ROOT
from backend.app.evaluation.runner import load_suite
from backend.app.evidence.repository import EvidenceRepository
from backend.app.main import app

class EvaluationRunnerTests(unittest.TestCase):
 def test_config_ground_truth_exports_services_and_api(self):
  suite=load_suite(ROOT/'config'/'evaluation_questions.json'); valid={n.id for n in EvidenceRepository(ROOT/'data'/'app.db').list_evidence_nodes()}
  self.assertGreaterEqual(len(suite.questions),8); self.assertTrue(all(q.required_groups for q in suite.questions)); self.assertTrue(all(set(g.acceptable_evidence_ids)<=valid for q in suite.questions for g in q.required_groups))
  self.assertTrue((ROOT/'data/evaluation/results_v1.json').is_file()); self.assertTrue((ROOT/'data/evaluation/results_v1.csv').is_file())
  client=TestClient(app); self.assertEqual(client.post('/api/baseline/query',json={'question':'HTTP 429','top_k':5}).status_code,200)
  with patch('backend.app.api.evaluation.run_evaluation',return_value={'status':'ok'}): self.assertEqual(client.post('/api/evaluate').status_code,200)
