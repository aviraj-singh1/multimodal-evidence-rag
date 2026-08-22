import unittest
from backend.app.api.baseline import get_baseline_service
from backend.app.api.query import ROOT
from backend.app.vector.chroma import ChromaStore

class BaselineSearchTests(unittest.TestCase):
 def test_collection_results_scores_and_idempotent_ids(self):
  service=get_baseline_service(); result=service.query('rules cache',5)
  self.assertTrue(result['results']); self.assertTrue(all(0<=x['score']<=1 and x['chunk_id'] in service.chunks for x in result['results']))
  store=ChromaStore(ROOT/'data'/'chroma'); self.assertEqual(store.collection('baseline_text').count(),len(service.chunks)); self.assertEqual(store.collection('text_evidence').count(),316)
