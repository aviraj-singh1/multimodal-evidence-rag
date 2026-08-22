import unittest
from backend.app.evaluation.metrics import group_metrics
from backend.app.evaluation.models import EvaluationQuestion,RequiredGroup

class EvaluationMetricTests(unittest.TestCase):
 def test_group_alternatives_recall_complete_and_chunk_mapping(self):
  q=EvaluationQuestion(id='Q',question='x',notes='x',required_groups=[RequiredGroup(name='a',required_modality_family='speech',acceptable_evidence_ids=['ev1','ev2']),RequiredGroup(name='b',required_modality_family='document',acceptable_evidence_ids=['ev3'])])
  hits,recall,complete=group_metrics(q,[{'ev2','other'}]); self.assertEqual(hits,{'a':True,'b':False}); self.assertEqual(recall,.5); self.assertFalse(complete)
  self.assertEqual(group_metrics(q,[{'ev2'},{'ev3'}])[1:],(1.0,True))
