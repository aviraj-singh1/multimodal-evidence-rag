import unittest
from backend.app.api.query import ROOT
from backend.app.baseline.builder import build_chunks
from backend.app.evidence.repository import EvidenceRepository

class BaselineBuilderTests(unittest.TestCase):
 def test_deterministic_text_only_overlap_members_and_raw_ocr(self):
  repo=EvidenceRepository(ROOT/'data'/'app.db'); nodes=repo.list_evidence_nodes(); a=build_chunks(nodes,repo.list_sources()); b=build_chunks(nodes,repo.list_sources())
  self.assertEqual(a,b); represented={x for c in a for x in c.member_evidence_ids}; by={n.id:n for n in nodes}
  self.assertTrue(all(by[x].modality in {'speech','ocr_text','pdf_text'} for x in represented))
  self.assertFalse(any(by[x].modality in {'video_frame','image'} for x in represented))
  self.assertTrue(any('AI Rate limiter' in c.text for c in a)); self.assertTrue(any('failure 529 HTTP code' in c.text for c in a))
  self.assertTrue(any(set(a[i].member_evidence_ids)&set(a[i+1].member_evidence_ids) for i in range(len(a)-1) if a[i].source_id==a[i+1].source_id))
