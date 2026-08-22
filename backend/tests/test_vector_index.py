"""Persistent Chroma collection and vector-reference tests."""

import unittest

from backend.app.api.query import ROOT, load_nodes
from backend.app.vector.chroma import ChromaStore, TEXT_COLLECTION, VISUAL_COLLECTION
from backend.app.vector.embeddings import CLIP_DIMENSION, MINILM_DIMENSION


class VectorIndexTests(unittest.TestCase):
    def test_collections_modalities_dimensions_and_references(self) -> None:
        store = ChromaStore(ROOT / "data" / "chroma")
        text = store.collection(TEXT_COLLECTION)
        visual = store.collection(VISUAL_COLLECTION)
        self.assertEqual(text.count(), 316)
        self.assertEqual(visual.count(), 17)
        self.assertEqual(len(text.get(limit=1, include=["embeddings"])["embeddings"][0]), MINILM_DIMENSION)
        self.assertEqual(len(visual.get(limit=1, include=["embeddings"])["embeddings"][0]), CLIP_DIMENSION)
        nodes = load_nodes()
        self.assertTrue(all(node.text_vector_id == f"text:{node.id}" for node in nodes
                            if node.modality in {"speech", "ocr_text", "pdf_text"}))
        self.assertTrue(all(node.visual_vector_id == f"visual:{node.id}" for node in nodes
                            if node.modality in {"video_frame", "image"}))
        self.assertEqual(len({node.text_vector_id for node in nodes if node.text_vector_id}), 316)
        self.assertEqual(len({node.visual_vector_id for node in nodes if node.visual_vector_id}), 17)
