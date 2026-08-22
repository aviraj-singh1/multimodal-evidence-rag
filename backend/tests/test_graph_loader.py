"""Persisted edge consistency, NetworkX loading, and rollback tests."""

from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from backend.app.db.sqlite import EvidenceStore
from backend.app.evidence.entities import enrich_entities, load_entity_config
from backend.app.evidence.relationships import build_relationships
from backend.app.graph.loader import load_graph
from test_entities import CACHE, IDS, ROOT, load_nodes


class GraphLoaderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        config = load_entity_config(ROOT / "config")
        cls.nodes = enrich_entities(load_nodes(), config)
        cls.edges, _ = build_relationships(cls.nodes, config.common_entity_threshold)

    def test_actual_sqlite_edges_cache_and_networkx_agree(self) -> None:
        store = EvidenceStore(ROOT / "data" / "app.db")
        payload = json.loads((CACHE / "graph" / "edges.json").read_text(encoding="utf-8"))
        graph = load_graph(self.nodes, store.load_edges())
        self.assertEqual(store.edge_count(), payload["edge_count"])
        self.assertEqual(graph.number_of_nodes(), len(self.nodes))
        self.assertEqual(graph.number_of_edges(), store.edge_count())
        self.assertEqual(store.foreign_key_violations(), [])
        for source_id in IDS:
            evidence = json.loads((CACHE / source_id / "evidence.json").read_text(encoding="utf-8"))
            self.assertEqual(evidence["evidence_count"], len(evidence["nodes"]))

    def test_rebuild_idempotency_and_failed_transaction_rollback(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "app.db"
            original = EvidenceStore(ROOT / "data" / "app.db")
            database.write_bytes((ROOT / "data" / "app.db").read_bytes())
            store = EvidenceStore(database)
            store.replace_graph(self.nodes, self.edges)
            ids = store.edge_ids()
            store.replace_graph(self.nodes, self.edges)
            self.assertEqual(ids, store.edge_ids())
            with self.assertRaises(sqlite3.IntegrityError):
                store.replace_graph(self.nodes, [self.edges[0], self.edges[0]])
            self.assertEqual(ids, store.edge_ids())
