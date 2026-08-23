"""SQLite and evidence cache consistency and transaction tests."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import tempfile
import unittest

from backend.app.cache.manager import read_json
from backend.app.db.sqlite import EvidenceStore
from backend.app.evidence.builder import build_evidence, discover_evidence_sources


REPO_ROOT = Path(__file__).resolve().parents[2]
CACHE_ROOT = REPO_ROOT / "data" / "cache"
DATASET_DIR = REPO_ROOT / "demo_test"


class EvidenceStorageTests(unittest.TestCase):
    def test_all_sources_store_with_valid_foreign_keys_and_idempotency(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = EvidenceStore(Path(directory) / "app.db")
            expected_total = 0
            for source_id in discover_evidence_sources(CACHE_ROOT):
                source, nodes = build_evidence(CACHE_ROOT / source_id, REPO_ROOT, DATASET_DIR)
                store.replace_source(source, nodes)
                expected_total += len(nodes)
                first_ids = store.evidence_ids(source_id)
                store.replace_source(source, nodes)
                self.assertEqual(store.evidence_ids(source_id), first_ids)
                cache = read_json(CACHE_ROOT / source_id / "evidence.json")
                self.assertEqual(cache["evidence_count"], store.evidence_count(source_id))
                self.assertEqual(store.evidence_count(source_id), len(nodes))
                self.assertEqual([node["id"] for node in cache["nodes"]], [node.id for node in nodes])
            self.assertEqual(store.source_count(), 4)
            self.assertEqual(store.evidence_count(), expected_total)
            self.assertEqual(store.foreign_key_violations(), [])

    def test_failed_rebuild_rolls_back_without_partial_records(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = EvidenceStore(Path(directory) / "app.db")
            source_id = "src_9899772d9cc4"
            source, nodes = build_evidence(CACHE_ROOT / source_id, REPO_ROOT, DATASET_DIR)
            store.replace_source(source, nodes)
            original_ids = store.evidence_ids(source_id)
            with self.assertRaises(sqlite3.IntegrityError):
                store.replace_source(source, [nodes[0], nodes[0]])
            self.assertEqual(store.evidence_ids(source_id), original_ids)
            self.assertEqual(store.evidence_count(source_id), len(nodes))
