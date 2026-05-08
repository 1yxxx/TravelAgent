from __future__ import annotations

import json
from pathlib import Path
from unittest import TestCase

from scripts.eval_layer_metrics import evaluate_layer_metrics


class TestEvalLayerMetrics(TestCase):
    def _write_metric(self, session_dir: Path, artifact_id: str, created_at: float, metric: dict) -> None:
        lm_dir = session_dir / "layer_metrics"
        lm_dir.mkdir(parents=True, exist_ok=True)
        payload_path = lm_dir / f"{artifact_id}.json"
        payload_path.write_text(
            json.dumps(
                {
                    "artifact_id": artifact_id,
                    "result": {
                        "session_id": session_dir.name,
                        "layer_metrics": metric,
                    },
                    "payload": {
                        "session_id": session_dir.name,
                        "layer_metrics": metric,
                    },
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        meta = [
            {
                "session_id": session_dir.name,
                "artifact_id": artifact_id,
                "node_id": "layer_metrics",
                "path": str(payload_path),
                "summary": "metrics",
                "created_at": created_at,
            }
        ]
        (session_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")

    def test_aggregate_layer_metrics_from_outputs(self):
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            s1 = root / "travel_a"
            s2 = root / "travel_b"
            s1.mkdir(parents=True, exist_ok=True)
            s2.mkdir(parents=True, exist_ok=True)

            self._write_metric(
                s1,
                "layer_metrics_a",
                100.0,
                {
                    "attempted_layers": 2,
                    "successful_layers": 2,
                    "layer_hit_rate": 1.0,
                    "total_attempts": 3,
                    "rollback_count": 1,
                    "rollback_rate": 1 / 3,
                    "per_layer_attempts": {"research": 2, "planning": 1},
                    "per_layer_success": {"research": True, "planning": True},
                },
            )

            self._write_metric(
                s2,
                "layer_metrics_b",
                200.0,
                {
                    "attempted_layers": 2,
                    "successful_layers": 1,
                    "layer_hit_rate": 0.5,
                    "total_attempts": 2,
                    "rollback_count": 1,
                    "rollback_rate": 0.5,
                    "per_layer_attempts": {"research": 1, "planning": 1},
                    "per_layer_success": {"research": True, "planning": False},
                },
            )

            summary = evaluate_layer_metrics(root)

            self.assertEqual(summary["session_count"], 2)
            self.assertAlmostEqual(summary["overall"]["avg_hit_rate"], 0.75)
            self.assertEqual(summary["overall"]["total_attempts"], 5)
            self.assertEqual(summary["overall"]["total_rollbacks"], 2)

            top_layers = summary["top_failing_layers"]
            self.assertTrue(any(x["layer"] == "planning" and x["failures"] >= 1 for x in top_layers))

    def test_ignore_sessions_without_metrics(self):
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "travel_empty").mkdir(parents=True, exist_ok=True)

            summary = evaluate_layer_metrics(root)
            self.assertEqual(summary["session_count"], 0)
            self.assertEqual(summary["sessions"], [])
