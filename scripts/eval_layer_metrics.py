from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


@dataclass
class SessionMetric:
    session_id: str
    artifact_id: str
    created_at: float
    layer_metrics: Dict[str, Any]


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _iter_session_dirs(outputs_dir: Path) -> Iterable[Path]:
    if not outputs_dir.exists():
        return []
    return [p for p in outputs_dir.iterdir() if p.is_dir()]


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _extract_metric_from_record(record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    payload = record.get("payload")
    if not isinstance(payload, dict):
        return None

    # 兼容两种结构：
    # 1) payload = {layer_metrics: {...}, layer_trace: [...]}（推荐）
    # 2) payload = {...直接是 metrics...}
    if isinstance(payload.get("layer_metrics"), dict):
        return payload["layer_metrics"]

    if "layer_hit_rate" in payload or "rollback_rate" in payload:
        return payload

    return None


def _load_latest_layer_metric(session_dir: Path) -> Optional[SessionMetric]:
    session_id = session_dir.name
    meta_path = session_dir / "meta.json"

    candidate_entries: List[Tuple[float, str, Path]] = []

    if meta_path.exists():
        try:
            meta = _read_json(meta_path)
            if isinstance(meta, list):
                for item in meta:
                    if not isinstance(item, dict):
                        continue
                    if item.get("node_id") != "layer_metrics":
                        continue
                    path_str = item.get("path")
                    if not isinstance(path_str, str):
                        continue
                    candidate_entries.append(
                        (
                            _safe_float(item.get("created_at")),
                            str(item.get("artifact_id") or ""),
                            Path(path_str),
                        )
                    )
        except Exception:
            pass

    # 兜底：直接扫描目录
    layer_metrics_dir = session_dir / "layer_metrics"
    if layer_metrics_dir.exists():
        for fp in layer_metrics_dir.glob("*.json"):
            candidate_entries.append((fp.stat().st_mtime, fp.stem, fp))

    if not candidate_entries:
        return None

    candidate_entries.sort(key=lambda x: x[0], reverse=True)

    for created_at, artifact_id, metric_path in candidate_entries:
        if not metric_path.exists():
            continue
        try:
            record = _read_json(metric_path)
            if not isinstance(record, dict):
                continue
            metric = _extract_metric_from_record(record)
            if metric is None:
                continue
            return SessionMetric(
                session_id=session_id,
                artifact_id=artifact_id,
                created_at=created_at,
                layer_metrics=metric,
            )
        except Exception:
            continue

    return None


def _quantile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    try:
        return statistics.quantiles(values, n=100)[max(0, min(99, int(q * 100) - 1))]
    except Exception:
        values_sorted = sorted(values)
        idx = int(round((len(values_sorted) - 1) * q))
        return values_sorted[idx]


def _aggregate(metrics: List[SessionMetric]) -> Dict[str, Any]:
    hit_rates = [_safe_float(m.layer_metrics.get("layer_hit_rate")) for m in metrics]
    rollback_rates = [_safe_float(m.layer_metrics.get("rollback_rate")) for m in metrics]

    total_attempts = sum(_safe_int(m.layer_metrics.get("total_attempts")) for m in metrics)
    total_rollbacks = sum(_safe_int(m.layer_metrics.get("rollback_count")) for m in metrics)

    per_layer_failures: Dict[str, int] = {}
    for m in metrics:
        attempts = m.layer_metrics.get("per_layer_attempts") or {}
        success = m.layer_metrics.get("per_layer_success") or {}
        if not isinstance(attempts, dict) or not isinstance(success, dict):
            continue
        for layer, cnt in attempts.items():
            cnt_i = _safe_int(cnt)
            ok = bool(success.get(layer, False))
            fail_count = cnt_i - 1 if ok else cnt_i
            per_layer_failures[layer] = per_layer_failures.get(layer, 0) + max(0, fail_count)

    top_failing_layers = [
        {"layer": layer, "failures": count}
        for layer, count in sorted(per_layer_failures.items(), key=lambda kv: kv[1], reverse=True)
        if count > 0
    ]

    return {
        "session_count": len(metrics),
        "overall": {
            "avg_hit_rate": statistics.fmean(hit_rates) if hit_rates else 0.0,
            "median_hit_rate": statistics.median(hit_rates) if hit_rates else 0.0,
            "p90_hit_rate": _quantile(hit_rates, 0.9),
            "avg_rollback_rate": statistics.fmean(rollback_rates) if rollback_rates else 0.0,
            "median_rollback_rate": statistics.median(rollback_rates) if rollback_rates else 0.0,
            "p90_rollback_rate": _quantile(rollback_rates, 0.9),
            "total_attempts": total_attempts,
            "total_rollbacks": total_rollbacks,
            "global_rollback_rate": (total_rollbacks / total_attempts) if total_attempts else 0.0,
        },
        "top_failing_layers": top_failing_layers,
        "sessions": [
            {
                "session_id": m.session_id,
                "artifact_id": m.artifact_id,
                "created_at": m.created_at,
                **m.layer_metrics,
            }
            for m in metrics
        ],
    }


def _write_csv(path: Path, metrics: List[SessionMetric]) -> None:
    fieldnames = [
        "session_id",
        "artifact_id",
        "created_at",
        "attempted_layers",
        "successful_layers",
        "layer_hit_rate",
        "total_attempts",
        "rollback_count",
        "rollback_rate",
    ]

    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for m in metrics:
            lm = m.layer_metrics
            writer.writerow(
                {
                    "session_id": m.session_id,
                    "artifact_id": m.artifact_id,
                    "created_at": m.created_at,
                    "attempted_layers": _safe_int(lm.get("attempted_layers")),
                    "successful_layers": _safe_int(lm.get("successful_layers")),
                    "layer_hit_rate": _safe_float(lm.get("layer_hit_rate")),
                    "total_attempts": _safe_int(lm.get("total_attempts")),
                    "rollback_count": _safe_int(lm.get("rollback_count")),
                    "rollback_rate": _safe_float(lm.get("rollback_rate")),
                }
            )


def evaluate_layer_metrics(outputs_dir: Path) -> Dict[str, Any]:
    metrics: List[SessionMetric] = []
    for session_dir in _iter_session_dirs(outputs_dir):
        metric = _load_latest_layer_metric(session_dir)
        if metric is not None:
            metrics.append(metric)

    metrics.sort(key=lambda m: m.created_at)
    return _aggregate(metrics)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate layered orchestration metrics across sessions.")
    parser.add_argument(
        "--outputs-dir",
        default="travel_outputs",
        help="Directory containing per-session outputs (default: travel_outputs)",
    )
    parser.add_argument(
        "--out-dir",
        default="travel_outputs/eval",
        help="Directory to write summary JSON/CSV (default: travel_outputs/eval)",
    )
    args = parser.parse_args()

    outputs_dir = Path(args.outputs_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = evaluate_layer_metrics(outputs_dir)

    ts = int(time.time())
    json_path = out_dir / f"layer_metrics_summary_{ts}.json"
    csv_path = out_dir / f"layer_metrics_sessions_{ts}.csv"

    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)

    metrics = [
        SessionMetric(
            session_id=s["session_id"],
            artifact_id=s.get("artifact_id", ""),
            created_at=_safe_float(s.get("created_at")),
            layer_metrics=s,
        )
        for s in summary.get("sessions", [])
    ]
    _write_csv(csv_path, metrics)

    overall = summary.get("overall", {})
    print(f"Sessions: {summary.get('session_count', 0)}")
    print(f"Avg hit rate: {overall.get('avg_hit_rate', 0.0):.3f}")
    print(f"Avg rollback rate: {overall.get('avg_rollback_rate', 0.0):.3f}")
    print(f"Global rollback rate: {overall.get('global_rollback_rate', 0.0):.3f}")
    print(f"JSON written to: {json_path}")
    print(f"CSV written to: {csv_path}")


if __name__ == "__main__":
    main()
