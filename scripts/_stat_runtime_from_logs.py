# -*- coding: utf-8 -*-
"""Stat single-experiment wall time from benchmark logs."""
from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
TS = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")
DONE = re.compile(r"Report generation complete|Experiment summary saved")
EVAL = re.compile(r"Evaluating: model=(\S+) strategy=(\S+) n_series=(\d+)")
START_LOAD = re.compile(r"Start loading 1 series")


def _parse_name(stem: str) -> tuple[str, str] | None:
    if stem.endswith("_score"):
        return stem[:-6].lower(), "score"
    if stem.endswith("_label"):
        return stem[:-6].lower(), "label"
    return None


def parse_log(path: Path) -> dict:
    first = last = None
    done = False
    model = strategy = None
    n_series_cfg = None
    n_loads = 0
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = TS.match(line)
            if m:
                t = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
                if first is None:
                    first = t
                last = t
            if DONE.search(line):
                done = True
            em = EVAL.search(line)
            if em:
                model, strategy, n_series_cfg = em.group(1), em.group(2), int(em.group(3))
            if START_LOAD.search(line):
                n_loads += 1
    wall = (last - first).total_seconds() if first and last else None
    return {
        "first": first,
        "last": last,
        "wall_s": wall,
        "done": done,
        "model_from_log": model,
        "strategy": strategy,
        "n_series_cfg": n_series_cfg,
        "n_series_started": n_loads,
    }


def find_elapsed(model: str, track: str) -> tuple[float | None, str | None]:
    results = ROOT / "results"
    if not results.exists():
        return None, None
    cands = []
    for h in results.rglob("experiment_summary.json"):
        s = str(h).replace("\\", "/").lower()
        if f"/{track}/" in s and model.lower() in Path(s).parts:
            cands.append(h)
    for h in cands:
        try:
            j = json.loads(h.read_text(encoding="utf-8"))
            el = j.get("elapsed_sec") or j.get("elapsed_seconds") or j.get("wall_time_sec")
            if el is not None:
                return float(el), str(h.relative_to(ROOT)).replace("\\", "/")
        except Exception:
            continue
    return None, None


def status_of(info: dict) -> str:
    n = info["n_series_started"] or 0
    if info["done"]:
        return "done"
    if n >= 90:
        return "eval_near_done"
    if n > 0:
        return f"partial_{n}/95"
    return "no_progress"


def main() -> None:
    all_logs: list[tuple[str, str, Path]] = []
    for p in list((ROOT / "logs").rglob("*.log")) + list(ROOT.glob("*.log")):
        if p.stat().st_size == 0:
            continue
        parsed = _parse_name(p.stem)
        if not parsed:
            continue
        model, track = parsed
        all_logs.append((model, track, p))

    best: dict[tuple[str, str], dict] = {}
    for model, track, p in all_logs:
        info = parse_log(p)
        rel = str(p.relative_to(ROOT)).replace("\\", "/")
        info["path"] = rel
        info["model"] = model
        info["track"] = track
        info["bytes"] = p.stat().st_size

        def rank(x: dict) -> tuple:
            return (
                1 if x["done"] else 0,
                x["n_series_started"] or 0,
                1 if x["path"].startswith("logs/") else 0,
                x["bytes"],
                x["wall_s"] or 0,
            )

        cur = best.get((model, track))
        if cur is None or rank(info) > rank(cur):
            best[(model, track)] = info

    rows = []
    for (model, track), info in sorted(best.items()):
        el, _ = find_elapsed(model, track)
        wall = info["wall_s"]
        rows.append(
            {
                "model": model,
                "track": track,
                "status": status_of(info),
                "wall_h": round(wall / 3600, 3) if wall is not None else None,
                "wall_min": round(wall / 60, 1) if wall is not None else None,
                "wall_s": round(wall) if wall is not None else None,
                "elapsed_min_json": round(el / 60, 1) if el is not None else None,
                "n_series_started": info["n_series_started"] or 0,
                "n_series_cfg": info["n_series_cfg"],
                "start": info["first"].strftime("%Y-%m-%d %H:%M:%S") if info["first"] else None,
                "end": info["last"].strftime("%Y-%m-%d %H:%M:%S") if info["last"] else None,
                "log": info["path"],
            }
        )

    df = pd.DataFrame(rows)
    out = ROOT / "export" / "experiment_runtime_from_logs.csv"
    out.parent.mkdir(exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Wrote {len(df)} rows -> {out}")

    for track in ["score", "label"]:
        sub = df[df.track == track].sort_values("wall_s", ascending=False)
        print(f"\n## {track} ({len(sub)} experiments)")
        cols = ["model", "status", "wall_h", "wall_min", "n_series_started", "start", "end"]
        print(sub[cols].to_string(index=False))

    # pivoted convenience view
    score = df[df.track == "score"].set_index("model")
    label = df[df.track == "label"].set_index("model")
    models = sorted(set(score.index) | set(label.index))
    piv_rows = []
    for m in models:
        s = score.loc[m] if m in score.index else None
        l = label.loc[m] if m in label.index else None
        piv_rows.append(
            {
                "model": m,
                "score_status": None if s is None else s["status"],
                "score_h": None if s is None else s["wall_h"],
                "score_min": None if s is None else s["wall_min"],
                "label_status": None if l is None else l["status"],
                "label_h": None if l is None else l["wall_h"],
                "label_min": None if l is None else l["wall_min"],
                "total_h_both": (
                    None
                    if s is None or l is None or pd.isna(s["wall_h"]) or pd.isna(l["wall_h"])
                    else round(float(s["wall_h"]) + float(l["wall_h"]), 3)
                ),
            }
        )
    piv = pd.DataFrame(piv_rows).sort_values("total_h_both", ascending=False, na_position="last")
    out2 = ROOT / "export" / "experiment_runtime_by_method.csv"
    piv.to_csv(out2, index=False)
    print(f"\nWrote method pivot -> {out2}")
    print(piv.to_string(index=False))


if __name__ == "__main__":
    main()
