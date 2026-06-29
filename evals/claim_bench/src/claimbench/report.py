from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


def write_summary(
    out_dir: Path,
    *,
    model_name: str,
    revision: str | None,
    settings: dict,
    env: dict,
    results: dict,
) -> dict:
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": model_name,
        "revision": revision,
        "settings": settings,
        "environment": env,
        "results": results,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (out_dir / "summary.md").write_text(render_markdown(summary), encoding="utf-8")
    return summary


def render_markdown(summary: dict) -> str:
    results = summary["results"]
    lines = [
        "# Benchmark Summary",
        "",
        f"- model: `{summary['model']}`",
        f"- revision: `{summary.get('revision') or 'default'}`",
        f"- created_at: `{summary['created_at']}`",
        "",
        "## Results",
        "",
    ]
    if "humaneval" in results:
        item = results["humaneval"]
        lines.append(
            f"- HumanEval Pass@1: **{item['pass_at_1'] * 100:.2f}%** "
            f"({item['passed']} / {item['total']})"
        )
    if "gsm8k" in results:
        item = results["gsm8k"]
        sample_note = ""
        if item.get("samples", 1) > 1:
            sample_note = f", {item['samples']} samples, {item.get('selection', 'majority')}"
        lines.append(
            f"- GSM8K exact match: **{item['exact_match'] * 100:.2f}%** "
            f"({item['correct']} / {item['total']}), {item['n_shot']}-shot{sample_note}"
        )
    lines.extend(
        [
            "",
            "## Settings",
            "",
            "```json",
            json.dumps(summary["settings"], indent=2, sort_keys=True),
            "```",
            "",
            "## Environment",
            "",
            "```json",
            json.dumps(summary["environment"], indent=2, sort_keys=True),
            "```",
            "",
        ]
    )
    return "\n".join(lines)
