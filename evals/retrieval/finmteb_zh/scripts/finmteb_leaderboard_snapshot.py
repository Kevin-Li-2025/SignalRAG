#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import re
import urllib.request
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

BENCHMARK_URL = "https://huggingface.co/spaces/FinanceMTEB/FinMTEB/resolve/main/benchmark.xlsx"
NS = {
    "a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
}


def _col_idx(cell_ref: str) -> int:
    letters = re.match(r"([A-Z]+)", cell_ref).group(1)
    value = 0
    for char in letters:
        value = value * 26 + ord(char) - 64
    return value - 1


def _read_xlsx(path: Path) -> dict[str, list[list[object]]]:
    with zipfile.ZipFile(path) as archive:
        shared_strings = []
        if "xl/sharedStrings.xml" in archive.namelist():
            root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            for item in root.findall("a:si", NS):
                text = "".join(node.text or "" for node in item.iter(f"{{{NS['a']}}}t"))
                shared_strings.append(text)

        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        rels = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        rel_targets = {rel.attrib["Id"]: rel.attrib["Target"] for rel in rels}

        sheets: dict[str, list[list[object]]] = {}
        for sheet in workbook.findall("a:sheets/a:sheet", NS):
            name = sheet.attrib["name"]
            rel_id = sheet.attrib[f"{{{NS['r']}}}id"]
            target = rel_targets[rel_id]
            sheet_path = "xl/" + target.lstrip("/") if not target.startswith("xl/") else target
            root = ET.fromstring(archive.read(sheet_path))
            rows = []
            for row in root.findall(".//a:sheetData/a:row", NS):
                values = {}
                for cell in row.findall("a:c", NS):
                    ref = cell.attrib.get("r", "A1")
                    idx = _col_idx(ref)
                    cell_type = cell.attrib.get("t")
                    node = cell.find("a:v", NS)
                    if node is None:
                        value: object = ""
                    elif cell_type == "s":
                        value = shared_strings[int(node.text)]
                    else:
                        value = node.text or ""
                        try:
                            number = float(value)
                            value = int(number) if number.is_integer() else number
                        except ValueError:
                            pass
                    values[idx] = value
                if values:
                    rows.append([values.get(idx, "") for idx in range(max(values) + 1)])
            sheets[name] = rows
    return sheets


def _model_average(row: list[object]) -> float | None:
    scores = []
    for value in row[2:]:
        if isinstance(value, int | float):
            scores.append(float(value))
        else:
            try:
                scores.append(float(str(value)))
            except ValueError:
                pass
    if not scores:
        return None
    return sum(scores) / len(scores)


def summarize(path: Path) -> dict[str, list[dict[str, object]]]:
    sheets = _read_xlsx(path)
    summary: dict[str, list[dict[str, object]]] = {}
    for sheet_name, rows in sheets.items():
        if not rows or not sheet_name.startswith("Reranking"):
            continue
        entries = []
        for row in rows[1:]:
            if not row or not row[0]:
                continue
            avg = _model_average(row)
            if avg is None:
                continue
            entries.append({"model": row[0], "average": avg, "scores": row[2:]})
        summary[sheet_name] = sorted(entries, key=lambda item: item["average"], reverse=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=Path("reports/finmteb_benchmark.xlsx"))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    args.cache.parent.mkdir(parents=True, exist_ok=True)
    if not args.cache.exists():
        urllib.request.urlretrieve(BENCHMARK_URL, args.cache)

    summary = summarize(args.cache)
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    for sheet, entries in summary.items():
        print(sheet)
        for entry in entries[:5]:
            print(f"  {entry['average']:.6f}  {entry['model']}  {entry['scores']}")


if __name__ == "__main__":
    main()

