"""Strict CSV / cached-value XLSX references; only the standard library is needed.

The XLSX reader is deliberately limited to a tabular PSPLIB reference sheet.
It does not evaluate formulas or interpret arbitrary workbook semantics.
"""
import csv
import math
import posixpath
import re
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
RID = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"
REQUIRED = {"FileName", "#Act", "CPM", "LBall", "UB-lit", "KeepOPT"}


def _xlsx_rows(path):
    with zipfile.ZipFile(path) as archive:
        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        sheet = next((s for s in workbook.findall("m:sheets/m:sheet", NS)
                      if s.get("name") == "PSPLIB"), None)
        if sheet is None:
            raise ValueError("Reference workbook must contain a PSPLIB sheet")
        relationships = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        relation = next(r for r in relationships if r.get("Id") == sheet.get(RID))
        if relation.get("TargetMode") == "External":
            raise ValueError("External worksheet relationships are unsupported")
        target = relation.get("Target")
        member = (target.lstrip("/") if target.startswith("/") else
                  posixpath.normpath(posixpath.join("xl", target)))
        strings = []
        if "xl/sharedStrings.xml" in archive.namelist():
            root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            strings = ["".join(t.text or "" for t in si.findall(".//m:t", NS))
                       for si in root.findall("m:si", NS)]
        header = None
        with archive.open(member) as stream:
            for _, row in ET.iterparse(stream, events=("end",)):
                if row.tag != "{" + NS["m"] + "}row":
                    continue
                cells = {}
                for cell in row.findall("m:c", NS):
                    column = re.sub(r"\d", "", cell.get("r", ""))
                    if header is not None and header.get(column) not in REQUIRED | {"Optimum"}:
                        continue
                    value = cell.find("m:v", NS)
                    kind = cell.get("t")
                    if kind == "inlineStr":
                        decoded = "".join(t.text or "" for t in cell.findall(".//m:t", NS))
                    elif value is None or value.text is None:
                        decoded = None
                    elif kind == "s":
                        decoded = strings[int(value.text)]
                    elif kind == "e":
                        raise ValueError("Reference cell contains an Excel error: " + cell.get("r"))
                    else:
                        decoded = value.text
                    cells[column] = decoded
                if header is None:
                    header = {key: str(value).strip() for key, value in cells.items() if value is not None}
                    if not REQUIRED <= set(header.values()):
                        raise ValueError("Missing required reference columns")
                else:
                    yield {header[key]: value for key, value in cells.items() if key in header}
                row.clear()


def _integer(value, label, optional=False):
    if value is None or str(value).strip() == "":
        if optional:
            return None
        raise ValueError("Missing reference value: " + label)
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError("Invalid reference value: " + label)
    if not math.isfinite(number) or number < 0 or not number.is_integer():
        raise ValueError("Reference must be a nonnegative integer: " + label)
    return int(number)


def load_references(path, targets=None):
    path = Path(path).resolve(strict=True)
    wanted = None if targets is None else {name.lower() for name in targets}
    if path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if not REQUIRED <= set(reader.fieldnames or []):
                raise ValueError("Missing required reference columns")
            rows = list(reader)
    elif path.suffix.lower() == ".xlsx":
        rows = _xlsx_rows(path)
    else:
        raise ValueError("References must be .csv or .xlsx")
    references = {}
    for row in rows:
        name = str(row.get("FileName") or "").strip().lower()
        if not name or (wanted is not None and name not in wanted):
            continue
        if name in references:
            raise ValueError("Duplicate reference instance: " + name)
        lb = _integer(row.get("LBall"), "LBall", optional=True)
        ub = _integer(row.get("UB-lit"), "UB-lit")
        optimum = _integer(row.get("Optimum"), "Optimum", optional=True)
        if lb is not None and lb > ub:
            raise ValueError("Reference LB exceeds UB: " + name)
        if optimum is not None and ((lb is not None and optimum < lb) or optimum > ub):
            raise ValueError("Optimum contradicts reference bounds: " + name)
        # KeepOPT is an opaque workbook annotation, never an optimality certificate.
        if optimum is None and lb == ub:
            optimum = ub
        references[name] = {"activities": _integer(row.get("#Act"), "#Act"),
                            "cpm": _integer(row.get("CPM"), "CPM"),
                            "lower_bound": lb, "bks": ub, "optimum": optimum,
                            "optimality_basis": ("explicit_Optimum" if row.get("Optimum") not in (None, "")
                                                 else "LB_equals_UB" if optimum is not None else None),
                            "reference_status": row.get("KeepOPT"), "source": str(path)}
    return references


def quality_metrics(makespan, reference):
    if not reference:
        return {"gap_to_optimum_percent": None, "deviation_to_bks_percent": None,
                "gap_to_lb_percent": None, "improves_bks": None}
    lb, optimum, bks = (reference.get(key) for key in ("lower_bound", "optimum", "bks"))
    if lb is not None and makespan < lb:
        raise ValueError("Feasible schedule contradicts reference lower bound")
    if optimum is not None and makespan < optimum:
        raise ValueError("Feasible schedule contradicts declared optimum")
    def deviation(value):
        return (makespan - value) / value * 100 if value else None
    return {"gap_to_optimum_percent": deviation(optimum),
            "deviation_to_bks_percent": deviation(bks), "gap_to_lb_percent": deviation(lb),
            "improves_bks": makespan < bks}
