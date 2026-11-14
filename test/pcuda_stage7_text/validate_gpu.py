#!/usr/bin/env python3
## @file
## @brief Independent SQLite oracle for real native Text execution and process-local reuse.
"""Correctness only: retain raw streams and native Bash process timers; no Python timer."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re
import sqlite3
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / "test/pcuda_stage7"))
from validate import process, unexpected_diagnostics


def fixture():
    """Independent SQL values, not the device layout/helpers being tested."""
    database = sqlite3.connect(":memory:")
    database.executescript("CREATE TABLE customer(c_custkey INTEGER,c_name TEXT,c_city INTEGER);"
                           "CREATE TABLE supplier(s_suppkey INTEGER,s_name TEXT,s_city INTEGER);"
                           "CREATE TABLE part(p_partkey INTEGER,p_name TEXT);")
    names = ["", None, "Alice", "Alice", "a ", "a", "Я", "1234567890123456789012345"]
    database.executemany("INSERT INTO customer VALUES(?,?,?)", [(i+1, name, i//2+1) for i, name in enumerate(names)])
    database.executemany("INSERT INTO supplier VALUES(?,?,?)", list(zip([3,4,1,2,5,6], ["Alice","Alice","",None,"a ","Z"], range(1,7))))
    database.executemany("INSERT INTO part VALUES(?,?)",[(1,"wide"),(2,"🙂")])
    return database


def workload():
    """Each ready flag requires all six generation/build/load counters to be zero."""
    items = []
    def add(sql, ready=False, oracle=None, error=None, types=None):
        items.append(dict(sql=sql, ready=ready, oracle=oracle, error=error, types=types))
    add("SELECT c_custkey,c_name FROM customer ORDER BY 1", types="IT")
    add("SELECT * FROM customer ORDER BY 1", types="ITI")
    add("SELECT p_partkey,p_name FROM part ORDER BY 1",types="IT")
    for op in ("=", "<>", "<", "<=", ">", ">="):
        for i, value in enumerate(("'Alice'", "''", "'a '", "'Я'", "'🙂'", "'"+"x"*513+"'", "NULL", "'Alice'")):
            add(f"SELECT c_custkey,c_name FROM customer WHERE c_name{op}{value} ORDER BY 1", ready=i>0, types="IT")
    for sql in (
        "SELECT c_custkey,c_name,c_name='Alice',c_name<>'' FROM customer ORDER BY 1",
        "SELECT c_name,c_name IS NULL,c_name IS NOT NULL FROM customer",
        "SELECT c_custkey,c_name FROM customer WHERE NOT(c_name='Alice') ORDER BY 1",
        "SELECT c_custkey,c_name FROM customer WHERE c_name=NULL OR c_city=1 ORDER BY 1",
        "SELECT c_custkey,c_name FROM customer WHERE c_name=NULL AND c_city=1 ORDER BY 1",
        "SELECT c_custkey,CASE WHEN c_name='Alice' THEN c_name ELSE '🙂' END FROM customer ORDER BY 1",
        "SELECT c_custkey,CASE WHEN c_city=1 THEN c_name ELSE NULL END FROM customer ORDER BY 1",
        "SELECT c_custkey,CASE WHEN c_name IS NULL THEN NULL ELSE c_name END FROM customer ORDER BY 1",
        "SELECT c_name,CASE WHEN c_name='' THEN 9007199254740993 ELSE 9007199254740994 END FROM customer",
        "SELECT c_name,CASE WHEN c_name='' THEN 0.5 ELSE -2 END FROM customer",
        "SELECT 'é','é','🙂',c_name FROM customer WHERE c_custkey=1",
        "SELECT COUNT(c_name),COUNT(*),MIN(c_name),MAX(c_name) FROM customer",
        "SELECT COUNT(c_name),MIN(c_name),MAX(c_name) FROM customer WHERE c_name IS NULL",
        "SELECT COUNT(c_name),MIN(c_name),MAX(c_name) FROM customer WHERE c_city=99",
        "SELECT c_name,COUNT(*),COUNT(c_name),MIN(c_name),MAX(c_name) FROM customer GROUP BY c_name",
        "SELECT COUNT(*),c_name,MIN(c_name),SUM(c_custkey),AVG(c_custkey) FROM customer GROUP BY c_name",
        "SELECT c_name,c_city,COUNT(*),SUM(c_custkey) FROM customer GROUP BY c_name,c_city",
        "SELECT c_city,c_name,COUNT(*) FROM customer GROUP BY c_city,c_name HAVING COUNT(*)>0",
        "SELECT c_name,COUNT(*),MIN(c_name) FROM customer GROUP BY c_name HAVING MIN(c_name)>='Alice'",
        "SELECT c_name,COUNT(*) FROM customer GROUP BY c_name HAVING c_name IS NULL",
        "SELECT CASE WHEN c_name='Alice' THEN 'same' ELSE c_name END,COUNT(*) FROM customer GROUP BY CASE WHEN c_name='Alice' THEN 'same' ELSE c_name END",
        "SELECT CASE WHEN c_city=1 THEN 'equal' ELSE 'equal' END,COUNT(*) FROM customer GROUP BY CASE WHEN c_city=1 THEN 'equal' ELSE 'equal' END",
        "SELECT CASE WHEN c_name IS NULL THEN NULL ELSE c_name END,COUNT(*) FROM customer GROUP BY CASE WHEN c_name IS NULL THEN NULL ELSE c_name END",
        "SELECT c_custkey,s_suppkey,c_name,s_name FROM customer,supplier WHERE c_name=s_name ORDER BY 1,2",
        "SELECT c_custkey,s_suppkey,c_name FROM customer,supplier WHERE c_name=s_name AND c_city=2 ORDER BY 1,2",
        "SELECT c_custkey,s_suppkey,c_name FROM customer,supplier WHERE c_name=s_name OR c_name=s_name ORDER BY 1,2",
        "SELECT c_custkey,s_suppkey,c_name FROM customer,supplier WHERE c_name=s_name OR c_custkey=s_suppkey ORDER BY 1,2",
        "SELECT c_name,COUNT(*),SUM(s_suppkey) FROM customer,supplier WHERE c_name=s_name GROUP BY c_name",
        "SELECT c_name,COUNT(*) FROM customer,supplier WHERE c_name=s_name AND s_name='missing' GROUP BY c_name",
    ):
        add(sql)
    for direction in ("ASC", "DESC"):
        for i, limit in enumerate((1, 8, 0, 100, 2)):
            sql = f"SELECT c_name,c_custkey FROM customer ORDER BY 1 {direction},2 LIMIT {limit}"
            add(sql, ready=i>0, oracle=sql.replace(f"1 {direction},", f"1 {direction} NULLS LAST,"), types="TI")
        sql = f"SELECT COUNT(*),c_name FROM customer GROUP BY c_name ORDER BY 2 {direction},1"
        add(sql, oracle=sql.replace(f"2 {direction},", f"2 {direction} NULLS LAST,"), types="UT")
    for sql in ("SELECT SUM(c_name) FROM customer", "SELECT AVG(c_name) FROM customer",
                "SELECT c_name FROM customer WHERE c_name=1", "SELECT c_name+1 FROM customer",
                "SELECT CASE WHEN c_city=1 THEN c_name ELSE 2 END FROM customer"):
        add(sql, error="Text")
    add("SELECT c_name,COUNT(*) FROM customer GROUP BY c_name HAVING MIN(c_name)>='Alice'", error="HAVING aggregate expression must also appear")
    add("SELECT c_custkey,c_name FROM customer ORDER BY 1", ready=True, types="IT")
    # Text remains a view parameter when four independent numeric slots reach the tagged fallback.
    for repeat in range(2):
        for mask in range(16):
            values=[str(value)+(".0" if mask&(1<<i) else "") for i,value in enumerate((0,9,1,4))]
            add("SELECT COUNT(c_name) FROM customer WHERE c_name>='' AND c_custkey>"+values[0]+" AND c_custkey<"+values[1]+
                " AND c_city>="+values[2]+" AND c_city<="+values[3],ready=repeat>0,types="U")
            if not repeat:
                items[-1]["numeric_schema_mask"]=mask
    return items


def decode(cell):
    if cell == "N":
        return None
    if cell.startswith("T"):
        return bytes.fromhex(cell[1:]).decode("utf-8", errors="strict")
    if cell[0] in "IU":
        return int(cell[1:])
    if cell[0] == "D":
        return float(cell[1:])
    raise ValueError("Unknown typed cell: " + cell)


def parse_rows(text):
    match = re.search(r"^__TEXT_ROWS__ (\d+) (\d+)\n(.*?)^__TEXT_EVENTS__ ([\d ]+)\n", text, re.M | re.S)
    if not match:
        raise ValueError("Missing typed rows/events: " + text[:500])
    rows, columns = int(match[1]), int(match[2])
    raw = [line.split("\t") for line in match[3].splitlines()]
    if len(raw) != rows or any(len(row) != columns for row in raw):
        raise ValueError("Typed row/column extent mismatch")
    return [tuple(decode(cell) for cell in row) for row in raw], raw, list(map(int, match[4].split()))


def check(item, text, database):
    if item["error"]:
        if "__TEXT_ERROR__" not in text or item["error"] not in text or "__TEXT_ROWS__" in text:
            raise ValueError("Expected diagnostic was not emitted")
        return dict(status="expected-error")
    if "__TEXT_ERROR__" in text:
        raise ValueError(text.split("__TEXT_ERROR__ ",1)[1].strip())
    actual, raw, events = parse_rows(text)
    expected = database.execute(item["oracle"] or item["sql"]).fetchall()
    ordered = "ORDER BY" in item["sql"]
    same = actual == expected if ordered else Counter(actual) == Counter(expected)
    if not same:
        raise ValueError(f"SQLite mismatch: actual={actual!r}, expected={expected!r}")
    if item["types"] and any(any(cell != "N" and cell[0] != kind for cell,kind in zip(row,item["types"])) for row in raw):
        raise ValueError("Typed result ABI mismatch")
    if len(events) != 6 or (item["ready"] and any(events)):
        raise ValueError("Compatible values triggered generation/build/load: " + str(events))
    if item.get("numeric_schema_mask",0)>8 and any(events[1:]):
        raise ValueError("A ready tagged numeric fallback rebuilt a module containing a Text parameter")
    return dict(status="executed", rows=actual, events=events)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--traces", action="store_true")
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix="gpu-traced-" if args.traces else "gpu-plain-", dir=args.output_root.resolve()))
    config = output / "crystal.conf"
    config.write_text("dump_generated_code=off\nreuse_scratch_buffers=true\nmemory_guard_enabled=on\n"
                      f"jit_keep_artifacts=on\njit_artifact_dir={output}\njit_traces={'on' if args.traces else 'off'}\n")
    items = workload()
    session, stdout, stderr = process(args.build.resolve()/"test_pcuda_text_gpu", [],
                                      "\n".join(item["sql"] for item in items)+"\n", output/"session", config)
    blocks = re.findall(r"^__TEXT_BEGIN__ (.*?)\n(.*?)^__TEXT_END__$", stdout, re.M | re.S)
    database = fixture()
    records = []
    for i,item in enumerate(items):
        record = dict(index=i, **item, success=False)
        try:
            if i >= len(blocks) or blocks[i][0] != item["sql"]:
                raise ValueError("Missing/incorrect query marker")
            record.update(check(item,blocks[i][1],database),success=True)
        except (ValueError,sqlite3.Error) as error:
            record["failure"] = str(error)
        records.append(record)
    diagnostics = unexpected_diagnostics(stderr)
    report = dict(**session, records=records, unexpected_diagnostics=diagnostics,
                  success=session["returncode"]==0 and not session["timeout"] and not diagnostics and len(blocks)==len(items) and all(r["success"] for r in records),
                  oracle="Independent SQLite BINARY UTF-8 SQL; NULLS LAST explicitly matched; ordered rows or duplicate-preserving multisets",
                  scope="Tiny real QueryEngine/PCUDA fixture, not SF10, device coverage, a performance study or final migration acceptance")
    (output/"report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    print(f"Artifacts: {output}\nChecks: {sum(r['success'] for r in records)}/{len(records)}; success={report['success']}")
    for record in records:
        if not record["success"]:
            print(f"{record['index']}: {record['sql']}\n  {record['failure']}")
    return 0 if report["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
