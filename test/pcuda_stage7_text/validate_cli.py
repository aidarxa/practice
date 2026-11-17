#!/usr/bin/env python3
## @file
## @brief Real db_cli SF10 Text matrix with independent physical-file/SQLite/NumPy oracles.
"""Correctness/diagnostic series, not a Text performance claim or final acceptance."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import re
import sqlite3
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.dont_write_bytecode = True
sys.path.insert(0,str(ROOT/"test/pcuda_stage7"))
from validate import process, module, unexpected_diagnostics

# Independent physical source description; deliberately not imported from the C++ schema.
SOURCES = {
    "customer": [("c_custkey",0),("c_name",25),("c_address",25),("c_city",10),("c_nation",15),("c_region",12),("c_phone",15),("c_mktsegment",10)],
    "supplier": [("s_suppkey",0),("s_name",25),("s_address",25),("s_city",10),("s_nation",15),("s_region",12),("s_phone",15)],
    "part": [("p_partkey",0),("p_name",22),("p_mfgr",6),("p_category",7),("p_brand1",10),("p_color",11),("p_type",25),("p_size",0),("p_container",10)],
    "ddate": [("d_datekey",0),("d_date",18),("d_dayofweek",9),("d_month",9),("d_year",0),("d_yearmonthnum",0),("d_yearmonth",7),("d_daynuminweek",0),("d_daynuminmonth",0),("d_daynuminyear",0),("d_monthnuminyear",0),("d_weeknuminyear",0),("d_sellingseason",12),("d_lastdayinweekfl",-1),("d_lastdayinmonthfl",-1),("d_holidayfl",-1),("d_weekdayfl",-1)]
}


def source_columns(data, table):
    """Decode exact records, bounded NUL, significant spaces and ASCII flags independently."""
    import numpy as np
    columns = []
    for index,(_,width) in enumerate(SOURCES[table]):
        path = data/(table.upper()+str(index))
        if width > 0:
            raw = path.read_bytes()
            if len(raw)%width:
                raise ValueError("Truncated independent text source")
            columns.append([raw[i:i+width].split(b"\0",1)[0].decode("utf-8",errors="strict") for i in range(0,len(raw),width)])
        elif width < 0:
            raw = path.read_bytes()
            if any(byte not in (48,49) for byte in raw):
                raise ValueError("Invalid independent flag source")
            columns.append([byte-48 for byte in raw])
        else:
            columns.append(list(map(int,np.fromfile(path,dtype="<i4"))))
    if len({len(column) for column in columns}) != 1:
        raise ValueError("Independent column extents differ")
    return columns


def sqlite_oracle(data):
    database = sqlite3.connect(":memory:")
    for table,source in SOURCES.items():
        database.execute("CREATE TABLE "+table+"("+",".join(name+(" TEXT" if width>0 else " INTEGER") for name,width in source)+")")
        database.executemany("INSERT INTO "+table+" VALUES("+",".join("?" for _ in source)+")",zip(*source_columns(data,table)))
    return database


def custom2(data):
    """Fact data are streamed offline, using exact signed sums, not engine hash/compare helpers."""
    import numpy as np
    segments = source_columns(data,"customer")[7]
    building = np.array([False]+[s=="BUILDING" for s in segments])
    furniture = np.array([False]+[s=="FURNITURE" for s in segments])
    customers = np.memmap(data/"LINEORDER2",dtype="<i4",mode="r")
    dates = np.memmap(data/"LINEORDER5",dtype="<i4",mode="r")
    revenue = np.memmap(data/"LINEORDER12",dtype="<i4",mode="r")
    date_columns = source_columns(data,"ddate")
    year_by_date = dict(zip(date_columns[0],date_columns[4]))
    totals = Counter()
    for start in range(0,len(customers),1_000_000):
        keys, when, value = customers[start:start+1_000_000],dates[start:start+1_000_000],revenue[start:start+1_000_000]
        if keys.min() < 1 or keys.max() >= len(building):
            raise ValueError("Independent oracle requires valid SF10 customer keys")
        for year,mask in ((1998,building[keys]),(1997,furniture[keys])):
            valid_dates = np.array([date for date,mapped in year_by_date.items() if mapped==year],dtype=np.int32)
            selected = mask & np.isin(when,valid_dates)
            if np.any(selected):
                totals[year] += int(value[selected].sum(dtype=np.int64))
    return list(totals.items())


def workload():
    items = []
    def add(sql, ready=False, error=None, oracle=None):
        items.append(dict(sql=sql,ready=ready,error=error,oracle=oracle))
    for query_index,sql in enumerate((
        "SELECT * FROM supplier ORDER BY s_suppkey LIMIT 7",
        "SELECT * FROM customer ORDER BY c_custkey LIMIT 7",
        "SELECT * FROM part ORDER BY p_partkey LIMIT 7",
        "SELECT * FROM ddate ORDER BY d_datekey LIMIT 7",
        "SELECT c_mktsegment,COUNT(*),MIN(c_name),MAX(c_name) FROM customer GROUP BY c_mktsegment ORDER BY 1",
        "SELECT COUNT(c_phone),MIN(c_phone),MAX(c_phone) FROM customer",
        "SELECT p_color,p_category,COUNT(*),SUM(p_size) FROM part GROUP BY p_color,p_category ORDER BY 1,2 LIMIT 32",
        "SELECT COUNT(*),d_dayofweek FROM ddate GROUP BY d_dayofweek ORDER BY 2 DESC",
        "SELECT d_sellingseason,COUNT(*),MIN(d_date),MAX(d_date) FROM ddate GROUP BY d_sellingseason HAVING COUNT(*)>0 ORDER BY 1",
        "SELECT CASE WHEN c_region<'ASIA' THEN c_mktsegment ELSE 'other' END,COUNT(*) FROM customer GROUP BY CASE WHEN c_region<'ASIA' THEN c_mktsegment ELSE 'other' END ORDER BY 1",
        "SELECT c_name,c_address FROM customer WHERE c_name<'Customer#000000008' ORDER BY 1",
    )):
        add(sql)
        if query_index==0:
            for _ in range(8):
                add(sql,ready=True)
    for i,value in enumerate(("'BUILDING'","'FURNITURE'","''","'🙂'","NULL","'BUILDING'")):
        add(f"SELECT COUNT(*),MIN(c_name),MAX(c_name) FROM customer WHERE c_mktsegment={value}",ready=i>0)
    for i,(a,b) in enumerate((("NULL","a ';|b\n\t\"\\Я🙂"),("","a "),("é","é"),(None,"other"),("same","same"))):
        literal = lambda text: "NULL" if text is None else "'"+text.replace("'","''")+"'"
        add(f"SELECT c_custkey,CASE WHEN c_custkey<4 THEN {literal(a)} ELSE {literal(b)} END FROM customer WHERE c_custkey<8 ORDER BY 1",ready=i>0)
    for i,value in enumerate((1,32,0,7)):
        add(f"SELECT c_mktsegment,c_custkey FROM customer ORDER BY 1 DESC,2 LIMIT {value}",ready=i>0)
    add(re.sub(r"--[^\n]*","",(ROOT/"olap_bench/sql/08_custom/custom2.sql").read_text()).strip().rstrip(";"),oracle="custom2")
    for sql in ("SELECT SUM(c_name) FROM customer", "SELECT AVG(c_name) FROM customer",
                "SELECT c_name FROM customer WHERE c_name=1", "SELECT NOT c_name FROM customer",
                "SELECT CASE WHEN c_name THEN 1 ELSE 0 END FROM customer"):
        add(sql,error="Text")
    add("SELECT COUNT(*),MIN(c_name),MAX(c_name) FROM customer WHERE c_mktsegment='FURNITURE'",ready=True)
    return items


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timing-model",choices=("pcuda-modules-v1","pcuda-modules-overlap-exp-v1"),default="pcuda-modules-v1")
    parser.add_argument("--build",type=Path,required=True)
    parser.add_argument("--data",type=Path,default=ROOT/"data/ssb/sf10")
    parser.add_argument("--output-root",type=Path,required=True)
    parser.add_argument("--traces",action="store_true")
    args = parser.parse_args()
    os.environ["CRYSTAL_DATA_DIR"] = str(args.data.resolve())
    os.environ["CRYSTAL_JIT_OVERLAP_EXPERIMENT"]="on" if args.timing_model=="pcuda-modules-overlap-exp-v1" else "off"
    args.output_root.mkdir(parents=True,exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix="text-cli-traced-" if args.traces else "text-cli-plain-",dir=args.output_root.resolve()))
    config = output/"crystal.conf"
    config.write_text("dump_generated_code=off\nextended_timing=on\noutput_limit=off\nreuse_scratch_buffers=true\n"
                      f"memory_guard_enabled=on\njit_keep_artifacts=on\njit_artifact_dir={output}\njit_traces={'on' if args.traces else 'off'}\n")
    items = workload()
    commands = ["\\dump off","\\timing on","\\limit 5000"]
    for i,item in enumerate(items):
        commands += [f"\\echo __TEXT_CLI_{i}__",item["sql"]+";"]
    commands += [f"\\echo __TEXT_CLI_{len(items)}__","\\q"]
    session,stdout,stderr = process(args.build.resolve()/"db_cli",[],"\n".join(commands)+"\n",output/"session",config,True)
    # No oracle calculation, source scan or online trace analysis during the measured process.
    database = sqlite_oracle(args.data.resolve())
    parser_rows = module("text_cli_parser","test/pcuda_stage1/validate_cli.py")
    timing = module("text_cli_timing","olap_bench/tools/run_olap_bench.py")
    records = []
    for i,item in enumerate(items):
        text = stdout.split(f"__TEXT_CLI_{i}__\n",1)[-1].split(f"__TEXT_CLI_{i+1}__\n",1)[0]
        record = dict(index=i,**item,success=False,**timing.parse_metrics(text))
        try:
            if stdout.count(f"__TEXT_CLI_{i}__\n")!=1 or stdout.count(f"__TEXT_CLI_{i+1}__\n")!=1:
                raise ValueError("Missing/duplicate marker")
            if item["error"]:
                if item["error"] not in text or "[ERROR]" not in text or "--- Result ---" in text:
                    raise ValueError("Expected early Text error missing")
                record.update(status="expected-error",success=True)
            else:
                if timing.has_sql_error(text) or record.get("reported_timing_model")!=args.timing_model:
                    raise ValueError("SQL/timing diagnostic: "+text[-1000:])
                columns,actual,count = parser_rows.parse_rows(text)
                expected = custom2(args.data.resolve()) if item["oracle"]=="custom2" else database.execute(item["sql"]).fetchall()
                same = actual==expected if "ORDER BY" in item["sql"] else Counter(actual)==Counter(expected)
                record.update(columns=columns,rows=actual,expected=expected,row_count=count)
                if not same or count!=len(expected):
                    raise ValueError("Independent physical SQLite/NumPy mismatch")
                if item["ready"] and any(record.get("module_"+key+"_calls")!=0 for key in ("plan_codegen","module_codegen","compiler","linker","loads","symbols")):
                    raise ValueError("Compatible Text parameters caused module work")
                record.update(status="executed",success=True)
        except (ValueError,IndexError,sqlite3.Error) as error:
            record["failure"] = str(error)
        records.append(record)
    diagnostics = unexpected_diagnostics(stderr)
    report = dict(**session,records=records,traces=args.traces,unexpected_diagnostics=diagnostics,
                  success=session["returncode"]==0 and not session["timeout"] and not diagnostics and all(r["success"] for r in records),
                  oracle="Offline independent exact physical files + SQLite BINARY; custom2 NumPy chunked signed sums",
                   measurement=args.timing_model+" continuous summary is total_query_ms; native multiquery process wall is not first-result startup; no SLA claim")
    (output/"report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    print(f"Artifacts: {output}\nChecks: {sum(r['success'] for r in records)}/{len(records)}; success={report['success']}")
    for record in records:
        if not record["success"]:
            print(f"{record['index']}: {record['sql']}\n  {record['failure']}")
    return 0 if report["success"] else 1


if __name__=="__main__":
    raise SystemExit(main())
