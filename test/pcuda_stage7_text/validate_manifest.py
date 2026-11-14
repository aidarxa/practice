#!/usr/bin/env python3
## @file
## @brief Offline independent counts and every visible Text/star row of the current manifest.
"""Run only after db_cli exits. No SQL/device helpers or timer are used as the oracle."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True

import numpy as np
from validate_cli import SOURCES, custom2

FACT = [("lo_orderkey",0),("lo_linenumber",0),("lo_custkey",0),("lo_partkey",0),("lo_suppkey",0),("lo_orderdate",0),
        ("lo_orderpriority",16),("lo_shippriority",-1),("lo_quantity",0),("lo_extendedprice",0),("lo_ordtotalprice",0),
        ("lo_discount",0),("lo_revenue",0),("lo_supplycost",0),("lo_tax",0),("lo_commitdate",0),("lo_shipmode",10)]


class PhysicalOracle:
    def __init__(self,data):
        self.tables = dict(SOURCES,lineorder=FACT)
        self.arrays, self.table, self.width = {}, {}, {}
        for table,columns in self.tables.items():
            rows = None
            for i,(name,width) in enumerate(columns):
                dtype = f"S{width}" if width>0 else ("u1" if width<0 else "<i4")
                values = np.memmap(data/(table.upper()+str(i)),dtype=dtype,mode="r")
                if rows is not None and len(values)!=rows:
                    raise ValueError("Independent physical column extent mismatch")
                self.arrays[name], self.table[name], self.width[name] = values,table,width
                rows = len(values)
        for table,key in (("customer","c_custkey"),("supplier","s_suppkey")):
            if not np.array_equal(self.arrays[key],np.arange(1,len(self.arrays[key])+1)):
                raise ValueError("SF10 oracle requires independently verified consecutive unique dimension keys")

    def value(self,name,row):
        value = self.arrays[name][row]
        if self.width[name]>0:
            return bytes(value).split(b"\0",1)[0].decode("utf-8",errors="strict")
        if self.width[name]<0:
            if value not in (48,49):
                raise ValueError("Invalid physical ASCII flag")
            return int(value)-48
        return int(value)

    def fact_row(self,values):
        keys = self.arrays["lo_orderkey"]
        key=values["lo_orderkey"]
        if key<np.iinfo(np.int32).min or key>np.iinfo(np.int32).max:
            raise ValueError("Fact key is outside the independent physical Int32 domain")
        # Match the physical dtype: avoid promoting/copying the entire 60M-row array for each scalar lookup.
        key=np.int32(key)
        begin,end = keys.searchsorted(key,side="left"),keys.searchsorted(key,side="right")
        lines = self.arrays["lo_linenumber"][begin:end]
        candidates = np.flatnonzero(lines==values["lo_linenumber"])
        if len(candidates)!=1:
            raise ValueError("Visible result does not identify one physical fact row")
        return int(begin+candidates[0])

    def count(self,kind):
        if kind=="ddate":
            return len(self.arrays["d_datekey"])
        count = 0
        customers,suppliers = self.arrays["lo_custkey"],self.arrays["lo_suppkey"]
        for start in range(0,len(customers),1_000_000):
            c,s = customers[start:start+1_000_000],suppliers[start:start+1_000_000]
            if c.min()<1 or c.max()>len(self.arrays["c_custkey"]):
                raise ValueError("Invalid independent customer FK")
            if kind=="customer-region":
                count += int(np.count_nonzero(self.arrays["c_region"][c-1]==1))
            else:
                if s.min()<1 or s.max()>len(self.arrays["s_suppkey"]):
                    raise ValueError("Invalid independent supplier FK")
                count += int(np.count_nonzero(self.arrays["c_nation"][c-1]==self.arrays["s_nation"][s-1]))
        return count

    def rows(self,record,kind):
        seen = Counter()
        for row in record["rows"]:
            visible = dict(zip(record["columns"],row))
            if len(visible)!=len(row):
                raise ValueError("Duplicate/missing visible column identity")
            if kind=="ddate":
                keys=self.arrays["d_datekey"]
                identity=visible["d_datekey"]
                position=int(keys.searchsorted(np.int32(identity)))
                if position==len(keys) or int(keys[position])!=identity:
                    raise ValueError("Visible DDATE key is not a physical row")
                indexes = {"ddate":position}
            else:
                fact = self.fact_row(visible)
                indexes = {"lineorder":fact,"customer":self.value("lo_custkey",fact)-1,"supplier":self.value("lo_suppkey",fact)-1}
                identity = fact
                if kind=="customer-region" and self.value("c_region",indexes["customer"])!=1:
                    raise ValueError("Visible row violates customer filter")
                if kind=="same-nation" and self.value("c_nation",indexes["customer"])!=self.value("s_nation",indexes["supplier"]):
                    raise ValueError("Visible row violates cross-table filter")
            seen[identity] += 1
            for name,value in visible.items():
                if value!=self.value(name,indexes[self.table[name]]):
                    raise ValueError("Physical Text/numeric payload mismatch at "+name)
        if any(n!=1 for n in seen.values()):
            raise ValueError("Visible sample duplicates a unique SF10 joined row")
        return len(seen)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report",type=Path,required=True)
    parser.add_argument("--data",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    source = json.loads(args.report.read_text())
    oracle = PhysicalOracle(args.data)
    records = []
    counts = {kind:oracle.count(kind) for kind in ("ddate","customer-region","same-nation")}
    for record in source["records"]:
        name = record["id"]
        kind = "ddate" if name.startswith("proj_05_select_star_small_dimension-") else (
            "customer-region" if name.startswith(("proj_join_02_star_customer-","stress_01_wide_star_customer-")) else (
            "same-nation" if name.startswith("stress_02_wide_star_customer_supplier-") else None))
        if not kind and not name.startswith("custom2-"):
            continue
        result = dict(id=name,success=False)
        try:
            if not record["success"]:
                raise ValueError("Original execution was not successful")
            if kind:
                if record["row_count"]!=counts[kind]:
                    raise ValueError("Independent complete row-count mismatch")
                result.update(expected_count=counts[kind],visible_rows_checked=oracle.rows(record,kind))
            else:
                expected = custom2(args.data)
                if Counter(map(tuple,record["rows"]))!=Counter(expected):
                    raise ValueError("Independent custom2 signed aggregation mismatch")
                result.update(expected=expected,visible_rows_checked=len(expected))
            result["success"]=True
        except (ValueError,KeyError,IndexError) as error:
            result["error"]=str(error)
        records.append(result)
    report = dict(success=len(records)==5 and all(r["success"] for r in records),source=str(args.report.resolve()),records=records,
                  scope="Full independent row counts and every visible row of all five previously unsupported Text/star manifest forms; display window is not a full 12M-row CPU materialization")
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))
    return 0 if report["success"] else 1


if __name__=="__main__":
    raise SystemExit(main())
