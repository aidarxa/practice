#!/usr/bin/env python3
import argparse
import csv
import json
import pathlib
import re
from typing import Callable


NATIONS = """ALGERIA
ARGENTINA
BRAZIL
CANADA
EGYPT
ETHIOPIA
FRANCE
GERMANY
INDIA
INDONESIA
IRAN
IRAQ
JAPAN
JORDAN
KENYA
MOROCCO
MOZAMBIQUE
PERU
CHINA
ROMANIA
SAUDI ARABIA
VIETNAM
RUSSIA
UNITED KINGDOM
UNITED STATES""".splitlines()

REGIONS = """AFRICA
AMERICA
ASIA
EUROPE
MIDDLE EAST""".splitlines()

# TPC-H/SSB market segment domain.
# The exact order is fixed to make encoded SQL predicates deterministic.
# AUTOMOBILE=0, BUILDING=1, FURNITURE=2, HOUSEHOLD=3, MACHINERY=4
MKTSEGMENTS = """AUTOMOBILE
BUILDING
FURNITURE
HOUSEHOLD
MACHINERY""".splitlines()

NATION_TO_ID = {value: idx for idx, value in enumerate(NATIONS)}
REGION_TO_ID = {value: idx for idx, value in enumerate(REGIONS)}
MKTSEGMENT_TO_ID = {value: idx for idx, value in enumerate(MKTSEGMENTS)}


def parse_delimiter(value: str | None, default: str = "|") -> str:
    if value is None:
        return default
    aliases = {
        "pipe": "|",
        "|": "|",
        "comma": ",",
        ",": ",",
        "tab": "\t",
        "\\t": "\t",
    }
    if value in aliases:
        return aliases[value]
    if len(value) == 1:
        return value
    raise ValueError(f"Unsupported delimiter: {value!r}. Use '|', ',', 'pipe', 'comma', or one character.")


def strip_dbgen_trailing_empty(row: list[str], expected_len: int) -> list[str]:
    # ssb-dbgen .tbl files usually end each line with a delimiter:
    # a|b|c|
    # csv.reader returns ['a', 'b', 'c', ''].
    if len(row) == expected_len + 1 and row[-1] == "":
        return row[:-1]
    return row


def add_dbgen_trailing_empty(row: list[str], emit_trailing_delimiter: bool) -> list[str]:
    return row + [""] if emit_trailing_delimiter else row


def require_len(row: list[str], expected_len: int, table: str, line_no: int) -> None:
    if len(row) != expected_len:
        raise ValueError(f"{table}:{line_no}: expected {expected_len} fields, got {len(row)}: {row!r}")


def map_value(mapping: dict[str, int], value: str, table: str, column: str, line_no: int) -> int:
    try:
        return mapping[value]
    except KeyError as exc:
        raise ValueError(f"{table}:{line_no}: unknown {column} value: {value!r}") from exc


def city_to_id(city: str, nation_id: int, table: str, line_no: int) -> int:
    # Original Crystal converter uses the last city digit:
    # city_id = nation_id * 10 + int(city[-1])
    city = city.rstrip()
    if not city:
        raise ValueError(f"{table}:{line_no}: empty city")
    if not city[-1].isdigit():
        raise ValueError(f"{table}:{line_no}: city does not end with a digit: {city!r}")
    return nation_id * 10 + int(city[-1])


def require_non_empty_primary_key(
    row: list[str],
    table: str,
    line_no: int,
    fix_empty_leading_key: bool,
) -> None:
    if row[0] != "":
        return

    if fix_empty_leading_key:
        derived = derive_key_from_name(table, row)
        if derived is not None:
            row[0] = str(derived)
            return

    raise ValueError(
        f"{table}:{line_no}: empty primary key in the first column. "
        f"Fix input data or pass --fix-empty-leading-key."
    )


def derive_key_from_name(table: str, row: list[str]) -> int | None:
    if len(row) < 2:
        return None

    patterns = {
        "customer": r"^Customer#0*([0-9]+)$",
        "supplier": r"^Supplier#0*([0-9]+)$",
    }
    pattern = patterns.get(table)
    if pattern is None:
        return None

    match = re.match(pattern, row[1])
    if match is None:
        return None

    return int(match.group(1))


def convert_supplier(row: list[str], line_no: int, args: argparse.Namespace) -> list[str]:
    row = strip_dbgen_trailing_empty(row, expected_len=7)
    require_len(row, 7, "supplier", line_no)
    require_non_empty_primary_key(row, "supplier", line_no, args.fix_empty_leading_key)

    nation_id = map_value(NATION_TO_ID, row[4], "supplier", "s_nation", line_no)
    region_id = map_value(REGION_TO_ID, row[5], "supplier", "s_region", line_no)
    city_id = city_to_id(row[3], nation_id, "supplier", line_no)

    row[3] = str(city_id)
    row[4] = str(nation_id)
    row[5] = str(region_id)
    return row


def convert_customer(row: list[str], line_no: int, args: argparse.Namespace) -> list[str]:
    row = strip_dbgen_trailing_empty(row, expected_len=8)
    require_len(row, 8, "customer", line_no)
    require_non_empty_primary_key(row, "customer", line_no, args.fix_empty_leading_key)

    nation_id = map_value(NATION_TO_ID, row[4], "customer", "c_nation", line_no)
    region_id = map_value(REGION_TO_ID, row[5], "customer", "c_region", line_no)
    city_id = city_to_id(row[3], nation_id, "customer", line_no)

    row[3] = str(city_id)
    row[4] = str(nation_id)
    row[5] = str(region_id)

    if args.encode_mktsegment:
        mktsegment_id = map_value(MKTSEGMENT_TO_ID, row[7], "customer", "c_mktsegment", line_no)
        row[7] = str(mktsegment_id)

    return row


def mfgr_to_id(value: str, table: str, line_no: int) -> int:
    try:
        return int(value.split("#")[-1]) - 1
    except Exception as exc:
        raise ValueError(f"{table}:{line_no}: invalid p_mfgr: {value!r}") from exc


def category_to_id(value: str, mfgr_id: int, table: str, line_no: int) -> int:
    try:
        suffix = int(value.split("#")[-1][-1])
        return mfgr_id * 5 + suffix - 1
    except Exception as exc:
        raise ValueError(f"{table}:{line_no}: invalid p_category: {value!r}") from exc


def brand_to_id(value: str, category_id: int, table: str, line_no: int) -> int:
    try:
        raw = value.split("#")[-1]
        brand_suffix = int(raw[2:])
        return category_id * 40 + brand_suffix - 1
    except Exception as exc:
        raise ValueError(f"{table}:{line_no}: invalid p_brand1: {value!r}") from exc


def convert_part(row: list[str], line_no: int, args: argparse.Namespace) -> list[str]:
    row = strip_dbgen_trailing_empty(row, expected_len=9)
    require_len(row, 9, "part", line_no)
    require_non_empty_primary_key(row, "part", line_no, args.fix_empty_leading_key)

    mfgr_id = mfgr_to_id(row[2], "part", line_no)
    category_id = category_to_id(row[3], mfgr_id, "part", line_no)
    brand_id = brand_to_id(row[4], category_id, "part", line_no)

    row[2] = str(mfgr_id)
    row[3] = str(category_id)
    row[4] = str(brand_id)
    return row


CONVERTERS: dict[str, Callable[[list[str], int, argparse.Namespace], list[str]]] = {
    "supplier": convert_supplier,
    "customer": convert_customer,
    "part": convert_part,
}


def convert_table(
    table: str,
    data_dir: pathlib.Path,
    args: argparse.Namespace,
    input_delimiter: str,
    output_delimiter: str,
) -> int:
    input_path = data_dir / f"{table}.tbl"
    output_path = data_dir / f"{table}.tbl.p"

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    converter = CONVERTERS[table]
    rows_written = 0

    with input_path.open("r", newline="", encoding=args.encoding) as src, \
         output_path.open("w", newline="", encoding=args.encoding) as dst:
        reader = csv.reader(src, delimiter=input_delimiter, quotechar=args.quotechar)
        writer = csv.writer(
            dst,
            delimiter=output_delimiter,
            quotechar=args.quotechar,
            lineterminator="\n",
            quoting=csv.QUOTE_MINIMAL,
        )

        for line_no, row in enumerate(reader, start=1):
            if not row:
                continue

            out_row = converter(row, line_no, args)
            out_row = add_dbgen_trailing_empty(out_row, args.trailing_delimiter)
            writer.writerow(out_row)
            rows_written += 1

    return rows_written


def write_metadata(data_dir: pathlib.Path, args: argparse.Namespace, input_delimiter: str, output_delimiter: str) -> None:
    if not args.write_metadata:
        return

    metadata = {
        "input_files": ["supplier.tbl", "customer.tbl", "part.tbl"],
        "output_files": ["supplier.tbl.p", "customer.tbl.p", "part.tbl.p"],
        "input_delimiter": input_delimiter,
        "output_delimiter": output_delimiter,
        "trailing_delimiter": args.trailing_delimiter,
        "encoded_columns": {
            "supplier": ["s_city", "s_nation", "s_region"],
            "customer": ["c_city", "c_nation", "c_region"] + (["c_mktsegment"] if args.encode_mktsegment else []),
            "part": ["p_mfgr", "p_category", "p_brand1"],
        },
        "regions": REGION_TO_ID,
        "nations": NATION_TO_ID,
        "mktsegments": MKTSEGMENT_TO_ID if args.encode_mktsegment else None,
        "formulas": {
            "supplier.s_city": "nation_id * 10 + int(last_digit(s_city))",
            "customer.c_city": "nation_id * 10 + int(last_digit(c_city))",
            "part.p_mfgr": "int(after_hash(p_mfgr)) - 1",
            "part.p_category": "p_mfgr_id * 5 + int(last_digit(p_category)) - 1",
            "part.p_brand1": "p_category_id * 40 + int(after_hash(p_brand1)[2:]) - 1",
        },
    }

    path = data_dir / "encoding_metadata.json"
    path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding=args.encoding)
    print(f"metadata: wrote {path}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Drop-in replacement for Crystal SSB converter. "
            "Reads supplier.tbl/customer.tbl/part.tbl and writes *.tbl.p."
        )
    )
    parser.add_argument("data_directory", type=str, help="Directory with ssb-dbgen .tbl files.")

    parser.add_argument(
        "--delimiter",
        default=None,
        help="Use the same delimiter for input and output. Examples: '|', ',', pipe, comma.",
    )
    parser.add_argument(
        "--input-delimiter",
        default=None,
        help="Input delimiter. Overrides --delimiter for reading.",
    )
    parser.add_argument(
        "--output-delimiter",
        default=None,
        help="Output delimiter. Overrides --delimiter for writing.",
    )
    parser.add_argument("--quotechar", default='"', help="CSV quote character.")
    parser.add_argument("--encoding", default="utf-8")

    parser.add_argument(
        "--no-trailing-delimiter",
        dest="trailing_delimiter",
        action="store_false",
        help="Do not emit the final empty field. Default keeps dbgen-style trailing delimiter.",
    )
    parser.set_defaults(trailing_delimiter=True)

    parser.add_argument(
        "--encode-mktsegment",
        action="store_true",
        help=(
            "Also encode customer.c_mktsegment. Use this if benchmark SQL uses numeric "
            "predicates like c_mktsegment = 1."
        ),
    )
    parser.add_argument(
        "--fix-empty-leading-key",
        action="store_true",
        help=(
            "Recover empty c_custkey/s_suppkey from Customer#/Supplier# name when possible. "
            "Useful for malformed CSV exports."
        ),
    )
    parser.add_argument(
        "--write-metadata",
        action="store_true",
        help="Write encoding_metadata.json next to the generated .tbl.p files.",
    )

    args = parser.parse_args()

    base_delimiter = parse_delimiter(args.delimiter, default="|")
    input_delimiter = parse_delimiter(args.input_delimiter, default=base_delimiter)
    output_delimiter = parse_delimiter(args.output_delimiter, default=base_delimiter)

    data_dir = pathlib.Path(args.data_directory)

    for table in ("supplier", "customer", "part"):
        rows = convert_table(table, data_dir, args, input_delimiter, output_delimiter)
        print(f"{table}: wrote {rows} rows -> {data_dir / (table + '.tbl.p')}")

    write_metadata(data_dir, args, input_delimiter, output_delimiter)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
