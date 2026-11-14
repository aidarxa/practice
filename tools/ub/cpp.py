"""Clang AST source identities, C++ lexical tokens and decision counting."""
import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import re
import shlex

from common import ToolError, digest, positive

FUNCTIONS = {"FunctionDecl", "CXXMethodDecl", "CXXConstructorDecl", "CXXDestructorDecl", "CXXConversionDecl"}
BODIES = {"CompoundStmt", "CXXTryStmt"}
DECISIONS = {"IfStmt", "WhileStmt", "DoStmt", "ForStmt", "CXXForRangeStmt", "CaseStmt",
             "ConditionalOperator", "BinaryConditionalOperator", "CXXCatchStmt"}
# Ordered alternatives: comments inside normal/raw literals are never comments.
LEXER = re.compile(
    r'(?P<space>\s+)|(?P<comment>//[^\n]*|/\*.*?\*/)|'
    r'(?P<raw>(?:u8|u|U|L)?R"(?P<delimiter>[^\s()\\]{0,16})\(.*?\)(?P=delimiter)"\w*)|'
    r'''(?P<literal>(?:u8|u|U|L)?(?:"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*')\w*)|'''
    r"(?P<number>(?:\d|\.\d)(?:[eEpP][+-]|[\w.'])*)|(?P<word>[^\W\d]\w*)|"
    r"(?P<operator><=>|>>=|<<=|->\*|\.\.\.|::|->|\.\*|\+\+|--|&&|\|\||[+*/%&|^!<>=-]=|<<|>>|[^\s])",
    re.DOTALL,
)


def tokens(text):
    return [match.group() for match in LEXER.finditer(text) if match.lastgroup not in ("space", "comment")]


def json_stream(text):
    decoder = json.JSONDecoder()
    position = 0
    while position < len(text):
        if text[position].isspace():
            position += 1
            continue
        value, position = decoder.raw_decode(text, position)
        yield value


def complexity(node):
    # Nested functions/lambdas are measured under their own identity, not twice.
    if node.get("kind") in FUNCTIONS | {"LambdaExpr"}:
        return 0
    own = node.get("kind") in DECISIONS or (node.get("kind") == "BinaryOperator" and node.get("opcode") in ("&&", "||"))
    return int(own) + sum(complexity(child) for child in node.get("inner", []) if child)


def locations(node):
    return [node.get("loc", {}), node.get("range", {}).get("begin", {}), node.get("range", {}).get("end", {})]


def annotate(node, directory, state):
    """Clang elides repeated file names across sibling nodes in serialized order."""
    for location in locations(node):
        location = location.get("expansionLoc", location)
        if "file" in location:
            state[0] = str((directory / location["file"]).resolve())
    node["_source"] = state[0]
    for child in node.get("inner", []):
        if child:
            annotate(child, directory, state)


def walk(node):
    yield node
    for child in node.get("inner", []):
        if child:
            yield from walk(child)


def extent(node):
    bounds = node.get("range", {})
    begin, end = bounds.get("begin", {}), bounds.get("end", {})
    if "offset" not in begin or "offset" not in end:
        return None  # Macro spelling/expansion ranges cannot be edited reliably.
    return begin["offset"], end["offset"] + end.get("tokLen", 0)


@dataclass
class Function:
    source: Path
    node: dict
    body: dict
    start: int
    end: int
    line: int
    text: str

    def public(self):
        return {"source": str(self.source), "name": self.node.get("name", "<anonymous>"),
                "mangled": self.node.get("mangledName"), "line": self.line,
                "start": self.start, "end": self.end}


def definitions(roots, sources, selected):
    encoded = {str(path): path.read_bytes() for path in sources}
    result = {}
    for root in roots:
        for node in walk(root):
            if (node.get("kind") not in FUNCTIONS or node.get("isImplicit")
                    or node.get("explicitlyDefaulted") or node.get("_source") not in encoded):
                continue
            if selected and node.get("name") not in selected:
                continue
            body = next((child for child in node.get("inner", []) if child.get("kind") in BODIES), None)
            if body is None:
                continue
            bounds = extent(body)
            if not bounds:
                raise ToolError(f"Macro-defined function body cannot be measured reliably: {node.get('name')}")
            start, end = bounds
            data = encoded[node["_source"]]
            if not (0 <= start < end <= len(data)) or data[end - 1:end] != b"}":
                raise ToolError(f"Unsupported source range for {node.get('name')}")
            identity = (node["_source"], start, end, node.get("mangledName"))
            result[identity] = Function(Path(node["_source"]), node, body, start, end,
                                        data.count(b"\n", 0, start) + 1, data[start:end].decode("utf-8"))
    # A template's pattern is not a second uncovered function beside its instantiations.
    instantiated = {key[:3] for key in result if key[3]}
    return [function for key, function in result.items() if key[3] or key[:3] not in instantiated]


def compile_flags(database, tu, index=None):
    path = database / "compile_commands.json" if database.is_dir() else database
    entries = json.loads(path.read_text(encoding="utf-8"))
    matches = [entry for entry in entries if (Path(entry["directory"]) / entry["file"]).resolve() == tu]
    if not matches or (index is None and len(matches) != 1):
        raise ToolError(f"Expected one compile command for {tu}, got {len(matches)}; use --compile-entry N or --flag")
    if index is not None and not 0 <= index < len(matches):
        raise ToolError(f"--compile-entry out of range for {tu}")
    record = matches[index or 0]
    argv = record.get("arguments") or shlex.split(record["command"])
    directory = Path(record["directory"]).resolve()
    result = []
    skip = False
    for arg in argv[1:]:
        if skip:
            skip = False
            continue
        if arg in ("-o", "-MF", "-MT", "-MQ", "-MJ", "--serialize-diagnostics"):
            skip = True
        elif arg in ("-c", "-S", "-MD", "-MMD", "-MP", "-MG", "-M", "-MM"):
            continue
        elif arg.startswith("@"):
            raise ToolError("Expand response files before supplying the compilation database")
        elif not arg.startswith("-") and (directory / arg).resolve() == tu:
            continue
        elif re.match(r"^-(?:MF|MT|MQ|MJ).+", arg):
            continue
        else:
            result.append(arg)
    return directory, result


def analysis_parser(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--source", type=Path, action="append", required=True)
    parser.add_argument("--tu", type=Path, help="Parse this TU when analyzing a header")
    parser.add_argument("--compile-db", type=Path)
    parser.add_argument("--compile-entry", type=int, help="Select among multiple entries for the TU (zero-based)")
    parser.add_argument("--clang", default="clang++")
    parser.add_argument("--flag", action="append", default=[], help="Repeat as --flag=-Ipath")
    parser.add_argument("--ast-filter", help="Optional Clang qualified-name substring filter")
    parser.add_argument("--function", action="append", default=[], help="Exact unqualified function name; repeatable")
    parser.add_argument("--timeout", type=positive, default=120)
    parser.add_argument("--output", type=Path, required=True, help="Must not already exist")
    return parser


def collect(args, session):
    sources = sorted({path.resolve(strict=True) for path in args.source})
    units = [args.tu.resolve(strict=True)] if args.tu else sources
    roots = []
    hashes = {str(path): digest(path) for path in sources}
    session.checked([args.clang, "--version"], Path.cwd(), "clang-version")
    for index, tu in enumerate(units):
        directory, flags = compile_flags(args.compile_db, tu, args.compile_entry) if args.compile_db else (Path.cwd(), [])
        command = [args.clang, *flags, *args.flag, "-std=c++17", "-x", "c++", str(tu),
                   "-fsyntax-only", "-o", str(session.output / f"syntax-{index}.unused"),
                   "-Xclang", "-ast-dump=json"]
        if args.ast_filter:
            command += ["-Xclang", "-ast-dump-filter=" + args.ast_filter]
        path = session.checked(command, directory, f"ast-{index}", timeout=args.timeout)
        state = [None]
        for root in json_stream(path.read_text(encoding="utf-8")):
            annotate(root, directory, state)
            roots.append(root)
    if any(digest(path) != expected for path, expected in hashes.items()):
        raise ToolError("Sources changed during AST collection")
    functions = definitions(roots, sources, set(args.function))
    if not functions or any(not any(fn.source == source for fn in functions) for source in sources):
        raise ToolError("Empty analysis scope for at least one --source; check --tu/--ast-filter/--function")
    if set(args.function) - {fn.node.get("name") for fn in functions}:
        raise ToolError("At least one requested --function was not found")
    session.event("scope", reason="Measure only explicitly selected C++17 source definitions", sha256=hashes,
                  functions=len(functions))
    return functions
