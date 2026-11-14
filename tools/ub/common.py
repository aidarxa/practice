"""Shared correctness-only command runner and explicit build recipes (stdlib only)."""
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import shutil
import subprocess
import sys


class ToolError(ValueError):
    """Invalid input or an infrastructure failure, never a successful quality gate."""


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def positive(value):
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ToolError("Expected a finite positive number")
    return result


class Session:
    """Own a fresh artifact directory; archive commands without shell interpretation."""
    def __init__(self, output):
        self.output = Path(output).absolute()
        self.output.mkdir(parents=True, exist_ok=False)
        self.event("start", reason="Exclusive directory prevents stale results and concurrent writers",
                   environment={key: os.environ[key] for key in ("CXX", "CXXFLAGS", "ACPP_TARGETS") if key in os.environ})

    def event(self, action, **details):
        with (self.output / "actions.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"action": action, **details}, ensure_ascii=False) + "\n")

    def run(self, command, cwd, label, timeout=120, env=None):
        timeout = positive(timeout)
        command = [str(arg) for arg in command]
        prefix = self.output / label
        prefix.parent.mkdir(parents=True, exist_ok=True)
        metadata = {"argv": command, "cwd": str(cwd), "environment": {"LC_ALL": "C", **(env or {})}}
        write_json(str(prefix) + ".command.json", metadata)
        self.event("command", reason="Preserve reproducible correctness checks, not performance timings", **metadata)
        with Path(str(prefix) + ".stdout").open("wb") as out, Path(str(prefix) + ".stderr").open("wb") as err:
            try:
                process = subprocess.Popen(command, cwd=cwd, env={**os.environ, **metadata["environment"]},
                                           stdout=out, stderr=err, start_new_session=True)
            except OSError as error:
                raise ToolError(f"Cannot execute {command[0]}: {error}") from error
            try:
                code = process.wait(timeout=timeout)
                result = {"status": "passed" if code == 0 else "failed", "returncode": code}
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass  # The group exited between the deadline and killpg.
                process.wait()
                result = {"status": "timeout", "returncode": None}
        write_json(str(prefix) + ".result.json", result)
        self.event("result", label=label, **result)
        return result

    def checked(self, command, cwd, label, timeout=120, env=None, strict_stderr=False):
        result = self.run(command, cwd, label, timeout, env)
        stderr = Path(str(self.output / label) + ".stderr")
        if result["status"] != "passed" or (strict_stderr and stderr.read_bytes().strip()):
            raise ToolError(f"{label} failed or emitted forbidden diagnostics; see {stderr}")
        return self.output / (label + ".stdout")

    def finish(self, report):
        write_json(self.output / "report.json", report)
        self.event("finish", success=report["success"])
        print(f"success={report['success']}; report: {self.output / 'report.json'}")
        return 0 if report["success"] else 1


def entry(main):
    """CLI convention: 0 pass, 1 completed but failed gate, 2 invalid/incomplete run."""
    try:
        return main()
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


def relative_path(value):
    if not isinstance(value, str):
        raise ToolError("Project-relative paths must be strings")
    path = Path(value)
    if not value or path.is_absolute() or ".." in path.parts or path == Path("."):
        raise ToolError(f"Expected a nonempty project-relative path: {value}")
    return path


def load_recipe(path, project):
    recipe = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(recipe, dict) or type(recipe.get("version")) is not int or recipe["version"] != 1:
        raise ToolError("Recipe must have version: 1")
    for phase in ("build", "test"):
        commands = recipe.get(phase)
        if not isinstance(commands, list) or not commands:
            raise ToolError(f"Recipe requires nonempty {phase} commands")
        if any(not isinstance(cmd, list) or not cmd or not all(isinstance(arg, str) for arg in cmd) for cmd in commands):
            raise ToolError(f"{phase} commands must be nonempty argv arrays, not shell strings")
    inputs = recipe.get("inputs")
    if not isinstance(inputs, list) or not inputs:
        raise ToolError("Recipe requires explicit inputs for isolation and provenance")
    files = set()
    for value in inputs:
        relative = relative_path(value)
        source = project / relative
        if source.resolve() != source.absolute() or not source.exists():
            raise ToolError(f"Missing/symlink input is not supported: {source}")
        candidates = [source] if source.is_file() else list(source.rglob("*"))
        for candidate in candidates:
            if candidate.is_symlink():
                raise ToolError(f"Symlink input is not isolated: {candidate}")
            if candidate.is_file():
                files.add(str(candidate.relative_to(project)))
    if not files:
        raise ToolError("Recipe inputs contain no files")
    return recipe, {name: digest(project / name) for name in sorted(files)}


def snapshot(project, files, destination):
    destination.mkdir(parents=True)
    for relative, expected in files.items():
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(project / relative, target)
        if digest(target) != expected:
            raise ToolError(f"Input changed while copying: {relative}")


def expand(command, root, out, cxx, flags=()):
    values = {"root": str(root), "out": str(out), "cxx": cxx}
    result = []
    for arg in command:
        if arg == "{flags}":
            result.extend(flags)
            continue
        if "{flags}" in arg:
            raise ToolError("{flags} must occupy a complete argv element")
        for key, value in values.items():
            arg = arg.replace("{" + key + "}", value)
        if re.search(r"\{(?:root|out|cxx|flags)\}", arg):
            raise ToolError(f"Unexpanded placeholder: {arg}")
        result.append(arg)
    return result


def run_phase(session, recipe, phase, root, out, cxx, timeout, flags=(), env=None, label=""):
    for index, command in enumerate(recipe[phase]):
        result = session.run(expand(command, root, out, cxx, flags), root,
                             f"{label}{phase}-{index}", timeout=timeout, env=env)
        if result["status"] != "passed":
            return result
    return {"status": "passed", "returncode": 0}
