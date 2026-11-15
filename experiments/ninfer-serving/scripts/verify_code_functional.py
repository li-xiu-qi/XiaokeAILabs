#!/usr/bin/env python3
"""Run functional checks for reviewed code patterns produced by quality_probe.py."""
import argparse
import ast
import builtins
import json
import subprocess
import sys
from pathlib import Path

from quality_probe import strip_code_fence

TESTS = {
    "code_palindrome": {
        "function": "is_palindrome",
        "cases": [
            (["A man, a plan, a canal: Panama"], True),
            (["race a car"], False),
            ([""], True),
            (["12321"], True),
            (["1234"], False),
        ],
    },
    "code_binary_search": {
        "function": "binary_search",
        "cases": [
            ([[1, 3, 5, 7], 5], 2),
            ([[1, 3, 5, 7], 1], 0),
            ([[1, 3, 5, 7], 7], 3),
            ([[1, 3, 5, 7], 4], -1),
            ([[], 1], -1),
        ],
    },
}
SAFE_FUNCTION_NAMES = frozenset(
    {
        "abs",
        "all",
        "any",
        "bool",
        "enumerate",
        "float",
        "int",
        "len",
        "list",
        "max",
        "min",
        "range",
        "reversed",
        "sorted",
        "str",
        "sum",
        "tuple",
        "zip",
    }
)
SAFE_FUNCTIONS = {name: getattr(builtins, name) for name in SAFE_FUNCTION_NAMES}
SAFE_METHODS = frozenset({"append", "casefold", "isalnum", "isalpha", "isdigit", "join", "lower"})
ALLOWED_NODES = (
    ast.Module,
    ast.FunctionDef,
    ast.arguments,
    ast.arg,
    ast.Assign,
    ast.AugAssign,
    ast.Name,
    ast.Load,
    ast.Store,
    ast.Constant,
    ast.Return,
    ast.If,
    ast.While,
    ast.For,
    ast.Break,
    ast.Continue,
    ast.Pass,
    ast.Expr,
    ast.Compare,
    ast.BoolOp,
    ast.BinOp,
    ast.UnaryOp,
    ast.Call,
    ast.Attribute,
    ast.Subscript,
    ast.Slice,
    ast.IfExp,
    ast.List,
    ast.Tuple,
    ast.Dict,
    ast.Set,
    ast.GeneratorExp,
    ast.comprehension,
    ast.keyword,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.FloorDiv,
    ast.Mod,
    ast.Eq,
    ast.NotEq,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
    ast.Is,
    ast.IsNot,
    ast.In,
    ast.NotIn,
    ast.And,
    ast.Or,
    ast.Not,
    ast.USub,
    ast.UAdd,
)


class CodeRejected(ValueError):
    """Raised when generated code leaves the small executable subset."""


def validate_code(code, function_name):
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        raise CodeRejected(f"invalid Python: {exc}") from exc
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        raise CodeRejected("expected exactly one top-level function")
    function = tree.body[0]
    if function.name != function_name:
        raise CodeRejected(f"expected function {function_name}, got {function.name}")
    if function.decorator_list:
        raise CodeRejected("decorators are not allowed")

    for node in ast.walk(tree):
        if not isinstance(node, ALLOWED_NODES):
            raise CodeRejected(f"unsupported syntax: {type(node).__name__}")
        if isinstance(node, ast.Name) and node.id.startswith("_"):
            raise CodeRejected("private and dunder names are not allowed")
        if isinstance(node, ast.Attribute) and node.attr not in SAFE_METHODS:
            raise CodeRejected(f"unsupported attribute: {node.attr}")
        if isinstance(node, ast.Expr):
            if not isinstance(node.value, ast.Constant) or not isinstance(node.value.value, str):
                raise CodeRejected("only function docstrings are allowed as expressions")
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                allowed = node.func.id in SAFE_FUNCTION_NAMES or node.func.id == function_name
                if not allowed:
                    raise CodeRejected(f"unsupported function call: {node.func.id}")
            elif not isinstance(node.func, ast.Attribute):
                raise CodeRejected("indirect function calls are not allowed")
    return tree


def limit_worker_resources():
    try:
        import resource
    except ImportError:
        return
    memory_limit = 256 * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_AS, (memory_limit, memory_limit))
    resource.setrlimit(resource.RLIMIT_CPU, (5, 5))


def worker():
    limit_worker_resources()
    payload = json.loads(sys.stdin.read())
    case_id = payload["case_id"]
    specification = TESTS[case_id]
    tree = validate_code(payload["code"], specification["function"])
    namespace = {"__builtins__": SAFE_FUNCTIONS}
    exec(compile(tree, "<model-generated>", "exec"), namespace)
    function = namespace[specification["function"]]

    outcomes = []
    failures = []
    for index, (arguments, expected) in enumerate(specification["cases"]):
        try:
            actual = function(*arguments)
            passed = actual == expected
            error = None if passed else f"expected {expected!r}, got {actual!r}"
        except Exception as exc:
            passed = False
            error = f"{type(exc).__name__}: {exc}"
        outcomes.append(passed)
        if not passed:
            failures.append({"case": index + 1, "error": error})
    print(
        json.dumps(
            {
                "tests": len(outcomes),
                "passed": sum(outcomes),
                "all_passed": all(outcomes),
                "failures": failures,
            }
        )
    )


def failure_result(case_id, message):
    return {
        "tests": len(TESTS[case_id]["cases"]),
        "passed": 0,
        "all_passed": False,
        "execution_completed": False,
        "error": message,
        "review": "AST allowlist, restricted builtins, subprocess resource limits",
    }


def run_case(case_id, code, timeout):
    try:
        completed = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--worker"],
            input=json.dumps({"case_id": case_id, "code": code}),
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return failure_result(case_id, f"execution exceeded {timeout} seconds")
    if completed.returncode != 0:
        error_lines = completed.stderr.strip().splitlines()
        message = error_lines[-1] if error_lines else f"worker exited with {completed.returncode}"
        return failure_result(case_id, message)
    try:
        result = json.loads(completed.stdout)
    except json.JSONDecodeError:
        return failure_result(case_id, "worker did not return a JSON result")
    result["execution_completed"] = True
    result["review"] = "AST allowlist, restricted builtins, subprocess resource limits"
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Execute the two coding cases from a quality probe with restricted syntax."
    )
    parser.add_argument("--quality", required=True, help="raw quality_probe.py JSON")
    parser.add_argument("--output", required=True)
    parser.add_argument("--timeout", type=float, default=10)
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout must be positive")

    quality = json.loads(Path(args.quality).read_text(encoding="utf-8"))
    code_cases = {}
    for case in quality.get("cases", []):
        if case.get("kind") == "python_syntax":
            if case["id"] in code_cases:
                raise SystemExit(f"duplicate code case: {case['id']}")
            code_cases[case["id"]] = case
    if set(code_cases) != set(TESTS):
        missing = sorted(set(TESTS) - set(code_cases))
        extra = sorted(set(code_cases) - set(TESTS))
        raise SystemExit(f"expected both coding cases; missing={missing}, extra={extra}")

    results = {}
    for case_id in TESTS:
        content = code_cases[case_id].get("response", {}).get("content") or ""
        code = strip_code_fence(content)
        results[case_id] = run_case(case_id, code, args.timeout)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"saved {output}")


if __name__ == "__main__":
    if sys.argv[1:] == ["--worker"]:
        worker()
    else:
        main()
