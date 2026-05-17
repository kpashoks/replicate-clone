import json
import re
from copy import deepcopy
from pathlib import Path
from typing import Any

from config import REPO_ROOT


WORKFLOWS_DIR = REPO_ROOT / "runpod" / "workflows"

# Matches __SOME_KEY__ anywhere inside a string. Used for embedded
# substitution (e.g. coordinates inside a JSON string literal).
# Non-greedy on the capture so that patterns like
# "__OUTPUT_PREFIX___zzz_debug_mask" (triple underscore between the
# placeholder close and a literal suffix) match OUTPUT_PREFIX rather
# than greedy-extending into OUTPUT_PREFIX_ and failing the param
# lookup.
_EMBEDDED_PLACEHOLDER = re.compile(r"__([A-Z][A-Z0-9_]*?)__")


def _substitute(node: Any, params: dict) -> Any:
    """Walk a parsed workflow dict; replace __KEY__ placeholders with values from params.

    Two modes:
    - Whole-string placeholder (e.g. "__STEPS__"): replaced with the typed
      value (int, float, etc.) so KSampler.steps stays an int, not a string.
    - Embedded placeholder (e.g. "[{\"x\":__SEED_X__}]"): replaced with the
      str()-coerced value. Required for things like SAM2 seed coords that
      live inside a JSON string widget.
    """
    if isinstance(node, dict):
        return {k: _substitute(v, params) for k, v in node.items()}
    if isinstance(node, list):
        return [_substitute(v, params) for v in node]
    if isinstance(node, str):
        # Whole-string placeholder -> typed substitution.
        if len(node) > 4 and node.startswith("__") and node.endswith("__"):
            key = node[2:-2].lower()
            if key in params:
                return params[key]
        # Embedded placeholder(s) inside a larger string -> str substitution.
        # Only applies if there's at least one __KEY__ somewhere in the string.
        if "__" in node and _EMBEDDED_PLACEHOLDER.search(node):
            def _repl(m: re.Match) -> str:
                key = m.group(1).lower()
                return str(params[key]) if key in params else m.group(0)
            return _EMBEDDED_PLACEHOLDER.sub(_repl, node)
    return node


def load_template(workflow_file: str) -> dict:
    path = WORKFLOWS_DIR / workflow_file
    if not path.exists():
        raise FileNotFoundError(f"Workflow template not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_workflow(workflow_file: str, params: dict) -> dict:
    template = load_template(workflow_file)
    return _substitute(deepcopy(template), params)
