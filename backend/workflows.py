import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from config import REPO_ROOT


WORKFLOWS_DIR = REPO_ROOT / "runpod" / "workflows"


def _substitute(node: Any, params: dict) -> Any:
    """Walk a parsed workflow dict; replace __KEY__ string placeholders with typed values from params."""
    if isinstance(node, dict):
        return {k: _substitute(v, params) for k, v in node.items()}
    if isinstance(node, list):
        return [_substitute(v, params) for v in node]
    if isinstance(node, str) and len(node) > 4 and node.startswith("__") and node.endswith("__"):
        key = node[2:-2].lower()
        if key in params:
            return params[key]
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
