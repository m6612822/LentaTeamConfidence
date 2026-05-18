"""Config loader: deep-merge a YAML file over packaged defaults."""
from __future__ import annotations

import copy
import os
from typing import Any, Dict

import yaml

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_CONFIG = os.path.join(_ROOT, "config", "default.yaml")
PATTERNS_FILE = os.path.join(_ROOT, "config", "patterns.yaml")


def _deep_merge(base: Dict[str, Any], over: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def load_config(path: str | None = None) -> Dict[str, Any]:
    cfg = _load_yaml(DEFAULT_CONFIG)
    if path and os.path.abspath(path) != os.path.abspath(DEFAULT_CONFIG):
        cfg = _deep_merge(cfg, _load_yaml(path))
    cfg["patterns"] = _load_yaml(PATTERNS_FILE)
    return cfg
