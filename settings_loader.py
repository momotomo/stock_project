import copy
import logging
import os
import re
from typing import Any, Dict, Optional

try:
    import yaml  # type: ignore
except ModuleNotFoundError:
    yaml = None


DEFAULT_SETTINGS: Dict[str, Any] = {
    "IS_PRODUCTION": False,
    "API_PASSWORD": "",
    "API_PASSWORD_SIM": "",
    "API_PASSWORD_PROD": "",
    "TRADE_PASSWORD": "",
}


def as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    return bool(value)


def as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _strip_inline_comment(value: str) -> str:
    in_single_quote = False
    in_double_quote = False
    result = []

    for char in value:
        if char == "'" and not in_double_quote:
            in_single_quote = not in_single_quote
        elif char == '"' and not in_single_quote:
            in_double_quote = not in_double_quote
        elif char == "#" and not in_single_quote and not in_double_quote:
            break
        result.append(char)

    return "".join(result).strip()


def _parse_scalar(value: str) -> Any:
    cleaned = _strip_inline_comment(value)
    if cleaned == "":
        return ""

    lower = cleaned.lower()
    if lower in {"true", "yes", "on"}:
        return True
    if lower in {"false", "no", "off"}:
        return False
    if lower in {"null", "none", "~"}:
        return None

    if (cleaned.startswith('"') and cleaned.endswith('"')) or (
        cleaned.startswith("'") and cleaned.endswith("'")
    ):
        return cleaned[1:-1]

    if re.fullmatch(r"[+-]?\d+", cleaned):
        return int(cleaned)
    if re.fullmatch(r"[+-]?(?:\d+\.\d*|\d*\.\d+)", cleaned):
        return float(cleaned)

    return cleaned


def _parse_simple_yaml(text: str) -> Dict[str, Any]:
    parsed: Dict[str, Any] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if raw_line[:1].isspace():
            continue
        if ":" not in line:
            continue

        key, raw_value = line.split(":", 1)
        key = key.strip()
        if not key:
            continue
        parsed[key] = _parse_scalar(raw_value.strip())
    return parsed


def _read_yaml(path: str, logger: Optional[logging.Logger] = None) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}

    try:
        with open(path, "r", encoding="utf-8") as handle:
            content = handle.read()

        if yaml is not None:
            data = yaml.safe_load(content) or {}
        else:
            data = _parse_simple_yaml(content)
        if not isinstance(data, dict):
            raise ValueError("root must be a mapping")
        if logger:
            logger.info(f"⚙️ 設定ファイル '{path}' を読み込みました。")
        return data
    except Exception as exc:
        if logger:
            logger.error(f"❌ 設定ファイル '{path}' の読み込みに失敗しました: {exc}")
        return {}


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_settings(
    defaults: Optional[Dict[str, Any]] = None,
    logger: Optional[logging.Logger] = None,
    settings_path: str = "settings.yml",
    local_settings_path: str = "settings.local.yml",
) -> Dict[str, Any]:
    settings = _deep_merge({}, defaults or DEFAULT_SETTINGS)
    settings = _deep_merge(settings, _read_yaml(settings_path, logger=logger))
    settings = _deep_merge(settings, _read_yaml(local_settings_path, logger=logger))
    return settings


def resolve_api_password(settings: Dict[str, Any], is_production: Optional[bool] = None) -> str:
    if is_production is None:
        is_production = as_bool(settings.get("IS_PRODUCTION"), False)

    if is_production:
        return str(settings.get("API_PASSWORD_PROD") or settings.get("API_PASSWORD") or "")
    return str(settings.get("API_PASSWORD_SIM") or settings.get("API_PASSWORD") or "")
