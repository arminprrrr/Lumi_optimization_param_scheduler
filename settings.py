from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import lichtfeld as lf
from lfs_plugins.props import BoolProperty, FloatProperty, IntProperty, PropertyGroup, StringProperty

PLUGIN_NAME = "Lumi_optimization_param_scheduler"
PERSIST_KEY = "param_scheduler_settings_v1_json"
LOCAL_PERSIST_PATH = Path(__file__).with_name("param_scheduler_settings.json")

PLUGIN_CONFIG_SECTION_KEYS = (
    "Lumi_optimization_param_scheduler",
    "optimization_param_scheduler",
    "param_scheduler",
)
PLUGIN_CONFIG_CONTAINER_KEYS = ("plugins", "plugin_configs", "plugin_settings")


class ParamSchedulerSettings(PropertyGroup):
    enabled = BoolProperty(
        default=True,
        name="Enabled",
        description="Enable scheduling of selected optimization parameters during training",
    )
    apply_every = IntProperty(
        default=1,
        min=1,
        max=100000,
        step=1,
        name="Apply every N steps",
        description="1 = apply every iteration; larger values reduce Python callback work",
    )
    log_every = IntProperty(
        default=500,
        min=0,
        max=1000000,
        step=100,
        name="Log every N steps",
        description="0 disables periodic scheduler log entries",
    )
    mid_ratio = FloatProperty(
        default=0.5,
        min=0.001,
        max=0.999,
        step=0.01,
        precision=6,
        name="Middle point ratio",
        description="Training progress ratio where mid values are reached",
    )
    strict_readback = BoolProperty(
        default=True,
        name="Verify readback",
        description="Read optimization params after applying values and warn if needed",
    )
    dry_run = BoolProperty(
        default=False,
        name="run without mutation",
        description="Compute and log scheduled values without mutating LichtFeld optimization params",
    )

    enable_lambda_dssim = BoolProperty(default=True, name="Schedule lambda_dssim")
    lambda_dssim_start = FloatProperty(default=0.20, min=0.0, max=1.0, step=0.01, precision=6, name="lambda_dssim start")
    lambda_dssim_mid = FloatProperty(default=0.20, min=0.0, max=1.0, step=0.01, precision=6, name="lambda_dssim mid")
    lambda_dssim_end = FloatProperty(default=0.20, min=0.0, max=1.0, step=0.01, precision=6, name="lambda_dssim end")

    enable_opacity_reg = BoolProperty(default=True, name="Schedule opacity_reg")
    opacity_reg_start = FloatProperty(default=0.0, min=0.0, max=1000000.0, step=0.001, precision=8, name="opacity_reg start")
    opacity_reg_mid = FloatProperty(default=0.0, min=0.0, max=1000000.0, step=0.001, precision=8, name="opacity_reg mid")
    opacity_reg_end = FloatProperty(default=0.0, min=0.0, max=1000000.0, step=0.001, precision=8, name="opacity_reg end")

    enable_scale_reg = BoolProperty(default=True, name="Schedule scale_reg")
    scale_reg_start = FloatProperty(default=0.0, min=0.0, max=1000000.0, step=0.001, precision=8, name="scale_reg start")
    scale_reg_mid = FloatProperty(default=0.0, min=0.0, max=1000000.0, step=0.001, precision=8, name="scale_reg mid")
    scale_reg_end = FloatProperty(default=0.0, min=0.0, max=1000000.0, step=0.001, precision=8, name="scale_reg end")

    enable_tv_loss_weight = BoolProperty(default=True, name="Schedule tv_loss_weight")
    tv_loss_weight_start = FloatProperty(default=10.0, min=0.0, max=1000000.0, step=0.1, precision=6, name="tv_loss_weight start")
    tv_loss_weight_mid = FloatProperty(default=10.0, min=0.0, max=1000000.0, step=0.1, precision=6, name="tv_loss_weight mid")
    tv_loss_weight_end = FloatProperty(default=10.0, min=0.0, max=1000000.0, step=0.1, precision=6, name="tv_loss_weight end")

    schedule_note = StringProperty(default="", maxlen=256, name="Save setting comment note")


FIELD_SPECS: dict[str, dict[str, Any]] = {
    "enabled": {"type": "bool"},
    "apply_every": {"type": "int", "min": 1, "max": 100000},
    "log_every": {"type": "int", "min": 0, "max": 1000000},
    "mid_ratio": {"type": "float", "min": 0.001, "max": 0.999},
    "strict_readback": {"type": "bool"},
    "dry_run": {"type": "bool"},
    "enable_lambda_dssim": {"type": "bool"},
    "lambda_dssim_start": {"type": "float", "min": 0.0, "max": 1.0},
    "lambda_dssim_mid": {"type": "float", "min": 0.0, "max": 1.0},
    "lambda_dssim_end": {"type": "float", "min": 0.0, "max": 1.0},
    "enable_opacity_reg": {"type": "bool"},
    "opacity_reg_start": {"type": "float", "min": 0.0, "max": 1000000.0},
    "opacity_reg_mid": {"type": "float", "min": 0.0, "max": 1000000.0},
    "opacity_reg_end": {"type": "float", "min": 0.0, "max": 1000000.0},
    "enable_scale_reg": {"type": "bool"},
    "scale_reg_start": {"type": "float", "min": 0.0, "max": 1000000.0},
    "scale_reg_mid": {"type": "float", "min": 0.0, "max": 1000000.0},
    "scale_reg_end": {"type": "float", "min": 0.0, "max": 1000000.0},
    "enable_tv_loss_weight": {"type": "bool"},
    "tv_loss_weight_start": {"type": "float", "min": 0.0, "max": 1000000.0},
    "tv_loss_weight_mid": {"type": "float", "min": 0.0, "max": 1000000.0},
    "tv_loss_weight_end": {"type": "float", "min": 0.0, "max": 1000000.0},
    "schedule_note": {"type": "string"},
}

CLI_ALIASES = {
    "lambda": "lambda_dssim_start",
    "lambda_start": "lambda_dssim_start",
    "lambda_mid": "lambda_dssim_mid",
    "lambda_end": "lambda_dssim_end",
    "lambda_dssim": "lambda_dssim_start",
    "opacity": "opacity_reg_start",
    "opacity_start": "opacity_reg_start",
    "opacity_mid": "opacity_reg_mid",
    "opacity_end": "opacity_reg_end",
    "scale": "scale_reg_start",
    "scale_start": "scale_reg_start",
    "scale_mid": "scale_reg_mid",
    "scale_end": "scale_reg_end",
    "tv": "tv_loss_weight_start",
    "tv_start": "tv_loss_weight_start",
    "tv_mid": "tv_loss_weight_mid",
    "tv_end": "tv_loss_weight_end",
}

_LAST_RUNTIME_INIT_REPORT: dict[str, Any] = {}


def _clamp(value: float, lo: float | None = None, hi: float | None = None):
    if lo is not None and value < lo:
        value = lo
    if hi is not None and value > hi:
        value = hi
    return value


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Invalid boolean value: {value}")


def _coerce_field(name: str, value: Any):
    spec = FIELD_SPECS[name]
    field_type = spec["type"]
    if field_type == "bool":
        return _parse_bool(value)
    if field_type == "int":
        return int(_clamp(int(value), spec.get("min"), spec.get("max")))
    if field_type == "float":
        return float(_clamp(float(value), spec.get("min"), spec.get("max")))
    if field_type == "string":
        return str(value)
    raise ValueError(f"Unsupported field type for {name}: {field_type}")


def settings_to_dict(settings: ParamSchedulerSettings) -> dict[str, Any]:
    return {name: getattr(settings, name) for name in FIELD_SPECS}


def apply_dict(settings: ParamSchedulerSettings, data: dict[str, Any], log_prefix: str = "settings"):
    changed = False
    for raw_name, raw_value in data.items():
        name = CLI_ALIASES.get(str(raw_name), str(raw_name))
        if name not in FIELD_SPECS:
            lf.log.warn(f"[{PLUGIN_NAME}] Unknown {log_prefix} field ignored: {raw_name}")
            continue
        try:
            value = _coerce_field(name, raw_value)
            if getattr(settings, name) != value:
                setattr(settings, name, value)
                changed = True
        except Exception as exc:
            lf.log.warn(f"[{PLUGIN_NAME}] Failed to apply {log_prefix} field {raw_name}: {exc}")
    return changed


def _get_plugin_store():
    try:
        return lf.plugins.settings(PLUGIN_NAME)
    except Exception as exc:
        lf.log.warn(f"[{PLUGIN_NAME}] Persistent settings unavailable: {exc}")
        return None


def _store_get(store, key: str, default=None):
    if store is None:
        return default
    try:
        getter = getattr(store, "get", None)
        if callable(getter):
            return getter(key, default)
    except Exception:
        pass
    try:
        return store[key]
    except Exception:
        pass
    try:
        return getattr(store, key)
    except Exception:
        return default


def _store_set(store, key: str, value):
    if store is None:
        return False
    try:
        setter = getattr(store, "set", None)
        if callable(setter):
            setter(key, value)
            return True
    except Exception:
        pass
    try:
        store[key] = value
        return True
    except Exception:
        pass
    try:
        setattr(store, key, value)
        return True
    except Exception:
        return False


def _normalize_config_blob(data: Any) -> tuple[dict[str, Any], str]:
    if not isinstance(data, dict):
        raise ValueError("Plugin config payload must be a JSON object")

    lower_keys = {str(k).strip().lower(): k for k in data.keys() if isinstance(k, str)}
    for logical_key in PLUGIN_CONFIG_SECTION_KEYS:
        actual_key = lower_keys.get(logical_key)
        if actual_key is not None and isinstance(data.get(actual_key), dict):
            return data[actual_key], actual_key

    for logical_key in PLUGIN_CONFIG_CONTAINER_KEYS:
        actual_key = lower_keys.get(logical_key)
        if actual_key is None or not isinstance(data.get(actual_key), dict):
            continue
        container = data[actual_key]
        nested_lookup = {str(k).strip().lower(): k for k in container.keys() if isinstance(k, str)}
        for nested_key in PLUGIN_CONFIG_SECTION_KEYS:
            actual_nested = nested_lookup.get(nested_key)
            if actual_nested is not None and isinstance(container.get(actual_nested), dict):
                return container[actual_nested], f"{actual_key}.{actual_nested}"

    if any(CLI_ALIASES.get(str(k), str(k)) in FIELD_SPECS for k in data.keys()):
        return data, "top_level"
    return {}, "no_plugin_fields"


def _read_json_file(path_like: os.PathLike[str] | str, source_label: str) -> tuple[dict[str, Any], str]:
    path = Path(path_like).expanduser()
    with path.open("r", encoding="utf-8") as f:
        blob = json.load(f)
    return _normalize_config_blob(blob)


def _load_local_payload() -> str:
    try:
        if LOCAL_PERSIST_PATH.exists():
            return LOCAL_PERSIST_PATH.read_text(encoding="utf-8").strip()
    except Exception as exc:
        lf.log.warn(f"[{PLUGIN_NAME}] Failed to read local settings file {LOCAL_PERSIST_PATH}: {exc}")
    return ""


def _save_local_payload(payload: str) -> bool:
    try:
        LOCAL_PERSIST_PATH.write_text(str(payload), encoding="utf-8")
        return True
    except Exception as exc:
        lf.log.warn(f"[{PLUGIN_NAME}] Failed to write local settings file {LOCAL_PERSIST_PATH}: {exc}")
        return False


def load_persistent_settings(settings: ParamSchedulerSettings):
    store = _get_plugin_store()
    payload = _store_get(store, PERSIST_KEY, "") or ""
    if not payload:
        payload = _load_local_payload()
    if not payload:
        return False

    try:
        data = json.loads(payload)
    except Exception as exc:
        lf.log.warn(f"[{PLUGIN_NAME}] Failed to parse saved settings: {exc}")
        return False
    if not isinstance(data, dict):
        lf.log.warn(f"[{PLUGIN_NAME}] Saved settings payload must be a dict")
        return False
    return bool(apply_dict(settings, data, log_prefix="saved") or data)


def save_persistent_settings(settings: ParamSchedulerSettings):
    try:
        payload = json.dumps(settings_to_dict(settings), sort_keys=True)
    except Exception as exc:
        lf.log.warn(f"[{PLUGIN_NAME}] Failed to serialize settings: {exc}")
        return False

    store = _get_plugin_store()
    store_ok = _store_set(store, PERSIST_KEY, payload) if store is not None else False
    file_ok = _save_local_payload(payload)
    if store is not None and not store_ok:
        lf.log.warn(f"[{PLUGIN_NAME}] Failed to save persistent settings to plugin store")
    if not store_ok and not file_ok:
        lf.log.warn(f"[{PLUGIN_NAME}] Failed to save persistent settings")
    return bool(store_ok or file_ok)


def _normalize_cli_name(name: str) -> str:
    return name.strip().lower().replace("-", "_")


def collect_runtime_overrides(argv: list[str] | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    argv = list(sys.argv[1:] if argv is None else argv)
    overrides: dict[str, Any] = {}
    sources: list[str] = []
    override_keys: list[str] = []

    def merge_blob(blob: dict[str, Any], source: str):
        if not isinstance(blob, dict) or not blob:
            return
        overrides.update(blob)
        sources.append(source)
        for key in blob.keys():
            normalized = CLI_ALIASES.get(str(key), str(key))
            if normalized not in override_keys:
                override_keys.append(normalized)

    for idx, arg in enumerate(argv):
        raw = str(arg).strip().strip('"')
        if raw.startswith("--lf-param-scheduler-file="):
            path = raw.split("=", 1)[1].strip().strip('"')
            try:
                blob, section = _read_json_file(path, "CLI config file")
                merge_blob(blob, f"cli_file:{path}#{section}")
            except Exception as exc:
                lf.log.warn(f"[{PLUGIN_NAME}] Failed to read CLI config file {path}: {exc}")
        elif raw == "--lf-param-scheduler-file" and idx + 1 < len(argv):
            path = str(argv[idx + 1]).strip().strip('"')
            try:
                blob, section = _read_json_file(path, "CLI config file")
                merge_blob(blob, f"cli_file:{path}#{section}")
            except Exception as exc:
                lf.log.warn(f"[{PLUGIN_NAME}] Failed to read CLI config file {path}: {exc}")
        elif raw.startswith("--lf-param-scheduler-"):
            payload = raw[len("--lf-param-scheduler-"):]
            if payload.startswith("no-"):
                key = _normalize_cli_name(payload[3:])
                if key in FIELD_SPECS and FIELD_SPECS[key]["type"] == "bool":
                    merge_blob({key: False}, f"cli_flag:{key}=false")
                continue
            if "=" in payload:
                key, value = payload.split("=", 1)
                merge_blob({_normalize_cli_name(key): value}, f"cli_value:{key}")
            else:
                key = _normalize_cli_name(payload)
                if key in FIELD_SPECS and FIELD_SPECS[key]["type"] == "bool":
                    merge_blob({key: True}, f"cli_flag:{key}=true")

    env_json = os.environ.get("LF_PARAM_SCHEDULER_JSON")
    if env_json:
        try:
            blob, section = _normalize_config_blob(json.loads(env_json))
            merge_blob(blob, f"env_json#{section}")
        except Exception as exc:
            lf.log.warn(f"[{PLUGIN_NAME}] Failed to parse env JSON overrides: {exc}")

    env_file = os.environ.get("LF_PARAM_SCHEDULER_FILE")
    if env_file:
        try:
            blob, section = _read_json_file(env_file, "env config file")
            merge_blob(blob, f"env_file:{env_file}#{section}")
        except Exception as exc:
            lf.log.warn(f"[{PLUGIN_NAME}] Failed to read env config file {env_file}: {exc}")

    return overrides, {"sources": sources, "override_keys": override_keys}


def get_last_runtime_init_report() -> dict[str, Any]:
    return dict(_LAST_RUNTIME_INIT_REPORT)


def initialize_runtime_settings(settings: ParamSchedulerSettings | None = None):
    global _LAST_RUNTIME_INIT_REPORT
    settings = settings or ParamSchedulerSettings.get_instance()

    persistent_loaded = load_persistent_settings(settings)
    overrides, meta = collect_runtime_overrides()
    if overrides:
        apply_dict(settings, overrides, log_prefix="runtime")
        save_persistent_settings(settings)

    _LAST_RUNTIME_INIT_REPORT = {
        "persistent_loaded": bool(persistent_loaded),
        "sources": list(meta.get("sources", [])),
        "override_keys": list(meta.get("override_keys", [])),
        "effective_settings": settings_to_dict(settings),
    }

    source_bits = []
    if persistent_loaded:
        source_bits.append("persistent_settings")
    source_bits.extend(_LAST_RUNTIME_INIT_REPORT["sources"])
    source_text = ", ".join(source_bits) if source_bits else "defaults_only"
    lf.log.info(f"[{PLUGIN_NAME}] Runtime config sources: {source_text}")
    if _LAST_RUNTIME_INIT_REPORT["override_keys"]:
        lf.log.info(f"[{PLUGIN_NAME}] Runtime override keys: {', '.join(_LAST_RUNTIME_INIT_REPORT['override_keys'])}")
    return settings
