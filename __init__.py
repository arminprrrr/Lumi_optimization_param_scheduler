from __future__ import annotations

import lichtfeld as lf

from .scheduler import install_scheduler, uninstall_scheduler, set_runtime_mode
from .settings import ParamSchedulerSettings, initialize_runtime_settings, save_persistent_settings

_PLUGIN_NAME = "Lumi_optimization_param_scheduler"
_REGISTERED_CLASSES = []
_SCHEDULER_INSTALLED = False


def _is_headless_ui_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return (
        "retained panel features require the retained ui manager" in msg
        or "unavailable in this runtime" in msg
        or "ui" in msg and "headless" in msg
    )


def _get_panel_classes():
    try:
        from .main_panel import OptimizationParamSchedulerPanel
    except Exception as exc:
        lf.log.warn(f"[{_PLUGIN_NAME}] Panel import failed; continuing without UI: {exc}")
        return []
    return [OptimizationParamSchedulerPanel]


def _register_optional_ui():
    global _REGISTERED_CLASSES
    for cls in _get_panel_classes():
        try:
            lf.register_class(cls)
            _REGISTERED_CLASSES.append(cls)
        except Exception as exc:
            if _is_headless_ui_error(exc):
                lf.log.info(f"[{_PLUGIN_NAME}] UI panel registration skipped in headless runtime.")
                continue
            raise


def on_load():
    global _SCHEDULER_INSTALLED
    settings = initialize_runtime_settings(ParamSchedulerSettings.get_instance())
    _register_optional_ui()

    mode = "ui" if _REGISTERED_CLASSES else "headless"
    set_runtime_mode(mode)
    install_scheduler(settings=settings, runtime_mode=mode)
    _SCHEDULER_INSTALLED = True

    lf.log.info(f"[{_PLUGIN_NAME}] Optimization parameter scheduler hooks installed")
    lf.log.info(f"{_PLUGIN_NAME} loaded ({mode})")


def on_unload():
    global _SCHEDULER_INSTALLED, _REGISTERED_CLASSES

    try:
        save_persistent_settings(ParamSchedulerSettings.get_instance())
    except Exception as exc:
        lf.log.warn(f"{_PLUGIN_NAME}: failed to save settings on unload: {exc}")

    if _SCHEDULER_INSTALLED:
        try:
            uninstall_scheduler()
        except Exception as exc:
            lf.log.warn(f"{_PLUGIN_NAME}: failed to uninstall scheduler: {exc}")
        _SCHEDULER_INSTALLED = False

    for cls in reversed(_REGISTERED_CLASSES):
        try:
            lf.unregister_class(cls)
        except Exception as exc:
            if not _is_headless_ui_error(exc):
                lf.log.warn(f"{_PLUGIN_NAME}: failed to unregister {cls}: {exc}")
    _REGISTERED_CLASSES = []
    lf.log.info(f"{_PLUGIN_NAME} unloaded")
