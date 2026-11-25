from __future__ import annotations

import lichtfeld as lf

from .scheduler import install_scheduler, uninstall_scheduler, set_runtime_mode
from .settings import ParamSchedulerSettings, initialize_runtime_settings, save_persistent_settings

_PLUGIN_NAME = "Lumi_optimization_param_scheduler"
_REGISTERED_CLASSES = []
_SCHEDULER_INSTALLED = False


def _is_headless_runtime():
    try:
        from lfs_plugins.ui import RuntimeState
    except ImportError:
        try:
            from lfs_plugins.ui.state import AppState as RuntimeState
        except ImportError:
            return False
    try:
        return bool(RuntimeState.is_headless.value)
    except (AttributeError, RuntimeError):
        return False


def _register_optional_ui():
    if _is_headless_runtime():
        lf.log.info(f"[{_PLUGIN_NAME}] UI panel registration skipped in headless runtime.")
        return
    try:
        from .main_panel import OptimizationParamSchedulerPanel
        lf.register_class(OptimizationParamSchedulerPanel)
        _REGISTERED_CLASSES.append(OptimizationParamSchedulerPanel)
        lf.log.info(f"[{_PLUGIN_NAME}] Registered panel: {OptimizationParamSchedulerPanel.id}")
    except Exception as exc:
        lf.log.error(f"[{_PLUGIN_NAME}] UI panel registration failed: {exc}")
        raise


def on_load():
    global _SCHEDULER_INSTALLED
    settings = initialize_runtime_settings(ParamSchedulerSettings.get_instance())
    _register_optional_ui()

    mode = "ui" if _REGISTERED_CLASSES else "headless"
    set_runtime_mode(mode)
    try:
        install_scheduler(settings=settings, runtime_mode=mode)
    except Exception:
        try:
            uninstall_scheduler()
        finally:
            for cls in reversed(_REGISTERED_CLASSES):
                try:
                    lf.unregister_class(cls)
                except Exception as exc:
                    lf.log.warn(f"[{_PLUGIN_NAME}] Failed to roll back panel: {exc}")
            _REGISTERED_CLASSES.clear()
        raise
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
            lf.log.warn(f"{_PLUGIN_NAME}: failed to unregister {cls}: {exc}")
    _REGISTERED_CLASSES = []
    lf.log.info(f"{_PLUGIN_NAME} unloaded")
