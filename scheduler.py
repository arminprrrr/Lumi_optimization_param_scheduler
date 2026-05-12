from __future__ import annotations

import math
import time
from collections import deque
from typing import Any

import lichtfeld as lf

from .settings import PLUGIN_NAME, ParamSchedulerSettings, save_persistent_settings

PARAM_SPECS = (
    ("lambda_dssim", "enable_lambda_dssim", "lambda_dssim_start", "lambda_dssim_mid", "lambda_dssim_end"),
    ("opacity_reg", "enable_opacity_reg", "opacity_reg_start", "opacity_reg_mid", "opacity_reg_end"),
    ("scale_reg", "enable_scale_reg", "scale_reg_start", "scale_reg_mid", "scale_reg_end"),
    ("tv_loss_weight", "enable_tv_loss_weight", "tv_loss_weight_start", "tv_loss_weight_mid", "tv_loss_weight_end"),
)

_SCHEDULER = None
_CALLBACK_HANDLER = None
_GLOBAL_HOOKS_REGISTERED = False
_RUNTIME_MODE = "unknown"


def set_runtime_mode(mode: str):
    global _RUNTIME_MODE
    _RUNTIME_MODE = str(mode)


def get_scheduler():
    return _SCHEDULER


def _dispatch_training_start(*args, **kwargs):
    scheduler = _SCHEDULER
    if scheduler is not None:
        return scheduler.on_training_start(*args, **kwargs)
    return None


def _dispatch_iteration_start(*args, **kwargs):
    scheduler = _SCHEDULER
    if scheduler is not None:
        return scheduler.on_iteration_start(*args, **kwargs)
    return None


def _dispatch_pre_optimizer_step(*args, **kwargs):
    scheduler = _SCHEDULER
    if scheduler is not None:
        return scheduler.on_pre_optimizer_step(*args, **kwargs)
    return None


def _dispatch_post_step(*args, **kwargs):
    scheduler = _SCHEDULER
    if scheduler is not None:
        return scheduler.on_post_step(*args, **kwargs)
    return None


def _dispatch_training_end(*args, **kwargs):
    scheduler = _SCHEDULER
    if scheduler is not None:
        return scheduler.on_training_end(*args, **kwargs)
    return None


def _register_global_hooks_once():

    global _GLOBAL_HOOKS_REGISTERED
    if _GLOBAL_HOOKS_REGISTERED:
        return

    lf.on_training_start(_dispatch_training_start)
    lf.on_iteration_start(_dispatch_iteration_start)

    pre_step = getattr(lf, "on_pre_optimizer_step", None)
    if callable(pre_step):
        pre_step(_dispatch_pre_optimizer_step)

    post_step = getattr(lf, "on_post_step", None)
    if callable(post_step):
        post_step(_dispatch_post_step)

    lf.on_training_end(_dispatch_training_end)
    _GLOBAL_HOOKS_REGISTERED = True


def _try_install_scoped_handler():
    handler_cls = getattr(lf, "ScopedHandler", None) or getattr(lf, "ControlSession", None)
    if handler_cls is None:
        return None

    try:
        handler = handler_cls()
    except Exception as exc:
        lf.log.warn(f"[{PLUGIN_NAME}] Scoped callback handler unavailable; falling back to global hooks: {exc}")
        return None

    def _register(handler_obj, name: str, callback):
        method = getattr(handler_obj, name, None)
        if callable(method):
            method(callback)
            return True
        return False

    registered_any = False
    for name, callback in (
        ("on_training_start", _dispatch_training_start),
        ("on_iteration_start", _dispatch_iteration_start),
        ("on_pre_optimizer_step", _dispatch_pre_optimizer_step),
        ("on_post_step", _dispatch_post_step),
        ("on_training_end", _dispatch_training_end),
    ):
        try:
            registered_any = _register(handler, name, callback) or registered_any
        except Exception as exc:
            lf.log.warn(f"[{PLUGIN_NAME}] Failed to register scoped callback {name}: {exc}")

    return handler if registered_any else None


def install_scheduler(settings: ParamSchedulerSettings | None = None, runtime_mode: str = "unknown"):
    global _SCHEDULER, _CALLBACK_HANDLER
    uninstall_scheduler()

    settings = settings or ParamSchedulerSettings.get_instance()
    _SCHEDULER = OptimizationParamScheduler(settings=settings)
    set_runtime_mode(runtime_mode)

    _CALLBACK_HANDLER = _try_install_scoped_handler()
    if _CALLBACK_HANDLER is None:
        _register_global_hooks_once()
        lf.log.info(f"[{PLUGIN_NAME}] Installed training hooks with global callback dispatch")
    else:
        lf.log.info(f"[{PLUGIN_NAME}] Installed training hooks with scoped callback handler")

    return _SCHEDULER


def uninstall_scheduler():
    global _SCHEDULER, _CALLBACK_HANDLER
    if _CALLBACK_HANDLER is not None:
        for method_name in ("clear", "close", "dispose", "unregister"):
            method = getattr(_CALLBACK_HANDLER, method_name, None)
            if callable(method):
                try:
                    method()
                    break
                except Exception as exc:
                    lf.log.warn(f"[{PLUGIN_NAME}] Failed to clear callback handler via {method_name}: {exc}")
        _CALLBACK_HANDLER = None

    if _SCHEDULER is not None:
        try:
            save_persistent_settings(_SCHEDULER.settings)
        except Exception as exc:
            lf.log.warn(f"[{PLUGIN_NAME}] Failed to save settings during uninstall: {exc}")
    _SCHEDULER = None


def _safe_float(value, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    if not math.isfinite(out):
        return default
    return out


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _lerp(a: float, b: float, t: float) -> float:
    return float(a) + (float(b) - float(a)) * float(t)


def _format_values(values: dict[str, float]) -> str:
    return ", ".join(f"{name}={value:.8g}" for name, value in values.items())


class OptimizationParamScheduler:
    def __init__(self, settings: ParamSchedulerSettings):
        self.settings = settings
        self.last_seen_iteration = -1
        self.last_applied_iteration = -1
        self.last_applied_source = ""
        self.last_progress = 0.0
        self.last_values: dict[str, float] = {}
        self.last_readback: dict[str, float | None] = {}
        self.last_sync_methods: list[str] = []
        self.status_message = "Idle"
        self.active_profile_label = "not started"
        self._status_log = deque(maxlen=80)
        self._warned_no_opt_params = False
        self._warned_readback_mismatch = False
        self._last_log_at = -1
        self._started_at = 0.0

    def clear_status_log(self):
        self._status_log.clear()

    def get_recent_status_lines(self, limit: int = 10):
        limit = max(0, int(limit))
        return list(self._status_log)[-limit:]

    def _log(self, message: str, level: str = "info"):
        text = f"[{PLUGIN_NAME}] {message}"
        self.status_message = message
        stamp = time.strftime("%H:%M:%S")
        self._status_log.append(f"{stamp} {message}")
        logger = getattr(lf.log, level, None)
        if not callable(logger):
            logger = getattr(lf.log, "info", None)
        if callable(logger):
            logger(text)

    def _warn_once_no_opt_params(self, message: str):
        if not self._warned_no_opt_params:
            self._warned_no_opt_params = True
            self._log(message, level="warn")

    def schedule_values(self, iteration: int | None = None, total_iterations: int | None = None) -> dict[str, float]:
        if iteration is None:
            iteration = self._current_iteration()
        if total_iterations is None:
            total_iterations = self._total_iterations()
        progress = self._progress_ratio(iteration, total_iterations)
        return self._schedule_for_progress(progress)

    def current_schedule_preview(self) -> dict[str, float]:
        return self.schedule_values()

    def on_training_start(self, *args, **kwargs):
        del args, kwargs
        self._started_at = time.monotonic()
        self.last_seen_iteration = 0
        self.last_applied_iteration = -1
        self._last_log_at = -1
        self._warned_readback_mismatch = False
        self._warned_no_opt_params = False
        self._log("Training started; applying parameter schedule at iteration 0")
        self.apply_for_iteration(0, source="training_start", force=True)

    def on_iteration_start(self, *args, **kwargs):
        del args, kwargs
        iteration = self._current_iteration()
        self.last_seen_iteration = iteration
        self.apply_for_iteration(iteration, source="iteration_start", force=False)

    def on_pre_optimizer_step(self, *args, **kwargs):
        del args, kwargs
        iteration = self._current_iteration()
        if iteration != self.last_applied_iteration:
            self.apply_for_iteration(iteration, source="pre_optimizer_step", force=False)

    def on_post_step(self, *args, **kwargs):
        del args, kwargs
        self.last_seen_iteration = self._current_iteration()

    def on_training_end(self, *args, **kwargs):
        del args, kwargs
        self._log(
            f"Training ended; last applied iteration={self.last_applied_iteration}, values=({_format_values(self.last_values)})"
        )
        try:
            save_persistent_settings(self.settings)
        except Exception as exc:
            self._log(f"Failed to save settings at training end: {exc}", level="warn")

    def request_apply_now(self):
        iteration = self._current_iteration()
        return self.apply_for_iteration(iteration, source="manual", force=True)

    def _context(self):
        try:
            fn = getattr(lf, "context", None)
            if callable(fn):
                return fn()
        except Exception:
            return None
        return None

    def _current_iteration(self) -> int:
        fn = getattr(lf, "trainer_current_iteration", None)
        if callable(fn):
            try:
                return max(0, int(fn()))
            except Exception:
                pass

        ctx = self._context()
        for obj in (ctx, getattr(ctx, "trainer", None), getattr(ctx, "training", None), getattr(ctx, "status", None)):
            if obj is None:
                continue
            for attr in ("current_iteration", "iteration", "step", "global_step"):
                try:
                    value = getattr(obj, attr)
                    if callable(value):
                        value = value()
                    return max(0, int(value))
                except Exception:
                    pass
        return max(0, int(self.last_seen_iteration if self.last_seen_iteration >= 0 else 0))

    def _total_iterations(self) -> int:
        fn = getattr(lf, "trainer_total_iterations", None)
        if callable(fn):
            try:
                return max(1, int(fn()))
            except Exception:
                pass

        ctx = self._context()
        for obj in (ctx, getattr(ctx, "trainer", None), getattr(ctx, "training", None), getattr(ctx, "status", None)):
            if obj is None:
                continue
            for attr in ("total_iterations", "max_iterations", "iterations", "num_iterations"):
                try:
                    value = getattr(obj, attr)
                    if callable(value):
                        value = value()
                    return max(1, int(value))
                except Exception:
                    pass

        params = self._get_optimization_params()
        if params is not None:
            value = self._read_param_field(params, "iterations")
            if value is not None:
                try:
                    return max(1, int(value))
                except Exception:
                    pass
        return max(1, int(self.last_seen_iteration + 1 if self.last_seen_iteration >= 0 else 1))

    def _progress_ratio(self, iteration: int, total_iterations: int) -> float:
        total = max(1, int(total_iterations))
        if total <= 1:
            return 0.0
        return _clamp01(float(max(0, int(iteration))) / float(total - 1))

    def _schedule_for_progress(self, progress: float) -> dict[str, float]:
        progress = _clamp01(progress)
        mid_ratio = min(0.999, max(0.001, _safe_float(getattr(self.settings, "mid_ratio", 0.5), 0.5)))
        values: dict[str, float] = {}
        for param_name, enable_attr, start_attr, mid_attr, end_attr in PARAM_SPECS:
            if not bool(getattr(self.settings, enable_attr, False)):
                continue
            start = _safe_float(getattr(self.settings, start_attr, 0.0))
            mid = _safe_float(getattr(self.settings, mid_attr, start))
            end = _safe_float(getattr(self.settings, end_attr, mid))
            if progress <= mid_ratio:
                local_t = progress / mid_ratio
                value = _lerp(start, mid, local_t)
            else:
                local_t = (progress - mid_ratio) / (1.0 - mid_ratio)
                value = _lerp(mid, end, local_t)
            values[param_name] = float(value)
        return values

    def apply_for_iteration(self, iteration: int, source: str, force: bool = False) -> bool:
        if not bool(getattr(self.settings, "enabled", True)):
            self.status_message = "Disabled"
            return False

        iteration = max(0, int(iteration))
        total_iterations = self._total_iterations()
        progress = self._progress_ratio(iteration, total_iterations)
        self.last_seen_iteration = iteration
        self.last_progress = progress
        self.active_profile_label = f"iter {iteration}/{max(total_iterations - 1, 0)} ({progress * 100.0:.2f}%)"

        apply_every = max(1, int(getattr(self.settings, "apply_every", 1)))
        if not force and iteration == self.last_applied_iteration:
            return False
        if not force and (iteration % apply_every) != 0:
            return False

        values = self._schedule_for_progress(progress)
        if not values:
            self.status_message = "No parameters enabled"
            return False

        if bool(getattr(self.settings, "dry_run", False)):
            self.last_values = dict(values)
            self.last_readback = {key: None for key in values}
            self.last_applied_iteration = iteration
            self.last_applied_source = source
            self._maybe_periodic_log(iteration, total_iterations, progress, values, dry_run=True)
            return True

        params = self._get_optimization_params()
        if params is None:
            self._warn_once_no_opt_params(
                "lf.optimization_params() was not available; cannot apply Optimization parameter schedule in this runtime"
            )
            return False

        written: dict[str, float] = {}
        for name, value in values.items():
            if self._write_param_field(params, name, value):
                written[name] = value
            else:
                self._log(f"Could not write optimization param field: {name}", level="warn")

        sync_methods = self._sync_optimization_params(params)
        readback = self._readback_values(written.keys()) if bool(getattr(self.settings, "strict_readback", True)) else {}
        retained = self._values_retained(written, readback) if readback else True

        self.last_values = dict(written)
        self.last_readback = readback
        self.last_sync_methods = list(sync_methods)
        self.last_applied_iteration = iteration
        self.last_applied_source = source

        if readback and not retained and not self._warned_readback_mismatch:
            self._warned_readback_mismatch = True
            self._log(
                "Optimization params were written but readback did not match. This LichtFeld build may return a copy from optimization_params() and may not expose a runtime setter/sync API.",
                level="warn",
            )
        else:
            self._maybe_periodic_log(iteration, total_iterations, progress, written, dry_run=False)

        return bool(written)

    def _maybe_periodic_log(self, iteration: int, total_iterations: int, progress: float, values: dict[str, float], dry_run: bool):
        log_every = int(getattr(self.settings, "log_every", 500))
        if log_every <= 0:
            self.status_message = f"Applied at iter {iteration}: {_format_values(values)}"
            return
        if iteration == 0 or iteration >= max(0, total_iterations - 1) or abs(iteration - self._last_log_at) >= log_every:
            prefix = "Dry run" if dry_run else "Applied"
            sync_text = f" via {', '.join(self.last_sync_methods)}" if self.last_sync_methods else ""
            self._log(f"{prefix} at iter {iteration}/{max(total_iterations - 1, 0)} ({progress * 100.0:.2f}%){sync_text}: {_format_values(values)}")
            self._last_log_at = iteration
        else:
            self.status_message = f"Applied at iter {iteration}: {_format_values(values)}"

    def _get_optimization_params(self):
        fn = getattr(lf, "optimization_params", None)
        if callable(fn):
            try:
                return fn()
            except Exception as exc:
                self._log(f"lf.optimization_params() failed: {exc}", level="warn")
                return None

        ctx = self._context()
        for attr in ("optimization_params", "optimization", "opt_params"):
            if ctx is None:
                break
            try:
                value = getattr(ctx, attr)
                if callable(value):
                    value = value()
                if value is not None:
                    return value
            except Exception:
                pass
        return None

    def _read_param_field(self, params, name: str):
        try:
            if hasattr(params, name):
                return getattr(params, name)
        except Exception:
            pass
        try:
            return params[name]
        except Exception:
            pass
        try:
            getter = getattr(params, "get", None)
            if callable(getter):
                return getter(name)
        except Exception:
            pass
        return None

    def _write_param_field(self, params, name: str, value: float) -> bool:
        try:
            params[name] = float(value)
            return True
        except Exception:
            pass

        try:
            setattr(params, name, float(value))
            return True
        except Exception:
            pass

        setter_name = f"set_{name}"
        try:
            setter = getattr(params, setter_name, None)
            if callable(setter):
                setter(float(value))
                return True
        except Exception:
            pass
        return False

    def _sync_optimization_params(self, params) -> list[str]:
        called: list[str] = []

        def call_with_params(owner, owner_name: str, method_names: tuple[str, ...]):
            for method_name in method_names:
                method = getattr(owner, method_name, None)
                if not callable(method):
                    continue
                try:
                    method(params)
                    called.append(f"{owner_name}.{method_name}(params)")
                except TypeError:
                    pass
                except Exception as exc:
                    self._log(f"{owner_name}.{method_name}(params) failed: {exc}", level="warn")

        def call_no_args(owner, owner_name: str, method_names: tuple[str, ...]):
            for method_name in method_names:
                method = getattr(owner, method_name, None)
                if not callable(method):
                    continue
                try:
                    method()
                    called.append(f"{owner_name}.{method_name}()")
                except TypeError:
                    pass
                except Exception as exc:
                    self._log(f"{owner_name}.{method_name}() failed: {exc}", level="warn")

        with_params = (
            "set_optimization_params",
            "update_optimization_params",
            "apply_optimization_params",
            "setOptimizationParams",
        )
        no_args = (
            "sync_strategy_optimization_params",
            "sync_optimization_params",
            "refresh_optimization_params",
        )

        call_with_params(lf, "lf", with_params)
        call_no_args(lf, "lf", no_args)

        ctx = self._context()
        if ctx is not None:
            call_with_params(ctx, "context", with_params)
            call_no_args(ctx, "context", no_args)
            for child_name in ("trainer", "training", "strategy"):
                child = getattr(ctx, child_name, None)
                if child is not None:
                    call_with_params(child, f"context.{child_name}", with_params)
                    call_no_args(child, f"context.{child_name}", no_args)

        return called

    def _readback_values(self, names) -> dict[str, float | None]:
        params = self._get_optimization_params()
        out: dict[str, float | None] = {}
        if params is None:
            return {name: None for name in names}
        for name in names:
            value = self._read_param_field(params, name)
            try:
                out[name] = float(value) if value is not None else None
            except Exception:
                out[name] = None
        return out

    def _values_retained(self, expected: dict[str, float], readback: dict[str, float | None]) -> bool:
        for name, expected_value in expected.items():
            got = readback.get(name)
            if got is None:
                return False
            tolerance = max(1e-7, abs(expected_value) * 1e-5)
            if abs(float(got) - float(expected_value)) > tolerance:
                return False
        return True
