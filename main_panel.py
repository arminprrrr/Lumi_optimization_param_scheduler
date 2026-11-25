from __future__ import annotations

import lichtfeld as lf

from .scheduler import PARAM_SPECS, get_scheduler
from .settings import ParamSchedulerSettings, load_persistent_settings, save_persistent_settings


class OptimizationParamSchedulerPanel(lf.ui.Panel):
    id = "Lumi_optimization_param_scheduler.panel"
    label = "Optimization Param Scheduler"
    space = lf.ui.PanelSpace.MAIN_PANEL_TAB
    order = 230
    update_policy = "interval"
    update_interval_ms = 250
    poll_dependencies = {lf.ui.PollDependency.SCENE, lf.ui.PollDependency.TRAINING}

    @classmethod
    def poll(cls, context) -> bool:
        del context
        return True

    def _redraw(self):
        try:
            lf.ui.request_redraw()
        except Exception:
            pass

    def on_update(self, doc):
        return True

    def _save(self, settings):
        save_persistent_settings(settings)
        self._redraw()

    def _safe_text(self, value):
        return str(value).replace("<", "(").replace(">", ")")

    def _draw_bool(self, layout, settings, attr: str, label: str):
        changed, value = layout.checkbox(label, bool(getattr(settings, attr)))
        if changed:
            setattr(settings, attr, bool(value))
            self._save(settings)

    def _draw_input_int(self, layout, settings, attr: str, label: str, step: int = 1, step_fast: int = 100):
        changed, value = layout.input_int(label, int(getattr(settings, attr)), step=step, step_fast=step_fast)
        if changed:
            setattr(settings, attr, int(value))
            self._save(settings)

    def _draw_input_float(self, layout, settings, attr: str, label: str, step: float = 0.0, step_fast: float = 0.0, fmt: str = "%.6f"):
        changed, value = layout.input_float(label, float(getattr(settings, attr)), step=step, step_fast=step_fast, format=fmt)
        if changed:
            setattr(settings, attr, float(value))
            self._save(settings)

    def _draw_input_text(self, layout, settings, attr: str, label: str):
        changed, value = layout.input_text(label, str(getattr(settings, attr)))
        if changed:
            setattr(settings, attr, str(value))
            self._save(settings)

    def _draw_param_block(self, layout, settings, param_name: str, enable_attr: str, start_attr: str, mid_attr: str, end_attr: str):
        layout.separator()
        layout.label(param_name)
        self._draw_bool(layout, settings, enable_attr, f"Enable {param_name}")
        self._draw_input_float(layout, settings, start_attr, f"Start##{param_name}", step=0.0, step_fast=0.0, fmt="%.8f")
        self._draw_input_float(layout, settings, mid_attr, f"Mid##{param_name}", step=0.0, step_fast=0.0, fmt="%.8f")
        self._draw_input_float(layout, settings, end_attr, f"End##{param_name}", step=0.0, step_fast=0.0, fmt="%.8f")

    def _draw_runtime(self, layout, scheduler):
        layout.separator()
        layout.label("Runtime")
        if scheduler is None:
            layout.text_colored("Scheduler not initialized.", (1.0, 0.5, 0.5, 1.0))
            return

        layout.label(f"Profile: {self._safe_text(scheduler.active_profile_label)}")
        layout.label(f"Last seen iteration: {scheduler.last_seen_iteration}")
        layout.label(f"Last applied iteration: {scheduler.last_applied_iteration}")
        layout.label(f"Last applied source: {self._safe_text(scheduler.last_applied_source)}")

        if scheduler.last_values:
            values = ", ".join(f"{k}={v:.8g}" for k, v in scheduler.last_values.items())
            layout.text_colored(f"Last values: {self._safe_text(values)}", (0.8, 0.9, 0.8, 1.0))
        if scheduler.last_readback:
            readback = ", ".join(f"{k}={v}" for k, v in scheduler.last_readback.items())
            layout.text_colored(f"Readback: {self._safe_text(readback)}", (0.8, 0.8, 0.95, 1.0))
        if scheduler.last_sync_methods:
            sync = ", ".join(scheduler.last_sync_methods)
            layout.text_colored(f"Sync methods: {self._safe_text(sync)}", (0.85, 0.85, 0.95, 1.0))

        preview = scheduler.current_schedule_preview()
        if preview:
            preview_text = ", ".join(f"{k}={v:.8g}" for k, v in preview.items())
            layout.text_colored(f"Current preview: {self._safe_text(preview_text)}", (0.75, 0.9, 1.0, 1.0))

        layout.text_colored(f"Status: {self._safe_text(scheduler.status_message)}", (0.7, 0.9, 1.0, 1.0))

        scale = layout.get_dpi_scale()
        if layout.button("Apply current scheduled values now", (-1, 30 * scale)):
            scheduler.request_apply_now()
            self._redraw()

        layout.label("Recent runtime log")
        for line in scheduler.get_recent_status_lines(limit=10):
            layout.text_colored(self._safe_text(line), (0.85, 0.85, 0.85, 1.0))
        if layout.button("Clear runtime log", (-1, 24 * scale)):
            scheduler.clear_status_log()
            self._redraw()

    def _draw_settings_io(self, layout, settings):
        layout.separator()
        layout.label("Persistence")
        scale = layout.get_dpi_scale()
        if layout.button("Reload saved settings", (-1, 28 * scale)):
            load_persistent_settings(settings)
            self._redraw()
        if layout.button("Save settings now", (-1, 28 * scale)):
            save_persistent_settings(settings)
            self._redraw()

    def draw(self, layout):
        settings = ParamSchedulerSettings.get_instance()
        scheduler = get_scheduler()

        layout.label("General")
        self._draw_bool(layout, settings, "enabled", "Enabled")
        self._draw_bool(layout, settings, "dry_run", "run without mutation")
        self._draw_bool(layout, settings, "strict_readback", "Verify readback")
        self._draw_input_float(layout, settings, "mid_ratio", "Mid ratio", step=0.0, step_fast=0.0, fmt="%.6f")
        self._draw_input_int(layout, settings, "apply_every", "Apply every N steps", step=1, step_fast=10)
        self._draw_input_int(layout, settings, "log_every", "Log every N steps", step=100, step_fast=1000)
        self._draw_input_text(layout, settings, "schedule_note", "Save setting comment note")

        layout.separator()
        layout.label("Schedule values")
        layout.text_colored("Each enabled parameter uses start -> mid -> end linear interpolation over training progress.", (0.85, 0.85, 0.85, 1.0))
        for spec in PARAM_SPECS:
            self._draw_param_block(layout, settings, *spec)

        self._draw_runtime(layout, scheduler)
        self._draw_settings_io(layout, settings)
