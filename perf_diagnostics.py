"""Optional host-boundary diagnostics. Installed only with GGUF perf stats enabled."""
import functools
import inspect
import logging
import sys
import threading
import time

import torch

_lock = threading.Lock()
_totals = {}
_events = []
_rows = []
_decisions = []
_execution_rows = []
_slow_syncs = []
_execution_limit = 2048
_slow_sync_limit = 64
_slow_sync_ms = 50.0
_hooks = {}
_limit = 128
_origin = time.perf_counter()


def reset():
    global _origin
    with _lock:
        _totals.clear()
        _events.clear()
        _rows.clear()
        _execution_rows.clear()
        _slow_syncs.clear()
        _origin = time.perf_counter()
    # Loading decisions describe still-cached models; retain across prompt resets.


def decision(row):
    with _lock:
        _decisions.append(row)
        del _decisions[:-32]
    logging.info("ComfyUI-GGUF perf gate/up: %s", row)


def _record(name, started, failed, event_pair=None, detail=None):
    elapsed = (time.perf_counter() - started) * 1000
    with _lock:
        total = _totals.setdefault(name, {"calls": 0, "failed_calls": 0, "cpu_wall_ms": 0.0, "max_cpu_wall_ms": 0.0})
        total["calls"] += 1
        total["failed_calls"] += int(failed)
        total["cpu_wall_ms"] += elapsed
        total["max_cpu_wall_ms"] = max(total["max_cpu_wall_ms"], elapsed)
        if name in ("model_apply", "step_callback", "cleanup_prefetch_queues", "cleanup_malloc_graph") and len(_rows) < _limit:
            row = {"phase": name, "start_ms": (started - _origin) * 1000, "cpu_wall_ms": elapsed, "failed": failed}
            if detail is not None:
                row["step_zero_based"] = detail
            _rows.append(row)
            if event_pair:
                _events.append((row, event_pair))


def _wrap(owner, attr, name, gpu=False):
    original = getattr(owner, attr, None)
    if not callable(original):
        _hooks[name] = "unavailable"
        return
    if getattr(original, "_gguf_perf_boundary", False):
        return

    @functools.wraps(original)
    def wrapped(*args, **kwargs):
        started = time.perf_counter()
        pair = None
        if gpu:
            x = args[1] if len(args) > 1 else kwargs.get("x")
            with _lock:
                room = len(_rows) < _limit
            if room and isinstance(x, torch.Tensor) and x.is_cuda:
                with torch.cuda.device(x.device):
                    stream = torch.cuda.current_stream(x.device)
                    pair = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True), stream)
                    pair[0].record(stream)
        failed = True
        try:
            result = original(*args, **kwargs)
            failed = False
            return result
        finally:
            if pair:
                pair[1].record(pair[2])
            if name in ("host_synchronize", "host_sync_stream"):
                elapsed = (time.perf_counter() - started) * 1000
                if elapsed >= _slow_sync_ms:
                    caller = sys._getframe(1)
                    site = {"module": caller.f_globals.get("__name__"), "function": caller.f_code.co_name,
                            "line": caller.f_lineno}
                    with _lock:
                        if len(_slow_syncs) < _slow_sync_limit:
                            _slow_syncs.append({
                                "phase": name, "start_ms": (started - _origin) * 1000,
                                "cpu_wall_ms": elapsed, "caller": site, "failed": failed,
                            })
            _record(name, started, failed, pair)

    wrapped._gguf_perf_boundary = True
    setattr(owner, attr, wrapped)
    _hooks[name] = "installed"


def _wrap_execution(owner, attr, phase):
    original = getattr(owner, attr, None)
    if not original or not inspect.iscoroutinefunction(original):
        _hooks[phase] = "unavailable"
        return
    if getattr(original, "_gguf_perf_boundary", False):
        return
    signature = inspect.signature(original)
    required = ("prompt_id", "dynprompt", "current_item") if phase == "node_execute" else ("prompt_id",)
    if any(name not in signature.parameters for name in required):
        _hooks[phase] = "unsupported_signature"
        return

    @functools.wraps(original)
    async def measured(*args, **kwargs):
        bound = signature.bind_partial(*args, **kwargs).arguments
        prompt_id = str(bound.get("prompt_id"))
        detail = {}
        if phase == "node_execute" and "current_item" in bound:
            node_id = bound["current_item"]
            detail["node_id"] = str(node_id)
            try:
                detail["node_type"] = bound["dynprompt"].get_node(node_id).get("class_type")
            except (AttributeError, KeyError, TypeError):
                pass
        started = time.perf_counter()
        failed = True
        try:
            result = await original(*args, **kwargs)
            if phase == "node_execute" and isinstance(result, tuple) and result:
                detail["status"] = getattr(result[0], "name", None)
                failed = detail["status"] == "FAILURE"
            else:
                failed = False
            return result
        finally:
            ended = time.perf_counter()
            with _lock:
                if len(_execution_rows) < _execution_limit or phase == "prompt_execute":
                    _execution_rows.append({
                        "phase": phase, "prompt_id": prompt_id, "start_ms": (started - _origin) * 1000,
                        "cpu_wall_ms": (ended - started) * 1000, "failed": failed, **detail,
                    })

    measured._gguf_perf_boundary = True
    setattr(owner, attr, measured)
    _hooks[phase] = "installed"


def _prompt_breakdown(rows):
    summaries = []
    for prompt in rows:
        if prompt["phase"] != "prompt_execute":
            continue
        start = prompt["start_ms"]
        duration = prompt["cpu_wall_ms"]
        end = start + duration
        intervals = sorted(
            (max(start, node["start_ms"]), min(end, node["start_ms"] + node["cpu_wall_ms"]))
            for node in rows
            if node["phase"] == "node_execute" and node["prompt_id"] == prompt["prompt_id"]
        )
        covered = 0.0
        cursor = start
        for left, right in intervals:
            if right > max(left, cursor):
                covered += right - max(left, cursor)
                cursor = right
        summaries.append({
            "prompt_id": prompt["prompt_id"], "executor_wall_ms": duration,
            "node_wall_union_ms": covered, "outside_recorded_node_intervals_ms": max(0.0, duration - covered),
        })
    return summaries


def install():
    import comfy.model_base
    import comfy.model_management
    import comfy.model_prefetch
    import latent_preview
    import execution
    _wrap_execution(execution.PromptExecutor, "execute_async", "prompt_execute")
    _wrap_execution(execution, "execute", "node_execute")
    _wrap(comfy.model_base.BaseModel, "apply_model", "model_apply", gpu=True)
    _wrap(comfy.model_management, "synchronize", "host_synchronize")
    _wrap(comfy.model_management, "sync_stream", "host_sync_stream")
    _wrap(comfy.model_management, "free_memory", "free_memory")
    _wrap(comfy.model_prefetch, "prefetch_queue_pop", "prefetch_queue_pop")
    _wrap(comfy.model_prefetch, "pin_modules", "pin_modules")
    _wrap(comfy.model_prefetch, "malloc_graph_end", "malloc_graph_end")
    _wrap(comfy.model_prefetch, "cleanup_prefetch_queues", "cleanup_prefetch_queues")
    _wrap(comfy.model_prefetch, "cleanup_malloc_graph", "cleanup_malloc_graph")
    _wrap(latent_preview, "preview_to_image", "preview_to_cpu_image")
    _wrap(latent_preview.LatentPreviewer, "decode_latent_to_preview_image", "preview_decode")
    original = latent_preview.prepare_callback
    if not getattr(original, "_gguf_perf_boundary", False):
        @functools.wraps(original)
        def prepare(*args, **kwargs):
            callback = original(*args, **kwargs)
            @functools.wraps(callback)
            def measured(*cb_args, **cb_kwargs):
                started = time.perf_counter()
                failed = True
                try:
                    result = callback(*cb_args, **cb_kwargs)
                    failed = False
                    return result
                finally:
                    step = cb_args[0] if cb_args else cb_kwargs.get("step")
                    _record("step_callback", started, failed, detail=step)
            return measured
        prepare._gguf_perf_boundary = True
        latent_preview.prepare_callback = prepare
    _hooks["step_callback"] = "installed"
    logging.info("ComfyUI-GGUF perf boundary hooks: %s; GPU events collected only at report", _hooks)


def collect():
    with _lock:
        events = list(_events)
        rows = list(_rows)
        execution_rows = list(_execution_rows)
        slow_syncs = list(_slow_syncs)
        _execution_rows.clear()
        _slow_syncs.clear()
        totals = {k: dict(v) for k, v in _totals.items()}
        decisions = list(_decisions)
        _events.clear()
        _rows.clear()
        _totals.clear()
    rows.sort(key=lambda row: row["start_ms"])
    last_callback = None
    for row in rows:
        if row["phase"] == "step_callback":
            if last_callback is not None:
                row["since_previous_callback_start_ms"] = row["start_ms"] - last_callback
            last_callback = row["start_ms"]
    for row, (start, end, stream) in events:
        # This is the explicit report boundary, never the sampler hot path.
        end.synchronize()
        row["gpu_stream_elapsed_ms"] = start.elapsed_time(end)
    return {
        "slow_synchronizations": sorted(slow_syncs, key=lambda row: row["start_ms"]),
        "slow_sync_threshold_ms": _slow_sync_ms, "slow_sync_limit": _slow_sync_limit,
        "execution_timeline": sorted(execution_rows, key=lambda row: row["start_ms"]),
        "node_timeline_limit": _execution_limit,
        "prompt_breakdown": _prompt_breakdown(execution_rows),
        "hooks": dict(_hooks), "cpu_phases": totals, "timeline_sampled": rows,
        "timeline_sample_limit": _limit, "gate_up_load_decisions": decisions,
        "scope": "Host-wide while enabled. CPU times are inclusive and overlap; GPU stream intervals include waits and are not pure kernel time. No per-step synchronization added.",
    }
