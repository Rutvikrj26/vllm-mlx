# SPDX-License-Identifier: Apache-2.0
"""Helpers for binding MLX generation streams to worker threads."""

import importlib
import inspect
import logging
import threading
from collections.abc import Iterable

import mlx.core as mx

logger = logging.getLogger(__name__)

# Serialize stream rebinding so module-level generation_stream references are
# updated atomically across concurrent engine threads.
_STREAM_REBIND_LOCK = threading.Lock()


def bind_generation_streams(
    module_names: Iterable[str] = ("mlx_lm.generate", "mlx_vlm.generate"),
) -> object:
    """Bind mlx-lm/mlx-vlm generation streams to the current thread.

    MLX streams are thread-local. If a model is loaded on one thread and
    generation runs on another, module-level generation streams created during
    import can point at a stream that does not exist in the worker thread.
    """
    with _STREAM_REBIND_LOCK:
        default_stream = mx.new_stream(mx.default_device())
        mx.set_default_stream(default_stream)
        for module_name in module_names:
            try:
                module = importlib.import_module(module_name)
            except ImportError:
                continue
            if hasattr(module, "generation_stream"):
                setattr(module, "generation_stream", default_stream)
        return default_stream


def _safe_eval(targets: list) -> None:
    """``mx.eval`` that survives a worker-thread stream mismatch.

    On the cross-thread MLX-stream error, bind a fresh thread-local default
    stream and retry once; if that still can't materialise the targets,
    fall back to lazy evaluation (the consumer thread will eval the values
    when it next touches them).
    """
    if not targets:
        return
    try:
        mx.eval(*targets)
        return
    except RuntimeError as exc:
        if "no Stream(" not in str(exc):
            raise
    try:
        mx.set_default_stream(mx.new_stream(mx.default_device()))
        mx.eval(*targets)
    except RuntimeError:
        # Defer to lazy evaluation on the consumer thread.
        return


_PATCHED_MARKER = "_vllm_mlx_stream_recovery_patched"


def patch_mlx_lm_prompt_eval() -> bool:
    """Wrap ``mlx_lm.generate.PromptProcessingBatch.prompt`` with stream recovery.

    Upstream ``mlx-lm`` calls ``mx.eval([c.state for c in self.prompt_cache])``
    inside ``PromptProcessingBatch.prompt`` (twice — once after each
    prefill chunk, once after the right-pad finalize). Both eval calls hit
    the cross-thread stream issue when a cached prompt_cache state was
    materialised on a different thread's GPU stream — see also our matching
    fix in ``memory_cache._eval_with_stream_recovery``. We can't reach those
    eval sites without modifying upstream code, so wrap the whole ``prompt``
    method here at import time and route both implicit eval points through
    ``_safe_eval``.

    Idempotent — multiple calls install the patch only once. Returns True
    when the patch is in place.
    """
    try:
        gen = importlib.import_module("mlx_lm.generate")
    except ImportError:
        logger.debug("mlx_lm.generate not importable; skipping prompt-eval patch")
        return False

    cls = getattr(gen, "PromptProcessingBatch", None)
    if cls is None:
        logger.debug(
            "PromptProcessingBatch not present in mlx_lm.generate (mlx-lm version "
            "older than expected?); skipping prompt-eval patch"
        )
        return False

    if getattr(cls.prompt, _PATCHED_MARKER, False):
        return True

    # Fail loud if upstream restructures `prompt`. Our replacement body is a
    # near-verbatim copy of the upstream method with the two ``mx.eval`` calls
    # swapped for ``_safe_eval``. If mlx-lm bumps the prefill loop to a
    # different shape (extra padding step, fused eval, single-call API, etc.),
    # silently installing our stale body would produce wrong-looking output.
    # Anchor on stable structural invariants and skip the patch if they drift.
    try:
        upstream_src = inspect.getsource(cls.prompt)
    except (OSError, TypeError):
        upstream_src = ""
    expected_invariants = (
        ("mx.eval(", 2),
        ("_right_pad_prompts", 1),
        ("prefill_step_size", 1),
    )
    if not all(upstream_src.count(token) == n for token, n in expected_invariants):
        logger.warning(
            "mlx_lm.generate.PromptProcessingBatch.prompt structure has drifted "
            "from the version this patch was written against; skipping. "
            "Cross-thread stream errors during prefill may resurface — pin "
            "mlx-lm and re-validate the patch body."
        )
        return False

    original_prompt = cls.prompt

    def _patched_prompt(self, tokens):  # noqa: ANN001 — match upstream signature
        # Replicate the upstream method, swapping the two ``mx.eval`` calls
        # for the stream-recovery wrapper. Keeping the rest verbatim
        # avoids drift if upstream tweaks padding semantics.
        if len(self.uids) != len(tokens):
            raise ValueError("The batch length doesn't match the number of inputs")
        if not tokens:
            return

        for sti, ti in zip(self.tokens, tokens):
            sti += ti

        lengths = [len(p) for p in tokens]
        max_length = max(lengths)
        padding = [max_length - lng for lng in lengths]
        max_padding = max(padding)

        if max_padding > 0:
            from mlx_lm.generate import _right_pad_prompts  # local import

            tokens = _right_pad_prompts(tokens, max_length=max_length)
            for c in self.prompt_cache:
                c.prepare(lengths=lengths, right_padding=padding)
        else:
            tokens = mx.array(tokens)

        while tokens.shape[1] > 0:
            n_to_process = min(self.prefill_step_size, tokens.shape[1])
            self.model(tokens[:, :n_to_process], cache=self.prompt_cache)
            _safe_eval([c.state for c in self.prompt_cache])
            mx.clear_cache()
            tokens = tokens[:, n_to_process:]

        if max_padding > 0:
            for c in self.prompt_cache:
                c.finalize()
            _safe_eval([c.state for c in self.prompt_cache])
            mx.clear_cache()

    setattr(_patched_prompt, _PATCHED_MARKER, True)
    cls.prompt = _patched_prompt
    logger.info(
        "Patched mlx_lm.generate.PromptProcessingBatch.prompt with stream-recovery wrapper"
    )
    return True
