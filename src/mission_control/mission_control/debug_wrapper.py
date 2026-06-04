"""
debug_wrapper.py
----------------
DebugContext + DebuggableState — single-process knobs for stepping through
the mission state machine.

The contract:

  * ``pause`` — every DebuggableState polls this flag and spins idle while
    it's true. Long-running states should also poll it inside their inner
    work loop (see ``DebugContext.wait_if_paused``).

  * ``step_mode`` — when true, every state blocks **after** running its
    inner logic and **before** returning its outcome, until the user
    signals ``step`` (one transition advance) or clears the mode.

  * ``force_outcome`` — one-shot. If set to a valid outcome of the
    currently-executing state, that outcome is returned instead of the
    real one. Cleared automatically after consumption.

  * ``abort`` — hard stop. States return immediately with a configured
    "abort" outcome (typically ``stop`` or ``failed``).

The DebugContext is a plain object shared between the SM node and every
state — no ROS dependency here, so the states stay testable.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Callable, Iterable, Optional

from yasmin import State, Blackboard

logger = logging.getLogger(__name__)


class DebugContext:
    """Thread-safe flags driving the debugger."""

    POLL_INTERVAL_S = 0.05

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._pause = False
        self._step_mode = False
        self._step_pending = False        # one-shot release for step_mode
        self._force_outcome: Optional[str] = None
        self._abort = False
        # Last state that announced it was waiting on a step release.
        self._waiting_state: Optional[str] = None

    # ------------------------------------------------------------------ setters
    def set_pause(self, value: bool) -> None:
        with self._lock:
            self._pause = bool(value)

    def set_step_mode(self, value: bool) -> None:
        with self._lock:
            self._step_mode = bool(value)
            if not self._step_mode:
                # Leaving step mode releases any pending wait.
                self._step_pending = True

    def request_step(self) -> None:
        with self._lock:
            self._step_pending = True

    def set_force_outcome(self, outcome: Optional[str]) -> None:
        with self._lock:
            self._force_outcome = outcome

    def set_abort(self, value: bool) -> None:
        with self._lock:
            self._abort = bool(value)

    # ------------------------------------------------------------------ getters
    @property
    def paused(self) -> bool:
        with self._lock:
            return self._pause

    @property
    def step_mode(self) -> bool:
        with self._lock:
            return self._step_mode

    @property
    def aborted(self) -> bool:
        with self._lock:
            return self._abort

    @property
    def waiting_state(self) -> Optional[str]:
        with self._lock:
            return self._waiting_state

    def snapshot(self) -> dict:
        """Read-only view for /sm/blackboard snapshots."""
        with self._lock:
            return {
                "pause":         self._pause,
                "step_mode":     self._step_mode,
                "force_outcome": self._force_outcome,
                "abort":         self._abort,
                "waiting_state": self._waiting_state,
            }

    # ------------------------------------------------------------------ helpers
    def wait_if_paused(self) -> None:
        """Block while paused. Returns immediately on abort."""
        while True:
            with self._lock:
                if not self._pause or self._abort:
                    return
            time.sleep(self.POLL_INTERVAL_S)

    def consume_force_outcome(self, allowed: Iterable[str]) -> Optional[str]:
        """If force_outcome is set and is in `allowed`, return + clear it."""
        with self._lock:
            forced = self._force_outcome
            if forced is not None and forced in allowed:
                self._force_outcome = None
                return forced
            return None

    def wait_for_step(self, state_name: str) -> None:
        """Block until either step_pending is consumed or abort is raised."""
        with self._lock:
            self._waiting_state = state_name
            self._step_pending = False
        try:
            while True:
                with self._lock:
                    if self._abort:
                        return
                    if self._step_pending:
                        self._step_pending = False
                        return
                    if not self._step_mode:
                        # User left step mode while we were waiting.
                        return
                time.sleep(self.POLL_INTERVAL_S)
        finally:
            with self._lock:
                self._waiting_state = None


# ---------------------------------------------------------------------------
# DebuggableState
# ---------------------------------------------------------------------------


class DebuggableState(State):
    """YASMIN state base class that respects DebugContext.

    Subclasses must:
      * Pass ``outcomes`` and ``name`` to ``__init__``.
      * Implement ``run(blackboard) -> str`` instead of ``execute``.
      * Optionally call ``self._debug.wait_if_paused()`` inside long polling
        loops to honour pause requests mid-state.

    The base ``execute`` handles:
      * Pre-state pause / abort.
      * Calling ``run``.
      * Forced outcome override.
      * Post-state step wait.
      * Publishing the transition (``on_transition`` callback injected by
        the node so the state stays free of ROS imports).
    """

    DEFAULT_ABORT_OUTCOME = "stop"

    def __init__(
        self,
        name: str,
        outcomes: list[str],
        debug_ctx: DebugContext,
        *,
        on_enter: Optional[Callable[[str], None]] = None,
        on_transition: Optional[Callable[[str, str], None]] = None,
        abort_outcome: Optional[str] = None,
        clears_abort: bool = False,
    ) -> None:
        super().__init__(outcomes=outcomes)
        self._state_name = name
        self._debug = debug_ctx
        self._on_enter = on_enter
        self._on_transition = on_transition
        # Terminal / idle "rest" states set this: on entry they consume a pending
        # abort so the machine settles here instead of bouncing through the
        # abort_outcome chain forever (IDLE→SEARCH→FAILED→IDLE→…).
        self._clears_abort = clears_abort
        chosen_abort = abort_outcome or self.DEFAULT_ABORT_OUTCOME
        if chosen_abort not in outcomes:
            # Fall back to the first declared outcome if "stop" isn't valid.
            chosen_abort = outcomes[0]
        self._abort_outcome = chosen_abort

    # Subclass hook.
    def run(self, blackboard: Blackboard) -> str:  # pragma: no cover - abstract
        raise NotImplementedError

    # YASMIN's entry point.
    def execute(self, blackboard: Blackboard) -> str:
        if self._on_enter is not None:
            try:
                self._on_enter(self._state_name)
            except Exception:  # noqa: BLE001
                logger.exception("on_enter hook raised")

        # Rest states (IDLE / terminals) consume a pending abort on entry so the
        # mission ends here and the machine waits, instead of looping.
        if self._clears_abort and self._debug.aborted:
            self._debug.set_abort(False)

        # Pre-state pause.
        self._debug.wait_if_paused()
        if self._debug.aborted:
            return self._finish(self._abort_outcome)

        # Inner work.
        try:
            outcome = self.run(blackboard)
        except Exception:  # noqa: BLE001
            logger.exception("[%s] run() raised — returning abort outcome.", self._state_name)
            return self._finish(self._abort_outcome)

        # Forced outcome override (one-shot).
        forced = self._debug.consume_force_outcome(self.get_outcomes())
        if forced is not None:
            logger.warning(
                "[%s] force_outcome consumed: %r overrides %r",
                self._state_name, forced, outcome,
            )
            outcome = forced

        # Post-state step wait.
        if self._debug.step_mode and not self._debug.aborted:
            logger.info(
                "[%s] step_mode — waiting for /sm/control step (outcome=%r).",
                self._state_name, outcome,
            )
            self._debug.wait_for_step(self._state_name)

        return self._finish(outcome)

    # ------------------------------------------------------------------
    def _finish(self, outcome: str) -> str:
        if self._on_transition is not None:
            try:
                self._on_transition(self._state_name, outcome)
            except Exception:  # noqa: BLE001
                logger.exception("on_transition hook raised")
        return outcome
