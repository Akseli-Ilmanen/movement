"""Undo history for edits to a movement Points layer."""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field

import numpy as np
from napari.layers import Points
from napari.layers.base import ActionType

from movement.napari.layer_styles import (
    DEFAULT_POINT_SYMBOL,
    EDITED_POINT_SYMBOL,
)
from movement.napari.layer_wiring import sync_tracks_layer

# Number of edits that can be undone, beyond which the oldest are dropped
MAX_UNDO_STEPS = 50


@dataclass
class EditStep:
    """The state of the points changed by one undoable edit.

    An edit is a single drag of one or more points, or a whole
    interpolation (which moves points in several batches). Each entry
    of ``changes`` holds the row indices of a batch of moved points,
    and their positions, confidence and edited flags before the move.
    """

    changes: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = (
        field(default_factory=list)
    )
    # Number of spans the edit added to the edited-frames timeline
    n_interpolated_spans: int = 0


class EditHistory:
    """Record the moves of a Points layer's points, so they can be undone.

    The history follows the layer's ``events.data``: every move
    (``ActionType.CHANGED``) is recorded as an :class:`EditStep` holding
    the previous state of the moved points. Moves made inside a
    :meth:`group` block are recorded as a single step.

    Adding or removing points changes the rows of the layer, which the
    recorded steps refer to by index, so doing either clears the history:
    only moves made after the last addition or removal can be undone.

    Parameters
    ----------
    layer
        A movement Points layer.
    on_change
        Called without arguments whenever a step is added to or removed
        from the history, e.g. to update an "Undo" button.

    """

    def __init__(self, layer: Points, on_change: Callable | None = None):
        """Start recording the layer's edits."""
        self.layer = layer
        self.on_change = on_change
        self._steps: list[EditStep] = []
        self._open_group: EditStep | None = None
        self._copy_layer_state()
        layer.events.data.connect(self._on_data_changed)

    @property
    def can_undo(self) -> bool:
        """Whether there is an edit to undo."""
        return bool(self._steps)

    def _copy_layer_state(self) -> None:
        """Keep a copy of the layer's state to compare later edits with.

        By the time an edit is announced the layer already holds the new
        positions, and whether its properties were updated yet depends on
        the order of the callbacks. The previous state of the edited
        points is thus read from this copy instead.
        """
        n_points = len(self.layer.data)
        props = self.layer.properties
        self._data = np.array(self.layer.data, dtype=float)
        self._confidence = np.array(
            props.get("confidence", np.full(n_points, np.nan)), dtype=float
        )
        self._edited = np.array(
            props.get("edited", np.full(n_points, False)), dtype=bool
        )

    def _on_data_changed(self, event) -> None:
        """Record a move, or start over if points were added or removed."""
        if event.action == ActionType.CHANGED:
            self._record_move(np.array(list(event.data_indices), dtype=int))
        elif event.action in (ActionType.ADDED, ActionType.REMOVED):
            self._copy_layer_state()
            self._open_group = None
            if self._steps:
                self._steps.clear()
                self._notify()

    def _record_move(self, indices: np.ndarray) -> None:
        """Store the previous state of the moved points as (part of) a step."""
        change = (
            indices,
            self._data[indices].copy(),
            self._confidence[indices].copy(),
            self._edited[indices].copy(),
        )
        # A moved point is flagged as edited and loses its confidence
        # (see layer_wiring.on_points_data_changed)
        self._data[indices] = self.layer.data[indices]
        self._confidence[indices] = np.nan
        self._edited[indices] = True

        if self._open_group is not None:
            self._open_group.changes.append(change)
        else:
            self._add_step(EditStep(changes=[change]))

    def _add_step(self, step: EditStep) -> None:
        """Add a step to the history, dropping the oldest if it is full."""
        self._steps.append(step)
        del self._steps[:-MAX_UNDO_STEPS]
        self._notify()

    @contextmanager
    def group(self) -> Iterator[EditStep]:
        """Record all moves made inside the ``with`` block as one step.

        Yields the step, e.g. to note how many timeline spans the edit
        added. The step is only kept if any point was actually moved.
        """
        step = EditStep()
        self._open_group = step
        try:
            yield step
        finally:
            # The group is discarded if points were added or removed
            if self._open_group is step:
                self._open_group = None
                if step.changes:
                    self._add_step(step)

    def undo(self) -> EditStep | None:
        """Put the points of the last edit back as they were before it.

        Their positions, confidence, edited flags and marker symbols are
        restored, and the Tracks layer is updated to match.

        Returns
        -------
        EditStep or None
            The step that was undone, or None if there was none.

        """
        if not self._steps:
            return None
        step = self._steps.pop()
        layer = self.layer
        for indices, data, confidence, edited in reversed(step.changes):
            self._data[indices] = data
            self._confidence[indices] = confidence
            self._edited[indices] = edited
        restored = np.unique(
            np.concatenate([indices for indices, *_ in step.changes])
        )

        layer.data[restored] = self._data[restored]
        props = layer.properties
        if "confidence" in props:
            props["confidence"] = self._confidence.copy()
        props["edited"] = self._edited.copy()
        layer.properties = props
        symbols = np.asarray(layer.symbol).copy()
        symbols[restored] = np.where(
            self._edited[restored], EDITED_POINT_SYMBOL, DEFAULT_POINT_SYMBOL
        )
        layer.symbol = symbols
        sync_tracks_layer(layer, restored.tolist())
        layer.refresh()
        self._notify()
        return step

    def _notify(self) -> None:
        """Tell the listener that the history has changed."""
        if self.on_change is not None:
            self.on_change()
