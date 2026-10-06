"""Layer callbacks that must outlive the widget connecting them.

Put a callback here if it should be active as long as the viewer
or a layer exists (rather than sharing lifetime with a widget that
can be closed).

- Typical candidates are callbacks that mutate layer state (data, properties,
  symbols, ``editable``): that state is part of the user's work and may be
  saved to a file later, so it must survive the widget being closed.
- A callback that only refreshes a widget's own UI (a dropdown, a table, a
  button) can stay a method on that widget: once the widget is gone there is
  nothing left to update, which is fine.

"""

import warnings
import weakref
from functools import partial
from weakref import WeakSet

import cv2
import numpy as np
from napari.components.dims import RangeTuple
from napari.layers import Image, Points
from napari.layers.base import ActionType
from scipy.interpolate import interp1d

from movement.napari.layer_styles import EDITED_POINT_SYMBOL

# Metadata keys stored on the movement Points layer.
# - POINTS_LAYER_KEY marks the layer as movement-created.
# - POINTS_PROPERTIES_KEY holds the full properties df, incl. the NaN rows
#   dropped from the live layer, needed to reconstruct the dataset.
# - DATASET_ATTRS_KEY holds the source dataset's attrs (source_software, fps…).
# - TRACKS_LAYER_KEY holds a reference to the companion Tracks layer.
# - MAX_FRAME_IDX_KEY holds the last frame index of the source data,
#   including leading/trailing all-NaN frames (which are dropped from the
#   napari layer data array).
POINTS_LAYER_KEY: str = "movement_points_layer"
POINTS_PROPERTIES_KEY: str = "movement_points_properties"
DATASET_ATTRS_KEY: str = "movement_dataset_attrs"
TRACKS_LAYER_KEY: str = "movement_tracks_layer"
MAX_FRAME_IDX_KEY: str = "movement_max_frame_idx"

# Interpolation methods offered when filling a run of frames between two
# anchor points (the ``kind`` argument of ``scipy.interpolate.interp1d``).
INTERPOLATION_METHODS: tuple[str, ...] = ("linear", "nearest", "cubic")
# An alternative to the above, which follows the image content of a video
# from one anchor to the other (Lucas-Kanade optical flow) rather than
# fitting a curve through the track's positions.
OPTICAL_FLOW_METHOD: str = "optical flow"

# Keep a set of viewers already wired by connect_viewer_callbacks,
# so we don't wire them twice. We use a WeakSet so tracking a viewer here
# does not prevent it from being garbage-collected when it is no longer used.
_WIRED_VIEWERS: WeakSet = WeakSet()


# ---- Layer helpers -------------------------------------
def is_movement_points_layer(layer) -> bool:
    """Return ``True`` for a movement-loaded napari Points layer."""
    layer = getattr(layer, "__wrapped__", layer)
    return isinstance(layer, Points) and bool(
        layer.metadata.get(POINTS_LAYER_KEY)
    )


def active_movement_points_layer(viewer):
    """Return the active movement Points layer, else the last one, else None.

    Prefers napari's active selection, but falls back to scanning the layer
    list so a Points layer is still found when the active selection is
    something else or unset.
    """
    active = viewer.layers.selection.active
    if is_movement_points_layer(active):
        return getattr(active, "__wrapped__", active)
    for layer in reversed(viewer.layers):
        if is_movement_points_layer(layer):
            return getattr(layer, "__wrapped__", layer)
    return None


def find_video_layer(viewer):
    """Return the topmost Image layer holding a stack of frames, else None."""
    for layer in reversed(viewer.layers):
        layer = getattr(layer, "__wrapped__", layer)
        if isinstance(layer, Image) and layer.ndim == 3:
            return layer
    return None


def track_row_indices(
    layer: Points, individual: str, keypoint: str | None = None
) -> np.ndarray:
    """Return the row indices of one track's points in a Points layer.

    A track is all the points of one ``individual`` and, if the layer
    has a ``keypoint`` property, one ``keypoint``. Pass ``keypoint=None``
    for layers without keypoints (e.g. bounding boxes).
    """
    props = layer.properties
    mask = np.asarray(props["individual"]) == individual
    if keypoint is not None:
        mask &= np.asarray(props["keypoint"]) == keypoint
    return np.flatnonzero(mask)


def edited_frames(
    layer: Points, individual: str, keypoint: str | None = None
) -> np.ndarray:
    """Return the sorted frames on which a track's point was edited.

    See :func:`track_row_indices` for what makes up a track.
    """
    edited = layer.properties.get("edited")
    if edited is None:
        return np.array([], dtype=int)
    rows = track_row_indices(layer, individual, keypoint)
    return np.sort(layer.data[rows[edited[rows]], 0]).astype(int)


def bind_viewer_shortcut(viewer, shortcut: str, widget, method_name: str):
    """Bind a viewer keyboard shortcut to a method of a widget.

    The viewer only holds a weak reference to the widget, so that the
    shortcut does not keep the widget (and whatever it references)
    alive once the widget is gone: from then on the shortcut does
    nothing. Pass ``None`` as the ``widget`` to release the shortcut.
    """
    if widget is None:
        viewer.bind_key(shortcut, None, overwrite=True)
        return
    widget_ref = weakref.ref(widget)

    def callback(viewer):
        widget = widget_ref()
        if widget is not None:
            getattr(widget, method_name)()

    viewer.bind_key(shortcut, callback, overwrite=True)


# ---- Callbacks with viewer lifetime --------------------
def connect_viewer_callbacks(viewer) -> None:
    """Wire the layer callbacks to a viewer, skipping if already wired.

    These wirings last as long as the viewer, no matter which widget requested
    them.
    """
    # Check if viewer has been wired by this function
    viewer = getattr(viewer, "__wrapped__", viewer)
    if viewer in _WIRED_VIEWERS:
        return

    # Connect relevant layer callbacks to the viewer.
    # Connect frame slider range update to layer events
    for action in (
        "inserted",
        "removed",
    ):
        getattr(viewer.layers.events, action).connect(
            partial(update_frame_slider_range, viewer)
        )

    # Point drags are only guaranteed to stay within their own frame
    # when frame is the sliced (non-displayed) axis in a 2D view. If
    # axes are rolled or a 3D view is used in the viewer, disable editing
    # rather than risk a drag moving a point onto a different frame.
    for event in (
        viewer.dims.events.order,
        viewer.dims.events.ndisplay,
    ):
        event.connect(partial(update_points_layers_editable, viewer))

    # Update set
    _WIRED_VIEWERS.add(viewer)


def update_frame_slider_range(viewer, event=None):
    """Widen the frame slider range to cover NaN-trimmed frames.

    napari derives ``viewer.dims.range`` from the world-coordinate union of
    all layer extents, and it does so before this callback runs. Movement's
    layers have their leading/trailing all-NaN rows dropped, so a dataset
    that starts or ends with NaNs yields an extent — and therefore a slider
    range — narrower than the real frame span.

    Extend napari's range to cover the true span of every movement layer,
    but never replace it: layers movement did not create (a video the user
    opened, another plugin's layer) keep the range napari computed for them,
    including any scale or translate they carry.
    """
    max_frame_indices = [
        ly.metadata[MAX_FRAME_IDX_KEY]
        for ly in viewer.layers
        if MAX_FRAME_IDX_KEY in getattr(ly, "metadata", {})
    ]
    if not max_frame_indices:
        return

    current_range = viewer.dims.range[0]
    start = min(current_range.start, 0.0)
    stop = max(current_range.stop, *max_frame_indices)

    if (start, stop) != (current_range.start, current_range.stop):
        viewer.dims.range = (
            RangeTuple(start=start, stop=stop, step=current_range.step),
        ) + viewer.dims.range[1:]


def frame_axis_is_sliced(viewer) -> bool:
    """Determine whether the frame axis is the sliced axis in a 2D view."""
    return viewer.dims.ndisplay == 2 and viewer.dims.order[0] == 0


def update_points_layers_editable(viewer, event=None):
    """Disable point editing while the frame axis isn't sliced.

    Connected to ``viewer.dims.events.order``/``ndisplay``.
    It disables editing on every movement Points layer if
    the viewer axes are rolled or switched to 3D; napari greys
    out the select/add/delete controls.
    """
    is_editable = frame_axis_is_sliced(viewer)
    for layer in viewer.layers:
        if is_movement_points_layer(layer):
            layer.editable = is_editable


# ---- Callbacks with layer lifetime --------------------
def set_point_symbol_by_edited(layer: Points) -> None:
    """Show points flagged as edited with a distinct marker symbol."""
    edited = layer.properties.get("edited")
    if edited is None or not edited.any():
        return
    symbols = np.asarray(layer.symbol).copy()
    symbols[edited] = EDITED_POINT_SYMBOL
    layer.symbol = symbols


def on_points_data_changed(event):
    """Keep the corresponding Tracks layer in sync with the Points layer.

    Connected to ``points_layer.events.data``. Handles two actions:

    - ``ActionType.CHANGED`` (a point was dragged): sets the
      confidence score of moved points to NaN, marks them as
      edited, and changes their marker symbol to
      ``EDITED_POINT_SYMBOL`` so edited points are visually
      distinguishable. The Tracks layer row is updated in place
      via `sync_tracks_layer`.
    - ``ActionType.REMOVED`` (one or more points were deleted):
      removes the corresponding rows from
      the Tracks layer via `remove_from_tracks_layer`.
    """
    layer = event.source
    if not isinstance(layer, Points):
        return

    if event.action == ActionType.CHANGED:
        moved_indices = list(event.data_indices)
        props = layer.properties
        props["confidence"] = props["confidence"].copy()
        props["confidence"][moved_indices] = float("nan")
        if "edited" in props:
            props["edited"] = props["edited"].copy()
        else:
            props["edited"] = np.full(len(props["confidence"]), False)
        props["edited"][moved_indices] = True
        layer.properties = props
        set_point_symbol_by_edited(layer)
        sync_tracks_layer(layer, moved_indices)

    elif event.action == ActionType.REMOVED:
        removed_indices = list(event.data_indices)
        remove_from_tracks_layer(layer, removed_indices)


def sync_tracks_layer(points_layer, moved_indices):
    """Update the corresponding Tracks layer to match an edited point.

    A moved point's new (frame, y, x) is written to the same row
    in the Tracks layer, so the track segment connecting the
    previous frame to this one terminates at the dragged position.
    """
    tracks_layer = points_layer.metadata.get(TRACKS_LAYER_KEY)
    if tracks_layer is None:
        return  # the data was loaded without a Tracks layer

    # Points and Tracks layers are built from the same NaN-filtered
    # array in the same row order (see _add_points_layer/
    # _add_tracks_layer). The Tracks layer only has an extra
    # leading track_id column.
    tracks_data = tracks_layer.data
    tracks_data[moved_indices, 1:] = points_layer.data[moved_indices]

    set_tracks_layer_data(tracks_layer, tracks_data, tracks_layer.properties)


def remove_from_tracks_layer(points_layer, removed_indices):
    """Remove the rows corresponding to deleted points.

    Users edit the Points layer directly, either dragging points
    or removing inaccurate predictions. The Tracks layer has no
    interactive editing of its own, so it must be kept in sync
    with the Points layer instead.

    ``removed_indices`` are indices in the Points layer which line
    up with rows in the Tracks layer, the same way
    :func:`sync_tracks_layer` relies on for edits.
    """
    tracks_layer = points_layer.metadata.get(TRACKS_LAYER_KEY)
    if tracks_layer is None:
        return  # the data was loaded without a Tracks layer

    tracks_data = np.delete(tracks_layer.data, removed_indices, axis=0)
    tracks_properties = {
        key: np.delete(np.asarray(values), removed_indices, axis=0)
        for key, values in tracks_layer.properties.items()
    }

    set_tracks_layer_data(tracks_layer, tracks_data, tracks_properties)


def interpolate_track_between(
    layer: Points,
    individual: str,
    keypoint: str | None,
    start_frame: int,
    end_frame: int,
    method: str = "linear",
    video=None,
) -> list[int]:
    """Overwrite a track's points between two anchor frames by interpolation.

    The points of the (``individual``, ``keypoint``) track that lie
    strictly between ``start_frame`` and ``end_frame`` are moved to
    positions interpolated from the rest of the track, with the two
    anchor frames left untouched. This is meant for runs of consecutive
    frames where a keypoint is consistently misplaced: correct the frame
    before and the frame after the run, then interpolate across it.

    The moved points are announced through the layer's ``events.data``
    with ``ActionType.CHANGED`` (exactly as napari does after a drag), so
    :func:`on_points_data_changed` marks them as edited, sets their
    confidence to NaN and keeps the Tracks layer in sync.

    Parameters
    ----------
    layer
        A movement Points layer.
    individual
        Name of the individual whose track to interpolate.
    keypoint
        Name of the keypoint to interpolate, or None for a layer
        without a ``keypoint`` property.
    start_frame, end_frame
        The two anchor frames. Both must hold a point for this track.
    method
        Interpolation method, one of ``INTERPOLATION_METHODS``
        (passed as ``kind`` to :class:`scipy.interpolate.interp1d`)
        or ``OPTICAL_FLOW_METHOD``.
        With ``cubic``, every point of the track outside the range
        supports the spline, not just the two anchors.
        With ``OPTICAL_FLOW_METHOD``, the points follow the image
        content of ``video`` between the two anchors
        (see :func:`track_point_by_optical_flow`).
    video
        The frames the points were tracked on, indexed by frame number
        (e.g. the data of a napari Image layer holding a video).
        Only required if ``method`` is ``OPTICAL_FLOW_METHOD``.

    Returns
    -------
    list[int]
        Row indices (in ``layer.data``) of the points that were moved.
        Empty if the track has no points strictly between the anchors,
        e.g. because those frames are missing (NaN) -- such gaps are
        not filled.

    Raises
    ------
    ValueError
        If either anchor frame has no point for this track, if the
        track has too few points for ``method``, or if ``method`` is
        ``OPTICAL_FLOW_METHOD`` and ``video`` is missing or too short.

    """
    if start_frame >= end_frame:
        raise ValueError("start_frame must be smaller than end_frame.")
    track_name = f"'{individual}'" + (
        f" / '{keypoint}'" if keypoint is not None else ""
    )
    rows = track_row_indices(layer, individual, keypoint)
    frames = layer.data[rows, 0]
    for anchor in (start_frame, end_frame):
        if not np.any(frames == anchor):
            raise ValueError(
                f"Cannot interpolate {track_name}: "
                f"no point at anchor frame {anchor}."
            )

    in_between = (frames > start_frame) & (frames < end_frame)
    if not in_between.any():
        return []
    try:
        if method == OPTICAL_FLOW_METHOD:
            if video is None:
                raise ValueError("no video to follow the points on.")
            tracked_points = track_point_by_optical_flow(
                video,
                start_frame,
                end_frame,
                start_point=layer.data[rows[frames == start_frame][0], 1:],
                end_point=layer.data[rows[frames == end_frame][0], 1:],
            )
            new_points = tracked_points[
                (frames[in_between] - start_frame).astype(int)
            ]
        else:
            interpolator = interp1d(
                frames[~in_between],
                layer.data[rows[~in_between], 1:],
                kind=method,  # type: ignore[call-overload]
                axis=0,
            )
            new_points = interpolator(frames[in_between])
    except ValueError as e:
        raise ValueError(
            f"Cannot interpolate {track_name} with method '{method}': {e}"
        ) from e

    moved_indices = rows[in_between].tolist()
    layer.data[moved_indices, 1:] = new_points
    layer.refresh()
    layer.events.data(
        value=layer.data,
        action=ActionType.CHANGED,
        data_indices=tuple(moved_indices),
        vertex_indices=((),),
    )
    layer.events.features()
    return moved_indices


class GrayscaleVideo:
    """Read-only view of a video as grayscale ``uint8`` frames.

    Frames are converted the first time they are requested and kept, so
    that tracking several points over the same frames (or the same point
    forward and backward) reads each frame from the video only once.

    Parameters
    ----------
    video
        The video frames, indexed by frame number. Each frame is an
        array of shape (height, width) or (height, width, channels),
        with the channels in RGB(A) order.

    """

    def __init__(self, video):
        """Wrap the video frames."""
        self._video = video
        self._frames: dict[int, np.ndarray] = {}

    def __len__(self) -> int:
        """Return the number of frames in the video."""
        return len(self._video)

    def __getitem__(self, frame_idx: int) -> np.ndarray:
        """Return a frame of the video as a grayscale ``uint8`` image."""
        if frame_idx not in self._frames:
            frame = np.asarray(self._video[frame_idx])
            if frame.dtype != np.uint8:
                # Stretch the frame's values over the uint8 range
                frame = frame.astype(float) - frame.min()
                frame = (frame / (frame.max() or 1) * 255).astype(np.uint8)
            if frame.ndim == 3:
                frame = cv2.cvtColor(
                    np.ascontiguousarray(frame[..., :3]), cv2.COLOR_RGB2GRAY
                )
            self._frames[frame_idx] = frame
        return self._frames[frame_idx]


def track_point_by_optical_flow(
    video, start_frame: int, end_frame: int, start_point, end_point
) -> np.ndarray:
    """Follow a point through a video between two frames it is known on.

    The point is tracked frame by frame with pyramidal Lucas-Kanade
    optical flow (:func:`cv2.calcOpticalFlowPyrLK`), once forward from
    ``start_point`` and once backward from ``end_point``. The two paths
    are blended with a weight that shifts linearly from the forward path
    (at ``start_frame``) to the backward path (at ``end_frame``), so
    that the result lands exactly on both known points and the drift
    each path accumulates is suppressed where it is largest.

    If the point is lost along one path, the other path is used on its
    own from there on. Frames reached by neither path fall back to the
    straight line between the two known points.

    Parameters
    ----------
    video
        The video frames, indexed by frame number, or a
        :class:`GrayscaleVideo` wrapping them.
    start_frame, end_frame
        The first and last frame to track the point over.
    start_point, end_point
        The (y, x) position of the point, in pixels, on ``start_frame``
        and ``end_frame``.

    Returns
    -------
    numpy.ndarray
        Array of shape (end_frame - start_frame + 1, 2) holding the
        (y, x) position of the point on each frame from ``start_frame``
        to ``end_frame`` (both included).

    Raises
    ------
    ValueError
        If the video does not hold all frames up to ``end_frame``.

    """
    if not isinstance(video, GrayscaleVideo):
        video = GrayscaleVideo(video)
    if end_frame >= len(video):
        raise ValueError(
            f"the video has {len(video)} frames, "
            f"so it does not reach frame {end_frame}."
        )
    start_point = np.asarray(start_point, dtype=float)
    end_point = np.asarray(end_point, dtype=float)
    frame_indices = list(range(start_frame, end_frame + 1))

    forward = _follow_point(video, frame_indices, start_point)
    backward = _follow_point(video, frame_indices[::-1], end_point)[::-1]

    weights = np.linspace(0, 1, len(frame_indices))[:, np.newaxis]
    blended = (1 - weights) * forward + weights * backward
    straight_line = (1 - weights) * start_point + weights * end_point
    # NaN marks the frames on which a path had lost the point
    for fallback in (forward, backward, straight_line):
        blended = np.where(np.isnan(blended), fallback, blended)
    return blended


def _follow_point(
    video: GrayscaleVideo, frame_indices: list[int], point: np.ndarray
) -> np.ndarray:
    """Track a (y, x) point across consecutive frames of a video.

    The point is given on the first of ``frame_indices``. Returns its
    position on each of them, with NaN from the frame it is lost on.
    """
    path = np.full((len(frame_indices), 2), np.nan)
    path[0] = point
    for i in range(1, len(frame_indices)):
        # OpenCV expects points as float32 (x, y), shaped (n_points, 1, 2)
        previous_point = path[i - 1, ::-1].astype(np.float32).reshape(1, 1, 2)
        next_point, status, _ = cv2.calcOpticalFlowPyrLK(  # type: ignore[call-overload]
            video[frame_indices[i - 1]],
            video[frame_indices[i]],
            previous_point,
            None,
        )
        if not status[0, 0]:
            break
        path[i] = next_point[0, 0, ::-1]
    return path


def set_tracks_layer_data(tracks_layer, data, properties):
    """Set a Tracks layer's data and properties, preserving color_by.

    Setting ``.data`` on a napari Tracks layer resets its internal
    features to empty, which transiently invalidates ``color_by``
    (napari warns and falls back to "track_id") even though we
    restore the same properties and colour-by property right
    after. Suppress that spurious warning around the sequence.
    """
    color_by = tracks_layer.color_by
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=".*Previous color_by key.*",
            category=UserWarning,
        )
        tracks_layer.data = data
        tracks_layer.properties = properties
        tracks_layer.color_by = color_by
