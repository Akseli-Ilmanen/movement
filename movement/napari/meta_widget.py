"""The main napari widget for the ``movement`` package."""

from itertools import pairwise
from typing import TYPE_CHECKING

import numpy as np
from napari.layers.base import ActionType
from napari.utils.notifications import show_warning
from napari.viewer import Viewer
from qt_niu.collapsible_widget import CollapsibleWidgetContainer
from qtpy.QtCore import QTimer
from qtpy.QtWidgets import QMessageBox

if TYPE_CHECKING:
    from qtpy.QtWidgets import QWidget

from movement.napari.batch_widget import BatchLoader
from movement.napari.edit_history import EditHistory
from movement.napari.edit_timeline_widget import (
    METHOD_SHORTCUTS,
    EditControlsWidget,
    EditTimelineWidget,
)
from movement.napari.layer_wiring import (
    OPTICAL_FLOW_METHOD,
    GrayscaleVideo,
    active_movement_points_layer,
    bind_viewer_shortcut,
    edited_frames,
    find_video_layer,
    interpolate_track_between,
    is_movement_points_layer,
    track_row_indices,
)
from movement.napari.loader_widgets import DataLoader
from movement.napari.regions_widget import RegionsWidget
from movement.napari.save_widget import DataSaver


class MovementMetaWidget(CollapsibleWidgetContainer):
    """The widget to rule all ``movement`` napari widgets.

    This is a container of collapsible widgets, each responsible
    for handing specific tasks in the movement napari workflow.
    """

    def __init__(self, napari_viewer: Viewer, parent=None):
        """Initialize the meta-widget."""
        super().__init__()
        self._viewer = napari_viewer
        self.edit_timeline_widget: EditTimelineWidget | None = None
        # Undo history of each movement Points layer, by layer id
        self._edit_histories: dict[int, EditHistory] = {}
        self._edit_timeline_dock_widget: QWidget | None = None

        # Add the data loader widget
        data_loader = DataLoader(napari_viewer, parent=self)
        self.add_widget(
            data_loader,
            collapsible=True,
            widget_title="Load tracked data",
        )

        # Add the batch loader widget, which steps through the files
        # of a folder using the settings of the data loader widget
        self._batch_loader = BatchLoader(
            napari_viewer, data_loader, parent=self
        )
        self.add_widget(
            self._batch_loader,
            collapsible=True,
            widget_title="Load folder of tracked data",
        )

        # A collapsible "edit controls" widget that can be used
        # to show/hide and configure the edit timeline docked
        # to the bottom of the viewer.
        self.edit_controls = EditControlsWidget(parent=self)
        self.edit_controls.show_individuals_toggled.connect(
            self._on_show_individuals_toggled
        )
        self.edit_controls.interpolate_mode_toggled.connect(
            self._on_interpolate_mode_toggled
        )
        self.edit_controls.interpolate_all_clicked.connect(
            self._on_interpolate_all_clicked
        )
        self.edit_controls.undo_clicked.connect(self._on_undo_clicked)
        self.add_widget(
            self.edit_controls,
            collapsible=True,
            widget_title="Edit tracked data",
        )
        self._edit_timeline_collapsible = self.collapsible_widgets[-1]
        self._edit_timeline_collapsible.toggled.connect(
            self._on_edit_timeline_widget_toggled
        )

        # Add the Save widget
        self.add_widget(
            DataSaver(napari_viewer, parent=self),
            collapsible=True,
            widget_title="Save tracked data",
        )

        # Add the Regions widget
        self.add_widget(
            RegionsWidget(napari_viewer, parent=self),
            collapsible=True,
            widget_title="Define regions of interest",
        )

        loader_collapsible = self.collapsible_widgets[0]
        loader_collapsible.expand()  # expand the loader widget by default

        napari_viewer.layers.events.inserted.connect(self._on_layer_inserted)
        napari_viewer.layers.events.removed.connect(self._on_layer_removed)
        napari_viewer.layers.selection.events.active.connect(
            self._update_undo_button
        )

        self.edit_controls.show_individuals_checkbox.setEnabled(False)
        napari_viewer.layers.selection.events.active.connect(
            self._show_individuals_enabled
        )
        napari_viewer.layers.selection.events.active.connect(
            self._update_track_choices
        )

        # One key per interpolation method, each also switching on
        # picking anchors on the timeline
        for key, method in METHOD_SHORTCUTS.items():
            bind_viewer_shortcut(
                napari_viewer,
                key,
                self,
                f"_pick_{method.replace(' ', '_')}_method",
            )

    def _pick_linear_method(self) -> None:
        """Pick the linear method and switch on picking anchors."""
        self.edit_controls.pick_method_and_anchors("linear")

    def _pick_nearest_method(self) -> None:
        """Pick the nearest method and switch on picking anchors."""
        self.edit_controls.pick_method_and_anchors("nearest")

    def _pick_cubic_method(self) -> None:
        """Pick the cubic method and switch on picking anchors."""
        self.edit_controls.pick_method_and_anchors("cubic")

    def _pick_optical_flow_method(self) -> None:
        """Pick the optical flow method and switch on picking anchors."""
        self.edit_controls.pick_method_and_anchors(OPTICAL_FLOW_METHOD)

    def closeEvent(self, event):
        """Release the keyboard shortcuts when the widget is closed."""
        for key in METHOD_SHORTCUTS:
            bind_viewer_shortcut(self._viewer, key, None, "")
        super().closeEvent(event)

    def _on_layer_inserted(self, event) -> None:
        """Keep the edit timeline section collapsed until a point is edited."""
        layer = event.value
        if not is_movement_points_layer(layer):
            return  # ignore any layer that is not a movement Points layer
        self._show_individuals_enabled()
        self._update_track_choices()
        # Open the edit timeline section as soon as a point is edited
        # on this layer.
        layer.events.data.connect(self._on_points_edited)
        self._edit_history(layer)  # start recording edits, for undoing
        # Stepping through a folder of files keeps the section as it is,
        # so the timeline stays in view from one file to the next.
        if not self._batch_loader.is_loading_file:
            self._edit_timeline_collapsible.collapse(False)

    def _on_layer_removed(self, event) -> None:
        """Forget the undo history of a removed layer."""
        layer = getattr(event.value, "__wrapped__", event.value)
        self._edit_histories.pop(id(layer), None)
        self._update_undo_button()

    def _edit_history(self, layer) -> EditHistory:
        """Return the undo history of a movement Points layer."""
        layer = getattr(layer, "__wrapped__", layer)
        if id(layer) not in self._edit_histories:
            self._edit_histories[id(layer)] = EditHistory(
                layer, on_change=self._update_undo_button
            )
        return self._edit_histories[id(layer)]

    def _update_undo_button(self, *_) -> None:
        """Enable "Undo last edit" if the active layer has an edit to undo."""
        layer = active_movement_points_layer(self._viewer)
        history = self._edit_histories.get(id(layer))
        self.edit_controls.undo_button.setEnabled(
            history is not None and history.can_undo
        )

    def _on_undo_clicked(self) -> None:
        """Undo the last drag or interpolation on the active layer."""
        layer = active_movement_points_layer(self._viewer)
        history = self._edit_histories.get(id(layer))
        step = history.undo() if history is not None else None
        if step is None:
            return
        self._batch_loader.mark_edited(layer)
        if self.edit_timeline_widget is not None:
            self.edit_timeline_widget.remove_last_interpolated_spans(
                step.n_interpolated_spans
            )

    def _on_points_edited(self, event) -> None:
        """Expand the edit timeline section when a point is dragged or removed.

        Expanding creates the timeline widget on first edit. We defer this
        until the event loop is next free (via ``QTimer.singleShot``). This
        allows the layer's ``edited`` property to be fully set before
        the timeline widget reads it, and thus ensures the first edit
        is not missed.
        """
        if event.action in (ActionType.CHANGED, ActionType.REMOVING) and (
            not self._edit_timeline_collapsible.isExpanded()
        ):
            QTimer.singleShot(0, self._edit_timeline_collapsible.expand)

    def _on_edit_timeline_widget_toggled(self, expanded: bool) -> None:
        """Show/hide the edited-frames timeline docked at the bottom."""
        if not expanded:
            if self._edit_timeline_dock_widget is not None:
                self._edit_timeline_dock_widget.hide()
            return
        self._autoselect_points_layer()
        if self.edit_timeline_widget is None:
            self.edit_timeline_widget = EditTimelineWidget(self._viewer)
            self.edit_timeline_widget.set_show_individuals(
                self.edit_controls.show_individuals_checkbox.isChecked()
            )
            self.edit_timeline_widget.set_interpolate_mode(
                self.edit_controls.interpolate_button.isChecked()
            )
            self.edit_timeline_widget.anchors_selected.connect(
                self._on_anchors_selected
            )
            self._edit_timeline_dock_widget = (
                self._viewer.window.add_dock_widget(
                    self.edit_timeline_widget,
                    area="bottom",
                    name="edited frames",
                )
            )
            # Handle closing the dock via its title-bar "X"
            self._edit_timeline_dock_widget.destroyed.connect(
                self._on_edit_timeline_dock_gone
            )
        elif self._edit_timeline_dock_widget is not None:
            self._edit_timeline_dock_widget.show()

    def _on_edit_timeline_dock_gone(self, _=None) -> None:
        """Reset state after the docked timeline is closed via its X."""
        self.edit_timeline_widget = None
        self._edit_timeline_dock_widget = None
        # Collapse the edit controls, in line with the now-missing dock.
        self._edit_timeline_collapsible.collapse(False)

    def _autoselect_points_layer(self) -> None:
        """Make a movement Points layer active for the timeline.

        Leave the active layer alone if it is already a movement Points
        layer; otherwise select the last one in the layer list.
        """
        if is_movement_points_layer(self._viewer.layers.selection.active):
            return
        layer = active_movement_points_layer(self._viewer)
        if layer is not None:
            self._viewer.layers.selection.active = layer

    def _on_show_individuals_toggled(self, checked: bool) -> None:
        """Forward the "Display individuals" checkbox to the timeline."""
        if self.edit_timeline_widget is not None:
            self.edit_timeline_widget.set_show_individuals(checked)

    def _on_interpolate_mode_toggled(self, checked: bool) -> None:
        """Forward the "Interpolate between anchors" button to the timeline.

        Entering the mode also opens the edit section (and with it the
        timeline), since the anchors are picked by clicking on it.
        """
        if checked and not self._edit_timeline_collapsible.isExpanded():
            self._edit_timeline_collapsible.expand()
        if self.edit_timeline_widget is not None:
            self.edit_timeline_widget.set_interpolate_mode(checked)

    def _update_track_choices(self, *_) -> None:
        """Populate the individual/keypoint dropdowns from the active layer."""
        layer = active_movement_points_layer(self._viewer)
        if layer is None:
            return
        props = layer.properties
        individuals = list(dict.fromkeys(props["individual"]))
        keypoints = (
            list(dict.fromkeys(props["keypoint"]))
            if "keypoint" in props
            else None
        )
        self.edit_controls.set_track_choices(individuals, keypoints)

    def _on_anchors_selected(self, start_frame: int, end_frame: int) -> None:
        """Interpolate the chosen track(s) between two anchor frames.

        The track(s) and method come from the dropdowns in the edit
        controls. Points between the anchors that were already edited
        by hand are only overwritten after the user confirms.
        """
        timeline = self.edit_timeline_widget
        layer = timeline.active_layer if timeline is not None else None
        if timeline is None or layer is None:
            return
        setup = self._interpolation_setup(layer)
        if setup is None:
            return
        individual, keypoints, method, video = setup

        if not self._confirm_overwriting_edited_points(
            layer, individual, keypoints, start_frame, end_frame
        ):
            return

        n_moved, problems = 0, []
        # Undone as a whole, however many keypoints are interpolated
        with self._edit_history(layer).group() as edit_step:
            for kpt in keypoints:
                try:
                    n_moved += len(
                        interpolate_track_between(
                            layer,
                            individual,
                            kpt,
                            start_frame,
                            end_frame,
                            method,
                            video=video,
                        )
                    )
                except ValueError as e:
                    problems.append(str(e))
            edit_step.n_interpolated_spans = 1 if n_moved else 0
        if problems:
            show_warning("\n".join(problems))
        if n_moved:
            timeline.add_interpolated_span(start_frame, end_frame, individual)

    def _on_interpolate_all_clicked(self) -> None:
        """Interpolate the chosen track(s) between all their edited points.

        Every edited point of a track serves as an anchor: the points
        between each anchor and the next are interpolated, separately
        for each of the chosen keypoints. Since only the points between
        consecutive edited points move, no edited point is overwritten.
        """
        layer = active_movement_points_layer(self._viewer)
        if layer is None:
            return
        setup = self._interpolation_setup(layer)
        if setup is None:
            return
        individual, keypoints, method, video = setup

        spans: set[tuple[int, int]] = set()
        problems: list[str] = []
        # Undone as a whole, however many stretches are interpolated
        with self._edit_history(layer).group() as edit_step:
            for kpt in keypoints:
                # Fix the anchors upfront: interpolated points get flagged
                # as edited too, but must not become anchors themselves.
                anchors = edited_frames(layer, individual, kpt)
                for start_frame, end_frame in pairwise(anchors.tolist()):
                    try:
                        moved = interpolate_track_between(
                            layer,
                            individual,
                            kpt,
                            start_frame,
                            end_frame,
                            method,
                            video=video,
                        )
                    except ValueError as e:
                        problems.append(str(e))
                    else:
                        if moved:
                            spans.add((start_frame, end_frame))
            edit_step.n_interpolated_spans = len(spans)
        self._report_interpolate_all(spans, problems, individual)

    def _report_interpolate_all(self, spans, problems, individual) -> None:
        """Show the outcome of interpolating between all edited points."""
        if problems:
            show_warning("\n".join(problems))
        if not spans and not problems:
            show_warning(
                "Nothing to interpolate: edit a point on at least two "
                "frames (more than one frame apart) of the chosen "
                "keypoint(s) first."
            )
        if self.edit_timeline_widget is not None:
            for start_frame, end_frame in sorted(spans):
                self.edit_timeline_widget.add_interpolated_span(
                    start_frame, end_frame, individual
                )

    def _interpolation_setup(self, layer):
        """Return what the edit controls say to interpolate, and how.

        Returns the individual, the list of keypoints, the method and
        (for the optical flow method) the video to follow the points on,
        or None if that method is chosen without a video being loaded.
        """
        individual = self.edit_controls.individual_combo.currentText()
        keypoint = self.edit_controls.selected_keypoint
        method = self.edit_controls.method_combo.currentText()
        props = layer.properties
        if keypoint is None and "keypoint" in props:
            keypoints = list(dict.fromkeys(props["keypoint"]))
        else:
            keypoints = [keypoint]

        video = None
        if method == OPTICAL_FLOW_METHOD:
            video_layer = find_video_layer(self._viewer)
            if video_layer is None:
                show_warning(
                    f"The '{OPTICAL_FLOW_METHOD}' method follows the "
                    "points on a video: load the video first."
                )
                return None
            # Shared by the keypoints, so each frame is read only once
            video = GrayscaleVideo(video_layer.data)
        return individual, keypoints, method, video

    def _confirm_overwriting_edited_points(
        self, layer, individual, keypoints, start_frame, end_frame
    ) -> bool:
        """Ask before interpolating over points already edited by hand."""
        edited = layer.properties.get("edited")
        if edited is None:
            return True
        frames = layer.data[:, 0]
        rows = np.concatenate(
            [track_row_indices(layer, individual, kpt) for kpt in keypoints]
        ).astype(int)
        in_between = (frames[rows] > start_frame) & (frames[rows] < end_frame)
        n_edited = int(np.count_nonzero(edited[rows][in_between]))
        if n_edited == 0:
            return True
        answer = QMessageBox.question(
            self,
            "Overwrite edited points?",
            f"{n_edited} point(s) of '{individual}' between frames "
            f"{start_frame} and {end_frame} were already edited. "
            "Overwrite them with interpolated positions?",
        )
        return answer == QMessageBox.StandardButton.Yes

    def _show_individuals_enabled(self, *_) -> None:
        """Enable "Display individuals" only for multi-individual data.

        With a single individual the checkbox does nothing useful, so it
        is disabled (and unchecked, falling back to the single-colour
        shared lane).
        """
        layer = active_movement_points_layer(self._viewer)
        if layer is None:
            return
        individuals = layer.properties.get("individual")
        multiple = individuals is not None and len(set(individuals)) > 1
        checkbox = self.edit_controls.show_individuals_checkbox
        checkbox.setEnabled(multiple)
        if not multiple and checkbox.isChecked():
            checkbox.setChecked(False)
