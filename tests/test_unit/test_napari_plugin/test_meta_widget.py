"""Test the napari plugin meta widget."""

import numpy as np
import pytest
from napari.layers.base import ActionType
from qtpy.QtWidgets import QMessageBox

from movement.io import save_poses
from movement.napari.batch_widget import BatchLoader
from movement.napari.edit_timeline_widget import (
    ALL_KEYPOINTS,
    METHOD_SHORTCUTS,
)
from movement.napari.loader_widgets import DataLoader
from movement.napari.meta_widget import MovementMetaWidget


def test_meta_widget_instantiation(make_napari_viewer_proxy):
    """Test that the meta widget can be properly instantiated."""
    viewer = make_napari_viewer_proxy()
    meta_widget = MovementMetaWidget(viewer)

    # number of collapsible widgets
    assert len(meta_widget.collapsible_widgets) == 5
    assert meta_widget.edit_timeline_widget is None

    first_widget = meta_widget.collapsible_widgets[0]
    assert first_widget._text == "Load tracked data"
    assert first_widget.isExpanded()

    expected_titles = [
        "Load folder of tracked data",
        "Edit tracked data",
        "Save tracked data",
        "Define regions of interest",
    ]
    for widget, title in zip(
        meta_widget.collapsible_widgets[1:], expected_titles, strict=True
    ):
        assert widget._text == title
        assert not widget.isExpanded()


def test_edit_timeline_widget_collapsable_roundtrip(
    make_napari_viewer_proxy,
):
    """Expand, collapse, then re-expand the "Edit tracked data" section."""
    viewer = make_napari_viewer_proxy()
    meta_widget = MovementMetaWidget(viewer)
    edit_timeline_collapsible = meta_widget.collapsible_widgets[2]

    edit_timeline_collapsible.expand(animate=False)
    assert meta_widget.edit_timeline_widget is not None
    assert not meta_widget._edit_timeline_dock_widget.isHidden()

    edit_timeline_collapsible.collapse(animate=False)
    assert (
        meta_widget.edit_timeline_widget is not None
    )  # not torn down, just hidden
    assert meta_widget._edit_timeline_dock_widget.isHidden()

    edit_timeline_collapsible.expand(animate=False)
    assert not meta_widget._edit_timeline_dock_widget.isHidden()


def test_closing_edit_timeline_dock_via_its_x_resets_state(
    make_napari_viewer_proxy,
):
    """Closing the docked timeline via its title-bar "X" resets state.

    ``_on_edit_timeline_dock_gone`` is connected to the dock widget's
    ``destroyed`` signal (see ``MovementMetaWidget.__init__``), which
    fires when napari tears the dock down after the user closes it that
    way -- unlike collapsing the "Edit tracked data" section, which
    only hides it (see ``test_edit_timeline_widget_collapsable_roundtrip``).
    """
    viewer = make_napari_viewer_proxy()
    meta_widget = MovementMetaWidget(viewer)
    edit_timeline_collapsible = meta_widget.collapsible_widgets[2]
    edit_timeline_collapsible.expand(animate=False)
    assert meta_widget.edit_timeline_widget is not None

    meta_widget._on_edit_timeline_dock_gone()

    assert meta_widget.edit_timeline_widget is None
    assert meta_widget._edit_timeline_dock_widget is None
    assert not edit_timeline_collapsible.isExpanded()


def test_show_individuals_checkbox_edit_timeline_widget(
    make_napari_viewer_proxy,
):
    """The sidebar checkbox controls the docked timeline's lane display."""
    viewer = make_napari_viewer_proxy()
    meta_widget = MovementMetaWidget(viewer)
    edit_timeline_collapsible = meta_widget.collapsible_widgets[2]
    edit_timeline_collapsible.expand(animate=False)

    meta_widget.edit_controls.show_individuals_checkbox.setChecked(True)
    assert meta_widget.edit_timeline_widget._show_individuals is True

    meta_widget.edit_controls.show_individuals_checkbox.setChecked(False)
    assert meta_widget.edit_timeline_widget._show_individuals is False


@pytest.mark.parametrize(
    "individuals, expect_enabled",
    [
        pytest.param(["id_0"], False, id="single_individual"),
        pytest.param(["id_0", "id_1"], True, id="multiple_individuals"),
    ],
)
def test_show_individuals_checkbox_enabled_only_for_multiple(
    make_napari_viewer_proxy, add_movement_points, individuals, expect_enabled
):
    """Disable "Display individuals" for single-individual datasets."""
    viewer = make_napari_viewer_proxy()
    meta_widget = MovementMetaWidget(viewer)
    checkbox = meta_widget.edit_controls.show_individuals_checkbox

    assert not checkbox.isEnabled()  # nothing loaded yet

    layer = add_movement_points(viewer, individuals)
    viewer.layers.selection.active = layer

    assert checkbox.isEnabled() is expect_enabled


def test_show_individuals_enabled_noop_without_a_movement_layer(
    make_napari_viewer_proxy, add_movement_points
):
    """Selecting a non-movement layer leaves the checkbox state untouched.

    ``_show_individuals_enabled`` fires on every active-layer change,
    but with no movement Points layer left to check for individuals it
    has nothing to enable/disable for and must return early -- without
    that guard it would crash reading ``.properties`` off ``None``.
    """
    viewer = make_napari_viewer_proxy()
    meta_widget = MovementMetaWidget(viewer)
    checkbox = meta_widget.edit_controls.show_individuals_checkbox

    multi = add_movement_points(viewer, ["id_0", "id_1"])
    viewer.layers.selection.active = multi
    assert checkbox.isEnabled()  # sanity: enabled for multi-individual data

    viewer.layers.remove(multi)
    other = viewer.add_points(np.zeros((1, 2)))  # not a movement layer
    viewer.layers.selection.active = other

    assert checkbox.isEnabled()  # unchanged: no movement layer to check


def test_show_individuals_unchecked_when_switching_to_single(
    make_napari_viewer_proxy, add_movement_points
):
    """Switching to a single-individual layer clears an active check."""
    viewer = make_napari_viewer_proxy()
    meta_widget = MovementMetaWidget(viewer)
    checkbox = meta_widget.edit_controls.show_individuals_checkbox

    multi = add_movement_points(viewer, ["id_0", "id_1"])
    viewer.layers.selection.active = multi
    checkbox.setChecked(True)

    single = add_movement_points(viewer)
    viewer.layers.selection.active = single

    assert not checkbox.isEnabled()
    assert not checkbox.isChecked()


def test_expanding_edit_section_autoselects_points_layer(
    make_napari_viewer_proxy, add_movement_points
):
    """Expanding the section makes a movement Points layer active."""
    viewer = make_napari_viewer_proxy()
    meta_widget = MovementMetaWidget(viewer)
    edit_timeline_collapsible = meta_widget.collapsible_widgets[2]

    points_layer = add_movement_points(viewer)
    # An unrelated layer stealing the active selection.
    other_layer = viewer.add_points(np.zeros((1, 2)))
    viewer.layers.selection.active = other_layer

    edit_timeline_collapsible.expand(animate=False)

    assert viewer.layers.selection.active.name == points_layer.name
    assert (
        meta_widget.edit_timeline_widget.active_layer.name == points_layer.name
    )


def test_expanding_edit_section_keeps_movement_layer_active(
    make_napari_viewer_proxy, add_movement_points
):
    """A movement Points layer already active is left selected."""
    viewer = make_napari_viewer_proxy()
    meta_widget = MovementMetaWidget(viewer)
    edit_timeline_collapsible = meta_widget.collapsible_widgets[2]

    add_movement_points(viewer)
    second_layer = add_movement_points(viewer)
    viewer.layers.selection.active = second_layer

    edit_timeline_collapsible.expand(animate=False)

    assert viewer.layers.selection.active.name == second_layer.name


@pytest.mark.parametrize(
    "edited, pre_expanded",
    [
        pytest.param(False, True, id="no_edits_forces_collapse"),
        pytest.param(True, True, id="prior_edits_still_collapse"),
    ],
)
def test_edit_section_stays_collapsed_on_load(
    make_napari_viewer_proxy, add_movement_points, edited, pre_expanded
):
    """Loading a layer always collapses the edit timeline section.

    Prior edits in the dataset no longer auto-open the section; it
    only opens once a point is edited in this session. A manual open
    is overridden on load.
    """
    viewer = make_napari_viewer_proxy()
    meta_widget = MovementMetaWidget(viewer)
    edit_timeline_collapsible = meta_widget.collapsible_widgets[2]
    if pre_expanded:
        edit_timeline_collapsible.expand(
            animate=False
        )  # simulate a manual open
    else:
        assert (
            not edit_timeline_collapsible.isExpanded()
        )  # collapsed by default

    add_movement_points(viewer, edited=[edited])

    assert not edit_timeline_collapsible.isExpanded()


@pytest.mark.parametrize(
    "action, expect_expanded",
    [
        pytest.param(ActionType.CHANGED, True, id="drag_expands"),
        pytest.param(ActionType.REMOVING, True, id="remove_expands"),
        pytest.param(ActionType.ADDED, False, id="add_does_not_expand"),
    ],
)
def test_editing_points_expands_edit_section(
    make_napari_viewer_proxy,
    add_movement_points,
    mocker,
    action,
    expect_expanded,
):
    """Editing a point opens the section, but only while it is collapsed.

    Dragging or removing a point expands the "Edit tracked data" section;
    a repeat edit must not re-expand an already-open one (which would
    replay the open animation on every edit).
    """
    # ``_on_points_edited`` defers the expand via ``QTimer.singleShot``;
    # run the callback synchronously so the test does not pump the loop.
    mocker.patch(
        "movement.napari.meta_widget.QTimer.singleShot",
        side_effect=lambda _ms, cb: cb(),
    )
    viewer = make_napari_viewer_proxy()
    meta_widget = MovementMetaWidget(viewer)
    edit_timeline_collapsible = meta_widget.collapsible_widgets[2]
    # Spy on this collapsible's expand specifically
    expand = mocker.spy(edit_timeline_collapsible, "expand")

    layer = add_movement_points(viewer)
    assert not edit_timeline_collapsible.isExpanded()

    def edit():
        layer.events.data(
            value=layer.data,
            action=action,
            data_indices=(0,),
            vertex_indices=((),),
        )

    edit()
    assert edit_timeline_collapsible.isExpanded() is expect_expanded

    edit()  # A repeat edit should not re-expand the collapsible
    assert expand.call_count == (1 if expect_expanded else 0)


# ---- Interpolation between anchors --------------------


@pytest.fixture
def meta_widget_with_data(
    make_napari_viewer_proxy, valid_poses_path_and_ds, loaded_data_loader
):
    """Return a ``MovementMetaWidget`` with a poses dataset loaded through
    its own loader and its "Edit tracked data" section expanded.
    """
    viewer = make_napari_viewer_proxy()
    meta_widget = MovementMetaWidget(viewer)
    filepath, ds = valid_poses_path_and_ds
    loaded_data_loader(filepath, ds, loader=meta_widget.findChild(DataLoader))
    meta_widget.collapsible_widgets[2].expand(animate=False)
    return meta_widget


def _items(combo):
    return [combo.itemText(i) for i in range(combo.count())]


def test_track_choices_follow_the_loaded_layer(meta_widget_with_data):
    """Loading data fills the individual/keypoint dropdowns and enables
    the interpolate button.
    """
    controls = meta_widget_with_data.edit_controls
    assert _items(controls.individual_combo) == ["id_0", "id_1"]
    assert _items(controls.keypoint_combo) == [
        ALL_KEYPOINTS,
        "centroid",
        "left",
        "right",
    ]
    assert controls.interpolate_button.isEnabled()
    assert controls.selected_keypoint is None  # "all keypoints" by default


def test_interpolate_button_drives_the_timeline_mode(meta_widget_with_data):
    """Toggling the button switches the timeline's click behaviour."""
    timeline = meta_widget_with_data.edit_timeline_widget
    button = meta_widget_with_data.edit_controls.interpolate_button

    button.setChecked(True)
    assert timeline._interpolate_mode is True
    button.setChecked(False)
    assert timeline._interpolate_mode is False


@pytest.mark.parametrize(
    "keypoint, expected_keypoints",
    [
        pytest.param("centroid", {"centroid"}, id="single_keypoint"),
        pytest.param(
            ALL_KEYPOINTS, {"centroid", "left", "right"}, id="all_keypoints"
        ),
    ],
)
def test_anchors_selected_interpolates_the_chosen_tracks(
    meta_widget_with_data, keypoint, expected_keypoints
):
    """Picking two anchors interpolates the dropdown-selected track(s)
    and marks the span on the timeline.
    """
    controls = meta_widget_with_data.edit_controls
    timeline = meta_widget_with_data.edit_timeline_widget
    layer = timeline.active_layer
    controls.individual_combo.setCurrentText("id_1")
    controls.keypoint_combo.setCurrentText(keypoint)

    timeline.anchors_selected.emit(2, 6)

    edited = layer.properties["edited"]
    assert set(layer.properties["keypoint"][edited]) == expected_keypoints
    assert set(layer.properties["individual"][edited]) == {"id_1"}
    assert sorted(set(layer.data[edited, 0])) == [3, 4, 5]
    assert timeline._interpolated_spans == [(2, 6, "id_1")]


@pytest.mark.parametrize("confirmed", [True, False])
def test_overwriting_edited_points_asks_first(
    meta_widget_with_data, move_point, mocker, confirmed
):
    """Points already edited between the anchors are only overwritten
    after the user agrees.
    """
    controls = meta_widget_with_data.edit_controls
    timeline = meta_widget_with_data.edit_timeline_widget
    loader = meta_widget_with_data.findChild(DataLoader)
    move_point(loader, 4, "centroid", "id_0", new_y=100, new_x=200)
    controls.individual_combo.setCurrentText("id_0")
    controls.keypoint_combo.setCurrentText("centroid")
    answer = QMessageBox.StandardButton.Yes
    if not confirmed:
        answer = QMessageBox.StandardButton.No
    question = mocker.patch.object(
        QMessageBox, "question", return_value=answer
    )

    timeline.anchors_selected.emit(2, 6)

    question.assert_called_once()
    layer = timeline.active_layer
    edited_frames = sorted(set(layer.data[layer.properties["edited"], 0]))
    assert edited_frames == ([3, 4, 5] if confirmed else [4])


def test_anchors_without_points_are_reported(
    meta_widget_with_data, remove_point, mocker
):
    """A track lacking a point at an anchor is skipped with a warning,
    while the other tracks are still interpolated.
    """
    controls = meta_widget_with_data.edit_controls
    timeline = meta_widget_with_data.edit_timeline_widget
    loader = meta_widget_with_data.findChild(DataLoader)
    remove_point(loader, 2, "left", "id_0")
    controls.individual_combo.setCurrentText("id_0")
    controls.keypoint_combo.setCurrentText(ALL_KEYPOINTS)
    show_warning = mocker.patch("movement.napari.meta_widget.show_warning")

    timeline.anchors_selected.emit(2, 6)

    show_warning.assert_called_once()
    assert "'left'" in show_warning.call_args.args[0]
    layer = timeline.active_layer
    edited = layer.properties["edited"]
    in_between = (layer.data[:, 0] > 2) & (layer.data[:, 0] < 6)
    assert set(layer.properties["keypoint"][edited & in_between]) == {
        "centroid",
        "right",
    }


def test_optical_flow_method_needs_a_video(meta_widget_with_data, mocker):
    """Without a video layer, the optical flow method warns and moves
    nothing.
    """
    controls = meta_widget_with_data.edit_controls
    timeline = meta_widget_with_data.edit_timeline_widget
    controls.method_combo.setCurrentText("optical flow")
    mock_warning = mocker.patch("movement.napari.meta_widget.show_warning")

    timeline.anchors_selected.emit(2, 6)

    assert "load the video first" in mock_warning.call_args.args[0]
    assert "edited" not in timeline.active_layer.properties
    assert timeline._interpolated_spans == []


def test_optical_flow_method_follows_the_video_layer(meta_widget_with_data):
    """With a video layer loaded, the optical flow method moves the
    points between the anchors and marks the span on the timeline.
    """
    controls = meta_widget_with_data.edit_controls
    timeline = meta_widget_with_data.edit_timeline_widget
    layer = timeline.active_layer
    video = np.random.default_rng(0).integers(
        0, 255, size=(10, 256, 256), dtype=np.uint8
    )
    meta_widget_with_data._viewer.add_image(video, name="video")
    meta_widget_with_data._viewer.layers.selection.active = layer
    controls.individual_combo.setCurrentText("id_1")
    controls.keypoint_combo.setCurrentText("centroid")
    controls.method_combo.setCurrentText("optical flow")

    timeline.anchors_selected.emit(2, 6)

    edited = layer.properties["edited"]
    assert set(layer.properties["keypoint"][edited]) == {"centroid"}
    assert sorted(layer.data[edited, 0]) == [3, 4, 5]
    assert timeline._interpolated_spans == [(2, 6, "id_1")]


# ---- Interpolation between all edited points --------------------


def _drag_point(layer, frame, keypoint, individual, new_y, new_x):
    """Simulate the user dragging one point of a Points layer."""
    props = layer.properties
    idx = int(
        np.flatnonzero(
            (layer.data[:, 0] == frame)
            & (props["keypoint"] == keypoint)
            & (props["individual"] == individual)
        )[0]
    )
    layer.data[idx, 1:] = (new_y, new_x)
    layer.events.data(
        value=layer.data,
        action=ActionType.CHANGED,
        data_indices=(idx,),
        vertex_indices=((),),
    )
    return idx


def test_interpolate_all_uses_every_edited_point_as_anchor(
    meta_widget_with_data,
):
    """The points between each edited point and the next are interpolated
    for the chosen keypoint, leaving the edited points and the other
    keypoints where they are.
    """
    controls = meta_widget_with_data.edit_controls
    timeline = meta_widget_with_data.edit_timeline_widget
    layer = timeline.active_layer
    data_before = layer.data.copy()
    anchors = {1: (0.0, 0.0), 4: (30.0, 60.0), 8: (10.0, 20.0)}
    for frame, (y, x) in anchors.items():
        _drag_point(layer, frame, "centroid", "id_1", y, x)
    controls.individual_combo.setCurrentText("id_1")
    controls.keypoint_combo.setCurrentText("centroid")

    controls.interpolate_all_button.click()

    props = layer.properties
    track = (props["individual"] == "id_1") & (props["keypoint"] == "centroid")
    frames = layer.data[track, 0]
    expected = np.column_stack(
        [
            np.interp(
                frames, list(anchors), [yx[i] for yx in anchors.values()]
            )
            for i in range(2)
        ]
    )
    in_range = (frames >= 1) & (frames <= 8)
    np.testing.assert_allclose(
        layer.data[track, 1:][in_range], expected[in_range]
    )
    # Points outside the anchors, and other tracks, are untouched
    np.testing.assert_array_equal(
        layer.data[track][~in_range], data_before[track][~in_range]
    )
    np.testing.assert_array_equal(layer.data[~track], data_before[~track])
    assert timeline._interpolated_spans == [(1, 4, "id_1"), (4, 8, "id_1")]


def test_interpolate_all_treats_each_keypoint_separately(
    meta_widget_with_data,
):
    """With all keypoints chosen, each keypoint is interpolated between
    its own edited points only.
    """
    controls = meta_widget_with_data.edit_controls
    layer = meta_widget_with_data.edit_timeline_widget.active_layer
    for frame in (1, 4):
        _drag_point(layer, frame, "centroid", "id_1", frame, frame)
    for frame in (5, 8):
        _drag_point(layer, frame, "left", "id_1", frame, frame)
    controls.individual_combo.setCurrentText("id_1")
    controls.keypoint_combo.setCurrentText(ALL_KEYPOINTS)

    controls.interpolate_all_button.click()

    props = layer.properties
    edited = props["edited"]
    for keypoint, expected_frames in (
        ("centroid", [1, 2, 3, 4]),
        ("left", [5, 6, 7, 8]),
    ):
        frames = layer.data[edited & (props["keypoint"] == keypoint), 0]
        assert sorted(frames) == expected_frames
    assert set(props["keypoint"][edited]) == {"centroid", "left"}


def test_interpolate_all_without_anchors_warns(meta_widget_with_data, mocker):
    """With fewer than two edited points there is nothing to interpolate."""
    controls = meta_widget_with_data.edit_controls
    layer = meta_widget_with_data.edit_timeline_widget.active_layer
    _drag_point(layer, 3, "centroid", "id_1", 1.0, 1.0)
    data_before = layer.data.copy()
    controls.individual_combo.setCurrentText("id_1")
    mock_warning = mocker.patch("movement.napari.meta_widget.show_warning")

    controls.interpolate_all_button.click()

    assert "Nothing to interpolate" in mock_warning.call_args.args[0]
    np.testing.assert_array_equal(layer.data, data_before)


# ---- Stepping through a folder of files --------------------


def test_edit_section_stays_open_when_stepping_through_folder(
    make_napari_viewer_proxy, valid_poses_dataset, tmp_path
):
    """Loading the next file of a folder keeps the edit section and its
    timeline open, with the timeline following the new Points layer.
    """
    for file_name in ("clip_0.csv", "clip_1.csv"):
        save_poses.to_dlc_file(
            valid_poses_dataset, tmp_path / file_name, split_individuals=False
        )
    viewer = make_napari_viewer_proxy()
    meta_widget = MovementMetaWidget(viewer)
    batch_loader = meta_widget.findChild(BatchLoader)
    batch_loader.loader.source_software_combo.setCurrentText("DeepLabCut")
    batch_loader.file_suffix_combo.setCurrentText("csv")
    batch_loader.folder_path_edit.setText(str(tmp_path))
    batch_loader._on_load_folder_clicked()
    edit_timeline_collapsible = meta_widget.collapsible_widgets[2]
    edit_timeline_collapsible.expand(animate=False)

    batch_loader.next_button.click()

    assert edit_timeline_collapsible.isExpanded()
    assert not meta_widget._edit_timeline_dock_widget.isHidden()
    timeline = meta_widget.edit_timeline_widget
    assert timeline.active_layer.name == "points: clip_1.csv"


# ---- Undoing edits --------------------


def test_undo_restores_a_dragged_point(meta_widget_with_data):
    """Undoing a drag puts the point back, with its confidence, edited
    flag and symbol as before, and keeps the Tracks layer in sync.
    """
    controls = meta_widget_with_data.edit_controls
    layer = meta_widget_with_data.edit_timeline_widget.active_layer
    tracks_layer = meta_widget_with_data.findChild(DataLoader).tracks_layer
    data_before = layer.data.copy()
    confidence_before = layer.properties["confidence"].copy()
    symbols_before = [str(symbol) for symbol in layer.symbol]
    assert not controls.undo_button.isEnabled()

    idx = _drag_point(layer, 3, "centroid", "id_1", 100.0, 200.0)
    assert controls.undo_button.isEnabled()
    assert layer.properties["edited"][idx]

    controls.undo_button.click()

    np.testing.assert_array_equal(layer.data, data_before)
    np.testing.assert_array_equal(
        layer.properties["confidence"], confidence_before
    )
    assert not layer.properties["edited"].any()
    assert [str(symbol) for symbol in layer.symbol] == symbols_before
    np.testing.assert_array_equal(tracks_layer.data[:, 1:], layer.data)
    assert not controls.undo_button.isEnabled()


def test_undo_reverts_an_interpolation_then_its_anchors(
    meta_widget_with_data,
):
    """An interpolation over several keypoints is undone in one go,
    leaving the anchors, which are then undone one by one.
    """
    controls = meta_widget_with_data.edit_controls
    timeline = meta_widget_with_data.edit_timeline_widget
    layer = timeline.active_layer
    data_before = layer.data.copy()
    _drag_point(layer, 2, "centroid", "id_1", 0.0, 0.0)
    _drag_point(layer, 6, "centroid", "id_1", 40.0, 80.0)
    data_with_anchors = layer.data.copy()
    controls.individual_combo.setCurrentText("id_1")
    controls.keypoint_combo.setCurrentText(ALL_KEYPOINTS)
    timeline.anchors_selected.emit(2, 6)
    assert timeline._interpolated_spans == [(2, 6, "id_1")]

    controls.undo_button.click()

    np.testing.assert_array_equal(layer.data, data_with_anchors)
    assert sorted(layer.data[layer.properties["edited"], 0]) == [2, 6]
    assert timeline._interpolated_spans == []

    controls.undo_button.click()
    controls.undo_button.click()

    np.testing.assert_array_equal(layer.data, data_before)
    assert not controls.undo_button.isEnabled()


def test_undo_reverts_interpolation_between_all_edited_points(
    meta_widget_with_data,
):
    """All the stretches filled by one press of the button are undone
    together, along with their spans on the timeline.
    """
    controls = meta_widget_with_data.edit_controls
    timeline = meta_widget_with_data.edit_timeline_widget
    layer = timeline.active_layer
    for frame in (1, 4, 8):
        _drag_point(layer, frame, "centroid", "id_1", frame, frame)
    data_with_anchors = layer.data.copy()
    controls.individual_combo.setCurrentText("id_1")
    controls.keypoint_combo.setCurrentText("centroid")
    controls.interpolate_all_button.click()
    assert len(timeline._interpolated_spans) == 2

    controls.undo_button.click()

    np.testing.assert_array_equal(layer.data, data_with_anchors)
    assert sorted(layer.data[layer.properties["edited"], 0]) == [1, 4, 8]
    assert timeline._interpolated_spans == []


def test_removing_a_point_clears_the_undo_history(meta_widget_with_data):
    """Moves made before a point is deleted can no longer be undone."""
    controls = meta_widget_with_data.edit_controls
    layer = meta_widget_with_data.edit_timeline_widget.active_layer
    _drag_point(layer, 3, "centroid", "id_1", 100.0, 200.0)
    assert controls.undo_button.isEnabled()

    layer.selected_data = {0}
    layer.remove_selected()

    assert not controls.undo_button.isEnabled()


def test_undo_history_is_dropped_with_its_layer(meta_widget_with_data):
    """Removing a layer forgets its history and disables the button."""
    controls = meta_widget_with_data.edit_controls
    viewer = meta_widget_with_data._viewer
    layer = meta_widget_with_data.edit_timeline_widget.active_layer
    _drag_point(layer, 3, "centroid", "id_1", 100.0, 200.0)

    viewer.layers.clear()

    assert meta_widget_with_data._edit_histories == {}
    assert not controls.undo_button.isEnabled()


# ---- Keyboard shortcuts for the interpolation methods --------------------


def _viewer_keymap(viewer):
    return {str(key): func for key, func in viewer.keymap.items()}


@pytest.mark.parametrize("key, method", METHOD_SHORTCUTS.items())
def test_method_shortcut_picks_method_and_anchor_mode(
    meta_widget_with_data, key, method
):
    """Pressing a method's key selects it in the dropdown and switches
    on picking anchors, so two clicks on the timeline interpolate.
    """
    controls = meta_widget_with_data.edit_controls
    timeline = meta_widget_with_data.edit_timeline_widget
    viewer = meta_widget_with_data._viewer
    assert not controls.interpolate_button.isChecked()

    _viewer_keymap(viewer)[key](viewer)

    assert controls.method_combo.currentText() == method
    assert controls.interpolate_button.isChecked()
    assert timeline._interpolate_mode


def test_method_shortcut_is_a_noop_without_data(make_napari_viewer_proxy):
    """Before any data is loaded the shortcuts change nothing."""
    viewer = make_napari_viewer_proxy()
    meta_widget = MovementMetaWidget(viewer)
    controls = meta_widget.edit_controls
    method_before = controls.method_combo.currentText()

    _viewer_keymap(viewer)["O"](viewer)

    assert controls.method_combo.currentText() == method_before
    assert not controls.interpolate_button.isChecked()
    assert not meta_widget.collapsible_widgets[2].isExpanded()
