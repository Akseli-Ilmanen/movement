"""Test the napari plugin meta widget."""

import numpy as np
import pytest
from napari.layers.base import ActionType
from qtpy.QtWidgets import QMessageBox

from movement.napari.edit_timeline_widget import ALL_KEYPOINTS
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
