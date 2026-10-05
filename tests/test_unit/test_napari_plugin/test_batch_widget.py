"""Unit tests for the batch loader widget in the napari plugin."""

import numpy as np
import pytest
import xarray as xr
from napari.layers import Image, Points, Tracks
from napari.layers.base import ActionType

from movement.io import save_poses
from movement.napari.batch_widget import (
    EDITED_FILE_SUFFIX,
    NEXT_FILE_SHORTCUT,
    NO_FOLDER_STATUS,
    PREVIOUS_FILE_SHORTCUT,
    BatchLoader,
    match_video,
)
from movement.napari.loader_widgets import DataLoader

FILE_NAMES = ["clip_0.csv", "clip_1.csv", "clip_2.csv"]


@pytest.fixture
def poses_folder(valid_poses_dataset, tmp_path):
    """Return a folder holding three DeepLabCut csv files."""
    folder = tmp_path / "poses"
    folder.mkdir()
    for file_name in FILE_NAMES:
        save_poses.to_dlc_file(
            valid_poses_dataset, folder / file_name, split_individuals=False
        )
    return folder


@pytest.fixture
def batch_loader(make_napari_viewer_proxy):
    """Return a ``BatchLoader`` set up to load DeepLabCut csv files."""
    viewer = make_napari_viewer_proxy()
    batch_loader = BatchLoader(viewer, DataLoader(viewer))
    batch_loader.loader.source_software_combo.setCurrentText("DeepLabCut")
    batch_loader.file_suffix_combo.setCurrentText("csv")
    return batch_loader


@pytest.fixture
def batch_loader_with_folder(batch_loader, poses_folder):
    """Return a ``BatchLoader`` with the poses folder loaded."""
    batch_loader.folder_path_edit.setText(str(poses_folder))
    batch_loader._on_load_folder_clicked()
    return batch_loader


def _points_layer(batch_loader):
    return next(
        ly for ly in batch_loader.viewer.layers if isinstance(ly, Points)
    )


def _drag_first_point(batch_loader, new_y=100.0, new_x=200.0):
    """Simulate the user dragging the first point of the Points layer."""
    layer = _points_layer(batch_loader)
    layer.data[0, 1:] = (new_y, new_x)
    layer.events.data(
        value=layer.data,
        action=ActionType.CHANGED,
        data_indices=(0,),
        vertex_indices=((),),
    )


def test_batch_loader_instantiation(batch_loader):
    """Test that the widget starts without a queue or usable navigation."""
    assert batch_loader.files == []
    assert batch_loader.status_label.text() == NO_FOLDER_STATUS
    assert not batch_loader.previous_button.isEnabled()
    assert not batch_loader.next_button.isEnabled()
    assert not batch_loader.save_edits_button.isEnabled()


def test_file_suffixes_follow_source_software(batch_loader):
    """Test that the suffix options follow the loader's source software."""
    combo = batch_loader.file_suffix_combo
    batch_loader.loader.source_software_combo.setCurrentText("SLEAP")
    assert [combo.itemText(i) for i in range(combo.count())] == ["h5", "slp"]


@pytest.mark.parametrize(
    "folder, expected_warning",
    [
        pytest.param("", "No folder path specified.", id="no_path"),
        pytest.param("missing", "is not a folder", id="not_a_folder"),
        pytest.param("empty", "No '.csv' files found", id="no_files"),
    ],
)
def test_load_folder_without_files_warns(
    folder, expected_warning, batch_loader, tmp_path, mocker
):
    """Test that an unusable folder path warns and loads nothing."""
    (tmp_path / "empty").mkdir()
    mock_warning = mocker.patch("movement.napari.batch_widget.show_warning")
    batch_loader.folder_path_edit.setText(
        str(tmp_path / folder) if folder else ""
    )

    batch_loader._on_load_folder_clicked()

    assert expected_warning in mock_warning.call_args.args[0]
    assert batch_loader.files == []
    assert len(batch_loader.viewer.layers) == 0


def test_load_folder_queues_files_and_loads_first(
    batch_loader, poses_folder, valid_poses_dataset
):
    """Test that loading a folder queues the files with the chosen suffix
    (skipping files with saved edits) and loads only the first one.
    """
    (poses_folder / "notes.txt").write_text("not tracked data")
    valid_poses_dataset.to_netcdf(poses_folder / f"clip_0{EDITED_FILE_SUFFIX}")
    batch_loader.folder_path_edit.setText(str(poses_folder))

    batch_loader._on_load_folder_clicked()

    assert [file.name for file in batch_loader.files] == FILE_NAMES
    assert (
        batch_loader.status_label.text()
        == "1/3: clip_0.csv (no matching video)"
    )
    assert [type(ly) for ly in batch_loader.viewer.layers] == [Points, Tracks]
    assert not batch_loader.previous_button.isEnabled()
    assert batch_loader.next_button.isEnabled()


def test_navigation_swaps_layers(batch_loader_with_folder, poses_folder):
    """Test that next/previous replace the layers with those of the
    neighbouring file, leaving other layers alone and saving nothing
    when no edits were made.
    """
    batch_loader = batch_loader_with_folder
    viewer = batch_loader.viewer
    image_layer = viewer.add_image(np.zeros((10, 10)), name="background")

    batch_loader.next_button.click()
    batch_loader.next_button.click()

    assert (
        batch_loader.status_label.text()
        == "3/3: clip_2.csv (no matching video)"
    )
    assert [ly.name for ly in viewer.layers] == [
        "background",
        "points: clip_2.csv",
        "tracks: clip_2.csv",
    ]
    assert not batch_loader.next_button.isEnabled()

    batch_loader.previous_button.click()

    assert (
        batch_loader.status_label.text()
        == "2/3: clip_1.csv (no matching video)"
    )
    assert image_layer in viewer.layers
    assert len(viewer.layers) == 3
    assert not list(poses_folder.glob(f"*{EDITED_FILE_SUFFIX}"))


def test_next_saves_edits_and_previous_reloads_original(
    batch_loader_with_folder, poses_folder, valid_poses_dataset
):
    """Test that edits are saved to a netCDF file when stepping to the
    next file, and that stepping back loads the unedited source file.
    """
    batch_loader = batch_loader_with_folder
    original_point = _points_layer(batch_loader).data[0].copy()

    _drag_first_point(batch_loader)
    assert batch_loader.status_label.text().endswith("(unsaved edits)")
    assert batch_loader.save_edits_button.isEnabled()

    batch_loader.next_button.click()

    saved_ds = xr.open_dataset(poses_folder / f"clip_0{EDITED_FILE_SUFFIX}")
    assert saved_ds["edited"].sum() == 1
    assert not saved_ds.position.equals(valid_poses_dataset.position)
    assert (
        batch_loader.status_label.text()
        == "2/3: clip_1.csv (no matching video)"
    )

    batch_loader.previous_button.click()

    np.testing.assert_array_equal(
        _points_layer(batch_loader).data[0], original_point
    )
    assert "edited" not in _points_layer(batch_loader).properties
    assert not batch_loader.save_edits_button.isEnabled()


def test_save_edits_button_saves_current_file(
    batch_loader_with_folder, poses_folder
):
    """Test that edits can be saved without leaving the current file."""
    batch_loader = batch_loader_with_folder
    _drag_first_point(batch_loader)

    batch_loader.save_edits_button.click()

    assert (poses_folder / f"clip_0{EDITED_FILE_SUFFIX}").exists()
    assert (
        batch_loader.status_label.text()
        == "1/3: clip_0.csv (no matching video)"
    )
    assert not batch_loader.save_edits_button.isEnabled()


def test_failed_save_keeps_current_file(batch_loader_with_folder, mocker):
    """Test that the current file and its edits are kept if saving fails."""
    batch_loader = batch_loader_with_folder
    _drag_first_point(batch_loader)
    points_layer = _points_layer(batch_loader)
    mocker.patch(
        "movement.napari.batch_widget.save_points_layer",
        side_effect=OSError("disk full"),
    )
    mock_error = mocker.patch("movement.napari.batch_widget.show_error")

    batch_loader.next_button.click()

    assert "disk full" in mock_error.call_args.args[0]
    assert batch_loader.status_label.text().startswith(
        "1/3: clip_0.csv (no matching video)"
    )
    assert _points_layer(batch_loader) is points_layer


def test_failed_load_is_reported(batch_loader_with_folder, mocker):
    """Test that a file that fails to load is reported and can be skipped."""
    batch_loader = batch_loader_with_folder
    batch_loader.files[1].write_text("not a DeepLabCut file")
    mock_error = mocker.patch("movement.napari.batch_widget.show_error")

    batch_loader.next_button.click()

    assert "clip_1.csv" in mock_error.call_args.args[0]
    assert len(batch_loader.viewer.layers) == 0

    batch_loader.next_button.click()

    assert (
        batch_loader.status_label.text()
        == "3/3: clip_2.csv (no matching video)"
    )
    assert len(batch_loader.viewer.layers) == 2


@pytest.mark.parametrize(
    "file_name, expected_video",
    [
        pytest.param("clip-1.h5", "clip-1.mp4", id="same_name"),
        pytest.param(
            "clip-1DLC_resnet50_shuffle1.h5", "clip-1.mp4", id="name_prefix"
        ),
        pytest.param("clip-10.h5", "clip-10.mp4", id="longest_prefix"),
        pytest.param(
            "Clip_-1DLC_resnet50.h5", "clip-1.mp4", id="other_separators"
        ),
        pytest.param(
            "clip_-10DLC_resnet50.h5", "clip-10.mp4", id="longest_loose"
        ),
        pytest.param("other.h5", None, id="no_match"),
    ],
)
def test_match_video(file_name, expected_video, tmp_path):
    """Test that a file is matched to the video its name starts with."""
    videos = [tmp_path / name for name in ("clip-1.mp4", "clip-10.mp4")]

    video = match_video(tmp_path / file_name, videos)

    assert (video.name if video else None) == expected_video


@pytest.mark.parametrize(
    "video_folder_name",
    [
        pytest.param("", id="videos_next_to_files"),
        pytest.param("videos", id="separate_video_folder"),
    ],
)
def test_matching_video_is_swapped_with_file(
    video_folder_name, batch_loader, poses_folder, tmp_path, mocker
):
    """Test that a file's matching video is loaded underneath its layers
    and replaced along with them, and that a file without one still loads.
    """
    video_folder = tmp_path / video_folder_name if video_folder_name else None
    for name in ("clip_0.mp4", "clip_2.mp4"):
        folder = video_folder or poses_folder
        folder.mkdir(exist_ok=True)
        (folder / name).touch()
    viewer = batch_loader.viewer
    mocker.patch.object(
        batch_loader,
        "_open_video",
        side_effect=lambda path: viewer.add_image(
            np.zeros((10, 8, 8)), name=path.name
        ),
    )
    batch_loader.folder_path_edit.setText(str(poses_folder))
    batch_loader.video_folder_path_edit.setText(str(video_folder or ""))

    batch_loader._on_load_folder_clicked()

    assert batch_loader.status_label.text() == "1/3: clip_0.csv"
    assert [ly.name for ly in viewer.layers] == [
        "clip_0.mp4",
        "points: clip_0.csv",
        "tracks: clip_0.csv",
    ]

    batch_loader.next_button.click()

    assert [type(ly) for ly in viewer.layers] == [Points, Tracks]

    batch_loader.next_button.click()

    assert [type(ly) for ly in viewer.layers] == [Image, Points, Tracks]
    assert viewer.layers[0].name == "clip_2.mp4"


def test_failed_video_load_still_loads_file(
    batch_loader, poses_folder, mocker
):
    """Test that a video that fails to load is reported, without
    preventing the tracked data from loading.
    """
    (poses_folder / "clip_0.mp4").write_text("not a video")
    mocker.patch.object(
        batch_loader, "_open_video", side_effect=OSError("bad video")
    )
    mock_error = mocker.patch("movement.napari.batch_widget.show_error")
    batch_loader.folder_path_edit.setText(str(poses_folder))

    batch_loader._on_load_folder_clicked()

    assert "bad video" in mock_error.call_args.args[0]
    assert [type(ly) for ly in batch_loader.viewer.layers] == [Points, Tracks]


def test_points_only_skips_tracks_layer(batch_loader, poses_folder):
    """Test that files can be loaded, edited and saved without a Tracks
    layer, and that the data loader adds tracks again afterwards.
    """
    batch_loader.points_only_checkbox.setChecked(True)
    batch_loader.folder_path_edit.setText(str(poses_folder))

    batch_loader._on_load_folder_clicked()

    assert [type(ly) for ly in batch_loader.viewer.layers] == [Points]
    assert batch_loader.loader.add_tracks

    _drag_first_point(batch_loader)
    batch_loader.next_button.click()

    assert (poses_folder / f"clip_0{EDITED_FILE_SUFFIX}").exists()
    assert [type(ly) for ly in batch_loader.viewer.layers] == [Points]


def test_loaded_points_layer_is_in_select_mode(batch_loader_with_folder):
    """Test that each file's Points layer is ready for selecting points."""
    batch_loader = batch_loader_with_folder
    assert _points_layer(batch_loader).mode == "select"

    batch_loader.next_button.click()

    assert _points_layer(batch_loader).mode == "select"


def test_keyboard_shortcuts_step_through_files(batch_loader_with_folder):
    """Test that the viewer's shortcuts load the next and previous file,
    and are released when the widget is closed.
    """
    batch_loader = batch_loader_with_folder
    viewer = batch_loader.viewer
    keymap = {str(key): func for key, func in viewer.keymap.items()}

    keymap[NEXT_FILE_SHORTCUT](viewer)
    assert batch_loader.index == 1

    keymap[PREVIOUS_FILE_SHORTCUT.replace("-", "+")](viewer)
    assert batch_loader.index == 0

    batch_loader.close()
    assert not viewer.keymap


def test_subfolders_are_queued_with_their_own_videos(
    batch_loader, valid_poses_dataset, tmp_path, mocker
):
    """Test that the files in subfolders are queued in order, each
    matched to the video in its own subfolder even if a video of the
    same name exists elsewhere.
    """
    folder = tmp_path / "sessions"
    for session in ("session_1", "session_2"):
        (folder / session).mkdir(parents=True)
        save_poses.to_dlc_file(
            valid_poses_dataset,
            folder / session / "clipDLC.csv",
            split_individuals=False,
        )
        (folder / session / "clip.mp4").touch()
    (folder / "session_2" / "other.csv").write_text("")
    (folder / "session_2" / "other.csv").rename(
        folder / "session_2" / f"clip{EDITED_FILE_SUFFIX}"
    )
    mock_open_video = mocker.patch.object(batch_loader, "_open_video")
    batch_loader.folder_path_edit.setText(str(folder))

    batch_loader._on_load_folder_clicked()

    assert batch_loader.status_label.text() == "1/2: session_1/clipDLC.csv"
    assert batch_loader.videos == [
        folder / "session_1" / "clip.mp4",
        folder / "session_2" / "clip.mp4",
    ]

    _drag_first_point(batch_loader)
    batch_loader.next_button.click()

    assert batch_loader.status_label.text() == "2/2: session_2/clipDLC.csv"
    assert (folder / "session_1" / f"clipDLC{EDITED_FILE_SUFFIX}").exists()
    assert mock_open_video.call_args.args[0] == (
        folder / "session_2" / "clip.mp4"
    )


def test_video_in_another_subfolder_is_used_as_fallback(
    batch_loader, poses_folder, tmp_path
):
    """Test that a video is still found if the video folder is not
    organised in the same subfolders as the files.
    """
    video_folder = tmp_path / "videos"
    (video_folder / "elsewhere").mkdir(parents=True)
    (video_folder / "elsewhere" / "clip_1.mp4").touch()
    batch_loader.folder = poses_folder
    batch_loader.files = sorted(poses_folder.glob("*.csv"))

    videos = batch_loader._match_videos(video_folder)

    assert videos == [None, video_folder / "elsewhere" / "clip_1.mp4", None]
