"""Widget for stepping through a folder of tracked data files."""

from pathlib import Path

from napari.layers.base import ActionType
from napari.utils.notifications import show_error, show_info, show_warning
from napari.viewer import Viewer
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QWidget,
)

from movement.napari.layer_wiring import is_movement_points_layer
from movement.napari.loader_widgets import SUPPORTED_DATA_FILES, DataLoader
from movement.napari.save_widget import save_points_layer
from movement.utils.logging import logger

# Appended to a file's stem to name the netCDF file its edits are saved to
EDITED_FILE_SUFFIX = "_edited.nc"

# Suffixes of the video files that napari-video can read
VIDEO_SUFFIXES = (".mp4", ".mov", ".avi")

NO_FOLDER_STATUS = "No folder loaded"


def _unwrap(obj):
    """Return the object behind a napari public proxy (or ``obj`` itself)."""
    return getattr(obj, "__wrapped__", obj)


def match_video(file: Path, videos: list[Path]) -> Path | None:
    """Return the video that a tracked data file was derived from, if any.

    A video matches if its name without the suffix is the start of the
    file's name, e.g. ``clip-1.mp4`` matches both ``clip-1.h5`` and
    ``clip-1DLC_resnet50_shuffle1_200000.h5``. Among several matches,
    the video with the longest name wins.
    """
    matches = [video for video in videos if file.stem.startswith(video.stem)]
    return max(matches, key=lambda video: len(video.stem), default=None)


class BatchLoader(QWidget):
    """Widget for loading and refining a folder of files, one at a time.

    The files in the chosen folder form a queue that the user steps
    through with the "Previous" and "Next" buttons. Only the layers of
    the current file are kept in the viewer: they are removed before
    the layers of the next file are added.

    If a video matching the current file is found (see
    :func:`match_video`), it is loaded underneath the tracked data and
    swapped along with it.

    Files are loaded through the ``DataLoader`` widget, using the
    source software and fps set there. The files in the queue are never
    modified: when stepping to another file, any edits made to the
    current one are saved next to it as a movement (netCDF) file named
    ``<stem>_edited.nc``. Stepping back to a file loads it from the
    queue again, i.e. without the edits saved earlier.
    """

    def __init__(self, napari_viewer: Viewer, loader: DataLoader, parent=None):
        """Initialize the batch loader widget."""
        super().__init__(parent=parent)
        self.viewer = napari_viewer
        self.loader = loader
        self.files: list[Path] = []
        self.videos: list[Path | None] = []
        self.index: int = -1
        self._source_software: str = ""
        self._current_layers: list = []
        self._has_unsaved_edits: bool = False
        self.setLayout(QFormLayout())

        self._create_file_suffix_widget()
        self._create_folder_path_widget()
        self._create_video_folder_path_widget()
        self._create_points_only_checkbox()
        self._create_load_folder_button()
        self._create_navigation_widgets()
        self._update_navigation_state()

    def _create_file_suffix_widget(self):
        """Create a combo box for the suffix of the files to queue.

        Its options follow the source software selected in the
        data loader widget.
        """
        self.file_suffix_combo = QComboBox()
        self.file_suffix_combo.setObjectName("file_suffix_combo")
        self.file_suffix_combo.setToolTip(
            "Only files with this suffix are loaded from the folder.\n"
            "The options depend on the source software selected in the\n"
            "'Load tracked data' menu."
        )
        self.loader.source_software_combo.currentTextChanged.connect(
            self._on_source_software_changed
        )
        self._on_source_software_changed(
            self.loader.source_software_combo.currentText()
        )
        self.layout().addRow("file suffix:", self.file_suffix_combo)

    def _on_source_software_changed(self, current_text: str):
        """Offer the file suffixes supported by the source software."""
        self.file_suffix_combo.clear()
        self.file_suffix_combo.addItems(
            SUPPORTED_DATA_FILES.get(current_text, [])
        )

    def _create_folder_path_widget(self):
        """Create a line edit and browse button for selecting the folder."""
        self.folder_path_edit = QLineEdit()
        self.folder_path_edit.setObjectName("folder_path_edit")
        self.browse_button = QPushButton("Browse")
        self.browse_button.setObjectName("browse_folder_button")
        self.browse_button.clicked.connect(
            lambda: self._on_browse_clicked(
                self.folder_path_edit,
                "Open folder containing tracked data files",
            )
        )

        self.folder_path_layout = QHBoxLayout()
        self.folder_path_layout.addWidget(self.folder_path_edit)
        self.folder_path_layout.addWidget(self.browse_button)
        self.layout().addRow("folder path:", self.folder_path_layout)

    def _create_video_folder_path_widget(self):
        """Create a line edit and browse button for the videos' folder."""
        self.video_folder_path_edit = QLineEdit()
        self.video_folder_path_edit.setObjectName("video_folder_path_edit")
        self.video_folder_path_edit.setPlaceholderText("same as folder path")
        self.video_folder_path_edit.setToolTip(
            "Folder holding the videos the files were derived from.\n"
            "A video is loaded along with a file if its name (without\n"
            "the suffix) is the start of the file's name.\n"
            "Leave empty to look for videos in the folder of the files."
        )
        self.browse_video_folder_button = QPushButton("Browse")
        self.browse_video_folder_button.setObjectName(
            "browse_video_folder_button"
        )
        self.browse_video_folder_button.clicked.connect(
            lambda: self._on_browse_clicked(
                self.video_folder_path_edit, "Open folder containing videos"
            )
        )

        self.video_folder_path_layout = QHBoxLayout()
        self.video_folder_path_layout.addWidget(self.video_folder_path_edit)
        self.video_folder_path_layout.addWidget(
            self.browse_video_folder_button
        )
        self.layout().addRow("video folder:", self.video_folder_path_layout)

    def _create_points_only_checkbox(self):
        """Create a checkbox for loading files without a Tracks layer."""
        self.points_only_checkbox = QCheckBox("Load points only")
        self.points_only_checkbox.setObjectName("points_only_checkbox")
        self.points_only_checkbox.setToolTip(
            "Load each file without its tracks layer.\n"
            "The tracks are not needed for editing points,\n"
            "and files load faster without them."
        )
        self.layout().addRow(self.points_only_checkbox)

    def _create_load_folder_button(self):
        """Create a button to queue the folder's files and load the first."""
        self.load_folder_button = QPushButton("Load folder")
        self.load_folder_button.setObjectName("load_folder_button")
        self.load_folder_button.setToolTip(
            "Queue the files in the folder and load the first one, using\n"
            "the source software and fps set in the 'Load tracked data' menu."
        )
        self.load_folder_button.clicked.connect(self._on_load_folder_clicked)
        self.layout().addRow(self.load_folder_button)

    def _create_navigation_widgets(self):
        """Create the status label and the previous/next/save buttons."""
        self.status_label = QLabel(NO_FOLDER_STATUS)
        self.status_label.setObjectName("batch_status_label")
        self.status_label.setWordWrap(True)
        self.layout().addRow(self.status_label)

        save_tooltip = (
            f"Edits to the current file are saved next to it,\n"
            f"as a movement (netCDF) file ending in '{EDITED_FILE_SUFFIX}'."
        )
        self.previous_button = QPushButton("Previous")
        self.previous_button.setObjectName("previous_file_button")
        self.previous_button.setToolTip(
            "Load the previous file in the folder.\n" + save_tooltip
        )
        self.previous_button.clicked.connect(self._on_previous_clicked)

        self.next_button = QPushButton("Next")
        self.next_button.setObjectName("next_file_button")
        self.next_button.setToolTip(
            "Load the next file in the folder.\n" + save_tooltip
        )
        self.next_button.clicked.connect(self._on_next_clicked)

        self.save_edits_button = QPushButton("Save edits")
        self.save_edits_button.setObjectName("save_edits_button")
        self.save_edits_button.setToolTip(save_tooltip)
        self.save_edits_button.clicked.connect(self._save_edits)

        self.navigation_layout = QHBoxLayout()
        self.navigation_layout.addWidget(self.previous_button)
        self.navigation_layout.addWidget(self.next_button)
        self.navigation_layout.addWidget(self.save_edits_button)
        self.layout().addRow(self.navigation_layout)

    def _on_browse_clicked(self, path_edit: QLineEdit, caption: str):
        """Open a dialog to select a folder and show it in ``path_edit``."""
        folder_path = QFileDialog.getExistingDirectory(self, caption=caption)
        # A blank string is returned if the user cancels the dialog
        if not folder_path:
            return
        path_edit.setText(folder_path)

    def _on_load_folder_clicked(self):
        """Queue the matching files in the folder and load the first one."""
        folder_path = self.folder_path_edit.text()
        if not folder_path:
            show_warning("No folder path specified.")
            return
        folder = Path(folder_path)
        if not folder.is_dir():
            show_warning(f"'{folder}' is not a folder.")
            return
        video_folder = Path(self.video_folder_path_edit.text() or folder)
        if not video_folder.is_dir():
            show_warning(f"'{video_folder}' is not a folder.")
            return

        suffix = self.file_suffix_combo.currentText()
        # Files holding edits saved by this widget are outputs,
        # so they are not queued when stepping through netCDF files.
        files = sorted(
            file
            for file in folder.iterdir()
            if file.is_file()
            and file.suffix.lower() == f".{suffix}"
            and not file.name.endswith(EDITED_FILE_SUFFIX)
        )
        if not files:
            show_warning(f"No '.{suffix}' files found in '{folder}'.")
            return

        # Keep any edits to the file of a previously loaded folder
        if not self._save_edits():
            return
        self._remove_current_layers()
        self.files = files
        video_files = [
            file
            for file in video_folder.iterdir()
            if file.is_file() and file.suffix.lower() in VIDEO_SUFFIXES
        ]
        self.videos = [match_video(file, video_files) for file in files]
        self._source_software = self.loader.source_software_combo.currentText()
        logger.info(f"Queued {len(files)} '.{suffix}' files from '{folder}'.")
        self._load_file(0)

    def _on_previous_clicked(self):
        """Save any edits and load the previous file in the queue."""
        self._go_to(self.index - 1)

    def _on_next_clicked(self):
        """Save any edits and load the next file in the queue."""
        self._go_to(self.index + 1)

    def _go_to(self, index: int):
        """Save any edits to the current file and load the one at ``index``.

        The current file is kept if its edits cannot be saved,
        so that they are not lost.
        """
        if not 0 <= index < len(self.files):
            return
        if not self._save_edits():
            return
        self._remove_current_layers()
        self._load_file(index)

    def _load_file(self, index: int):
        """Load the file at ``index`` in the queue via the data loader.

        The matching video, if any, is loaded first so that it sits
        underneath the tracked data.
        """
        self.index = index
        file_path = self.files[index]
        video_path = self.videos[index]
        existing_layers = {id(_unwrap(ly)) for ly in self.viewer.layers}

        if video_path is not None:
            try:
                self._open_video(video_path)
            except Exception as e:
                show_error(f"Failed to load video '{video_path}': {e}")

        self.loader.source_software_combo.setCurrentText(self._source_software)
        self.loader.file_path_edit.setText(str(file_path))
        self.loader.add_tracks = not self.points_only_checkbox.isChecked()
        try:
            self.loader._on_load_clicked()
        except Exception as e:
            show_error(f"Failed to load '{file_path}': {e}")
        finally:
            self.loader.add_tracks = True

        self._current_layers = [
            _unwrap(ly)
            for ly in self.viewer.layers
            if id(_unwrap(ly)) not in existing_layers
        ]
        self._has_unsaved_edits = False
        points_layer = self._current_points_layer()
        if points_layer is not None:
            points_layer.events.data.connect(self._on_points_data_changed)
        self._update_navigation_state()

    def _open_video(self, video_path: Path):
        """Add a video to the viewer as an image layer."""
        self.viewer.open(str(video_path), plugin="napari_video")

    def _current_points_layer(self):
        """Return the current file's Points layer if still in the viewer."""
        viewer_layers = {id(_unwrap(ly)) for ly in self.viewer.layers}
        for layer in self._current_layers:
            if is_movement_points_layer(layer) and id(layer) in viewer_layers:
                return layer
        return None

    def _remove_current_layers(self):
        """Remove the layers of the current file from the viewer."""
        current_layers = {id(ly) for ly in self._current_layers}
        for layer in list(self.viewer.layers):
            if id(_unwrap(layer)) in current_layers:
                self.viewer.layers.remove(layer)
        self._current_layers = []
        self._has_unsaved_edits = False

    def _on_points_data_changed(self, event):
        """Flag the current file as edited when a point is moved or removed."""
        if event.action in (ActionType.CHANGED, ActionType.REMOVED):
            self._has_unsaved_edits = True
            self._update_navigation_state()

    def _save_edits(self) -> bool:
        """Save any unsaved edits to the current file as a netCDF file.

        Returns False only if there were edits that could not be saved.
        """
        points_layer = self._current_points_layer()
        if not self._has_unsaved_edits or points_layer is None:
            return True
        file_path = self.files[self.index]
        save_path = file_path.with_name(file_path.stem + EDITED_FILE_SUFFIX)
        try:
            save_points_layer(points_layer, save_path)
        except Exception as e:
            show_error(f"Failed to save edits to '{save_path}': {e}")
            return False
        self._has_unsaved_edits = False
        self._update_navigation_state()
        show_info(f"Saved edits to '{save_path}'.")
        return True

    def _update_navigation_state(self):
        """Update the status label and enable the buttons that apply."""
        self.previous_button.setEnabled(self.index > 0)
        self.next_button.setEnabled(0 <= self.index < len(self.files) - 1)
        self.save_edits_button.setEnabled(self._has_unsaved_edits)
        if not self.files:
            self.status_label.setText(NO_FOLDER_STATUS)
            return
        status = (
            f"{self.index + 1}/{len(self.files)}: "
            f"{self.files[self.index].name}"
        )
        if self.videos[self.index] is None:
            status += " (no matching video)"
        if self._has_unsaved_edits:
            status += " (unsaved edits)"
        self.status_label.setText(status)
