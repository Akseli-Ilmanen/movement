[![Python Version](https://img.shields.io/pypi/pyversions/movement.svg)](https://pypi.org/project/movement)
[![PyPI Version](https://img.shields.io/pypi/v/movement.svg)](https://pypi.org/project/movement)
[![Conda Forge Version](https://anaconda.org/conda-forge/movement/badges/version.svg)](https://anaconda.org/conda-forge/movement)
[![Downloads](https://pepy.tech/badge/movement)](https://pepy.tech/project/movement)
[![License](https://img.shields.io/badge/License-BSD_3--Clause-orange.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![CI](https://img.shields.io/github/actions/workflow/status/neuroinformatics-unit/movement/test_and_deploy.yml?label=CI)](https://github.com/neuroinformatics-unit/movement/actions)
[![codecov](https://codecov.io/gh/neuroinformatics-unit/movement/branch/main/graph/badge.svg?token=P8CCH3TI8K)](https://codecov.io/gh/neuroinformatics-unit/movement)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/neuroinformatics-unit/movement/gh-pages?filepath=notebooks/examples)
[![Code style: Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/format.json)](https://github.com/astral-sh/ruff)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![project chat](https://img.shields.io/badge/zulip-join_chat-brightgreen.svg)](https://neuroinformatics.zulipchat.com/#narrow/stream/406001-Movement/topic/Welcome!)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.12755724.svg)](https://zenodo.org/doi/10.5281/zenodo.12755724)

# movement

A Python toolbox for analysing animal body movements across space and time.

> [!Important]
> **This is a fork** of [neuroinformatics-unit/movement](https://github.com/neuroinformatics-unit/movement)
> that extends the napari plugin for correcting pose tracks across many
> short clips. See [What this fork adds](#what-this-fork-adds) below.
> Everything else is as upstream.

## What this fork adds

All of this lives in the napari plugin (`movement launch`), in the
"Load folder of tracked data" and "Edit tracked data" sections.

**Batch refinement of a folder of files**
([upstream issue #1118](https://github.com/neuroinformatics-unit/movement/issues/1118))
- *Load folder*: queue every tracked data file in a folder, including its
  subfolders, and step through them with *Previous* / *Next*
  (or `Shift-Left` / `Shift-Right`). Only the current file's layers are
  kept in the viewer.
- The matching video is loaded underneath each file. A video matches if
  its name (without the suffix) starts the file's name, e.g.
  `clip-1.mp4` for `clip-1DLC_resnet50_….h5`; separators and letter case
  are ignored if nothing matches exactly.
- Stepping away from a file saves any edits next to it as
  `<name>_edited.nc` (movement's netCDF format). The original files are
  never modified. Going back to a file reloads the original.
- *Load points only* skips the tracks layer, which is not needed for
  editing and makes files load faster.

**Interpolation between corrected frames**
- *Interpolate between anchors*: correct the frame before and after a run
  of misplaced frames, click those two frames on the edited-frames
  timeline, and the points in between are re-positioned.
- Methods: `linear`, `nearest`, `cubic` (a spline through the track's
  points outside the range) and `optical flow`, which follows the image
  content of the loaded video between the anchors with Lucas-Kanade
  tracking (forward and backward, blended so both anchors are hit).
- Shortcuts over the viewer: `L`, `N`, `C`, `O` pick the method and
  switch on anchor picking in one go, so a correction is
  "press `O`, click two frames on the timeline".
- *Interpolate between all edited points*: every corrected frame of the
  chosen keypoint(s) acts as an anchor, and each stretch between
  consecutive anchors is interpolated.
- *Undo last edit*: reverts the last drag or the last interpolation as a
  whole, up to 50 steps back. Deleting a point clears the history.


![](docs/source/_static/movement_overview.png)

## Quick install

Create and activate a conda environment with movement installed (including the GUI):
```bash
conda create -n movement-env -c conda-forge movement napari pyqt6
conda activate movement-env
```


> [!Note]
> Read the [documentation](https://movement.neuroinformatics.dev/latest) for more information, including [full installation instructions](https://movement.neuroinformatics.dev/latest/user_guide/installation.html) and [examples](https://movement.neuroinformatics.dev/latest/examples/index.html).

## Overview

Deep learning methods for motion tracking have revolutionised a range of
scientific disciplines, from neuroscience and biomechanics, to conservation
and ethology. Tools such as
[DeepLabCut](https://mlabofai.org/deeplabcut/) and
[SLEAP](https://sleap.ai/) now allow researchers to track animal movements
in videos with remarkable accuracy, without requiring physical markers.
However, there is still a need for standardised, easy-to-use methods
to process the tracks generated by these tools.

`movement` aims to provide a consistent, modular interface for analysing
motion tracks, enabling steps such as data cleaning, visualisation,
and motion quantification. We aim to support all popular animal tracking
frameworks and file formats.

Find out more on our [mission and scope](https://movement.neuroinformatics.dev/latest/community/mission-scope.html) statement and our [roadmap](https://movement.neuroinformatics.dev/latest/community/roadmaps.html).

<!-- Start Admonitions -->

> [!Tip]
> If you prefer analysing your data in R, we recommend checking out the
> [animovement](https://animovement.dev/) toolbox, which is similar in scope.
> We are working together with its developer
> to gradually converge on common data standards and workflows.

<!-- End Admonitions -->

## Join the movement

`movement` is made possible by the generous contributions of many [people](https://movement.neuroinformatics.dev/latest/community/people.html).

We welcome and encourage contributions in any form—whether it is fixing a bug, developing a new feature, or improving the documentation—as long as you follow our [code of conduct](CODE_OF_CONDUCT.md).

Go to our [community page](https://movement.neuroinformatics.dev/latest/community/index.html) to find out how to connect with us and get involved.


## Citation

If you use movement in your work, please cite the following Zenodo DOI:

> Nikoloz Sirmpilatze, Chang Huan Lo, Sofía Miñano, Brandon D. Peri, Dhruv Sharma, Laura Porta, Iván Varela & Adam L. Tyson (2024). neuroinformatics-unit/movement. Zenodo. https://zenodo.org/doi/10.5281/zenodo.12755724

## License
⚖️ [BSD 3-Clause](./LICENSE)

## Package template
This package layout and configuration (including pre-commit hooks and GitHub actions) have been copied from the [python-cookiecutter](https://github.com/neuroinformatics-unit/python-cookiecutter) template.
