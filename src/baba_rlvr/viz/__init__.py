"""Trajectory & gameplay visualizer for Baba Is You.

Renders worlds to PIL images and stitches them into animated GIFs / strips.
Used for:
  - The pitch-deck before/after demo video.
  - Inline Jupyter rendering in the Colab notebook.
  - Quick visual debugging of the engine and PCG output.
"""

from .renderer import (
    actions_from_strings,
    render_trajectory_gif,
    render_trajectory_strip,
    render_world,
    rollout_actions,
)

__all__ = [
    "actions_from_strings",
    "render_trajectory_gif",
    "render_trajectory_strip",
    "render_world",
    "rollout_actions",
]
