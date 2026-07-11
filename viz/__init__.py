"""Visualisation layer (optional). Requires plotly; the core engine does not."""
from .viz3d import cognitive_map_html, phylogeny_3d_html

__all__ = ["cognitive_map_html", "phylogeny_3d_html"]
