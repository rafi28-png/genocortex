"""
The 3-D cognitive map -- what the brain sees.
=============================================

Two interactive, fully **offline** 3-D views (Plotly is embedded inline, so the
HTML opens with no internet and no CDN):

* ``cognitive_map_html`` -- the *surprise landscape*. The fractal mind-space is
  the floor (x, y = Chaos Game coordinates); height and colour are how surprised
  the brain is at each location. Valleys are motifs the brain has mastered;
  peaks are novelty, mutations, or foreign DNA. The learned prototypes float
  above their territory, coloured by the symbol they predict -- you are quite
  literally looking at the contents of the brain's memory.

* ``phylogeny_3d_html`` -- an unsupervised 3-D "constellation" of organisms
  placed by the brain's cross-entropy distance (via MDS), joined by a minimum
  spanning tree. Clusters and lineages appear with no labels and no alignment.

Colours follow a validated, colourblind-safe design system: a single-hue blue
sequential ramp for surprise (magnitude), and a fixed-order categorical set for
the symbols / clusters (identity). Dark surface throughout.
"""

from __future__ import annotations

import numpy as np

import plotly.graph_objects as go

try:
    from scipy.ndimage import gaussian_filter
    _HAVE_NDIMAGE = True
except Exception:  # pragma: no cover
    _HAVE_NDIMAGE = False

from scipy.sparse.csgraph import minimum_spanning_tree

# -- design tokens (dark surface) ---------------------------------------
SURFACE = "#1a1a19"
PAGE = "#0d0d0d"
INK = "#ffffff"
INK2 = "#c3c2b7"
MUTED = "#898781"
GRID = "#2c2c2a"

# Sequential blue ramp for surprise: low (mastered) = dark -> recedes into the
# dark surface; high (surprising) = bright.
SURPRISE_SCALE = [
    [0.00, "#0d1b3a"],
    [0.20, "#104281"],
    [0.40, "#1c5cab"],
    [0.60, "#2a78d6"],
    [0.80, "#5598e7"],
    [1.00, "#cde2fb"],
]

# Fixed-order categorical hues (dark column), chosen well-separated for CVD.
CATEGORICAL = ["#3987e5", "#d95926", "#008300", "#9085e9",
               "#c98500", "#199e70", "#e66767", "#d55181"]


def _nan_smooth(grid: np.ndarray, mask: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian-smooth ``grid`` while respecting a validity ``mask`` (NaN-safe)."""
    if not _HAVE_NDIMAGE or sigma <= 0:
        return grid
    filled = np.where(mask, grid, 0.0)
    num = gaussian_filter(filled, sigma=sigma, mode="nearest")
    den = gaussian_filter(mask.astype(float), sigma=sigma, mode="nearest")
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(den > 1e-6, num / den, np.nan)
    return out


def _surprise_grid(points: np.ndarray, bits: np.ndarray, grid: int):
    """Mean surprise per Chaos-Game cell -> a (grid, grid) terrain + validity."""
    z = np.full((grid, grid), np.nan)
    valid = np.isfinite(bits)
    p = points[valid]
    b = bits[valid]
    if len(p) == 0:
        return z, np.zeros_like(z, dtype=bool)
    ij = np.clip((p * grid).astype(int), 0, grid - 1)
    acc = np.zeros((grid, grid))
    cnt = np.zeros((grid, grid))
    np.add.at(acc, (ij[:, 1], ij[:, 0]), b)
    np.add.at(cnt, (ij[:, 1], ij[:, 0]), 1.0)
    m = cnt > 0
    z[m] = acc[m] / cnt[m]
    return z, m


def cognitive_map_html(cortex, ids, out_path: str, grid: int = 96,
                       title: str = "Fractal Predictive Cortex - cognitive map",
                       smooth: float = 1.1, show_prototypes: bool = True,
                       max_prototypes_shown: int = 4000, png_path: str | None = None) -> str:
    """Render the surprise landscape + learned prototypes to a standalone HTML.

    Parameters
    ----------
    cortex:   a (trained) :class:`fpc.FractalPredictiveCortex`.
    ids:      the encoded sequence to visualise (int array).
    out_path: where to write the HTML.
    grid:     resolution of the terrain grid.
    smooth:   Gaussian smoothing sigma (in cells) for the terrain.
    """
    ids = np.asarray(ids)
    P = cortex.embedding.embed(ids)
    track = cortex.score(ids)
    z, mask = _surprise_grid(P, track.bits, grid)
    z = _nan_smooth(z, mask, smooth)

    xs = (np.arange(grid) + 0.5) / grid
    ys = (np.arange(grid) + 0.5) / grid

    surface = go.Surface(
        x=xs, y=ys, z=z,
        colorscale=SURPRISE_SCALE, connectgaps=True,
        opacity=0.98,
        colorbar=dict(title=dict(text="surprise<br>(bits)", font=dict(color=INK2)),
                      tickfont=dict(color=INK2), outlinewidth=0, len=0.6),
        contours={"z": {"show": True, "usecolormap": True,
                        "highlightcolor": "#ffffff", "project_z": False,
                        "width": 1}},
        name="surprise terrain", showscale=True,
        hovertemplate="CGR-x %{x:.3f}<br>CGR-y %{y:.3f}<br>surprise %{z:.2f} bits<extra></extra>",
    )
    data = [surface]

    if show_prototypes and cortex.n_prototypes > 0:
        pf = cortex.prototype_frame()
        m = len(pf["x"])
        idx = np.arange(m)
        if m > max_prototypes_shown:
            # keep the most-used prototypes
            idx = np.argsort(pf["usage"])[::-1][:max_prototypes_shown]
        gx = np.clip((pf["x"][idx] * grid).astype(int), 0, grid - 1)
        gy = np.clip((pf["y"][idx] * grid).astype(int), 0, grid - 1)
        zc = z[gy, gx]
        zc = np.where(np.isfinite(zc), zc, np.nanmin(z))
        lift = 0.04 * (np.nanmax(z) - np.nanmin(z) + 1e-9)
        sym_names = cortex.alphabet.symbols
        dominant = pf["dominant"][idx]
        for s in range(cortex.a):
            sel = dominant == s
            if not np.any(sel):
                continue
            usage = pf["usage"][idx][sel]
            size = 4 + 10 * (usage / (usage.max() + 1e-9))
            data.append(go.Scatter3d(
                x=pf["x"][idx][sel], y=pf["y"][idx][sel], z=zc[sel] + lift,
                mode="markers",
                marker=dict(size=size, color=CATEGORICAL[s % len(CATEGORICAL)],
                            line=dict(width=0.5, color=SURFACE), opacity=0.95),
                name=f"predicts '{sym_names[s]}'",
                hovertemplate=(f"prototype -> '{sym_names[s]}'<br>"
                               "CGR (%{x:.3f}, %{y:.3f})<extra></extra>"),
            ))

    fig = go.Figure(data=data)
    _style_scene(fig, title,
                 xaxis="Chaos-Game x", yaxis="Chaos-Game y", zaxis="surprise (bits)")
    return _write(fig, out_path, png_path)


def phylogeny_3d_html(coords: np.ndarray, labels, out_path: str,
                      names=None, distance_matrix: np.ndarray | None = None,
                      title: str = "Cross-entropy phylogeny (unsupervised, alignment-free)",
                      png_path: str | None = None) -> str:
    """3-D MDS constellation of organisms, coloured by cluster, joined by an MST.

    Parameters
    ----------
    coords:           (n, 3) MDS coordinates (see ``fpc.tasks.mds_embedding``).
    labels:           cluster / class id per point (drives colour).
    names:            optional hover names.
    distance_matrix:  optional (n, n) distances; if given, a minimum spanning
                      tree is drawn as faint edges (the "lineage skeleton").
    """
    coords = np.asarray(coords, dtype=float)
    n = coords.shape[0]
    if coords.shape[1] < 3:
        coords = np.column_stack([coords, np.zeros((n, 3 - coords.shape[1]))])
    labels = list(labels)
    names = list(names) if names is not None else [str(i) for i in range(n)]

    data = []
    # MST skeleton first (drawn under the nodes).
    if distance_matrix is not None:
        mst = minimum_spanning_tree(np.asarray(distance_matrix)).toarray()
        ex, ey, ez = [], [], []
        for i in range(n):
            for j in range(n):
                if mst[i, j] > 0:
                    ex += [coords[i, 0], coords[j, 0], None]
                    ey += [coords[i, 1], coords[j, 1], None]
                    ez += [coords[i, 2], coords[j, 2], None]
        data.append(go.Scatter3d(x=ex, y=ey, z=ez, mode="lines",
                                 line=dict(color=MUTED, width=2),
                                 name="lineage (MST)", hoverinfo="skip"))

    uniq = sorted(set(labels), key=lambda v: str(v))
    for k, lab in enumerate(uniq):
        sel = [i for i in range(n) if labels[i] == lab]
        data.append(go.Scatter3d(
            x=coords[sel, 0], y=coords[sel, 1], z=coords[sel, 2],
            mode="markers+text",
            text=[names[i] for i in sel],
            textposition="top center",
            textfont=dict(color=INK2, size=9),
            marker=dict(size=7, color=CATEGORICAL[k % len(CATEGORICAL)],
                        line=dict(width=0.5, color=SURFACE)),
            name=f"cluster {lab}",
            hovertemplate="%{text}<extra>cluster " + str(lab) + "</extra>",
        ))

    fig = go.Figure(data=data)
    _style_scene(fig, title, xaxis="MDS-1", yaxis="MDS-2", zaxis="MDS-3")
    return _write(fig, out_path, png_path)


# -- shared styling / output --------------------------------------------
def _style_scene(fig, title, xaxis, yaxis, zaxis):
    def ax(name):
        return dict(title=dict(text=name, font=dict(color=INK2)),
                    backgroundcolor=SURFACE, gridcolor=GRID, zerolinecolor=GRID,
                    color=MUTED, showbackground=True,
                    tickfont=dict(color=MUTED, size=9))
    fig.update_layout(
        title=dict(text=title, font=dict(color=INK, size=18), x=0.5, xanchor="center"),
        paper_bgcolor=PAGE, plot_bgcolor=PAGE,
        font=dict(family='system-ui, -apple-system, "Segoe UI", sans-serif', color=INK2),
        scene=dict(
            xaxis=ax(xaxis), yaxis=ax(yaxis), zaxis=ax(zaxis),
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.1)),
            aspectmode="cube",
        ),
        legend=dict(font=dict(color=INK2), bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=0, r=0, t=50, b=0),
    )


def _write(fig, out_path: str, png_path: str | None = None) -> str:
    # include_plotlyjs="inline" -> a fully self-contained, offline HTML file.
    fig.write_html(out_path, include_plotlyjs="inline", full_html=True,
                   config={"displaylogo": False, "responsive": True})
    if png_path:
        try:
            fig.write_image(png_path, width=1400, height=1000, scale=2)
        except Exception as exc:  # pragma: no cover - kaleido/chrome optional
            print(f"  (PNG export skipped: {exc})")
    return out_path
