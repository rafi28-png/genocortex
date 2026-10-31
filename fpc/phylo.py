"""
Alignment-free Neighbour-Joining phylogenetics & Newick export.
===============================================================

Pure-NumPy implementation of Saitou & Nei (1987) Neighbour-Joining on any
symmetric pairwise distance matrix (such as :func:`fpc.tasks.distance_matrix`).

For any additive tree metric, :func:`neighbor_joining` recovers the exact tree
topology and exact branch lengths to IEEE-754 floating-point precision, with
zero external phylogenetics dependencies required at runtime.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
import numpy as np

_UNSAFE_NEWICK_RE = re.compile(r"[\s():;,'\[\]]")


def _format_newick_label(name: str) -> str:
    """Format a taxon label safely for standard Newick parsers."""
    clean = name.strip()
    if not clean:
        return "unnamed"
    if _UNSAFE_NEWICK_RE.search(clean):
        # Replace single quotes and wrap in single quotes per Newick specification
        escaped = clean.replace("'", "_")
        return f"'{escaped}'"
    return clean


@dataclass
class PhyloNode:
    """A node in an unrooted or rooted phylogenetic tree."""
    name: str | None = None
    branch_length: float | None = None
    children: list["PhyloNode"] = field(default_factory=list)

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def leaf_names(self) -> list[str]:
        if self.is_leaf:
            return [self.name or ""]
        out: list[str] = []
        for ch in self.children:
            out.extend(ch.leaf_names())
        return out

    def _to_newick_body(self, precision: int = 6) -> str:
        if self.children:
            inner = ",".join(ch._to_newick_body(precision=precision) for ch in self.children)
            body = f"({inner})"
            if self.name:
                body += _format_newick_label(self.name)
        else:
            body = _format_newick_label(self.name or "")

        if self.branch_length is not None:
            body += f":{self.branch_length:.{precision}f}"
        return body

    def to_newick(self, precision: int = 6) -> str:
        """Render the tree as a standard ``;``-terminated Newick string."""
        return self._to_newick_body(precision=precision) + ";"


def neighbor_joining(
    D: np.ndarray,
    names: list[str],
    clamp_negative: bool = True,
) -> PhyloNode:
    """Construct a Neighbour-Joining phylogenetic tree (Saitou & Nei 1987).

    Parameters
    ----------
    D:
        ``(n, n)`` symmetric distance matrix.
    names:
        Sequence / taxon names of length ``n``.
    clamp_negative:
        If True (default), clamps negative branch lengths caused by non-additive
        empirical noise to ``0.0`` while preserving exact lengths on additive trees.
    """
    mat = np.asarray(D, dtype=np.float64)
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        raise ValueError(f"Distance matrix must be square 2-D, got shape {mat.shape}")
    n = mat.shape[0]
    if len(names) != n:
        raise ValueError(f"Expected {n} taxon names, got {len(names)}")
    if n == 0:
        raise ValueError("Cannot build phylogenetic tree from 0 taxa")
    if not np.all(np.isfinite(mat)):
        raise ValueError("Distance matrix contains non-finite (NaN or Inf) values")
    if n == 1:
        return PhyloNode(name=str(names[0]), branch_length=0.0)

    # Symmetrise and zero diagonal
    dist = 0.5 * (mat + mat.T)
    np.fill_diagonal(dist, 0.0)

    if n == 2:
        half = 0.5 * float(dist[0, 1])
        if clamp_negative:
            half = max(0.0, half)
        return PhyloNode(
            children=[
                PhyloNode(name=str(names[0]), branch_length=half),
                PhyloNode(name=str(names[1]), branch_length=half),
            ]
        )

    nodes: list[PhyloNode] = [PhyloNode(name=str(nm)) for nm in names]

    while len(nodes) > 3:
        m = len(nodes)
        r = dist.sum(axis=1)
        # Q_ij = (m - 2) * d_ij - r_i - r_j for i < j
        Q = (m - 2.0) * dist - r[:, None] - r[None, :]
        np.fill_diagonal(Q, np.inf)

        flat_idx = int(np.argmin(Q))
        i, j = divmod(flat_idx, m)
        if i > j:
            i, j = j, i

        dij = float(dist[i, j])
        delta_i = 0.5 * dij + (float(r[i]) - float(r[j])) / (2.0 * (m - 2.0))
        delta_j = dij - delta_i
        if clamp_negative:
            delta_i = max(0.0, delta_i)
            delta_j = max(0.0, delta_j)

        child_i = nodes[i]
        child_j = nodes[j]
        child_i.branch_length = delta_i
        child_j.branch_length = delta_j
        parent = PhyloNode(children=[child_i, child_j])

        # Compute distances from new parent node u to remaining active nodes k
        keep = [k for k in range(m) if k != i and k != j]
        d_new = 0.5 * (dist[i, keep] + dist[j, keep] - dij)
        if clamp_negative:
            d_new = np.maximum(0.0, d_new)

        sub = dist[np.ix_(keep, keep)]
        next_dist = np.zeros((m - 1, m - 1), dtype=np.float64)
        next_dist[:-1, :-1] = sub
        next_dist[-1, :-1] = d_new
        next_dist[:-1, -1] = d_new

        nodes = [nodes[k] for k in keep] + [parent]
        dist = next_dist

    # Exact 3-taxon star resolution for unrooted NJ tree
    d01 = float(dist[0, 1])
    d02 = float(dist[0, 2])
    d12 = float(dist[1, 2])
    b0 = 0.5 * (d01 + d02 - d12)
    b1 = 0.5 * (d01 + d12 - d02)
    b2 = 0.5 * (d02 + d12 - d01)
    if clamp_negative:
        b0 = max(0.0, b0)
        b1 = max(0.0, b1)
        b2 = max(0.0, b2)

    nodes[0].branch_length = b0
    nodes[1].branch_length = b1
    nodes[2].branch_length = b2
    return PhyloNode(children=[nodes[0], nodes[1], nodes[2]])


def patristic_distance_matrix(root: PhyloNode, names: list[str] | None = None) -> np.ndarray:
    """Compute exact leaf-to-leaf path-length matrix across a :class:`PhyloNode` tree."""
    if names is None:
        names = root.leaf_names()
    idx_map = {nm: idx for idx, nm in enumerate(names)}
    n = len(names)
    P = np.zeros((n, n), dtype=np.float64)

    def _collect(node: PhyloNode) -> list[tuple[int, float]]:
        """Return list of (leaf_index, dist_to_this_node) in subtree."""
        if node.is_leaf:
            if node.name in idx_map:
                return [(idx_map[node.name], 0.0)]
            return []
        sub_lists: list[list[tuple[int, float]]] = []
        for ch in node.children:
            bl = float(ch.branch_length or 0.0)
            sub = [(li, d + bl) for li, d in _collect(ch)]
            sub_lists.append(sub)
        for a_idx in range(len(sub_lists)):
            for b_idx in range(a_idx + 1, len(sub_lists)):
                for li, di in sub_lists[a_idx]:
                    for lj, dj in sub_lists[b_idx]:
                        P[li, lj] = P[lj, li] = di + dj
        merged: list[tuple[int, float]] = []
        for sub in sub_lists:
            merged.extend(sub)
        return merged

    _collect(root)
    return P


def newick_from_distance_matrix(
    D: np.ndarray,
    names: list[str],
    precision: int = 6,
    clamp_negative: bool = True,
) -> str:
    """Build a Neighbour-Joining tree from ``D`` and format it as Newick."""
    tree = neighbor_joining(D, names, clamp_negative=clamp_negative)
    return tree.to_newick(precision=precision)


def write_newick(tree_or_newick: PhyloNode | str, path: str, precision: int = 6) -> None:
    """Write a :class:`PhyloNode` or Newick string to disk."""
    nwk = (
        tree_or_newick.to_newick(precision=precision)
        if isinstance(tree_or_newick, PhyloNode)
        else str(tree_or_newick).strip()
    )
    if not nwk.endswith(";"):
        nwk += ";"
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(nwk + "\n")


def write_distance_tsv(
    D: np.ndarray,
    names: list[str],
    path: str,
    precision: int = 6,
) -> None:
    """Write a labeled symmetric distance matrix as tab-separated values."""
    mat = np.asarray(D, dtype=np.float64)
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        raise ValueError(f"Distance matrix must be square 2-D, got shape {mat.shape}")
    n = mat.shape[0]
    if len(names) != n:
        raise ValueError(f"Expected {n} names, got {len(names)}")
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("taxon\t" + "\t".join(names) + "\n")
        for i, nm in enumerate(names):
            vals = "\t".join(f"{mat[i, j]:.{precision}f}" for j in range(n))
            fh.write(f"{nm}\t{vals}\n")


__all__ = [
    "PhyloNode",
    "neighbor_joining",
    "patristic_distance_matrix",
    "newick_from_distance_matrix",
    "write_newick",
    "write_distance_tsv",
]
