"""Element-specific persistent-homology node featurizer for TopoQA interfaces.

Clean-room reimplementation from Han et al. (2025), *Briefings in Bioinformatics*
26(2), bbaf083 ("TopoQA"), reproduced from the paper — NOT the unlicensed source.

For each interface residue, atoms within ``NEIGHBOR_RADIUS`` of its Cα form a local
point cloud. Seven element selections are processed independently; each yields 20
persistence summaries (5 from VR H0 + 15 from alpha H1), for 7 x 20 = 140 features.

Conventions taken from the paper's §3 / feature description:
  - VR complex -> H0; alpha complex -> H1 (GUDHI backends).
  - H0 bars kept if death <= 8 (drops the one essential/infinite component) and
    lifetime >= 0.01. For VR H0 births are 0, so death == lifetime (the paper's
    "death or lifetime" for H0 is unambiguous here); the death values are summarized.
  - H1 bars kept if lifetime >= 0.01. Lifetime, birth, and death are each summarized.
  - Summary = (sum, min, max, mean, std) with population std (ddof=0); empty -> zeros.

Assumptions the paper does not pin down explicitly (documented, not guessed silently):
  - Alpha-complex filtration values are GUDHI's squared circumradii; we take sqrt so
    birth/death live on a distance (Å) scale, consistent with the 0.01 Å lifetime cut.
  - std is population std (ddof=0).
"""

from __future__ import annotations

import gudhi
import numpy as np

# Seven element selections, processed independently (paper §2 / Figure 2).
CHANNELS: tuple[frozenset[str], ...] = (
    frozenset({"C"}),
    frozenset({"N"}),
    frozenset({"O"}),
    frozenset({"C", "N"}),
    frozenset({"C", "O"}),
    frozenset({"N", "O"}),
    frozenset({"C", "N", "O"}),
)

NEIGHBOR_RADIUS = 8.0  # atoms within 8 Å of the residue Cα form the local cloud
H0_DEATH_MAX = 8.0     # H0 bars kept only if death <= 8
LIFETIME_MIN = 0.01    # all retained bars must have lifetime >= 0.01

_STATS_PER_SUMMARY = 5           # sum, min, max, mean, std
CHANNEL_WIDTH = 4 * _STATS_PER_SUMMARY   # H0(1 series) + H1(3 series) = 20
TOPO_WIDTH = len(CHANNELS) * CHANNEL_WIDTH  # 140


def _summary(values: np.ndarray) -> np.ndarray:
    """Five order-preserving summaries of a 1-D array; zeros for an empty array."""
    if values.size == 0:
        return np.zeros(_STATS_PER_SUMMARY)
    return np.array(
        [values.sum(), values.min(), values.max(), values.mean(), values.std()]
    )


def _h0_deaths(points: np.ndarray) -> np.ndarray:
    """Finite VR H0 death values, filtered to death <= 8 and lifetime >= 0.01."""
    if len(points) < 2:
        return np.empty(0)

    rips = gudhi.RipsComplex(points=points, max_edge_length=H0_DEATH_MAX)
    st = rips.create_simplex_tree(max_dimension=1)  # edges suffice for H0
    st.compute_persistence(persistence_dim_max=False)
    bars = st.persistence_intervals_in_dimension(0)
    if bars.size == 0:
        return np.empty(0)
    births, deaths = bars[:, 0], bars[:, 1]
    finite = np.isfinite(deaths)
    lifetimes = deaths - births
    # `deaths <= H0_DEATH_MAX` is belt-and-suspenders: max_edge_length already makes
    # any over-cap merge an infinite (dropped) bar, so `finite` alone suffices today.
    keep = finite & (deaths <= H0_DEATH_MAX) & (lifetimes >= LIFETIME_MIN)
    return deaths[keep]  # births are 0, so death == lifetime here


def _h1_intervals(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Alpha H1 (lifetime, birth, death) on a distance scale, lifetime >= 0.01."""
    empty = (np.empty(0), np.empty(0), np.empty(0))
    if len(points) < 4:  # an alpha H1 class needs >=4 points to enclose a void
        return empty

    alpha = gudhi.AlphaComplex(points=points)
    st = alpha.create_simplex_tree()  # GUDHI returns squared circumradii
    st.compute_persistence()
    bars = st.persistence_intervals_in_dimension(1)
    if bars.size == 0:
        return empty
    # clip guards against CGAL's inexact kernel emitting tiny-negative squared values
    births = np.sqrt(np.clip(bars[:, 0], 0.0, None))
    deaths = np.sqrt(np.clip(bars[:, 1], 0.0, None))
    lifetimes = deaths - births
    keep = np.isfinite(deaths) & (lifetimes >= LIFETIME_MIN)
    return lifetimes[keep], births[keep], deaths[keep]


def channel_features(points: np.ndarray) -> np.ndarray:
    """20 persistence summaries for one element channel's point cloud.

    Layout: [H0 death (5) | H1 lifetime (5) | H1 birth (5) | H1 death (5)].
    """
    points = np.asarray(points, dtype=float)
    h0 = _h0_deaths(points)
    life, birth, death = _h1_intervals(points)
    return np.concatenate(
        [_summary(h0), _summary(life), _summary(birth), _summary(death)]
    )


def residue_topology_features(
    elements: np.ndarray, coords: np.ndarray, ca_coord: np.ndarray
) -> np.ndarray:
    """140 topological features for one interface residue.

    Args:
        elements: (n,) array of element symbols for candidate atoms.
        coords: (n, 3) atom coordinates.
        ca_coord: (3,) coordinate of the residue's Cα.

    Atoms within ``NEIGHBOR_RADIUS`` of the Cα are gathered once, then each of the
    seven element channels is featurized on its subset.
    """
    elements = np.asarray(elements)
    coords = np.asarray(coords, dtype=float)
    ca_coord = np.asarray(ca_coord, dtype=float)

    if len(coords) == 0:
        return np.zeros(TOPO_WIDTH)

    within = np.linalg.norm(coords - ca_coord, axis=1) <= NEIGHBOR_RADIUS
    local_elements = elements[within]
    local_coords = coords[within]

    blocks = []
    for channel in CHANNELS:
        sel = np.array([e in channel for e in local_elements], dtype=bool)
        blocks.append(channel_features(local_coords[sel]))
    return np.concatenate(blocks)
