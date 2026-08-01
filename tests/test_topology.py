"""Red-to-green unit tests for the element-specific PH core.

No upstream oracle exists (the released code carries the `(x, y, y)` coordinate bug
and #5 retrains anyway), so these assert the paper spec + analytically-known
topology on hand-built point clouds, not bit-exact parity.
"""

import numpy as np
import pytest

from open_topoqa_featurizer.topology import (
    CHANNELS,
    CHANNEL_WIDTH,
    TOPO_WIDTH,
    _h0_deaths,
    _h1_intervals,
    channel_features,
    residue_topology_features,
)


def _jittered_ring(n=8, radius=3.0, seed=0):
    """n points on a slightly perturbed planar ring (avoids cocircular degeneracy)."""
    rng = np.random.RandomState(seed)
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    r = radius + rng.uniform(-0.05, 0.05, n)
    return np.column_stack([r * np.cos(theta), r * np.sin(theta)])


def test_layout_constants():
    assert len(CHANNELS) == 7
    assert CHANNEL_WIDTH == 20
    assert TOPO_WIDTH == 140


def test_h0_two_points_death_equals_distance():
    pts = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    deaths = _h0_deaths(pts)
    assert deaths.shape == (1,)  # one finite bar; the essential component is dropped
    assert deaths[0] == pytest.approx(3.0, abs=1e-9)


def test_h0_death_cap_and_lifetime_filter():
    # distance 9 > H0_DEATH_MAX -> never merges within the cap -> filtered out
    far = np.array([[0.0, 0.0, 0.0], [9.0, 0.0, 0.0]])
    assert _h0_deaths(far).size == 0
    # distance 0.005 < LIFETIME_MIN -> filtered out
    near = np.array([[0.0, 0.0, 0.0], [0.005, 0.0, 0.0]])
    assert _h0_deaths(near).size == 0


def test_h1_ring_single_loop():
    life, birth, death = _h1_intervals(_jittered_ring(radius=3.0))
    assert life.shape == (1,)  # a single dominant 1-cycle
    assert birth[0] > 0
    assert death[0] > birth[0]
    assert life[0] == pytest.approx(death[0] - birth[0], abs=1e-9)
    # Pins the 2*sqrt diameter-scale assumption: the hole fills at 2x the circumradius
    # (~6 for r=3), NOT the radius (~3) or the squared value (~9).
    assert death[0] == pytest.approx(6.0, abs=0.4)


def test_h1_has_no_death_cap():
    # Spec asymmetry vs H0: H1 bars have no death<=8 filter. An r=5 ring's hole fills at
    # diameter ~10 (> 8), so it must still be retained.
    life, birth, death = _h1_intervals(_jittered_ring(radius=5.0))
    assert life.shape == (1,)
    assert death[0] > 8.0  # would be dropped if an H0-style death cap leaked into H1
    assert death[0] == pytest.approx(10.0, abs=0.5)


def test_channel_features_shape_and_empty():
    assert channel_features(np.empty((0, 3))).shape == (CHANNEL_WIDTH,)
    assert np.all(channel_features(np.empty((0, 3))) == 0)
    # a single point yields no finite H0 bar and no H1 loop
    assert np.all(channel_features(np.array([[1.0, 2.0, 3.0]])) == 0)


def test_channel_features_h0_block():
    pts = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    feat = channel_features(pts)
    # H0 block = [sum, min, max, mean, std] of the single death value 3.0
    assert feat[:5] == pytest.approx([3.0, 3.0, 3.0, 3.0, 0.0], abs=1e-9)
    assert np.all(feat[5:] == 0)  # no H1


def test_residue_features_shape():
    elements = np.array(["C", "N", "O"])
    coords = np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0]])
    feat = residue_topology_features(elements, coords, ca_coord=np.zeros(3))
    assert feat.shape == (TOPO_WIDTH,)


def test_residue_features_element_selection():
    # O atoms form a ring; a lone N atom forms nothing.
    ring = _jittered_ring()
    ring3d = np.column_stack([ring, np.zeros(len(ring))])
    elements = np.array(["O"] * len(ring) + ["N"])
    coords = np.vstack([ring3d, [[0.5, 0.5, 0.0]]])
    feat = residue_topology_features(elements, coords, ca_coord=np.zeros(3))

    o_block = feat[2 * CHANNEL_WIDTH : 3 * CHANNEL_WIDTH]  # channel index 2 = {O}
    n_block = feat[1 * CHANNEL_WIDTH : 2 * CHANNEL_WIDTH]  # channel index 1 = {N}
    assert o_block[5] > 0  # {O} H1 lifetime sum nonzero (the ring closed a loop)
    assert np.all(n_block == 0)  # a single N atom contributes nothing


def test_residue_features_radius_cutoff():
    ca = np.zeros(3)
    inside = np.array([[3.0, 0, 0], [0, 0, 3.0]])  # both within 8 Å of Cα
    feat_in = residue_topology_features(np.array(["C", "C"]), inside, ca)
    assert feat_in[0] > 0  # {C} H0 sum nonzero (two atoms merge)

    outside = np.array([[3.0, 0, 0], [0, 0, 9.0]])  # second atom beyond 8 Å
    feat_out = residue_topology_features(np.array(["C", "C"]), outside, ca)
    assert np.all(feat_out[:CHANNEL_WIDTH] == 0)  # only one C remains -> zeros
