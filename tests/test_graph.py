"""Tests for the interface graph, node/edge features, and full assembly.

The DSSP-dependent path (mkdssp binary) is exercised via the ``dssp_map`` injection
seam, so these run hermetically. A real end-to-end DSSP run is gated on mkdssp.
"""

import shutil
from pathlib import Path

import numpy as np
import pytest

from open_topoqa_featurizer.graph import (
    AA_WIDTH,
    CONVENTIONAL_WIDTH,
    EDGE_WIDTH,
    NODE_WIDTH,
    SS_WIDTH,
    amino_acid_one_hot,
    conventional_node_features,
    edge_features,
    featurize_complex,
    interface_nodes,
    parse_structure,
    residue_edges,
)

FIXTURE = str(Path(__file__).parent / "fixtures" / "complex_2fns.pdb")


@pytest.fixture(scope="module")
def model():
    return parse_structure(FIXTURE)


def test_layout_constants():
    assert AA_WIDTH == 21
    assert SS_WIDTH == 8
    assert CONVENTIONAL_WIDTH == 32
    assert NODE_WIDTH == 172
    assert EDGE_WIDTH == 11


def test_amino_acid_one_hot():
    ala = amino_acid_one_hot("ALA")
    assert ala.shape == (AA_WIDTH,)
    assert ala.sum() == 1.0 and ala[0] == 1.0
    unknown = amino_acid_one_hot("XYZ")  # non-standard -> last slot
    assert unknown[AA_WIDTH - 1] == 1.0 and unknown.sum() == 1.0


def test_conventional_features_dim_and_content():
    feat = conventional_node_features("LEU", ss8="H", rel_sasa=0.5, phi=np.pi, psi=None)
    assert feat.shape == (CONVENTIONAL_WIDTH,)
    assert feat[:AA_WIDTH].sum() == 1.0  # one AA
    ss = feat[AA_WIDTH : AA_WIDTH + SS_WIDTH]
    assert ss.sum() == 1.0 and ss[0] == 1.0  # 'H' is index 0
    assert feat[AA_WIDTH + SS_WIDTH] == 0.5  # rel SASA
    assert feat[-2] == pytest.approx(1.0)  # phi = pi -> pi/pi = 1.0
    assert feat[-1] == 0.0  # psi None -> 0


def test_conventional_unknown_ss_maps_to_coil():
    feat = conventional_node_features("ALA", ss8="?", rel_sasa=0.0, phi=None, psi=None)
    ss = feat[AA_WIDTH : AA_WIDTH + SS_WIDTH]
    assert ss[SS_WIDTH - 1] == 1.0  # '-' (coil) is the last slot


def test_interface_and_edges(model):
    nodes = interface_nodes(model)
    assert len(nodes) > 0
    edges = residue_edges(nodes)
    assert len(edges) > 0
    chains = [cid for cid, _res in nodes]
    # every edge is strictly cross-chain (no intra-chain edges)
    for i, j in edges:
        assert chains[i] != chains[j]


def test_edge_features_histogram(model):
    nodes = interface_nodes(model)
    i, j = residue_edges(nodes)[0]
    feat = edge_features(nodes[i][1], nodes[j][1])
    assert feat.shape == (EDGE_WIDTH,)
    assert feat[0] > 0  # Cα distance positive
    # histogram counts are non-negative; total <= all atom pairs
    hist = feat[1:]
    assert np.all(hist >= 0)
    n_pairs = len(list(nodes[i][1].get_atoms())) * len(list(nodes[j][1].get_atoms()))
    assert hist.sum() <= n_pairs


def test_edge_histogram_uses_xyz_not_xyy():
    # A synthetic pair separated purely in z: the (x,y,y) bug would zero the z
    # component and mis-bin; correct (x,y,z) puts the single pair at ~5 Å -> bin [5,6).
    from Bio.PDB.Atom import Atom
    from Bio.PDB.Residue import Residue

    def _res(coord):
        r = Residue((" ", 1, " "), "ALA", " ")
        a = Atom("CA", np.array(coord, dtype=float), 0.0, 1.0, " ", "CA", 1, "C")
        r.add(a)
        return r

    ri = _res([0.0, 0.0, 0.0])
    rj = _res([0.0, 0.0, 5.5])  # only z differs
    feat = edge_features(ri, rj)
    assert feat[0] == pytest.approx(5.5)  # Cα distance sees z
    # single atom pair at 5.5 Å -> bin index floor(5.5)-1 = 4 (the [5,6) bin)
    hist = feat[1:]
    assert hist[4] == 1.0 and hist.sum() == 1.0


def test_featurize_complex_shapes_via_injected_dssp(model):
    # Inject an empty DSSP map -> SS defaults to coil, rel-SASA 0; exercises the full
    # node (172) + edge (11) assembly without needing the mkdssp binary.
    graph = featurize_complex(FIXTURE, dssp_map={})
    n = len(graph["node_ids"])
    assert n > 0
    assert graph["node_features"].shape == (n, NODE_WIDTH)
    assert np.all(np.isfinite(graph["node_features"]))
    e = len(graph["edge_index"])
    assert graph["edge_index"].shape == (e, 2)
    assert graph["edge_features"].shape == (e, EDGE_WIDTH)


@pytest.mark.skipif(
    shutil.which("mkdssp") is None and shutil.which("dssp") is None,
    reason="mkdssp binary not available",
)
def test_featurize_complex_end_to_end_with_dssp():
    graph = featurize_complex(FIXTURE)
    n = len(graph["node_ids"])
    assert graph["node_features"].shape == (n, NODE_WIDTH)
    # with real DSSP some residues should be non-coil / have nonzero SASA
    ss_block = graph["node_features"][:, AA_WIDTH : AA_WIDTH + SS_WIDTH]
    assert ss_block.sum() == n  # exactly one SS state per node
