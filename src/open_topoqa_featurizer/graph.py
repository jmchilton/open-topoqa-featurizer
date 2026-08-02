"""Interface graph + node/edge features for the TopoQA featurizer.

Clean-room from Han et al. (2025), bbaf083 — reproduced from the paper, not the
unlicensed source. Assembles the residue-level interface graph:

  - Nodes = interface residues (Cα within 10 Å of a Cα on another chain).
  - Edges = cross-chain residue pairs with Cα separation < 10 Å (no intra-chain edges).
  - Node features (172) = 32 conventional + 140 topological (see ``topology``).
  - Edge features (11) = Cα distance + a 10-bin all-atom distance histogram.

The 32 conventional features are 21 one-hot amino acid + 8-state DSSP secondary
structure + relative SASA + normalized φ/ψ. The DSSP-derived pieces (SS, rel-SASA)
are *injected* into ``conventional_node_features`` so the graph/edge/assembly logic
is testable without the mkdssp binary; ``run_dssp`` is the thin wrapper that needs it.

Edge histogram uses real (x, y, z) atom coordinates — this is the corrected form of
the released code's (x, y, y) coordinate defect (correct by construction here).
"""

from __future__ import annotations

import os
import shutil
import tempfile

import numpy as np
from Bio.PDB import PDBIO, PDBParser
from Bio.PDB.Polypeptide import PPBuilder

from open_topoqa_featurizer.topology import residue_topology_features

CA_CUTOFF = 10.0          # interface / edge Cα distance cutoff (Å)
TOPO_ELEMENTS = frozenset({"C", "N", "O"})

# 20 standard amino acids + unknown -> 21-dim one-hot.
STANDARD_AA = (
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
)
_AA_INDEX = {name: i for i, name in enumerate(STANDARD_AA)}
AA_WIDTH = len(STANDARD_AA) + 1  # + unknown

# DSSP 8-state secondary-structure alphabet ('-' = coil/none).
DSSP_SS = "HBEGITS-"
_SS_INDEX = {c: i for i, c in enumerate(DSSP_SS)}
SS_WIDTH = len(DSSP_SS)

CONVENTIONAL_WIDTH = AA_WIDTH + SS_WIDTH + 1 + 2  # 21 + 8 + 1 (SASA) + 2 (φ,ψ) = 32
NODE_WIDTH = CONVENTIONAL_WIDTH + 140              # 172

# 10 histogram bins: [1,2), [2,3), ..., [9,10), [10, inf).
_HIST_EDGES = np.arange(1.0, 11.0)  # 1..10 -> 10 left edges (last bin is 10+)
EDGE_WIDTH = 1 + len(_HIST_EDGES)   # Cα distance + 10 bins = 11


def parse_structure(pdb_path: str):
    """Parse the first model of a PDB file (permissive parser)."""
    parser = PDBParser(QUIET=True)
    return parser.get_structure("complex", pdb_path)[0]


def _ca(residue):
    return residue["CA"].coord if residue.has_id("CA") else None


def interface_nodes(model, cutoff: float = CA_CUTOFF) -> list[tuple[str, object]]:
    """Interface residues: those whose Cα is within ``cutoff`` of another chain's Cα.

    Returns an ordered list of (chain_id, residue) for residues carrying a Cα.
    """
    residues = []  # (chain_id, residue, ca_coord)
    for chain in model:
        for residue in chain:
            ca = _ca(residue)
            if ca is not None:
                residues.append((chain.id, residue, ca))

    nodes = []
    for chain_id, residue, ca in residues:
        for other_id, _other_res, other_ca in residues:
            if other_id != chain_id and np.linalg.norm(ca - other_ca) <= cutoff:
                nodes.append((chain_id, residue))
                break
    return nodes


def residue_edges(
    nodes: list[tuple[str, object]], cutoff: float = CA_CUTOFF
) -> list[tuple[int, int]]:
    """Undirected cross-chain edges between node residues with Cα distance < cutoff."""
    cas = [_ca(res) for _cid, res in nodes]
    chains = [cid for cid, _res in nodes]
    edges = []
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            if chains[i] != chains[j] and np.linalg.norm(cas[i] - cas[j]) < cutoff:
                edges.append((i, j))
    return edges


def edge_features(res_i, res_j) -> np.ndarray:
    """11 edge features: Cα distance + 10-bin all-atom distance histogram.

    Distances are computed from real (x, y, z) coordinates (the corrected form of the
    released code's (x, y, y) defect). Bins are [1,2), ..., [9,10), [10, inf); pairs
    below 1 Å (none in practice) are not counted.
    """
    ca_dist = float(np.linalg.norm(_ca(res_i) - _ca(res_j)))
    coords_i = np.array([a.coord for a in res_i.get_atoms()], dtype=float)
    coords_j = np.array([a.coord for a in res_j.get_atoms()], dtype=float)
    pairwise = np.linalg.norm(
        coords_i[:, None, :] - coords_j[None, :, :], axis=2
    ).ravel()
    # bin index = floor(d) - 1 for d in [1,10); everything >=10 into the last bin.
    hist = np.zeros(len(_HIST_EDGES))
    idx = np.floor(pairwise).astype(int) - 1
    idx = np.clip(idx, -1, len(_HIST_EDGES) - 1)  # <1 -> -1 (dropped); >=10 -> last bin
    for k in idx:
        if k >= 0:
            hist[k] += 1
    return np.concatenate([[ca_dist], hist])


def amino_acid_one_hot(resname: str) -> np.ndarray:
    vec = np.zeros(AA_WIDTH)
    vec[_AA_INDEX.get(resname, AA_WIDTH - 1)] = 1.0  # unknown -> last slot
    return vec


def conventional_node_features(
    resname: str, ss8: str, rel_sasa: float, phi: float | None, psi: float | None
) -> np.ndarray:
    """32 conventional features from injected DSSP + backbone values.

    φ/ψ are given in radians (Biopython convention); normalized by π to (-1, 1].
    Missing angles (chain ends / breaks) -> 0. ``ss8`` is a single DSSP code.
    """
    ss = np.zeros(SS_WIDTH)
    ss[_SS_INDEX.get(ss8, _SS_INDEX["-"])] = 1.0
    phi_n = 0.0 if phi is None else phi / np.pi
    psi_n = 0.0 if psi is None else psi / np.pi
    return np.concatenate(
        [amino_acid_one_hot(resname), ss, [float(rel_sasa)], [phi_n, psi_n]]
    )


def _structure_cno_atoms(model) -> tuple[np.ndarray, np.ndarray]:
    """Elements and coords of all C/N/O atoms in the model (topology candidates)."""
    elements, coords = [], []
    for atom in model.get_atoms():
        el = atom.element.strip().upper()
        if el in TOPO_ELEMENTS:
            elements.append(el)
            coords.append(atom.coord)
    if not coords:
        return np.empty(0, dtype="<U2"), np.empty((0, 3))
    return np.array(elements), np.array(coords, dtype=float)


def phi_psi_map(model) -> dict:
    """Map residue.id (per chain) -> (phi, psi) in radians; ends/breaks -> (None, None)."""
    ppb = PPBuilder()
    out = {}
    for chain in model:
        for pp in ppb.build_peptides(chain):
            angles = pp.get_phi_psi_list()
            for residue, (phi, psi) in zip(pp, angles):
                out[(chain.id, residue.id)] = (phi, psi)
    return out


def run_dssp(model, pdb_path: str | None = None) -> dict:
    """Map (chain_id, res_id) -> (ss8_char, rel_sasa) via mkdssp.

    Requires the mkdssp binary. Raises RuntimeError if it is unavailable.

    DSSP runs on a copy of ``model`` re-serialized through Biopython's ``PDBIO``, not on the
    original file: mkdssp's strict fixed-column parser rejects some minimally-formatted decoy
    PDBs (unpadded records, residues it reports it "could not map"), while the round-tripped copy
    is canonical and parses cleanly. ``pdb_path`` is accepted for backward compatibility but
    ignored — the parsed ``model`` is the sole source of coordinates.
    """
    if shutil.which("mkdssp") is None and shutil.which("dssp") is None:
        raise RuntimeError("mkdssp/dssp binary not found on PATH")
    from Bio.PDB.DSSP import DSSP

    io = PDBIO()
    io.set_structure(model)
    fd, norm_path = tempfile.mkstemp(suffix=".pdb")
    os.close(fd)
    try:
        io.save(norm_path)
        dssp = DSSP(model, norm_path)
        out = {}
        for key in dssp.keys():
            chain_id, res_id = key
            _dssp_index, _aa, ss, rel_asa = dssp[key][:4]
            rel = 0.0 if rel_asa in ("NA", None) else float(rel_asa)
            out[(chain_id, res_id)] = (ss if ss != " " else "-", rel)
        return out
    finally:
        os.unlink(norm_path)


def featurize_complex(pdb_path: str, dssp_map: dict | None = None) -> dict:
    """Full interface graph for a complex.

    Runs mkdssp for the SS/rel-SASA node features unless ``dssp_map`` (a precomputed
    (chain_id, res_id) -> (ss8_char, rel_sasa) mapping) is supplied — the injection
    seam that lets the assembly run without the binary. Returns dict with node_ids
    [(chain, res_id)], node_features (N, 172), edge_index (E, 2), edge_features (E, 11).
    """
    model = parse_structure(pdb_path)
    nodes = interface_nodes(model)
    edges = residue_edges(nodes)
    elements, coords = _structure_cno_atoms(model)
    angles = phi_psi_map(model)
    dssp = run_dssp(model) if dssp_map is None else dssp_map

    node_feats, node_ids = [], []
    for chain_id, residue in nodes:
        key = (chain_id, residue.id)
        ss8, rel_sasa = dssp.get(key, ("-", 0.0))
        phi, psi = angles.get(key, (None, None))
        conv = conventional_node_features(
            residue.resname.strip(), ss8, rel_sasa, phi, psi
        )
        topo = residue_topology_features(elements, coords, _ca(residue))
        node_feats.append(np.concatenate([conv, topo]))
        node_ids.append(key)

    edge_feats = [
        edge_features(nodes[i][1], nodes[j][1]) for i, j in edges
    ]
    return {
        "node_ids": node_ids,
        "node_features": np.array(node_feats) if node_feats else np.empty((0, NODE_WIDTH)),
        "edge_index": np.array(edges) if edges else np.empty((0, 2), dtype=int),
        "edge_features": np.array(edge_feats) if edge_feats else np.empty((0, EDGE_WIDTH)),
    }
