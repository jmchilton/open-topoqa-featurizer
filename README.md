# open-topoqa-featurizer

Open **clean-room** reimplementation of the [TopoQA](https://doi.org/10.1093/bib/bbaf083)
interface featurizer (Han et al., 2025, *Briefings in Bioinformatics* 26(2), bbaf083),
built on GUDHI. MIT-licensed.

**Provenance.** Reproduced **from the paper**, its supplement, and its published
feature specification — **not** from the upstream code, which is unlicensed
(`yubingapril/TopoQA`) and was **not read, cloned, or decompiled**. The methods/math
of persistent homology are not copyrightable; this independent implementation is ours,
released MIT. Part of the [bio-topo-foundry](https://github.com/jmchilton/bio-topo-foundry)
cleanroom epic (#1), issue #4.

## Why a reimplementation exists

TopoQA's **scorer weights + inference code** ship without a software license, and the
released code carries a known coordinate defect (the all-atom edge histogram is built
as `(x, y, y)` rather than `(x, y, z)`). This package reproduces the **feature
computation from the paper spec** — correct by construction — so an open, retrainable
TopoQA vertical (foundry pipeline P1) does not depend on unlicensed code. Because the
correct featurizer diverges from the released checkpoint, matching upstream is a
**non-goal**; a retrained scorer is tracked separately (foundry #5).

## Status

- ✅ **Topological node core** (`open_topoqa_featurizer.topology`): the 140 element-specific
  persistent-homology features — 7 element channels × 20 summaries. Tested red-to-green on
  analytically-known point clouds.
- ✅ **Interface graph + node/edge features** (`open_topoqa_featurizer.graph`): interface
  extraction (Cα < 10 Å across chains), cross-chain edges, 32 conventional node features
  (21 one-hot AA · 8-state DSSP · relative SASA · normalized φ/ψ) → node dim **172**, and 11
  edge features (Cα distance + 10-bin all-atom histogram, correct `(x, y, z)`).
- ⬜ Packaging (foundry L1 recipe + env) and a retrained scorer (foundry #5).

```python
from open_topoqa_featurizer.graph import featurize_complex
g = featurize_complex("complex.pdb")   # runs mkdssp
g["node_features"]  # (N, 172)   g["edge_index"]  # (E, 2)   g["edge_features"]  # (E, 11)
```

**DSSP.** Secondary structure (8-state) and relative SASA both come from DSSP (paper-faithful,
single source), via `Bio.PDB.DSSP` → needs the `mkdssp` binary. The binary is required only for
that one step: `featurize_complex(pdb, dssp_map=...)` accepts a precomputed
`(chain, res_id) → (ss, rel_sasa)` map, the injection seam that lets the rest of the graph +
feature assembly (and its tests) run without mkdssp. The real-DSSP end-to-end test skips when the
binary is absent.

**Tests.** 18 pass + 1 gated (real DSSP); `tests/fixtures/complex_2fns.pdb` is a public 2-chain
interface structure exercising the geometry, edges, and 172-dim node assembly.

## The topological features (paper §3)

For each interface residue, atoms within **8 Å** of its Cα form a local cloud. Seven
element selections `{C} {N} {O} {C,N} {C,O} {N,O} {C,N,O}` are featurized independently:

| Source | Complex | Bars kept | Summaries |
|---|---|---|---|
| H0 | Vietoris–Rips | death ≤ 8, lifetime ≥ 0.01 | death: sum/min/max/mean/std (5) |
| H1 | alpha | lifetime ≥ 0.01 | lifetime, birth, death × 5 each (15) |

20 per channel × 7 = **140**.

```python
from open_topoqa_featurizer.topology import residue_topology_features
feat = residue_topology_features(elements, coords, ca_coord)  # -> (140,)
```

### Documented assumptions (paper does not pin these down)

- **Alpha filtration is GUDHI's squared circumradius**; endpoints are `2*sqrt`-ed onto a
  **diameter** distance (Å) scale. This places a single alpha edge at the full pairwise
  distance — the same length unit as a VR-H0 edge — so the one shared 0.01 Å lifetime cut
  applies consistently across H0 and H1. It also matches the sibling `open-topodockq-featurizer`.
  (Sanity: a radius-3 ring gives an H1 bar dying at ≈ 6.0 = 2× the circumradius.) The scorer is
  retrained (#5), so this scale is learnable and not load-bearing for accuracy — chosen for
  cross-featurizer consistency, not to match any oracle.
- **std is population std (ddof=0).**
- For VR H0, births are 0, so "death or lifetime" (paper) is the same value; deaths are summarized.

## Develop

```
uv sync
uv run pytest -q
```
