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
  analytically-known point clouds (`tests/test_topology.py`, 9 tests).
- ⬜ Interface extraction + residue graph (Cα < 10 Å across chains).
- ⬜ 32 conventional node features (21 one-hot AA, 8-state DSSP, relative SASA, normalized φ/ψ) → node dim 172.
- ⬜ 11 edge features (Cα distance + 10-bin all-atom distance histogram, `(x, y, z)`).

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

- **Alpha filtration is GUDHI's squared circumradius**; endpoints are `sqrt`-ed so
  birth/death live on a distance (Å) scale, consistent with the 0.01 Å lifetime cut.
  (Sanity: a radius-3 ring gives an H1 bar dying at ≈ 3.0 — the hole fills at the circumradius.)
- **std is population std (ddof=0).**
- For VR H0, births are 0, so "death or lifetime" (paper) is the same value; deaths are summarized.

## Develop

```
uv sync
uv run pytest -q
```
