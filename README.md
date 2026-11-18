# Reference-robust evaluation

Compare two segmentation models against the **same admissible reference**. Report whether either model remains ahead, whether admissible references reverse the ranking, or whether the evidence is unresolved at a declared score margin.

Version 0.4.0 provides three interfaces:

1. Exact decisions over supplied common-reference score records.
2. Exact extrema and independently checkable certificates over compiled Cartesian ROI-component tables.
3. A 3-D Boolean-mask compiler preserving union boundaries, area weights and bidirectional Surface Dice coverage.

## Install

Python 3.10 or newer. From this repository or the archived source directory:

```sh
python -m pip install .
# For the optional image-array compiler:
python -m pip install '.[geometry]'
```

This release is installable with pip from source; it is not a PyPI publication.
The two score/table CLIs use the Python standard library. Geometry requires NumPy and SciPy.

## Run the synthetic examples

```sh
reference-robust examples/common_reference_scores.csv --margin 1/100
reference-robust-pair solve examples/compiled_pair.json certificate.json
reference-robust-pair verify examples/compiled_pair.json certificate.json
python -m unittest discover -s tests -v
```

The score-record command covers only the supplied references. It does not infer that a sampled set contains every admissible reference. Ranking reversal requires opposite signs under two common references; overlapping separate model ranges are insufficient.

The paired solver uses exact integer/rational arithmetic, direct root support queries and lazy support-hull construction for branching. The independent checker imports no search routine. It verifies endpoint scores and a complete, prefix-free partition of the supplied joint choice space. The algorithms use standard hull preprocessing; this is not a new general optimization theory.

## Compile aligned 3-D masks

```python
from reference_robust.geometry import compile_pair
from reference_robust.adaptive_search import search
from reference_robust.verify_adaptive import verify

# A, B: same-shape boolean arrays; groups[j]: coherent alternatives for ROI j.
q = compile_pair(A, B, groups, spacing=(0.8, 0.8, 1.5), tolerance=1.0)
r = search(q["a"], q["b"], seconds=30, node_cap=1000000)
if r["complete"]:
    verify(q["a"], q["b"], r)
```

Grid registration, label semantics and admissibility are caller responsibilities. The interface accepts aligned 3-D arrays; it does not resample NIfTI files or infer shared-reader constraints. The declared metric uses stored binary64 area-table values and an exact tolerance predicate. It does not certify continuous anatomical surfaces. Paired optimization requires positive denominators and a common reference contribution. The public interface refuses a joint component above 4096 states; compilation and search can still be expensive. Count limits are not peak-memory guarantees.

The CLI refuses to overwrite a certificate. It uses a cooperative directory lock, a flushed temporary file, read-back checks and atomic replacement. Inspect stale locks after interruption; no automatic restart or retry occurs. An incomplete search cannot pass verification.

## Reproducibility and data

The examples and tests are synthetic. No clinical images, annotations, checkpoints, access credentials or unpublished manuscript are included. Obtain clinical resources from their original providers under their terms. See `docs/VALIDATION.md`, `docs/INPUT_CONTRACT.md` and `NOTICE`.

## License and citation

Apache License 2.0. Google DeepMind lookup-table attribution is retained in `NOTICE` and the source header. Cite the software using `CITATION.cff`. Research claims remain limited to the documented experiments; speed comparisons do not establish universal superiority over SCIP.

## Archived version

Version 0.4.0 is archived at https://doi.org/10.5281/zenodo.24326443. The source ZIP is byte-identical to the audited release package.

## Synthetic solver-limit benchmark

The [solver-limit benchmark](benchmarks/solver-limits/) includes the generated inputs, computation protocols, all 14 family outcomes, SCIP endpoint bounds, and independently checkable certificates.

Versioned benchmark archive: https://doi.org/10.5281/zenodo.24331572 (software 0.4.0 with benchmark supplement 1).
