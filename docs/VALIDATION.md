# Validation and limits

Version 0.4.0 was checked on 256 new small integer families (19,185 exhaustive assignments) with 60 certificate-corruption challenges. The qualification covered root closure and branching (up to 23 visited nodes). A separate 3-D compiler integration test compared 18 whole-union assignments at three tolerances, with anisotropic spacing and mixed ROI domains, to direct distance enumeration.

A frozen synthetic confirmation panel contained 18 conditions: 24, 72, 144 or 288 ROIs; three alternatives or mixed two/three/four alternatives; isolated or four-ROI-block layouts; and two connected chains. Sixteen compiled and both chains were refused at the common 4096 component-state cap. Four methods were measured three times on every compiled condition, giving 192 new runs. Exact paired endpoints agreed for every completed run.

Summed per-condition median paired-call times were 0.406148 s (adaptive), 0.423026 s (eager hull), 0.406819 s (direct residual scan), and 2.323035 s (strengthened SCIP 10.0.2). Independent verification cost 0.502791 s for adaptive. Adaptive plus checking was faster than the SCIP call on all 16 compiled conditions. Adaptive call was faster than eager hull on 12/16. Maximum method-worker RSS was 74.69 MiB for adaptive and 277.92 MiB for SCIP; those figures exclude compilation. Compiler and input costs remain required end-to-end costs.

These are paired synthetic implementation measurements on one Mac, not evidence of universal solver superiority, clinical generalization or a polynomial-time algorithm. All four methods completed all 16 compiled conditions; no capacity advantage was established. Older failure conditions are retained in the research archive. The package contains no clinical data and no model weights.

Run `python -m unittest discover -s tests -v` for interface and geometry checks. Geometry checks require the optional dependencies. The exact search/checker only require the standard library.
