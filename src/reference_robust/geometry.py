"""Compile aligned 3-D boolean ROI alternatives. NIfTI alignment is caller-owned."""
from itertools import product
import numpy as np
from .vendor.surface_distance import lookup_tables
from .categorical_envelope import categorical_envelope
from .input_contract import validate

def boundary_codes(mask):
    a=np.pad(np.asarray(mask,dtype=np.uint8),1)
    result=np.zeros(tuple(n+1 for n in mask.shape),np.uint8)
    for b in product((0,1),repeat=3):
        view=a[tuple(slice(b[j],b[j]+mask.shape[j]+1) for j in range(3))]
        result |= view * int(lookup_tables.ENCODE_NEIGHBOURHOOD_3D_KERNEL[b])
    return result

def compile_pair(prediction_a,prediction_b,roi_alternatives,spacing,tolerance,component_cap=4096):
    """Return exact component tables for a Cartesian family of Boolean unions.

    Inputs must be pre-aligned boolean 3-D arrays. Spacing follows array axes.
    No cross-ROI rater constraint is inferred. Zero-denominator paired families
    and components over the declared cap are rejected, never truncated.
    """
    A=np.asarray(prediction_a);B=np.asarray(prediction_b)
    if A.ndim!=3 or A.dtype!=bool or B.dtype!=bool or A.shape!=B.shape or min(A.shape)<1:
        raise ValueError('Same-shape nonempty 3-D boolean prediction grids required')
    groups=[list(g) for g in roi_alternatives]
    if not groups or any(not g for g in groups):raise ValueError('At least one alternative per ROI required')
    for g in groups:
        for x in g:
            if np.asarray(x).dtype!=bool or np.asarray(x).shape!=A.shape:
                raise ValueError('ROI grids must match the boolean prediction grid')
    codes=[np.stack([boundary_codes(np.asarray(x)) for x in g]) for g in groups]
    C=np.zeros(codes[0].shape[1:],np.uint8);sites={}
    for v,a in enumerate(codes):
        common=np.bitwise_and.reduce(a,axis=0);C|=common
        possible=np.bitwise_or.reduce(a,axis=0)
        for q in np.argwhere(common!=possible):
            q=tuple(map(int,q));sites.setdefault(q,[]).append((v,tuple(map(int,a[(slice(None),)+q]))))
    kw=dict(spacing=spacing,tau=tolerance,max_component_states=component_cap,return_option_tables=True)
    domains=[len(g) for g in groups]
    result=dict(a=categorical_envelope(boundary_codes(A),C,sites,domains,**kw),
                b=categorical_envelope(boundary_codes(B),C,sites,domains,**kw))
    validate(result['a'],result['b'])
    return result
