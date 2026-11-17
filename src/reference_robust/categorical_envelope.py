"""Exact attainable area-weighted Surface Dice over categorical ROI alternatives.

Inputs are pre-extracted site-code fields, not probabilistic segmentations.
Alternative ROI masks combine by Boolean union, hence their local codes combine
by bitwise OR. The operation preserves the dyadic stored-LUT/tolerance contract
of completion_envelope and reuses its standard exact ratio-table solver.
"""
from fractions import Fraction
from itertools import product
from math import isfinite, prod
from time import perf_counter

import numpy as np
from scipy import ndimage

try:
    from .completion_envelope import (CompletionEnvelopeError, EnvelopeResourceLimit,
        _weights, _exact_footprint, _weighted_codes, _DisjointSet, _solve, _ratio)
except ImportError:
    from completion_envelope import (CompletionEnvelopeError, EnvelopeResourceLimit,
        _weights, _exact_footprint, _weighted_codes, _DisjointSet, _solve, _ratio)


def _validated(pred_codes, common_codes, variable_sites, domains, spacing, tau):
    P, C = np.asarray(pred_codes), np.asarray(common_codes)
    if P.ndim != 3 or C.ndim != 3 or P.shape != C.shape or P.dtype != np.uint8 or C.dtype != np.uint8:
        raise CompletionEnvelopeError("prediction/common code grids must be same-shape 3-D uint8 arrays")
    if any(n < 2 for n in P.shape):
        raise CompletionEnvelopeError("site-code grid must have at least two sites along every axis")
    domains = list(domains)
    if any(not isinstance(d, (int, np.integer)) or isinstance(d, (bool, np.bool_)) or d < 1 for d in domains):
        raise CompletionEnvelopeError("domains must contain positive integer alternative counts")
    domains = [int(d) for d in domains]
    try:
        spacing = tuple(float(s) for s in spacing)
        tau = float(tau)
    except (ValueError, TypeError, OverflowError) as exc:
        raise CompletionEnvelopeError("spacing/tau must be finite real values") from exc
    if len(spacing) != 3 or not all(isfinite(s) and s > 0 for s in spacing) or not isfinite(tau) or tau < 0:
        raise CompletionEnvelopeError("need three positive finite spacings and finite nonnegative tau")
    if not hasattr(variable_sites, "items"):
        raise CompletionEnvelopeError("variable_sites must map coordinates to (variable, alternative codes) lists")
    references = []
    for coordinate, terms in variable_sites.items():
        q = tuple(coordinate)
        if len(q) != 3 or any(not isinstance(v, (int, np.integer)) or isinstance(v, (bool, np.bool_)) for v in q):
            raise CompletionEnvelopeError("site coordinates must be integer triples")
        q = tuple(int(v) for v in q)
        if any(q[j] < 0 or q[j] >= P.shape[j] for j in range(3)):
            raise CompletionEnvelopeError("variable site lies outside supplied code grid: %r" % (q,))
        base = int(C[q])
        combined = {}
        for variable, values in terms:
            if not isinstance(variable, (int, np.integer)) or isinstance(variable, (bool, np.bool_)) or not 0 <= variable < len(domains):
                raise CompletionEnvelopeError("invalid variable index at %r" % (q,))
            variable = int(variable)
            values = tuple(values)
            if len(values) != domains[variable] or any(not isinstance(v, (int, np.integer)) or isinstance(v, (bool, np.bool_)) or not 0 <= v <= 255 for v in values):
                raise CompletionEnvelopeError("alternative-code count/range mismatch for variable %d" % variable)
            previous = combined.setdefault(variable, [0] * domains[variable])
            for a, code in enumerate(values):
                previous[a] |= int(code)
        # A full code is immutable under OR: it never contributes a boundary.
        if base == 255:
            continue
        normalized = tuple((v, tuple(values)) for v, values in sorted(combined.items()))
        possible = base
        for _, values in normalized:
            for code in values:
                possible |= code
        if not normalized or possible == base:
            continue  # The fixed common code already represents this site.
        references.append({"coordinate": q, "base": base, "terms": normalized, "possible": possible})
    references.sort(key=lambda f: f["coordinate"])
    return P, C, references, domains, spacing, tau


def _optimize_empty_prediction(constant_d, components, tables, domains):
    """With no predicted surface, score is one exactly for empty references."""
    low_choices, high_choices = [], []
    min_d = max_d = constant_d
    for options in tables:
        lo = min(range(len(options)), key=lambda a: options[a][1])
        hi = max(range(len(options)), key=lambda a: options[a][1])
        min_d += options[lo][1]
        max_d += options[hi][1]
        high_choices.append(lo)
        low_choices.append(hi)
    lower = Fraction(1 if max_d == 0 else 0)
    upper = Fraction(1 if min_d == 0 else 0)
    return lower, upper, low_choices, high_choices, min_d, max_d


def categorical_envelope(pred_codes, common_codes, variable_sites, domains,
                         spacing, tau, max_component_states=59049,
                         witness_mask=True, max_structure_volume=1_000_000,
                         max_factor_links=2_000_000,
                         max_neighbor_checks=50_000_000,
                         max_solver_iterations=10_000,
                         return_option_tables=False):
    """Return exact min/max over Cartesian products of ROI alternative IDs.

    ``pred_codes`` and ``common_codes`` use the padded (n+1)-site grid.
    ``variable_sites[q]`` contains ``(variable, tuple(code_per_alternative))``.
    A completion code at q is common_codes[q] OR all chosen alternative codes.
    Codes need not be disjoint: OR correctly handles overlapping ROI masks.
    The caller is responsible for deriving site codes from valid same-grid masks.

    Assignments are alternative indices in domains order. Per-component option
    numbers use little-endian mixed radix, with the first variable varying fastest.
    Redundant variables may remain in a conservative dependency graph; no feasible
    alternative is discarded. Resource refusal is explicit, not approximate.
    """
    start = perf_counter()
    P, C, references, domains, spacing, tau = _validated(pred_codes, common_codes, variable_sites, domains, spacing, tau)
    caps = (max_component_states, max_structure_volume, max_factor_links, max_neighbor_checks, max_solver_iterations)
    if any(not isinstance(v, (int, np.integer)) or isinstance(v, (bool, np.bool_)) or v < 1 for v in caps):
        raise CompletionEnvelopeError("resource caps must be positive integers")
    if not isinstance(witness_mask, (bool, np.bool_)):
        raise CompletionEnvelopeError("witness_mask must be Boolean")
    weights, area_denominator = _weights(spacing)
    structure, offsets, radii = _exact_footprint(spacing, tau, tuple(n - 1 for n in P.shape), max_structure_volume)
    p_active = (P != 0) & (P != 255)
    p_empty = not np.any(p_active)
    affected = np.zeros(P.shape, dtype=np.bool_)
    always_active = (C != 0) & (C != 255)
    for f in references:
        q = f["coordinate"]
        affected[q] = True
        always_active[q] = f["base"] != 0 and f["possible"] != 255
    fixed_p_hit = ndimage.binary_dilation(p_active, structure=structure, border_value=0)
    certain_prediction_hit = ndimage.binary_dilation(always_active, structure=structure, border_value=0)
    prediction_area = _weighted_codes(P, weights)
    constant_d = prediction_area + _weighted_codes(C[~affected], weights)
    constant_n = _weighted_codes(C[(~affected) & fixed_p_hit], weights)
    constant_n += _weighted_codes(P[certain_prediction_hit], weights)
    for f in references:
        f["hit_p"] = bool(fixed_p_hit[f["coordinate"]])

    checks = 0 if p_empty else len(references) * len(offsets)
    if checks > max_neighbor_checks:
        raise EnvelopeResourceLimit("categorical coverage compilation needs %d checks, cap=%d" % (checks, max_neighbor_checks))
    nearby, links = {}, 0
    if not p_empty:
        for ri, f in enumerate(references):
            q = f["coordinate"]
            for delta in offsets:
                target = (q[0] + delta[0], q[1] + delta[1], q[2] + delta[2])
                if any(target[j] < 0 or target[j] >= P.shape[j] for j in range(3)) or not p_active[target]:
                    continue
                if witness_mask and certain_prediction_hit[target]:
                    continue
                nearby.setdefault(target, []).append(ri)
                links += 1
                if links > max_factor_links:
                    raise EnvelopeResourceLimit("categorical coverage factor links exceed %d" % max_factor_links)
    coverage = []
    for q in sorted(nearby):
        ref_indices = tuple(nearby[q])
        scope = tuple(sorted({v for ri in ref_indices for v, _ in references[ri]["terms"]}))
        certain = bool(certain_prediction_hit[q])
        weight = weights[int(P[q])]
        if certain:
            constant_n -= weight
        coverage.append({"references": ref_indices, "scope": scope, "certain": certain, "weight": weight})

    dsu = _DisjointSet(len(domains))
    for f in references:
        dsu.union_scope(v for v, _ in f["terms"])
    for f in coverage:
        dsu.union_scope(f["scope"])
    groups = {}
    for v in range(len(domains)):
        groups.setdefault(dsu.find(v), []).append(v)
    components = sorted(groups.values(), key=lambda vs: vs[0])
    states = [prod(domains[v] for v in vs) for vs in components]
    if any(n > max_component_states for n in states):
        raise EnvelopeResourceLimit("categorical component has %d states, cap=%d; sizes=%r, witness_mask=%r"
                                    % (max(states), max_component_states, states, bool(witness_mask)))
    which = {v: ci for ci, vs in enumerate(components) for v in vs}
    ref_groups, cov_groups = [[] for _ in components], [[] for _ in components]
    for ri, f in enumerate(references):
        ref_groups[which[f["terms"][0][0]]].append(ri)
    for f in coverage:
        cov_groups[which[f["scope"][0]]].append(f)
    compile_end = perf_counter()

    tables = []
    compressed_counts = []
    for ci, variables in enumerate(components):
        local_position = {v: j for j, v in enumerate(variables)}
        # Coordinates with identical code functions can be aggregated exactly.
        # Coverage uses their active Boolean function; duplicate witnesses are OR.
        signatures, signature_index, ref_lookup = [], {}, {}
        total_counts, hit_counts = [], []
        for ri in ref_groups[ci]:
            f = references[ri]
            key = (f["base"], tuple((local_position[v], codes) for v, codes in f["terms"]))
            if key not in signature_index:
                signature_index[key] = len(signatures)
                signatures.append(key)
                total_counts.append(0)
                hit_counts.append(0)
            si = signature_index[key]
            total_counts[si] += 1
            hit_counts[si] += int(f["hit_p"])
            ref_lookup[ri] = si
        cov_by_function = {}
        for f in cov_groups[ci]:
            key = (f["certain"], tuple(sorted({ref_lookup[ri] for ri in f["references"]})))
            cov_by_function[key] = cov_by_function.get(key, 0) + f["weight"]
        local_coverage = [(weight, certain, ref_indices)
                          for (certain, ref_indices), weight in cov_by_function.items()]
        sizes = [domains[v] for v in variables]
        options = []
        for assignment in range(states[ci]):
            value = assignment
            chosen = []
            for size in sizes:
                chosen.append(value % size)
                value //= size
            numerator, denominator = 0, 0
            active = []
            for si, (base, terms) in enumerate(signatures):
                code = base
                for position, alternatives in terms:
                    code |= alternatives[chosen[position]]
                weight = weights[code]
                denominator += weight * total_counts[si]
                numerator += weight * hit_counts[si]
                active.append(code != 0 and code != 255)
            for weight, certain, ref_indices in local_coverage:
                if certain or any(active[ri] for ri in ref_indices):
                    numerator += weight
            options.append((numerator, denominator))
        tables.append(options)
        compressed_counts.append({"reference_code_functions": len(signatures),
                                  "coverage_functions": len(local_coverage)})
    enumeration_end = perf_counter()

    if p_empty:
        lower, upper, lower_choices, upper_choices, min_d, max_d = _optimize_empty_prediction(constant_d, components, tables, domains)
        loose_lower, loose_upper = lower, upper
        min_n = max_n = 0
        low_iterations = high_iterations = 0
    else:
        lower, lower_choices, low_iterations = _solve(constant_n, constant_d, tables, False, max_solver_iterations)
        upper, upper_choices, high_iterations = _solve(constant_n, constant_d, tables, True, max_solver_iterations)
        min_n = constant_n + sum(min(n for n, _ in options) for options in tables)
        max_n = constant_n + sum(max(n for n, _ in options) for options in tables)
        min_d = constant_d + sum(min(d for _, d in options) for options in tables)
        max_d = constant_d + sum(max(d for _, d in options) for options in tables)
        loose_lower = max(Fraction(0), Fraction(min_n, max_d))
        loose_upper = min(Fraction(1), Fraction(max_n, min_d))
    if not (0 <= loose_lower <= lower <= upper <= loose_upper <= 1):
        raise ArithmeticError("categorical completion bounds violate score contract")

    def expand(choices):
        assignment = [0] * len(domains)
        for vs, option in zip(components, choices):
            for v in vs:
                assignment[v] = option % domains[v]
                option //= domains[v]
        return assignment

    result = {
        "lower": _ratio(lower), "upper": _ratio(upper),
        "lower_assignment": expand(lower_choices), "upper_assignment": expand(upper_choices),
        "loose_lower": _ratio(loose_lower), "loose_upper": _ratio(loose_upper),
        "domains": domains,
        "components": [{"variables": vs, "option_count": state,
                        "lower_option": lo, "upper_option": hi}
                       for vs, state, lo, hi in zip(components, states, lower_choices, upper_choices)],
        "certificate": {"contract": "exact dyadic stored-area and tolerance metric; categorical Boolean-union alternatives",
                        "attainable": True, "area_weight_denominator": str(area_denominator),
                        "constant_numerator_units": str(constant_n), "constant_denominator_units": str(constant_d),
                        "lower_terminal_reduced_cost": "0" if not p_empty else None,
                        "upper_terminal_reduced_cost": "0" if not p_empty else None,
                        "independent_numerator_min": str(min_n), "independent_numerator_max": str(max_n),
                        "independent_denominator_min": str(min_d), "independent_denominator_max": str(max_d)},
        "stats": {"variables": len(domains), "global_completions": str(prod(domains)),
                  "component_count": len(components), "max_component_variables": max((len(vs) for vs in components), default=0),
                  "max_component_states": max(states, default=0), "option_count": sum(states),
                  "options_per_component": states, "reference_factors": len(references),
                  "coverage_factors": len(coverage), "coverage_factor_links": links,
                  "compressed_factors": compressed_counts, "neighbor_checks": checks,
                  "witness_mask": bool(witness_mask), "footprint_offsets": len(offsets),
                  "footprint_radii": list(radii), "empty_prediction_special_case": bool(p_empty),
                  "lower_solver_iterations": low_iterations, "upper_solver_iterations": high_iterations,
                  "compile_seconds": compile_end - start,
                  "enumeration_seconds": enumeration_end - compile_end,
                  "solve_seconds": perf_counter() - enumeration_end,
                  "total_seconds": perf_counter() - start},
    }
    if return_option_tables:
        result["option_tables"] = [[{"numerator_units": str(n), "denominator_units": str(d)}
                                   for n, d in options] for options in tables]
    return result
