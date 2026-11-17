"""Attainable Surface Dice envelopes over independent unknown reference bits.

This prototype is resident, not a bounded-memory surface evaluator. It preserves
DeepMind's site/code/area-table representation, treats each stored binary64 area
as an exact dyadic constant, and uses an exact dyadic tolerance predicate.
Factor decomposition and Dinkelbach optimization are standard techniques.
No claim of bitwise identity with a floating EDT/KD reduction is made.

The only public operation is ``completion_envelope``. Resource limits refuse
rather than truncate the assignment space. Input masks are never modified.
"""
from fractions import Fraction
from itertools import product
from math import gcd, isfinite
from time import perf_counter

import numpy as np
from scipy import ndimage

try:
    from .vendor.surface_distance import lookup_tables
except ImportError:
    from vendor.surface_distance import lookup_tables


class CompletionEnvelopeError(ValueError):
    """Input violates the explicitly supported evaluation contract."""


class EnvelopeResourceLimit(RuntimeError):
    """Exact execution exceeds an explicit, nontruncating resource cap."""


# Alias for callers that prefer a candidate-specific exception name.
E4ResourceLimit = EnvelopeResourceLimit
_KERNEL = np.asarray([[[128, 64], [32, 16]], [[8, 4], [2, 1]]], dtype=np.uint8)


def _ratio(value):
    value = Fraction(value)
    return {"numerator": str(value.numerator), "denominator": str(value.denominator)}


def _validate(prediction, reference, unknown, spacing, tau):
    arrays = [np.asarray(x) for x in (prediction, reference, unknown)]
    if any(a.ndim != 3 or a.dtype != np.bool_ for a in arrays):
        raise CompletionEnvelopeError("prediction/reference/unknown must be 3-D bool arrays")
    if any(a.shape != arrays[0].shape for a in arrays[1:]):
        raise CompletionEnvelopeError("all masks must have the same grid shape")
    if any(n <= 0 for n in arrays[0].shape):
        raise CompletionEnvelopeError("every grid dimension must be positive")
    if np.any(arrays[1][arrays[2]]):
        raise CompletionEnvelopeError("reference[unknown] must be zero; known and unknown bits overlap")
    try:
        s = tuple(float(x) for x in spacing)
        t = float(tau)
    except (TypeError, ValueError, OverflowError) as exc:
        raise CompletionEnvelopeError("spacing and tau must be finite real scalars") from exc
    if len(s) != 3 or not all(isfinite(x) and x > 0 for x in s):
        raise CompletionEnvelopeError("spacing must contain three positive finite binary64 values")
    if not isfinite(t) or t < 0:
        raise CompletionEnvelopeError("tau must be finite and nonnegative")
    return (*arrays, s, t)


def _codes(mask):
    # At q, correlate reads mask[q - 1 + b]; only high-end padding is needed.
    padded = np.pad(mask.astype(np.uint8), ((0, 1),) * 3, mode="constant")
    return ndimage.correlate(padded, _KERNEL, mode="constant", cval=0, output=np.uint8)


def _weights(spacing):
    areas = np.asarray(lookup_tables.create_table_neighbour_code_to_surface_area(spacing), dtype=np.float64)
    if areas.shape != (256,) or not np.all(np.isfinite(areas)):
        raise CompletionEnvelopeError("area LUT is nonfinite or has an unsupported shape")
    if areas[0] != 0 or areas[255] != 0 or np.any(areas[1:255] <= 0):
        raise CompletionEnvelopeError("area LUT must have positive selected-site areas and zero empty/full codes")
    dyadics = [float(a).as_integer_ratio() for a in areas]
    denominator = max(d for _, d in dyadics)  # All denominators are powers of two.
    weights = [n * (denominator // d) for n, d in dyadics]
    return weights, denominator


def _exact_footprint(spacing, tau, shape, max_structure_volume):
    s = [Fraction.from_float(x) for x in spacing]
    t = Fraction.from_float(tau)
    radii = tuple(min(int(t // x), int(n)) for x, n in zip(s, shape))
    structure_shape = tuple(2 * r + 1 for r in radii)
    volume = int(structure_shape[0]) * int(structure_shape[1]) * int(structure_shape[2])
    if volume > max_structure_volume:
        raise EnvelopeResourceLimit(
            "exact footprint bounding volume %d exceeds max_structure_volume=%d (radii=%r)"
            % (volume, max_structure_volume, radii))
    squared = [x * x for x in s]
    t2 = t * t
    denominator = max([x.denominator for x in squared] + [t2.denominator])
    coefficients = [x.numerator * (denominator // x.denominator) for x in squared]
    threshold = t2.numerator * (denominator // t2.denominator)
    structure = np.zeros(structure_shape, dtype=np.bool_)
    offsets = []
    for x in range(-radii[0], radii[0] + 1):
        first = coefficients[0] * x * x
        for y in range(-radii[1], radii[1] + 1):
            partial = first + coefficients[1] * y * y
            if partial > threshold:
                continue
            for z in range(-radii[2], radii[2] + 1):
                if partial + coefficients[2] * z * z <= threshold:
                    offsets.append((x, y, z))
                    structure[x + radii[0], y + radii[1], z + radii[2]] = True
    return structure, offsets, radii


def _weighted_codes(codes, weights):
    counts = np.bincount(np.asarray(codes, dtype=np.uint8).ravel(), minlength=256)
    return sum(int(counts[c]) * weights[c] for c in range(1, 255))


class _DisjointSet:
    def __init__(self, size):
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union_scope(self, variables):
        variables = tuple(variables)
        if len(variables) < 2:
            return
        root = self.find(variables[0])
        for v in variables[1:]:
            other = self.find(v)
            if root == other:
                continue
            if self.rank[root] < self.rank[other]:
                root, other = other, root
            self.parent[other] = root
            if self.rank[root] == self.rank[other]:
                self.rank[root] += 1


def _solve(constant_n, constant_d, tables, maximize, max_iterations):
    """Finite exact Dinkelbach iteration over independent option tables.

    ``tables[i][a] = (N_i(a), D_i(a))``. Denominators for the combined
    assignment must be positive; the caller guarantees fixed prediction area.
    """
    lam = Fraction(constant_n, constant_d)
    for iteration in range(1, max_iterations + 1):
        num, den, choices = constant_n, constant_d, []
        for options in tables:
            best_index = 0
            n0, d0 = options[0]
            best = n0 * lam.denominator - lam.numerator * d0
            for index in range(1, len(options)):
                n, d = options[index]
                reduced = n * lam.denominator - lam.numerator * d
                if (reduced > best) if maximize else (reduced < best):
                    best, best_index = reduced, index
            n, d = options[best_index]
            num += n
            den += d
            choices.append(best_index)
        if den <= 0:
            raise ArithmeticError("nonpositive denominator in nonempty-prediction solve")
        residual = num * lam.denominator - lam.numerator * den
        ratio = Fraction(num, den)
        if residual == 0:
            return ratio, choices, iteration
        lam = ratio
    raise EnvelopeResourceLimit("exact fractional optimization exceeded max_solver_iterations=%d" % max_iterations)


def completion_envelope(prediction, reference, unknown, spacing, tau,
                        max_component_bits=12, witness_mask=True,
                        max_structure_volume=1_000_000,
                        max_factor_links=2_000_000,
                        max_neighbor_checks=20_000_000,
                        max_solver_iterations=10_000,
                        return_option_tables=False):
    """Return attainable minimum/maximum area-weighted Surface Dice.

    All unknown reference bits vary independently. ``reference[unknown]`` must
    equal zero. Assignment bit lists follow ``np.argwhere(unknown)`` order.
    ``lower``/``upper`` and loose bounds are JSON-ready rational dictionaries.
    ``witness_mask=False`` retains geometrically adjacent coverage dependencies
    even when a certain reference witness already makes coverage true. It is
    a correct, deliberately less-refined decomposition ablation.

    The resource caps concern exact prototype work, not a promised RSS budget.
    Footprint offsets, resident code/dilation volumes, per-site factor links,
    and enumerated option tables are all retained or allocated explicitly.
    """
    start = perf_counter()
    P, R, U, spacing, tau = _validate(prediction, reference, unknown, spacing, tau)
    limits = (max_component_bits, max_structure_volume, max_factor_links,
              max_neighbor_checks, max_solver_iterations)
    if any(not isinstance(v, (int, np.integer)) or isinstance(v, (bool, np.bool_)) or v < 1 for v in limits):
        raise CompletionEnvelopeError("all resource caps must be positive integers")
    if not isinstance(witness_mask, (bool, np.bool_)):
        raise CompletionEnvelopeError("witness_mask must be Boolean")
    unknown_coordinates = np.argwhere(U)
    n_unknown = len(unknown_coordinates)
    if not np.any(P):
        zero = [0] * n_unknown
        one = zero.copy()
        if np.any(R):
            low = high = Fraction(0)
        elif n_unknown:
            low, high = Fraction(0), Fraction(1)
            one[0] = 1
        else:
            low = high = Fraction(1)
        return {
            "lower": _ratio(low), "upper": _ratio(high),
            "lower_assignment": one, "upper_assignment": zero,
            "loose_lower": _ratio(low), "loose_upper": _ratio(high),
            "unknown_coordinates": unknown_coordinates.tolist(),
            "components": [],
            "stats": {"unknown_bits": n_unknown, "component_count": 0,
                      "max_component_bits": 0, "option_count": 0,
                      "options_per_component": [], "empty_prediction_special_case": True,
                      "compile_seconds": perf_counter() - start,
                      "enumeration_seconds": 0.0, "solve_seconds": 0.0,
                      "total_seconds": perf_counter() - start},
            "certificate": {"contract": "exact empty-mask wrapper", "attainable": True},
        }

    weights, area_denominator = _weights(spacing)
    structure, offsets, radii = _exact_footprint(spacing, tau, P.shape, max_structure_volume)
    p_codes, r_base, unknown_codes = _codes(P), _codes(R), _codes(U)
    r_possible = np.bitwise_or(r_base, unknown_codes)
    p_active = (p_codes != 0) & (p_codes != 255)
    always_active = (r_base != 0) & (r_possible != 255)
    fixed_p_hit = ndimage.binary_dilation(p_active, structure=structure, border_value=0)
    certain_prediction_hit = ndimage.binary_dilation(always_active, structure=structure, border_value=0)
    affected = unknown_codes != 0
    constant_d = _weighted_codes(p_codes, weights) + _weighted_codes(r_base[~affected], weights)
    constant_n = _weighted_codes(r_base[(~affected) & fixed_p_hit], weights)
    constant_n += _weighted_codes(p_codes[certain_prediction_hit], weights)

    # Every unknown voxel v affects exactly q=v+d, d in {0,1}^3.
    site_variables = {}
    for variable, voxel in enumerate(unknown_coordinates):
        for d in product((0, 1), repeat=3):
            q = tuple(int(voxel[j]) + d[j] for j in range(3))
            bit = int(_KERNEL[1 - d[0], 1 - d[1], 1 - d[2]])
            site_variables.setdefault(q, []).append((variable, bit))
    site_coordinates = sorted(site_variables)
    references = []
    for q in site_coordinates:
        references.append({"coordinate": q, "base": int(r_base[q]),
                           "terms": tuple(site_variables[q]), "hit_p": bool(fixed_p_hit[q])})

    # Generate only prediction sites near a variable reference site. This still
    # uses a finite exact footprint and has an explicit (not hidden) probe cap.
    checks = len(references) * len(offsets)
    if checks > max_neighbor_checks:
        raise EnvelopeResourceLimit("coverage compilation needs %d neighbor checks, cap=%d"
                                    % (checks, max_neighbor_checks))
    nearby = {}
    links = 0
    shape = p_codes.shape
    for reference_index, factor in enumerate(references):
        q = factor["coordinate"]
        for offset in offsets:
            target = (q[0] + offset[0], q[1] + offset[1], q[2] + offset[2])
            if any(target[j] < 0 or target[j] >= shape[j] for j in range(3)):
                continue
            if not p_active[target]:
                continue
            if witness_mask and certain_prediction_hit[target]:
                continue
            nearby.setdefault(target, []).append(reference_index)
            links += 1
            if links > max_factor_links:
                raise EnvelopeResourceLimit("coverage factor links exceed max_factor_links=%d"
                                            % max_factor_links)

    coverage = []
    for q in sorted(nearby):
        ref_indices = tuple(nearby[q])
        scope = tuple(sorted({v for ri in ref_indices for v, _ in references[ri]["terms"]}))
        certain = bool(certain_prediction_hit[q])
        weight = weights[int(p_codes[q])]
        if certain:
            # In the unmasked ablation this contribution is moved, not copied,
            # from the constant numerator into its retained dependency factor.
            constant_n -= weight
        coverage.append({"references": ref_indices, "scope": scope,
                         "certain": certain, "weight": weight})

    dsu = _DisjointSet(n_unknown)
    for factor in references:
        dsu.union_scope(v for v, _ in factor["terms"])
    for factor in coverage:
        dsu.union_scope(factor["scope"])
    grouped = {}
    for v in range(n_unknown):
        grouped.setdefault(dsu.find(v), []).append(v)
    components = sorted(grouped.values(), key=lambda vs: vs[0])
    maximum = max((len(vs) for vs in components), default=0)
    if maximum > max_component_bits:
        sizes = [len(vs) for vs in components]
        raise EnvelopeResourceLimit("dependency component has %d bits, cap=%d; component sizes=%r; witness_mask=%r"
                                    % (maximum, max_component_bits, sizes, bool(witness_mask)))
    component_for_variable = {v: ci for ci, vs in enumerate(components) for v in vs}
    ref_groups = [[] for _ in components]
    coverage_groups = [[] for _ in components]
    for ri, factor in enumerate(references):
        ref_groups[component_for_variable[factor["terms"][0][0]]].append(ri)
    for factor in coverage:
        coverage_groups[component_for_variable[factor["scope"][0]]].append(factor)
    compile_end = perf_counter()

    tables = []
    for ci, variables in enumerate(components):
        local_bit = {v: (1 << j) for j, v in enumerate(variables)}
        local_refs = []
        ref_lookup = {}
        for ri in ref_groups[ci]:
            factor = references[ri]
            ref_lookup[ri] = len(local_refs)
            local_refs.append((factor["base"], tuple((local_bit[v], bit) for v, bit in factor["terms"]), factor["hit_p"]))
        local_coverage = [
            (f["weight"], f["certain"], tuple(ref_lookup[ri] for ri in f["references"]))
            for f in coverage_groups[ci]
        ]
        options = []
        for assignment in range(1 << len(variables)):
            numerator, denominator = 0, 0
            active_sites = []
            for base, terms, hit_p in local_refs:
                code = base
                for assignment_bit, code_bit in terms:
                    if assignment & assignment_bit:
                        code |= code_bit
                weight = weights[code]
                denominator += weight
                if hit_p:
                    numerator += weight
                active_sites.append(code != 0 and code != 255)
            for weight, certain, ref_indices in local_coverage:
                if certain or any(active_sites[ri] for ri in ref_indices):
                    numerator += weight
            options.append((numerator, denominator))
        tables.append(options)
    enumeration_end = perf_counter()

    lower, lower_choices, lower_iterations = _solve(constant_n, constant_d, tables, False, max_solver_iterations)
    upper, upper_choices, upper_iterations = _solve(constant_n, constant_d, tables, True, max_solver_iterations)
    min_n = constant_n + sum(min(n for n, _ in opts) for opts in tables)
    max_n = constant_n + sum(max(n for n, _ in opts) for opts in tables)
    min_d = constant_d + sum(min(d for _, d in opts) for opts in tables)
    max_d = constant_d + sum(max(d for _, d in opts) for opts in tables)
    loose_lower = max(Fraction(0), Fraction(min_n, max_d))
    loose_upper = min(Fraction(1), Fraction(max_n, min_d))
    if not (0 <= loose_lower <= lower <= upper <= loose_upper <= 1):
        raise ArithmeticError("envelope/independent bounds violate the score contract")

    def expand(choices):
        bits = [0] * n_unknown
        for vs, assignment in zip(components, choices):
            for j, v in enumerate(vs):
                bits[v] = int(bool(assignment & (1 << j)))
        return bits

    result = {
        "lower": _ratio(lower), "upper": _ratio(upper),
        "lower_assignment": expand(lower_choices), "upper_assignment": expand(upper_choices),
        "loose_lower": _ratio(loose_lower), "loose_upper": _ratio(loose_upper),
        "unknown_coordinates": unknown_coordinates.tolist(),
        "components": [{"variables": vs, "option_count": len(options),
                        "lower_option": lo, "upper_option": hi}
                       for vs, options, lo, hi in zip(components, tables, lower_choices, upper_choices)],
        "certificate": {"contract": "exact dyadic stored-area and tolerance metric",
                        "attainable": True, "area_weight_denominator": str(area_denominator),
                        "constant_numerator_units": str(constant_n),
                        "constant_denominator_units": str(constant_d),
                        "lower_terminal_reduced_cost": "0", "upper_terminal_reduced_cost": "0",
                        "independent_numerator_min": str(min_n), "independent_numerator_max": str(max_n),
                        "independent_denominator_min": str(min_d), "independent_denominator_max": str(max_d)},
        "stats": {"unknown_bits": n_unknown, "component_count": len(components),
                  "max_component_bits": maximum, "option_count": sum(len(x) for x in tables),
                  "options_per_component": [len(x) for x in tables],
                  "reference_factors": len(references), "coverage_factors": len(coverage),
                  "coverage_factor_links": links, "neighbor_checks": checks,
                  "witness_mask": bool(witness_mask), "footprint_offsets": len(offsets),
                  "footprint_radii": list(radii), "empty_prediction_special_case": False,
                  "lower_solver_iterations": lower_iterations, "upper_solver_iterations": upper_iterations,
                  "compile_seconds": compile_end - start,
                  "enumeration_seconds": enumeration_end - compile_end,
                  "solve_seconds": perf_counter() - enumeration_end,
                  "total_seconds": perf_counter() - start},
    }
    if return_option_tables:
        result["option_tables"] = [[{"numerator_units": str(n), "denominator_units": str(d)}
                                   for n, d in options] for options in tables]
    return result
