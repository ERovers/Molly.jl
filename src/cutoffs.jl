# Cutoff strategies for long-range interactions

export
    NoCutoff,
    DistanceCutoff,
    ShiftedPotentialCutoff,
    ShiftedForceCutoff,
    CubicSplineCutoff

"""
    NoCutoff()

Placeholder cutoff that does not alter forces or potentials.
"""
struct NoCutoff end

cutoff_points(::Type{NoCutoff}) = 0

"""
    DistanceCutoff(dist_cutoff)

Cutoff that sets the potential and force to be zero past a specified cutoff point.
"""
struct DistanceCutoff{D, S, I}
    dist_cutoff::D
    sqdist_cutoff::S
    inv_sqdist_cutoff::I
end

function DistanceCutoff(dist_cutoff)
    return DistanceCutoff(dist_cutoff, dist_cutoff ^ 2, inv(dist_cutoff ^ 2))
end

cutoff_points(::Type{DistanceCutoff{D, S, I}}) where {D, S, I} = 1

force_divr_cutoff(::DistanceCutoff, r2, inter, params) = force_divr(inter, r2, inv(r2), params)
pe_apply_cutoff(::DistanceCutoff, inter, r2, params) = pairwise_pe(inter, r2, params)

"""
    ShiftedPotentialCutoff(dist_cutoff)

Cutoff that shifts the potential to be continuous at a specified cutoff point.
"""
struct ShiftedPotentialCutoff{D, S, I}
    dist_cutoff::D
    sqdist_cutoff::S
    inv_sqdist_cutoff::I
end

function ShiftedPotentialCutoff(dist_cutoff)
    return ShiftedPotentialCutoff(dist_cutoff, dist_cutoff ^ 2, inv(dist_cutoff ^ 2))
end

cutoff_points(::Type{ShiftedPotentialCutoff{D, S, I}}) where {D, S, I} = 1

function force_divr_cutoff(::ShiftedPotentialCutoff, r2, inter, params)
    return force_divr(inter, r2, inv(r2), params)
end

function pe_apply_cutoff(cutoff::ShiftedPotentialCutoff, inter, r2, params)
    pe_r = pairwise_pe(inter, r2, params)
    pe_cut = pairwise_pe(inter, cutoff.sqdist_cutoff, params)
    return pe_r - pe_cut
end

"""
    ShiftedForceCutoff(dist_cutoff)

Cutoff that shifts the force to be continuous at a specified cutoff point.
"""
struct ShiftedForceCutoff{D, S, I}
    dist_cutoff::D
    sqdist_cutoff::S
    inv_sqdist_cutoff::I
end

function ShiftedForceCutoff(dist_cutoff)
    return ShiftedForceCutoff(dist_cutoff, dist_cutoff ^ 2, inv(dist_cutoff ^ 2))
end

cutoff_points(::Type{ShiftedForceCutoff{D, S, I}}) where {D, S, I} = 1

function force_divr_cutoff(cutoff::ShiftedForceCutoff, r2, inter, params)
    return force_divr(inter, r2, inv(r2), params) -
           force_divr(inter, cutoff.sqdist_cutoff, cutoff.inv_sqdist_cutoff, params)
end

function pe_apply_cutoff(cutoff::ShiftedForceCutoff, inter, r2, params)
    r = sqrt(r2)
    fc = force_divr(inter, cutoff.sqdist_cutoff, cutoff.inv_sqdist_cutoff, params) * r
    pe_r = pairwise_pe(inter, r2, params)
    pe_cut = pairwise_pe(inter, cutoff.sqdist_cutoff, params)
    return pe_r + (r - cutoff.dist_cutoff) * fc - pe_cut
end

"""
    CubicSplineCutoff(dist_activation, dist_cutoff)

Cutoff that interpolates the true potential and zero between an activation point
and a cutoff point, using a cubic Hermite spline.
"""
struct CubicSplineCutoff{D, S, I}
    dist_activation::D
    dist_cutoff::D
    sqdist_activation::S
    inv_sqdist_activation::I
    sqdist_cutoff::S
    inv_sqdist_cutoff::I
end

function CubicSplineCutoff(dist_activation, dist_cutoff)
    if dist_cutoff <= dist_activation
        error("the cutoff radius must be strictly larger than the activation radius")
    end
    D, S, I = typeof(dist_cutoff), typeof(dist_cutoff^2), typeof(inv(dist_cutoff^2))
    return CubicSplineCutoff{D, S, I}(dist_activation, dist_cutoff, dist_activation^2,
                                      inv(dist_activation^2), dist_cutoff^2, inv(dist_cutoff^2))
end

cutoff_points(::Type{CubicSplineCutoff{D, S, I}}) where {D, S, I} = 2

function force_divr_cutoff(cutoff::CubicSplineCutoff, r2, inter, params)
    r = √r2
    t = (r - cutoff.dist_activation) / (cutoff.dist_cutoff - cutoff.dist_activation)
    Va = pairwise_pe(inter, cutoff.sqdist_activation, params)
    dVa = -force_divr(inter, cutoff.sqdist_activation, cutoff.inv_sqdist_activation, params) *
            cutoff.dist_activation
    return -((6t^2 - 6t) * Va / (cutoff.dist_cutoff-cutoff.dist_activation) + (3t^2 - 4t + 1) * dVa)/r
end

function pe_apply_cutoff(cutoff::CubicSplineCutoff, inter, r2, params)
    r = √r2
    t = (r - cutoff.dist_activation) / (cutoff.dist_cutoff-cutoff.dist_activation)
    Va = pairwise_pe(inter, cutoff.sqdist_activation, params)
    dVa = -force_divr(inter, cutoff.sqdist_activation, cutoff.inv_sqdist_activation, params) *
            cutoff.dist_activation
    return (2t^3 - 3t^2 + 1) * Va + (t^3 - 2t^2 + t) *
           (cutoff.dist_cutoff-cutoff.dist_activation) * dVa
end

Base.:+(c1::T, ::T) where {T <: Union{NoCutoff, DistanceCutoff, ShiftedPotentialCutoff,
                                      ShiftedForceCutoff, CubicSplineCutoff}} = c1

function force_divr_with_cutoff(inter, r2, params, cutoff::C, force_units) where C
    if cutoff_points(C) == 0
        return force_divr(inter, r2, inv(r2), params)
    elseif cutoff_points(C) == 1
        return force_divr_cutoff(cutoff, r2, inter, params) * (r2 <= cutoff.sqdist_cutoff)
    elseif cutoff_points(C) == 2
        return ifelse(
            r2 < cutoff.sqdist_activation,
            force_divr(inter, r2, inv(r2), params),
            force_divr_cutoff(cutoff, r2, inter, params) * (r2 <= cutoff.sqdist_cutoff),
        )
    end
end

function pe_cutoff(cutoff::C, inter, r2, params, energy_units) where C
    if cutoff_points(C) == 0
        return pairwise_pe(inter, r2, params)
    elseif cutoff_points(C) == 1
        return pe_apply_cutoff(cutoff, inter, r2, params) * (r2 <= cutoff.sqdist_cutoff)
    elseif cutoff_points(C) == 2
        return ifelse(
            r2 < cutoff.sqdist_activation,
            pairwise_pe(inter, r2, params),
            pe_apply_cutoff(cutoff, inter, r2, params) * (r2 <= cutoff.sqdist_cutoff),
        )
    end
end
