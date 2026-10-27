# CHARMM force switching, as in the CHARMM/OpenMM interface (J. Comput. Chem. 2009, 30, 1545)

export
    CHARMMForceSwitchCutoff,
    LennardJones14Cutoff,
    LennardJones14CutoffScaled

struct CHARMMForceSwitchCutoff{P, D, L, E} <: AbstractCutoff{1}
    dist_activation::D
    dist_cutoff::D
    lj::L
    elec::E
end

function CHARMMForceSwitchCutoff(dist_activation, dist_cutoff)
    r_on, r_off = promote(dist_activation, dist_cutoff)
    if r_off <= r_on
        throw(ArgumentError("the cutoff radius $dist_cutoff must be larger than the " *
                            "activation radius $dist_activation"))
    end
    if r_on <= zero(r_on)
        throw(ArgumentError("the activation radius $dist_activation must be positive"))
    end
    on2, off2 = r_on^2, r_off^2
    on3, off3 = on2 * r_on, off2 * r_off
    on6, off6 = on3^2, off3^2
    lj = (
        onoff3 = inv(on3 * off3),
        onoff6 = inv(on6 * off6),
        ofdif3 = off3 / (off3 - on3),
        ofdif6 = off6 / (off6 - on6),
        recof3 = inv(off3),
        recof6 = inv(off6),
    )
    Δ = inv((off2 - on2)^3)
    a = off2^2 * (off2 - 3 * on2) * Δ
    b = 6 * on2 * off2 * Δ
    c = -(on2 + off2) * Δ
    d = 2 * Δ / 5
    elec = (
        a = a,
        b = b,
        c = c,
        d = d,
        eadd = 8 * Δ * (on2 * off2 * (r_off - r_on) - (off3 * off2 - on3 * on2) / 5),
        k = b * r_off - a / r_off + c * off3 + d * off3 * off2,
    )
    return CHARMMForceSwitchCutoff{1, typeof(r_on), typeof(lj), typeof(elec)}(r_on, r_off, lj, elec)
end

# The r^-12 and r^-6 terms are switched separately, so the parameters are needed as
#   A = 4ϵσ^12 and B = 4ϵσ^6 rather than as σ and ϵ
@inline function force_switch_lj_ab((σ2, ϵ))
    σ6 = σ2^3
    B = 4 * ϵ * σ6
    return B * σ6, B
end

@inline force_switch_q((ke, qi, qj)::Tuple{Any, Any, Any}) = ke * qi * qj
@inline force_switch_q((ke, qij)::Tuple{Any, Any}) = ke * qij

@inline function force_switch_pe_lj(cutoff, A, B, r)
    lj = cutoff.lj
    r3 = inv(r^3)
    r6 = r3^2
    if r <= cutoff.dist_activation
        return A * (r6^2 - lj.onoff6) - B * (r6 - lj.onoff3)
    else
        return A * lj.ofdif6 * (r6 - lj.recof6)^2 - B * lj.ofdif3 * (r3 - lj.recof3)^2
    end
end

@inline function force_switch_force_lj(cutoff, A, B, r)
    lj = cutoff.lj
    r3 = inv(r^3)
    r6 = r3^2
    if r <= cutoff.dist_activation
        return (12 * A * r6^2 - 6 * B * r6) / r
    else
        return (12 * A * lj.ofdif6 * (r6 - lj.recof6) * r6 -
                 6 * B * lj.ofdif3 * (r3 - lj.recof3) * r3) / r
    end
end

@inline function force_switch_pe_elec(cutoff, C, r)
    el = cutoff.elec
    if r <= cutoff.dist_activation
        return C * (inv(r) + el.eadd)
    else
        r2 = r^2
        return C * ((el.a - r2 * (el.b + r2 * (el.c + el.d * r2))) / r + el.k)
    end
end

@inline function force_switch_force_elec(cutoff, C, r)
    el = cutoff.elec
    if r <= cutoff.dist_activation
        return C / r^2
    else
        r2 = r^2
        return C * (el.a / r2 + el.b + 3 * el.c * r2 + 5 * el.d * r2^2)
    end
end

# The 1-4 Lennard-Jones pairs of a force field that gives them their own σ and ϵ are a specific
# interaction, so they need a version that takes a cutoff to be switched like every other pair
struct LennardJones14Cutoff{C, S, E, W}
    cutoff::C
    σ14_mixed::S
    ϵ14_mixed::E
    weight_14::W
end

function Base.zero(l::LennardJones14Cutoff)
    return LennardJones14Cutoff(l.cutoff, zero(l.σ14_mixed), zero(l.ϵ14_mixed),
                                zero(l.weight_14))
end

function Base.:+(l1::LennardJones14Cutoff, l2::LennardJones14Cutoff)
    return LennardJones14Cutoff(
        l1.cutoff,
        l1.σ14_mixed + l2.σ14_mixed,
        l1.ϵ14_mixed + l2.ϵ14_mixed,
        l1.weight_14 + l2.weight_14,
    )
end

parameter_prefix(::LennardJones14Cutoff, inter_type) = "inter_LJ14_$(inter_type)_"
parameter_fields(::Type{<:LennardJones14Cutoff}) =
    ((:σ14_mixed, "σ14"), (:ϵ14_mixed, "ϵ14"), (:weight_14, "weight_14"))

lj14_params(inter) = (inter.σ14_mixed^2, inter.ϵ14_mixed)

@inline function force(inter::LennardJones14Cutoff, coords_i, coords_l, boundary, args...)
    dr = vector(coords_i, coords_l, boundary)
    r = norm(dr)
    f = force_cutoff(inter.cutoff, inter, r, lj14_params(inter))
    fl = (inter.weight_14 * f / r) * dr
    return SpecificForce2Atoms(-fl, fl)
end

@inline function potential_energy(inter::LennardJones14Cutoff, coords_i, coords_l, boundary,
                                  args...)
    r = norm(vector(coords_i, coords_l, boundary))
    return inter.weight_14 * pe_cutoff(inter.cutoff, inter, r, lj14_params(inter))
end

pairwise_force(::LennardJones14Cutoff, r, params) = pairwise_force(LennardJones(), r, params)
pairwise_pe(::LennardJones14Cutoff, r, params) = pairwise_pe(LennardJones(), r, params)

# λ version of the above, with the pair energy scaled like any other Lennard-Jones pair
@kwdef struct LennardJones14CutoffScaled{C, S, E, W, LM, SCH}
    cutoff::C
    σ14_mixed::S
    ϵ14_mixed::E
    weight_14::W
    λ_mixing::LM = MinimumMixing()
    scheduler::SCH = DefaultLambdaScheduler()
end

function Base.zero(l::LennardJones14CutoffScaled)
    return LennardJones14CutoffScaled(l.cutoff, zero.(l.σ14_mixed), zero.(l.ϵ14_mixed),
                                      zero(l.weight_14), l.λ_mixing, l.scheduler)
end

function Base.:+(l1::LennardJones14CutoffScaled, l2::LennardJones14CutoffScaled)
    return LennardJones14CutoffScaled(
        l1.cutoff,
        l1.σ14_mixed .+ l2.σ14_mixed,
        l1.ϵ14_mixed .+ l2.ϵ14_mixed,
        l1.weight_14 + l2.weight_14,
        l1.λ_mixing,
        l1.scheduler,
    )
end

parameter_prefix(::LennardJones14CutoffScaled, inter_type) = "inter_LJ14_$(inter_type)_"
parameter_fields(::Type{<:LennardJones14CutoffScaled}) =
    ((:σ14_mixed, "σ14"), (:ϵ14_mixed, "ϵ14"), (:weight_14, "weight_14"))

function to_lambda_function(inter::LennardJones14Cutoff, ::ScaledSoftCore;
                            λ_mixing=MinimumMixing(), scheduler=DefaultLambdaScheduler(), args...)
    return LennardJones14CutoffScaled(cutoff=inter.cutoff, σ14_mixed=inter.σ14_mixed,
                                      ϵ14_mixed=inter.ϵ14_mixed, weight_14=inter.weight_14,
                                      λ_mixing=λ_mixing, scheduler=scheduler)
end

@inline function lj14_lambda_params(inter::LennardJones14CutoffScaled, atom_i, atom_l)
    λ_glob = λ_mixing(inter.λ_mixing, (atom_i, atom_l))
    pair_role = mix_roles(inter.scheduler, (atom_i.alch_role, atom_l.alch_role); lj=true)
    λ, λR, λ_params = scale_sterics_dual(inter.scheduler, λ_glob, pair_role)
    return λ, λR, params_mixing(λ_params, inter.σ14_mixed), params_mixing(λ_params, inter.ϵ14_mixed)
end

@inline function force(inter::LennardJones14CutoffScaled, coords_i, coords_l, boundary, atom_i,
                       atom_l, force_units, args...)
    dr = vector(coords_i, coords_l, boundary)
    λ, λR, σ, ϵ = lj14_lambda_params(inter, atom_i, atom_l)
    r = norm(dr)
    if λ <= 0 || iszero_value(r)
        return SpecificForce2Atoms(zero_pairwise_force(dr, force_units),
                                   zero_pairwise_force(dr, force_units))
    end

    f = λ * inter.weight_14 * force_cutoff(inter.cutoff, inter, r, (σ^2, ϵ))
    fl = (f / r) * dr
    return SpecificForce2Atoms(-fl, fl)
end

@inline function potential_energy(inter::LennardJones14CutoffScaled, coords_i, coords_l, boundary,
                                  atom_i, atom_l, energy_units, args...)
    dr = vector(coords_i, coords_l, boundary)
    λ, λR, σ, ϵ = lj14_lambda_params(inter, atom_i, atom_l)
    r = norm(dr)
    if λ <= 0 || iszero_value(r)
        return zero_pairwise_energy(dr, energy_units)
    end

    return λ * inter.weight_14 * pe_cutoff(inter.cutoff, inter, r, (σ^2, ϵ))
end

pairwise_force(::LennardJones14CutoffScaled, r, params) = pairwise_force(LennardJones(), r, params)
pairwise_pe(::LennardJones14CutoffScaled, r, params) = pairwise_pe(LennardJones(), r, params)

# The switched form differs for the Lennard-Jones and the Coulomb terms, so the cutoff has no
# generic version and each interaction is routed to the form its parameters take
const CHARMMForceSwitchLJ = Union{LennardJones, LennardJonesScaled, LennardJones14Cutoff,
                                  LennardJones14CutoffScaled}
const CHARMMForceSwitchCoulomb = Union{Coulomb, CoulombScaled}

pe_apply_cutoff(cutoff::CHARMMForceSwitchCutoff, inter::CHARMMForceSwitchLJ, r, params) =
    force_switch_pe_lj(cutoff, force_switch_lj_ab(params)..., r)
force_apply_cutoff(cutoff::CHARMMForceSwitchCutoff, inter::CHARMMForceSwitchLJ, r, params) =
    force_switch_force_lj(cutoff, force_switch_lj_ab(params)..., r)
pe_apply_cutoff(cutoff::CHARMMForceSwitchCutoff, inter::CHARMMForceSwitchCoulomb, r, params) =
    force_switch_pe_elec(cutoff, force_switch_q(params), r)
force_apply_cutoff(cutoff::CHARMMForceSwitchCutoff, inter::CHARMMForceSwitchCoulomb, r, params) =
    force_switch_force_elec(cutoff, force_switch_q(params), r)

function force_switch_unsupported(inter)
    throw(ArgumentError("CHARMMForceSwitchCutoff has no switched form for " *
                        "$(typeof(inter).name.name), it is only defined for the Lennard-Jones " *
                        "and Coulomb interactions"))
end

pe_apply_cutoff(::CHARMMForceSwitchCutoff, inter, r, params) = force_switch_unsupported(inter)
force_apply_cutoff(::CHARMMForceSwitchCutoff, inter, r, params) = force_switch_unsupported(inter)

# The 1-4 pairs with their own parameters, with or without a cutoff, as the alchemical code
# treats them alike
const LennardJones14Any = Union{LennardJones14, LennardJones14Cutoff}

# A cutoff that changes the form of the potential has to be carried into the 1-4 pairs, which are
# a specific interaction, so that they are switched like every other pair
lj14_interaction(cutoff, σ14, ϵ14, weight_14) = LennardJones14(σ14, ϵ14, weight_14)
lj14_interaction(cutoff::CHARMMForceSwitchCutoff, σ14, ϵ14, weight_14) =
    LennardJones14Cutoff(cutoff, σ14, ϵ14, weight_14)
