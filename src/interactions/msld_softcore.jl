# The soft core of multi-site λ dynamics (J. Phys. Chem. B 2017, 121, 3626), which replaces the
# distance of a pair by a radius that stays finite as the pair is decoupled

export
    MSLDSoftCore,
    LennardJonesSoftCoreMSLD,
    CoulombSoftCoreMSLD

struct MSLDSoftCore <: SoftCore end

const msld_r_soft = 0.4u"nm"

# The softened radius and its derivative with respect to r, which the chain rule needs for the
# force. Below `r_cut` the radius flattens off to `r_cut / 2` at overlap, above it nothing changes.
# CHARMM gives the 1-4 pairs their own parameters and leaves them unsoftened
@inline function msld_radius(r, r_cut, special=false)
    if special || r >= r_cut
        return r, one(r / r_cut)
    end
    x = r / r_cut
    return r_cut * (x^3 - x^4 / 2 + one(x) / 2), (3 - 2 * x) * x^2
end

@inline msld_r_cut(inter, λ) = inter.r_soft * (1 - λ)

"""
    LennardJonesSoftCoreMSLD(; cutoff, r_soft, use_neighbors, shortcut, σ_mixing, ϵ_mixing,
                             λ_mixing, scheduler, weight_special)

The Lennard-Jones 6-12 interaction between two atoms with the multi-site λ dynamics soft core.
"""
@kwdef struct LennardJonesSoftCoreMSLD{C, R, H, S, E, LM, SCH, W} <: PairwiseInteraction
    cutoff::C = NoCutoff()
    r_soft::R = msld_r_soft
    use_neighbors::Bool = false
    shortcut::H = LJZeroShortcut()
    σ_mixing::S = LorentzMixing()
    ϵ_mixing::E = GeometricMixing()
    λ_mixing::LM = MinimumMixing()
    scheduler::SCH = DefaultLambdaScheduler()
    weight_special::W = 1
end

use_neighbors(inter::LennardJonesSoftCoreMSLD) = inter.use_neighbors

required_atom_fields(inter::LennardJonesSoftCoreMSLD) = (:σ, :ϵ, :λ, :alch_role,
            mixing_atom_fields(inter.σ_mixing)..., mixing_atom_fields(inter.ϵ_mixing)...,
            mixing_atom_fields(inter.λ_mixing)...)

function Base.zero(lj::LennardJonesSoftCoreMSLD)
    return LennardJonesSoftCoreMSLD(
        lj.cutoff,
        zero(lj.r_soft),
        lj.use_neighbors,
        lj.shortcut,
        lj.σ_mixing,
        lj.ϵ_mixing,
        lj.λ_mixing,
        lj.scheduler,
        zero(lj.weight_special),
    )
end

function Base.:+(l1::LennardJonesSoftCoreMSLD, l2::LennardJonesSoftCoreMSLD)
    return LennardJonesSoftCoreMSLD(
        l1.cutoff,
        l1.r_soft + l2.r_soft,
        l1.use_neighbors,
        l1.shortcut,
        l1.σ_mixing,
        l1.ϵ_mixing,
        l1.λ_mixing,
        l1.scheduler,
        l1.weight_special + l2.weight_special,
    )
end

parameter_prefix(::LennardJonesSoftCoreMSLD) = "inter_LJ_"
parameter_fields(::Type{<:LennardJonesSoftCoreMSLD}) =
    ((:r_soft, "r_soft"), (:weight_special, "weight_14"))

function to_lambda_function(inter::LennardJones, ::MSLDSoftCore; r_soft=msld_r_soft,
                            λ_mixing=MinimumMixing(), scheduler=DefaultLambdaScheduler(),
                            float_type=Float32, args...)
    return LennardJonesSoftCoreMSLD(cutoff=inter.cutoff,
                                    r_soft=float_type(ustrip(r_soft)) * unit(r_soft),
                                    use_neighbors=inter.use_neighbors, shortcut=inter.shortcut,
                                    σ_mixing=inter.σ_mixing, ϵ_mixing=inter.ϵ_mixing,
                                    λ_mixing=λ_mixing, scheduler=scheduler,
                                    weight_special=inter.weight_special)
end

@inline function force(inter::LennardJonesSoftCoreMSLD,
                       dr,
                       atom_i,
                       atom_j,
                       force_units=u"kJ * mol^-1 * nm^-1",
                       special=false,
                       args...)
    λ, λR, λ_params, σ, ϵ = λ_params_function(inter.scheduler, inter.λ_mixing, inter.σ_mixing,
                                              inter.ϵ_mixing, atom_i, atom_j, special)
    r = sqrt(sum(abs2, dr))
    if λ <= 0 || iszero_value(r) || shortcut_pair(inter.shortcut, atom_i, atom_j)
        return zero_pairwise_force(dr, force_units)
    end

    rp, drp_dr = msld_radius(r, msld_r_cut(inter, λ), special)
    f = λ * drp_dr * force_cutoff(inter.cutoff, inter, rp, (σ^2, ϵ))
    fdr = (f / r) * dr
    return special ? fdr * inter.weight_special : fdr
end

@inline function potential_energy(inter::LennardJonesSoftCoreMSLD,
                                  dr,
                                  atom_i,
                                  atom_j,
                                  energy_units=u"kJ * mol^-1",
                                  special=false,
                                  args...)
    λ, λR, λ_params, σ, ϵ = λ_params_function(inter.scheduler, inter.λ_mixing, inter.σ_mixing,
                                              inter.ϵ_mixing, atom_i, atom_j, special)
    if λ <= 0 || shortcut_pair(inter.shortcut, atom_i, atom_j)
        return zero_pairwise_energy(dr, energy_units)
    end

    r = sqrt(sum(abs2, dr))
    rp, drp_dr = msld_radius(r, msld_r_cut(inter, λ), special)
    if iszero_value(rp)
        return zero_pairwise_energy(dr, energy_units)
    end

    pe = λ * pe_cutoff(inter.cutoff, inter, rp, (σ^2, ϵ))
    return special ? pe * inter.weight_special : pe
end

pairwise_force(::LennardJonesSoftCoreMSLD, r, params) = pairwise_force(LennardJones(), r, params)
pairwise_pe(::LennardJonesSoftCoreMSLD, r, params) = pairwise_pe(LennardJones(), r, params)

"""
    CoulombSoftCoreMSLD(; cutoff, r_soft, use_neighbors, λ_mixing, scheduler, weight_special,
                        coulomb_const)

The Coulomb electrostatic interaction between two atoms with the multi-site λ dynamics soft core.
"""
@kwdef struct CoulombSoftCoreMSLD{C, R, LM, SCH, W, T} <: PairwiseInteraction
    cutoff::C = NoCutoff()
    r_soft::R = msld_r_soft
    use_neighbors::Bool = false
    λ_mixing::LM = MinimumMixing()
    scheduler::SCH = DefaultLambdaScheduler()
    weight_special::W = 1
    coulomb_const::T = coulomb_const
end

use_neighbors(inter::CoulombSoftCoreMSLD) = inter.use_neighbors

required_atom_fields(inter::CoulombSoftCoreMSLD) = (:charge, :λ, :alch_role,
            mixing_atom_fields(inter.λ_mixing)...)

function Base.zero(coul::CoulombSoftCoreMSLD)
    return CoulombSoftCoreMSLD(
        coul.cutoff,
        zero(coul.r_soft),
        coul.use_neighbors,
        coul.λ_mixing,
        coul.scheduler,
        zero(coul.weight_special),
        zero(coul.coulomb_const),
    )
end

function Base.:+(c1::CoulombSoftCoreMSLD, c2::CoulombSoftCoreMSLD)
    return CoulombSoftCoreMSLD(
        c1.cutoff,
        c1.r_soft + c2.r_soft,
        c1.use_neighbors,
        c1.λ_mixing,
        c1.scheduler,
        c1.weight_special + c2.weight_special,
        c1.coulomb_const + c2.coulomb_const,
    )
end

parameter_prefix(::CoulombSoftCoreMSLD) = "inter_CO_"
parameter_fields(::Type{<:CoulombSoftCoreMSLD}) =
    ((:r_soft, "r_soft"), (:weight_special, "weight_14"), (:coulomb_const, "coulomb_const"))

function to_lambda_function(inter::Coulomb, ::MSLDSoftCore; r_soft=msld_r_soft,
                            λ_mixing=MinimumMixing(), scheduler=DefaultLambdaScheduler(),
                            float_type=Float32, args...)
    return CoulombSoftCoreMSLD(cutoff=inter.cutoff,
                               r_soft=float_type(ustrip(r_soft)) * unit(r_soft),
                               use_neighbors=inter.use_neighbors, λ_mixing=λ_mixing,
                               scheduler=scheduler, weight_special=inter.weight_special,
                               coulomb_const=inter.coulomb_const)
end

@inline function force(inter::CoulombSoftCoreMSLD,
                       dr,
                       atom_i,
                       atom_j,
                       force_units=u"kJ * mol^-1 * nm^-1",
                       special=false,
                       args...)
    pair_role, λ, λR, λ_params, qij = softcore_pair_elec_lambda(inter, atom_i, atom_j, special)
    r = sqrt(sum(abs2, dr))
    if λ <= 0 || iszero_value(qij) || iszero_value(r)
        return zero_pairwise_force(dr, force_units)
    end

    rp, drp_dr = msld_radius(r, msld_r_cut(inter, λ), special)
    f = λ * drp_dr * force_cutoff(inter.cutoff, inter, rp, (inter.coulomb_const, qij))
    fdr = (f / r) * dr
    return special ? fdr * inter.weight_special : fdr
end

@inline function potential_energy(inter::CoulombSoftCoreMSLD,
                                  dr,
                                  atom_i,
                                  atom_j,
                                  energy_units=u"kJ * mol^-1",
                                  special=false,
                                  args...)
    pair_role, λ, λR, λ_params, qij = softcore_pair_elec_lambda(inter, atom_i, atom_j, special)
    if λ <= 0 || iszero_value(qij)
        return zero_pairwise_energy(dr, energy_units)
    end

    r = sqrt(sum(abs2, dr))
    rp, drp_dr = msld_radius(r, msld_r_cut(inter, λ), special)
    if iszero_value(rp)
        return zero_pairwise_energy(dr, energy_units)
    end

    pe = λ * pe_cutoff(inter.cutoff, inter, rp, (inter.coulomb_const, qij))
    return special ? pe * inter.weight_special : pe
end

pairwise_force(::CoulombSoftCoreMSLD, r, params) = pairwise_force(CoulombScaled(), r, params)
pairwise_pe(::CoulombSoftCoreMSLD, r, params) = pairwise_pe(CoulombScaled(), r, params)

pe_apply_cutoff(cutoff::CHARMMForceSwitchCutoff, inter::LennardJonesSoftCoreMSLD, r, params) =
    force_switch_pe_lj(cutoff, force_switch_lj_ab(params)..., r)
force_apply_cutoff(cutoff::CHARMMForceSwitchCutoff, inter::LennardJonesSoftCoreMSLD, r, params) =
    force_switch_force_lj(cutoff, force_switch_lj_ab(params)..., r)
pe_apply_cutoff(cutoff::CHARMMForceSwitchCutoff, inter::CoulombSoftCoreMSLD, r, params) =
    force_switch_pe_elec(cutoff, force_switch_q(params), r)
force_apply_cutoff(cutoff::CHARMMForceSwitchCutoff, inter::CoulombSoftCoreMSLD, r, params) =
    force_switch_force_elec(cutoff, force_switch_q(params), r)

# The 1-4 pairs of the force field keep their own parameters and take no soft core, as in CHARMM
function to_lambda_function(inter::LennardJones14Cutoff, ::MSLDSoftCore;
                            λ_mixing=MinimumMixing(), scheduler=DefaultLambdaScheduler(), args...)
    return to_lambda_function(inter, ScaledSoftCore(); λ_mixing=λ_mixing, scheduler=scheduler)
end

function to_lambda_function(inter::LennardJones14, ::MSLDSoftCore; λ_mixing=MinimumMixing(),
                            scheduler=DefaultLambdaScheduler(), args...)
    return to_lambda_function(inter, ScaledSoftCore(); λ_mixing=λ_mixing, scheduler=scheduler)
end
