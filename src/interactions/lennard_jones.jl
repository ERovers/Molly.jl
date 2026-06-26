export
    LennardJones,
    LJDispersionCorrection,
    LJDispersionCorrectionλ,
    LennardJonesSoftCoreBeutler,
    LennardJonesSoftCoreGapsys,
    AshbaughHatch,
    LennardJones14,
    LennardJones14SoftCoreGapsys

@doc raw"""
    LennardJones(; cutoff, use_neighbors, shortcut, σ_mixing, ϵ_mixing, weight_special)

The Lennard-Jones 6-12 interaction between two atoms.

The potential energy is defined as
```math
V(r_{ij}) = 4\varepsilon_{ij} \left[\left(\frac{\sigma_{ij}}{r_{ij}}\right)^{12} - \left(\frac{\sigma_{ij}}{r_{ij}}\right)^{6}\right]
```
and the force on each atom by
```math
\begin{aligned}
\vec{F}_i &= 24\varepsilon_{ij} \left(2\frac{\sigma_{ij}^{12}}{r_{ij}^{13}} - \frac{\sigma_{ij}^6}{r_{ij}^{7}}\right) \frac{\vec{r}_{ij}}{r_{ij}} \\
&= \frac{24\varepsilon_{ij}}{r_{ij}^2} \left[2\left(\frac{\sigma_{ij}}{r_{ij}}\right)^{12} -\left(\frac{\sigma_{ij}}{r_{ij}}\right)^{6}\right] \vec{r}_{ij}
\end{aligned}
```

Should be used alongside the [`LJDispersionCorrection`](@ref) general interaction
when the long-range correction to the potential energy is required.
"""
@kwdef struct LennardJones{C, H, S, E, W} <: PairwiseInteraction
    cutoff::C = NoCutoff()
    use_neighbors::Bool = false
    shortcut::H = LJZeroShortcut()
    σ_mixing::S = LorentzMixing()
    ϵ_mixing::E = GeometricMixing()
    weight_special::W = 1
end

use_neighbors(inter::LennardJones) = inter.use_neighbors

function Base.zero(lj::LennardJones{C, H, S, E, W}) where {C, H, S, E, W}
    return LennardJones(
        lj.cutoff,
        lj.use_neighbors,
        lj.shortcut,
        lj.σ_mixing,
        lj.ϵ_mixing,
        zero(W),
    )
end

function Base.:+(l1::LennardJones, l2::LennardJones)
    return LennardJones(
        l1.cutoff,
        l1.use_neighbors,
        l1.shortcut,
        l1.σ_mixing,
        l1.ϵ_mixing,
        l1.weight_special + l2.weight_special,
    )
end

function inject_interaction(inter::LennardJones, params_dic)
    key_prefix = "inter_LJ_"
    return LennardJones(
        inter.cutoff,
        inter.use_neighbors,
        inter.shortcut,
        inter.σ_mixing,
        inter.ϵ_mixing,
        dict_get(params_dic, key_prefix * "weight_14", inter.weight_special),
    )
end

function extract_parameters!(params_dic, inter::LennardJones, ff)
    key_prefix = "inter_LJ_"
    params_dic[key_prefix * "weight_14"] = inter.weight_special
    return params_dic
end

@inline function force(inter::LennardJones,
                       dr,
                       atom_i,
                       atom_j,
                       force_units=u"kJ * mol^-1 * nm^-1",
                       special=false,
                       args...)
    if shortcut_pair(inter.shortcut, atom_i, atom_j, special)
        return zero_pairwise_force(dr, force_units)
    end
    σ = σ_mixing(inter.σ_mixing, atom_i, atom_j, special)
    ϵ = ϵ_mixing(inter.ϵ_mixing, atom_i, atom_j, special)

    cutoff = inter.cutoff
    r = sqrt(sum(abs2, dr))
    σ2 = σ^2
    params = (σ2, ϵ)

    f = force_cutoff(cutoff, inter, r, params)
    fdr = (f / r) * dr
    if special
        return fdr * inter.weight_special
    else
        return fdr
    end
end

function pairwise_force(::LennardJones, r, (σ2, ϵ))
    six_term = (σ2 / r^2) ^ 3
    return (24ϵ / r) * (2 * six_term ^ 2 - six_term)
end

@inline function potential_energy(inter::LennardJones,
                                  dr,
                                  atom_i,
                                  atom_j,
                                  energy_units=u"kJ * mol^-1",
                                  special=false,
                                  args...)
    if shortcut_pair(inter.shortcut, atom_i, atom_j, special)
        return ustrip(zero(dr[1])) * energy_units
    end
    σ = σ_mixing(inter.σ_mixing, atom_i, atom_j, special)
    ϵ = ϵ_mixing(inter.ϵ_mixing, atom_i, atom_j, special)

    cutoff = inter.cutoff
    r = sqrt(sum(abs2, dr))
    σ2 = σ^2
    params = (σ2, ϵ)

    pe = pe_cutoff(cutoff, inter, r, params)
    if special
        return pe * inter.weight_special
    else
        return pe
    end
end

function pairwise_pe(::LennardJones, r, (σ2, ϵ))
    six_term = (σ2 / r^2) ^ 3
    return 4ϵ * (six_term ^ 2 - six_term)
end

@doc raw"""
    LJDispersionCorrection(atoms, dist_cutoff, σ_mixing=LorentzMixing(),
                           ϵ_mixing=GeometricMixing())

The long-range dispersion correction for the [`LennardJones`](@ref) interaction.

Approximately represents contributions from beyond the cutoff distance.
Should be used alongside the [`LennardJones`](@ref) pairwise interaction when the long-range
correction to the potential energy is required.
The potential energy is defined as
```math
E = \frac{8 \pi N^2}{V} \left( \frac{\left< \epsilon_{ij} \sigma_{ij}^{12} \right>}{9 r_c^9} - \frac{\left< \epsilon_{ij} \sigma_{ij}^{6} \right>}{3 r_c^3} \right)
```
The forces are zero.

The number of atoms and atom σ and ϵ values are assumed not to change after setup (the box
volume can change).
Only compatible with 3D systems.
Not compatible with cutoffs other than [`DistanceCutoff`](@ref).
"""
struct LJDispersionCorrection{F, C}
    factor::F
    dist_cutoff::C
end

function LJDispersionCorrection(atoms, dist_cutoff, σ_mix, ϵ_mix)
    T = typeof(ustrip(dist_cutoff))
    n_atoms = length(atoms)
    atoms_cpu = from_device(atoms)
    at = atoms_cpu[1]
    ϵσ12_sum, ϵσ6_sum = zero(at.ϵ * at.σ^12), zero(at.ϵ * at.σ^6)
    for i in 1:n_atoms
        atom_i = atoms_cpu[i]
        for j in 1:i
            atom_j = atoms_cpu[j]
            σ = σ_mixing(σ_mix, atom_i, atom_j)
            ϵ = ϵ_mixing(ϵ_mix, atom_i, atom_j)
            ϵσ12_sum += ϵ * σ^12
            ϵσ6_sum  += ϵ * σ^6
        end
    end
    n_pairs = (n_atoms * (n_atoms + 1)) ÷ 2
    ϵσ12_mean = ϵσ12_sum / n_pairs
    ϵσ6_mean  = ϵσ6_sum  / n_pairs
    inner_term = (ϵσ12_mean / (9 * dist_cutoff^9) - ϵσ6_mean / (3 * dist_cutoff^3))
    factor = 8 * T(π) * n_atoms^2 * inner_term
    return LJDispersionCorrection(factor, dist_cutoff)
end

Base.zero(dc::LJDispersionCorrection) = LJDispersionCorrection(zero(dc.factor), zero(dc.dist_cutoff))

function Base.:+(dc1::LJDispersionCorrection, dc2::LJDispersionCorrection)
    return LJDispersionCorrection(dc1.factor + dc2.factor, dc1.dist_cutoff + dc2.dist_cutoff)
end

AtomsCalculators.@generate_interface function AtomsCalculators.potential_energy(sys,
                                                        inter::LJDispersionCorrection; kwargs...)
    return inter.factor / volume(sys)
end

AtomsCalculators.@generate_interface function AtomsCalculators.forces!(fs, sys,
                                                        inter::LJDispersionCorrection; kwargs...)
    return fs
end

@doc raw"""
    LJDispersionCorrectionλ(atoms, dist_cutoff, σ_mixing=LorentzMixing(),
                           ϵ_mixing=GeometricMixing())

The long-range dispersion correction for the [`LennardJones`](@ref) interaction scaled by λ for alchemical transformations.

Approximately represents contributions from beyond the cutoff distance.
Should be used alongside the [`LennardJones`](@ref) pairwise interaction when the long-range
correction to the potential energy is required.
The potential energy is defined as
```math
E = \frac{8 \pi N^2}{V} \left( \frac{\left< \epsilon_{ij} \sigma_{ij}^{12} \right>}{9 r_c^9} - \frac{\left< \epsilon_{ij} \sigma_{ij}^{6} \right>}{3 r_c^3} \right)
```
The forces are zero.

The number of atoms and atom σ and ϵ values are assumed not to change after setup (the box
volume can change).
Only compatible with 3D systems.
Not compatible with cutoffs other than [`DistanceCutoff`](@ref).
"""
struct LJDispersionCorrectionλ{F,D,PS,PE}
    factor::F
    dist_cutoff::D
    p_σ::PS
    p_ϵ::PE
end

function LJDispersionCorrectionλ(atoms, dist_cutoff, scheduler, λ_mix, σ_mix, ϵ_mix)
    T = typeof(ustrip(dist_cutoff))
    n_atoms = length(atoms)
    atoms_cpu = from_device(atoms)
    at = atoms_cpu[1]
    ϵσ12_sum, ϵσ6_sum = zero(at.ϵ * at.σ^12), zero(at.ϵ * at.σ^6)
    nλ_atoms = T(0)
    for i in 1:n_atoms
        atom_i = atoms_cpu[i]
        λ, λ_params = scale_sterics(scheduler, atom_i.λ, atom_i.alch_role)
        nλ_atoms += (atom_i.alch_role==CoreIRole || atom_i.alch_role==CoreDRole ? λ_params : λ)
        for j in 1:i
            atom_j = atoms_cpu[j]
            # Still have to figure out a better way of doing this, maybe include an 
            # eligibility matrix where the alchemical groups = false and rest true
            if atom_i.alch_role in [InsertRole, CoreIRole] && atom_j.alch_role in [DeleteRole, CoreDRole]
                continue
            elseif atom_i.alch_role in [DeleteRole, CoreDRole] && atom_j.alch_role in [InsertRole, CoreIRole]
                continue
            end

            λ_glob = T(λ_mixing(λ_mix, (atom_i.λ, atom_j.λ)))
            role_i = atom_i.alch_role
            role_j = atom_j.alch_role
            pair_role = mix_roles(scheduler, (role_i, role_j))
            λ, λ_params = scale_sterics(scheduler, λ_glob, pair_role)

            σ = λ_params * σ_mixing(σ_mix, atom_i, atom_j)
            ϵ = λ_params * ϵ_mixing(ϵ_mix, atom_i, atom_j)
            ϵσ12_sum += λ * ϵ * σ^12
            ϵσ6_sum  += λ * ϵ * σ^6
        end
    end
    n_pairs = (nλ_atoms * (nλ_atoms + 1)) ÷ 2
    ϵσ12_mean = ϵσ12_sum / n_pairs
    ϵσ6_mean  = ϵσ6_sum  / n_pairs
    inner_term = (ϵσ12_mean / (9 * dist_cutoff^9) - ϵσ6_mean / (3 * dist_cutoff^3))
    factor = 8 * T(π) * nλ_atoms^2 * inner_term
    return LJDispersionCorrectionλ(factor, dist_cutoff, σ_mix, ϵ_mix)
end

Base.zero(dc::LJDispersionCorrectionλ) = LJDispersionCorrectionλ(zero(dc.factor), zero(dc.cutoff), dc.p_σ, dc.p_ϵ)

function Base.:+(dc1::LJDispersionCorrectionλ, dc2::LJDispersionCorrectionλ)
    return LJDispersionCorrectionλ(dc1.factor + dc2.factor, dc1.cutoff + dc2.cutoff, dc1.p_σ, dc1.p_ϵ)
end

AtomsCalculators.@generate_interface function AtomsCalculators.potential_energy(sys,
                                                        inter::LJDispersionCorrectionλ; kwargs...)
    return inter.factor / volume(sys)
end

AtomsCalculators.@generate_interface function AtomsCalculators.forces!(fs, sys,
                                                        inter::LJDispersionCorrectionλ; kwargs...)
    return fs
end


@doc raw"""
    LennardJonesSoftCoreBeutler(; cutoff, α, λ, use_neighbors, shortcut, σ_mixing,
                                ϵ_mixing, weight_special)

The Lennard-Jones 6-12 interaction between two atoms with a soft core, used for
the appearing and disappearing of atoms.

See [Beutler et al. 1994](https://doi.org/10.1016/0009-2614(94)00397-1).
The potential energy is defined as
```math
V(r_{ij}) = \lambda \left(\frac{C^{(12)}}{r_{LJ}^{12}} - \frac{C^{(6)}}{r_{LJ}^{6}}\right)
```
and the force on each atom by
```math
\vec{F}_i = \lambda \left(\left(\frac{12C^{(12)}}{r_{LJ}^{13}} - \frac{6C^{(6)}}{r_{LJ}^7}\right)\left(\frac{r_{ij}}{r_{LJ}}\right)^5\right) \frac{\vec{r_{ij}}}{r_{ij}}
```
where
```math
r_{LJ} = \left(\frac{\alpha(1-\lambda)C^{(12)}}{C^{(6)}}+r^6\right)^{1/6}
```
and
```math
C^{(12)} = 4\epsilon\sigma^{12}
C^{(6)} = 4\epsilon\sigma^{6}
```

If ``\lambda`` is 1.0, this gives the standard [`LennardJones`](@ref) potential and means
the atom is fully turned on.
If ``\lambda`` is zero the interaction is turned off.
``\alpha`` determines the strength of softening the function.
"""
@kwdef struct LennardJonesSoftCoreBeutler{C, A, H, S, E, LM, SCH, W} <: PairwiseInteraction
    cutoff::C = NoCutoff()
    α::A = 1.0
    use_neighbors::Bool = false
    shortcut::H = LJZeroShortcut()
    σ_mixing::S = LorentzMixing()
    ϵ_mixing::E = GeometricMixing()
    λ_mixing::LM = MinimumMixing()
    scheduler::SCH = DefaultLambdaScheduler()
    weight_special::W = 1
end

use_neighbors(inter::LennardJonesSoftCoreBeutler) = inter.use_neighbors

function Base.zero(lj::LennardJonesSoftCoreBeutler{C, A, H, S, E, LM, SCH, W}) where {C, A, H, S, E, LM, SCH, W}
    return LennardJonesSoftCoreBeutler(
        lj.cutoff,
        zero(A),
        lj.use_neighbors,
        lj.shortcut,
        lj.σ_mixing,
        lj.ϵ_mixing,
        lj.λ_mixing,
        lj.scheduler,
        zero(W),
    )
end

function Base.:+(l1::LennardJonesSoftCoreBeutler, l2::LennardJonesSoftCoreBeutler)
    return LennardJonesSoftCoreBeutler(
        l1.cutoff,
        l1.α + l2.α,
        l1.use_neighbors,
        l1.shortcut,
        l1.σ_mixing,
        l1.ϵ_mixing,
        l1.λ_mixing,
        l1.scheduler,
        l1.weight_special + l2.weight_special
    )
end

function to_lambda_function(inter::LennardJones, gapsys::Val{false}; α=T(1.0), λ_mixing=MinimumMixing(), scheduler=DefaultLambdaScheduler())
    return LennardJonesSoftCoreBeutler(cutoff=inter.cutoff, α=α, use_neighbors=inter.use_neighbors, shortcut=inter.shortcut, 
                                        σ_mixing=inter.σ_mixing, ϵ_mixing=inter.ϵ_mixing, λ_mixing=λ_mixing, scheduler=scheduler, 
                                        weight_special=inter.weight_special)
end

@inline function force(inter::LennardJonesSoftCoreBeutler,
                       dr,
                       atom_i,
                       atom_j,
                       force_units=u"kJ * mol^-1 * nm^-1",
                       special=false,
                       args...)
    # Mix Lambda
    T = typeof(ustrip(atom_i.λ))
    λ_glob = T(λ_mixing(inter.λ_mixing, (atom_i.λ, atom_j.λ)))

    # 1. Fetch alchemical roles from the contiguous array
    role_i = atom_i.alch_role
    role_j = atom_j.alch_role
    pair_role = mix_roles(inter.scheduler, (role_i, role_j))

    # 2. Dispatch to the scheduler for the effective sterics lambda
    λ, λ_params = scale_sterics(inter.scheduler, λ_glob, pair_role)

    if λ <= 0
        return zero_pairwise_force(dr, force_units)
    end

    if shortcut_pair(inter.shortcut, atom_i, atom_j)
        return zero_pairwise_force(dr, force_units)
    end

    r = sqrt(sum(abs2, dr))
    if iszero_value(r)
        return zero_pairwise_force(dr, force_units)
    end

    # If lambda is 1, the soft core formula reduces to standard LJ
    # We explicity branch to save compute.
    if λ >= 1

        σ = σ_mixing(inter.σ_mixing, atom_i, atom_j)
        ϵ = ϵ_mixing(inter.ϵ_mixing, atom_i, atom_j)
        σ2 = σ^2
        params = (σ2, ϵ, nothing, nothing)
        
        # Call standard LJ cutoff logic.
        f = force_cutoff(inter.cutoff, inter, r, params) 
        fdr = radial_force_vector(f, r, dr, force_units)
        
        return special ? fdr * inter.weight_special : fdr
    end

    σ = λ_params*σ_mixing(inter.σ_mixing, atom_i, atom_j)
    ϵ = λ_params*ϵ_mixing(inter.ϵ_mixing, atom_i, atom_j)
    σ6 = σ^6

    C6 = 4 * ϵ * σ6
    C12 = C6 * σ6
    σ6_fac = inter.α * (1 - λ)
    params = (C12, C6, σ6_fac, λ)

    f = force_cutoff(inter.cutoff, inter, r, params)
    fdr = radial_force_vector(f, r, dr, force_units)
    
    return special ? fdr * inter.weight_special : fdr
end

# Dispatch 1: Standard LJ Logic
@inline function pairwise_force(::LennardJonesSoftCoreBeutler, r, (σ2, ϵ, _, _)::Tuple{Any, Any, Nothing, Nothing})
    six_term = (σ2 / r^2)^3
    return (24 * ϵ / r) * (2 * six_term^2 - six_term)
end

# Dispatch 2: Soft Core Logic
function pairwise_force(::LennardJonesSoftCoreBeutler, r, (C12, C6, σ6_fac, λ)::Tuple{Any, Any, Any, Any})
    R = sqrt(cbrt((σ6_fac*(C12/C6))+r^6))
    R6 = R^6
    return λ*(((12*C12)/(R6*R6*R)) - ((6*C6)/(R6*R)))*((r/R)^5)
end

@inline function potential_energy(inter::LennardJonesSoftCoreBeutler,
                                  dr,
                                  atom_i,
                                  atom_j,
                                  energy_units=u"kJ * mol^-1",
                                  special=false,
                                  args...)
    # Mix Lambda
    T = typeof(ustrip(atom_i.λ))
    λ_glob = T(λ_mixing(inter.λ_mixing, (atom_i.λ, atom_j.λ)))

    # 1. Fetch alchemical roles from the contiguous array
    role_i = atom_i.alch_role
    role_j = atom_j.alch_role
    pair_role = mix_roles(inter.scheduler, (role_i, role_j))

    # 2. Dispatch to the scheduler for the effective sterics lambda
    λ, λ_params = scale_sterics(inter.scheduler, λ_glob, pair_role)

    if λ <= 0
        return ustrip(zero(dr[1])) * energy_units
    end

    if shortcut_pair(inter.shortcut, atom_i, atom_j)
        return ustrip(zero(dr[1])) * energy_units
    end


    # If lambda is 1, the soft core formula reduces to standard LJ
    # We explicity branch to save compute.
    if λ >= 1

        σ = σ_mixing(inter.σ_mixing, atom_i, atom_j)
        ϵ = ϵ_mixing(inter.ϵ_mixing, atom_i, atom_j)

        r = sqrt(sum(abs2, dr))
        σ2 = σ^2
        params = (σ2, ϵ, nothing, nothing)

        pe = pe_cutoff(inter.cutoff, inter, r, params)
        
        if special
            return pe * inter.weight_special
        else
            return pe
        end
    end

    σ = λ_params*σ_mixing(inter.σ_mixing, atom_i, atom_j)
    ϵ = λ_params*ϵ_mixing(inter.ϵ_mixing, atom_i, atom_j)
    σ6 = σ^6

    cutoff = inter.cutoff
    r = sqrt(sum(abs2, dr))
    C6 = 4 * ϵ * σ6
    C12 = C6 * σ6
    σ6_fac = inter.α * (1 - λ)
    params = (C12, C6, σ6_fac, λ)

    pe = pe_cutoff(cutoff, inter, r, params)
    if special
        return pe * inter.weight_special
    else
        return pe
    end
end

# Dispatch 1: Standard LJ Logic
@inline function pairwise_pe(::LennardJonesSoftCoreBeutler, r, (σ2, ϵ, _, _)::Tuple{Any, Any, Nothing, Nothing})
    inv_r2 = inv(r^2)
    six_term = (σ2 * inv_r2)^3
    return 4 * ϵ * (six_term^2 - six_term)
end

# Dispatch 2: Soft Core Logic (Matches Tuple length 4)
function pairwise_pe(::LennardJonesSoftCoreBeutler, r, (C12, C6, σ6_fac, λ)::Tuple{Any, Any, Any, Any})
    R6 = (σ6_fac * (C12 / C6)) + r^6
    return λ * ((C12 / (R6 * R6)) - (C6 / R6))
end

@doc raw"""
    LennardJonesSoftCoreGapsys(; cutoff, α, λ, use_neighbors, shortcut, σ_mixing,
                               ϵ_mixing, weight_special)

The Lennard-Jones 6-12 interaction between two atoms with a soft core potential, used for
the appearing and disappearing of atoms.

See [Gapsys et al. 2012](https://doi.org/10.1021/ct300220p).
The potential energy is defined as
```math
V(r_{ij}) = \left\{ \begin{array}{cl}
\lambda \left( \frac{C^{(12)}}{r_{ij}^{12}} - \frac{C^{(6)}}{r_{ij}^{6}} \right), & \text{if} & r \ge r_{LJ} \\
\lambda \left( (\frac{78C^{(12)}}{r_{LJ}^{14}}-\frac{21C^{(6)}}{r_{LJ}^{8}})r_{ij}^2 - (\frac{168C^{(12)}}{r_{LJ}^{13}}-\frac{48C^{(6)}}{r_{LJ}^{7}})r_{ij} + \frac{91C^{(12)}}{r_{LJ}^{12}}-\frac{28C^{(6)}}{r_{LJ}^{6}} \right), & \text{if} & r \lt r_{LJ} \\
\end{array} \right.
```
and the force on each atom by
```math
\vec{F}_i = \left\{ \begin{array}{cl}
\lambda \left( \frac{12C^{(12)}}{r_{ij}^{13}} - \frac{6C^{(6)}}{r_{ij}^{7}} \right)\frac{\vec{r_{ij}}}{r_{ij}}, & \text{if} & r \ge r_{LJ} \\
\lambda \left( (\frac{-156C^{(12)}}{r_{LJ}^{14}}+\frac{42C^{(6)}}{r_{LJ}^{8}})r_{ij} - (\frac{168C^{(12)}}{r_{LJ}^{13}}-\frac{48C^{(6)}}{r_{LJ}^{7}}) \right)\frac{\vec{r_{ij}}}{r_{ij}}, & \text{if} & r \lt r_{LJ} \\
\end{array} \right.
```
where
```math
r_{LJ} = \alpha \left( \frac{26C^{(12)}(1-\lambda)}{7C^{(6)}} \right)^{\frac{1}{6}}
```
and
```math
C^{(12)} = 4\epsilon\sigma^{12}
C^{(6)} = 4\epsilon\sigma^{6}
```

If ``\lambda`` is 1.0, this gives the standard [`LennardJones`](@ref) potential and means
the atom is fully turned on.
If ``\lambda`` is zero the interaction is turned off.
``\alpha`` determines the strength of softening the function.
"""
@kwdef struct LennardJonesSoftCoreGapsys{C, A, H, S, E, LM, SCH, W} <: PairwiseInteraction
    cutoff::C = NoCutoff()
    α::A = 0.85
    use_neighbors::Bool = false
    shortcut::H = LJZeroShortcut()
    σ_mixing::S = LorentzMixing()
    ϵ_mixing::E = GeometricMixing()
    λ_mixing::LM = MinimumMixing()
    scheduler::SCH = DefaultLambdaScheduler()
    weight_special::W = 1
end

use_neighbors(inter::LennardJonesSoftCoreGapsys) = inter.use_neighbors

function Base.zero(lj::LennardJonesSoftCoreGapsys{C, A, H, S, E, LM, SCH, W}) where {C, A, H, S, E, LM, SCH, W}
    return LennardJonesSoftCoreGapsys(
        lj.cutoff,
        zero(A),
        lj.use_neighbors,
        lj.shortcut,
        lj.σ_mixing,
        lj.ϵ_mixing,
        lj.λ_mixing,
        lj.scheduler,
        zero(W),
    )
end

function Base.:+(l1::LennardJonesSoftCoreGapsys, l2::LennardJonesSoftCoreGapsys)
    return LennardJonesSoftCoreGapsys(
        l1.cutoff,
        l1.α + l2.α,
        l1.use_neighbors,
        l1.shortcut,
        l1.σ_mixing,
        l1.ϵ_mixing,
        l1.λ_mixing,
        l1.scheduler,
        l1.weight_special + l2.weight_special,
    )
end

function to_lambda_function(inter::LennardJones, gapsys::Val{true}; α=T(0.85), λ_mixing=MinimumMixing(), scheduler=DefaultLambdaScheduler())
    return LennardJonesSoftCoreGapsys(cutoff=inter.cutoff, α=α, use_neighbors=inter.use_neighbors, shortcut=inter.shortcut, 
                                        σ_mixing=inter.σ_mixing, ϵ_mixing=inter.ϵ_mixing, λ_mixing=λ.mixing, scheduler=scheduler, 
                                        weight_special=inter.weight_special)
end

@inline function force(inter::LennardJonesSoftCoreGapsys,
                       dr,
                       atom_i,
                       atom_j,
                       force_units=u"kJ * mol^-1 * nm^-1",
                       special=false,
                       args...)

    T = typeof(ustrip(atom_i.mass))
    λ_glob = T(λ_mixing(inter.λ_mixing, (atom_i.λ, atom_j.λ)))

    # 1. Fetch alchemical roles from the contiguous array
    role_i = atom_i.alch_role
    role_j = atom_j.alch_role
    pair_role = mix_roles(inter.scheduler, (role_i, role_j))

    # 2. Dispatch to the scheduler for the effective sterics lambda
    # Changed scale_elec to scale_sterics
    λ, λ_params = scale_sterics(inter.scheduler, λ_glob, pair_role)

    if λ <= 0
        return zero_pairwise_force(dr, force_units)
    end

    if shortcut_pair(inter.shortcut, atom_i, atom_j)
        return zero_pairwise_force(dr, force_units)
    end

    cutoff = inter.cutoff
    r = sqrt(sum(abs2, dr))
    if iszero_value(r)
        return zero_pairwise_force(dr, force_units)
    end

    σ = λ_params*σ_mixing(inter.σ_mixing, atom_i, atom_j)
    ϵ = λ_params*ϵ_mixing(inter.ϵ_mixing, atom_i, atom_j)
    σ2 = σ^2
    σ6 = σ2^3

    # 3. Fast Path: Standard Lennard Jones
    if λ >= 1
        # Pass standard LJ params tuple (Length 2)
        params = (σ^2, ϵ, nothing, nothing)
        f = force_cutoff(cutoff, inter, r, params) * dr
        fdr = f * inv(r)
        return special ? fdr * inter.weight_special : fdr
    end

    # 4. Alchemical Path: Soft Core Gapsys
    C6 = 4 * ϵ * σ6
    C12 = C6 * σ6
    val = (26 * σ6 * (1 - λ)) / 7
    R = inter.α * sqrt(cbrt(val))

    # Pass SoftCore params tuple (Length 4)
    params = (C12, C6, λ, R)
    f = force_cutoff(cutoff, inter, r, params)
    fdr = (f * inv(r)) * dr
    return special ? fdr * inter.weight_special : fdr
end

# Dispatch 1: Standard LJ Logic (Matches Tuple length 2)
@inline function pairwise_force(::LennardJonesSoftCoreGapsys, r, (σ2, ϵ, _, _)::Tuple{Any, Any, Nothing, Nothing})
    six_term = (σ2 / r^2)^3
    return (24 * ϵ / r) * (2 * six_term^2 - six_term)
end

# Dispatch 2: Soft Core Logic (Matches Tuple length 4)
@inline function pairwise_force(::LennardJonesSoftCoreGapsys, r, (C12, C6, λ, R)::Tuple{Any, Any, Any, Any})
    r2 = r^2
    r6 = r2^3
    if r >= R
        return λ * (((12*C12)/(r6*r6*r)) - ((6*C6)/(r6*r)))
    else
        invR = inv(R)
        invR2 = invR^2
        invR6 = invR2^3
        return λ * (((-156*C12*(invR6*invR6*invR2)) + (42*C6*(invR2*invR6)))*r +
                    (168*C12*(invR6*invR6*invR)) - (48*C6*(invR6*invR)))
    end
end

@inline function potential_energy(inter::LennardJonesSoftCoreGapsys,
                                  dr,
                                  atom_i,
                                  atom_j,
                                  energy_units=u"kJ * mol^-1",
                                  special=false,
                                  args...)
    T = typeof(ustrip(atom_i.mass))
    λ_glob = T(λ_mixing(inter.λ_mixing, (atom_i.λ, atom_j.λ)))

    # 1. Fetch alchemical roles from the contiguous array
    role_i = atom_i.alch_role
    role_j = atom_j.alch_role
    pair_role = mix_roles(inter.scheduler, (role_i, role_j))

    # 2. Dispatch to the scheduler for the effective sterics lambda
    # Changed scale_elec to scale_sterics
    λ, λ_params = scale_sterics(inter.scheduler, λ_glob, pair_role)

    if λ <= 0
        return ustrip(zero(dr[1])) * energy_units
    end

    if shortcut_pair(inter.shortcut, atom_i, atom_j)
        return ustrip(zero(dr[1])) * energy_units
    end

    cutoff = inter.cutoff
    r = sqrt(sum(abs2, dr))
    σ = λ_params*σ_mixing(inter.σ_mixing, atom_i, atom_j)
    ϵ = λ_params*ϵ_mixing(inter.ϵ_mixing, atom_i, atom_j)
    σ6 = σ^6

    # 3. Fast Path: Standard Lennard Jones
    if λ >= 1
        # Pass standard LJ params tuple (Length 2)
        params = (σ^2, ϵ, nothing, nothing)
        pe = pe_cutoff(cutoff, inter, r, params)
        return special ? pe * inter.weight_special : pe
    end

    # 4. Alchemical Path: Soft Core Gapsys
    C6 = 4 * ϵ * σ6
    C12 = C6 * σ6
    val = (26 * σ6 * (1 - λ)) / 7
    R = inter.α * sqrt(cbrt(val))

    # Pass SoftCore params tuple (Length 4)
    params = (C12, C6, λ, R)
    pe = pe_cutoff(cutoff, inter, r, params)
    return special ? pe * inter.weight_special : pe
end

# Dispatch 1: Standard LJ Logic (Matches Tuple length 2)
@inline function pairwise_pe(::LennardJonesSoftCoreGapsys, r, (σ2, ϵ, _, _)::Tuple{Any, Any, Nothing, Nothing})
    inv_r2 = inv(r^2)
    six_term = (σ2 * inv_r2)^3
    return 4 * ϵ * (six_term^2 - six_term)
end

# Dispatch 2: Soft Core Logic (Matches Tuple length 4)
@inline function pairwise_pe(::LennardJonesSoftCoreGapsys, r, (C12, C6, λ, R)::Tuple{Any, Any, Any, Any})
    r6 = r^6
    if r >= R
        return λ * ((C12/(r6*r6)) - (C6/(r6)))
    else
        invR = inv(R)
        invR2 = invR^2
        invR6 = invR^6
        return λ * (((78*C12*(invR6*invR6*invR2)) - (21*C6*(invR2*invR6)))*(r^2) -
                   ((168*C12*(invR6*invR6*invR)) - (48*C6*(invR6*invR)))*r +
                   (91*C12*(invR6*invR6)) - (28*C6*(invR6)))
    end
end

@inline function force_λ(inter::LennardJonesSoftCoreGapsys,
                       dr,
                       atom_i,
                       atom_j,
                       force_units=u"kJ * mol^-1 * nm^-1",
                       special=false,
                       args...)

    T = typeof(ustrip(atom_i.λ))
    if !isa(inter.λ_mixing, typeof(Molly.ProductMixing()))
        error("force with respect to λ only possible with ProductMixing(), currently using: ", typeof(inter.λ_mixing))
    end
    λ_glob = T(λ_mixing(inter.λ_mixing, (atom_i.λ, atom_j.λ)))
    d_λ = zero_pairwise_force(dr, force_units)

    # 1. Fetch alchemical roles from the contiguous array
    role_i = atom_i.alch_role
    role_j = atom_j.alch_role
    pair_role = mix_roles(inter.scheduler, (role_i, role_j))
    di = atom_i.alch_role==ProbRole
    dj = atom_j.alch_role==ProbRole

    # 2. Dispatch to the scheduler for the effective sterics lambda
    # Changed scale_elec to scale_sterics
    λ, λ_params = scale_sterics(inter.scheduler, λ_glob, pair_role)

    if λ <= 0
        return d_λ
    end

    if shortcut_pair(inter.shortcut, atom_i, atom_j)
        return d_λ
    end

    cutoff = inter.cutoff
    r = norm(dr)
    σ = λ_params*σ_mixing(inter.σ_mixing, atom_i, atom_j)
    ϵ = λ_params*ϵ_mixing(inter.ϵ_mixing, atom_i, atom_j)
    σ6 = σ^6

    C6 = 4 * ϵ * σ6
    C12 = C6 * σ6
    val = (26 * σ6 * (1 - λ)) / 7
    R = inter.α * sqrt(cbrt(val))
    
    if r >= R
        r6 = r^6
        pe_i = atom_j.λ * ((C12/(r6*r6)) - (C6/(r6)))
        pe_j = atom_i.λ * ((C12/(r6*r6)) - (C6/(r6)))
    else
        α_term = inter.α * sqrt(cbrt((26*(C12/C6)/7)))
        α_term2 = α_term^2
        α_term6 = α_term2^3
        r2 = r^2
        A1 = (78*C12*r2) / (α_term6*α_term6*α_term2)
        B1 = (-21*C6*r2) / (α_term6*α_term2)
        C1 = (-168*C12*r) / (α_term6*α_term6*α_term)
        D1 = (48*C6*r) / (α_term6*α_term)
        E1 = (91*C12) / (α_term6*α_term6)
        F1 = (-28*C6) / α_term6
        λ1 = (1-λ)
        λ13 = cbrt(λ1)
        λ16 = sqrt(λ13)
        pe_i = A1*((atom_j.λ*((4*λ)+3))/(3*λ1^3*λ13)) + B1*((atom_j.λ*(λ+3))/(3*λ1^2*λ13)) +
                C1*((atom_j.λ*((7*λ)+6))/(6*λ1^3*λ16)) + D1*((atom_j.λ*(λ+6))/(6*λ1^2*λ16)) +
                E1*((atom_j.λ*(λ+1))/(λ1^3)) + F1*(atom_j.λ/(λ1^2))
        pe_j = A1*((atom_i.λ*((4*λ)+3))/(3*λ1^3*λ13)) + B1*((atom_i.λ*(λ+3))/(3*λ1^2*λ13)) +
                C1*((atom_i.λ*((7*λ)+6))/(6*λ1^3*λ16)) + D1*((atom_i.λ*(λ+6))/(6*λ1^2*λ16)) +
                E1*((atom_i.λ*(λ+1))/(λ1^3)) + F1*(atom_i.λ/(λ1^2))
    end

    if special
        pe_i = pe_i * inter.weight_special * (r <= inter.cutoff.dist_cutoff) * di
        pe_j = pe_j * inter.weight_special * (r <= inter.cutoff.dist_cutoff) * dj
        tmp = SVector{3,T}(ustrip(pe_i),ustrip(pe_j),0.0)
        return d_λ .+ (tmp*force_units)
    else
        pe_i = pe_i * (r <= inter.cutoff.dist_cutoff) * di
        pe_j = pe_j * (r <= inter.cutoff.dist_cutoff) * dj
        tmp = SVector{3,T}(ustrip(pe_i),ustrip(pe_j),0.0)
        return d_λ .+ (tmp*force_units)
    end
end

@doc raw"""
    AshbaughHatch(; cutoff, use_neighbors, shortcut, ϵ_mixing, σ_mixing,
                  λ_mixing, weight_special)

The Ashbaugh-Hatch potential ($V_{\text{AH}}$) is a modified Lennard-Jones ($V_{\text{LJ}}$)
6-12 interaction between two atoms.

The potential energy is defined as
```math
V_{\text{LJ}}(r_{ij}) = 4\varepsilon_{ij} \left[\left(\frac{\sigma_{ij}}{r_{ij}}\right)^{12} - \left(\frac{\sigma_{ij}}{r_{ij}}\right)^{6}\right] \\
```
```math
V_{\text{AH}}(r_{ij}) =
    \begin{cases}
      V_{\text{LJ}}(r_{ij}) +\varepsilon_{ij}(1-λ_{ij}) &,  r_{ij}\leq  2^{1/6}σ  \\
       λ_{ij}V_{\text{LJ}}(r_{ij})  &,  2^{1/6}σ \leq r_{ij}
    \end{cases}
```
and the force on each atom by
```math
\vec{F}_{\text{AH}} =
    \begin{cases}
      F_{\text{LJ}}(r_{ij})  &,  r_{ij} \leq  2^{1/6}σ  \\
       λ_{ij}F_{\text{LJ}}(r_{ij})  &,  2^{1/6}σ \leq r_{ij}
    \end{cases}
```
where
```math
\begin{aligned}
\vec{F}_{\text{LJ}}\
&= \frac{24\varepsilon_{ij}}{r_{ij}^2} \left[2\left(\frac{\sigma_{ij}}{r_{ij}}\right)^{12} -\left(\frac{\sigma_{ij}}{r_{ij}}\right)^{6}\right]  \vec{r_{ij}}
\end{aligned}
```

If ``\lambda`` is one this gives the standard [`LennardJones`](@ref) potential.
"""
@kwdef struct AshbaughHatch{C, H, S, E, L, W} <: PairwiseInteraction
    cutoff::C = NoCutoff()
    use_neighbors::Bool = false
    shortcut::H = LJZeroShortcut()
    σ_mixing::S = LorentzMixing()
    ϵ_mixing::E = LorentzMixing()
    λ_mixing::L = LorentzMixing()
    weight_special::W = 1
end

use_neighbors(inter::AshbaughHatch) = inter.use_neighbors

function Base.zero(lj::AshbaughHatch{C, H, S, E, L, W}) where {C, H, S, E, L, W}
    return AshbaughHatch(
        lj.cutoff,
        lj.use_neighbors,
        lj.shortcut,
        lj.σ_mixing,
        lj.ϵ_mixing,
        lj.λ_mixing,
        zero(W),
    )
end

function Base.:+(l1::AshbaughHatch, l2::AshbaughHatch)
    return AshbaughHatch(
        l1.cutoff,
        l1.use_neighbors,
        l1.shortcut,
        l1.σ_mixing,
        l1.ϵ_mixing,
        l1.λ_mixing,
        l1.weight_special + l2.weight_special,
    )
end

@kwdef struct AshbaughHatchAtom{T, M, C, S, E, L}
    index::Int = 1
    atom_type::T = 1
    mass::M = 1.0u"g/mol"
    charge::C = 0.0
    σ::S = 0.0u"nm"
    ϵ::E = 0.0u"kJ * mol^-1"
    λ::L = 1.0
end

@inline function force(inter::AshbaughHatch,
                       dr,
                       atom_i,
                       atom_j,
                       force_units=u"kJ * mol^-1 * nm^-1",
                       special::Bool=false,
                       args...)
    if shortcut_pair(inter.shortcut, atom_i, atom_j, special)
        return zero_pairwise_force(dr, force_units)
    end

    λ_glob = T(λ_mixing(inter.λ_mixing, (atom_i.λ, atom_j.λ)))
    role_i = atom_i.alch_role
    role_j = atom_j.alch_role
    pair_role = mix_roles(inter.scheduler, (role_i, role_j))
    λ, λ_params = scale_sterics(inter.scheduler, λ_glob, pair_role)

    σ = λ_params*σ_mixing(inter.σ_mixing, atom_i, atom_j, special)
    ϵ = λ_params*ϵ_mixing(inter.ϵ_mixing, atom_i, atom_j, special)

    cutoff = inter.cutoff
    r = sqrt(sum(abs2, dr))
    σ2 = σ^2
    params = (σ2, ϵ, λ)

    f = force_cutoff(cutoff, inter, r, params)
    fdr = (f / r) * dr
    if special
        return fdr * inter.weight_special
    else
        return fdr
    end
end

@inline function pairwise_force(::AshbaughHatch, r, (σ2, ϵ, λ))
    r2 = r^2
    six_term = (σ2 / r2) ^ 3
    lj_term = (24ϵ / r) * (2 * six_term ^ 2 - six_term)
    if r2 < (2^(1/3) * σ2)
        return lj_term
    else
        return λ * lj_term
    end
end

@inline function potential_energy(inter::AshbaughHatch,
                                  dr,
                                  atom_i,
                                  atom_j,
                                  energy_units=u"kJ * mol^-1",
                                  special::Bool=false,
                                  args...)
    if shortcut_pair(inter.shortcut, atom_i, atom_j, special)
        return ustrip(zero(dr[1])) * energy_units
    end

    λ_glob = T(λ_mixing(inter.λ_mixing, (atom_i.λ, atom_j.λ)))
    role_i = atom_i.alch_role
    role_j = atom_j.alch_role
    pair_role = mix_roles(inter.scheduler, (role_i, role_j))
    λ, λ_params = scale_sterics(inter.scheduler, λ_glob, pair_role)

    σ = λ_params*σ_mixing(inter.σ_mixing, atom_i, atom_j, special)
    ϵ = λ_params*ϵ_mixing(inter.ϵ_mixing, atom_i, atom_j, special)

    cutoff = inter.cutoff
    r = sqrt(sum(abs2, dr))
    σ2 = σ^2
    params = (σ2, ϵ, λ)

    pe = pe_cutoff(cutoff, inter, r, params)
    if special
        return pe * inter.weight_special
    else
        return pe
    end
end

@inline function pairwise_pe(::AshbaughHatch, r, (σ2, ϵ, λ))
    r2 = r^2
    six_term = (σ2 / r2) ^ 3
    lj_term = 4ϵ * (six_term ^ 2 - six_term)
    if r2 < (2^(1/3) * σ2)
        return lj_term + ϵ * (1 - λ)
    else
        return λ * lj_term
    end
end

# Specific interaction used to allow different σ/ϵ for 1-4 interactions
# Assumes no 1-4 Lennard-Jones interaction via the pairwise interactions (weight_special = 0)
struct LennardJones14{S, E, W}
    σ14_mixed::S
    ϵ14_mixed::E
    weight_14::W
end

function Base.zero(::LennardJones14{S, E, W}) where {S, E, W}
    return LennardJones14(zero(S), zero(E), zero(W))
end

function Base.:+(l1::LennardJones14, l2::LennardJones14)
    return LennardJones14(
        l1.σ14_mixed + l2.σ14_mixed,
        l1.ϵ14_mixed + l2.ϵ14_mixed,
        l1.weight_14 + l2.weight_14,
    )
end

@inline function force(inter::LennardJones14, coords_i, coords_l, boundary, args...)
    σ2 = inter.σ14_mixed ^ 2
    dr = vector(coords_i, coords_l, boundary)
    r2 = sum(abs2, dr)
    six_term = (σ2 / r2) ^ 3
    fl = inter.weight_14 * (24 * inter.ϵ14_mixed / r2) * (2 * six_term ^ 2 - six_term) * dr
    fi = -fl
    return SpecificForce2Atoms(fi, fl)
end

@inline function potential_energy(inter::LennardJones14, coords_i, coords_l, boundary, args...)
    σ2 = inter.σ14_mixed ^ 2
    r2 = sum(abs2, vector(coords_i, coords_l, boundary))
    six_term = (σ2 / r2) ^ 3
    return inter.weight_14 * 4 * inter.ϵ14_mixed * (six_term ^ 2 - six_term)
end

@inline function force_λ(b::LennardJones14, coord_i, coord_j, boundary, atoms_i, atoms_j, F, args...)
    dr = vector(coord_i, coord_j, boundary)
    return SpecificForce2Atoms(zero_pairwise_force(dr, F), zero_pairwise_force(dr, F))
end

# Specific interaction used to allow different σ/ϵ for 1-4 interactions
# Assumes no 1-4 Lennard-Jones interaction via the pairwise interactions (weight_special = 0)
@kwdef struct LennardJones14SoftCoreGapsys{S, E, W, A, LM, SCH}
    σ14_mixed::S
    ϵ14_mixed::E
    weight_14::W
    α::A = 0.85
    λ_mixing::LM = MinimumMixing()
    scheduler::SCH = DefaultLambdaScheduler()
end

function Base.zero(lj::LennardJones14SoftCoreGapsys{S, E, W, A, LM, SCH}) where {S, E, W, A, LM, SCH}
    return LennardJones14SoftCoreGapsys(
        zero(S), 
        zero(E), 
        zero(W),
        zero(A),
        lj.λ_mixing,
        lj.scheduler,
        )
end

function Base.:+(l1::LennardJones14SoftCoreGapsys, l2::LennardJones14SoftCoreGapsys)
    return LennardJones14SoftCoreGapsys(
        l1.σ14_mixed + l2.σ14_mixed,
        l1.ϵ14_mixed + l2.ϵ14_mixed,
        l1.weight_14 + l2.weight_14,
        l1.α + l2.α,
        l1.λ_mixing,
        l1.scheduler,
    )
end

function to_lambda_function(inter::LennardJones14, gapsys::Val{true}; α=T(0.85), λ_mixing=MinimumMixing(), scheduler=DefaultLambdaScheduler())
    return LennardJones14SoftCoreGapsys(σ14_mixed=inter.σ14_mixed, ϵ14_mixed=inter.ϵ14_mixed, weight_14=inter.weight_14, α=α, λ_mixing=λ.mixing, scheduler=scheduler)
end

@inline function force(inter::LennardJones14SoftCoreGapsys, coords_i, coords_l, boundary, atoms_i, atoms_l, force_units, args...)
    T = typeof(ustrip(atoms_i.σ))
    dr = vector(coords_i, coords_l, boundary)
    λ_glob = T(λ_mixing(inter.λ_mixing, (atoms_i.λ, atoms_l.λ)))

    # 1. Fetch alchemical roles from the contiguous array
    role_i = atoms_i.alch_role
    role_l = atoms_l.alch_role
    pair_role = mix_roles(inter.scheduler, (role_i, role_l))

    # 2. Dispatch to the scheduler for the effective sterics lambda
    # Changed scale_elec to scale_sterics
    λ, λ_params = scale_sterics(inter.scheduler, λ_glob, pair_role)

    if λ <= 0
        return SpecificForce2Atoms(zero(dr)*force_units, zero(dr)*force_units)
    end

    r = norm(dr)
    if iszero_value(r)
        return SpecificForce2Atoms(zero_pairwise_force(dr, force_units), zero_pairwise_force(dr, force_units))
    end

    if λ >= 1
        σ2 = (λ_params * inter.σ14_mixed) ^ 2
        dr = vector(coords_i, coords_l, boundary)
        r2 = sum(abs2, dr)
        six_term = (σ2 / r2) ^ 3
        fl = inter.weight_14 * (24 * λ_params * inter.ϵ14_mixed / r2) * (2 * six_term ^ 2 - six_term) * dr
        fi = -fl
        return SpecificForce2Atoms(fi, fl)
    else
        σ6 = (λ_params*inter.σ14_mixed)^6
        r6 = r^6
        C6 = 4 * λ_params * inter.ϵ14_mixed * σ6
        C12 = C6 * σ6
        val = (26 * σ6 * (1 - λ)) / 7
        R = inter.α * sqrt(cbrt(val))

        if r >= R
            σ2 = (λ_params*inter.σ14_mixed) ^ 2
            dr = vector(coords_i, coords_l, boundary)
            r2 = sum(abs2, dr)
            six_term = (σ2 / r2) ^ 3
            fl = λ * inter.weight_14 * (24 * λ_params * inter.ϵ14_mixed / r2) * (2 * six_term ^ 2 - six_term) * dr
            fi = -fl
            return SpecificForce2Atoms(fi, fl)
        else
            invR = inv(R)
            invR2 = invR^2
            invR6 = invR^6
            fl = λ * inter.weight_14 * (((-156*C12*(invR6*invR6*invR2)) + (42*C6*(invR2*invR6)))*r +
                        (168*C12*(invR6*invR6*invR)) - (48*C6*(invR6*invR))) / r * dr
            fi = -fl
            return SpecificForce2Atoms(fi, fl)
        end
    end
end

@inline function potential_energy(inter::LennardJones14SoftCoreGapsys, coords_i, coords_l, boundary, atoms_i, atoms_l, energy_units, args...)
    T = typeof(ustrip(atoms_i.σ))
    dr = vector(coords_i, coords_l, boundary)
    λ_glob = T(λ_mixing(inter.λ_mixing, (atoms_i.λ, atoms_l.λ)))

    # 1. Fetch alchemical roles from the contiguous array
    role_i = atoms_i.alch_role
    role_l = atoms_l.alch_role
    pair_role = mix_roles(inter.scheduler, (role_i, role_l))

    # 2. Dispatch to the scheduler for the effective sterics lambda
    # Changed scale_elec to scale_sterics
    λ, λ_params = scale_sterics(inter.scheduler, λ_glob, pair_role)

    if λ <= 0
        return ustrip(zero(dr[1])) * energy_units
    end

    r = norm(dr)
    if iszero_value(r)
        return ustrip(zero(dr[1])) * energy_units
    end

    if λ >= 1
        σ2 = (λ_params*inter.σ14_mixed) ^ 2
        r2 = r^2
        six_term = (σ2 / r2) ^ 3
        return inter.weight_14 * 4 * λ_params * inter.ϵ14_mixed * (six_term ^ 2 - six_term)
    else
        σ6 = (λ_params*inter.σ14_mixed)^6
        C6 = 4 * λ_params * inter.ϵ14_mixed * σ6
        C12 = C6 * σ6
        val = (26 * σ6 * (1 - λ)) / 7
        R = inter.α * sqrt(cbrt(val))

        r6 = r^6
        if r >= R
            return λ * inter.weight_14 * ((C12/(r6*r6)) - (C6/(r6)))
        else
            invR = inv(R)
            invR2 = invR^2
            invR6 = invR^6
            return λ * inter.weight_14 * (((78*C12*(invR6*invR6*invR2)) - (21*C6*(invR2*invR6)))*(r^2) -
                    ((168*C12*(invR6*invR6*invR)) - (48*C6*(invR6*invR)))*r +
                    (91*C12*(invR6*invR6)) - (28*C6*(invR6)))
        end
    end
end

@inline function force_λ(inter::LennardJones14SoftCoreGapsys, coords_i, coords_l, boundary, atoms_i, atoms_l, force_units, args...)
    T = typeof(ustrip(atoms_i.σ))
    if !isa(inter.λ_mixing, typeof(ProductMixing()))
        error("force with respect to λ only possible with ProductMixing(), currently using: ", typeof(inter.λ_mixing))
    end
    dr = vector(coords_i, coords_l, boundary)
    λ_glob = T(λ_mixing(inter.λ_mixing, (atoms_i.λ, atoms_l.λ)))

    # 1. Fetch alchemical roles from the contiguous array
    role_i = atoms_i.alch_role
    role_j = atoms_l.alch_role
    pair_role = mix_roles(inter.scheduler, (role_i, role_j))
    di = atoms_i.alch_role==ProbRole
    dj = atoms_l.alch_role==ProbRole

    # 2. Dispatch to the scheduler for the effective sterics lambda
    # Changed scale_elec to scale_sterics
    λ, λ_params = scale_sterics(inter.scheduler, λ_glob, pair_role)

    # Different cutoffs
    if λ <= 0
        return SpecificForce2Atoms(zero_pairwise_force(dr, force_units),zero_pairwise_force(dr, force_units))
    end

    r = norm(dr)
    if iszero_value(r)
        return SpecificForce2Atoms(zero_pairwise_force(dr, force_units),zero_pairwise_force(dr, force_units))
    end

    σ6 = (λ_params*inter.σ14_mixed)^6

    C6 = 4 * λ_params* inter.ϵ14_mixed * σ6
    C12 = C6 * σ6
    val = (26 * σ6 * (1 - λ)) / 7
    R = inter.α * sqrt(cbrt(val))
    
    if r >= R
        r6 = r^6
        pe_i = inter.weight_14 * ((C12/(r6*r6))-(C6/(r6))) * atoms_l.λ
        pe_l = inter.weight_14 * ((C12/(r6*r6))-(C6/(r6))) * atoms_i.λ
    else
        α_term = inter.α * sqrt(cbrt((26*(C12/C6)/7)))
        α_term2 = α_term^2
        α_term6 = α_term2^3
        r2 = r^2
        A1 = (78*C12*r2) / (α_term6*α_term6*α_term2)
        B1 = (-21*C6*r2) / (α_term6*α_term2)
        C1 = (-168*C12*r) / (α_term6*α_term6*α_term)
        D1 = (48*C6*r) / (α_term6*α_term)
        E1 = (91*C12) / (α_term6*α_term6)
        F1 = (-28*C6) / α_term6
        λ1 = (1-λ)
        λ13 = cbrt(λ1)
        λ16 = sqrt(λ13)
        pe_i = inter.weight_14 * (A1*((atoms_l.λ*((4*λ)+3))/(3*λ1^3*λ13)) + B1*((atoms_l.λ*(λ+3))/(3*λ1^2*λ13)) +
                C1*((atoms_l.λ*((7*λ)+6))/(6*λ1^3*λ16)) + D1*((atoms_l.λ*(λ+6))/(6*λ1^2*λ16)) +
                E1*((atoms_l.λ*(λ+1))/(λ1^3)) + F1*(atoms_l.λ/(λ1^2)))
        pe_l = inter.weight_14 * (A1*((atoms_i.λ*((4*λ)+3))/(3*λ1^3*λ13)) + B1*((atoms_i.λ*(λ+3))/(3*λ1^2*λ13)) +
                C1*((atoms_i.λ*((7*λ)+6))/(6*λ1^3*λ16)) + D1*((atoms_i.λ*(λ+6))/(6*λ1^2*λ16)) +
                E1*((atoms_i.λ*(λ+1))/(λ1^3)) + F1*(atoms_i.λ/(λ1^2)))
    end
    fi = SVector{3,T}(ustrip(pe_i),0.0,0.0)*force_units
    fl = SVector{3,T}(ustrip(pe_l),0.0,0.0)*force_units
    return SpecificForce2Atoms(fi, fl)
end