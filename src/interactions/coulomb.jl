export
    Coulomb,
    CoulombSoftCoreBeutler,
    CoulombSoftCoreGapsys,
    CoulombReactionField,
    CoulombSoftCoreBeutlerReactionField,
    CoulombSoftCoreGapsysReactionField,
    CoulombEwald,
    CoulombSoftCoreBeutlerEwald,
    CoulombSoftCoreGapsysEwald,
    Yukawa

const coulomb_const = 138.93545764u"kJ * mol^-1 * nm" # 1 / 4πϵ0

@doc raw"""
    Coulomb(; cutoff, use_neighbors, weight_special, coulomb_const)

The Coulomb electrostatic interaction between two atoms.

The potential energy is defined as
```math
V(r_{ij}) = \frac{q_i q_j}{4 \pi \varepsilon_0 r_{ij}}
```
"""
@kwdef struct Coulomb{C, W, T} <: PairwiseInteraction
    cutoff::C = NoCutoff()
    use_neighbors::Bool = false
    weight_special::W = 1
    coulomb_const::T = coulomb_const
end

use_neighbors(inter::Coulomb) = inter.use_neighbors

function Base.zero(coul::Coulomb{C, W, T}) where {C, W, T}
    return Coulomb(coul.cutoff, coul.use_neighbors, zero(W), zero(T))
end

function Base.:+(c1::Coulomb, c2::Coulomb)
    return Coulomb(
        c1.cutoff,
        c1.use_neighbors,
        c1.weight_special + c2.weight_special,
        c1.coulomb_const + c2.coulomb_const,
    )
end

function inject_interaction(inter::Coulomb, params_dic)
    key_prefix = "inter_CO_"
    return Coulomb(
        inter.cutoff,
        inter.use_neighbors,
        dict_get(params_dic, key_prefix * "weight_14", inter.weight_special),
        dict_get(params_dic, key_prefix * "coulomb_const", inter.coulomb_const),
    )
end

function extract_parameters!(params_dic, inter::Coulomb, ff)
    key_prefix = "inter_CO_"
    params_dic[key_prefix * "weight_14"] = inter.weight_special
    params_dic[key_prefix * "coulomb_const"] = inter.coulomb_const
    return params_dic
end

@inline function force(inter::Coulomb{C},
                       dr,
                       atom_i,
                       atom_j,
                       force_units=u"kJ * mol^-1 * nm^-1",
                       special=false,
                       args...) where C
    r = norm(dr)
    cutoff = inter.cutoff
    ke = inter.coulomb_const
    qi, qj = atom_i.charge, atom_j.charge
    params = (ke, qi, qj)

    f = force_cutoff(cutoff, inter, r, params)
    fdr = (f / r) * dr
    if special
        return fdr * inter.weight_special
    else
        return fdr
    end
end

function pairwise_force(::Coulomb, r, (ke, qi, qj))
    return (ke * qi * qj) / r^2
end

@inline function potential_energy(inter::Coulomb{C},
                                  dr,
                                  atom_i,
                                  atom_j,
                                  energy_units=u"kJ * mol^-1",
                                  special=false,
                                  args...) where C
    r = norm(dr)
    cutoff = inter.cutoff
    ke = inter.coulomb_const
    qi, qj = atom_i.charge, atom_j.charge
    params = (ke, qi, qj)

    pe = pe_cutoff(cutoff, inter, r, params)
    if special
        return pe * inter.weight_special
    else
        return pe
    end
end

function pairwise_pe(::Coulomb, r, (ke, qi, qj))
    return (ke * qi * qj) * inv(r)
end

@doc raw"""
    CoulombSoftCoreBeutler(; cutoff, α, λ, use_neighbors, σ_mixing, ϵ_mixing,
                           weight_special, coulomb_const)

The Coulomb electrostatic interaction between two atoms with a soft core, used for
the appearing and disappearing of atoms.

See [Beutler et al. 1994](https://doi.org/10.1016/0009-2614(94)00397-1).
The potential energy is defined as
```math
V(r_{ij}) = \lambda \frac{1}{4\pi\epsilon_0} \frac{q_iq_j}{r_Q^{1/6}}
```
and the force on each atom by
```math
\vec{F}_i = \lambda \frac{1}{4\pi\epsilon_0} \frac{q_iq_jr_{ij}^5}{r_Q^{7/6}}\frac{\vec{r_{ij}}}{r_{ij}}
```
where
```math
r_{Q} = \left(\frac{\alpha(1-\lambda)C^{(12)}}{C^{(6)}}\right)+r_{ij}^6
```
and
```math
C^{(12)} = 4\epsilon\sigma^{12}
C^{(6)} = 4\epsilon\sigma^{6}
```
If ``\lambda`` is 1.0, this gives the standard [`Coulomb`](@ref) potential and means
the atom is fully turned on.
If ``\lambda`` is zero the interaction is turned off.
``\alpha`` determines the strength of softening the function.
"""
struct CoulombSoftCoreBeutler{C, A, S, E, LM, SCH, W, T} <: PairwiseInteraction
    cutoff::C
    α::A
    use_neighbors::Bool 
    σ_mixing::S 
    ϵ_mixing::E 
    λ_mixing::LM 
    scheduler::SCH 
    weight_special::W 
    coulomb_const::T
end

function CoulombSoftCoreBeutler(; cutoff=NoCutoff(), α=1.0, use_neighbors=false,
                                    σ_mixing=LorentzMixing(), ϵ_mixing=GeometricMixing(), 
                                    λ_mixing=MinimumMixing(),
                                    scheduler=DefaultLambdaScheduler(), weight_special=1,
                                    coulomb_const=coulomb_const)
    T = typeof(ustrip(α))
    return CoulombSoftCoreBeutler(
        cutoff,
        α,
        use_neighbors,
        σ_mixing,
        ϵ_mixing,
        λ_mixing,
        scheduler,
        weight_special,
        coulomb_const,
    )
end

use_neighbors(inter::CoulombSoftCoreBeutler) = inter.use_neighbors

function Base.zero(coul::CoulombSoftCoreBeutler{C, A, S, E, LM, SCH, W, T}) where {C, A, S, E, LM, SCH, W, T}
    return CoulombSoftCoreBeutler(
        coul.cutoff,
        zero(A),
        coul.use_neighbors,
        coul.σ_mixing,
        coul.ϵ_mixing,
        coul.λ_mixing,
        coul.scheduler,
        zero(W),
        zero(T),
    )
end

function Base.:+(c1::CoulombSoftCoreBeutler, c2::CoulombSoftCoreBeutler)
    return CoulombSoftCoreBeutler(
        c1.cutoff,
        c1.α + c2.α,
        c1.use_neighbors,
        c1.σ_mixing,
        c1.ϵ_mixing,
        c1.λ_mixing,
        c1.scheduler,
        c1.weight_special + c2.weight_special,
        c1.coulomb_const + c2.coulomb_const,
    )
end

function to_lambda_function(inter::Coulomb, gapsys::Val{false}; α=1.0, λ_mixing=MinimumMixing(), scheduler=DefaultLambdaScheduler())
    return CoulombSoftCoreBeutler(cutoff=inter.cutoff, α=α, use_neighbors=inter.use_neighbors, σ_mixing=inter.σ_mixing, ϵ_mixing=inter.ϵ_mixing,
                                        λ_mixing=λ_mixing, scheduler=scheduler, weight_special=inter.weight_special, coulomb_const=inter.coulomb_const)
end

@inline function force(inter::CoulombSoftCoreBeutler,
                       dr,
                       atom_i,
                       atom_j,
                       force_units=u"kJ * mol^-1 * nm^-1",
                       special=false,
                       args...)
    # 1. Type stability
    ke = inter.coulomb_const
    T = typeof(ustrip(ke))
    λ_glob = T(λ_mixing(inter.λ_mixing, (atom_i.λ, atom_j.λ)))

    # 1. Fetch alchemical roles from the contiguous array
    role_i = atom_i.alch_role
    role_j = atom_j.alch_role
    pair_role = mix_roles(inter.scheduler, (role_i, role_j))

    # 2. Dispatch to the scheduler for the effective sterics lambda
    # Changed scale_elec to scale_sterics
    λ, λ_params = scale_elec(inter.scheduler, λ_glob, pair_role)

    if λ <= 0
        return zero_pairwise_force(dr, force_units)
    end

    r = norm(dr)
    if iszero_value(r)
        return zero_pairwise_force(dr, force_units)
    end

    qij = (λ_params*atom_i.charge) * (λ_params*atom_j.charge)
    cutoff = inter.cutoff

    # 2. Fast Path: Standard Coulomb (λ >= 1.0)
    # Use tuple padding (Nothing) to match length 5 of the alchemical path
    if λ >= 1
        params = (ke, qij, nothing, nothing)
        f = force_cutoff(cutoff, inter, r, params)
        fdr = radial_force_vector(f, r, dr, force_units)
        return special ? fdr * inter.weight_special : fdr
    end

    # 3. Alchemical Path
    σ6 = σ_mixing(inter.σ_mixing, atom_i, atom_j)^6
    σ6_fac = inter.α * (1 - λ) * σ6

    params = (ke, qij, σ6_fac, λ)
    f = force_cutoff(cutoff, inter, r, params)
    fdr = radial_force_vector(f, r, dr, force_units)
    return special ? fdr * inter.weight_special : fdr
end

# Dispatch 1: Standard Coulomb Logic (Matches Tuple length 5 with Nothings)
@inline function pairwise_force(::CoulombSoftCoreBeutler, r, (ke, qij, _, _)::Tuple{Any, Any, Any, Nothing, Nothing})
    return (ke * qij) / r^2
end

# Dispatch 2: Soft Core Logic (Matches Tuple length 5 with Real/Quantities)
@inline function pairwise_force(::CoulombSoftCoreBeutler, r, (ke, qij, σ6_fac, λ)::Tuple{Any, Any, Any, Any, Any})
    r3 = r^3
    term = σ6_fac + (r3 * r3)
    R = term * sqrt(cbrt(term))
    return λ * ke * ((qij) / R) * (r3 * r * r)
end

@inline function potential_energy(inter::CoulombSoftCoreBeutler,
                                  dr,
                                  atom_i,
                                  atom_j,
                                  energy_units=u"kJ * mol^-1",
                                  special=false,
                                  args...)
    # 1. Type stability
    ke = inter.coulomb_const
    T = typeof(ustrip(ke))
    λ_glob = T(λ_mixing(inter.λ_mixing, (atom_i.λ, atom_j.λ)))

    # 1. Fetch alchemical roles from the contiguous array
    role_i = atom_i.alch_role
    role_j = atom_j.alch_role
    pair_role = mix_roles(inter.scheduler, (role_i, role_j))

    # 2. Dispatch to the scheduler for the effective sterics lambda
    # Changed scale_elec to scale_sterics
    λ, λ_params = scale_elec(inter.scheduler, λ_glob, pair_role)

    if λ <= 0
        return ustrip(zero(dr[1])) * energy_units
    end

    r = norm(dr)
    qij = (λ_params*atom_i.charge) * (λ_params*atom_j.charge)
    cutoff = inter.cutoff

    # 2. Fast Path: Standard Coulomb (λ >= 1.0)
    if λ >= 1
        params = (ke, qij, nothing, nothing)
        pe = pe_cutoff(cutoff, inter, r, params)
        return special ? pe * inter.weight_special : pe
    end

    # 3. Alchemical Path
    σ6 = σ_mixing(inter.σ_mixing, atom_i, atom_j)^6
    σ6_fac = inter.α * (1 - λ) * σ6

    params = (ke, qij, σ6_fac, λ)
    pe = pe_cutoff(cutoff, inter, r, params)
    return special ? pe * inter.weight_special : pe
end

# Dispatch 1: Standard Coulomb PE
@inline function pairwise_pe(::CoulombSoftCoreBeutler, r, (ke, qij, _, _)::Tuple{Any, Any, Any, Nothing, Nothing})
    return (ke * qij) * inv(r)
end

# Dispatch 2: Soft Core PE
@inline function pairwise_pe(::CoulombSoftCoreBeutler, r, (ke, qij, σ6_fac, λ)::Tuple{Any, Any, Any, Any, Any})
    R = sqrt(cbrt(σ6_fac + r^6))
    return λ * ke * ((qij) / R)
end

@doc raw"""
    CoulombSoftCoreGapsys(; cutoff, α, λ, σQ, use_neighbors, weight_special, coulomb_const)

The Coulomb electrostatic interaction between two atoms with a soft core, used for
the appearing and disappearing of atoms.

See [Gapsys et al. 2012](https://doi.org/10.1021/ct300220p).
The potential energy is defined as
```math
V(r_{ij}) = \left\{ \begin{array}{cl}
\lambda \frac{1}{4\pi\epsilon_0} \frac{q_iq_j}{r_{ij}}, & \text{if} & r \ge r_{LJ} \\
\lambda \frac{1}{4\pi\epsilon_0} (\frac{q_iq_j}{r_{Q}^3}r_{ij}^2-\frac{3q_iq_j}{r_{Q}^2}r_{ij}+\frac{3q_iq_j}{r_{Q}}), & \text{if} & r \lt r_{LJ} \\
\end{array} \right.
```
and the force on each atom by
```math
\vec{F}_i = \left\{ \begin{array}{cl}
\lambda \frac{1}{4\pi\epsilon_0} \frac{q_iq_j}{r_{ij}^2}\frac{\vec{r_{ij}}}{r_{ij}}, & \text{if} & r \ge r_{LJ} \\
\lambda \frac{1}{4\pi\epsilon_0} (\frac{-2q_iq_j}{r_{Q}^3}r_{ij}+\frac{3q_iq_j}{r_{Q}^2})\frac{\vec{r_{ij}}}{r_{ij}}, & \text{if} & r \lt r_{LJ} \\
\end{array} \right.
```
where
```math
r_{Q} = \alpha(1-\lambda)^{1/6}(1+σ_Q|qi*qj|)
```

If ``\lambda`` is 1.0, this gives the standard [`Coulomb`](@ref) potential and means
the atom is fully turned on.
If ``\lambda`` is zero the interaction is turned off.
``\alpha`` determines the strength of softening the function.
"""
struct CoulombSoftCoreGapsys{C, A, S, LM, SCH, W, T} <: PairwiseInteraction
    cutoff::C
    α::A
    σQ::S
    use_neighbors::Bool
    λ_mixing::LM
    scheduler::SCH
    weight_special::W
    coulomb_const::T
end

function CoulombSoftCoreGapsys(; cutoff=NoCutoff(), α=1.0, σQ=1.0u"nm", use_neighbors=false,
                                    λ_mixing=MinimumMixing(),
                                    scheduler=DefaultLambdaScheduler(), weight_special=1,
                                    coulomb_const=coulomb_const)
    T = typeof(ustrip(α))
    return CoulombSoftCoreGapsys(
        cutoff,
        α,
        use_neighbors,
        σ_mixing,
        ϵ_mixing,
        λ_mixing,
        scheduler,
        weight_special,
        coulomb_const,
    )
end

use_neighbors(inter::CoulombSoftCoreGapsys) = inter.use_neighbors

function Base.zero(coul::CoulombSoftCoreGapsys{C, A, S, LM, SCH, W, T}) where {C, A, S, LM, SCH, W, T}
    return CoulombSoftCoreGapsys(
        coul.cutoff,
        zero(A),
        coul.σQ,
        coul.use_neighbors,
        coul.λ_mixing,
        coul.scheduler,
        zero(W),
        zero(T),
    )
end

function Base.:+(c1::CoulombSoftCoreGapsys, c2::CoulombSoftCoreGapsys)
    return CoulombSoftCoreGapsys(
        c1.cutoff,
        c1.α + c2.α,
        c1.σQ + c2.σQ,
        c1.use_neighbors,
        c1.λ_mixing,
        c1.scheduler,
        c1.weight_special + c2.weight_special,
        c1.coulomb_const + c2.coulomb_const,
    )
end

function to_lambda_function(inter::Coulomb, gapsys::Val{true}; α=1.0, σQ=1.0u"nm", λ_mixing=MinimumMixing(), scheduler=DefaultLambdaScheduler())
    return CoulombSoftCoreGapsys(cutoff=inter.cutoff, α=α, σQ=σQ, use_neighbors=inter.use_neighbors, λ_mixing=λ_mixing, scheduler=scheduler, 
                                    weight_special=inter.weight_special, coulomb_const=inter.coulomb_const)
end

@inline function force(inter::CoulombSoftCoreGapsys, 
                       dr,
                       atom_i,
                       atom_j,
                       force_units=u"kJ * mol^-1 * nm^-1",
                       special=false,
                       args...)

    ke = inter.coulomb_const
    T = typeof(ustrip(ke))
    λ_glob = T(λ_mixing(inter.λ_mixing, (atom_i.λ, atom_j.λ)))

    # 1. Fetch alchemical roles from the contiguous array
    role_i = atom_i.alch_role
    role_j = atom_j.alch_role
    pair_role = mix_roles(inter.scheduler, (role_i, role_j))

    # 2. Dispatch to the scheduler for the effective sterics lambda
    # Changed scale_elec to scale_sterics
    λ, λ_params = scale_elec(inter.scheduler, λ_glob, pair_role)

    if λ <= 0
        return zero_pairwise_force(dr, force_units)
    end

    r = norm(dr)
    if iszero_value(r)
        return zero_pairwise_force(dr, force_units)
    end

    qij = (λ_params*atom_i.charge) * (λ_params*atom_j.charge)
    cutoff = inter.cutoff

    # Fast Path: Standard Coulomb (Length 4)
    if λ >= 1
        params = (ke, qij, nothing, nothing)
        f = force_cutoff(cutoff, inter, r, params)
        fdr = radial_force_vector(f, r, dr, force_units)
        return special ? fdr * inter.weight_special : fdr
    end

    # Alchemical Path
    # R is precomputed here, saving absolute value and unit math inside the inner loop
    σ6_fac = inter.α * sqrt(cbrt(1 - λ))
    R = σ6_fac * (oneunit(r) + (inter.σQ * abs(qij)))
    params = (ke, qij, λ, R)

    f = force_cutoff(cutoff, inter, r, params)
    fdr = radial_force_vector(f, r, dr, force_units)
    return special ? fdr * inter.weight_special : fdr
end

@inline function pairwise_force(::CoulombSoftCoreGapsys, r, (ke, qij, _, _)::Tuple{Any, Any, Nothing, Nothing})
    return (ke * qij) / r^2
end

@inline function pairwise_force(::CoulombSoftCoreGapsys, r, (ke, qij, λ, R)::Tuple{Any, Any, Any, Any})
    if r >= R
        return λ * ke * (qij / (r^2))
    else
        return λ * ke * (-(((2 * qij) / (R^3)) * r) + ((3 * qij) / (R^2)))
    end
end

@inline function potential_energy(inter::CoulombSoftCoreGapsys,
                                  dr,
                                  atom_i,
                                  atom_j,
                                  energy_units=u"kJ * mol^-1",
                                  special=false,
                                  args...)

    ke = inter.coulomb_const
    T = typeof(ustrip(ke))
    λ_glob = T(λ_mixing(inter.λ_mixing, (atom_i.λ, atom_j.λ)))

    # 1. Fetch alchemical roles from the contiguous array
    role_i = atom_i.alch_role
    role_j = atom_j.alch_role
    pair_role = mix_roles(inter.scheduler, (role_i, role_j))

    # 2. Dispatch to the scheduler for the effective sterics lambda
    # Changed scale_elec to scale_sterics
    λ, λ_params = scale_elec(inter.scheduler, λ_glob, pair_role)

    if λ <= 0
        return ustrip(zero(dr[1])) * energy_units
    end

    r = norm(dr)
    qij = (λ_params*atom_i.charge) * (λ_params*atom_j.charge)
    cutoff = inter.cutoff

    if λ >= 1
        params = (ke, qij, nothing, nothing)
        pe = pe_cutoff(cutoff, inter, r, params)
        return special ? pe * inter.weight_special : pe
    end

    # Precompute R
    σ6_fac = inter.α * sqrt(cbrt(1 - λ))
    R = σ6_fac * (oneunit(r) + (inter.σQ * abs(qij)))
    params = (ke, qij, λ, R)

    pe = pe_cutoff(cutoff, inter, r, params)
    return special ? pe * inter.weight_special : pe
end

@inline function pairwise_pe(::CoulombSoftCoreGapsys, r, (ke, qij, _, _)::Tuple{Any, Any, Nothing, Nothing})
    return (ke * qij) * inv(r)
end

@inline function pairwise_pe(::CoulombSoftCoreGapsys, r, (ke, qij, λ, R)::Tuple{Any, Any, Any, Any})
    if r >= R
        return λ * (ke * (qij/r))
    else
        return λ * (ke * (((qij/(R^3))*(r^2))-(((3*qij)/(R^2))*r)+((3*qij)/R)))
    end
end

const crf_solvent_dielectric = 78.3

@doc raw"""
    CoulombReactionField(; dist_cutoff, solvent_dielectric, use_neighbors, weight_special,
                            coulomb_const)

The Coulomb electrostatic interaction modified using the reaction field approximation
between two atoms.

The potential energy is defined as
```math
V(r_{ij}) = \frac{q_i q_j}{4 \pi \varepsilon_0} \left( \frac{1}{r_{ij}} + k_\mathrm{rf} r_{ij}^2 - c_\mathrm{rf} \right)
```
where
```math
k_\mathrm{rf} = \frac{1}{r_\mathrm{c}^3} \frac{\varepsilon_\mathrm{rf} - 1}{2\varepsilon_\mathrm{rf} + 1}, \quad
c_\mathrm{rf} = \frac{1}{r_\mathrm{c}} \frac{3\varepsilon_\mathrm{rf}}{2\varepsilon_\mathrm{rf} + 1}
```
`solvent_dielectric` corresponds to ``\varepsilon_\mathrm{rf}``.
Setting `solvent_dielectric=Inf` gives conducting boundary conditions
(``k_\mathrm{rf} = 1/(2r_\mathrm{c}^3)``, ``c_\mathrm{rf} = 3/(2r_\mathrm{c})``).
"""
@kwdef struct CoulombReactionField{D, S, W, T} <: PairwiseInteraction
    dist_cutoff::D
    solvent_dielectric::S = crf_solvent_dielectric
    use_neighbors::Bool = false
    weight_special::W = 1
    coulomb_const::T = coulomb_const
end

use_neighbors(inter::CoulombReactionField) = inter.use_neighbors

function Base.zero(coul::CoulombReactionField{D, S, W, T}) where {D, S, W, T}
    return CoulombReactionField{D, S, W, T}(
        zero(D),
        zero(S),
        coul.use_neighbors,
        zero(W),
        zero(T),
    )
end

function Base.:+(c1::CoulombReactionField, c2::CoulombReactionField)
    return CoulombReactionField(
        c1.dist_cutoff + c2.dist_cutoff,
        c1.solvent_dielectric + c2.solvent_dielectric,
        c1.use_neighbors,
        c1.weight_special + c2.weight_special,
        c1.coulomb_const + c2.coulomb_const,
    )
end

function inject_interaction(inter::CoulombReactionField, params_dic)
    key_prefix = "inter_CRF_"
    return CoulombReactionField(
        dict_get(params_dic, key_prefix * "dist_cutoff", inter.dist_cutoff),
        dict_get(params_dic, key_prefix * "solvent_dielectric", inter.solvent_dielectric),
        inter.use_neighbors,
        dict_get(params_dic, key_prefix * "weight_14", inter.weight_special),
        dict_get(params_dic, key_prefix * "coulomb_const", inter.coulomb_const),
    )
end

function extract_parameters!(params_dic, inter::CoulombReactionField, ff)
    key_prefix = "inter_CRF_"
    params_dic[key_prefix * "dist_cutoff"] = inter.dist_cutoff
    params_dic[key_prefix * "solvent_dielectric"] = inter.solvent_dielectric
    params_dic[key_prefix * "weight_14"] = inter.weight_special
    params_dic[key_prefix * "coulomb_const"] = inter.coulomb_const
    return params_dic
end

@inline function force(inter::CoulombReactionField,
                       dr,
                       atom_i,
                       atom_j,
                       force_units=u"kJ * mol^-1 * nm^-1",
                       special=false,
                       args...)
    r2 = sum(abs2, dr)
    ke = inter.coulomb_const
    qi, qj = atom_i.charge, atom_j.charge
    r = sqrt(r2)

    if special
        # 1-4 interactions do not use the reaction field approximation
        krf = inv(inter.dist_cutoff^3) * 0
    else
        # These values could be pre-computed but this way is easier for AD
        if isinf(inter.solvent_dielectric) # Conducting boundary conditions
            krf = inv(2 * inter.dist_cutoff^3)
        else
            krf = inv(inter.dist_cutoff^3) * (inter.solvent_dielectric - 1) / (2 * inter.solvent_dielectric + 1)
        end
    end

    f = (ke * qi * qj) * (inv(r) - 2 * krf * r2) * inv(r2)

    if special
        return f * dr * inter.weight_special * (r <= inter.dist_cutoff)
    else
        return f * dr * (r <= inter.dist_cutoff)
    end
end

@inline function potential_energy(inter::CoulombReactionField,
                                  dr,
                                  atom_i,
                                  atom_j,
                                  energy_units=u"kJ * mol^-1",
                                  special=false,
                                  args...)
    r2 = sum(abs2, dr)
    ke = inter.coulomb_const
    qi, qj = atom_i.charge, atom_j.charge
    r = sqrt(r2)

    if special
        # 1-4 interactions do not use the reaction field approximation
        krf = inv(inter.dist_cutoff^3) * 0
        crf = inv(inter.dist_cutoff) * 0
    else
        if isinf(inter.solvent_dielectric) # conducting boundary conditions
            krf = inv(2 * inter.dist_cutoff^3)
            crf = 3 * inv(2 * inter.dist_cutoff)
        else
            krf = inv(inter.dist_cutoff^3) * (inter.solvent_dielectric - 1) / (2 * inter.solvent_dielectric + 1)
            crf = inv(inter.dist_cutoff) * (3 * inter.solvent_dielectric) / (2 * inter.solvent_dielectric + 1)
        end
    end

    pe = (ke * qi * qj) * (inv(r) + krf * r2 - crf)

    if special
        return pe * inter.weight_special * (r <= inter.dist_cutoff)
    else
        return pe * (r <= inter.dist_cutoff)
    end
end

@doc raw"""
    CoulombSoftCoreBeutlerReactionField(; dist_cutoff, solvent_dielectric, α, use_neighbors,
                                         σ_mixing, ϵ_mixing, λ_mixing, scheduler,
                                         weight_special, coulomb_const)

The Coulomb electrostatic interaction with Beutler soft core and reaction field correction,
used for the appearing and disappearing of atoms in alchemical simulations.

At ``\lambda = 1`` this reduces to [`CoulombReactionField`](@ref).
At ``\lambda = 0`` the interaction is zero.
Special (1-4) pairs disable the reaction field correction and use plain soft-core Coulomb
scaled by `weight_special`.
"""
struct CoulombSoftCoreBeutlerReactionField{D, S, A, SM, EM, LM, SCH, W, T} <: PairwiseInteraction
    dist_cutoff::D
    solvent_dielectric::S 
    α::A 
    use_neighbors::Bool 
    σ_mixing::SM 
    ϵ_mixing::EM 
    λ_mixing::LM 
    scheduler::SCH 
    weight_special::W 
    coulomb_const::T
end

function CoulombSoftCoreBeutlerReactionField(; dist_cutoff, solvent_dielectric=crf_solvent_dielectric, α=1.0,
                                              use_neighbors=false, σ_mixing=LorentzMixing(),
                                              ϵ_mixing=GeometricMixing(), λ_mixing=MinimumMixing(),
                                              scheduler=DefaultLambdaScheduler(), weight_special=1,
                                              coulomb_const=coulomb_const)
    T = typeof(ustrip(dist_cutoff))
    return CoulombSoftCoreBeutlerReactionField(
        dist_cutoff,
        solvent_dielectric,
        α,
        use_neighbors,
        σ_mixing,
        ϵ_mixing,
        λ_mixing,
        scheduler,
        weight_special,
        coulomb_const,
    )
end

use_neighbors(inter::CoulombSoftCoreBeutlerReactionField) = inter.use_neighbors

function Base.zero(coul::CoulombSoftCoreBeutlerReactionField{D, S, A, SM, EM, LM, SCH, W, T}) where {D, S, A, SM, EM, LM, SCH, W, T}
    return CoulombSoftCoreBeutlerReactionField(
        zero(D),
        zero(S),
        zero(A),
        coul.use_neighbors,
        coul.σ_mixing,
        coul.ϵ_mixing,
        coul.λ_mixing,
        coul.scheduler,
        zero(W),
        zero(T),
    )
end

function Base.:+(c1::CoulombSoftCoreBeutlerReactionField, c2::CoulombSoftCoreBeutlerReactionField)
    return CoulombSoftCoreBeutlerReactionField(
        c1.dist_cutoff + c2.dist_cutoff,
        c1.solvent_dielectric + c2.solvent_dielectric,
        c1.α + c2.α,
        c1.use_neighbors,
        c1.σ_mixing,
        c1.ϵ_mixing,
        c1.λ_mixing,
        c1.scheduler,
        c1.weight_special + c2.weight_special,
        c1.coulomb_const + c2.coulomb_const,
    )
end

function to_lambda_function(inter::CoulombReactionField, gapsys::Val{false}; α=1.0, λ_mixing=MinimumMixing(), scheduler=DefaultLambdaScheduler())
    return CoulombSoftCoreBeutlerReactionField(dist_cutoff=inter.dist_cutoff, solvent_dielectric=inter.solvent_dielectric, α=α, 
                                                use_neighbors=inter.use_neighbors, σ_mixing=inter.σ_mixing, ϵ_mixing=inter.ϵ_mixing,
                                                λ_mixing=λ_mixing, scheduler=scheduler, weight_special=inter.weight_special, coulomb_const=inter.coulomb_const)
end

@inline function force(inter::CoulombSoftCoreBeutlerReactionField,
                       dr,
                       atom_i,
                       atom_j,
                       force_units=u"kJ * mol^-1 * nm^-1",
                       special=false,
                       args...)
    ke = inter.coulomb_const
    T = typeof(ustrip(ke))
    λ_glob = T(λ_mixing(inter.λ_mixing, (atom_i.λ, atom_j.λ)))
    pair_role = mix_roles(inter.scheduler, (atom_i.alch_role, atom_j.alch_role))
    λ, λ_params = scale_elec(inter.scheduler, λ_glob, pair_role)

    if λ <= 0
        return zero_pairwise_force(dr, force_units)
    end

    r2 = sum(abs2, dr)
    r = sqrt(r2)
    if iszero_value(r)
        return zero_pairwise_force(dr, force_units)
    end

    qij = (λ_params*atom_i.charge) * (λ_params*atom_j.charge)
    rc = inter.dist_cutoff

    if special
        krf = inv(rc^3) * zero(T)
    else
        krf = inv(rc^3) * ((inter.solvent_dielectric - 1) / (2 * inter.solvent_dielectric + 1))
    end

    if λ >= 1
        f = (ke * qij) * (inv(r) - 2 * krf * r2) * inv(r2)
        if special
            return f * dr * inter.weight_special * (r <= rc)
        else
            return f * dr * (r <= rc)
        end
    end

    σ6 = σ_mixing(inter.σ_mixing, atom_i, atom_j)^6
    term = inter.α * (1 - λ) * σ6 + r2^3
    f = λ * (ke * qij) * (r2^2 / (term * sqrt(cbrt(term))) - 2 * krf)

    if special
        return f * dr * inter.weight_special * (r <= rc)
    else
        return f * dr * (r <= rc)
    end
end

@inline function potential_energy(inter::CoulombSoftCoreBeutlerReactionField,
                                  dr,
                                  atom_i,
                                  atom_j,
                                  energy_units=u"kJ * mol^-1",
                                  special=false,
                                  args...)
    ke = inter.coulomb_const
    T = typeof(ustrip(ke))
    λ_glob = T(λ_mixing(inter.λ_mixing, (atom_i.λ, atom_j.λ)))
    pair_role = mix_roles(inter.scheduler, (atom_i.alch_role, atom_j.alch_role))
    λ, λ_params = scale_elec(inter.scheduler, λ_glob, pair_role)

    if λ <= 0
        return ustrip(zero(dr[1])) * energy_units
    end

    r2 = sum(abs2, dr)
    r = sqrt(r2)
    qij = (λ_params*atom_i.charge) * (λ_params*atom_j.charge)
    rc = inter.dist_cutoff

    if λ >= 1
        if special
            krf = inv(rc^3) * zero(T)
            crf = inv(rc) * zero(T)
        else
            krf = inv(rc^3) * ((inter.solvent_dielectric - 1) / (2 * inter.solvent_dielectric + 1))
            crf = inv(rc) * ((3 * inter.solvent_dielectric) / (2 * inter.solvent_dielectric + 1))
        end
        pe = (ke * qij) * (inv(r) + krf * r2 - crf)
        if special
            return pe * inter.weight_special * (r <= rc)
        else
            return pe * (r <= rc)
        end
    end

    σ6 = σ_mixing(inter.σ_mixing, atom_i, atom_j)^6
    term = inter.α * (1 - λ) * σ6 + r2^3
    R_eff = sqrt(cbrt(term))

    if special
        pe = λ * (ke * qij) * inv(R_eff)
        return pe * inter.weight_special * (r <= rc)
    else
        krf = inv(rc^3) * ((inter.solvent_dielectric - 1) / (2 * inter.solvent_dielectric + 1))
        crf_λ = inv(sqrt(cbrt(inter.α * (1 - λ) * σ6 + rc^6))) + krf * rc^2
        pe = λ * (ke * qij) * (inv(R_eff) + krf * r2 - crf_λ)
        return pe * (r <= rc)
    end
end

@doc raw"""
    CoulombSoftCoreGapsysReactionField(; dist_cutoff, solvent_dielectric, α, σQ,
                                        use_neighbors, λ_mixing, scheduler,
                                        weight_special, coulomb_const)

The Coulomb electrostatic interaction with Gapsys soft core and reaction field correction,
used for the appearing and disappearing of atoms in alchemical simulations.

At ``\lambda = 1`` this reduces to [`CoulombReactionField`](@ref).
At ``\lambda = 0`` the interaction is zero.
Special (1-4) pairs disable the reaction field correction and use the standard Gapsys
polynomial scaled by `weight_special`.
"""
struct CoulombSoftCoreGapsysReactionField{D, S, A, SQ, LM, SCH, W, T} <: PairwiseInteraction
    dist_cutoff::D
    solvent_dielectric::S 
    α::A 
    σQ::SQ 
    use_neighbors::Bool 
    λ_mixing::LM 
    scheduler::SCH 
    weight_special::W 
    coulomb_const::T 
end

function CoulombSoftCoreGapsysReactionField(; dist_cutoff, solvent_dielectric=crf_solvent_dielectric, α=1.0,
                                              σQ=1.0u"nm", use_neighbors=false, λ_mixing=MinimumMixing(),
                                              scheduler=DefaultLambdaScheduler(), weight_special=1,
                                              coulomb_const=coulomb_const)
    T = typeof(ustrip(dist_cutoff))
    return CoulombSoftCoreBeutlerReactionField(
        dist_cutoff,
        solvent_dielectric,
        α,
        use_neighbors,
        σ_mixing,
        ϵ_mixing,
        λ_mixing,
        scheduler,
        weight_special,
        coulomb_const,
    )
end

use_neighbors(inter::CoulombSoftCoreGapsysReactionField) = inter.use_neighbors

function Base.zero(coul::CoulombSoftCoreGapsysReactionField{D, S, A, SQ, LM, SCH, W, T}) where {D, S, A, SQ, LM, SCH, W, T}
    return CoulombSoftCoreGapsysReactionField(
        zero(D),
        zero(S),
        zero(A),
        coul.σQ,
        coul.use_neighbors,
        coul.λ_mixing,
        coul.scheduler,
        zero(W),
        zero(T),
    )
end

function Base.:+(c1::CoulombSoftCoreGapsysReactionField, c2::CoulombSoftCoreGapsysReactionField)
    return CoulombSoftCoreGapsysReactionField(
        c1.dist_cutoff + c2.dist_cutoff,
        c1.solvent_dielectric + c2.solvent_dielectric,
        c1.α + c2.α,
        c1.σQ + c2.σQ,
        c1.use_neighbors,
        c1.λ_mixing,
        c1.scheduler,
        c1.weight_special + c2.weight_special,
        c1.coulomb_const + c2.coulomb_const,
    )
end

function to_lambda_function(inter::CoulombReactionField, gapsys::Val{true}; α=1.0, σQ=1.0u"nm", λ_mixing=MinimumMixing(), scheduler=DefaultLambdaScheduler())
    return CoulombSoftCoreGapsysReactionField(dist_cutoff=inter.dist_cutoff, solvent_dielectric=inter.solvent_dielectric, α=α, σQ=σQ, 
                                    use_neighbors=inter.use_neighbors, λ_mixing=λ_mixing, scheduler=scheduler, 
                                    weight_special=inter.weight_special, coulomb_const=inter.coulomb_const)
end

@inline function force(inter::CoulombSoftCoreGapsysReactionField,
                       dr,
                       atom_i,
                       atom_j,
                       force_units=u"kJ * mol^-1 * nm^-1",
                       special=false,
                       args...)
    ke = inter.coulomb_const
    T = typeof(ustrip(ke))
    λ_glob = T(λ_mixing(inter.λ_mixing, (atom_i.λ, atom_j.λ)))
    pair_role = mix_roles(inter.scheduler, (atom_i.alch_role, atom_j.alch_role))
    λ, λ_params = scale_elec(inter.scheduler, λ_glob, pair_role)

    if λ <= 0
        return zero_pairwise_force(dr, force_units)
    end

    r2 = sum(abs2, dr)
    r = sqrt(r2)
    if iszero_value(r)
        return zero_pairwise_force(dr, force_units)
    end

    qij = (λ_params*atom_i.charge) * (λ_params*atom_j.charge)
    rc = inter.dist_cutoff

    if special
        krf = inv(rc^3) * zero(T)
    else
        krf = inv(rc^3) * ((inter.solvent_dielectric - 1) / (2 * inter.solvent_dielectric + 1))
    end

    if λ >= 1
        f = (ke * qij) * (inv(r) - 2 * krf * r2) * inv(r2)
        if special
            return f * dr * inter.weight_special * (r <= rc)
        else
            return f * dr * (r <= rc)
        end
    end

    R = inter.α * sqrt(cbrt(1 - λ)) * (oneunit(r) + inter.σQ * abs(qij))
    if r >= R
        f = λ * (ke * qij) * (inv(r) - 2 * krf * r2) * inv(r2)
    else
        f = λ * ke * (-2 * qij * (inv(R^3) + krf) + 3 * qij * inv(R^2) * inv(r))
    end

    if special
        return f * dr * inter.weight_special * (r <= rc)
    else
        return f * dr * (r <= rc)
    end
end

@inline function potential_energy(inter::CoulombSoftCoreGapsysReactionField,
                                  dr,
                                  atom_i,
                                  atom_j,
                                  energy_units=u"kJ * mol^-1",
                                  special=false,
                                  args...)
    ke = inter.coulomb_const
    T = typeof(ustrip(ke))
    λ_glob = T(λ_mixing(inter.λ_mixing, (atom_i.λ, atom_j.λ)))
    pair_role = mix_roles(inter.scheduler, (atom_i.alch_role, atom_j.alch_role))
    λ, λ_params = scale_elec(inter.scheduler, λ_glob, pair_role)

    if λ <= 0
        return ustrip(zero(dr[1])) * energy_units
    end

    r2 = sum(abs2, dr)
    r = sqrt(r2)
    qij = (λ_params*atom_i.charge) * (λ_params*atom_j.charge)
    rc = inter.dist_cutoff

    if special
        krf = inv(rc^3) * zero(T)
        crf = inv(rc) * zero(T)
    else
        krf = inv(rc^3) * ((inter.solvent_dielectric - 1) / (2 * inter.solvent_dielectric + 1))
        crf = inv(rc) * ((3 * inter.solvent_dielectric) / (2 * inter.solvent_dielectric + 1))
    end

    if λ >= 1
        pe = (ke * qij) * (inv(r) + krf * r2 - crf)
        if special
            return pe * inter.weight_special * (r <= rc)
        else
            return pe * (r <= rc)
        end
    end

    R = inter.α * sqrt(cbrt(1 - λ)) * (oneunit(r) + inter.σQ * abs(qij))
    if r >= R
        pe = λ * (ke * qij) * (inv(r) + krf * r2 - crf)
    else
        A = qij * (inv(R^3) + krf)
        B = -3 * qij * inv(R^2)
        C = qij * (3 * inv(R) - crf)
        pe = λ * ke * (A * r2 + B * r + C)
    end

    if special
        return pe * inter.weight_special * (r <= rc)
    else
        return pe * (r <= rc)
    end
end

"""
    CoulombEwald(; dist_cutoff, error_tol=0.0005, use_neighbors=false, weight_special=1,
                 coulomb_const=coulomb_const, approximate_erfc=true)

The short range Ewald electrostatic interaction between two atoms.

Should be used alongside the [`Ewald`](@ref) or [`PME`](@ref) general interaction,
which provide the long-range term.
`dist_cutoff` and `error_tol` should match the general interaction.

`dist_cutoff` is the cutoff distance for short range interactions.
`approximate_erfc` determines whether to use a fast approximation to the erfc function.
"""
struct CoulombEwald{T, D, W, C, A} <: PairwiseInteraction
    dist_cutoff::D
    error_tol::T
    use_neighbors::Bool
    weight_special::W
    coulomb_const::C
    α::A
    approximate_erfc::Bool
end

function CoulombEwald(; dist_cutoff, error_tol=0.0005, use_neighbors=false,
                      weight_special=1, coulomb_const=coulomb_const, approximate_erfc=true)
    α = inv(dist_cutoff) * sqrt(-log(2 * error_tol))
    return CoulombEwald(dist_cutoff, error_tol, use_neighbors, weight_special, coulomb_const,
                        α, approximate_erfc)
end

use_neighbors(inter::CoulombEwald) = inter.use_neighbors

function Base.zero(coul::CoulombEwald{T, D, W, C, A}) where {T, D, W, C, A}
    return CoulombEwald(
        zero(D),
        zero(T),
        coul.use_neighbors,
        zero(W),
        zero(C),
        zero(A),
        coul.approximate_erfc,
    )
end

function Base.:+(c1::CoulombEwald, c2::CoulombEwald)
    return CoulombEwald(
        c1.dist_cutoff + c2.dist_cutoff,
        c1.error_tol + c2.error_tol,
        c1.use_neighbors,
        c1.weight_special + c2.weight_special,
        c1.coulomb_const + c2.coulomb_const,
        c1.α + c2.α,
        c1.approximate_erfc,
    )
end

function inject_interaction(inter::CoulombEwald, params_dic)
    key_prefix = "inter_CE_"
    return CoulombEwald(
        dict_get(params_dic, key_prefix * "dist_cutoff", inter.dist_cutoff),
        inter.error_tol,
        inter.use_neighbors,
        dict_get(params_dic, key_prefix * "weight_14", inter.weight_special),
        dict_get(params_dic, key_prefix * "coulomb_const", inter.coulomb_const),
        inter.α,
        inter.approximate_erfc,
    )
end

function extract_parameters!(params_dic, inter::CoulombEwald, ff)
    key_prefix = "inter_CE_"
    params_dic[key_prefix * "dist_cutoff"] = inter.dist_cutoff
    params_dic[key_prefix * "weight_14"] = inter.weight_special
    params_dic[key_prefix * "coulomb_const"] = inter.coulomb_const
    return params_dic
end

function calc_erfc(αr::T, exp_mαr2, approximate_erfc) where T
    if approximate_erfc
        # See the OpenMM source code, Abramowitz and Stegun 1964, and Hastings 1995
        t = inv(one(T) + T(0.3275911) * αr)
        return (T(0.254829592)+(T(-0.284496736)+(T(1.421413741)+
                                    (T(-1.453152027)+T(1.061405429)*t)*t)*t)*t)*t*exp_mαr2
    else
        return erfc(αr)
    end
end

@inline function force(inter::CoulombEwald{T},
                       dr,
                       atom_i,
                       atom_j,
                       force_units=u"kJ * mol^-1 * nm^-1",
                       special=false,
                       args...) where T
    r2 = sum(abs2, dr)
    ke, α = inter.coulomb_const, inter.α
    qi, qj = atom_i.charge, atom_j.charge
    r = sqrt(r2)
    inv_r = inv(r)
    αr = α * r
    exp_mαr2 = exp(-αr^2)
    erfc_αr = calc_erfc(αr, exp_mαr2, inter.approximate_erfc)
    f = ke * qi * qj * inv_r^3
    if special
        # Special interactions excluded from reciprocal calculation
        # so have a standard interaction
        return f * dr * inter.weight_special * (r <= inter.dist_cutoff)
    else
        return f * dr * (erfc_αr + 2 * αr * exp_mαr2 / sqrt(T(π))) * (r <= inter.dist_cutoff)
    end
end

@inline function potential_energy(inter::CoulombEwald,
                                  dr,
                                  atom_i,
                                  atom_j,
                                  energy_units=u"kJ * mol^-1",
                                  special=false,
                                  args...)
    r2 = sum(abs2, dr)
    ke, α = inter.coulomb_const, inter.α
    qi, qj = atom_i.charge, atom_j.charge
    r = sqrt(r2)
    inv_r = inv(r)
    αr = α * r
    exp_mαr2 = exp(-αr^2)
    erfc_αr = calc_erfc(αr, exp_mαr2, inter.approximate_erfc)
    pe = ke * qi * qj * inv_r
    if special
        return pe * inter.weight_special * (r <= inter.dist_cutoff)
    else
        return pe * erfc_αr * (r <= inter.dist_cutoff)
    end
end

@inline function softcore_pair_elec_lambda(inter, atom_i, atom_j)
    T = typeof(ustrip(inter.coulomb_const))
    λ_glob = T(λ_mixing(inter.λ_mixing, (atom_i.λ, atom_j.λ)))
    pair_role = mix_roles(inter.scheduler, (atom_i.alch_role, atom_j.alch_role))
    λ, λ_params = scale_elec(inter.scheduler, λ_glob, pair_role)
    return pair_role, λ, λ_params
end

@inline function softcore_ewald_screen(inter, r)
    T = typeof(ustrip(inter.coulomb_const))
    αr = inter.α_ewald * r
    exp_mαr2 = exp(-αr^2)
    erfc_αr = calc_erfc(αr, exp_mαr2, inter.approximate_erfc)
    force_screen = 2 * inter.α_ewald * exp_mαr2 / sqrt(T(π))
    return erfc_αr, force_screen
end

@doc raw"""
    CoulombSoftCoreBeutlerEwald(; dist_cutoff, error_tol=0.0005, α=1.0,
                                use_neighbors=false, σ_mixing=LorentzMixing(),
                                ϵ_mixing=GeometricMixing(), λ_mixing=MinimumMixing(),
                                scheduler=DefaultLambdaScheduler(), weight_special=1,
                                coulomb_const=coulomb_const, approximate_erfc=true)

The short-range Ewald electrostatic interaction between two atoms with the
Beutler soft-core distance modification.

Should be used alongside [`Ewald`](@ref) or [`PME`](@ref) configured with the same
electrostatic scheduler. Special pairs are excluded from the reciprocal-space
calculation, so their interaction uses the undamped soft-core form.
"""
struct CoulombSoftCoreBeutlerEwald{ET, D, A, SM, EM, LM, SCH, W, C, EA} <: PairwiseInteraction
    dist_cutoff::D
    error_tol::ET
    α::A
    use_neighbors::Bool
    σ_mixing::SM
    ϵ_mixing::EM
    λ_mixing::LM
    scheduler::SCH
    weight_special::W
    coulomb_const::C
    α_ewald::EA
    approximate_erfc::Bool
end

function CoulombSoftCoreBeutlerEwald(; dist_cutoff, error_tol=0.0005, α=1.0,
                                     use_neighbors=false, σ_mixing=LorentzMixing(),
                                     ϵ_mixing=GeometricMixing(), λ_mixing=MinimumMixing(),
                                     scheduler=DefaultLambdaScheduler(), weight_special=1,
                                     coulomb_const=coulomb_const, approximate_erfc=true)
    T = typeof(ustrip(dist_cutoff))
    α_ewald = inv(dist_cutoff) * sqrt(-log(2 * error_tol))
    return CoulombSoftCoreBeutlerEwald(
        dist_cutoff,
        error_tol,
        α,
        use_neighbors,
        σ_mixing,
        ϵ_mixing,
        λ_mixing,
        scheduler,
        weight_special,
        coulomb_const,
        α_ewald,
        approximate_erfc,
    )
end

use_neighbors(inter::CoulombSoftCoreBeutlerEwald) = inter.use_neighbors

function Base.zero(coul::CoulombSoftCoreBeutlerEwald{ET, D, A, SM, EM, LM, SCH, W, C, EA}) where {ET, D, A, SM, EM, LM, SCH, W, C, EA}
    return CoulombSoftCoreBeutlerEwald(
        zero(D),
        zero(ET),
        zero(A),
        coul.use_neighbors,
        coul.σ_mixing,
        coul.ϵ_mixing,
        coul.λ_mixing,
        coul.scheduler,
        zero(W),
        zero(C),
        zero(EA),
        coul.approximate_erfc,
    )
end

function Base.:+(c1::CoulombSoftCoreBeutlerEwald, c2::CoulombSoftCoreBeutlerEwald)
    return CoulombSoftCoreBeutlerEwald(
        c1.dist_cutoff + c2.dist_cutoff,
        c1.error_tol + c2.error_tol,
        c1.α + c2.α,
        c1.use_neighbors,
        c1.σ_mixing,
        c1.ϵ_mixing,
        c1.λ_mixing,
        c1.scheduler,
        c1.weight_special + c2.weight_special,
        c1.coulomb_const + c2.coulomb_const,
        c1.α_ewald + c2.α_ewald,
        c1.approximate_erfc,
    )
end

function to_lambda_function(inter::CoulombEwald, gapsys::Val{false}; α=1.0, λ_mixing=MinimumMixing(), scheduler=DefaultLambdaScheduler())
    return CoulombSoftCoreBeutlerEwald(dist_cutoff=inter.dist_cutoff, error_tol=inter.error_tol, α=α, 
                                                use_neighbors=inter.use_neighbors, σ_mixing=inter.σ_mixing, ϵ_mixing=inter.ϵ_mixing,
                                                λ_mixing=λ_mixing, scheduler=scheduler, weight_special=inter.weight_special, coulomb_const=inter.coulomb_const,
                                                α_ewald=inter.α_ewald, approximate_erfc=inter.approximate_erfc)
end

@inline function force(inter::CoulombSoftCoreBeutlerEwald,
                       dr,
                       atom_i,
                       atom_j,
                       force_units=u"kJ * mol^-1 * nm^-1",
                       special=false,
                       args...)
    pair_role, λ, λ_params = softcore_pair_elec_lambda(inter, atom_i, atom_j)
    if λ <= 0
        return zero_pairwise_force(dr, force_units)
    end

    r = norm(dr)
    if iszero_value(r)
        return zero_pairwise_force(dr, force_units)
    end

    qij = (λ_params*atom_i.charge) * (λ_params*atom_j.charge)
    term = inter.α * (1 - λ) * σ_mixing(inter.σ_mixing, atom_i, atom_j)^6 + r^6
    pe_soft = λ * inter.coulomb_const * ((qij) / sqrt(cbrt(term)))
    f_soft = λ * inter.coulomb_const * ((qij) / (term * sqrt(cbrt(term)))) * r^5

    if special
        return radial_force_vector(f_soft, r, dr, force_units) * inter.weight_special * (r <= inter.dist_cutoff)
    end

    erfc_αr, force_screen = softcore_ewald_screen(inter, r)
    f = (f_soft * erfc_αr) + (pe_soft * force_screen)
    return radial_force_vector(f, r, dr, force_units) * (r <= inter.dist_cutoff)
end

@inline function potential_energy(inter::CoulombSoftCoreBeutlerEwald,
                                  dr,
                                  atom_i,
                                  atom_j,
                                  energy_units=u"kJ * mol^-1",
                                  special=false,
                                  args...)
    pair_role, λ, λ_params = softcore_pair_elec_lambda(inter, atom_i, atom_j)
    if λ <= 0
        return ustrip(zero(dr[1])) * energy_units
    end

    r = norm(dr)
    qij = (λ_params*atom_i.charge) * (λ_params*atom_j.charge)
    term = inter.α * (1 - λ) * σ_mixing(inter.σ_mixing, atom_i, atom_j)^6 + r^6
    pe_soft = inter.coulomb_const * ((qij) / sqrt(cbrt(term)))

    if special
        return pe_soft * inter.weight_special * (r <= inter.dist_cutoff)
    end

    erfc_αr, _ = softcore_ewald_screen(inter, r)
    return pe_soft * erfc_αr * (r <= inter.dist_cutoff)
end

@doc raw"""
    CoulombSoftCoreGapsysEwald(; dist_cutoff, error_tol=0.0005, α=0.3, σQ=1.0u"nm",
                               use_neighbors=false, λ_mixing=MinimumMixing(),
                               scheduler=DefaultLambdaScheduler(), weight_special=1,
                               coulomb_const=coulomb_const, approximate_erfc=true)

The short-range Ewald electrostatic interaction between two atoms with the
Gapsys soft-core polynomial near the origin.

Should be used alongside [`Ewald`](@ref) or [`PME`](@ref) configured with the same
electrostatic scheduler. Special pairs are excluded from the reciprocal-space
calculation, so their interaction uses the undamped soft-core form.
"""
struct CoulombSoftCoreGapsysEwald{ET, D, A, SQ, LM, SCH, W, C, EA} <: PairwiseInteraction
    dist_cutoff::D
    error_tol::ET
    α::A
    σQ::SQ
    use_neighbors::Bool
    λ_mixing::LM
    scheduler::SCH
    weight_special::W
    coulomb_const::C
    α_ewald::EA
    approximate_erfc::Bool
end

function CoulombSoftCoreGapsysEwald(; dist_cutoff, error_tol=0.0005, α=0.3,
                                    σQ=1.0u"nm", use_neighbors=false,
                                    λ_mixing=MinimumMixing(),
                                    scheduler=DefaultLambdaScheduler(),
                                    weight_special=1, coulomb_const=coulomb_const,
                                    approximate_erfc=true)
    T = typeof(ustrip(dist_cutoff))
    α_ewald = inv(dist_cutoff) * sqrt(-log(2 * error_tol))
    return CoulombSoftCoreGapsysEwald(
        dist_cutoff,
        error_tol,
        α,
        σQ,
        use_neighbors,
        λ_mixing,
        scheduler,
        weight_special,
        coulomb_const,
        α_ewald,
        approximate_erfc,
    )
end

use_neighbors(inter::CoulombSoftCoreGapsysEwald) = inter.use_neighbors

function Base.zero(coul::CoulombSoftCoreGapsysEwald{ET, D, A, SQ, LM, SCH, W, C, EA}) where {ET, D, A, SQ, LM, SCH, W, C, EA}
    return CoulombSoftCoreGapsysEwald(
        zero(D),
        zero(ET),
        zero(A),
        coul.σQ,
        coul.use_neighbors,
        coul.λ_mixing,
        coul.scheduler,
        zero(W),
        zero(C),
        zero(EA),
        coul.approximate_erfc,
    )
end

function Base.:+(c1::CoulombSoftCoreGapsysEwald, c2::CoulombSoftCoreGapsysEwald)
    return CoulombSoftCoreGapsysEwald(
        c1.dist_cutoff + c2.dist_cutoff,
        c1.error_tol + c2.error_tol,
        c1.α + c2.α,
        c1.σQ + c2.σQ,
        c1.use_neighbors,
        c1.λ_mixing,
        c1.scheduler,
        c1.weight_special + c2.weight_special,
        c1.coulomb_const + c2.coulomb_const,
        c1.α_ewald + c2.α_ewald,
        c1.approximate_erfc,
    )
end

function to_lambda_function(inter::CoulombEwald, gapsys::Val{true}; α=1.0, σQ=1.0u"nm", λ_mixing=MinimumMixing(), scheduler=DefaultLambdaScheduler())
    return CoulombSoftCoreGapsysEwald(dist_cutoff=inter.dist_cutoff, error_tol=inter.error_tol, α=α, σQ=σQ, 
                                    use_neighbors=inter.use_neighbors, λ_mixing=λ_mixing, scheduler=scheduler, 
                                    weight_special=inter.weight_special, coulomb_const=inter.coulomb_const,
                                    α_ewald=inter.α_ewald, approximate_erfc=inter.approximate_erfc)
end

@inline function force(inter::CoulombSoftCoreGapsysEwald,
                       dr,
                       atom_i,
                       atom_j,
                       force_units=u"kJ * mol^-1 * nm^-1",
                       special=false,
                       args...)
    pair_role, λ, λ_params = softcore_pair_elec_lambda(inter, atom_i, atom_j)
    if λ <= 0
        return zero_pairwise_force(dr, force_units)
    end

    r = norm(dr)
    if iszero_value(r)
        return zero_pairwise_force(dr, force_units)
    end

    qij = (λ_params*atom_i.charge) * (λ_params*atom_j.charge)
    inv_r = inv(r)
    inv_r2 = inv_r^2
    
    if λ >= 1
        pe_soft = inter.coulomb_const * (qij * inv_r)
        f_soft = inter.coulomb_const * (qij * inv_r2)
    else
        R = inter.α * sqrt(cbrt(1 - λ)) * (oneunit(r) + inter.σQ * abs(qij))
        if r >= R
            pe_soft = λ * inter.coulomb_const * (qij * inv_r)
            f_soft = λ * inter.coulomb_const * (qij * inv_r2)
        else
            R2 = R^2
            R3 = R2 * R
            pe_soft = λ * inter.coulomb_const * (((qij / R3) * r^2) - (((3 * qij) / R2) * r) +
                    ((3 * qij) / R))
            f_soft = λ * inter.coulomb_const * (-(((2 * qij) / R3) * r) + ((3 * qij) / R2))
        end
    end

    if special
        return (f_soft * inv_r) * dr * inter.weight_special * (r <= inter.dist_cutoff)
    end

    erfc_αr, force_screen = softcore_ewald_screen(inter, r)
    f = (f_soft * erfc_αr) + (pe_soft * force_screen)
    return f * inv_r * dr * (r <= inter.dist_cutoff)
end

@inline function potential_energy(inter::CoulombSoftCoreGapsysEwald,
                                  dr,
                                  atom_i,
                                  atom_j,
                                  energy_units=u"kJ * mol^-1",
                                  special=false,
                                  args...)
    pair_role, λ, λ_params = softcore_pair_elec_lambda(inter, atom_i, atom_j)
    if λ <= 0
        return ustrip(zero(dr[1])) * energy_units
    end

    r = norm(dr)
    qij = (λ_params*atom_i.charge) * (λ_params*atom_j.charge)

    if λ >= 1
        pe_soft = inter.coulomb_const * (qij / r)
    else
        R = inter.α * sqrt(cbrt(1 - λ)) * (oneunit(r) + inter.σQ * abs(qij))
        if r >= R
            pe_soft = λ * inter.coulomb_const * (qij / r)
        else
            pe_soft = λ * inter.coulomb_const * (((qij / R^3) * r^2) - (((3 * qij) / R^2) * r) +
                    ((3 * qij) / R))
        end
    end

    if special
        return 0*pe_soft
        # return pe_soft * inter.weight_special * (r <= inter.dist_cutoff)
    end

    erfc_αr, _ = softcore_ewald_screen(inter, r)
    return pe_soft * erfc_αr * (r <= inter.dist_cutoff)
end

@inline function force_λ(inter::CoulombSoftCoreGapsysEwald,
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
    d_λ = zero_pairwise_force(dr, force_units)
    pair_role, λ, λ_params = softcore_pair_elec_lambda(inter, atom_i, atom_j)
    di = atom_i.alch_role==ProbRole
    dj = atom_j.alch_role==ProbRole

    r = norm(dr)
    if iszero_value(r)
        return d_λ
    end

    qij = (λ_params*atom_i.charge) * (λ_params*atom_j.charge)
    R = inter.α * sqrt(cbrt(1 - λ)) * (oneunit(r) + inter.σQ * abs(qij))

    if r >= R
        pe_i = (inter.coulomb_const * (qij/r)) * atom_j.λ
        pe_j = (inter.coulomb_const * (qij/r)) * atom_i.λ
    else
        σ_term = inter.α*(oneunit(r)+(inter.σQ*abs(qij))) 
        kqij = inter.coulomb_const * qij
        A = (kqij*r^2)/(σ_term^3)
        B = (-3*kqij*r)/(σ_term^2)
        C = (3*kqij)/σ_term
        pe_i = (2-λ)/(2*(1-λ)*sqrt(1-λ))*A*atom_j.λ  + (3-2*λ)/(3*(1-λ)*cbrt(1-λ)) * B * atom_j.λ + 
                (6-5*λ)/(6*(1-λ)*sqrt(cbrt(1-λ))) * C * atom_j.λ
        pe_j = (2-λ)/(2*(1-λ)*sqrt(1-λ))*A*atom_i.λ  + (3-2*λ)/(3*(1-λ)*cbrt(1-λ)) * B * atom_i.λ + 
                (6-5*λ)/(6*(1-λ)*sqrt(cbrt(1-λ))) * C * atom_i.λ
    end

    if special
        pe_i = pe_i * inter.weight_special * (r <= inter.dist_cutoff) * di
        pe_j = pe_j * inter.weight_special * (r <= inter.dist_cutoff) * dj
        tmp = SVector{3,T}(ustrip(pe_i),ustrip(pe_j),0.0)
        return d_λ .+ (tmp*force_units)
    else
        erfc_αr, _ = softcore_ewald_screen(inter, r)
        pe_i = pe_i * erfc_αr * (r <= inter.dist_cutoff) * di
        pe_j = pe_j * erfc_αr * (r <= inter.dist_cutoff) * dj
        tmp = SVector{3,T}(ustrip(pe_i),ustrip(pe_j),0.0)
        return d_λ .+ (tmp*force_units)
    end
end

@doc raw"""
    Yukawa(; cutoff, use_neighbors, weight_special, coulomb_const, kappa)

The Yukawa electrostatic interaction between two atoms.

The potential energy is defined as
```math
V(r_{ij}) = \frac{q_i q_j}{4 \pi \varepsilon_0 r_{ij}} \exp(-\kappa r_{ij})
```
and the force on each atom by
```math
F(r_{ij}) = \frac{q_i q_j}{4 \pi \varepsilon_0 r_{ij}^2} \exp(-\kappa r_{ij})\left(\kappa r_{ij} + 1\right) \vec{r}_{ij}
```
"""
@kwdef struct Yukawa{C, W, T, K} <: PairwiseInteraction
    cutoff::C = NoCutoff()
    use_neighbors::Bool = false
    weight_special::W = 1
    coulomb_const::T = coulomb_const
    kappa::K = 1.0*u"nm^-1"
end

use_neighbors(inter::Yukawa) = inter.use_neighbors

function Base.zero(yukawa::Yukawa{C, W, T, K}) where {C, W, T, K}
    return Yukawa(
        yukawa.cutoff,
        yukawa.use_neighbors,
        zero(W),
        zero(T),
        zero(K),
    )
end

function Base.:+(c1::Yukawa, c2::Yukawa)
    return Yukawa(
        c1.cutoff,
        c1.use_neighbors,
        c1.weight_special + c2.weight_special,
        c1.coulomb_const + c2.coulomb_const,
        c1.kappa + c2.kappa,
    )
end

@inline function force(inter::Yukawa,
                       dr,
                       atom_i,
                       atom_j,
                       force_units=u"kJ * mol^-1 * nm^-1",
                       special=false,
                       args...)
    r = norm(dr)
    cutoff = inter.cutoff
    coulomb_const = inter.coulomb_const
    qi, qj = atom_i.charge, atom_j.charge
    kappa = inter.kappa
    params = (coulomb_const, qi, qj, kappa)

    f = force_cutoff(cutoff, inter, r, params)
    fdr = (f / r) * dr
    if special
        return fdr * inter.weight_special
    else
        return fdr
    end
end

function pairwise_force(::Yukawa, r, (coulomb_const, qi, qj, kappa))
    return (coulomb_const * qi * qj) * exp(-kappa * r) * (kappa * r + 1) / r^2
end

@inline function potential_energy(inter::Yukawa,
                                  dr,
                                  atom_i,
                                  atom_j,
                                  energy_units=u"kJ * mol^-1",
                                  special::Bool=false,
                                  args...)
    r = norm(dr)
    cutoff = inter.cutoff
    coulomb_const = inter.coulomb_const
    qi, qj = atom_i.charge, atom_j.charge
    params = (coulomb_const, qi, qj, inter.kappa)

    pe = pe_cutoff(cutoff, inter, r, params)
    if special
        return pe * inter.weight_special
    else
        return pe
    end
end

function pairwise_pe(::Yukawa, r, (coulomb_const, qi, qj, kappa))
    return (coulomb_const * qi * qj) * inv(r) * exp(-kappa * r)
end
