export
    lambda4,
    Atom_L


function lambda4(l1, l2, l3, l4)
    if l1 == l2 == l3 == l4
        return min((l1, l2, l3, l4))
    else
        return one(l1)
    end
end

"""
    Atom_L(; <keyword arguments>)

Similar to Atom Type, but with λ scaling for alchemical transformations.

# Arguments
- `index::Int`: the index of the atom in the system.
- `atom_type::T`: the type of the atom.
- `mass::M=1.0u"g/mol"`: the mass of the atom.
- `charge::C=0.0`: the charge of the atom, used for electrostatic interactions.
- `σ::S=0.0u"nm"`: the Lennard-Jones finite distance at which the inter-particle
    potential is zero.
- `ϵ::E=0.0u"kJ * mol^-1"`: the Lennard-Jones depth of the potential well.
- `λ::L=1.0: the λ scaling factor (if λ is 1.0, all potentials are regular energy potentials)
"""

@kwdef struct Atom_L{T, M, C, S, E, L}
    index::Int = 1
    atom_type::T = 1
    mass::M = 1.0u"g/mol"
    charge::C = 0.0
    σ::S = 0.0u"nm"
    ϵ::E = 0.0u"kJ * mol^-1"
    λ::L = 1.0
end

function Base.show(io::IO, a::Atom_L)
    print(io, "Atom with index=", a.index, ", atom_type=", a.atom_type, ", mass=", mass(a),
          ", charge=", charge(a), ", σ=", a.σ, ", ϵ=", a.ϵ, ", λ=", a.λ)
end

# Shortcuts
function lj_λ_less_one_shortcut(atom_i, atom_j)
    return iszero_value(atom_i.ϵ) || iszero_value(atom_j.ϵ) ||
           iszero_value(atom_i.σ) || iszero_value(atom_j.σ) ||
           (atom_i.λ<1) || (atom_j.λ<1)
end

function lj_λ_one_shortcut(atom_i, atom_j)
    return ( iszero(atom_i.ϵ) || iszero(atom_j.ϵ) ||
           iszero(atom_i.σ) || iszero(atom_j.σ) ) ||
           ( (atom_i.λ==1) && (atom_j.λ==1) )
end

function coul_zero_shortcut(atom_i, atom_j)
    return iszero_value(atom_i.charge) || iszero_value(atom_j.charge)
end

function coul_λ_less_one_shortcut(atom_i, atom_j)
    return iszero_value(atom_i.charge) || iszero_value(atom_j.charge) ||
           (atom_i.λ<1) || (atom_j.λ<1)
end

function coul_λ_one_shortcut(atom_i, atom_j)
    return ( iszero(atom_i.charge) || iszero(atom_j.charge) ) ||
           ( (atom_i.λ==1) && (atom_j.λ==1) )
end


# Torsion functions
@inline function force(d::PeriodicTorsion, coords_i, coords_j, coords_k,
                       coords_l, boundary, atom_i::Atom_L, atom_j::Atom_L, 
                        atom_k::Atom_L, atom_l::Atom_L, args...)
    ab, bc, cd, cross_ab_bc, cross_bc_cd, bc_norm, θ = periodic_torsion_vectors(
                                        coords_i, coords_j, coords_k, coords_l, boundary)
    fs = sum(zip(d.periodicities, d.phases, d.ks)) do (periodicity, phase, k)
        fi, fj, fk, fl = periodic_torsion_force(periodicity, phase, k, ab, bc, cd, cross_ab_bc,
                                                cross_bc_cd, bc_norm, θ)
        l = lambda4(atom_i.λ, atom_j.λ, atom_k.λ, atom_l.λ)
        return SpecificForce4Atoms(l * fi, l * fj, l * fk, l * fl)
    end
    return fs
end

@inline function force_gpu(d::PeriodicTorsion{N}, coords_i, coords_j, coords_k,
                           coords_l, boundary, atom_i::Atom_L, atom_j::Atom_L, 
                        atom_k::Atom_L, atom_l::Atom_L, args...) where N
    ab, bc, cd, cross_ab_bc, cross_bc_cd, bc_norm, θ = periodic_torsion_vectors(
                                        coords_i, coords_j, coords_k, coords_l, boundary)
    fi_sum, fj_sum, fk_sum, fl_sum = periodic_torsion_force(d.periodicities[1], d.phases[1],
                                        d.ks[1], ab, bc, cd, cross_ab_bc, cross_bc_cd, bc_norm, θ)
    l = min(atom_i.λ, atom_j.λ, atom_k.λ, atom_l.λ)
    for i in 2:N
        fi, fj, fk, fl = periodic_torsion_force(d.periodicities[i], d.phases[i], d.ks[i], ab, bc,
                                                cd, cross_ab_bc, cross_bc_cd, bc_norm, θ)
        fi_sum += fi
        fj_sum += fj
        fk_sum += fk
        fl_sum += fl
    end
    return SpecificForce4Atoms(l * fi_sum, l * fj_sum, l * fk_sum, l * fl_sum)
end

# Interactions
function Base.append!(il1::InteractionList1Atoms{I, T}, il2::InteractionList1Atoms{I, T}) where {I, T}
    return InteractionList1Atoms{I, T}(
        append!(il1.is,il2.is),
        append!(il1.inters,il2.inters),
        append!(il1.types,il2.types)
    )
end

function Base.append!(il1::InteractionList2Atoms{I, T}, il2::InteractionList2Atoms{I, T}) where {I, T}
    return InteractionList2Atoms{I, T}(
        append!(il1.is,il2.is),
        append!(il1.js,il2.js),
        append!(il1.inters,il2.inters),
        append!(il1.types,il2.types)
    )
end

function Base.append!(il1::InteractionList3Atoms{I, T}, il2::InteractionList3Atoms{I, T}) where {I, T}
    return InteractionList3Atoms{I, T}(
        append!(il1.is,il2.is),
        append!(il1.js,il2.js),
        append!(il1.ks,il2.ks),
        append!(il1.inters,il2.inters),
        append!(il1.types,il2.types)
    )
end

function Base.append!(il1::InteractionList4Atoms{I, T}, il2::InteractionList4Atoms{I, T}) where {I, T}
    return InteractionList4Atoms{I, T}(
        append!(il1.is,il2.is),
        append!(il1.js,il2.js),
        append!(il1.ks,il2.ks),
        append!(il1.ls,il2.ls),
        append!(il1.inters,il2.inters),
        append!(il1.types,il2.types)
    )
end