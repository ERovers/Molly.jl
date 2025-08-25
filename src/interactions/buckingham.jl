export Buckingham

function buckingham_zero_shortcut(atom_i, atom_j)
    return (iszero_value(atom_i.A) || iszero_value(atom_j.A)) &&
           (iszero_value(atom_i.C) || iszero_value(atom_j.C))
end

function geometric_A_mixing(atom_i, atom_j)
    return sqrt(atom_i.A * atom_j.A)
end

function geometric_B_mixing(atom_i, atom_j)
    return sqrt(atom_i.B * atom_j.B)
end

function geometric_C_mixing(atom_i, atom_j)
    return sqrt(atom_i.C * atom_j.C)
end

function inverse_B_mixing(atom_i, atom_j)
    return 2 / (inv(atom_i.B) + inv(atom_j.B))
end

@doc raw"""
    Buckingham(; cutoff, use_neighbors, shortcut, A_mixing, B_mixing,
               C_mixing, weight_special)

The Buckingham interaction between two atoms.

The potential energy is defined as
```math
V(r_{ij}) = A_{ij} \exp(-B_{ij} r_{ij}) - \frac{C_{ij}}{r_{ij}^6}
```
and the force on each atom by
```math
\vec{F}_i = \left( A_{ij} B_{ij} \exp(-B_{ij} r_{ij}) - 6 \frac{C_{ij}}{r_{ij}^7} \right) \frac{\vec{r}_{ij}}{r_{ij}}
```
The parameters are derived from the atom parameters according to
```math
\begin{aligned}
A_{ij} &= (A_{ii} A_{jj})^{1/2} \\
B_{ij} &= \frac{2}{\frac{1}{B_{ii}} + \frac{1}{B_{jj}}} \\
C_{ij} &= (C_{ii} C_{jj})^{1/2}
\end{aligned}
```
so atoms that use this interaction should have fields `A`, `B` and `C` available.
"""
@kwdef struct Buckingham{C, S, A, B, M, W}
    cutoff::C = NoCutoff()
    use_neighbors::Bool = false
    shortcut::S = buckingham_zero_shortcut
    A_mixing::A = geometric_A_mixing
    B_mixing::B = inverse_B_mixing
    C_mixing::M = geometric_C_mixing
    weight_special::W = 1
end

use_neighbors(inter::Buckingham) = inter.use_neighbors

function Base.zero(b::Buckingham{C, W}) where {C, W}
    return Buckingham(b.cutoff, b.use_neighbors, b.shortcut, b.A_mixing,
                      b.B_mixing, b.C_mixing, zero(W))
end

function Base.:+(b1::Buckingham, b2::Buckingham)
    return Buckingham(b1.cutoff, b1.use_neighbors, b1.shortcut, b1.A_mixing, b1.B_mixing,
                      b1.C_mixing, b1.weight_special + b2.weight_special)
end

@inline function force(inter::Buckingham,
                       dr,
                       atom_i,
                       atom_j,
                       force_units=u"kJ * mol^-1 * nm^-1",
                       special=false,
                       args...)
    if inter.shortcut(atom_i, atom_j)
        return ustrip.(zero(dr)) * force_units
    end
    A = inter.A_mixing(atom_i, atom_j)
    B = inter.B_mixing(atom_i, atom_j)
    C = inter.C_mixing(atom_i, atom_j)

    cutoff = inter.cutoff
    r = norm(dr)
    params = (A, B, C)

    f = force_cutoff(cutoff, inter, r, params, force_units)
    fdr = (f / r) * dr
    if special
        return fdr * inter.weight_special
    else
        return fdr
    end
end

function pairwise_force(::Buckingham, r, (A, B, C))
    return A * B * exp(-B * r) - 6 * C * inv(r)^7
end

@inline function potential_energy(inter::Buckingham,
                                  dr,
                                  atom_i,
                                  atom_j,
                                  energy_units=u"kJ * mol^-1",
                                  special=false,
                                  args...)
    if inter.shortcut(atom_i, atom_j)
        return ustrip(zero(dr[1])) * energy_units
    end
    A = inter.A_mixing(atom_i, atom_j)
    B = inter.B_mixing(atom_i, atom_j)
    C = inter.C_mixing(atom_i, atom_j)

    cutoff = inter.cutoff
    r = norm(dr)
    params = (A, B, C)

    pe = pe_cutoff(cutoff, inter, r, params, energy_units)
    if special
        return pe * inter.weight_special
    else
        return pe
    end
end

function pairwise_pe(::Buckingham, r, (A, B, C))
    return A * exp(-B * r) - C * inv(r)^6
end
