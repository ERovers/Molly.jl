export ImproperTorsion

"""
    ImproperTorsion(; k, θ0)

A improper torsion angle between four atoms.

Only compatible with 3D systems.
"""
struct ImproperTorsion{K, D}
    k::K
    θ0::D
end

ImproperTorsion(; k, θ0) = ImproperTorsion{typeof(k),typeof(θ0)}(k, θ0)

@inline function force(d::ImproperTorsion, coords_i, coords_j, coords_k, coords_l, boundary, args...)
    ab = vector(coords_i, coords_j, boundary)
    bc = vector(coords_j, coords_k, boundary)
    cd = vector(coords_k, coords_l, boundary)
    cross_ab_bc = ab × bc
    cross_bc_cd = bc × cd
    bc_norm = norm(bc)
    θ = atan(
        ustrip(dot(cross_ab_bc × cross_bc_cd, bc / bc_norm)),
        ustrip(dot(cross_ab_bc, cross_bc_cd)),
    )
    
    dEdθ = d.k * (θ - d.θ0) + d.k * (θ - d.θ0)
    fi =  dEdθ * bc_norm * cross_ab_bc / dot(cross_ab_bc, cross_ab_bc)
    fl = -dEdθ * bc_norm * cross_bc_cd / dot(cross_bc_cd, cross_bc_cd)
    v = (dot(-ab, bc) / bc_norm^2) * fi - (dot(-cd, bc) / bc_norm^2) * fl
    fj =  v - fi
    fk = -v - fl
    return SpecificForce4Atoms(fi, fj, fk, fl)
end

@inline function potential_energy(d::ImproperTorsion, coords_i, coords_j, coords_k,
                                  coords_l, boundary, args...)
    θ = torsion_angle(coords_i, coords_j, coords_k, coords_l, boundary)
    return d.k * (θ - d.θ0)^2
end
