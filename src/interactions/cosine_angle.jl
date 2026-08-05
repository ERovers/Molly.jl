export CosineAngle

@doc raw"""
    CosineAngle(; k, θ0)

A cosine bond angle between three atoms.

`θ0` is in radians.
The potential energy is defined as
```math
V(\theta) = k(1 + \cos(\theta - \theta_0))
```
"""
@kwdef struct CosineAngle{K, D}
    k::K
    θ0::D
end

@inline function force(a::CosineAngle, coords_i, coords_j, coords_k, boundary, args...)
    # In 2D we use then eliminate the cross product
    ba = vector_pad3D(coords_j, coords_i, boundary)
    bc = vector_pad3D(coords_j, coords_k, boundary)
    cross_ba_bc = ba × bc
    if iszero_value(cross_ba_bc)
        zf = zero(a.k ./ trim3D(ba, boundary))
        return SpecificForce3Atoms(zf, zf, zf)
    end
    pa = normalize(trim3D( ba × cross_ba_bc, boundary))
    pc = normalize(trim3D(-bc × cross_ba_bc, boundary))
    θ = bond_angle(ba, bc)
    angle_term = a.k * sin(θ - a.θ0)
    fa = (angle_term / norm(ba)) * pa
    fc = (angle_term / norm(bc)) * pc
    fb = -fa - fc
    return SpecificForce3Atoms(fa, fb, fc)
end

@inline function potential_energy(a::CosineAngle, coords_i, coords_j,
                                  coords_k, boundary, args...)
    θ = bond_angle(coords_i, coords_j, coords_k, boundary)
    return a.k * (1 + cos(θ - a.θ0))
end

@doc raw"""
    CosineAngleλ(; k, θ0)

A cosine bond angle between three atoms scaled by λ for core atoms.

`θ0` is in radians.
The potential energy is defined as
```math
V(\theta) = k(1 + \cos(\theta - \theta_0))
```
"""
@kwdef struct CosineAngleλ{K, D}
    k::K
    θ0::D
end

function to_lambda_function(inter::CosineAngle; λ_mixing=MinimumMixing(), scheduler=DefaultLambdaScheduler())
    return CosineAngleλ(k=inter.k, θ0=inter.θ0, λ_mixing=λ_mixing, scheduler=scheduler)
end

@inline function force(a::CosineAngleλ, coords_i, coords_j, coords_k, boundary, args...)
    # In 2D we use then eliminate the cross product
    ba = vector_pad3D(coords_j, coords_i, boundary)
    bc = vector_pad3D(coords_j, coords_k, boundary)
    cross_ba_bc = ba × bc
    if iszero_value(cross_ba_bc)
        zf = zero(a.k ./ ba)
        return SpecificForce3Atoms(zf, zf, zf)
    end
    pa = normalize(trim3D( ba × cross_ba_bc, boundary))
    pc = normalize(trim3D(-bc × cross_ba_bc, boundary))
    θ = bond_angle(ba, bc)
    angle_term = a.k * sin(θ - a.θ0)
    fa = (angle_term / norm(ba)) * pa
    fc = (angle_term / norm(bc)) * pc
    fb = -fa - fc

    λ_glob = T(λ_mixing(a.λ_mixing, (atom_i.λ, atom_j.λ, atom_k.λ)))    
    pair_role = mix_roles(a.scheduler, (atom_i.alch_role, atom_j.alch_role, atom_k.alch_role))
    λ = scale(a.scheduler, λ_glob, pair_role)

    return SpecificForce3Atoms(λ*fa, λ*fb, λ*fc)
end

@inline function potential_energy(a::CosineAngleλ, coords_i, coords_j,
                                  coords_k, boundary, args...)
    θ = bond_angle(coords_i, coords_j, coords_k, boundary)

    λ_glob = T(λ_mixing(a.λ_mixing, (atom_i.λ, atom_j.λ, atom_k.λ)))    
    pair_role = mix_roles(a.scheduler, (atom_i.alch_role, atom_j.alch_role, atom_k.alch_role))
    λ = scale(a.scheduler, λ_glob, pair_role)

    return λ * a.k * (1 + cos(θ - a.θ0))
end
