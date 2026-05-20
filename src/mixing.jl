# Mixing functions for non-bonded parameters

shortcut_pair(::Nothing, args...) = false

struct LJZeroShortcut end

function shortcut_pair(::LJZeroShortcut, atom_i, atom_j, args...)
    return iszero_value(atom_i.ϵ) || iszero_value(atom_j.ϵ) ||
           iszero_value(atom_i.σ) || iszero_value(atom_j.σ) 
end

struct BuckinghamZeroShortcut end

function shortcut_pair(::BuckinghamZeroShortcut, atom_i, atom_j, args...)
    return (iszero_value(atom_i.A) || iszero_value(atom_j.A)) &&
           (iszero_value(atom_i.C) || iszero_value(atom_j.C))
end

struct LorentzMixing end

xy_mixing(::LorentzMixing, x, y, args...) = (x + y) / 2
σ_mixing(m::LorentzMixing, atom_i, atom_j, args...) = xy_mixing(m, atom_i.σ , atom_j.σ, args...)
ϵ_mixing(m::LorentzMixing, atom_i, atom_j, args...) = xy_mixing(m, atom_i.ϵ , atom_j.ϵ, args...)
λ_mixing(m::LorentzMixing, atom_i, atom_j, args...) = xy_mixing(m, atom_i.λ , atom_j.λ, args...)
A_mixing(m::LorentzMixing, atom_i, atom_j, args...) = xy_mixing(m, atom_i.A , atom_j.A, args...)
B_mixing(m::LorentzMixing, atom_i, atom_j, args...) = xy_mixing(m, atom_i.B , atom_j.B, args...)
C_mixing(m::LorentzMixing, atom_i, atom_j, args...) = xy_mixing(m, atom_i.C , atom_j.C, args...)

struct GeometricMixing end

xy_mixing(::GeometricMixing, x, y, args...) = sqrt(x * y)
σ_mixing(m::GeometricMixing, atom_i, atom_j, args...) = xy_mixing(m, atom_i.σ, atom_j.σ, args...)
ϵ_mixing(m::GeometricMixing, atom_i, atom_j, args...) = xy_mixing(m, atom_i.ϵ, atom_j.ϵ, args...)
λ_mixing(m::GeometricMixing, atom_i, atom_j, args...) = xy_mixing(m, atom_i.λ, atom_j.λ, args...)
A_mixing(m::GeometricMixing, atom_i, atom_j, args...) = xy_mixing(m, atom_i.A, atom_j.A, args...)
B_mixing(m::GeometricMixing, atom_i, atom_j, args...) = xy_mixing(m, atom_i.B, atom_j.B, args...)
C_mixing(m::GeometricMixing, atom_i, atom_j, args...) = xy_mixing(m, atom_i.C, atom_j.C, args...)

struct WaldmanHaglerMixing end

function σ_mixing(::WaldmanHaglerMixing, atom_i, atom_j, args...)
    T = typeof(ustrip(atom_i.σ))
    return ((atom_i.σ^6 + atom_j.σ^6) / 2) ^ T(1/6)
end

function ϵ_mixing(::WaldmanHaglerMixing, atom_i, atom_j, args...)
    return 2 * sqrt(atom_i.ϵ * atom_j.ϵ) * ((atom_i.σ^3 * atom_j.σ^3) / (atom_i.σ^6 + atom_j.σ^6))
end

struct FenderHalseyMixing end

function ϵ_mixing(::FenderHalseyMixing, atom_i, atom_j, args...)
    return (2 * atom_i.ϵ * atom_j.ϵ) / (atom_i.ϵ + atom_j.ϵ)
end

struct InverseMixing end

B_mixing(::InverseMixing, atom_i, atom_j, args...) = 2 / (inv(atom_i.B) + inv(atom_j.B))

# Dict can be used on CPU but doesn't seem faster than ExceptionList for a few exceptions
function get_pair(d::Dict, i, j, default)
    k1 = (i, j)
    k2 = (j, i)
    if haskey(d, k1)
        return d[k1]
    elseif haskey(d, k2)
        return d[k2]
    else
        return default
    end
end

# GPU-compatible dictionary-like object for pair lookup
struct ExceptionList{N, K, V}
    keys::SVector{N, K}
    values::SVector{N, V}
end

function ExceptionList(d::AbstractDict)
    n = length(d)
    ks = SVector{n}(collect(keys(d)))
    vs = SVector{n}(d[k] for k in ks)
    return ExceptionList(ks, vs)
end

# Avoiding branches helps GPU performance
function get_pair(d::ExceptionList{N}, i, j, default) where N
    k1 = (i, j)
    k2 = (j, i)
    val = default
    for ki in 1:N
        if d.keys[ki] == k1 || d.keys[ki] == k2
            val = d.values[ki]
        end
    end
    return val
end

# Provide exceptions (NBFix) for specific pairs of atom types
struct MixingException{M, E}
    mixing::M
    exceptions::E
end

function σ_mixing(me::MixingException, atom_i, atom_j, args...)
    default = σ_mixing(me.mixing, atom_i, atom_j, args...)
    return get_pair(me.exceptions, atom_i.atom_type, atom_j.atom_type, default)
end

function ϵ_mixing(me::MixingException, atom_i, atom_j, args...)
    default = ϵ_mixing(me.mixing, atom_i, atom_j, args...)
    return get_pair(me.exceptions, atom_i.atom_type, atom_j.atom_type, default)
end

function λ_mixing(me::MixingException, atom_i, atom_j, args...)
    default = λ_mixing(me.mixing, atom_i, atom_j, args...)
    return get_pair(me.exceptions, atom_i.atom_type, atom_j.atom_type, default)
end

struct MinimumMixing end

function λ_mixing(m::MinimumMixing, lambdas::Tuple{Vararg{T}}, args...) where T
    return min(T(1.0), min(lambdas...))
end

struct ProductMixing end

function λ_mixing(m::ProductMixing, (a, b), args...)
    return a.λ*b.λ
end

# CoreMixing
#= 
The logic found in this file is a reinterpretation of how OpenFE deals
with alchemical transformations specifically around core atoms that change properties. 

The original logic can be found in:
https://github.com/OpenFreeEnergy/openfe/blob/main/src/openfe/protocols/openmm_rfe/_rfe_utils/relative.py

=#
struct ParamsList{N, K, V}
    keys::SVector{N, K}
    values::V
end

function ParamsList(d::AbstractDict)
    n = length(d)
    ks = SVector{n}(collect(keys(d)))
    vs = SVector{n}(d[k] for k in ks)
    return ParamsList(ks, vs)
end

function get_params(d::ParamsList{N,K,V}, i, default) where {N,K,V}
    val = (default, default)
    for ki in 1:N
        if d.keys[ki] == i
            val = d.values[ki]
        end
    end
    return val
end

struct CoreMixing{M, P}
    mixing::M
    params::P
end

function σ_mixing(me::CoreMixing, atom_i, atom_j, λ, role, args...)
    if role == EnvRole
        return xy_mixing(me.mixing, atom_i.σ, atom_j.σ, args...)
    end
    sigmaA1, sigmaB1 = get_params(me.params, atom_i.index, atom_i.σ)
    sigmaA2, sigmaB2 = get_params(me.params, atom_j.index, atom_j.σ)
    sigmaA = xy_mixing(me.mixing, sigmaA1, sigmaA2, args...)
    sigmaB = xy_mixing(me.mixing, sigmaB1, sigmaB2, args...)
    return (1-λ)*sigmaA + λ*sigmaB
end

function ϵ_mixing(me::CoreMixing, atom_i, atom_j, λ, role, args...)
    if role == EnvRole
        return xy_mixing(me.mixing, atom_i.ϵ, atom_j.ϵ, args...)
    end
    epsA1, epsB1 = get_params(me.params, atom_i.index, atom_i.ϵ)
    epsA2, epsB2 = get_params(me.params, atom_j.index, atom_j.ϵ)
    epsA = xy_mixing(me.mixing, epsA1, epsA2, args...)
    epsB = xy_mixing(me.mixing, epsB1, epsB2, args...)
    return (1-λ)*epsA + λ*epsB
end

function scale_charge(params::ParamsList{N,K,V}, atom_i, atom_j, λ, role, args...) where {N,K,V}
    if role == EnvRole
        return atom_i.charge, atom_j.charge
    end
    qiA, qiB = get_params(params, atom_i.index, atom_i.charge)
    qjA, qjB = get_params(params, atom_j.index, atom_j.charge)
    return (1-λ)*qiA + λ*qiB, (1-λ)*qjA + λ*qjB
end

function inter_mixing((pA,pB), λ, args...)
    return (1-λ)*pA + λ*pB
end

@inline function torsion_mixing(ks_tuple::NTuple{2, NTuple{N, E}}, λ, role, args...) where {N, E}
    ks_A, ks_B = ks_tuple
    return ntuple(i -> (1-λ) * ks_A[i] + λ * ks_B[i], N)
end