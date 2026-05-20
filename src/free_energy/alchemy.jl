const AlchemicalRole = Int

const EnvRole::AlchemicalRole    = 0
const CoreRole::AlchemicalRole   = 1
const InsertRole::AlchemicalRole = 2
const DeleteRole::AlchemicalRole = 3
const ProbRole::AlchemicalRole   = 4

#= 
The logic found in this file is a reinterpretation of how OpenFE deals
with alchemical transformations. The original logic can be found in:

https://github.com/OpenFreeEnergy/openfe/blob/main/src/openfe/protocols/openmm_rfe/_rfe_utils/lambdaprotocol.py

=#

# Rule for combining roles during a pairwise interaction.
# Dispatched on the scheduler to allow custom overriding by users.
@inline function mix_roles(::Any, roles::Tuple{Vararg{AlchemicalRole}})
    if any(x->x==InsertRole, roles)
        return InsertRole
    elseif any(x->x==DeleteRole, roles)
        return DeleteRole
    elseif any(x->x==CoreRole, roles)
        return CoreRole
    else
        return EnvRole
    end
end

struct ProbabilityLambdaScheduler end

@inline function scale_sterics(::ProbabilityLambdaScheduler, λ::T, role::AlchemicalRole) where T
    if role == ProbRole
        return λ
    else
        return T(1.0)
    end
end

@inline function scale_elec(::ProbabilityLambdaScheduler, λ::T, role::AlchemicalRole) where T
    if role == ProbRole
        return λ
    else
        return T(1.0)
    end
end

@inline function mix_roles(::ProbabilityLambdaScheduler, role_i::AlchemicalRole, role_j::AlchemicalRole)
    if role_i == ProbRole || role_j == ProbRole
        return ProbRole
    else
        return CoreRole
    end
end

struct DefaultLambdaScheduler end


@inline function scale(::DefaultLambdaScheduler, λ::T, role::AlchemicalRole) where T
    if role == CoreRole
        return λ
    else
        return one(λ)
    end
end

@inline function scale_torsion(::DefaultLambdaScheduler, λ::T, role::AlchemicalRole) where T
    if role == InsertRole
        return λ, λ
    elseif role== DeleteRole
        return (1-λ), λ
    else
        return one(λ), λ
    end
end

@inline function scale_sterics(::DefaultLambdaScheduler, λ::T, role::AlchemicalRole) where T
    if role == InsertRole
        λ = λ < T(0.5) ? T(2.0) * λ : T(1.0)
        return λ, λ
    elseif role == DeleteRole
        λ = λ < T(0.5) ? T(0.0) : T(2.0) * (λ - T(0.5))
        return (1-λ), λ
    else
        return one(λ), λ
    end
end

@inline function scale_elec(::DefaultLambdaScheduler, λ::T, role::AlchemicalRole) where T
    if role == InsertRole
        λ = T(λ < T(0.5) ? T(0.0) : T(2.0) * (λ - T(0.5)))
        return λ, λ
    elseif role == DeleteRole
        λ = T(λ < T(0.5) ? T(2.0) * λ : T(1.0))
        return (1-λ), λ
    else
        return one(λ), λ
    end
end

struct NAMDLambdaScheduler end

@inline function scale_sterics(::NAMDLambdaScheduler, λ::T, role::AlchemicalRole) where T
    if role == InsertRole
        λ = T(λ < (T(2.0) / T(3.0)) ? (T(3.0) / T(2.0)) * λ : T(1.0))
        return λ, λ
    elseif role == DeleteRole
        λ = T(λ < (T(1.0) / T(3.0)) ? T(0.0) : (λ - (T(1.0) / T(3.0))) * (T(3.0) / T(2.0)))
        return (1-λ), λ
    else
        return one(λ), λ
    end
end

@inline function scale_elec(::NAMDLambdaScheduler, λ::T, role::AlchemicalRole) where T
    if role == InsertRole
        λ = T(λ < T(0.5) ? T(0.0) : T(2.0) * (λ - T(0.5)))
        return λ, λ
    elseif role == DeleteRole
        λ = T(λ < T(0.5) ? T(2.0) * λ : T(1.0))
        return (1-λ), λ
    else
        return one(λ), λ
    end
end

struct QuartersLambdaScheduler end

@inline function scale_sterics(::QuartersLambdaScheduler, λ::T, role::AlchemicalRole) where T
    if role == InsertRole
        λ = λ < T(0.5) ? T(0.0) : (λ > T(0.75) ? T(1.0) : T(4.0) * (λ - T(0.5)))
        return λ, λ
    elseif role == DeleteRole
        λ = λ < T(0.25) ? T(0.0) : (λ > T(0.5) ? T(1.0) : T(4.0) * (λ - T(0.25)))
        return (1-λ), λ
    else
        return one(λ), λ
    end
end

@inline function scale_elec(::QuartersLambdaScheduler, λ::T, role::AlchemicalRole) where T
    if role == InsertRole
        λ = λ < T(0.75) ? T(0.0) : T(4.0) * (λ - T(0.75))
        return λ, λ
    elseif role == DeleteRole
        λ = λ < T(0.25) ? T(4.0) * λ : T(1.0)
        return (1-λ), λ
    else
        return one(λ), λ
    end
end

struct EleScaledLambdaScheduler end

@inline function scale_sterics(::EleScaledLambdaScheduler, λ::T, role::AlchemicalRole) where T
    if role == InsertRole
        λ = λ < T(0.5) ? T(2.0) * λ : T(1.0)
        return λ, λ
    elseif role == DeleteRole
        λ = λ < T(0.5) ? T(0.0) : T(2.0) * (λ - T(0.5))
        return (1-λ), λ
    else
        return one(λ), λ
    end
end

@inline function scale_elec(::EleScaledLambdaScheduler, λ::T, role::AlchemicalRole) where T
    if role == InsertRole
        λ = λ < T(0.5) ? T(0.0) : sqrt(T(2.0) * (λ - T(0.5)))
        return λ, λ
    elseif role == DeleteRole
        λ = λ < T(0.5) ? (T(2.0) * λ)^2 : T(1.0)
        return (1-λ), λ
    else
        return one(λ), λ
    end
end