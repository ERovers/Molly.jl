const AlchemicalRole = Int

const EnvRole::AlchemicalRole    = 0
const CoreIRole::AlchemicalRole  = 1
const CoreDRole::AlchemicalRole  = 2
const InsertRole::AlchemicalRole = 3
const DeleteRole::AlchemicalRole = 4
const ProbRole::AlchemicalRole   = 5

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
    elseif any(x->x==CoreIRole, roles)
        return CoreIRole
    elseif any(x->x==CoreDRole, roles)
        return CoreDRole
    else
        return EnvRole
    end
end

struct ProbabilityLambdaScheduler end

@inline function scale_sterics(::ProbabilityLambdaScheduler, λ::T, role::AlchemicalRole, args...) where T
    if role == ProbRole
        return λ
    else
        return T(1.0)
    end
end

@inline function scale_elec(::ProbabilityLambdaScheduler, λ::T, role::AlchemicalRole, args...) where T
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


@inline function scale(::DefaultLambdaScheduler, λ::T, role::AlchemicalRole, args...) where T
    if role == CoreIRole
        return λ
    elseif role == CoreDRole
        return (1-λ)
    else
        return one(λ)
    end
end

@inline function scale_torsion(::DefaultLambdaScheduler, λ::T, role::AlchemicalRole, args...) where T
    if role == InsertRole
        return λ
    elseif role == DeleteRole
        return (1-λ)
    elseif role == CoreIRole
        return λ
    elseif role == CoreDRole
        return (1-λ)
    else
        return one(λ)
    end
end

@inline function scale_sterics(::DefaultLambdaScheduler, λ::T, role::AlchemicalRole, args...) where T
    if role == InsertRole
        λ = λ < T(0.5) ? T(2.0) * λ : T(1.0)
        return λ, one(λ)
    elseif role == DeleteRole
        λ = λ < T(0.5) ? T(0.0) : T(2.0) * (λ - T(0.5))
        return (1-λ), one(λ)
    elseif role == CoreIRole
        return one(λ), λ
    elseif role == CoreDRole
        return one(λ), (1-λ)
    else
        return one(λ), one(λ)
    end
end

@inline function scale_elec(::DefaultLambdaScheduler, λ::T, role::AlchemicalRole, args...) where T
    if role == InsertRole
        λ = T(λ < T(0.5) ? T(0.0) : T(2.0) * (λ - T(0.5)))
        return λ, one(λ)
    elseif role == DeleteRole
        λ = T(λ < T(0.5) ? T(2.0) * λ : T(1.0))
        return (1-λ), one(λ)
    elseif role == CoreIRole
        return one(λ), λ
    elseif role == CoreDRole
        return one(λ), (1-λ)
    else
        return one(λ), one(λ)
    end
end

struct OpenMMTestScheduler end


@inline function scale(::OpenMMTestScheduler, λ::T, role::AlchemicalRole, args...) where T
    if role == CoreIRole
        return λ
    elseif role == CoreDRole
        return (1-λ)
    else
        return one(λ)
    end
end

@inline function scale_torsion(::OpenMMTestScheduler, λ::T, role::AlchemicalRole, args...) where T
    if role == CoreIRole
        return λ
    elseif role == CoreDRole
        return (1-λ)
    else
        return one(λ)
    end
end

@inline function scale_sterics(::OpenMMTestScheduler, λ::T, role::AlchemicalRole, args...) where T
    if role == InsertRole
        λ = λ < T(0.5) ? T(2.0) * λ : T(1.0)
        return λ, one(λ)
    elseif role == DeleteRole
        λ = λ < T(0.5) ? T(0.0) : T(2.0) * (λ - T(0.5))
        return (1-λ), one(λ)
    elseif role == CoreIRole
        return one(λ), λ
    elseif role == CoreDRole
        return one(λ), (1-λ)
    else
        return one(λ), one(λ)
    end
end

@inline function scale_elec(::OpenMMTestScheduler, λ::T, role::AlchemicalRole, args...) where T
    if role == InsertRole
        λ = T(λ < T(0.5) ? T(0.0) : T(2.0) * (λ - T(0.5)))
        return λ, one(λ)
    elseif role == DeleteRole
        λ = T(λ < T(0.5) ? T(2.0) * λ : T(1.0))
        return (1-λ), one(λ)
    elseif role == CoreIRole
        return one(λ), λ
    elseif role == CoreDRole
        return one(λ), (1-λ)
    else
        return one(λ), one(λ)
    end
end

struct NAMDLambdaScheduler end

@inline function scale_sterics(::NAMDLambdaScheduler, λ::T, role::AlchemicalRole, args...) where T
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

@inline function scale_elec(::NAMDLambdaScheduler, λ::T, role::AlchemicalRole, args...) where T
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

@inline function scale_sterics(::QuartersLambdaScheduler, λ::T, role::AlchemicalRole, args...) where T
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

@inline function scale_elec(::QuartersLambdaScheduler, λ::T, role::AlchemicalRole, args...) where T
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

@inline function scale_sterics(::EleScaledLambdaScheduler, λ::T, role::AlchemicalRole, args...) where T
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

@inline function scale_elec(::EleScaledLambdaScheduler, λ::T, role::AlchemicalRole, args...) where T
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