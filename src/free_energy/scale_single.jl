#= 
The logic found in this file is a reinterpretation of how OpenFE deals
with alchemical transformations. The original logic can be found in:

https://github.com/OpenFreeEnergy/openfe/blob/main/src/openfe/protocols/openmm_rfe/_rfe_utils/lambdaprotocol.py

=#

# Default Lambda Scheduler
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

# OpenMM Test Scheduler
@inline function scale(::OpenMMTestScheduler, λ::T, role::AlchemicalRole, dual::Val{false}, args...) where T
    if role == CoreRole
        return one(λ), λ
    elseif role == InsertRole
        return one(λ), (1-λ)
    elseif role == DeleteRole
        return one(λ), λ
    else
        return one(λ), zero(λ)
    end
end

@inline function scale_torsion(::OpenMMTestScheduler, λ::T, role::AlchemicalRole, dual::Val{false}, args...) where T
    if role == CoreRole
        return ((1-λ),(1-λ),(1-λ),(1-λ),(1-λ),(1-λ),λ,λ,λ,λ,λ,λ)
    else
        λ = one(λ)
        return (λ,λ,λ,λ,λ,λ,λ,λ,λ,λ,λ,λ)
    end
end

@inline function scale_sterics(::OpenMMTestScheduler, λ::T, role::AlchemicalRole, dual::Val{false}, args...) where T
    if role == InsertRole
        λ = λ < T(0.5) ? T(2.0) * λ : T(1.0)
        return one(λ), λ
    elseif role == DeleteRole
        λ = λ < T(0.5) ? T(0.0) : T(2.0) * (λ - T(0.5))
        return one(λ), λ
    elseif role == CoreRole
        return one(λ), λ
    else
        return one(λ), one(λ)
    end
end

@inline function scale_elec(::OpenMMTestScheduler, λ::T, role::AlchemicalRole, dual::Val{false}, args...) where T
    if role == InsertRole
        λ = T(λ < T(0.5) ? T(0.0) : T(2.0) * (λ - T(0.5)))
        return one(λ), λ
    elseif role == DeleteRole
        λ = T(λ < T(0.5) ? T(2.0) * λ : T(1.0))
        return one(λ), λ
    elseif role == CoreRole
        return one(λ), λ
    else
        return one(λ), one(λ)
    end
end

# NAMD Lambda Scheduler

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

# Quarters Lambda Scheduler
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

# Electrostatics Scaled Lambda Scheduler
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

#Probability Lambda Scheduler
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

# @inline function mix_roles(::ProbabilityLambdaScheduler, role_i::AlchemicalRole, role_j::AlchemicalRole)
#     if role_i == ProbRole || role_j == ProbRole
#         return ProbRole
#     else
#         return CoreRole
#     end
# end