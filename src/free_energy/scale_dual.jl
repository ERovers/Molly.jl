# Default Lambda Scheduler

# OpenMM Test Scheduler
@inline function scale(::OpenMMTestScheduler, λ::T, role::AlchemicalRole, dual::Val{true}, args...) where T
    if role == CoreIRole
        return λ, one(λ)
    elseif role == CoreDRole
        return (1-λ), one(λ)
    else
        return one(λ), one(λ)
    end
end

@inline function scale_torsion(::OpenMMTestScheduler, λ::T, role::AlchemicalRole, dual::Val{true}, args...) where T
    if role == CoreIRole
        return (λ,λ,λ,λ,λ,λ)
    elseif role == CoreDRole
        return ((1-λ),(1-λ),(1-λ),(1-λ),(1-λ),(1-λ))
    else
        return (one(λ),one(λ),one(λ),one(λ),one(λ),one(λ))
    end
end

@inline function scale_sterics(::OpenMMTestScheduler, λ::T, role::AlchemicalRole, dual::Val{true}, args...) where T
    if role == InsertRole
        λ = λ < T(0.5) ? T(2.0) * λ : T(1.0)
        return λ, one(λ)
    elseif role == DeleteRole
        λ = λ < T(0.5) ? T(0.0) : T(2.0) * (λ - T(0.5))
        return (1-λ), one(λ)
    elseif role == CoreIRole
        return λ, one(λ)
    elseif role == CoreDRole
        return (1-λ), one(λ)
    else
        return one(λ), one(λ)
    end
end

@inline function scale_elec(::OpenMMTestScheduler, λ::T, role::AlchemicalRole, dual::Val{true}, args...) where T
    if role == InsertRole
        λ = T(λ < T(0.5) ? T(0.0) : T(2.0) * (λ - T(0.5)))
        return λ, one(λ)
    elseif role == DeleteRole
        λ = T(λ < T(0.5) ? T(2.0) * λ : T(1.0))
        return (1-λ), one(λ)
    elseif role == CoreIRole
        return λ, one(λ)
    elseif role == CoreDRole
        return (1-λ), one(λ)
    else
        return one(λ), one(λ)
    end
end