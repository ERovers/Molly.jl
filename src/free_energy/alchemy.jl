const AlchemicalRole = Int32

const EnvRole::AlchemicalRole    = 0
const CoreRole::AlchemicalRole   = 1
const CoreIRole::AlchemicalRole  = 2
const CoreDRole::AlchemicalRole  = 3
const InsertRole::AlchemicalRole = 4
const DeleteRole::AlchemicalRole = 5
const ProbRole::AlchemicalRole   = 6

# Rule for combining roles during a pairwise interaction.
# Dispatched on the scheduler to allow custom overriding by users.
@inline function mix_roles(::Any, role_i::AlchemicalRole, role_j::AlchemicalRole)
    if role_i == InsertRole || role_j == InsertRole
        return InsertRole
    elseif role_i == DeleteRole || role_j == DeleteRole
        return DeleteRole
    else
        return CoreRole
    end
end

@inline function mix_roles(::Any, roles::Tuple{Vararg{AlchemicalRole}})
    if any(x->x==InsertRole, roles)
        return InsertRole
    elseif any(x->x==DeleteRole, roles)
        return DeleteRole
    elseif any(x->x==CoreRole, roles)
        return CoreRole
    elseif any(x->x==CoreIRole, roles)
        return CoreIRole
    elseif any(x->x==CoreDRole, roles)
        return CoreDRole
    else
        return EnvRole
    end
end

function switchAB(alch_role::Val{DeleteRole}, A, B)
    return A,A
end

function switchAB(alch_role::Val{InsertRole}, A, B)
    return B,B
end

@kwdef struct DefaultLambdaScheduler
    dual::Bool = true
end
@kwdef struct OpenMMTestScheduler 
    dual::Bool = true
end
@kwdef struct NAMDLambdaScheduler
    dual::Bool = true
end
@kwdef struct QuartersLambdaScheduler 
    dual::Bool = true
end
@kwdef struct EleScaledLambdaScheduler 
    dual::Bool = true
end
@kwdef struct ProbabilityLambdaScheduler 
    dual::Bool = true
end
