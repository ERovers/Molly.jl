const AlchemicalRole = Int32

const EnvRole::AlchemicalRole    = Int32(0)
const CoreRole::AlchemicalRole   = Int32(1)
const CoreIRole::AlchemicalRole  = Int32(2)
const CoreDRole::AlchemicalRole  = Int32(3)
const InsertRole::AlchemicalRole = Int32(4)
const DeleteRole::AlchemicalRole = Int32(5)

# Rule for combining roles during a pairwise interaction.
# Dispatched on the scheduler to allow custom overriding by users.
@inline function mix_roles(::Any, role_i::AlchemicalRole, role_j::AlchemicalRole)
    if role_i == InsertRole || role_j == InsertRole
        return InsertRole
    elseif role_i == DeleteRole || role_j == DeleteRole
        return DeleteRole
    elseif role_i == CoreIRole || role_j == CoreIRole
        return CoreIRole
    elseif role_i == CoreDRole || role_j == CoreDRole
        return CoreDRole
    elseif role_i == CoreRole || role_j == CoreRole
        return CoreRole
    else
        return EnvRole
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
