# Alchemical Roles
const AlchemicalRole = Int32

const EnvRole::AlchemicalRole    = Int32(0)
const CoreRole::AlchemicalRole   = Int32(1)
const CoreIRole::AlchemicalRole  = Int32(2)
const CoreDRole::AlchemicalRole  = Int32(3)
const InsertRole::AlchemicalRole = Int32(4)
const DeleteRole::AlchemicalRole = Int32(5)

# SoftCore potential options
abstract type SoftCore end

struct DefaultSoftCore <: SoftCore end
struct BeutlerSoftCore <: SoftCore end
struct GapsysSoftCore  <: SoftCore end
struct ScaledSoftCore  <: SoftCore end

# Lambda Schedulers
@kwdef struct DefaultLambdaScheduler
    dual::Bool = true
    LJindividual::Bool = false
    LJspecial::Bool = false
    Cindividual::Bool = false
    Cspecial::Bool = false
    intraLJ::Bool = false
    intraC::Bool = true
end
@kwdef struct LinearLambdaScheduler
    dual::Bool = true
    LJindividual::Bool = false
    LJspecial::Bool = false
    Cindividual::Bool = false
    Cspecial::Bool = false
    intraLJ::Bool = false
    intraC::Bool = true
end
@kwdef struct OpenFEScheduler 
    dual::Bool = false
    LJindividual::Bool = false
    LJspecial::Bool = false
    Cindividual::Bool = true
    Cspecial::Bool = false
    intraLJ::Bool = false
    intraC::Bool = true
end
@kwdef struct NAMDLambdaScheduler
    dual::Bool = true
    LJindividual::Bool = false
    LJspecial::Bool = false
    Cindividual::Bool = false
    Cspecial::Bool = false
    intraLJ::Bool = false
    intraC::Bool = true
end
@kwdef struct QuartersLambdaScheduler 
    dual::Bool = true
    LJindividual::Bool = false
    LJspecial::Bool = false
    Cindividual::Bool = false
    Cspecial::Bool = false
    intraLJ::Bool = false
    intraC::Bool = true
end
@kwdef struct EleScaledLambdaScheduler 
    dual::Bool = true
    LJindividual::Bool = false
    LJspecial::Bool = false
    Cindividual::Bool = false
    Cspecial::Bool = false
    intraLJ::Bool = false
    intraC::Bool = true
end

@inline function mix_default(roles::Tuple{Vararg{AlchemicalRole}})
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

@inline function mix_special(roles::Tuple{Vararg{AlchemicalRole}})
    if all(x->x==InsertRole, roles)
        return EnvRole
    elseif any(x->x==InsertRole, roles)
        return InsertRole
    elseif all(x->x==DeleteRole, roles)
        return EnvRole
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

@inline function mix_roles(inter::Any, roles::Tuple{Vararg{AlchemicalRole}}; type="")
    if type=="coulomb" && inter.intraC
        return mix_special(roles)
    elseif type=="LJ" && inter.intraLJ
        return mix_special(roles)
    else
        return mix_default(roles)
    end
end

function switchAB(alch_role::Val{DeleteRole}, A, B)
    return A,A
end

function switchAB(alch_role::Val{InsertRole}, A, B)
    return B,B
end

