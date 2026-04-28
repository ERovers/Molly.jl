# Alchemical helper functions for protein design
# Project Molly v0.23.0
# ENZYME version: v"0.13.116"
# Julia version: 1.11

export
    lambda4,
    Atom_L,
    CMAPTorsion_L,
    print_interaction,
    kabsch,
    step_logger,
    lambda

function print_interaction(interaction, start_idx=1, end_idx=1)
    field_names = getfield.((interaction,), fieldnames(typeof(interaction)))
    if end_idx==1
        end_idx = length(field_names)
    end
    for (i,ft) in enumerate(zip(field_names[1:end-1]...))
        if i>start_idx && i<end_idx
            n_fields = length(ft)
            println("i:",i,"   ",[string(ft[i])*"," for i in 1:n_fields-2]..., string(ft[end]))
            println(ft[end-1])
        end
    end
end

function print_yaml(data; indent=0)
    if typeof(data) == Dict{Any,Any}
        for (key, value) in data
            println(" " ^ indent * "• ", key)
            # Recursively print nested data structures
            print_yaml(value, indent=indent+2)
        end
    else
        println(" " ^ indent * "• ", data)  # Handle non-Dict types
    end
end

function step_logger(sys, buffers, neighbors, step_n::Integer; n_threads::Integer,
                                  current_potential_energy=nothing, kwargs...)
    if isnothing(current_potential_energy)
        energy = potential_energy(sys, neighbors; n_threads=n_threads)
        println("Step "*string(step_n)*" : potential_energy = "*string(energy))
        flush(stdout)
        return energy
    else
        println("Step "*string(step_n)*" : potential_energy = "*string(current_potential_energy))
        flush(stdout)
        return current_potential_energy
    end
end

function kabsch(P,Q)
    P = ustrip(P')
    Q = ustrip(Q')
    cP = mean(P, dims=2)
    cQ = mean(Q, dims=2)
    
    P_centered = P .- cP
    Q_centered = Q .- cQ
    
    cov = P_centered * Q_centered'
    svd_res = svd(ustrip.(cov))
    Ut = transpose(svd_res.U)
    d = sign(det(svd_res.V * Ut))
    dmat = [1 0 0; 0 1 0; 0 0 d]
    rot = svd_res.V * dmat * Ut
    
    t = Q[:,1] - (rot * P)[:,1]
    return rot, t
end

@inline function lambda4(l1, l2, l3, l4)
    minλ = min(min(l1, l2), min(l3, l4))
    all_equal = (l1 == l2) & (l2 == l3) & (l3 == l4)
    return ifelse(all_equal, one(l1), minλ)
end

function lambda_mix(inter, atom_i, atom_j)
    if atom_i.λ == one(atom_i.λ)
        return atom_j.λ
    elseif atom_j.λ == one(atom_j.λ)
        return atom_i.λ
    end
    # return lorentz_λ_mixing(atom_i, atom_j)
    return atom_i.λ * atom_j.λ
end


function charmm_pdb_file(data_dir, filename, filename2)
    file_content = readlines(joinpath(data_dir,filename))
    final_lines = []
    current_res = ""
    current_idx = 0
    index = 1
    NME_flag = false
    for line in file_content
        if occursin("REMARK", line)
            push!(final_lines, line)
        elseif occursin("CRYST", line)
            push!(final_lines, line)
        elseif occursin("END", line)
            push!(final_lines, line)
        elseif occursin("Y", line[13:17])
            len = 78-length(line)
            push!(final_lines, line[1:17]*"ACE"*line[21:22]*lpad(string(1), 4)*line[27:end])
        elseif occursin("T", line[13:17])
            NME_flag = true
            len = 78-length(line)
            if parse(Int, line[23:27])==current_idx
                index += 1
                current_idx -= 1
            end
            push!(final_lines, line[1:17]*"NME"*line[21:22]*lpad(string(index), 4)*line[27:end])
        elseif occursin("TER", line)
            push!(final_lines, line[1:22]*lpad(string(index), 4))
        elseif occursin("HOH", line)
            NME_flag = false
            len = 78-length(line)
            if (parse(Int, line[23:27])!=current_idx || line[18:20]!=current_res)
                index += 1
                current_idx = parse(Int, line[23:27])
                current_res = line[18:20]
            end
            push!(final_lines, line[1:22]*lpad(string(index), 4)*line[27:end])
        else
            len = 78-length(line)
            if (parse(Int, line[23:27])!=current_idx || line[18:20]!=current_res) && !NME_flag
                index += 1
                current_idx = parse(Int, line[23:27])
                current_res = line[18:20]
            end
            if !NME_flag
                push!(final_lines, line[1:22]*lpad(string(index), 4)*line[27:end])
            else
                push!(final_lines, line[1:17]*"NME"*line[21:22]*lpad(string(index), 4)*line[27:end])
            end
        end
    end
    
    file = open(joinpath(data_dir,filename2), "w")
    for line in final_lines
        println(file, line)
    end
    close(file)
end

"""
    Atom_L(; <keyword arguments>)

Similar to Atom Type, but with λ scaling for alchemical transformations.

# Arguments
- `index::Int`: the index of the atom in the system.
- `atom_type::T`: the type of the atom.
- `mass::M=1.0u"g/mol"`: the mass of the atom.
- `charge::C=0.0`: the charge of the atom, used for electrostatic interactions.
- `σ::S=0.0u"nm"`: the Lennard-Jones finite distance at which the inter-particle
    potential is zero.
- `ϵ::E=0.0u"kJ * mol^-1"`: the Lennard-Jones depth of the potential well.
- `λ::L=1.0: the λ scaling factor (if λ is 1.0, all potentials are regular energy potentials)
"""

@kwdef struct Atom_L{T, M, C, S, E, L}
    index::Int = 1
    atom_type::T = 1
    mass::M = 1.0u"g/mol"
    charge::C = 0.0
    σ::S = 0.0u"nm"
    ϵ::E = 0.0u"kJ * mol^-1"
    σ14::S = 0.0u"nm"
    ϵ14::E = 0.0u"kJ * mol^-1"
    λ::L = 1.0
    charge_type::T = 0
    solvent::T = 0
end

lambda(atom) = atom.λ

function Base.show(io::IO, a::Atom_L)
    print(io, "Atom with index=", a.index, ", atom_type=", a.atom_type, ", mass=", mass(a),
          ", charge=", charge(a), ", σ=", a.σ, ", ϵ=", a.ϵ, ", λ=", a.λ, ", charge_type=", a.charge_type, ", solvent=", a.solvent)
end

function dict_get(dic, key, atom::Atom_L, at_data, default)
    if haskey(dic, key)
        if at_data.res_name!="ALC"
            return dic[key][at_data.res_id]
        else
            return default
        end
    else
        return default
    end
end

function inject_atom(at::Atom_L, at_data, params_dic)
    key_prefix = "residue_" * string(at_data.res_number) * "_λ"
    return Atom_L(
                at.index,
                at.atom_type,
                at.mass,
                at.charge, 
                at.σ,
                at.ϵ,
                at.σ14,
                at.ϵ14,
                dict_get(params_dic, key_prefix, at, at_data, at.λ),
                at.charge_type,
                at.solvent,
            )
end

"""
    AtomData_L(; atom_type="?", atom_name="?", res_number=1, res_name="???", res_id=1,
             chain_id="A", element="?", hetero_atom=false)

Data associated with an atom.

Storing this separately allows the [`Atom`](@ref) types to be bits types and hence
work on the GPU.
"""
@kwdef struct AtomData_L
    atom_type::String = "?"
    atom_name::String = "?"
    res_number::Int = 1
    res_name::String = "???"
    res_id::Int = 1
    chain_id::String = "A"
    element::String = "?"
    hetero_atom::Bool = false
end

function BioStructures.AtomRecord(at_data::AtomData_L, i, coord)
    return BioStructures.AtomRecord(
        at_data.hetero_atom, i, at_data.atom_name, ' ', at_data.res_name,
        at_data.chain_id, at_data.res_number, ' ', coord, 1.0, 0.0,
        at_data.element == "?" ? "  " : at_data.element, "  "
    )
end

# Rotamer search
"""
    rotate_side_chain(coords, p1_idx, p2_idx, moving_indices, theta)

Rotates a set of atoms in `coords` around an axis defined by atoms at `p1_idx` and `p2_idx`.
- `coords`: Matrix of size (3, N) or (N, 3). This code assumes (N, 3).
- `theta`: Angle in radians.
"""
function rotate_side_chain(coords::Vector{Any}, p1_idx::Int, p2_idx::Int, moving_indices::Vector{Any}, theta::Float64)
    # 1. Define the rotation axis
    p1 = coords[p1_idx, :]
    p2 = coords[p2_idx, :]
    axis = p2 - p1
    axis /= norm(axis)
    
    # Pre-calculate rotation components
    cos_t = cos(theta)
    sin_t = sin(theta)
    one_minus_cos = 1.0 - cos_t
    
    # 2. Process each moving atom
    for i in moving_indices
        # Translate atom so p1 is at origin
        v = coords[i, :] - p1
        
        # 3. Rodrigues' Rotation Formula
        v_rot = v * cos_t + 
                [cross(axis[1], v[1])] * sin_t + 
                axis * dot(axis, v) * one_minus_cos
        
        # 4. Translate back and update
        coords[i, :] = v_rot + p1
    end
    
    return coords
end

"""
    count_clashes(moving_coords, static_coords; cutoff=2.0)

Returns the number of atom pairs that are closer than the `cutoff` distance.
`moving_coords`: Matrix (M, 3) of the side chain being tested.
`static_coords`: Matrix (N, 3) of the rest of the protein.
"""
function count_clashes(moving_coords::Vector{Any}, static_coords; cutoff::Float64=0.15)
    clash_count = 0
    
    # Iterate through each atom in the side chain
    for i in 1:size(moving_coords, 1)
        m_atom = moving_coords[i, :]
        
        # Compare against every 'static' atom in the environment
        for j in 1:size(static_coords, 1)
            s_atom = static_coords[j, :]
            
            # Squared distance is faster (avoids sqrt)
            dist_sq = sqrt(sum((m_atom .- s_atom)[1].^2))
            
            if dist_sq < cutoff
                clash_count += 1
            end
        end
    end
    return clash_count
end

# Shortcuts
function lj_λ_less_one_shortcut(atom_i, atom_j)
    return iszero_value(atom_i.ϵ) || iszero_value(atom_j.ϵ) ||
           iszero_value(atom_i.σ) || iszero_value(atom_j.σ) ||
           (atom_i.λ<1) || (atom_j.λ<1)
end

function lj_λ_one_shortcut(atom_i, atom_j)
    return ( iszero_value(atom_i.ϵ) || iszero_value(atom_j.ϵ) ||
           iszero_value(atom_i.σ) || iszero_value(atom_j.σ) ) ||
           ( (atom_i.λ==1) && (atom_j.λ==1) )
end

function coul_zero_shortcut(atom_i, atom_j)
    return iszero_value(atom_i.charge) || iszero_value(atom_j.charge)
end

function coul_λ_less_one_shortcut(atom_i, atom_j)
    return iszero_value(atom_i.charge) || iszero_value(atom_j.charge) ||
           (atom_i.λ<1) || (atom_j.λ<1)
end

function coul_λ_one_shortcut(atom_i, atom_j)
    return ( iszero_value(atom_i.charge) || iszero_value(atom_j.charge) ) ||
           ( (atom_i.λ==1) && (atom_j.λ==1) )
end

# Torsion functions
@inline function force(d::PeriodicTorsion, coords_i, coords_j, coords_k,
                       coords_l, boundary, atom_i::Atom_L, atom_j::Atom_L, 
                       atom_k::Atom_L, atom_l::Atom_L, args...)
    ab, bc, cd, cross_ab_bc, cross_bc_cd, bc_norm, θ = periodic_torsion_vectors(
                                        coords_i, coords_j, coords_k, coords_l, boundary)
    fs = sum(zip(d.periodicities, d.phases, d.ks)) do (periodicity, phase, k)
        fi, fj, fk, fl = periodic_torsion_force(periodicity, phase, k, ab, bc, cd, cross_ab_bc,
                                                cross_bc_cd, bc_norm, θ)
        
        if atom_i.λ<one(atom_i.λ) && atom_j.λ<one(atom_j.λ) && atom_k.λ<one(atom_k.λ) && atom_l.λ<one(atom_l.λ)
            l = one(atom_i.λ)
        else
            l = minimum((atom_i.λ,atom_j.λ,atom_k.λ,atom_l.λ))
        end
    
        return SpecificForce4Atoms(l*fi, l*fj, l*fk, l*fl)
    end
    return fs
end

@inline function force_gpu(d::PeriodicTorsion{N}, coords_i, coords_j, coords_k,
                           coords_l, boundary, atom_i::Atom_L, atom_j::Atom_L, atom_k::Atom_L, 
                           atom_l::Atom_L, args...) where N
    ab, bc, cd, cross_ab_bc, cross_bc_cd, bc_norm, θ = periodic_torsion_vectors(
                                        coords_i, coords_j, coords_k, coords_l, boundary)
    fi_sum, fj_sum, fk_sum, fl_sum = periodic_torsion_force(d.periodicities[1], d.phases[1],
                                        d.ks[1], ab, bc, cd, cross_ab_bc, cross_bc_cd, bc_norm, θ)
    
    if atom_i.λ<one(atom_i.λ) && atom_j.λ<one(atom_j.λ) && atom_k.λ<one(atom_k.λ) && atom_l.λ<one(atom_l.λ)
        l = one(atom_i.λ)
    else
        l = minimum((atom_i.λ,atom_j.λ,atom_k.λ,atom_l.λ))
    end
    
    for i in 2:N
        fi, fj, fk, fl = periodic_torsion_force(d.periodicities[i], d.phases[i], d.ks[i], ab, bc,
                                                cd, cross_ab_bc, cross_bc_cd, bc_norm, θ)
        fi_sum += fi
        fj_sum += fj
        fk_sum += fk
        fl_sum += fl
    end
    return SpecificForce4Atoms(l*fi_sum,l*fj_sum, l*fk_sum, l*fl_sum)
end

@inline function potential_energy(d::PeriodicTorsion{N}, coords_i, coords_j, coords_k,
                                  coords_l, boundary, atom_i::Atom_L, atom_j::Atom_L,
                                  atom_k::Atom_L, atom_l::Atom_L, args...) where N
    θ = torsion_angle(coords_i, coords_j, coords_k, coords_l, boundary)
    k1 = d.ks[1]
    E = k1 + k1 * cos((d.periodicities[1] * θ) - d.phases[1])

    if atom_i.λ<one(atom_i.λ) && atom_j.λ<one(atom_j.λ) && atom_k.λ<one(atom_k.λ) && atom_l.λ<one(atom_l.λ)
        l = one(atom_i.λ)
    else
        l = minimum((atom_i.λ,atom_j.λ,atom_k.λ,atom_l.λ))
    end
    
    for i in 2:N
        k = d.ks[i]
        E += k + k * cos((d.periodicities[i] * θ) - d.phases[i])
    end
    return l * E
end

# CMAPtorsion
@kwdef struct CMAPTorsion_L{I,L}
    index::I
    size::I
    λ::L
    res_num::I
    res_id::I
end

Base.zero(::CMAPTorsion_L) = CMAPTorsion_L(index=0, size=0, λ=0.0, res_num=0, res_id=0)

function dict_get(dic, key, inter::CMAPTorsion_L, default)
    if haskey(dic, key)
        return dic[key][inter.res_id]
    else
        return default
    end
end

function inject_interaction(inter::CMAPTorsion_L{I,L}, inter_type, params_dic) where {I,L}
    key_prefix = "residue_" * string(inter.res_num) * "_λ"
    return CMAPTorsion_L{I,L}(
        inter.index,
        inter.size,
        dict_get(params_dic, key_prefix, inter, inter.λ),
        inter.res_num,
        inter.res_id,
    )
end

@inline function force(inter::CMAPTorsion_L, coords_i, coords_j, coords_k, coords_l, 
                            coords_m, boundary, atoms_i, atoms_j, atoms_k, atoms_l, 
                            atoms_m, force_units, velocities_i, velocities_j,
                            velocities_k, velocities_l, velocities_m, step_n, data)
    # First angle
    v0a = vector(coords_j, coords_i, boundary)
    v1a = vector(coords_j, coords_k, boundary)
    v2a = vector(coords_l, coords_k, boundary)
    cp0a = cross(v0a, v1a)
    cp1a = cross(v1a, v2a)
    cosangle = dot(cp0a/norm(cp0a), cp1a/norm(cp1a))
    F = typeof(cosangle)
    if cosangle > F(0.99) || cosangle < F(-0.99)
        cross_prod = cp0a × cp1a
        scale = dot(cp0a,cp0a) * dot(cp1a,cp1a)
        angleA = asin(sqrt(dot(cross_prod,cross_prod)/scale))
        if cosangle < F(0.0)
            angleA = pi - angleA
        end
    else
        angleA = acos(cosangle)
    end
    angleA = (ustrip(dot(v0a,cp1a))>=0) ? angleA : -angleA
    angleA = mod(angleA + F(2*pi), F(2*pi))
    
    # Second angle
    v0b = vector(coords_k, coords_j, boundary)
    v1b = vector(coords_k, coords_l, boundary)
    v2b = vector(coords_m, coords_l, boundary)
    cp0b = cross(v0b, v1b)
    cp1b = cross(v1b, v2b)
    cosangle = dot(cp0b/norm(cp0b), cp1b/norm(cp1b))
    if cosangle > F(0.99) || cosangle < F(-0.99)
        cross_prod = cross(cp0b, cp1b)
        scale = dot(cp0b,cp0b) * dot(cp1b,cp1b)
        angleB = asin(sqrt(dot(cross_prod,cross_prod)/scale))
        if cosangle < F(0.0)
            angleB = pi - angleB
        end
    else
        angleB = acos(ustrip(cosangle))
    end
    angleB = (ustrip(dot(v0b,cp1b))>=0) ? angleB : -angleB
    angleB = mod(angleB + F(2*pi), F(2*pi))

    # Identify Patch
    delta = F(2*pi) / inter.size
    s = Int(trunc(min(angleA/delta, inter.size-1)))
    t = Int(trunc(min(angleB/delta, inter.size-1)))
    idx = inter.index+(4*(s+inter.size*t))+1
    # c0 = @view data[idx,:]
    # c1 = @view data[idx+1,:]
    # c2 = @view data[idx+2,:]
    # c3 = @view data[idx+3,:]
    da = angleA/delta - s
    db = angleB/delta - t

     # Evaluate the spline to determine the energy and gradients.
    dEdA = (3*data[idx+3,4]*da + 2*data[idx+2,4])*da + data[idx+1,4]
    dEdB = (3*data[idx+3,4]*db + 2*data[idx+3,3])*db + data[idx+3,2]
    dEdA = db*dEdA + (3*data[idx+3,3]*da + 2*data[idx+2,3])*da + data[idx+1,3]
    dEdB = da*dEdB + (3*data[idx+2,4]*db + 2*data[idx+2,3])*db + data[idx+2,2]
    dEdA = db*dEdA + (3*data[idx+3,2]*da + 2*data[idx+2,2])*da + data[idx+1,2]
    dEdB = da*dEdB + (3*data[idx+1,4]*db + 2*data[idx+1,3])*db + data[idx+1,2]
    dEdA = db*dEdA + (3*data[idx+3,1]*da + 2*data[idx+2,1])*da + data[idx+1,1]
    dEdB = da*dEdB + (3*data[idx,4]*db + 2*data[idx,3])*db + data[idx,2]
    dEdA /= delta
    dEdB /= delta

    # Calculate the force to the first torsion.
    normCross1 = dot(cp0a, cp0a)
    normSqrBC = dot(v1a, v1a)
    normBC = sqrt(normSqrBC)
    normCross2 = dot(cp1a, cp1a)
    dp = 1/normSqrBC
    ff = ((-dEdA*normBC)/normCross1, dot(v0a, v1a)*dp, dot(v2a, v1a)*dp, (dEdA*normBC)/normCross2)
    force1 = ff[1]*cp0a
    force4 = ff[4]*cp1a
    d = ff[2]*force1 - ff[3]*force4
    force2 = d-force1
    force3 = -d-force4

    # Calculate the force to the second torsion.
    normCross1 = dot(cp0b, cp0b)
    normSqrBC = dot(v1b, v1b)
    normBC = sqrt(normSqrBC)
    normCross2 = dot(cp1b, cp1b)
    dp = 1/normSqrBC
    ff = ((-dEdB*normBC)/normCross1, dot(v0b, v1b)*dp, dot(v2b, v1b)*dp, (dEdB*normBC)/normCross2)
    force5 = ff[1]*cp0b
    force8 = ff[4]*cp1b
    d = ff[2]*force5 - ff[3]*force8
    force6 = d-force5
    force7 = -d-force8

    # Apply the forces to the atoms
    fi = force1
    fj = force2 + force5
    fk = force3 + force6
    fl = force4 + force7
    fm =          force8
    return SpecificForce5Atoms(inter.λ*fi, inter.λ*fj, inter.λ*fk, inter.λ*fl, inter.λ*fm)
end

@inline function potential_energy(inter::CMAPTorsion_L,coords_i, coords_j, coords_k, coords_l, 
                            coords_m, boundary, atoms_i::Atom_L, atoms_j::Atom_L, atoms_k::Atom_L, atoms_l::Atom_L, 
                            atoms_m::Atom_L, energy_units, velocities_i, velocities_j, velocities_k, velocities_l, 
                            velocities_m, step_n, data)
    # First angle
    v0a = vector(coords_j, coords_i, boundary)
    v1a = vector(coords_j, coords_k, boundary)
    v2a = vector(coords_l, coords_k, boundary)
    cp0a = ustrip.(cross(v0a, v1a))
    cp1a = ustrip.(cross(v1a, v2a))
    cosangle = dot(cp0a/norm(cp0a), cp1a/norm(cp1a))
    F = typeof(cosangle)
    if cosangle > F(0.99) || cosangle < F(-0.99)
        cross_prod = cp0a × cp1a
        scale = dot(cp0a,cp0a) * dot(cp1a,cp1a)
        angleA = asin(sqrt(dot(cross_prod,cross_prod)/scale))
        if cosangle < F(0.0)
            angleA = pi - angleA
        end
    else
        angleA = acos(cosangle)
    end
    angleA = (ustrip(dot(v0a,cp1a))>=0) ? angleA : -angleA
    angleA = mod(angleA + F(2*pi), F(2*pi))
    
    # Second angle
    v0b = vector(coords_k, coords_j, boundary)
    v1b = vector(coords_k, coords_l, boundary)
    v2b = vector(coords_m, coords_l, boundary)
    cp0b = ustrip.(cross(v0b, v1b))
    cp1b = ustrip.(cross(v1b, v2b))
    cosangle = dot(cp0b/norm(cp0b), cp1b/norm(cp1b))
    if cosangle > F(0.99) || cosangle < F(-0.99)
        cross_prod = cross(cp0b, cp1b)
        scale = dot(cp0b,cp0b) * dot(cp1b,cp1b)
        angleB = asin(sqrt(dot(cross_prod,cross_prod)/scale))
        if cosangle < F(0.0)
            angleB = pi - angleB
        end
    else
        angleB = acos(ustrip(cosangle))
    end
    angleB = (ustrip(dot(v0b,cp1b))>=0) ? angleB : -angleB
    angleB = mod(angleB + F(2*pi), F(2*pi))

    # Identify Patch
    delta = F(2*pi) / inter.size
    s = Int(trunc(min(angleA/delta, inter.size-1)))
    t = Int(trunc(min(angleB/delta, inter.size-1)))
    idx = inter.index+(4*(s+inter.size*t))+1
    # c0 = @view data[idx,:]
    # c1 = @view data[idx+1,:]
    # c2 = @view data[idx+2,:]
    # c3 = @view data[idx+3,:]
    da = angleA/delta - s
    db = angleB/delta - t

    # Spline with coefficients
    energy = ((data[idx+3,4]*db + data[idx+3,3])*db + data[idx+3,2])*db + data[idx+3,1]
    energy = da*energy + ((data[idx+2,4]*db + data[idx+2,3])*db + data[idx+2,2])*db + data[idx+2,1]
    energy = da*energy + ((data[idx+1,4]*db + data[idx+1,3])*db + data[idx+1,2])*db + data[idx+1,1]
    energy = da*energy + ((data[idx,4]*db + data[idx,3])*db + data[idx,2])*db + data[idx,1]
    return inter.λ*energy
end

# Energy
function myfindneighbors(atoms, coords, boundary, neighborfinder)
     sys = System(
        atoms=atoms,
        coords=coords,
        boundary=boundary,
        pairwise_inters=(),
        specific_inter_lists=(),
        general_inters=(),
        neighbor_finder=neighborfinder,
        force_units=NoUnits,
        energy_units=NoUnits,
    )
    return find_neighbors(sys)
end

@inline function potential_energy(coords, boundary, neighbors, n_neighbors, velocities, energy_units,
                            atoms, b_g, a_g, p_g, i_g, c_g, cl_g, 
                            lj_g, lg_g, cg_g, pme_g)
    T = typeof(ustrip(zero(eltype(eltype(coords)))))
    pe = zero(T) * energy_units

    # Pairwise interactions
    for ni in 1:n_neighbors
        i, j, special = neighbors[ni]
        dr = vector(coords[i], coords[j], boundary)
        pe += potential_energy(lj_g, dr, atoms[i], atoms[j], energy_units, special,
                        coords[i], coords[j], boundary, velocities[i], velocities[j], 0)
        pe += potential_energy(lg_g, dr, atoms[i], atoms[j], energy_units, special,
                        coords[i], coords[j], boundary, velocities[i], velocities[j], 0)
        pe += potential_energy(cg_g, dr, atoms[i], atoms[j], energy_units, special,
                        coords[i], coords[j], boundary, velocities[i], velocities[j], 0)
    end

    # PME
    pe += potential_energy(pme_g, atoms, coords, boundary, NoUnits)

    # Bonds
    for (i, j, inter) in zip(b_g.is, b_g.js, b_g.inters)
        pe +=  potential_energy(inter, coords[i], coords[j], boundary, atoms[i], atoms[j],
                              energy_units, velocities[i], velocities[j], 0, b_g.data)
    end

    # Angles
    for (i, j, k, inter) in zip(a_g.is, a_g.js, a_g.ks, a_g.inters)
        pe += potential_energy(inter, coords[i], coords[j], coords[k], boundary, atoms[i],
                              atoms[j], atoms[k], energy_units, velocities[i], velocities[j],
                              velocities[k], 0, a_g.data)
    end

    # Periodic Torsions
    for (i, j, k, l, inter) in zip(p_g.is, p_g.js, p_g.ks, p_g.ls,
                                   p_g.inters)
        pe += potential_energy(inter, coords[i], coords[j], coords[k], coords[l], boundary,
                              atoms[i], atoms[j], atoms[k], atoms[l], energy_units,
                              velocities[i], velocities[j], velocities[k], velocities[l],
                              0, p_g.data)
    end

    # Custom Torsions
    for (i, j, k, l, inter) in zip(i_g.is, i_g.js, i_g.ks, i_g.ls,
                                   i_g.inters)
        pe += potential_energy(inter, coords[i], coords[j], coords[k], coords[l], boundary,
                              atoms[i], atoms[j], atoms[k], atoms[l], energy_units,
                              velocities[i], velocities[j], velocities[k], velocities[l],
                              0, i_g.data)
    end

    # CMAP Torsions
    for (i, j, k, l, m, inter) in zip(c_g.is, c_g.js, c_g.ks, c_g.ls,
                                   c_g.ms, c_g.inters)
        pe += potential_energy(inter, coords[i], coords[j], coords[k], coords[l], coords[m], 
                              boundary, atoms[i], atoms[j], atoms[k], atoms[l], atoms[m], energy_units,
                              velocities[i], velocities[j], velocities[k], velocities[l], velocities[m],
                              0, c_g.data)
    end

    # CMAP Torsions lambda
    for (i, j, k, l, m, inter) in zip(cl_g.is, cl_g.js, cl_g.ks, cl_g.ls,
                                   cl_g.ms, cl_g.inters)
        pe += potential_energy(inter, coords[i], coords[j], coords[k], coords[l], coords[m], 
                              boundary, atoms[i], atoms[j], atoms[k], atoms[l], atoms[m], energy_units,
                              velocities[i], velocities[j], velocities[k], velocities[l], velocities[m],
                              0, cl_g.data)
    end
    
    return pe
end

# Interactions
function merge(interactions)
    interactions_final = []
    cache = []
    for (i,inter) in enumerate(interactions)
        i in cache && continue
        idx = findall(x->typeof(x)==typeof(inter), interactions)
        inters = interactions[idx[1]]
        for j in idx[2:end]
            inters = append!(inters,interactions[j])
        end
        push!(interactions_final, inters)
        append!(cache,idx)
    end
    return interactions_final
end

function Base.append!(il1::InteractionList1Atoms{I, T, D}, il2::InteractionList1Atoms{I, T, D}) where {I, T, D}
    return InteractionList1Atoms{I, T, D}(
        append!(il1.is,il2.is),
        append!(il1.inters,il2.inters),
        append!(il1.types,il2.types),
        nothing,
    )
end

function Base.append!(il1::InteractionList2Atoms{I, T, D}, il2::InteractionList2Atoms{I, T, D}) where {I, T, D}
    return InteractionList2Atoms{I, T, D}(
        append!(il1.is,il2.is),
        append!(il1.js,il2.js),
        append!(il1.inters,il2.inters),
        append!(il1.types,il2.types),
        nothing
    )
end

function Base.append!(il1::InteractionList3Atoms{I, T, D}, il2::InteractionList3Atoms{I, T, D}) where {I, T, D}
    return InteractionList3Atoms{I, T, D}(
        append!(il1.is,il2.is),
        append!(il1.js,il2.js),
        append!(il1.ks,il2.ks),
        append!(il1.inters,il2.inters),
        append!(il1.types,il2.types),
        nothing
    )
end

function Base.append!(il1::InteractionList4Atoms{I, T, D}, il2::InteractionList4Atoms{I, T, D}) where {I, T, D}
    return InteractionList4Atoms{I, T, D}(
        append!(il1.is,il2.is),
        append!(il1.js,il2.js),
        append!(il1.ks,il2.ks),
        append!(il1.ls,il2.ls),
        append!(il1.inters,il2.inters),
        append!(il1.types,il2.types),
        nothing
    )
end

function Base.append!(il1::InteractionList5Atoms{I, T, D}, il2::InteractionList5Atoms{I, T, D}) where {I, T, D}
    tmp_inters = il1.inters
    cmaptorsion = typeof(il1.inters[1])
    matrix = typeof(il1.data)
    for inter in il2.inters
        push!(tmp_inters, cmaptorsion((4*tmp_inters[end].size*tmp_inters[end].size)+tmp_inters[end].index, inter.size, inter.λ, inter.res_num, inter.res_id))
    end
    # println(vcat(il1.data,il2.data))
    return InteractionList5Atoms{I, T, D}(
        append!(il1.is,il2.is),
        append!(il1.js,il2.js),
        append!(il1.ks,il2.ks),
        append!(il1.ls,il2.ls),
        append!(il1.ms,il2.ms),
        tmp_inters,
        append!(il1.types,il2.types),
        vcat(il1.data,il2.data),
    )
end

function to_device(il1::InteractionList1Atoms{I, T, D}, ::Type{AT}) where {I, T, D, AT}
    return InteractionList1Atoms(
        to_device(il1.is, AT),
        to_device(il1.inters,AT),
        il1.types,
        nothing,
    )
end

function to_device(il1::InteractionList2Atoms{I, T, D}, ::Type{AT}) where {I, T, D, AT}
    return InteractionList2Atoms(
        to_device(il1.is, AT),
        to_device(il1.js, AT),
        to_device(il1.inters,AT),
        il1.types,
        nothing,
    )
end

function to_device(il1::InteractionList3Atoms{I, T, D}, ::Type{AT}) where {I, T, D, AT}
    return InteractionList3Atoms(
        to_device(il1.is, AT),
        to_device(il1.js, AT),
        to_device(il1.ks, AT),
        to_device(il1.inters,AT),
        il1.types,
        nothing,
    )
end

function to_device(il1::InteractionList4Atoms{I, T, D}, ::Type{AT}) where {I, T, D, AT}
    return InteractionList4Atoms(
        to_device(il1.is, AT),
        to_device(il1.js, AT),
        to_device(il1.ks, AT),
        to_device(il1.ls, AT),
        to_device(il1.inters,AT),
        il1.types,
        nothing,
    )
end

function to_device(il1::InteractionList5Atoms{I, T, D}, ::Type{AT}) where {I, T, D, AT}
    return InteractionList5Atoms(
        to_device(il1.is, AT),
        to_device(il1.js, AT),
        to_device(il1.ks, AT),
        to_device(il1.ls, AT),
        to_device(il1.ms, AT),
        to_device(il1.inters,AT),
        il1.types,
        to_device(il1.data,AT),
    )
end