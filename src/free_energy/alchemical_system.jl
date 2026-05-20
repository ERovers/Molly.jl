export
    Hybrid_system

const aminos_dic = Dict("ALA" => 1, "ARG" => 2, "ASN" => 3,
                        "ASP" => 4, "CYS" => 5, "GLN" => 6,
                        "GLU" => 7, "GLY" => 8, "HIS" => 9,
                        "ILE" => 10, "LEU" => 11, "LYS" => 12, 
                        "MET" => 13, "PHE" => 14, "SER" => 15,
                        "THR" => 16, "TRP" => 17, "TYR" => 18,
                        "VAL" => 19, "ACE" => 0, "ALC" => 0,
                        "NME" => 0, "HOH" => 0, "MG" => 0, 
                        "CL" => 0, "NA" => 0)

const rotamer_dic = Dict("ARG" => [62,-177,-67,-62], "LYS" => [62,-177,-90,-67,-62],
                            "MET" => [62,-177,-67,-65], "GLU" => [62, -177,-67,-65],
                            "GLN" => [62,-70,-177,-65,-67], "ASP" => [62, -177,-70],
                            "ASN" => [62,-174,-177,-65], "ILE" => [62, -177,-65,-57],
                            "LEU" => [62,-177,-172,-85,-65], "HIS" => [62, -177,-65],
                            "TRP" => [62,-177,-65], "TYR" => [62, -177,-65],
                            "PHE" => [62,-177,-65], "THR" => [62, -175,-65],
                            "VAL" => [63,175,-60], "SER" => [62, -177,-65],
                            "CYS" => [62,-177,-65])

const interaction_mapping = Dict(HarmonicBond => HarmonicBondλ,
                                HarmonicAngle => HarmonicAngleλ,
                                PeriodicTorsion => PeriodicTorsionλ)

function to_lambda_param(inter::HarmonicBond)
    return HarmonicBondλ(k=(inter.k, inter.k), r0=(inter.r0, inter.r0))
end

function to_lambda_param(inter::HarmonicAngle)
    return HarmonicAngleλ(k=(inter.k, inter.k), θ0=(inter.θ0, inter.θ0))
end

function to_lambda_param(inter::PeriodicTorsion)
    return PeriodicTorsionλ(
        periodicities=(inter.periodicities, inter.periodicities),
        phases       =(inter.phases,        inter.phases),
        ks           =(inter.ks,            inter.ks),
        proper       =inter.proper,
    )
end

function merge_core_params(existing::HarmonicBondλ, b::HarmonicBond)
    return HarmonicBondλ(k=(existing.k[1], b.k), r0=(existing.r0[1], b.r0))
end

function merge_core_params(existing::HarmonicAngleλ, b::HarmonicAngle)
    return HarmonicAngleλ(k=(existing.k[1], b.k), θ0=(existing.θ0[1], b.θ0))
end

function merge_core_params(existing::PeriodicTorsionλ, b::PeriodicTorsion)
    return PeriodicTorsionλ(
        periodicities=(existing.periodicities[1], b.periodicities),
        phases       =(existing.phases[1],        b.phases),
        ks           =(existing.ks[1],            b.ks),
        proper       =existing.proper,
    )
end

function merge_core_params(existing, _)
    return existing
end

function find_and_merge_core_interaction!(Interactions, IT, mapped, b_param, b_type)
    for inter_list in Interactions
        if typeof(inter_list).name.wrapper !== IT
            continue
        end

        names = fieldnames(typeof(inter_list))
        n_indices = length(mapped)
        index_fields = names[1:n_indices]
        inters_field = names[n_indices + 1]
        types_field = names[n_indices + 2]

        index_arrays = getfield.((inter_list,), index_fields)
        types_arr = getfield(inter_list, types_field)
        inters_arr = getfield(inter_list, inters_field)

        for (i,key) in enumerate(zip(index_arrays...))
            if all(x->x in key, mapped)
                inters_arr[i] = merge_core_params(inters_arr[i], b_param)
                return true
            end
        end
    end
    return false
end

"""
    Hybrid_system(T, AT, ff, sys, res_num, aminos)

Mutates residue in a system to (multiple) λ-residues and returns a hybrid system of
original system and λ-mediated atoms.

# Arguments
- `T`: Float type used, often Float32
- `AT`: Array type, whether to run on CPU (Array) or GPU (CuArray)
- `ff`: MolecularForceField struct that is used to parametrize the system.
- `sys`: The system in which a residue is going to be mutated.
- `params_dic`: A dictionary of the substitutions and the λ-values
- `temp = T(298.0)u"K"`: Temperature to generate random velocities for the system, standard is 298K.
"""


function Hybrid_system(T, AT, ff, sys, traj_file, params_dic; direction=true, temp = T(298.0)u"K", units=true, aminos_d=aminos_dic)
    # initialize data groups for new system
    Atoms = []
    Data = []
    Coords = []
    Interactions = []
    Boundary = deepcopy(sys.boundary)
    aminos = Dict()
    for (res, res_vec) in params_dic
        res_n = parse(Int, match(r"\d+", res).match)
        for (i,l) in enumerate(res_vec)
            res_type = [key for (key,value) in aminos_d if value==i][1]
            if haskey(aminos, res_n)
                aminos[res_n][res_type] = l
            else
                aminos[res_n] = Dict(res_type => l)
            end
        end
    end
    res_num = sort(collect(keys(aminos)))
    
    # Remove all the atoms of selected residue except backbone and set λ to 1.0
    # Make map of backbone to map interactions on later for added side-chains
    map = Dict()
    residue = Dict()
    CAs = []
    for r in res_num
        tmp_dic = units ? Dict("backbone_map" => Dict(), "backbone_coords" => zeros(T,3,3)u"nm") : Dict("backbone_map" => Dict(), "backbone_coords" => zeros(T,3,3))
        residue[r] = tmp_dic
    end
    backbone_idx = Dict("HA"=>1,"CA"=>2,"CB"=>3, "HA2"=>1, "HA3"=>3)
    count = 1
    for (i,(a,d,c)) in enumerate(zip(sys.atoms, sys.atoms_data, sys.coords))
        if (d.res_number in res_num) && d.atom_name in ["HA","CA","CB","HA2","HA3"]
            residue[d.res_number]["backbone_coords"][backbone_idx[d.atom_name],:] = c
        end
        if any((r-d.res_number)==1 for r in res_num) && d.atom_name=="C"
            residue[d.res_number+1]["backbone_map"][d.atom_name*"*"] = count
        elseif any((d.res_number-r)==1 for r in res_num) && d.atom_name=="N"
            residue[d.res_number-1]["backbone_map"][d.atom_name*"*"] = count
        end
        if !(d.res_number in res_num)
            push!(Atoms, Atom(index=count, atom_type=a.atom_type, mass=a.mass, charge=a.charge, σ=a.σ, ϵ=a.ϵ, 
                    λ=T(1.0), alch_role=CoreRole))
            push!(Data, AtomData(atom_type=d.atom_type, atom_name=d.atom_name, res_number=d.res_number,
                                        res_name=d.res_name, chain_id=d.chain_id, element=d.element, hetero_atom=d.hetero_atom))
            push!(Coords, c)
            map[i] = count
            count += 1
        elseif (d.res_number in res_num) && d.atom_name in ["N","CA","C","O","H","HA","HA2"]
            push!(Atoms, Atom(index=count, mass=a.mass, charge=a.charge, σ=a.σ, ϵ=a.ϵ, 
                    λ=T(1.0)))
            push!(Data, AtomData(atom_type=d.atom_type, atom_name=d.atom_name, res_number=d.res_number,
                                        res_name="ALC", chain_id=d.chain_id, element=d.element, hetero_atom=d.hetero_atom))
            push!(Coords, c)
            map[i] = count
            residue[d.res_number]["backbone_map"][d.atom_name] = count
            if d.atom_name=="HA"
                residue[d.res_number]["backbone_map"]["HA2"] = count
            elseif d.atom_name=="HA2"
                residue[d.res_number]["backbone_map"]["HA"] = count
            elseif d.atom_name=="CA"
                push!(CAs, count)
            end
            count += 1
        end
    end
    
    # Add all the in the interactions except the ones part of the side-chain that has been removed
    for interaction in sys.specific_inter_lists
        IT = typeof(interaction).name.wrapper
        field_names = getfield.((interaction,), fieldnames(typeof(interaction)))
        tmp = [typeof(fn)() for fn in field_names[1:end-1]]
        name = ""
        if typeof(field_names[end-2][1]).name.name==:CMAPTorsion
            index = 0
            maps = []
            CMAP = typeof(field_names[end-2][1])
            for field_tuple in zip(field_names[1:end-1]...)
                n_fields = length(field_tuple)
                if !any(field_tuple[i] in CAs for i in 1:n_fields-2)
                    for i in 1:n_fields-2
                        tmp[i] = push!(tmp[i], map[field_tuple[i]])
                    end
                    tmp[n_fields-1] = push!(tmp[n_fields-1], CMAPTorsion(index, field_tuple[end-1].size))
                    at_data = Data[field_tuple[3]]
                    tmp[n_fields] = push!(tmp[n_fields], field_tuple[end])
                    push!(maps, interaction.data[index+1:index+(4*field_tuple[end-1].size*field_tuple[end-1].size), :])
                    index += 4*field_tuple[end-1].size*field_tuple[end-1].size
                end
            end
            maps = vcat(maps...)
            if length(tmp[1])>0
                Interactions = push!(Interactions, IT(tmp..., maps))
            end
        else
            for field_tuple in zip(field_names[1:end-1]...)
                n_fields = length(field_tuple)
                name = typeof(field_tuple[end-1]).name.name
                if all(haskey(map, field_tuple[i]) for i in 1:n_fields-2)
                    for i in 1:n_fields-2
                        tmp[i] = push!(tmp[i], map[field_tuple[i]])
                    end
                    
                    tmp[n_fields-1] = push!(tmp[n_fields-1], field_tuple[end-1])
                    tmp[n_fields] = push!(tmp[n_fields], field_tuple[end])
                end
            end
            Interactions = push!(Interactions, IT(tmp..., nothing))
        end
    end

    # Add new side-chains for selected residue
    map_AA = Dict()
    addition_groups = Dict()
    rotamer_search = Dict()
    rotamer_search["N_idx"] = zeros(Int64,maximum(res_num))
    rotamer_search["CA_idx"] = zeros(Int64,maximum(res_num))
    
    for r in res_num
        rotamer_search[r] = Dict()
        rotamer_search["N_idx"][r] = residue[r]["backbone_map"]["N"]
        rotamer_search["CA_idx"][r] = residue[r]["backbone_map"]["CA"]
        for AA in keys(aminos[r])
            rotamer_search[r][AA] = Dict()
            # Load residue system
            amino_dir = normpath(@__DIR__, "../..", "data/aminoacids")
            tmp_sys = System(amino_dir*"/"*AA*".pdb", ff;
                     nonbonded_method=:none, center_coords=false, units=units, array_type=AT)
            
            # Superimpose residue onto backbone
            backbone_coords2 = units ? zeros(T,3,3)u"nm" : zeros(T,3,3)
            for (d,c) in zip(tmp_sys.atoms_data, tmp_sys.coords)
                if d.atom_name in ["HA","CA","CB","HA2","HA3"] && (d.res_name!="TLA") 
                    backbone_coords2[backbone_idx[d.atom_name],:] = c
                end
            end
            rot, t = kabsch_AA(backbone_coords2, residue[r]["backbone_coords"])
            t = units ? (t)u"nm" : t
            coords2 = ((rot * hcat(tmp_sys.coords...)) .+ t)'
            coords2 = [SVector{length(c)}(c) for c in eachrow(coords2)]

            # Create map for interactions and add atoms
            sidechain_A = []
            count = isempty(map_AA) ? maximum(values(map))+1 : maximum(values(map_AA))+1
            map_AA = Dict()
            rotamer_search[r][AA]["index"] = []
            CMAP_idx = 0
            for (i,(a,d,c)) in enumerate(zip(tmp_sys.atoms, tmp_sys.atoms_data, coords2))
                if d.res_name=="TLA" && d.res_number==1 && d.atom_name=="C"
                    map_AA[i] = residue[r]["backbone_map"]["C*"]
                elseif d.res_name=="TLA" && d.res_number==3 && d.atom_name=="N"
                    map_AA[i] = residue[r]["backbone_map"]["N*"]
                elseif d.res_name == AA
                    if  d.atom_name in ["N","H","CA","HA","HA2","C","O"]
                        map_AA[i] = residue[r]["backbone_map"][d.atom_name]
                    else
                        push!(Atoms, Atom(index=count, atom_type=a.atom_type, mass=a.mass, charge=a.charge, σ=a.σ, ϵ=a.ϵ, 
                        λ=T(aminos[r][AA]), alch_role=ProbRole))
                        push!(Data, AtomData(d.atom_type, d.atom_name, r, d.res_name, d.chain_id, d.element, d.hetero_atom))
                        push!(Coords, c)
                        map_AA[i] = count
                        push!(sidechain_A, i)
                        if !haskey(addition_groups, r)
                            addition_groups[r]  = [count]
                        else
                            addition_groups[r] = push!(addition_groups[r], count)
                        end
                        if d.atom_name in ["CG","CG1","OG1","OG","SG"]
                            rotamer_search[r][AA]["G_idx"] = count
                        elseif d.atom_name=="CB"
                            rotamer_search[r][AA]["CB_idx"] = count
                            CMAP_idx = count
                        elseif d.atom_name=="HA3"
                            CMAP_idx = count
                        end
                        push!(rotamer_search[r][AA]["index"], count)
                        count += 1
                    end
                end
            end
        
            # Add all interactions for the new atoms
            for interaction in tmp_sys.specific_inter_lists
                IT = typeof(interaction).name.wrapper
                field_names = getfield.((interaction,), fieldnames(typeof(interaction)))
                tmp = [typeof(fn)() for fn in field_names[1:end-1]]
        
                if typeof(field_names[end-2][1]).name.name==:CMAPTorsion
                    n_fields = length(tmp)
                    tmp[n_fields-1] = CMAPTorsion_L{Int,T}[]
                    for field_tuple in zip(field_names[1:end-1]...)
                        name = typeof(field_tuple[end-1]).name.name
                        if all(haskey(map_AA, field_tuple[i]) for i in 1:n_fields-2)
                            for i in 1:n_fields-2
                                tmp[i] = push!(tmp[i], map_AA[field_tuple[i]])
                            end
                            
                            tmp[n_fields-1] = push!(tmp[n_fields-1], CMAPTorsion_L(field_tuple[end-1].index, field_tuple[end-1].size, T(aminos[r][AA]), r, aminos_d[AA], CMAP_idx))
                            tmp[n_fields] = push!(tmp[n_fields], "residue_$(r)_r$(aminos_d[AA])_λ")
                        end
                    end
                    tmp[n_fields-1] = collect(tmp[n_fields-1])
                    Interactions = push!(Interactions, IT(tmp..., interaction.data))
                elseif typeof(field_names[end-2][1]).name.name==:LennardJones14
                    n_fields = length(tmp)
                    typs = fieldtypes(typeof(field_names[end-2][1]))
                    tmp[n_fields-1] = LennardJones14SoftCoreGapsys{typs[1],typs[2],typs[3],T,typeof(ProductMixing()), typeof(ProbabilityLambdaScheduler())}[]
                    for field_tuple in zip(field_names[1:end-1]...)
                        name = typeof(field_tuple[end-1]).name.name
                        if all(haskey(map_AA, field_tuple[i]) for i in 1:n_fields-2)
                            for i in 1:n_fields-2
                                tmp[i] = push!(tmp[i], map_AA[field_tuple[i]])
                            end
                            
                            tmp[n_fields-1] = push!(tmp[n_fields-1], LennardJones14SoftCoreGapsys(field_tuple[end-1].σ14_mixed, field_tuple[end-1].ϵ14_mixed, field_tuple[end-1].weight_14,
                                                                                                     T(0.85), ProductMixing(), ProbabilityLambdaScheduler()))
                            tmp[n_fields] = push!(tmp[n_fields], field_tuple[end])
                        end
                    end
                    tmp[n_fields-1] = collect(tmp[n_fields-1])
                    Interactions = push!(Interactions, IT(tmp..., interaction.data))
                else
                    for field_tuple in zip(field_names[1:end-1]...)
                        n_fields = length(field_tuple)
                        name = typeof(field_tuple[end-1]).name.name
                        if any(field_tuple[i] in sidechain_A for i in 1:n_fields-2)
                            for i in 1:n_fields-2
                                tmp[i] = push!(tmp[i], map_AA[field_tuple[i]])
                            end
                            
                            tmp[n_fields-1] = push!(tmp[n_fields-1], field_tuple[end-1])
                            tmp[n_fields] = push!(tmp[n_fields], field_tuple[end])
                        end
                    end
                    Interactions = push!(Interactions, IT(tmp..., nothing))
                end
            end
        end
    end

    Interactions = merge(Interactions)
    specific_inter_lists = tuple(Interactions...)

    # Rotamer search
    if direction
        rota_order = res_num
    else
        rota_order = reverse(res_num)
    end

    for r in rota_order
        # println(r)
        for AA in keys(aminos[r])
            if AA in keys(rotamer_dic)
                # println(AA)
                N_idx = rotamer_search["N_idx"][r]
                CA_idx = rotamer_search["CA_idx"][r]
                CB_idx = rotamer_search[r][AA]["CB_idx"]
                G_idx = rotamer_search[r][AA]["G_idx"]
                indexes = rotamer_search[r][AA]["index"]
                best_angle = 0
                num_clashes = 10000
                for χ1 in rotamer_dic[AA]
                    # println("χ1:",χ1)
                    θ = torsion_angle(Coords[N_idx], Coords[CA_idx], Coords[CB_idx], Coords[G_idx], Boundary)
                    Δθ = deg2rad(χ1) - θ
                    new_coords = rotate_side_chain(Coords, CA_idx, CB_idx, indexes, Δθ)
                    result = filter(pair -> pair.first!=r, rotamer_search)
                    result = filter(pair -> pair.first!="N_idx", result)
                    result = filter(pair -> pair.first!="CA_idx", result)
                    clash_idx = collect(i for (k1, v1) in result for (k2, v2) in v1 for i in v2["index"])
                    clashes = count_clashes(new_coords[rotamer_search[r][AA]["index"]], new_coords[clash_idx])
                    # println(clashes)
                    if clashes<num_clashes
                        best_angle = χ1
                        num_clashes = clashes
                    end
                    if clashes==0
                        best_angle = χ1
                        num_clashes = clashes
                        break
                    end
                end
                # println("Best angle: ", best_angle, ", with ", num_clashes, " clashes")
                θ = torsion_angle(Coords[N_idx], Coords[CA_idx], Coords[CB_idx], Coords[G_idx], Boundary)
                Δθ = deg2rad(best_angle) - θ
                Coords = rotate_side_chain(Coords, CA_idx, CB_idx, indexes, Δθ)
                θ = torsion_angle(Coords[N_idx], Coords[CA_idx], Coords[CB_idx], Coords[G_idx], Boundary)
                # println("Check best angle:", deg2rad(best_angle), ", calculated angle:", θ)
            end
        end
    end

    # Ensure all arrays have correct typing
    Atoms = Vector{typeof(Atoms[1])}(Atoms)
    Coords = Vector{typeof(Coords[1])}(Coords)
    Data = Vector{typeof(Data[1])}(Data)
    specific_inter_lists = tuple(Interactions...)

    # Calculate matrix of pairs eligible for non-bonded interactions
    n_atoms = length(Coords)
    eligible = trues(n_atoms, n_atoms)
    eligible_PME = trues(n_atoms, n_atoms)
    for i in 1:n_atoms
        eligible[i, i] = false
        
        eligible_PME[i, i] = false
    end
    for (i, j) in zip(specific_inter_lists[1].is, specific_inter_lists[1].js)
        eligible[i, j] = false
        eligible[j, i] = false

        eligible_PME[i, j] = false
        eligible_PME[j, i] = false
    end
    for (i, k) in zip(specific_inter_lists[2].is, specific_inter_lists[2].ks)
        # Assume bonding is already specified
        eligible[i, k] = false
        eligible[k, i] = false

        eligible_PME[i, k] = false
        eligible_PME[k, i] = false
    end

    for k in keys(addition_groups)
        pairs = collect(Iterators.product(addition_groups[k], addition_groups[k]))
        for (i, j) in pairs
            eligible[i, j] = false
            eligible[j, i] = false
        end
    end
    
    # Calculate matrix of pairs eligible for halved non-bonded interactions
    # This applies to specified pairs in the topology file, usually 1-4 bonded
    special = falses(n_atoms, n_atoms)
    for (i, l) in zip(Interactions[3].is, Interactions[3].ls)
        special[i, l] = true
        special[l, i] = true
    end

    cut = units ? T(1.2)u"nm" : T(1.2)
    cou_const = units ? T(coulomb_const) : ustrip(T(coulomb_const))
    if AT <:AbstractGPUArray
        nf = GPUNeighborFinder(eligible=to_device(eligible, AT), dist_cutoff=cut, 
                                special=to_device(special, AT), n_steps_reorder=10)
    else
        nf = CellListMapNeighborFinder(eligible=to_device(eligible, AT), dist_cutoff=cut, 
                                        special=to_device(special, AT), n_steps=10)
    end
    σQ= units ? T(1.0)u"nm" : T(1.0)
    cutoff = DistanceCutoff(cut)
    pairwise_inters = (
                        LennardJonesSoftCoreGapsys(cutoff=cutoff, α=T(0.85), use_neighbors=true,  
                                                    shortcut=LJZeroShortcut(), λ_mixing=ProductMixing(),
                                                    scheduler=ProbabilityLambdaScheduler(), weight_special=T(0.0)),
                        CoulombSoftCoreGapsysEwald(dist_cutoff=(units ? T(1.0)u"nm" : T(1.0)), error_tol=T(0.0005), 
                                                    α=T(0.6), σQ=units ? T(1.0)u"nm" : T(1.0),
                                                    use_neighbors=true,  λ_mixing=ProductMixing(),
                                                    scheduler=ProbabilityLambdaScheduler(),
                                                    weight_special=T(1.0), 
                                                    coulomb_const=(units ? T(coulomb_const) : T(ustrip(coulomb_const))), 
                                                    approximate_erfc=true)
                        )

    
    general_inters = (
                        PME((units ? T(1.0)u"nm" : T(1.0)), to_device(Atoms, AT), Boundary; error_tol=T(0.0005), 
                            eligible=to_device(eligible_PME, AT), 
                            special=to_device(special, AT), grad_safe=true),
                        )

    # Set-up final system
    vels_gpu = [random_velocity(a.mass, temp) for a in Atoms]
    
    sys_final = System(
        atoms=to_device(Atoms, AT),
        coords=to_device(Coords, AT),
        atoms_data=Data,
        boundary=Boundary,
        velocities=to_device(vels_gpu, AT),
        pairwise_inters=pairwise_inters,
        specific_inter_lists=to_device.(specific_inter_lists,AT),
        neighbor_finder=nf,
        general_inters=general_inters,
        loggers=(
            writer=TrajectoryWriter(1_000, traj_file),
        ),
        force_units=(units ? u"kJ * mol^-1 * nm^-1" : NoUnits),
        energy_units=(units ? u"kJ * mol^-1" : NoUnits),
    )

    return sys_final
end

# System A is the main reference system from which the environment atoms are taken
# System B is only used for the Unique system B atoms and the parameters for Core atoms

function Hybrid_system(T, AT, sysA::System, sysB::System, global_λ, mapping, core_mapAB, traj_file; temp = T(298.0)u"K", units=true)
    # Collect correct datatypes and fill in missing gaps in the mappings
    S = typeof(sysA.atoms[1].σ)
    E = typeof(sysA.atoms[1].ϵ)
    C = typeof(sysA.atoms[1].charge)
    core_mapBA = dict_reverse(core_mapAB)
    env = []
    for i in 1:length(sysA.atoms)
        if !(i in mapping["unique_A"]) && !(i in mapping["core"])
            push!(env, i)
        end
    end
    mapping["env"] = env
    mapping_A = Dict()
    mapping_B = Dict()
    unique_groups = Dict("sysA"=>[], "sysB"=>[])

    # Initialize data groups for new system
    Atoms = []
    Data = []
    Coords = []
    Interactions = []
    p_charge = Dict{Int64, SVector{2, C}}()
    p_σ = Dict{Int64, SVector{2, S}}()
    p_ϵ = Dict{Int64, SVector{2, E}}()
    Boundary = deepcopy(sysA.boundary)

    # Add all the atoms with new numbering (save a mapping to adjust the interactions list later)
    # Add first core, then unique and then environment atoms
    counter = 1
    for i in mapping["core"]
        a = sysA.atoms[i]
        d = sysA.atoms_data[i]
        c = sysA.coords[i]
        push!(Atoms, Atom(index=counter, atom_type=a.atom_type, mass=a.mass, charge=a.charge, σ=a.σ, ϵ=a.ϵ, 
                λ=T(global_λ), alch_role=CoreRole))
        push!(Data, AtomData(atom_type=d.atom_type, atom_name=d.atom_name, res_number=d.res_number,
                                    res_name=d.res_name, chain_id=d.chain_id, element=d.element, hetero_atom=d.hetero_atom))
        push!(Coords, c)
        mapping_A[i] = counter
        mapping_B[core_mapBA[i]] = counter
        p_charge[counter] = SVector{2,C}(sysA.atoms[i].charge, sysB.atoms[core_mapAB[i]].charge)
        p_σ[counter] = SVector{2,S}(sysA.atoms[i].σ, sysB.atoms[core_mapAB[i]].σ)
        p_ϵ[counter] = SVector{2,E}(sysA.atoms[i].ϵ, sysB.atoms[core_mapAB[i]].ϵ)
        counter += 1
    end
    for i in mapping["unique_A"]
        a = sysA.atoms[i]
        d = sysA.atoms_data[i]
        c = sysA.coords[i]
        push!(Atoms, Atom(index=counter, atom_type=a.atom_type, mass=a.mass, charge=a.charge, σ=a.σ, ϵ=a.ϵ, 
                λ=T(global_λ), alch_role=DeleteRole))
        push!(Data, AtomData(atom_type=d.atom_type, atom_name=d.atom_name, res_number=d.res_number,
                                    res_name=d.res_name, chain_id=d.chain_id, element=d.element, hetero_atom=d.hetero_atom))
        push!(Coords, c)
        mapping_A[i] = counter
        push!(unique_groups["sysA"], counter)
        counter += 1
    end
    for i in mapping["unique_B"]
        a = sysB.atoms[i]
        d = sysB.atoms_data[i]
        c = sysB.coords[i]
        push!(Atoms, Atom(index=counter, atom_type=a.atom_type, mass=a.mass, charge=a.charge, σ=a.σ, ϵ=a.ϵ, 
                λ=T(global_λ), alch_role=InsertRole))
        push!(Data, AtomData(atom_type=d.atom_type, atom_name=d.atom_name, res_number=d.res_number,
                                    res_name=d.res_name, chain_id=d.chain_id, element=d.element, hetero_atom=d.hetero_atom))
        push!(Coords, c)
        mapping_B[i] = counter
        push!(unique_groups["sysB"], counter)
        counter += 1
    end
    for i in mapping["env"]
        a = sysA.atoms[i]
        d = sysA.atoms_data[i]
        c = sysA.coords[i]
        push!(Atoms, Atom(index=counter, atom_type=a.atom_type, mass=a.mass, charge=a.charge, σ=a.σ, ϵ=a.ϵ, 
                λ=T(1.0), alch_role=EnvRole))
        push!(Data, AtomData(atom_type=d.atom_type, atom_name=d.atom_name, res_number=d.res_number,
                                    res_name=d.res_name, chain_id=d.chain_id, element=d.element, hetero_atom=d.hetero_atom))
        push!(Coords, c)
        mapping_A[i] = counter
        counter += 1
    end

    # Adding all the interactions
    # Interactions from System A
    for interaction in sysA.specific_inter_lists
        IT = typeof(interaction).name.wrapper
        field_names = getfield.((interaction,), fieldnames(typeof(interaction)))
        field_types = fieldtypes(typeof(interaction))[1:end-1]
        converted_type = typeof(to_lambda_param(interaction.inters[1]))
        field_types = [field_types[1:end-2]..., Vector{converted_type}, field_types[end]]
        tmp = [T() for T in field_types]
        for field_tuple in zip(field_names[1:end-1]...)
            n_fields = length(field_tuple)
            name = typeof(field_tuple[end-1]).name.name
            if all(haskey(mapping_A, field_tuple[i]) for i in 1:n_fields-2)
                for i in 1:n_fields-2
                    tmp[i] = push!(tmp[i], mapping_A[field_tuple[i]])
                end
                tmp[n_fields-1] = push!(tmp[n_fields-1], to_lambda_param(field_tuple[end-1]))
                tmp[n_fields] = push!(tmp[n_fields], field_tuple[end])
            end
        end
        Interactions = push!(Interactions, IT(tmp..., nothing))

    end

    # Interactions from System B
    for interaction in sysB.specific_inter_lists
        IT = typeof(interaction).name.wrapper
        field_names = getfield.((interaction,), fieldnames(typeof(interaction)))
        field_types = fieldtypes(typeof(interaction))[1:end-1]
        converted_type = typeof(to_lambda_param(interaction.inters[1]))
        field_types = [field_types[1:end-2]..., Vector{converted_type}, field_types[end]]
        tmp = [T() for T in field_types]
        for field_tuple in zip(field_names[1:end-1]...)
            n_fields = length(field_tuple)
            name = typeof(field_tuple[end-1]).name.name
            if any(haskey(mapping_B, field_tuple[i]) for i in 1:n_fields-2)
                mapped = Vector{Int}(undef, n_fields-2)
                for i in 1:n_fields-2
                    if haskey(mapping_B, field_tuple[i])
                        mapped[i] = mapping_B[field_tuple[i]]
                    else
                        mapped[i] = mapping_A[core_mapBA[field_tuple[i]]]
                    end
                end

                if find_and_merge_core_interaction!(Interactions, IT, mapped, field_tuple[end-1], field_tuple[end])
                    continue
                end

                for i in 1:n_fields-2
                    tmp[i] = push!(tmp[i], mapped[i])
                end
                tmp[n_fields-1] = push!(tmp[n_fields-1], to_lambda_param(field_tuple[end-1]))
                tmp[n_fields] = push!(tmp[n_fields], field_tuple[end])
            end
        end
        Interactions = push!(Interactions, IT(tmp..., nothing))
    end

    Interactions = merge(Interactions)

    # Ensure all arrays have correct typing
    Atoms = Vector{typeof(Atoms[1])}(Atoms)
    Coords = Vector{typeof(Coords[1])}(Coords)
    Data = Vector{typeof(Data[1])}(Data)
    specific_inter_lists = tuple(Interactions...)

    # Pairwise Interactions
    n_atoms = length(Coords)
    eligible = trues(n_atoms, n_atoms)
    eligible_PME = trues(n_atoms, n_atoms)
    for i in 1:n_atoms
        eligible[i, i] = false
        
        eligible_PME[i, i] = false
    end
    for (i, j) in zip(specific_inter_lists[1].is, specific_inter_lists[1].js)
        eligible[i, j] = false
        eligible[j, i] = false

        eligible_PME[i, j] = false
        eligible_PME[j, i] = false
    end
    for (i, k) in zip(specific_inter_lists[2].is, specific_inter_lists[2].ks)
        eligible[i, k] = false
        eligible[k, i] = false

        eligible_PME[i, k] = false
        eligible_PME[k, i] = false
    end

    pairs = collect(Iterators.product(unique_groups["sysA"], unique_groups["sysB"]))
    for (i, j) in pairs
        eligible[i, j] = false
        eligible[j, i] = false
    end

    special = falses(n_atoms, n_atoms)
    for (i, l) in zip(Interactions[3].is, Interactions[3].ls)
        special[i, l] = true
        special[l, i] = true
    end

    if AT <:AbstractGPUArray
        nf = GPUNeighborFinder(eligible=to_device(eligible, AT), dist_cutoff=sysA.neighbor_finder.dist_cutoff, 
                                special=to_device(special, AT), n_steps_reorder=10)
    else
        nf = CellListMapNeighborFinder(eligible=to_device(eligible, AT), dist_cutoff=sysA.neighbor_finder.dist_cutoff, 
                                        special=to_device(special, AT), n_steps=10)
    end

    PairInteraction = []
    for inter in sysA.pairwise_inters
        if inter isa LennardJones
            push!(PairInteraction, LennardJonesSoftCoreGapsys(cutoff=inter.cutoff, α=T(0.85), use_neighbors=inter.use_neighbors,  
                                                    shortcut=inter.shortcut, σ_mixing=CoreMixing(inter.σ_mixing, ParamsList(p_σ)), 
                                                    ϵ_mixing=CoreMixing(inter.ϵ_mixing, ParamsList(p_ϵ)),
                                                    λ_mixing=MinimumMixing(),scheduler=DefaultLambdaScheduler(), weight_special=inter.weight_special))
        elseif inter isa Coulomb
            push!(PairInteraction, CoulombSoftCoreGapsys(cutoff=inter.cutoff,
                                                    α=T(0.6), σQ=units ? T(1.0)u"nm" : T(1.0),
                                                    use_neighbors=inter.use_neighbors, λ_mixing=MinimumMixing(),
                                                    charge_scaling=ParamsList(p_charge),
                                                    scheduler=DefaultLambdaScheduler(),
                                                    weight_special=inter.weight_special, 
                                                    coulomb_const=inter.coulomb_const))
        elseif inter isa CoulombEwald
            push!(PairInteraction, CoulombSoftCoreGapsysEwald(dist_cutoff=inter.dist_cutoff, error_tol=inter.error_tol, 
                                                    α=T(0.6), σQ=units ? T(1.0)u"nm" : T(1.0),
                                                    use_neighbors=inter.use_neighbors, λ_mixing=MinimumMixing(),
                                                    charge_scaling=ParamsList(p_charge),
                                                    scheduler=DefaultLambdaScheduler(),
                                                    weight_special=inter.weight_special, 
                                                    coulomb_const=inter.coulomb_const, 
                                                    approximate_erfc=inter.approximate_erfc))
        else
            error("Currently {$inter} is not yet implemented for relative binding free energy")
        end
    end

    pairwise_inters = tuple(PairInteraction...)

    # General interactions
    GenerInteraction = []
    for inter in sysA.general_inters
        if inter isa PME
            push!(GenerInteraction, PME(inter.dist_cutoff, to_device(Atoms, AT), Boundary; error_tol=inter.error_tol, 
                            fixed_charges=false, eligible=to_device(eligible_PME, AT), special=to_device(special, AT), 
                            scheduler=DefaultLambdaScheduler(), charge_scaling=ParamsList(p_charge), grad_safe=inter.grad_safe),
                        )
        elseif inter isa LJDispersionCorrection
            push!(GenerInteraction, LJDispersionCorrectionλ(to_device(Atoms, AT), inter.dist_cutoff, DefaultLambdaScheduler(), 
                            MinimumMixing(), CoreMixing(LorentzMixing(), ParamsList(p_σ)), 
                            CoreMixing(GeometricMixing(), ParamsList(p_ϵ)))
                            )
        end
    end

    general_inters = tuple(GenerInteraction...)

    # Setup new system
    vels_gpu = [random_velocity(a.mass, temp) for a in Atoms]

    sys_final = System(
        atoms=to_device(Atoms, AT),
        coords=to_device(Coords, AT),
        atoms_data=Data,
        boundary=Boundary,
        velocities=to_device(vels_gpu, AT),
        pairwise_inters=pairwise_inters,
        specific_inter_lists=to_device.(specific_inter_lists,AT),
        neighbor_finder=nf,
        general_inters=general_inters,
        loggers=(
            writer=TrajectoryWriter(1_000, traj_file),
        ),
        force_units=(units ? u"kJ * mol^-1 * nm^-1" : NoUnits),
        energy_units=(units ? u"kJ * mol^-1" : NoUnits),
    )

    return sys_final, mapping_A, mapping_B
end