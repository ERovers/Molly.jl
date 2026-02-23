export
    Hybrid_system

const aminos_dic = Dict("ALA" => 1, "ARG" => 2, "ASN" => 3,
                        "ASP" => 4, "CYS" => 5, "GLN" => 6,
                        "GLU" => 7, "GLY" => 8, "HIS" => 9,
                        "ILE" => 10, "LEU" => 11, "LYS" => 12, 
                        "MET" => 13, "PHE" => 14, "SER" => 15,
                        "THR" => 16, "TRP" => 17, "TYR" => 18,
                        "VAL" => 19, "ACE" => 0, "ALC" => 0,
                        "NME" => 0, "HOH" => 0)

const charge_aminos_dic = Dict("ALA" => 0, "ARG" => 1, "ASN" => 0,
                                "ASP" => -1, "CYS" => 0, "GLN" => 0,
                                "GLU" => -1, "GLY" => 0, "HIS" => 1,
                                "ILE" => 0, "LEU" => 0, "LYS" => 1, 
                                "MET" => 0, "PHE" => 0, "SER" => 0,
                                "THR" => 0, "TRP" => 0, "TYR" => 0,
                                "VAL" => 0, "ACE" => 0, "ALC" => 0,
                                "NME" => 0, "HOH" => 0)

const solvent_aminos_dic = Dict("ALA" => 0, "ARG" => 0, "ASN" => 0,
                                "ASP" => 0, "CYS" => 0, "GLN" => 0,
                                "GLU" => 0, "GLY" => 0, "HIS" => 0,
                                "ILE" => 0, "LEU" => 0, "LYS" => 0, 
                                "MET" => 0, "PHE" => 0, "SER" => 0,
                                "THR" => 0, "TRP" => 0, "TYR" => 0,
                                "VAL" => 0, "ACE" => 0, "ALC" => 0,
                                "NME" => 0, "HOH" => 1)

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


function Hybrid_system(T, AT, epoch, ff, sys, traj_file, params_dic, temp = T(298.0)u"K", units=true)
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
            res_type = [key for (key,value) in aminos_dic if value==i][1]
            if haskey(aminos, res_n)
                aminos[res_n][res_type] = l
            else
                aminos[res_n] = Dict(res_type => l)
            end
        end
    end
    res_num = keys(aminos)
    
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
            push!(Atoms, Atom_L(index=count, mass=a.mass, charge=a.charge, σ=a.σ, ϵ=a.ϵ, 
                    σ14=a.σ14, ϵ14=a.ϵ14, λ=T(1.0), charge_type=charge_aminos_dic[d.res_name], solvent=solvent_aminos_dic[d.res_name]))
            # push!(Data, d)
            push!(Data, AtomData_L(atom_type=d.atom_type, atom_name=d.atom_name, res_number=d.res_number,
                                        res_name=d.res_name, res_id=aminos_dic[d.res_name], element=d.element))
            push!(Coords, c)
            map[i] = count
            count += 1
        elseif (d.res_number in res_num) && d.atom_name in ["N","CA","C","O","H","HA","HA2"]
            push!(Atoms, Atom_L(index=count, mass=a.mass, charge=a.charge, σ=a.σ, ϵ=a.ϵ, 
                    σ14=a.σ14, ϵ14=a.ϵ14, λ=T(1.0), charge_type=charge_aminos_dic[d.res_name], solvent=solvent_aminos_dic[d.res_name]))
            push!(Data, AtomData_L(atom_type=d.atom_type, atom_name=d.atom_name, res_number=d.res_number,
                                        res_name="ALC", res_id=aminos_dic[d.res_name], element=d.element))
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
        IT = typeof(interaction)
        field_names = getfield.((interaction,), fieldnames(typeof(interaction)))
        tmp = [[] for _ in field_names[1:end-1]]
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
                Interactions = push!(Interactions, InteractionList5Atoms{IT.types[1], Vector{CMAPTorsion{CMAP.types[1]}}, IT.types[end]}(tmp..., maps))
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
    for r in res_num
        for AA in keys(aminos[r])
            # Load residue system
            amino_dir = normpath(@__DIR__, "..", "data/aminoacids")
            tmp_sys = System(amino_dir*"/"*AA*".pdb", ff;
                     nonbonded_method=:cutoff, center_coords=false, units=units)
            
            # Superimpose residue onto backbone
            backbone_coords2 = units ? zeros(T,3,3)u"nm" : zeros(T,3,3)
            for (d,c) in zip(tmp_sys.atoms_data, tmp_sys.coords)
                if d.atom_name in ["HA","CA","CB","HA2","HA3"] && (d.res_name!="ACE" && d.res_name!="NME") 
                    backbone_coords2[backbone_idx[d.atom_name],:] = c
                end
            end
            rot, t = kabsch(backbone_coords2, residue[r]["backbone_coords"])
            t = units ? (t)u"nm" : t
            coords2 = ((rot * hcat(tmp_sys.coords...)) .+ t)'
            
            # Create map for interactions and add atoms
            sidechain_A = []
            count = isempty(map_AA) ? maximum(values(map))+1 : maximum(values(map_AA))+1
            map_AA = Dict()
            for (i,(a,d,c)) in enumerate(zip(tmp_sys.atoms, tmp_sys.atoms_data, eachrow(coords2)))
                if d.res_name=="ACE" && d.atom_name=="C"
                    map_AA[i] = residue[r]["backbone_map"]["C*"]
                elseif d.res_name=="NME" && d.atom_name=="N"
                    map_AA[i] = residue[r]["backbone_map"]["N*"]
                elseif d.res_name == AA
                    if  d.atom_name in ["N","H","CA","HA","HA2","C","O"]
                        map_AA[i] = residue[r]["backbone_map"][d.atom_name]
                    else
                        push!(Atoms, Atom_L(index=count, mass=a.mass, charge=a.charge, σ=a.σ, ϵ=a.ϵ, 
                        σ14=a.σ14, ϵ14=a.ϵ14, λ=T(aminos[r][AA]), charge_type=charge_aminos_dic[d.res_name], solvent=solvent_aminos_dic[d.res_name]))
                        push!(Data, AtomData_L(d.atom_type, d.atom_name, r, d.res_name, aminos_dic[d.res_name], d.chain_id, d.element, d.hetero_atom))
                        push!(Coords, SVector{length(c)}(c))
                        map_AA[i] = count
                        push!(sidechain_A, i)
                        if !haskey(addition_groups, AA)
                            addition_groups[AA]  = [count]
                        else
                            addition_groups[AA] = push!(addition_groups[AA], count)
                        end
                        count += 1
                    end
                end
            end
        
            # Add all interactions for the new atoms
            for interaction in tmp_sys.specific_inter_lists
                IT = typeof(interaction)
                field_names = getfield.((interaction,), fieldnames(typeof(interaction)))
                tmp = [[] for _ in field_names[1:end-1]]
        
                if typeof(field_names[end-2][1]).name.name==:CMAPTorsion
                    CMAP = typeof(field_names[end-2][1])
                    for field_tuple in zip(field_names[1:end-1]...)
                        n_fields = length(field_tuple)
                        name = typeof(field_tuple[end-1]).name.name
                        if all(haskey(map_AA, field_tuple[i]) for i in 1:n_fields-2)
                            for i in 1:n_fields-2
                                tmp[i] = push!(tmp[i], map_AA[field_tuple[i]])
                            end
                            
                            tmp[n_fields-1] = push!(tmp[n_fields-1], CMAPTorsion_L(field_tuple[end-1].index, field_tuple[end-1].size, T(aminos[r][AA]), r, aminos_dic[AA]))
                            tmp[n_fields] = push!(tmp[n_fields], "residue_$(r)_r$(aminos_dic[AA])_λ")
                        end
                    end
                    Interactions = push!(Interactions, InteractionList5Atoms{IT.types[1], Vector{CMAPTorsion_L{CMAP.types[1], T}}, IT.types[end]}(tmp..., interaction.data))
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
    
    Interactions = Molly.merge(Interactions)

    # Calculate matrix of pairs eligible for non-bonded interactions
    n_atoms = length(Coords)
    eligible = trues(n_atoms, n_atoms)
    for i in 1:n_atoms
        eligible[i, i] = false
    end
    for (i, j) in zip(Interactions[1].is, Interactions[1].js)
        eligible[i, j] = false
        eligible[j, i] = false
    end
    for (i, k) in zip(Interactions[2].is, Interactions[2].ks)
        # Assume bonding is already specified
        eligible[i, k] = false
        eligible[k, i] = false
    end
    
    keys_dict = [(k1, k2) for (k1, k2) in Iterators.product(keys(addition_groups), keys(addition_groups)) if k1 < k2]
    for (k1, k2) in keys_dict
        pairs = collect(Iterators.product(addition_groups[k1], addition_groups[k2]))
        for (i, j) in pairs
            # Assume bonding is already specified
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
        nf = GPUNeighborFinder(eligible=Molly.to_device(eligible, AT), dist_cutoff=cut, special=Molly.to_device(special, AT), n_steps_reorder=10)
    else
        nf = CellListMapNeighborFinder(eligible=Molly.to_device(eligible, AT), dist_cutoff=cut, special=Molly.to_device(special, AT), n_steps=10)
    end
    σQ= units ? T(1.0)u"nm" : T(1.0)
    cutoff = DistanceCutoff(cut)
    pairwise_inters = (
                        LennardJones(use_neighbors=true, cutoff=cutoff, shortcut=lj_λ_less_one_shortcut,weight_special=T(1.0)),
                        Coulomb(use_neighbors=true, cutoff=cutoff, shortcut=coul_λ_less_one_shortcut, coulomb_const=cou_const, weight_special=T(1.0)),
                        LennardJonesSoftCoreGapsys(α=T(0.85), λ=nothing, use_neighbors=true, cutoff=cutoff, shortcut=lj_λ_one_shortcut, weight_special=T(1.0)),
                        CoulombSoftCoreGapsys(α=T(0.3), λ=nothing, σQ=σQ, use_neighbors=true, cutoff=cutoff, shortcut=coul_λ_one_shortcut, coulomb_const=cou_const, weight_special=T(1.0), epoch=epoch),
                        )
    
    # Ensure all arrays have correct typing
    Atoms = Vector{typeof(Atoms[1])}(Atoms)
    Coords = Vector{typeof(Coords[1])}(Coords)
    Data = Vector{typeof(Data[1])}(Data)
    specific_inter_lists = tuple(Interactions...)

    # Set-up final system
    vels_gpu = [random_velocity(a.mass, temp) for a in Atoms]
    
    sys_final = System(
        atoms=Molly.to_device(Atoms, AT),
        coords=Molly.to_device(Coords, AT),
        atoms_data=Data,
        boundary=Boundary,
        velocities=Molly.to_device(vels_gpu, AT),
        pairwise_inters=pairwise_inters,
        specific_inter_lists=Molly.to_device.(specific_inter_lists,AT),
        neighbor_finder=nf,
        general_inters=(),
        loggers=(
            writer=TrajectoryWriter(10_000, traj_file),
            step_log=GeneralObservableLogger(step_logger, typeof(one(T)), 10_000),
        ),
        force_units=(units ? u"kJ * mol^-1 * nm^-1" : NoUnits),
        energy_units=(units ? u"kJ * mol^-1" : NoUnits),
    )

    return sys_final
end