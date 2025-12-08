export
    Hybrid_system

"""
    Hybrid_system(T, AT, ff, sys, res_num, aminos)

Mutates residue in a system to (multiple) λ-residues and returns a hybrid system of
original system and λ-mediated atoms.

# Arguments
- `T`: Float type used, often Float32
- `AT`: Array type, whether to run on CPU (Array) or GPU (CuArray)
- `ff`: MolecularForceField struct that is used to parametrize the system.
- `sys`: The system in which a residue is going to be mutated.
- `res_num`: The residue number of the residue that we want to mutate.
- `aminos = Nothing`: A dictionary of the substitutions and the λ-values. If nothing, the whole library (all AA, except PRO) will be added with λ=0.05.
- `temp = T(298.0)u"K"`: Temperature to generate random velocities for the system, standard is 298K.
"""


function Hybrid_system(T, AT, ff, sys, traj_file, res_num, aminos=nothing, temp = T(298.0)u"K")
    # initialize data groups for new system
    Atoms = []
    Data = []
    Coords = []
    Interactions = []
    Boundary = deepcopy(sys.boundary)
    if isnothing(aminos)
        aminos = Dict("ARG"=>0.05,"HIS"=>0.05,"LYS"=>0.05,"ASP"=>0.05,
                    "GLU"=>0.05,"SER"=>0.05,"THR"=>0.05,"ASN"=>0.05,
                    "GLN"=>0.05,"CYS"=>0.05,"GLY"=>0.05,"ALA"=>0.05,
                    "VAL"=>0.05,"ILE"=>0.05,"LEU"=>0.05,"MET"=>0.05,
                    "PHE"=>0.05,"THR"=>0.05,"TRP"=>0.05)
    end
    
    # Remove all the atoms of selected residue except backbone and set λ to 1.0
    # Make map of backbone to map interactions on later for added side-chains
    map = Dict()
    residue = Dict()
    CAs = []
    for r in res_num
        residue[r] = Dict("backbone_map" => Dict(), "backbone_coords" => zeros(T,3,3)u"nm")
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
                    σ14=a.σ14, ϵ14=a.ϵ14, λ=T(1.0)))
            push!(Data, d)
            push!(Coords, c)
            map[i] = count
            count += 1
        elseif (d.res_number in res_num) && d.atom_name in ["N","CA","C","O","H","HA","HA2"]
            push!(Atoms, Atom_L(index=count, mass=a.mass, charge=a.charge, σ=a.σ, ϵ=a.ϵ, 
                    σ14=a.σ14, ϵ14=a.ϵ14, λ=T(1.0)))
            push!(Data, d)
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
                    tmp[n_fields] = push!(tmp[n_fields], field_tuple[end])
                    push!(maps, interaction.data[index+1:index+(4*field_tuple[end-1].size*field_tuple[end-1].size), :])
                    index += 4*field_tuple[end-1].size*field_tuple[end-1].size
                end
            end
            maps = vcat(maps...)
            Interactions = push!(Interactions, InteractionList5Atoms{IT.types[1], Vector{CMAPTorsion{CMAP.types[1]}}, IT.types[end]}(tmp..., maps))
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
        for AA in keys(aminos)
            # Load residue system
            amino_dir = normpath(@__DIR__, "..", "data/aminoacids")
            tmp_sys = System(amino_dir*"/"*AA*".pdb", ff;
                     nonbonded_method=:cutoff, center_coords=false)
            
            # Superimpose residue onto backbone
            backbone_coords2 = zeros(T,3,3)u"nm"
            for (d,c) in zip(tmp_sys.atoms_data, tmp_sys.coords)
                if d.atom_name in ["HA","CA","CB","HA2","HA3"] && (d.res_name!="ACE" && d.res_name!="NME") 
                    backbone_coords2[backbone_idx[d.atom_name],:] = c
                end
            end
            rot, t = kabsch(backbone_coords2, residue[r]["backbone_coords"])
            coords2 = ((rot * hcat(tmp_sys.coords...)) .+ (t)u"nm")'
            
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
                        σ14=a.σ14, ϵ14=a.ϵ14, λ=T(aminos[AA])))
                        push!(Data, AtomData(d.atom_type, d.atom_name, r, d.res_name, d.chain_id, d.element, d.hetero_atom))
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
                            
                            tmp[n_fields-1] = push!(tmp[n_fields-1], CMAPTorsion_L(field_tuple[end-1].index, field_tuple[end-1].size, T(aminos[AA])))
                            tmp[n_fields] = push!(tmp[n_fields], field_tuple[end]*"λ")
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
    
    cutoff = DistanceCutoff(T(1.2)u"nm")
    if AT <:AbstractGPUArray
        nf = GPUNeighborFinder(eligible=Molly.to_device(eligible, AT), dist_cutoff=T(1.2)u"nm", special=Molly.to_device(special, AT), n_steps_reorder=10)
    else
        nf = CellListMapNeighborFinder(eligible=Molly.to_device(eligible, AT), dist_cutoff=T(1.2)u"nm", special=Molly.to_device(special, AT), n_steps=10)
    end
    pairwise_inters = (
                        LennardJonesSoftCoreGapsys(α=T(0.85), λ=nothing, use_neighbors=true, cutoff=cutoff, shortcut=Molly.lj_λ_one_shortcut),
                        LennardJones(use_neighbors=true, cutoff=cutoff, shortcut=Molly.lj_λ_less_one_shortcut),
                        Coulomb(use_neighbors=true, cutoff=cutoff, shortcut=Molly.coul_λ_less_one_shortcut, coulomb_const=T(coulomb_const)),
                        CoulombSoftCoreGapsys(α=T(0.3), λ=nothing, σQ=T(1.0)u"nm", use_neighbors=true, cutoff=cutoff, shortcut=Molly.coul_λ_one_shortcut, coulomb_const=T(coulomb_const)),
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
            temp=TemperatureLogger(1000),
            writer=TrajectoryWriter(1000, traj_file),
            step_log=GeneralObservableLogger(step_logger, typeof(one(T)u"kJ * mol^-1"), 10_000),
        ),
    )

    return sys_final
end