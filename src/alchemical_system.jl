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


function Hybrid_system(T, AT, ff, sys, res_num, aminos=nothing, temp = T(298.0)u"K")
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
    backbone_map = Dict()
    backbone_coords = zeros(T,4,3)u"nm"
    backbone_idx = Dict("N"=>1,"CA"=>2,"C"=>3,"O"=>4)
    count = 1
    for (i,(a,d,c)) in enumerate(zip(sys.atoms, sys.atoms_data, sys.coords))
        if (res_num-d.res_number)==1 && d.atom_name=="C"
            backbone_map[d.atom_name*"*"] = count
        elseif (d.res_number-res_num)==1 && d.atom_name=="N"
            backbone_map[d.atom_name*"*"] = count
        end
        if d.res_number != res_num
            push!(Atoms, Atom_L(index=count, mass=a.mass, charge=a.charge, σ=a.σ, ϵ=a.ϵ, 
                    σ14=a.σ14, ϵ14=a.ϵ14, λ=T(1.0)))
            push!(Data, d)
            push!(Coords, c)
            map[i] = count
            count += 1
        elseif d.res_number == res_num && d.atom_name in ["N","CA","C","O","H","HA","HA2"]
            push!(Atoms, Atom_L(index=count, mass=a.mass, charge=a.charge, σ=a.σ, ϵ=a.ϵ, 
                    σ14=a.σ14, ϵ14=a.ϵ14, λ=T(1.0)))
            push!(Data, d)
            push!(Coords, c)
            map[i] = count
            backbone_map[d.atom_name] = count
            if d.atom_name=="HA"
                backbone_map["HA2"] = count
            elseif d.atom_name=="HA2"
                backbone_map["HA"] = count
            end
            count += 1
            if d.atom_name in ["N","CA","C","O"]
                backbone_coords[backbone_idx[d.atom_name],:] = c
            end
        end
    end
    
    # Add all the in the interactions except the ones part of the side-chain that has been removed
    for interaction in sys.specific_inter_lists
        interaction_type = typeof(interaction)
        field_names = getfield.((interaction,), fieldnames(typeof(interaction)))
        tmp = [[] for _ in field_names[1:end-1]]
        name = ""
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
        if name==:CMAPTorsion
            continue
        else
            Interactions = push!(Interactions, interaction_type(tmp..., nothing))
        end
    end

    # Add new side-chains for selected residue
    map_AA = Dict()
    for AA in keys(aminos)
        # Load residue system
        amino_dir = normpath(@__DIR__, "..", "data/aminoacids")
        tmp_sys = System(amino_dir*"/"*AA*".pdb", ff;
                 nonbonded_method=:cutoff, center_coords=false)
        
        # Superimpose residue onto backbone
        backbone_coords2 = zeros(T,4,3)u"nm"
        for (d,c) in zip(tmp_sys.atoms_data, tmp_sys.coords)
            if d.atom_name in ["N","CA","C","O"] && (d.res_name!="ACE" && d.res_name!="NME") 
                backbone_coords2[backbone_idx[d.atom_name],:] = c
            end
        end
        R, t = kabsch(backbone_coords2, backbone_coords)
        coords2 = hcat(tmp_sys.coords...)' * R .+ t'u"nm"
        
        # Create map for interactions and add atoms of the side-chains
        sidechain_A = []
        map_AA = Dict()
        count = isempty(map_AA) ? maximum(values(map))+1 : maximum(values(map_AA))+1
        for (i,(a,d,c)) in enumerate(zip(tmp_sys.atoms, tmp_sys.atoms_data, eachrow(coords2)))
            if d.res_name=="ACE" && d.atom_name=="C"
                map_AA[i] = backbone_map["C*"]
            elseif d.res_name=="NME" && d.atom_name=="N"
                map_AA[i] = backbone_map["N*"]
            elseif d.res_name == AA
                if  d.atom_name in ["N","H","CA","HA","HA2","C","O"]
                    map_AA[i] = backbone_map[d.atom_name]
                else
                    push!(Atoms, Atom_L(index=count, mass=a.mass, charge=a.charge, σ=a.σ, ϵ=a.ϵ, 
                    σ14=a.σ14, ϵ14=a.ϵ14, λ=T(aminos[AA])))
                    push!(Data, AtomData(d.atom_type, d.atom_name, res_num, d.res_name, d.chain_id, d.element, d.hetero_atom))
                    push!(Coords, SVector{length(c)}(c))
                    map_AA[i] = count
                    push!(sidechain_A, i)
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
                        tmp[n_fields] = push!(tmp[n_fields], field_tuple[end])
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

    # Merge all interaction lists
    Interactions = Molly.merge(Interactions)

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
        pairwise_inters=(),
        specific_inter_lists=Molly.to_device.(specific_inter_lists,AT),
        general_inters=(),
    )

    return sys_final
end