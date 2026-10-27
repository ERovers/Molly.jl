# CHARMM impropers from per-residue lists
#
# CHARMM has an improper because the residue topology lists it, not because the atom types of a
# centre and its three neighbours match a rule, and it can list two impropers on the same four
# atoms. OpenMM's CHARMM force field files carry these lists in a Python <Script> that Molly
# cannot run; `data/force_fields/charmm36m/charmm36_2024_impropers.xml`, written by
# `claude_lb/scripts/charmm_improper_tables.py`, holds the same tables as a <CHARMMImpropers>
# element. This file reads that element and builds the impropers from it, following the script.

# The atoms of one residue by their name in the matched template, plus the template it matched
struct CHARMMResidueAtoms
    res_name::String
    patch::String # "" when the matched template is unpatched
    atoms::Dict{String, Int} # Template atom name to global atom index
end

charmm_improper_classes(node::EzXML.Node) = ntuple(i -> get_ezxml(node, "class$i", ""), 4)

function read_charmm_impropers!(tables, entry, ff_file, units, T)
    for el in eachelement(entry)
        if el.name in ("Residue", "Patch", "PatchDelete")
            name = xml_attr(el, "name", ff_file)
            impropers = CHARMMImproper[]
            for imp in eachelement(el)
                imp.name == "Improper" || continue
                push!(impropers, ntuple(4) do i
                    # An omitted offset means the same residue, which is the common case
                    offset = (haskey(imp, "offset$i") ? parse_attr(Int, imp, "offset$i", ff_file) : 0)
                    (offset, xml_attr(imp, "atom$i", ff_file))
                end)
            end
            if el.name == "Residue"
                tables.residue_impropers[name] = impropers
                tables.residue_order[name] = parse_attr(Int, el, "order", ff_file)
            elseif el.name == "Patch"
                tables.patch_impropers[name] = impropers
                tables.patch_order[name] = parse_attr(Int, el, "order", ff_file)
            else
                tables.delete_impropers[name] = impropers
            end

        elseif el.name == "HarmonicImproper"
            k = add_units(parse_attr(T, el, "k", ff_file), u"kJ * mol^-1", units)
            θ0 = parse_attr(T, el, "theta0", ff_file)
            tables.harmonic_types[charmm_improper_classes(el)] = HarmonicTorsionType(k, θ0)

        elseif el.name == "PeriodicImproper"
            periodicity = parse_attr(Int, el, "periodicity", ff_file)
            phase = parse_attr(T, el, "phase", ff_file)
            k = add_units(parse_attr(T, el, "k", ff_file), u"kJ * mol^-1", units)
            tables.periodic_types[charmm_improper_classes(el)] = PeriodicTorsionType(
                                                    [periodicity], [phase], [k], false)
        end
    end
    return tables
end

# Wildcard positions in the order OpenMM's find_improper tries them: fewest wildcards first,
#   then earliest positions, which is what itertools.combinations gives
const CHARMM_IMPROPER_WILDCARDS = ((), (1,), (2,), (3,), (4,), (1, 2), (1, 3), (1, 4), (2, 3),
                                   (2, 4), (3, 4), (1, 2, 3), (1, 2, 4), (1, 3, 4), (2, 3, 4),
                                   (1, 2, 3, 4))

# The classes are looked up in the listed order only, since the tables hold both directions
function charmm_improper_params(types, classes::NTuple{4, String})
    for wildcards in CHARMM_IMPROPER_WILDCARDS
        key = ntuple(i -> (i in wildcards ? "" : classes[i]), 4)
        haskey(types, key) && return types[key]
    end
    return nothing
end

# OpenMM's script strips the variant suffix of a patch name, e.g. NTER_0 gives NTER
function patch_improper_key(patch)
    us = findlast('_', patch)
    return isnothing(us) ? patch : patch[1:us-1]
end

# The impropers of one residue: those of its template, then those its patch adds, without those
#   its patch deletes
# Molly applies at most one patch per template, where OpenMM can apply several and sorts them by
#   `residue_order` and `patch_order`, so those two tables are read but not needed here
function charmm_improper_list(tables, res_name, patch)
    impropers = get(tables.residue_impropers, res_name, NO_CHARMM_IMPROPERS)
    patch_key = patch_improper_key(patch)
    # A patch with no impropers of its own contributes nothing and its deletions are never
    #   reached, as in OpenMM
    haskey(tables.patch_impropers, patch_key) || return impropers
    patched = copy(impropers)
    for improper in get(tables.delete_impropers, patch_key, NO_CHARMM_IMPROPERS)
        di = findfirst(isequal(improper), patched)
        isnothing(di) || deleteat!(patched, di)
    end
    for improper in tables.patch_impropers[patch_key]
        improper in patched || push!(patched, improper)
    end
    return patched
end

# Offsets refer to the neighbouring residue numbers of the same chain, as the `-C` and `+N` bonds
#   of `create_bonds!` do
function charmm_improper_inds(improper, chain_atoms, res_id)
    inds = ntuple(4) do i
        offset, atom_name = improper[i]
        rec = get(chain_atoms, res_id + offset, nothing)
        isnothing(rec) ? 0 : get(rec.atoms, atom_name, 0)
    end
    return any(iszero, inds) ? nothing : inds
end

# OpenMM's is_valid_improper, which drops an improper whose offsets reached an unbonded residue
function valid_charmm_improper(inds, adj)
    central_first = all(i -> i in adj[inds[1]], (inds[2], inds[3], inds[4]))
    central_last  = all(i -> i in adj[inds[4]], (inds[1], inds[2], inds[3]))
    return central_first || central_last
end

function build_charmm_impropers(tables::CHARMMImproperTables{T, E}, residue_atoms, adj,
                                atom_type_of, type_to_class) where {T, E}
    periodic = Tuple{NTuple{4, Int}, NTuple{4, String}, PeriodicTorsionType{T, E}}[]
    harmonic = Tuple{NTuple{4, Int}, NTuple{4, String}, HarmonicTorsionType{T, E}}[]
    # Chains and residues in order so that the interaction lists are reproducible
    for chain in sort(collect(keys(residue_atoms)))
        chain_atoms = residue_atoms[chain]
        for res_id in sort(collect(keys(chain_atoms)))
            rec = chain_atoms[res_id]
            for improper in charmm_improper_list(tables, rec.res_name, rec.patch)
                inds = charmm_improper_inds(improper, chain_atoms, res_id)
                isnothing(inds) && continue
                valid_charmm_improper(inds, adj) || continue
                classes = ntuple(i -> type_to_class[atom_type_of[inds[i]]], 4)
                per = charmm_improper_params(tables.periodic_types, classes)
                har = charmm_improper_params(tables.harmonic_types, classes)
                if !isnothing(per) && !isnothing(har)
                    throw(ForceFieldXMLError("atom classes $(join(classes, ", ")) match both a " *
                        "periodic and a harmonic CHARMM improper"))
                elseif !isnothing(per)
                    push!(periodic, (inds, classes, per))
                elseif !isnothing(har)
                    push!(harmonic, (inds, classes, har))
                else
                    throw(ForceFieldXMLError("no CHARMM improper parameters for atom classes " *
                        "$(join(classes, ", ")), listed by residue $(rec.res_name)"))
                end
            end
        end
    end
    return periodic, harmonic
end
