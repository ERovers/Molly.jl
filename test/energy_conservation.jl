# Energy conservation test

using Molly
using CUDA

using Test

@testset "Lennard-Jones energy conservation" begin
    function test_energy_conservation(nl::Bool, array_type::AbstractArray, n_threads::Integer, n_steps::Integer)
        n_atoms = 2_000
        atom_mass = 40.0u"g/mol"
        temp = 1.0u"K"
        boundary = CubicBoundary(5.0u"nm")
        simulator = VelocityVerlet(dt=0.001u"ps", remove_CM_motion=false)
    
        atoms = [Atom(mass=atom_mass, charge=0.0, σ=0.05u"nm", ϵ=0.2u"kJ * mol^-1") for i in 1:n_atoms]
        dist_cutoff = 3.0u"nm"
        cutoffs = (
            DistanceCutoff(dist_cutoff),
            ShiftedPotentialCutoff(dist_cutoff),
            ShiftedForceCutoff(dist_cutoff),
            CubicSplineCutoff(dist_cutoff, dist_cutoff + 0.5u"nm"),
        )
    
        for cutoff in cutoffs
            coords = place_atoms(n_atoms, boundary; min_dist=0.1u"nm")
            neighbor_finder = NoNeighborFinder()
            if nl && gpu
                neighbor_finder=GPUNeighborFinder(
                    eligible=CuArray(trues(n_atoms, n_atoms)),
                    n_steps_reorder=10,
                    dist_cutoff=dist_cutoff,
                )
            end
            if nl && !gpu
                neighbor_finder=DistanceNeighborFinder(
                    eligible=trues(n_atoms, n_atoms),
                    n_steps=10,
                    dist_cutoff=dist_cutoff,
                )
            end
    
            sys = System(
                atoms=(array_type(atoms) : atoms),
                coords=(array_type(coords) : coords),
                boundary=boundary,
                pairwise_inters=(LennardJones(cutoff=cutoff, use_neighbors=ifelse(nl, true, false)),),
                neighbor_finder=neighbor_finder,
                loggers=(
                    coords=CoordinatesLogger(100),
                    energy=TotalEnergyLogger(100),
                ),
            )
            random_velocities!(sys, temp)
    
            E0 = total_energy(sys; n_threads=n_threads)
            simulate!(deepcopy(sys), simulator, 20; n_threads=n_threads)
            @time simulate!(sys, simulator, n_steps; n_threads=n_threads)
    
            Es = values(sys.loggers.energy)
            @test isapprox(Es[1], E0; atol=1e-7u"kJ * mol^-1")
    
            max_ΔE = maximum(abs.(Es .- E0))
            platform_str = gpu ? "GPU" : "CPU $n_threads thread(s)"
            cutoff_str = Base.typename(typeof(cutoff)).wrapper
            @info "$platform_str - $cutoff_str - max energy difference $max_ΔE"
            @test max_ΔE < 5e-4u"kJ * mol^-1"
    
            final_coords = last(values(sys.loggers.coords))
            @test all(all(c .> 0.0u"nm") for c in final_coords)
            @test all(all(c .< boundary) for c in final_coords)
        end
    end

    test_energy_conservation(true, Array, 1, 10_000)
    test_energy_conservation(false, Array, 1, 10_000)
    if Threads.nthreads() > 1
        test_energy_conservation(true, Array, Threads.nthreads(), 50_000)
        test_energy_conservation(false, Array, Threads.nthreads(), 50_000)
    end
    for array_type in array_list[2:end]
        test_energy_conservation(true, array_type, 1, 100_000)
        test_energy_conservation(false, array_type, 1, 100_000)
    end
end


