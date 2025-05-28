export
    SHAKE_RATTLE,
    check_position_constraints,
    check_velocity_constraints

"""
    SHAKE_RATTLE(constraints, n_atoms, dist_tolerance, vel_tolerance)

Constrain distances during a simulation using the SHAKE and RATTLE algorithms.

Velocity constraints will be imposed for simulators that integrate velocities such as
[`VelocityVerlet`](@ref).
See [Ryckaert et al. 1977](https://doi.org/10.1016/0021-9991(77)90098-5) for SHAKE,
[Andersen 1983](https://doi.org/10.1016/0021-9991(83)90014-1) for RATTLE and
[Elber et al. 2011](https://doi.org/10.1140%2Fepjst%2Fe2011-01525-9) for a derivation
of the linear system solved to satisfy the RATTLE algorithm.

Not currently compatible with GPU simulation.

# Arguments
- `constraints`: a vector of constraints to be imposed on the system.
- `n_atoms::Integer`: the number of atoms in the system.
- `dist_tolerance`: the tolerance used to end the iterative procedure when calculating
    position constraints, should have the same units as the coordinates.
- `vel_tolerance`: the tolerance used to end the iterative procedure when calculating
    velocity constraints, should have the same units as the velocities * the coordinates.
"""
struct SHAKE_RATTLE{C, D, V}
    clusters::C
    dist_tolerance::D
    vel_tolerance::V
    n_constrainted_atoms::Int
end

function SHAKE_RATTLE(constraints, n_atoms::Integer, dist_tolerance, vel_tolerance)
    clusters = build_clusters(n_atoms, constraints)
    n_constrainted_atoms = sum(num_unique.(clusters))
    return SHAKE_RATTLE{typeof(clusters), typeof(dist_tolerance), typeof(vel_tolerance)}(
            clusters, dist_tolerance, vel_tolerance, n_constrainted_atoms)
end

function apply_position_constraints!(sys, ca::SHAKE_RATTLE, coord_storage;
                                     n_threads::Integer=Threads.nthreads())
    # SHAKE updates
    converged = false

    while !converged
        Threads.@threads for cluster in ca.clusters
            for constraint in cluster.constraints
                k1, k2 = constraint.i, constraint.j

                # Vector between the atoms after unconstrained update (s)
                s12 = vector(sys.coords[k1], sys.coords[k2], sys.boundary)

                # Vector between the atoms before unconstrained update (r)
                r12 = vector(coord_storage[k1], coord_storage[k2], sys.boundary)

                if abs(norm(s12) - constraint.dist) > ca.dist_tolerance
                    m1_inv = inv(mass(sys.atoms[k1]))
                    m2_inv = inv(mass(sys.atoms[k2]))
                    a = (m1_inv + m2_inv)^2 * sum(abs2, r12)
                    b = -2 * (m1_inv + m2_inv) * dot(r12, s12)
                    c = sum(abs2, s12) - (constraint.dist)^2
                    D = b^2 - 4*a*c

                    if ustrip(D) < 0.0
                        @warn "SHAKE determinant negative, setting to 0.0"
                        D = zero(D)
                    end

                    # Quadratic solution for g
                    α1 = (-b + sqrt(D)) / (2*a)
                    α2 = (-b - sqrt(D)) / (2*a)

                    g = abs(α1) <= abs(α2) ? α1 : α2

                    # Update positions
                    δri1 = r12 .* (g*m1_inv)
                    δri2 = r12 .* (-g*m2_inv)

                    sys.coords[k1] += δri1
                    sys.coords[k2] += δri2
                end
            end
        end

        converged = check_position_constraints(sys, ca)
    end
    return sys
end

function apply_velocity_constraints!(sys, ca::SHAKE_RATTLE; n_threads::Integer=Threads.nthreads())
    # RATTLE updates
    converged = false

    while !converged
        Threads.@threads for cluster in ca.clusters
            for constraint in cluster.constraints
                k1, k2 = constraint.i, constraint.j

                inv_m1 = inv(mass(sys.atoms[k1]))
                inv_m2 = inv(mass(sys.atoms[k2]))

                # Vector between the atoms after SHAKE constraint
                r_k1k2 = vector(sys.coords[k1], sys.coords[k2], sys.boundary)

                # Difference between unconstrainted velocities
                v_k1k2 = sys.velocities[k2] .- sys.velocities[k1]

                err = abs(dot(r_k1k2, v_k1k2))
                if err > ca.vel_tolerance
                    # Re-arrange constraint equation to solve for Lagrange multiplier
                    # This has a factor of dt which cancels out in the velocity update
                    λₖ = -dot(r_k1k2, v_k1k2) / (dot(r_k1k2, r_k1k2) * (inv_m1 + inv_m2))

                    # Correct velocities
                    sys.velocities[k1] -= inv_m1 .* λₖ .* r_k1k2
                    sys.velocities[k2] += inv_m2 .* λₖ .* r_k1k2
                end
            end
        end

        converged = check_velocity_constraints(sys, ca)
    end
    return sys
end

"""
    check_position_constraints(sys, constraints)

Checks if the position constraints are satisfied by the current coordinates of `sys`.
"""
function check_position_constraints(sys, ca::SHAKE_RATTLE)
    max_err = typemin(float_type(sys)) * unit(eltype(eltype(sys.coords)))
    for cluster in ca.clusters
        for constraint in cluster.constraints
            dr = vector(sys.coords[constraint.i], sys.coords[constraint.j], sys.boundary)
            err = abs(norm(dr) - constraint.dist)
            if max_err < err
                max_err = err
            end
        end
    end
    return max_err < ca.dist_tolerance
end

"""
    check_velocity_constraints(sys, constraints)

Checks if the velocity constraints are satisfied by the current velocities of `sys`.
"""
function check_velocity_constraints(sys::System, ca::SHAKE_RATTLE)
    max_err = typemin(float_type(sys)) * unit(eltype(eltype(sys.velocities))) * unit(eltype(eltype(sys.coords)))
    for cluster in ca.clusters
        for constraint in cluster.constraints
            dr = vector(sys.coords[constraint.i], sys.coords[constraint.j], sys.boundary)
            v_diff = sys.velocities[constraint.j] .- sys.velocities[constraint.i]
            err = abs(dot(dr, v_diff))
            if max_err < err
                max_err = err
            end
        end
    end
    return max_err < ca.vel_tolerance
end
