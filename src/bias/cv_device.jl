# Evaluating a collective variable without leaving the device

#=
The host path gathers the atoms a CV reads, runs cv_gradient on the CPU and scatters the
forces back. On a GPU that is two blocking copies per step, which dominates the cost of a
bias long after the amount of data copied has stopped mattering.

Here the whole bias is a single kernel launch. One workgroup reduces over the CV's items
into local memory, every lane then evaluates the CV and the bias gradient from the result,
and the lanes scatter the forces. Nothing is copied back unless the virial or the energy is
asked for, so an ordinary force step never synchronizes.

A CV joins in by defining a tag type and the functions below; the kernel is shared. The
defaults cover a CV whose value needs no reduction, so a CV over a fixed few atoms only
defines cv_device_finish, cv_device_grad and cv_device_virial.

One workgroup is deliberate. A CV reads few atoms by definition, so a second kernel to
finish a multi-group reduction would cost more than it saves. A CV over more than roughly
1e5 items would want a two stage reduction instead.
=#

@public cv_has_device_kernel

# A CV type is not itself passable to a kernel, since its index vectors and Symbol fields
#   are not isbits, so the host translates it into a singleton tag plus device arrays
struct DevDist{DT, MIC} end # DT is :single, :min, :max or :cm; MIC uses the minimum image
struct DevAngle end
struct DevTorsion end

# Whether a CV can be evaluated entirely on the device
cv_has_device_kernel(cv) = false
cv_has_device_kernel(cv::Union{CalcDist, CalcAngle, CalcTorsion}) = true

dev_dist_kind(::CalcSingleDist) = :single
dev_dist_kind(::CalcMinDist   ) = :min
dev_dist_kind(::CalcMaxDist   ) = :max
dev_dist_kind(::CalcCMDist    ) = :cm

cv_device_tag(cv::CalcDist) = DevDist{dev_dist_kind(cv.dist_type),
                                      cv.dist_type.calc_type == :closest}()
cv_device_tag(cv::CalcAngle  ) = DevAngle()
cv_device_tag(cv::CalcTorsion) = DevTorsion()

# The unit of the CV value, which the kernel reduces to a plain number so that it can be
#   stored in local memory and in the output buffer
cv_device_unit(::DevDist, ::Type{C}) where C = oneunit(C)
cv_device_unit(::Union{DevAngle, DevTorsion}, ::Type{C}) where C = one(typeof(ustrip(oneunit(C))))

# What a kernel reads, assembled inside the kernel from the arguments so that it holds
#   device pointers rather than host arrays
struct CVCtx{C, I, E, B, D}
    coords::C
    inds::I   # System index of each selected atom
    extras::E # Per-CV device arrays, see cv_device_extras
    boundary::B
    n_sel::Int
end

@inline function CVCtx(coords, inds, extras, boundary, n_sel, ::Val{D}) where D
    return CVCtx{typeof(coords), typeof(inds), typeof(extras),
                 typeof(boundary), D}(coords, inds, extras, boundary, n_sel)
end

@inline dev_ndims(::CVCtx{C, I, E, B, D}) where {C, I, E, B, D} = D
@inline dev_float_type(ctx::CVCtx) = typeof(ustrip(oneunit(eltype(eltype(ctx.coords)))))

# The coordinate of selected atom k. correction=:pbc needs the whole system made whole,
#   which is host work, so a CV that asks for it stays on the host path
@inline cv_device_coord(ctx::CVCtx, k) = @inbounds ctx.coords[ctx.inds[k]]

# The protocol. A partial is (NTuple{P, T}, Int32), the tuple holding unitless numbers and
#   the index identifying a winning item, so that local memory stays on primitive types.

# Number of items the reduction runs over
cv_device_n_items(tag, ctx) = 1
# Width of the unitless part of a partial
cv_device_np(tag, ::Val{D}) where D = Val(1)
# Neutral element of the reduction
cv_device_init(tag, ::Val{P}, ::Type{T}) where {P, T} = (ntuple(_ -> zero(T), Val(P)), Int32(0))
# Partial for item i
cv_device_item(tag, ctx, i, np::Val, T::Type) = cv_device_init(tag, np, T)
# Associative combination of two partials
cv_device_combine(tag, a, b) = a
# (unitless CV value, auxiliary isbits state the gradient and the virial need) from the
#   finished reduction
function cv_device_finish end
# Gradient of the CV with respect to the coordinates of selected atom k
function cv_device_grad end
#=
The virial of the bias, built from the vectors cv_device_finish kept rather than from the
offsets of the individual atoms. Summing (r_k - r_ref) x f_k over the atoms would be equal
by translational invariance, but only where minimum image vectors add, which is exactly
what fails once a group separation approaches half the box; the pair that defines a maximum
distance is normally that far apart. Working from the CV's own vectors is both exact and
cheaper, since one lane writes the result and no atomics are needed.
=#
function cv_device_virial end

# Distances between two atoms or two groups of atoms

# Item i is the pair (a, b) in row major order, which is the loop order of cv_gradient on
#   the host, so a tie between equal distances resolves to the same pair
@inline function dev_pair(ctx, i)
    g1, g2 = ctx.extras
    n2 = length(g2)
    @inbounds return g1[(i - 1) ÷ n2 + 1], g2[(i - 1) % n2 + 1]
end

@inline function dev_dr(::DevDist{DT, MIC}, ctx, k1, k2) where {DT, MIC}
    r1 = cv_device_coord(ctx, k1)
    r2 = cv_device_coord(ctx, k2)
    return MIC ? vector(r1, r2, ctx.boundary) : r2 - r1
end

# CalcSingleDist reads only the first atom of each group, as cv_gradient does
cv_device_n_items(::DevDist{:single}, ctx) = 1
cv_device_n_items(::DevDist{:min}, ctx) = length(ctx.extras[1]) * length(ctx.extras[2])
cv_device_n_items(::DevDist{:max}, ctx) = length(ctx.extras[1]) * length(ctx.extras[2])

cv_device_init(::DevDist{:single}, ::Val{P}, ::Type{T}) where {P, T} = ((typemax(T),), Int32(0))
cv_device_init(::DevDist{:min   }, ::Val{P}, ::Type{T}) where {P, T} = ((typemax(T),), Int32(0))
cv_device_init(::DevDist{:max   }, ::Val{P}, ::Type{T}) where {P, T} = ((typemin(T),), Int32(0))

cv_device_combine(::DevDist{:single}, a, b) = a[1][1] <= b[1][1] ? a : b
cv_device_combine(::DevDist{:min   }, a, b) = a[1][1] <= b[1][1] ? a : b
cv_device_combine(::DevDist{:max   }, a, b) = a[1][1] >= b[1][1] ? a : b

@inline function cv_device_item(tag::DevDist{DT}, ctx, i, ::Val{P}, ::Type{T}) where {DT, P, T}
    k1, k2 = dev_pair(ctx, i)
    dr = dev_dr(tag, ctx, k1, k2)
    return ((T(ustrip(sum(abs2, dr))),), Int32(i))
end

@inline function cv_device_finish(tag::DevDist{DT}, ctx, p) where DT
    k1, k2 = dev_pair(ctx, p[2])
    dr = dev_dr(tag, ctx, k1, k2)
    d = norm(dr)
    # At zero distance dr is zero, so dividing by a unit gives the same dimensionless zero
    #   the host gradient uses without branching into a different type
    dir = d > zero(d) ? dr / d : dr / oneunit(d)
    return (ustrip(d), (dir, Int32(k1), Int32(k2)))
end

@inline function cv_device_grad(::DevDist{DT}, ctx, k, cv_value, aux) where DT
    dir, k1, k2 = aux
    g = zero(dir)
    k == k1 && (g -= dir)
    k == k2 && (g += dir)
    return g
end

@inline cv_device_virial(::DevDist, ctx, cv_value, aux, d_bias) =
    -(d_bias * cv_value) * (aux[1] * transpose(aux[1]))

# Distance between the centers of mass of two groups. The partial carries both weighted
#   center sums, and the per-atom mass fractions are precomputed so that an atom shared by
#   the two groups still needs only one write

cv_device_np(::DevDist{:cm}, ::Val{D}) where D = Val(2D)
cv_device_n_items(::DevDist{:cm}, ctx) = ctx.n_sel
cv_device_combine(::DevDist{:cm}, a, b) = (map(+, a[1], b[1]), Int32(0))

@inline function cv_device_item(::DevDist{:cm}, ctx, i, ::Val{P}, ::Type{T}) where {P, T}
    w1, w2 = ctx.extras
    r = cv_device_coord(ctx, i)
    D = P ÷ 2
    @inbounds v = ntuple(q -> q <= D ? T(w1[i] * ustrip(r[q])) : T(w2[i] * ustrip(r[q - D])),
                         Val(P))
    return (v, Int32(0))
end

@inline function cv_device_finish(tag::DevDist{:cm, MIC}, ctx, p) where MIC
    D = dev_ndims(ctx)
    u = oneunit(eltype(eltype(ctx.coords)))
    com1 = SVector(ntuple(q -> p[1][q]     * u, Val(D)))
    com2 = SVector(ntuple(q -> p[1][q + D] * u, Val(D)))
    r_12 = MIC ? vector(com1, com2, ctx.boundary) : com2 - com1
    d = norm(r_12)
    dir = d > zero(d) ? r_12 / d : r_12 / oneunit(d)
    return (ustrip(d), (dir,))
end

@inline function cv_device_grad(::DevDist{:cm}, ctx, k, cv_value, aux)
    w1, w2 = ctx.extras
    @inbounds return (w2[k] - w1[k]) * aux[1]
end

# Angle between three atoms

@inline function cv_device_finish(::DevAngle, ctx, p)
    g = ctx.extras[1]
    @inbounds ri, rj, rk = cv_device_coord(ctx, g[1]), cv_device_coord(ctx, g[2]),
                           cv_device_coord(ctx, g[3])
    ba = vector_pad3D(rj, ri, ctx.boundary)
    bc = vector_pad3D(rj, rk, ctx.boundary)
    cross_ba_bc = ba × bc
    theta = bond_angle(ri, rj, rk, ctx.boundary)
    z = zero(ri) / oneunit(eltype(ri))^2
    r_ji, r_jk = trim3D(ba, ctx.boundary), trim3D(bc, ctx.boundary)
    iszero_value(cross_ba_bc) && return (theta, (z, z, z, r_ji, r_jk))
    pa = normalize(trim3D( ba × cross_ba_bc, ctx.boundary))
    pc = normalize(trim3D(-bc × cross_ba_bc, ctx.boundary))
    grad_i = pa / norm(ba)
    grad_k = pc / norm(bc)
    return (theta, (grad_i, -(grad_i + grad_k), grad_k, r_ji, r_jk))
end

@inline function cv_device_virial(::DevAngle, ctx, cv_value, aux, d_bias)
    return aux[4] * transpose(-(d_bias * aux[1])) + aux[5] * transpose(-(d_bias * aux[3]))
end

@inline function cv_device_grad(::DevAngle, ctx, k, cv_value, aux)
    g = ctx.extras[1]
    out = zero(aux[1])
    @inbounds begin
        g[1] == k && (out += aux[1])
        g[2] == k && (out += aux[2])
        g[3] == k && (out += aux[3])
    end
    return out
end

# Torsion between four atoms

@inline dev_pos_finite(x) = isfinite(ustrip(x)) && x > zero(x)

@inline function cv_device_finish(::DevTorsion, ctx, p)
    g = ctx.extras[1]
    @inbounds ri, rj = cv_device_coord(ctx, g[1]), cv_device_coord(ctx, g[2])
    @inbounds rk, rl = cv_device_coord(ctx, g[3]), cv_device_coord(ctx, g[4])
    bnd = ctx.boundary
    b1 = vector(ri, rj, bnd)
    b2 = vector(rj, rk, bnd)
    b3 = vector(rk, rl, bnd)
    phi = torsion_angle(ri, rj, rk, rl, bnd)
    b1_norm, b2_norm, b3_norm = norm(b1), norm(b2), norm(b3)
    z = zero(ri) / oneunit(eltype(ri))^2
    r_ji, r_jk, r_jl = vector(rj, ri, bnd), vector(rj, rk, bnd), vector(rj, rl, bnd)
    # check_torsion_bond_norm throws on the host, which a kernel cannot do, so a degenerate
    #   bond returns a non-finite CV instead and is reported when the flag is next read
    if !(dev_pos_finite(b1_norm) && dev_pos_finite(b2_norm) && dev_pos_finite(b3_norm))
        return (oftype(phi, NaN), (z, z, z, z, r_ji, r_jk, r_jl))
    end

    m = b1 × b2
    n = b2 × b3
    tol = dev_float_type(ctx)(@inbounds ctx.extras[2][1])
    norm_floor = tol * max(b1_norm, max(b2_norm, b3_norm))
    b1_norm_eff = max(b1_norm, norm_floor)
    b2_norm_eff = max(b2_norm, norm_floor)
    b3_norm_eff = max(b3_norm, norm_floor)
    m_sq_eff = max(sum(abs2, m), (tol * b1_norm_eff * b2_norm_eff)^2)
    n_sq_eff = max(sum(abs2, n), (tol * b2_norm_eff * b3_norm_eff)^2)
    b2_sq_eff = max(b2_norm^2, b2_norm_eff^2)

    grad_i =  (b2_norm_eff / m_sq_eff) * m
    grad_l = -(b2_norm_eff / n_sq_eff) * n
    c1 = dot(b1, b2) / b2_sq_eff
    c3 = dot(b3, b2) / b2_sq_eff
    grad_j = -(1 + c1) * grad_i + c3 * grad_l
    grad_k = c1 * grad_i - (1 + c3) * grad_l
    return (phi, (-grad_i, -grad_j, -grad_k, -grad_l, r_ji, r_jk, r_jl))
end

@inline function cv_device_virial(::DevTorsion, ctx, cv_value, aux, d_bias)
    return aux[5] * transpose(-(d_bias * aux[1])) +
           aux[6] * transpose(-(d_bias * aux[3])) +
           aux[7] * transpose(-(d_bias * aux[4]))
end

@inline function cv_device_grad(::DevTorsion, ctx, k, cv_value, aux)
    g = ctx.extras[1]
    out = zero(aux[1])
    @inbounds begin
        g[1] == k && (out += aux[1])
        g[2] == k && (out += aux[2])
        g[3] == k && (out += aux[3])
        g[4] == k && (out += aux[4])
    end
    return out
end

# The partial of lane j, read and written with the index as an argument so that no closure
#   captures a variable the kernel reassigns, and with the component loop unrolled so that
#   the tuple is never indexed at run time
@inline dev_load(vals, idx, j, ::Val{P}) where P =
    (ntuple(q -> (@inbounds vals[q, j]), Val(P)), (@inbounds idx[j]))

@inline function dev_store!(vals, idx, j, p, ::Val{P}) where P
    ntuple(q -> (@inbounds vals[q, j] = p[1][q]), Val(P))
    @inbounds idx[j] = p[2]
    return nothing
end

#=
The bias in one launch.

Like the cell list search kernel this cannot run on the KernelAbstractions CPU backend,
which splits a kernel at each @synchronize and so does not carry per-lane state across the
barriers. It is only ever launched on a GPU array.
=#
@kernel inbounds=true function bias_kernel!(fs, vir, cv_out, flag, @Const(coords),
                        @Const(inds), extras, boundary, tag, bias_type,
                        n_sel, ::Val{gs}, ::Val{P}, ::Val{T}, ::Val{D},
                        ::Val{needs_vir}, ::Val{apply}) where {gs, P, T, D, needs_vir, apply}
    lane = @index(Local, Linear)
    ctx = CVCtx(coords, inds, extras, boundary, n_sel, Val(D))

    part_vals = @localmem T (P, gs)
    part_idx = @localmem Int32 (gs,)

    acc = cv_device_init(tag, Val(P), T)
    n_items = cv_device_n_items(tag, ctx)
    item_i = lane
    while item_i <= n_items
        acc = cv_device_combine(tag, acc, cv_device_item(tag, ctx, item_i, Val(P), T))
        item_i += gs
    end
    dev_store!(part_vals, part_idx, lane, acc, Val(P))
    @synchronize

    # The loop bound is the same for every lane in the group, so the barrier is reached by
    #   all of them
    stride = gs >> 1
    while stride > 0
        if lane <= stride
            merged = cv_device_combine(tag, dev_load(part_vals, part_idx, lane, Val(P)),
                                       dev_load(part_vals, part_idx, lane + stride, Val(P)))
            dev_store!(part_vals, part_idx, lane, merged, Val(P))
        end
        @synchronize
        stride >>= 1
    end

    # Every lane repeats the finish rather than broadcasting it through local memory, which
    #   would cost another barrier for a handful of arithmetic
    raw_cv, aux = cv_device_finish(tag, ctx, dev_load(part_vals, part_idx, 1, Val(P)))
    cv_value = raw_cv * cv_device_unit(tag, eltype(eltype(coords)))

    if lane == 1
        cv_out[1] = raw_cv
        isfinite(raw_cv) || (flag[1] = Int32(1))
    end

    if apply
        d_bias = bias_gradient(bias_type, cv_value)
        if needs_vir && lane == 1
            vir_cv = cv_device_virial(tag, ctx, cv_value, aux, d_bias)
            for dim in 1:D, alpha in 1:D
                vir[dim, alpha] = ustrip(vir_cv[dim, alpha])
            end
        end
        k = lane
        while k <= n_sel
            f = -(d_bias * cv_device_grad(tag, ctx, k, cv_value, aux))
            # The selected indices are unique, so no atom is written twice
            fs[inds[k]] += f
            bias_all_finite(f) || (flag[1] = Int32(1))
            k += gs
        end
    end
end

gpu_threads_bias() = prevpow(2, gpu_threads_env("MOLLY_GPUNTHREADS_BIAS", 256))

# Device arrays the kernel reads for a given CV, indexed into the selection
function cv_device_extras(cv::CalcDist, cv_local, n_sel, atoms, ::Type{T}, at) where T
    if cv.dist_type isa CalcCMDist
        return (to_device(dev_mass_fractions(cv_local.atom_inds_1, atoms, n_sel, T), at),
                to_device(dev_mass_fractions(cv_local.atom_inds_2, atoms, n_sel, T), at))
    end
    return (to_device(Int32.(cv_local.atom_inds_1), at),
            to_device(Int32.(cv_local.atom_inds_2), at))
end

cv_device_extras(cv::CalcAngle, cv_local, n_sel, atoms, ::Type{T}, at) where T =
    (to_device(Int32.(cv_local.atom_inds), at),)

cv_device_extras(cv::CalcTorsion, cv_local, n_sel, atoms, ::Type{T}, at) where T =
    (to_device(Int32.(cv_local.atom_inds), at),
     to_device([T(cv.gradient_singularity_tol)], at))

# m_k / M summed over every appearance of atom k in the group, which is the coefficient
#   cv_gradient accumulates for CalcCMDist
function dev_mass_fractions(inds_local, atoms, n_sel, ::Type{T}) where T
    ms = mass.(atoms[inds_local])
    M = sum(ms)
    w = zeros(T, n_sel)
    for (idx, k) in enumerate(inds_local)
        w[k] += T(ustrip(ms[idx] / M))
    end
    return w
end

# Items reduced over, used only to size the workgroup
dev_host_n_items(cv::CalcDist) = length(cv.atom_inds_1) * length(cv.atom_inds_2)
dev_host_n_items(cv) = 1

struct CVDeviceWorkspace{TG, E, V, CO, FL, NP, GS, BE}
    tag::TG
    extras::E
    vir::V      # Unitless D x D virial of this bias, its own since the shared device
                #   buffer is drained before general interactions are evaluated
    cv_out::CO  # The CV value, read when the energy is asked for
    flag::FL    # Set by the kernel when a value is not finite, read lazily
    np::NP
    gs::GS
    backend::BE
end

# The device workspace for a CV, or nothing when the CV stays on the host path
function cv_device_workspace(cv, sys, inds, cv_local, atoms)
    (sys.coords isa AbstractGPUArray && cv_has_device_kernel(cv)) || return nothing
    # :pbc needs the whole system made whole, which is host work by nature
    cv.correction == :pbc && return nothing

    at = array_type(sys.coords)
    T = typeof(ustrip(oneunit(eltype(eltype(sys.coords)))))
    D = length(eltype(sys.coords))
    n_sel = length(inds)
    tag = cv_device_tag(cv)
    backend = get_backend(sys.coords)
    gs = clamp(nextpow(2, max(n_sel, dev_host_n_items(cv), 1)), 1, gpu_threads_bias())
    return CVDeviceWorkspace(
        tag,
        cv_device_extras(cv, cv_local, n_sel, atoms, T, at),
        KernelAbstractions.zeros(backend, T, D, D),
        KernelAbstractions.zeros(backend, T, 1),
        KernelAbstractions.zeros(backend, Int32, 1),
        cv_device_np(tag, Val(D)),
        Val(gs),
        backend,
    )
end

# The device path is taken when the CV has a workspace and the bias itself can be passed to
#   a kernel, which rules out a bias holding host arrays such as Metadynamics
cv_device_ready(sel, bias) = !isnothing(sel.device) && isbitstype(typeof(bias.bias_type))

function launch_bias_kernel!(fs, sys, bias, sel, ws::CVDeviceWorkspace{TG, E, V, CO, FL,
                             Val{P}, Val{gs}}, ::Val{needs_vir},
                             ::Val{apply}) where {TG, E, V, CO, FL, P, gs, needs_vir, apply}
    T = typeof(ustrip(oneunit(eltype(eltype(sys.coords)))))
    D = length(eltype(sys.coords))
    kernel! = bias_kernel!(ws.backend, gs)
    kernel!(fs, ws.vir, ws.cv_out, ws.flag, sys.coords, sel.inds_dev, ws.extras,
            sys.boundary, ws.tag, bias.bias_type, length(sel.inds), Val(gs), Val(P), Val(T),
            Val(D), Val(needs_vir), Val(apply); ndrange=gs)
    return nothing
end

# Errors if the kernel has flagged a non-finite value since the flag was last read. Only
#   called where the host is already waiting on the device, so a force step stays free of
#   synchronization; a blown up bias is reported at the next energy or virial evaluation
function bias_check_flag(bias, ws::CVDeviceWorkspace; cv_sim=nothing)
    iszero(from_device(ws.flag)[1]) && return nothing
    fill!(ws.flag, Int32(0))
    msg = "$(bias_description(bias)) produced a non-finite collective variable or force"
    isnothing(cv_sim) || (msg *= ", cv_sim=$(cv_sim)")
    error(msg)
end

function bias_forces_device!(fs, sys, bias, sel, ws; needs_vir::Bool, buffers)
    want_vir = needs_vir && bias.cv_type.has_virial
    if want_vir
        launch_bias_kernel!(fs, sys, bias, sel, ws, Val(true), Val(true))
        buffers.virial .+= from_device(ws.vir) .* sys.energy_units
        bias_check_flag(bias, ws)
    else
        launch_bias_kernel!(fs, sys, bias, sel, ws, Val(false), Val(true))
    end
    return fs
end

function bias_energy_device(sys, bias, sel, ws; kwargs...)
    launch_bias_kernel!(nothing, sys, bias, sel, ws, Val(false), Val(false))
    cv_sim = from_device(ws.cv_out)[1] *
             cv_device_unit(ws.tag, eltype(eltype(sys.coords)))
    check_bias_finite(cv_sim, "collective variable", bias)
    # Reported here rather than every step, so a force step needs no synchronization
    bias_check_flag(bias, ws; cv_sim=cv_sim)
    pe = potential_energy(bias.bias_type, cv_sim; kwargs...)
    return check_bias_finite(pe, "potential energy", bias; cv_sim=cv_sim)
end
