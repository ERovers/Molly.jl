export CMAPTorsion

"""
    CMAPTorsion(; k, θ0)

CMAP torsion correction on 8 atoms.

Only compatible with 3D systems.
"""
@kwdef struct CMAPTorsion{S,AT}
    size::S
    coeff::AT
end

Base.zero(::CMAPTorsion) = CMAPTorsion(size=0, coeff=SMatrix(undef, 0))

# Setup functions (based on CMAPTorsionForceImpl.cpp in OpenMM)
@inline function calc_coefficients(size, energy::Vector{E}) where E
    c = calcMapDerivatives(size, energy)
    coeffMatrix = Matrix(undef, size*size*4, 4)
    
    for j in 1:(size*size)
        coeffMatrix[(j-1)*4+1, :] .= c[j, 1:4]
        coeffMatrix[(j-1)*4+2, :] .= c[j, 5:8]
        coeffMatrix[(j-1)*4+3, :] .= c[j, 9:12]
        coeffMatrix[(j-1)*4+4, :] .= c[j, 13:16]
    end
    return SMatrix{size*size*4, 4, E}(coeffMatrix)
end

@inline function calcMapDerivatives(size, energy::Vector{E}) where E
    tp = eltype(energy)
    x     = [(i * 2 * π / size) for i in 0:size]
    y     = zeros(tp, size+1)
    deriv = zeros(tp, size+1)
    d1    = zeros(tp, size*size)
    d2    = zeros(tp, size*size)
    d12   = zeros(tp, size*size)
    
    for i in 1:size
        for j in 1:size
            y[j] = energy[j+size*(i-1)] 
        end
        y[size+1] = energy[size*(i-1)+1]  
        deriv = createPeriodicSpline(x,y,deriv)
        for j in 1:size
            d1[j+size*(i-1)] = evaluateSplineDerivative(x, y, deriv, x[j])
        end
    end
    
    for i in 1:size
        for j in 1:size
            y[j] = energy[i+size*(j-1)]  
        end
        y[size+1] = energy[i]  
        
        deriv = createPeriodicSpline(x,y,deriv)
        for j in 1:size
            d2[i+size*(j - 1)] = evaluateSplineDerivative(x, y, deriv, x[j])
        end
    end
    
    for i in 1:size
        for j in 1:size
            y[j] = d2[j+size*(i-1)]
        end
        y[size+1] = d2[size*(i-1)+1]
        deriv = createPeriodicSpline(x,y,deriv)
        for j in 1:size
            d12[j+size*(i-1)] = evaluateSplineDerivative(x, y, deriv, x[j])
        end
    end
    
    wt = [
            1, 0, -3, 2, 0, 0, 0, 0, -3, 0, 9, -6, 2, 0, -6, 4,
            0, 0, 0, 0, 0, 0, 0, 0, 3, 0, -9, 6, -2, 0, 6, -4,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9, -6, 0, 0, -6, 4,
            0, 0, 3, -2, 0, 0, 0, 0, 0, 0, -9, 6, 0, 0, 6, -4,
            0, 0, 0, 0, 1, 0, -3, 2, -2, 0, 6, -4, 1, 0, -3, 2,
            0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 3, -2, 1, 0, -3, 2,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -3, 2, 0, 0, 3, -2,
            0, 0, 0, 0, 0, 0, 3, -2, 0, 0, -6, 4, 0, 0, 3, -2,
            0, 1, -2, 1, 0, 0, 0, 0, 0, -3, 6, -3, 0, 2, -4, 2,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 3, -6, 3, 0, -2, 4, -2,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -3, 3, 0, 0, 2, -2,
            0, 0, -1, 1, 0, 0, 0, 0, 0, 0, 3, -3, 0, 0, -2, 2,
            0, 0, 0, 0, 0, 1, -2, 1, 0, -2, 4, -2, 0, 1, -2, 1,
            0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 2, -1, 0, 1, -2, 1,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0, 0, -1, 1,
            0, 0, 0, 0, 0, 0, -1, 1, 0, 0, 2, -2, 0, 0, -1, 1
        ]
    
    rhs = zeros(tp, 16)
    delta = 2 * π / size
    c = zeros(tp, size * size, 16)

    for i in 1:size
        for j in 1:size
            nexti = (i % size) + 1
            nextj = (j % size) + 1
            e = [energy[i+size*(j-1)],energy[nexti+size*(j-1)], energy[nexti+size*(nextj-1)], energy[i+size*(nextj-1)]]
            e1 = [d1[i+size*(j-1)], d1[nexti+size*(j-1)], d1[nexti+size*(nextj-1)], d1[i+size*(nextj-1)]]
            e2 = [d2[i+size*(j-1)], d2[nexti+size*(j-1)], d2[nexti+size*(nextj-1)], d2[i+size*(nextj-1)]]
            e12 = [d12[i+size*(j-1)], d12[nexti+size*(j-1)], d12[nexti+size*(nextj-1)], d12[i+size*(nextj-1)]]
    
            for k in 1:4
                rhs[k] = e[k]
                rhs[k+4] = e1[k]*delta
                rhs[k+8] = e2[k]*delta
                rhs[k+12] = e12[k]*delta*delta
            end
            for k in 1:16
                sum = 0.0u"kJ * mol^-1"
                for m in 1:16
                    sum += wt[k+16*(m-1)]*rhs[m]
                end
                c[i+size*(j-1),k] = sum
            end
        end
    end
    return c
end

@inline function createPeriodicSpline(x,y,deriv)
    n = length(x)
    if length(y) != n
        # throw error
        return
    end
    if (n < 3)
        # throw error
        return
    end
    if y[1]!=y[end]
        # throw error
        return
    end
    tp1 = eltype(x)
    tp2 = eltype(deriv)
    # Create the system of equations to solve
    a   = zeros(tp1,n-1)
    b   = zeros(tp1,n-1)
    c   = zeros(tp1,n-1)
    rhs = zeros(tp2,n-1)
    a[1] = x[n]-x[n-1]
    b[1] = 2*(x[2]-x[1]+x[n]-x[n-1])
    c0 = x[2]-x[1]
    rhs[1] =  6*((y[2]-y[1])/(x[2]-x[1]) - (y[n]-y[n-1])/(x[n]-x[n-1]))
    for i in 2:n-1
            a[i] = x[i]-x[i-1]
            b[i] = 2*(x[i+1]-x[i-1])
            c[i] = x[i+1]-x[i]
            rhs[i] = 6*((y[i+1]-y[i])/(x[i+1]-x[i]) - (y[i]-y[i-1])/(x[i]-x[i-1]))
    end
    beta = a[1]
    alpha = c[n-1]
    gamma = -b[1]

    # This is a cyclic tridiagonal matrix. We solve it using the Sherman-Morrison method,
    # which involves solving two tridiagonal systems

    n-=1
    b[1] -= gamma
    b[n] -= alpha*beta/gamma
    deriv = solveTridiagonalMatrix(a, b, c, rhs, deriv)  
    u = zeros(n)
    z = zeros(n)
    u[1] = gamma
    u[n] = alpha
    z = solveTridiagonalMatrix(a,b,c,u,z)
    scale = (deriv[1]+beta*deriv[n]/gamma)/(1+z[1]+beta*z[n]/gamma)
    for i in 1:n
        deriv[i] -= scale*z[i]
    end
    deriv[n+1] = deriv[1]
    return deriv
end
    
@inline function solveTridiagonalMatrix(a,b,c,rhs,deriv)
    n = length(a)
    gamma = zeros(n)
    
    # Decompose the matrix
    deriv[1] = rhs[1]/b[1]
    beta = b[1]
    for i in 2:n
        gamma[i] = c[i-1]/beta
        beta = b[i]-a[i]*gamma[i]
        deriv[i] = (rhs[i]-a[i]*deriv[i-1])/beta
    end
    
    # Perform backsubstitution
    for i in n-1:-1:1
        deriv[i] -= gamma[i+1]*deriv[i+1]
    end
    return deriv
end

@inline function evaluateSplineDerivative(x, y, deriv, t)
    n = length(x)
    if (t<x[1] || t > x[n])
        println("NO")
    end
    
    lower = 1
    upper = n
    while (upper-lower > 1)
        middle = round(Int,(upper+lower)/2)
        if (x[middle] > t)
            upper = Int(middle)
        else
            lower = Int(middle)
        end
    end

    dx = x[upper] - x[lower]
    a = (x[upper]-t)/dx
    b = one(a)-a
    dadx = -one(dx)/dx
    return dadx*y[lower]-dadx*y[upper] + ((1-3*a*a)*deriv[lower] + ((3*b*b)-1)*deriv[upper])*dx/6
end

@inline function force(d::CMAPTorsion, coords_i, coords_j, coords_k, coords_l, 
                            coords_m, boundary, args...)

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
    angleA = mod(angleA + 2*pi, 2*pi)
    
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
    angleB = mod(angleB + 2*pi, 2*pi)

    # Identify Patch
    delta = 2*pi / d.size
    s = Int(trunc(min(angleA/delta, d.size-1)))
    t = Int(trunc(min(angleB/delta, d.size-1)))
    idx = (4*(s+d.size*t))+1
    c0 = d.coeff[idx,:]
    c1 = d.coeff[idx+1,:]
    c2 = d.coeff[idx+2,:]
    c3 = d.coeff[idx+3,:]
    da = angleA/delta - s
    db = angleB/delta - t

    # Evaluate the spline to determine the energy and gradients.
    dEdA = (3*c3[4]*da + 2*c2[4])*da + c1[4]
    dEdB = (3*c3[4]*db + 2*c3[3])*db + c3[2]
    dEdA = db*dEdA + (3*c3[3]*da + 2*c2[3])*da + c1[3]
    dEdB = da*dEdB + (3*c2[4]*db + 2*c2[3])*db + c2[2]
    dEdA = db*dEdA + (3*c3[2]*da + 2*c2[2])*da + c1[2]
    dEdB = da*dEdB + (3*c1[4]*db + 2*c1[3])*db + c1[2]
    dEdA = db*dEdA + (3*c3[1]*da + 2*c2[1])*da + c1[1]
    dEdB = da*dEdB + (3*c0[4]*db + 2*c0[3])*db + c0[2]
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
    return SpecificForce5Atoms(fi, fj, fk, fl, fm)
end

@inline function potential_energy(d::CMAPTorsion, coords_i, coords_j, coords_k, coords_l, 
                            coords_m, boundary, args...)
    
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
    angleA = mod(angleA + 2*pi, 2*pi)
    
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
    angleB = mod(angleB + 2*pi, 2*pi)

    # Identify Patch
    delta = 2*pi / d.size
    s = Int(trunc(min(angleA/delta, d.size-1)))
    t = Int(trunc(min(angleB/delta, d.size-1)))
    idx = (4*(s+d.size*t))+1
    c0 = d.coeff[idx,:]
    c1 = d.coeff[idx+1,:]
    c2 = d.coeff[idx+2,:]
    c3 = d.coeff[idx+3,:]
    da = angleA/delta - s
    db = angleB/delta - t

    # Spline with coefficients
    energy = ((c3[4]*db + c3[3])*db + c3[2])*db + c3[1]
    energy = da*energy + ((c2[4]*db + c2[3])*db + c2[2])*db + c2[1]
    energy = da*energy + ((c1[4]*db + c1[3])*db + c1[2])*db + c1[1]
    energy = da*energy + ((c0[4]*db + c0[3])*db + c0[2])*db + c0[1]
    return energy
end
