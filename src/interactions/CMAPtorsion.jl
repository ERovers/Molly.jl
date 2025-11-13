export CMAPTorsion

"""
    CMAPTorsion(; k, θ0)

CMAP torsion correction on 8 atoms.

Only compatible with 3D systems.
"""
struct CMAPTorsion{I, C}
    size::I
    COEFF::Vector{C, C}
end

CMAPTorsion(; k, θ0) = CMAPTorsion{typeof(k),typeof(θ0)}(k, θ0)

@inline function force(d::CMAPTorsion, coords_i, coords_j, coords_k, coords_l, 
                            coords_m, coords_n, coords_o, coords_p, boundary, args...)
    F = typeof(ustrip(coords_i[1]))
    # First angle
    v0a = vector(coords_i, coords_j, boundary)
    v1a = vector(coords_k, coords_j, boundary)
    v2a = vector(coords_k, coords_l, boundary)
    cp0a = v0a × v1a
    cp1a = v1a × v2a
    cosangle = dot(norm(cp0a), norm(cp1a))
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
    angleA = (dot(v0a,cp1a)>=0) ? angleA : -angleA
    angleA = fmod(angleA + F(2.0)*pi, F(2.0)*pi)

    # Second angle
    v0b = vector(coords_m, coords_n, boundary)
    v1b = vector(coords_o, coords_n, boundary)
    v2b = vector(coords_o, coords_p, boundary)
    cp0a = v0b × v1b
    cp1a = v1b × v2b
    cosangle = dot(norm(cp0b), norm(cp1b))
    if cosangle > F(0.99) || cosangle < F(-0.99)
        cross_prod = cp0b × cp1b
        scale = dot(cp0b,cp0b) * dot(cp1b,cp1b)
        angleB = asin(sqrt(dot(cross_prod,cross_prod)/scale))
        if cosangle < F(0.0)
            angleB = pi - angleB
        end
    else
        angleB = acos(cosangle)
    end
    angleB = (dot(v0b,cp1b)>=0) ? angleB : -angleB
    angleB = fmod(angleB + F(2.0)*pi, F(2.0)*pi)

    # 




    # Energy
    
    dEdθ = d.k * (θ - d.θ0) + d.k * (θ - d.θ0)
    fi =  dEdθ * bc_norm * cross_ab_bc / dot(cross_ab_bc, cross_ab_bc)
    fl = -dEdθ * bc_norm * cross_bc_cd / dot(cross_bc_cd, cross_bc_cd)
    v = (dot(-ab, bc) / bc_norm^2) * fi - (dot(-cd, bc) / bc_norm^2) * fl
    fj =  v - fi
    fk = -v - fl
    return SpecificForce4Atoms(fi, fj, fk, fl)
end

@inline function potential_energy(d::CMAPTorsion, coords_i, coords_j, coords_k, coords_l, 
                            coords_m, coords_n, coords_o, coords_p, boundary, args...)
    
    # First angle
    v0a = vector(coords_i, coords_j, boundary)
    v1a = vector(coords_k, coords_j, boundary)
    v2a = vector(coords_k, coords_l, boundary)
    cp0a = v0a × v1a
    cp1a = v1a × v2a
    cosangle = dot(norm(cp0a), norm(cp1a))
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
    angleA = (dot(v0a,cp1a)>=0) ? angleA : -angleA
    angleA = fmod(angleA + F(2.0)*pi, F(2.0)*pi)

    # Second angle
    v0b = vector(coords_m, coords_n, boundary)
    v1b = vector(coords_o, coords_n, boundary)
    v2b = vector(coords_o, coords_p, boundary)
    cp0a = v0b × v1b
    cp1a = v1b × v2b
    cosangle = dot(norm(cp0b), norm(cp1b))
    if cosangle > F(0.99) || cosangle < F(-0.99)
        cross_prod = cp0b × cp1b
        scale = dot(cp0b,cp0b) * dot(cp1b,cp1b)
        angleB = asin(sqrt(dot(cross_prod,cross_prod)/scale))
        if cosangle < F(0.0)
            angleB = pi - angleB
        end
    else
        angleB = acos(cosangle)
    end
    angleB = (dot(v0b,cp1b)>=0) ? angleB : -angleB
    angleB = fmod(angleB + F(2.0)*pi, F(2.0)*pi)

    # Identify Patch
    delta = 2*pi / d.size
    s = min(angleA/delta, d.size-1)
    t = min(angleB/delta, d.size-1)
    idx = 4*(s+d.size*t)
    c0 = d.COEFF[idx]
    c1 = d.COEFF[idx+1]
    c2 = d.COEFF[idx+2]
    c3 = d.COEFF[idx+3]
    da = angleA/delta - s
    db = angleB/delta - t

    # Spline with coefficients
    energy = ((c3[4]*db + c3[3])*db + c3[2])*db + c3[1]
    energy += ((c2[4]*db + c2[3])*db + c2[2])*db + c2[1]
    energy += ((c1[4]*db + c1[3])*db + c1[2])*db + c1[1]
    energy += ((c0[4]*db + c0[3])*db + c0[2])*db + c0[1]
    
    return energy
end
