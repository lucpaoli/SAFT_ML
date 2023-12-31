using Clapeyron
import Clapeyron: *, @comps
import Clapeyron: dot, a_assoc, N_A, Solvers, diagvalues, SAFTγMieconsts, SA, SingleComp
import Base: @kwdef

using Zygote, ChainRulesCore

function bmcs_hs(ζ0, ζ1, ζ2, ζ3)
    ζ3m1 = (1 - ζ3)
    ζ3m1² = ζ3m1 * ζ3m1
    res = 1 / ζ0 * (3ζ1 * ζ2 / ζ3m1 + ζ2^3 / (ζ3 * ζ3m1²) + (ζ2^3 / ζ3^2 - ζ0) * Solvers.log1p(-ζ3))
    return res
end

# This could likely be done instead by constructing SingleParam{T1} etc
@kwdef struct SAFTVRMieNNParams{T1<:Real,T2<:Real,T3<:Real,T4<:Real,T5<:Real,T6<:Real,T7<:Real,T8<:Real}
    Mw::Vector{T1}
    segment::Vector{T2}
    sigma::Vector{T3}
    lambda_a::Vector{T4}
    lambda_r::Vector{T5}
    epsilon::Vector{T6}
    epsilon_assoc::Vector{T7} = Float64[]
    bondvol::Vector{T8} = Float64[]
end
# Base.eltype(::Type{SAFTVRMieNNParams{T1,T2,T3,T4,T5,T6,T7,T8}}) where {T1,T2,T3,T4,T5,T6,T7,T8} = T2

@kwdef struct SAFTVRMieNN <: SAFTModel
    params::SAFTVRMieNNParams
    idealmodel::IdealModel = BasicIdeal(["methane"])
    components::Vector{String} = ["methane"]
end

function make_NN_model(Mw, m, σ, λ_a, λ_r, ϵ)
    model = SAFTVRMieNN(
        params=SAFTVRMieNNParams(
            Mw=[Mw],
            segment=[m],
            sigma=[σ * 1e-10],
            lambda_a=[λ_a],
            lambda_r=[λ_r],
            epsilon=[ϵ],
            epsilon_assoc=Float64[],
            bondvol=Float64[],
        )
    )
    return model
end

# Hack to get around Clapeyron model construction API
function make_model(Mw, m, σ, λ_a, λ_r, ϵ)
    model = SAFTVRMie(["methane"])

    model.params.Mw[1] = Mw
    model.params.segment[1] = m
    model.params.sigma[1] = σ * 1e-10
    model.params.lambda_a[1] = λ_a
    model.params.lambda_r[1] = λ_r
    model.params.epsilon[1] = ϵ

    return model
end

#! Not differentiable! 
# function critical_temperature_NN(X)
#     saft_model = make_model(X...)
#     Tc, pc, Vc = crit_pure(saft_model)

#     return Tc
# end

# Critical point solver obj. func defined by:
    # function f!(F,x)
    #     T_c = x[1]*T̄
    #     V_c = exp10(x[2])
    #     ∂²A∂V², ∂³A∂V³ = ∂²³f(model, V_c, T_c, SA[1.0])
    #     F[1] = -∂²A∂V²
    #     F[2] = -∂³A∂V³
    #     return F
    # end
# ∂p∂V = ForwardDiff.derivative(V -> pressure_NN(X, V, T), vL)
# v2 = vL - (pressure_NN(X, vL, T) - p) / ∂p∂V
# Define our Newton step as:
    # Tc2 = Tc - (∂²A∂V²(X, vc, Tc) - ∂²A∂V²) / ∂³A∂V³(X, vc, Tc)
#? This isn't right I don't think, and this is weird to define as a Newton iteration
#? How is Tc determined from ∂²A∂V²? Defining derivative of Tc
#* Temperature is marched within the critical point solver

# obj func T -> -∂²A∂V²
# Tc is T at ∂²A∂V² = 0
# ∂²A∂V² = f(T, Vc)
# Newton step is defined by
# Tc2 = Tc - (∂²A∂V²(X, vc, Tc) - ∂²A∂V²(X, vc, T)) / ∂²A∂V²(X, vc, Tc)

function pressure_NN(X, V, T)
    model = make_NN_model(X...)
    return -ForwardDiff.derivative(V -> eos(model, V, T), V)
end

# function ∂²A∂V²(X::Vector, V, T)
#     return ForwardDiff.derivative(V -> pressure_NN(X, V, T), V)
# end

# function ∂³A∂V²∂T(X::Vector, V, T)
#     return ForwardDiff.derivative(T -> ∂²A∂V²(X, V, T), T)
# end

# function ∂p∂V(X::Vector, V, T)
#     return ForwardDiff.derivative(V -> pressure_NN(X, V, T), V)
# end

# function ChainRulesCore.rrule(::typeof(critical_temperature_NN), X)
#     saft_model = make_model(X...)
#     Tc, pc, Vc = crit_pure(saft_model)

#     function f_pullback(Δy)
#         #* Newton step from perfect initialisation
#         # todo calculate both these derivatives in one pass
#         function f(X)
#             Tc2 = Tc - ∂²A∂V²(X, Vc, Tc)/∂³A∂V²∂T(X, Vc, Tc)
#             return Tc2
#         end

#         ∂X = @thunk(ForwardDiff.gradient(X -> f(X), X) .* Δy)
#         return (NoTangent(), ∂X)
#     end

#     return Tc, f_pullback
# end

# function saturation_pressure_NN(X, T)
#     model = make_model(X...)
#     p, Vₗ, Vᵥ = saturation_pressure(model, T)

#     return p
# end

# function ChainRulesCore.rrule(::typeof(saturation_pressure_NN), X, T)
#     model = make_model(X...)
#     p, Vₗ, Vᵥ = saturation_pressure(model, T)

#     function f_pullback(Δy)
#         #* Newton step from perfect initialisation
#         #! How do I define the gradients wrt Vₗ ?
#         function f(X, T)
#             NN_model = make_NN_model(X...)
#             p2 = -(eos(NN_model, Vᵥ, T) - eos(NN_model, Vₗ, T)) / (Vᵥ - Vₗ)
#             return p2
#         end

#         ∂X = @thunk(ForwardDiff.gradient(X -> f(X, T), X) .* Δy)
#         ∂T = @thunk(ForwardDiff.derivative(T -> f(X, T), T) .* Δy)
#         return (NoTangent(), ∂X, ∂T)
#     end

#     return p, f_pullback
# end


# function volume_NN(X, p, T)#, Vₗ = nothing)
#     model = make_model(X...)
#     V = volume(model, p, T; phase=:liquid)

#     return V
# end

# function ChainRulesCore.rrule(::typeof(volume_NN), X, p, T)
#     # model = make_model(X...)
#     # vL = volume(model, p, T, [1.0]; phase=:liquid)
#     vL = volume_NN(X, p, T) #* Is this cached?

#     function f_pullback(Δy)
#         #* Newton step from perfect initialisation
#         function f_V(X, p, T)
#             # model = make_NN_model(X...)
#             ∂p∂V = ForwardDiff.derivative(V -> pressure_NN(X, V, T), vL)
#             v2 = vL - (pressure_NN(X, vL, T) - p) / ∂p∂V
#             return v2
#         end

#         # ∂X = @thunk(ForwardDiff.gradient(X -> f_V(X, p, T), X) .* Δy)
#         ∂X = ForwardDiff.gradient(X -> f_V(X, p, T), X) .* Δy
#         ∂p = @thunk(ForwardDiff.derivative(p -> f_V(X, p, T), p) .* Δy)
#         ∂T = @thunk(ForwardDiff.derivative(T -> f_V(X, p, T), T) .* Δy)
#         return (NoTangent(), ∂X, ∂p, ∂T)
#     end

#     return vL, f_pullback
# end

# diagvalues(x<:Rea) = x
function diagvalues(x::T) where {T<:Real}
    return x
end


function data(model::SAFTVRMieNN, V, T, z)
    m̄ = dot(z, model.params.segment)
    _d = @f(d)
    ζi = @f(ζ0123, _d)
    _ζ_X, σ3x = @f(ζ_X_σ3, _d, m̄)
    _ρ_S = @f(ρ_S, m̄)
    _ζst = σ3x * _ρ_S * π / 6
    return (_d, _ρ_S, ζi, _ζ_X, _ζst, σ3x, m̄)
end

#fused chain and disp calculation
function Clapeyron.a_res(model::SAFTVRMieNN, V, T, z)
    _data = @f(data)
    return @f(a_hs, _data) + @f(a_dispchain, _data)# + @f(a_assoc, _data)
end

function a_mono(model::SAFTVRMieNN, V, T, z, _data=@f(data))
    return @f(a_hs, _data) + @f(a_disp, _data)
end

function a_hs(model::SAFTVRMieNN, V, T, z, _data=@f(data))
    _, _, ζi, _, _, _, m̄ = _data
    ζ0, ζ1, ζ2, ζ3 = ζi
    return m̄ * bmcs_hs(ζ0, ζ1, ζ2, ζ3) / sum(z)
end

function ρ_S(model::SAFTVRMieNN, V, T, z, m̄=dot(z, model.params.segment))
    T1 = eltype(V + T + first(z))
    return T1(N_A) / V * m̄
end

function ζ0123(model::SAFTVRMieNN, V, T, z, _d=@f(d), m̄=dot(z, model.params.segment))
    m = model.params.segment
    # _0 = zero(V + T + first(z) + one(eltype(model)))
    T1 = eltype(V + T + first(z))
    _0 = zero(T1)
    ζ0, ζ1, ζ2, ζ3 = _0, _0, _0, _0
    for i ∈ 1:length(z)
        di = _d[i]
        xS = z[i] * m[i] / m̄
        ζ0 += xS
        ζ1 += xS * di
        ζ2 += xS * di * di
        ζ3 += xS * di * di * di
    end
    c = T1(π) / 6 * T1(N_A) * m̄ / V
    ζ0, ζ1, ζ2, ζ3 = c * ζ0, c * ζ1, c * ζ2, c * ζ3
    return ζ0, ζ1, ζ2, ζ3
end

#=
SAFT-VR-Mie diameter:
Defined as:
```
C  = (λr/(λr-λa))*(λr/λa)^(λa/(λr-λa))
u(r) = C*ϵ*(x^-λr - x^-λa)
f(r) = exp(-u(r)/T)
d = σ*(1-integral(f(r),0,1))
```

we use a mixed approach, depending on T⋆ = T/ϵ:

if T⋆ < 1:
    5-point gauss-laguerre. we do the change of variables `y = r^-λr`
else:
    10-point modified gauss-legendre with cut. (pending)
=#
function _laguerrex(Base.@specialize(f), r, a, u, w)
    k = exp(-r * a) / r
    u1, w1 = u[1], w[1]
    rinv = 1 / r
    x1 = u1 * rinv + a
    f1 = f(x1)
    res = w1 * f1
    l = length(u)
    for i in 2:l
        xi = u[i] * rinv + a
        res += w[i] * f(xi)
    end
    return res * k
end

function laguerre5(Base.@specialize(f), r=1.0, a=0.0)
    laguerre5_u = (0.26356031971814109102031, 1.41340305910651679221800, 3.59642577104072208122300, 7.08581000585883755692200, 12.6408008442757826594300)
    laguerre5_w = (0.5217556105828086524759, 0.3986668110831759274500, 7.5942449681707595390e-2, 3.6117586799220484545e-3, 2.3369972385776227891e-5)
    return _laguerrex(f, r, a, laguerre5_u, laguerre5_w)
end

function d_vrmie(T, λa, λr, σ, ϵ)
    Tx = T / ϵ
    C = Cλ_mie(λa, λr)
    θ = C / Tx
    λrinv = 1 / λr
    λaλr = λa / λr

    f_laguerre(x) = x^(-λrinv) * exp(θ * x^(λaλr)) * λrinv / x
    ∑fi = laguerre5(f_laguerre, θ, one(θ))
    di = σ * (1 - ∑fi)
    return di
end

function d(model::SAFTVRMieNN, V, T, z)
    ϵ = diagvalues(model.params.epsilon)
    σ = diagvalues(model.params.sigma)
    λa = diagvalues(model.params.lambda_a)
    λr = diagvalues(model.params.lambda_r)
    n = length(z)
    _d = [d_vrmie(T, λa[k], λr[k], σ[k], ϵ[k]) for k ∈ 1:n]
    return _d
end


function d(model::SAFTVRMieNN, V, T, z, λa, λr, ϵ, σ)
    d_vrmie(T, λa, λr, σ, ϵ)
end

function Cλ(model::SAFTVRMieNN, V, T, z, λa, λr)
    return Cλ_mie(λa, λr)
end

Cλ_mie(λa, λr) = (λr / (λr - λa)) * (λr / λa)^(λa / (λr - λa))

function ζ_X(model::SAFTVRMieNN, V, T, z, _d=@f(d))
    _ζ_X, σ3x = @f(ζ_X_σ3, _d)
    return _ζ_X
end

function ζ_X_σ3(model::SAFTVRMieNN, V, T, z, _d=@f(d), m̄=dot(z, model.params.segment))
    T1 = eltype(V + T + first(z))
    m = model.params.segment
    m̄ = dot(z, m)
    m̄inv = 1 / m̄
    σ = model.params.sigma
    ρS = T1(N_A) / V * m̄
    comps = 1:length(z)
    _ζ_X = zero(V + T + first(z))
    kρS = ρS * π / 6 / 8
    σ3_x = _ζ_X

    for i ∈ comps
        x_Si = z[i] * m[i] * m̄inv
        σ3_x += x_Si * x_Si * (σ[i, i]^3)
        di = _d[i]
        r1 = kρS * x_Si * x_Si * (2 * di)^3
        _ζ_X += r1
        for j ∈ 1:(i-1)
            x_Sj = z[j] * m[j] * m̄inv
            σ3_x += 2 * x_Si * x_Sj * (σ[i, j]^3)
            dij = (di + _d[j])
            r1 = kρS * x_Si * x_Sj * dij^3
            _ζ_X += 2 * r1
        end
    end

    return _ζ_X, σ3_x
end

function aS_1(model::SAFTVRMieNN, V, T, z, λ, ζ_X_=@f(ζ_X))
    ζeff_ = @f(ζeff, λ, ζ_X_)
    return -1 / (λ - 3) * (1 - ζeff_ / 2) / (1 - ζeff_)^3
end

function ζeff(model::SAFTVRMieNN, V, T, z, λ, ζ_X_=@f(ζ_X))
    A = SAFTγMieconsts.A
    λ⁻¹ = one(λ) / λ
    Aλ⁻¹ = A * SA[one(λ); λ⁻¹; λ⁻¹ * λ⁻¹; λ⁻¹ * λ⁻¹ * λ⁻¹]
    return dot(Aλ⁻¹, SA[ζ_X_; ζ_X_^2; ζ_X_^3; ζ_X_^4])
end

function B(model::SAFTVRMieNN, V, T, z, λ, x_0, ζ_X_=@f(ζ_X))
    x_0_3λ = x_0^(3 - λ)
    ζ_X_m13 = (1 - ζ_X_)^3
    I = (1 - x_0_3λ) / (λ - 3)
    J = (1 - (λ - 3) * x_0^(4 - λ) + (λ - 4) * x_0_3λ) / ((λ - 3) * (λ - 4))
    return I * (1 - ζ_X_ / 2) / ζ_X_m13 - 9 * J * ζ_X_ * (ζ_X_ + 1) / (2 * ζ_X_m13)
end

function KHS(model::SAFTVRMieNN, V, T, z, ζ_X_=@f(ζ_X), ρS=@f(ρ_S))
    return (1 - ζ_X_)^4 / (1 + 4ζ_X_ + 4ζ_X_^2 - 4ζ_X_^3 + ζ_X_^4)
end

function f123456(model::SAFTVRMieNN, V, T, z, α)
    ϕ = SAFTVRMieconsts.ϕ
    _0 = zero(α)

    add_tuples(a, b) = map(+, a, b)
    in_1 = collect(ϕ[i] .* α^(i - 1) for i in 1:4)
    fa = reduce(add_tuples, in_1)
    fb = reduce(add_tuples, (ϕ[i] .* α^(i - 4) for i in 5:7))

    return fa ./ (one(_0) .+ fb)
end

function ζst(model::SAFTVRMieNN, V, T, z, _σ=model.params.sigma)
    T1 = eltype(V + T + first(z))
    m = model.params.segment
    m̄ = dot(z, m)
    m̄inv = 1 / m̄
    ρS = T1(N_A) / V * m̄
    comps = @comps
    _ζst = zero(V + T + first(z))
    for i ∈ comps
        x_Si = z[i] * m[i] * m̄inv
        _ζst += x_Si * x_Si * (_σ[i, i]^3)
        for j ∈ 1:i-1
            x_Sj = z[j] * m[j] * m̄inv
            _ζst += 2 * x_Si * x_Sj * (_σ[i, j]^3)
        end
    end

    return _ζst * ρS * π / 6
end

function g_HS(model::SAFTVRMieNN, V, T, z, x_0ij, ζ_X_=@f(ζ_X))
    ζX3 = (1 - ζ_X_)^3
    k_0 = -log(1 - ζ_X_) + evalpoly(ζ_X_, (0, 42, -39, 9, -2)) / (6 * ζX3)
    k_1 = evalpoly(ζ_X_, (0, -12, 6, 0, 1)) / (2 * ζX3)
    k_2 = -3 * ζ_X_^2 / (8 * (1 - ζ_X_)^2)
    k_3 = evalpoly(ζ_X_, (0, 3, 3, 0, -1)) / (6 * ζX3)
    return exp(evalpoly(x_0ij, (k_0, k_1, k_2, k_3)))
end

function ζeff_fdf(model::SAFTVRMieNN, V, T, z, λ, ζ_X_, ρ_S_)
    A = SAFTγMieconsts.A
    λ⁻¹ = one(λ) / λ
    Aλ⁻¹ = A * [one(λ); λ⁻¹; λ⁻¹ * λ⁻¹; λ⁻¹ * λ⁻¹ * λ⁻¹]
    _f = dot(Aλ⁻¹, [ζ_X_; ζ_X_^2; ζ_X_^3; ζ_X_^4])
    _df = dot(Aλ⁻¹, [1; 2ζ_X_; 3ζ_X_^2; 4ζ_X_^3]) * ζ_X_ / ρ_S_
    return _f, _df
end

function aS_1_fdf(model::SAFTVRMieNN, V, T, z, λ, ζ_X_=@f(ζ_X), ρ_S_=@f(ρ_S))
    ζeff_, ∂ζeff_ = @f(ζeff_fdf, λ, ζ_X_, ρ_S_)
    ζeff3 = (1 - ζeff_)^3
    ζeffm1 = (1 - ζeff_ * 0.5)
    ζf = ζeffm1 / ζeff3
    λf = -1 / (λ - 3)
    _f = λf * ζf
    _df = λf * (ζf + ρ_S_ * ∂ζeff_ * ((3 * ζeffm1 * (1 - ζeff_)^2 - 0.5 * ζeff3) / ζeff3^2))
    return _f, _df
end

function B_fdf(model::SAFTVRMieNN, V, T, z, λ, x_0, ζ_X_=@f(ζ_X), ρ_S_=@f(ρ_S))
    T1 = eltype(V + T + first(z))
    x_0_λ = x_0^(3 - λ)
    I = (1 - x_0_λ) / (λ - 3)
    J = (1 - (λ - 3) * x_0^(4 - λ) + (λ - 4) * x_0_λ) / ((λ - 3) * (λ - 4))
    ζX2 = (1 - ζ_X_)^2
    ζX3 = (1 - ζ_X_)^3
    ζX6 = ζX3 * ζX3

    _f = I * (1 - ζ_X_ / 2) / ζX3 - 9 * J * ζ_X_ * (ζ_X_ + 1) / (2 * ζX3)
    _df = (((1 - ζ_X_ / 2) * I / ζX3 - 9 * ζ_X_ * (1 + ζ_X_) * J / (2 * ζX3))
           +
           ζ_X_ * ((3 * (1 - ζ_X_ / 2) * ζX2
                    -
                    1 / 2 * ζX3) * I / ζX6
                   -
                   9 * J * ((1 + 2 * ζ_X_) * ζX3
                            +
                            ζ_X_ * (1 + ζ_X_) * 3 * ζX2) / (2 * ζX6)))
    # @show _df
    return _f, _df
end

function KHS_fdf(model::SAFTVRMieNN, V, T, z, ζ_X_, ρ_S_=@f(ρ_S))
    ζX4 = (1 - ζ_X_)^4
    denom1 = evalpoly(ζ_X_, (1, 4, 4, -4, 1))
    ∂denom1 = evalpoly(ζ_X_, (4, 8, -12, 4))
    _f = ζX4 / denom1
    _df = -(ζ_X_ / ρ_S_) * ((4 * (1 - ζ_X_)^3 * denom1 + ζX4 * ∂denom1) / denom1^2)
    return _f, _df
end

function ∂a_2╱∂ρ_S(model::SAFTVRMieNN, V, T, z, i)
    λr = diagvalues(model.params.lambda_r)
    λa = diagvalues(model.params.lambda_a)
    x_0ij = @f(x_0, i, i)
    ζ_X_ = @f(ζ_X)
    ρ_S_ = @f(ρ_S)
    ∂KHS╱∂ρ_S = -ζ_X_ / ρ_S_ *
                ((4 * (1 - ζ_X_)^3 * (1 + 4 * ζ_X_ + 4 * ζ_X_^2 - 4 * ζ_X_^3 + ζ_X_^4)
                  +
                  (1 - ζ_X_)^4 * (4 + 8 * ζ_X_ - 12 * ζ_X_^2 + 4 * ζ_X_^3)) / (1 + 4 * ζ_X_ + 4 * ζ_X_^2 - 4 * ζ_X_^3 + ζ_X_^4)^2)
    return 0.5 * @f(C, i, i)^2 *
           (@f(ρ_S) * ∂KHS╱∂ρ_S * (x_0ij^(2 * λa[i]) * (@f(aS_1, 2 * λa[i]) + @f(B, 2 * λa[i], x_0ij))
                                   -
                                   2 * x_0ij^(λa[i] + λr[i]) * (@f(aS_1, λa[i] + λr[i]) + @f(B, λa[i] + λr[i], x_0ij))
                                   +
                                   x_0ij^(2 * λr[i]) * (@f(aS_1, 2 * λr[i]) + @f(B, 2 * λr[i], x_0ij)))
            +
            @f(KHS) * (x_0ij^(2 * λa[i]) * (@f(∂aS_1╱∂ρ_S, 2 * λa[i]) + @f(∂B╱∂ρ_S, 2 * λa[i], x_0ij))
                       -
                       2 * x_0ij^(λa[i] + λr[i]) * (@f(∂aS_1╱∂ρ_S, λa[i] + λr[i]) + @f(∂B╱∂ρ_S, λa[i] + λr[i], x_0ij))
                       +
                       x_0ij^(2 * λr[i]) * (@f(∂aS_1╱∂ρ_S, 2 * λr[i]) + @f(∂B╱∂ρ_S, 2 * λr[i], x_0ij))))
end

function I(model::SAFTVRMieNN, V, T, z, Tr, _data=@f(data))
    _d, ρS, ζi, _ζ_X, _ζst, σ3_x = _data
    c = SAFTVRMieconsts.c
    res = zero(_ζst)
    ρr = ρS * σ3_x
    @inbounds for n ∈ 0:10
        ρrn = ρr^n
        res_m = zero(res)
        for m ∈ 0:(10-n)
            res_m += c[n+1, m+1] * Tr^m
        end
        res += res_m * ρrn
    end
    return res
end

function Δ(model::SAFTVRMieNN, V, T, z, i, j, a, b, _data=@f(data))
    ϵ = model.params.epsilon
    Tr = T / ϵ[i, j]
    _I = @f(I, Tr, _data)
    ϵ_assoc = model.params.epsilon_assoc
    K = model.params.bondvol
    F = expm1(ϵ_assoc[i, j][a, b] / T)
    return F * K[i, j][a, b] * _I
end

#optimized functions for maximum speed on default SAFTVRMie
function a_dispchain(model::SAFTVRMieNN, V, T, z, _data=@f(data))
    T1 = eltype(V + T + first(z))
    _d, ρS, ζi, ζₓ, _ζst, _, m̄ = _data
    comps = @comps
    ∑z = Clapeyron.∑(z)
    m = model.params.segment
    _ϵ = model.params.epsilon
    _λr = model.params.lambda_r
    _λa = model.params.lambda_a
    _σ = model.params.sigma
    m̄inv = 1 / m̄
    # a₁ = zero(V + T + first(z) + one(eltype(model)))
    a₁ = zero(V + T + first(z))
    a₂ = a₁
    a₃ = a₁
    achain = a₁
    _ζst5 = _ζst^5
    _ζst8 = _ζst^8
    _KHS, _∂KHS = @f(KHS_fdf, ζₓ, ρS)
    # @show _KHS, _∂KHS
    for i ∈ comps
        j = i
        mi = m[i]
        x_Si = z[i] * mi * m̄inv
        x_Sj = x_Si
        ϵ = _ϵ[i, j]
        λa = _λa[i, j]
        λr = _λr[i, j]
        σ = _σ[i, j]
        _C = @f(Cλ, λa, λr)
        dij = _d[i]
        x_0ij = σ / dij
        dij3 = dij^3
        τ = ϵ / T
        #precalculate exponentials of x_0ij
        x_0ij_λa = x_0ij^λa
        x_0ij_λr = x_0ij^λr
        x_0ij_2λa = x_0ij^(2 * λa)
        x_0ij_2λr = x_0ij^(2 * λr)
        x_0ij_λaλr = x_0ij^(λa + λr)

        #calculations for a1 - diagonal
        aS₁_a, ∂aS₁∂ρS_a = @f(aS_1_fdf, λa, ζₓ, ρS)
        aS₁_r, ∂aS₁∂ρS_r = @f(aS_1_fdf, λr, ζₓ, ρS)
        B_a, ∂B∂ρS_a = @f(B_fdf, λa, x_0ij, ζₓ, ρS)
        B_r, ∂B∂ρS_r = @f(B_fdf, λr, x_0ij, ζₓ, ρS)
        a1_ij = (2 * T1(π) * ϵ * dij3) * _C * ρS *
                (x_0ij_λa * (aS₁_a + B_a) - x_0ij_λr * (aS₁_r + B_r))

        #calculations for a2 - diagonal
        aS₁_2a, ∂aS₁∂ρS_2a = @f(aS_1_fdf, 2 * λa, ζₓ, ρS)
        aS₁_2r, ∂aS₁∂ρS_2r = @f(aS_1_fdf, 2 * λr, ζₓ, ρS)
        aS₁_ar, ∂aS₁∂ρS_ar = @f(aS_1_fdf, λa + λr, ζₓ, ρS)
        B_2a, ∂B∂ρS_2a = @f(B_fdf, 2 * λa, x_0ij, ζₓ, ρS)
        B_2r, ∂B∂ρS_2r = @f(B_fdf, 2 * λr, x_0ij, ζₓ, ρS)
        B_ar, ∂B∂ρS_ar = @f(B_fdf, λr + λa, x_0ij, ζₓ, ρS)
        α = _C * (1 / (λa - 3) - 1 / (λr - 3))
        f1, f2, f3, f4, f5, f6 = @f(f123456, α)
        _χ = f1 * _ζst + f2 * _ζst5 + f3 * _ζst8
        # @show x_0ij_2λa
        # @show aS₁_2a
        # @show B_2a
        # @show x_0ij_λaλr
        # @show aS₁_ar
        # @show B_ar
        # @show x_0ij_2λr
        # @show aS₁_2r
        # @show B_2r
        a2_ij = T1(π) * _KHS * (1 + _χ) * ρS * ϵ^2 * dij3 * _C^2 *
                (x_0ij_2λa * (aS₁_2a + B_2a)
                 -
                 2 * x_0ij_λaλr * (aS₁_ar + B_ar)
                 +
                 x_0ij_2λr * (aS₁_2r + B_2r)
                )

        #calculations for a3 - diagonal
        # @show ϵ
        # @show f4
        # @show _ζst
        # @show f5
        # @show f6
        a3_ij = -ϵ^3 * f4 * _ζst * exp(_ζst * (f5 + f6 * _ζst))
        #adding - diagonal
        a₁ += a1_ij * x_Si * x_Sj
        a₂ += a2_ij * x_Si * x_Sj
        a₃ += a3_ij * x_Si * x_Sj

        # @show a₁
        # @show a₂
        # @show a₃

        g_HSi = @f(g_HS, x_0ij, ζₓ)

        ∂a_1∂ρ_S = _C * (x_0ij_λa * (∂aS₁∂ρS_a + ∂B∂ρS_a)
                         -
                         x_0ij_λr * (∂aS₁∂ρS_r + ∂B∂ρS_r)
        )
        #calculus for g1
        g_1_ = 3 * ∂a_1∂ρ_S - _C * (λa * x_0ij_λa * (aS₁_a + B_a) - λr * x_0ij_λr * (aS₁_r + B_r))
        θ = expm1(τ)
        γc = 10 * (-tanh(10 * (0.57 - α)) + 1) * _ζst * θ * exp(_ζst * (-6.7 - 8 * _ζst))
        # @show _C
        # @show _∂KHS
        # @show x_0ij_2λa
        # @show aS₁_2a
        # @show B_2a
        # @show x_0ij_λaλr
        # @show aS₁_ar
        # @show B_ar
        # @show x_0ij_2λr
        # @show aS₁_2r
        # @show B_2r
        #* 
        # @show ∂aS₁∂ρS_2a
        # @show ∂B∂ρS_2a
        # @show ∂aS₁∂ρS_ar
        # @show ∂B∂ρS_ar
        # @show ∂aS₁∂ρS_2r
        # @show ∂B∂ρS_2r
        ∂a_2∂ρ_S = 0.5 * _C^2 *
                   (ρS * _∂KHS * (x_0ij_2λa * (aS₁_2a + B_2a)
                                  -
                                  2 * x_0ij_λaλr * (aS₁_ar + B_ar)
                                  +
                                  x_0ij_2λr * (aS₁_2r + B_2r)
                    )
                    +
                    _KHS * (x_0ij_2λa * (∂aS₁∂ρS_2a + ∂B∂ρS_2a)
                            -
                            2 * x_0ij_λaλr * (∂aS₁∂ρS_ar + ∂B∂ρS_ar)
                            +
                            x_0ij_2λr * (∂aS₁∂ρS_2r + ∂B∂ρS_2r)
                   )
                   )
        # @show ∂a_2∂ρ_S

        gMCA2 = 3 * ∂a_2∂ρ_S - _KHS * _C^2 *
                               (λr * x_0ij_2λr * (aS₁_2r + B_2r) -
                                (λa + λr) * x_0ij_λaλr * (aS₁_ar + B_ar) +
                                λa * x_0ij_2λa * (aS₁_2a + B_2a)
                               )
        # @show gMCA2

        g_2_ = (1 + γc) * gMCA2
        g_Mie_ = g_HSi * exp(τ * g_1_ / g_HSi + τ^2 * g_2_ / g_HSi)
        achain -= z[i] * (log(g_Mie_) * (mi - 1))
        for j ∈ 1:i-1
            x_Sj = z[j] * m[j] * m̄inv
            ϵ = _ϵ[i, j]
            λa = _λa[i, j]
            λr = _λr[i, j]
            σ = _σ[i, j]
            _C = @f(Cλ, λa, λr)
            dij = 0.5 * (_d[i] + _d[j])
            x_0ij = σ / dij
            dij3 = dij^3
            #calculations for a1
            a1_ij = (2 * T1(π) * ϵ * dij3) * _C * ρS *
                    (x_0ij^λa * (@f(aS_1, λa, ζₓ) + @f(B, λa, x_0ij, ζₓ)) - x_0ij^λr * (@f(aS_1, λr, ζₓ) + @f(B, λr, x_0ij, ζₓ)))

            #calculations for a2
            α = _C * (1 / (λa - 3) - 1 / (λr - 3))
            f1, f2, f3, f4, f5, f6 = @f(f123456, α)
            _χ = f1 * _ζst + f2 * _ζst5 + f3 * _ζst8
            a2_ij = T1(π) * _KHS * (1 + _χ) * ρS * ϵ^2 * dij3 * _C^2 *
                    (x_0ij^(2 * λa) * (@f(aS_1, 2 * λa, ζₓ) + @f(B, 2 * λa, x_0ij, ζₓ))
                     -
                     2 * x_0ij^(λa + λr) * (@f(aS_1, λa + λr, ζₓ) + @f(B, λa + λr, x_0ij, ζₓ))
                     +
                     x_0ij^(2 * λr) * (@f(aS_1, 2λr, ζₓ) + @f(B, 2 * λr, x_0ij, ζₓ)))

            #calculations for a3
            a3_ij = -ϵ^3 * f4 * _ζst * exp(_ζst * (f5 + f6 * _ζst))
            #adding
            a₁ += 2 * a1_ij * x_Si * x_Sj
            a₂ += 2 * a2_ij * x_Si * x_Sj
            a₃ += 2 * a3_ij * x_Si * x_Sj
        end
    end
    a₁ = a₁ * m̄ / T / ∑z
    a₂ = a₂ * m̄ / (T * T) / ∑z
    a₃ = a₃ * m̄ / (T * T * T) / ∑z
    adisp = a₁ + a₂ + a₃
    # @show adisp
    # @show achain
    return adisp + achain / ∑z
end

function a_disp(model::SAFTVRMieNN, V, T, z, _data=@f(data))
    _d, ρS, ζi, _ζ_X, _ζst, _, m̄ = _data
    comps = 1:length(z)
    #this is a magic trick. we normally (should) expect length(z) = length(model),
    #but on GC models, @comps != @groups
    #if we pass Xgc instead of z, the equation is exactly the same.
    #we need to add the divide the result by sum(z) later.
    m = model.params.segment
    _ϵ = model.params.epsilon
    _λr = model.params.lambda_r
    _λa = model.params.lambda_a
    _σ = model.params.sigma
    m̄inv = 1 / m̄
    # a₁ = zero(V + T + first(z) + one(eltype(model)))
    a₁ = zero(V + T + first(z))
    a₂ = a₁
    a₃ = a₁
    _ζst5 = _ζst^5
    _ζst8 = _ζst^8
    _KHS = @f(KHS, _ζ_X, ρS)
    for i ∈ comps
        j = i
        x_Si = z[i] * m[i] * m̄inv
        x_Sj = x_Si
        ϵ = _ϵ[i, j]
        λa = _λa[i, i]
        λr = _λr[i, i]
        σ = _σ[i, i]
        _C = @f(Cλ, λa, λr)
        dij = _d[i]
        dij3 = dij^3
        x_0ij = σ / dij
        #calculations for a1 - diagonal
        aS_1_a = @f(aS_1, λa, _ζ_X)
        aS_1_r = @f(aS_1, λr, _ζ_X)
        B_a = @f(B, λa, x_0ij, _ζ_X)
        B_r = @f(B, λr, x_0ij, _ζ_X)
        a1_ij = (2 * π * ϵ * dij3) * _C * ρS *
                (x_0ij^λa * (aS_1_a + B_a) - x_0ij^λr * (aS_1_r + B_r))

        #calculations for a2 - diagonal
        aS_1_2a = @f(aS_1, 2 * λa, _ζ_X)
        aS_1_2r = @f(aS_1, 2 * λr, _ζ_X)
        aS_1_ar = @f(aS_1, λa + λr, _ζ_X)
        B_2a = @f(B, 2 * λa, x_0ij, _ζ_X)
        B_2r = @f(B, 2 * λr, x_0ij, _ζ_X)
        B_ar = @f(B, λr + λa, x_0ij, _ζ_X)
        α = _C * (1 / (λa - 3) - 1 / (λr - 3))
        f1, f2, f3, f4, f5, f6 = @f(f123456, α)
        _χ = f1 * _ζst + f2 * _ζst5 + f3 * _ζst8
        a2_ij = π * _KHS * (1 + _χ) * ρS * ϵ^2 * dij3 * _C^2 *
                (x_0ij^(2 * λa) * (aS_1_2a + B_2a)
                 -
                 2 * x_0ij^(λa + λr) * (aS_1_ar + B_ar)
                 +
                 x_0ij^(2 * λr) * (aS_1_2r + B_2r))

        #calculations for a3 - diagonal
        a3_ij = -ϵ^3 * f4 * _ζst * exp(f5 * _ζst + f6 * _ζst^2)
        #adding - diagonal
        a₁ += a1_ij * x_Si * x_Si
        a₂ += a2_ij * x_Si * x_Si
        a₃ += a3_ij * x_Si * x_Si
        for j ∈ 1:(i-1)
            x_Sj = z[j] * m[j] * m̄inv
            ϵ = _ϵ[i, j]
            λa = _λa[i, j]
            λr = _λr[i, j]
            σ = _σ[i, j]
            _C = @f(Cλ, λa, λr)
            dij = 0.5 * (_d[i] + _d[j])
            x_0ij = σ / dij
            dij3 = dij^3
            x_0ij = σ / dij
            #calculations for a1
            a1_ij = (2 * π * ϵ * dij3) * _C * ρS *
                    (x_0ij^λa * (@f(aS_1, λa, _ζ_X) + @f(B, λa, x_0ij, _ζ_X)) - x_0ij^λr * (@f(aS_1, λr, _ζ_X) + @f(B, λr, x_0ij, _ζ_X)))

            #calculations for a2
            α = _C * (1 / (λa - 3) - 1 / (λr - 3))
            f1, f2, f3, f4, f5, f6 = @f(f123456, α)
            _χ = f1 * _ζst + f2 * _ζst5 + f3 * _ζst8
            a2_ij = π * _KHS * (1 + _χ) * ρS * ϵ^2 * dij3 * _C^2 *
                    (x_0ij^(2 * λa) * (@f(aS_1, 2 * λa, _ζ_X) + @f(B, 2 * λa, x_0ij, _ζ_X))
                     -
                     2 * x_0ij^(λa + λr) * (@f(aS_1, λa + λr, _ζ_X) + @f(B, λa + λr, x_0ij, _ζ_X))
                     +
                     x_0ij^(2 * λr) * (@f(aS_1, 2λr, _ζ_X) + @f(B, 2 * λr, x_0ij, _ζ_X)))

            #calculations for a3
            a3_ij = -ϵ^3 * f4 * _ζst * exp(f5 * _ζst + f6 * _ζst^2)
            #adding
            a₁ += 2 * a1_ij * x_Si * x_Sj
            a₂ += 2 * a2_ij * x_Si * x_Sj
            a₃ += 2 * a3_ij * x_Si * x_Sj
        end
    end
    a₁ = a₁ * m̄ / T #/sum(z)
    a₂ = a₂ * m̄ / (T * T)  #/sum(z)
    a₃ = a₃ * m̄ / (T * T * T)  #/sum(z)
    adisp = a₁ + a₂ + a₃
    return adisp
end

function a_chain(model::SAFTVRMieNN, V, T, z, _data=@f(data))
    _d, ρS, ζi, _ζ_X, _ζst, _, m̄ = _data
    l = length(z)
    comps = 1:l
    ∑z = ∑(z)
    m = model.params.segment
    _ϵ = model.params.epsilon
    _λr = model.params.lambda_r
    _λa = model.params.lambda_a
    _σ = model.params.sigma
    m̄inv = 1 / m̄
    # a₁ = zero(V + T + first(z) + one(eltype(model)))
    a₁ = zero(V + T + first(z))
    a₂ = a₁
    a₃ = a₁
    achain = a₁
    _ζst5 = _ζst^5
    _ζst8 = _ζst^8
    _KHS, _∂KHS = @f(KHS_fdf, _ζ_X, ρS)
    for i ∈ comps
        x_Si = z[i] * m[i] * m̄inv
        x_Sj = x_Si
        ϵ = _ϵ[i, i]
        λa = _λa[i, i]
        λr = _λr[i, i]
        σ = _σ[i, i]
        _C = @f(Cλ, λa, λr)
        dij = _d[i]
        x_0ij = σ / dij
        dij3 = dij^3
        x_0ij = σ / dij
        #calculations for a1 - diagonal
        aS_1_a, ∂aS_1∂ρS_a = @f(aS_1_fdf, λa, _ζ_X, ρS)
        aS_1_r, ∂aS_1∂ρS_r = @f(aS_1_fdf, λr, _ζ_X, ρS)
        B_a, ∂B∂ρS_a = @f(B_fdf, λa, x_0ij, _ζ_X, ρS)
        B_r, ∂B∂ρS_r = @f(B_fdf, λr, x_0ij, _ζ_X, ρS)
        a1_ij = (2 * π * ϵ * dij3) * _C * ρS *
                (x_0ij^λa * (aS_1_a + B_a) - x_0ij^λr * (aS_1_r + B_r))

        #calculations for a2 - diagonal
        aS_1_2a, ∂aS_1∂ρS_2a = @f(aS_1_fdf, 2 * λa, _ζ_X, ρS)
        aS_1_2r, ∂aS_1∂ρS_2r = @f(aS_1_fdf, 2 * λr, _ζ_X, ρS)
        aS_1_ar, ∂aS_1∂ρS_ar = @f(aS_1_fdf, λa + λr, _ζ_X, ρS)
        B_2a, ∂B∂ρS_2a = @f(B_fdf, 2 * λa, x_0ij, _ζ_X, ρS)
        B_2r, ∂B∂ρS_2r = @f(B_fdf, 2 * λr, x_0ij, _ζ_X, ρS)
        B_ar, ∂B∂ρS_ar = @f(B_fdf, λr + λa, x_0ij, _ζ_X, ρS)
        α = _C * (1 / (λa - 3) - 1 / (λr - 3))
        f1, f2, f3, f4, f5, f6 = @f(f123456, α)
        _χ = f1 * _ζst + f2 * _ζst5 + f3 * _ζst8
        a2_ij = π * _KHS * (1 + _χ) * ρS * ϵ^2 * dij3 * _C^2 *
                (x_0ij^(2 * λa) * (aS_1_2a + B_2a)
                 -
                 2 * x_0ij^(λa + λr) * (aS_1_ar + B_ar)
                 +
                 x_0ij^(2 * λr) * (aS_1_2r + B_2r))

        #calculations for a3 - diagonal
        a3_ij = -ϵ^3 * f4 * _ζst * exp(f5 * _ζst + f6 * _ζst^2)
        #adding - diagonal
        a₁ += a1_ij * x_Si * x_Sj
        a₂ += a2_ij * x_Si * x_Sj
        a₃ += a3_ij * x_Si * x_Sj

        g_HSi = @f(g_HS, x_0ij, _ζ_X)
        ∂a_1∂ρ_S = _C * (x_0ij^λa * (∂aS_1∂ρS_a + ∂B∂ρS_a)
                         -
                         x_0ij^λr * (∂aS_1∂ρS_r + ∂B∂ρS_r))

        g_1_ = 3 * ∂a_1∂ρ_S - _C * (λa * x_0ij^λa * (aS_1_a + B_a) - λr * x_0ij^λr * (aS_1_r + B_r))
        θ = exp(ϵ / T) - 1
        γc = 10 * (-tanh(10 * (0.57 - α)) + 1) * _ζst * θ * exp(-6.7 * _ζst - 8 * _ζst^2)
        ∂a_2∂ρ_S = 0.5 * _C^2 *
                   (ρS * _∂KHS * (x_0ij^(2 * λa) * (aS_1_2a + B_2a)
                                  -
                                  2 * x_0ij^(λa + λr) * (aS_1_ar + B_ar)
                                  +
                                  x_0ij^(2 * λr) * (aS_1_2r + B_2r))
                    +
                    _KHS * (x_0ij^(2 * λa) * (∂aS_1∂ρS_2a + ∂B∂ρS_2a)
                            -
                            2 * x_0ij^(λa + λr) * (∂aS_1∂ρS_ar + ∂B∂ρS_ar)
                            +
                            x_0ij^(2 * λr) * (∂aS_1∂ρS_2r + ∂B∂ρS_2r)))

        gMCA2 = 3 * ∂a_2∂ρ_S - _KHS * _C^2 *
                               (λr * x_0ij^(2 * λr) * (aS_1_2r + B_2r) -
                                (λa + λr) * x_0ij^(λa + λr) * (aS_1_ar + B_ar) +
                                λa * x_0ij^(2 * λa) * (aS_1_2a + B_2a))
        g_2_ = (1 + γc) * gMCA2
        g_Mie_ = g_HSi * exp(ϵ / T * g_1_ / g_HSi + (ϵ / T)^2 * g_2_ / g_HSi)
        achain += z[i] * (log(g_Mie_) * (m[i] - 1))
    end
    return -achain / ∑z
end
const SAFTVRMieconsts = (
    A=SA[0.81096 1.7888 -37.578 92.284
        1.02050 -19.341 151.26 -463.50
        -1.90570 22.845 -228.14 973.92
        1.08850 -6.1962 106.98 -677.64], ϕ=((7.5365557, -359.440, 1550.9, -1.199320, -1911.2800, 9236.9),
        (-37.604630, 1825.60, -5070.1, 9.063632, 21390.175, -129430.0),
        (71.745953, -3168.00, 6534.6, -17.94820, -51320.700, 357230.0),
        (-46.835520, 1884.20, -3288.7, 11.34027, 37064.540, -315530.0),
        (-2.4679820, -0.82376, -2.7171, 20.52142, 1103.7420, 1390.2),
        (-0.5027200, -3.19350, 2.0883, -56.63770, -3264.6100, -4518.2),
        (8.0956883, 3.70900, 0.0000, 40.53683, 2556.1810, 4241.6)), c=[0.0756425183020431 -0.128667137050961 0.128350632316055 -0.0725321780970292 0.0257782547511452 -0.00601170055221687 0.000933363147191978 -9.55607377143667e-05 6.19576039900837e-06 -2.30466608213628e-07 3.74605718435540e-09
        0.134228218276565 -0.182682168504886 0.0771662412959262 -0.000717458641164565 -0.00872427344283170 0.00297971836051287 -0.000484863997651451 4.35262491516424e-05 -2.07789181640066e-06 4.13749349344802e-08 0
        -0.565116428942893 1.00930692226792 -0.660166945915607 0.214492212294301 -0.0388462990166792 0.00406016982985030 -0.000239515566373142 7.25488368831468e-06 -8.58904640281928e-08 0 0
        -0.387336382687019 -0.211614570109503 0.450442894490509 -0.176931752538907 0.0317171522104923 -0.00291368915845693 0.000130193710011706 -2.14505500786531e-06 0 0 0
        2.13713180911797 -2.02798460133021 0.336709255682693 0.00118106507393722 -0.00600058423301506 0.000626343952584415 -2.03636395699819e-05 0 0 0 0
        -0.300527494795524 2.89920714512243 -0.567134839686498 0.0518085125423494 -0.00239326776760414 4.15107362643844e-05 0 0 0 0 0
        -6.21028065719194 -1.92883360342573 0.284109761066570 -0.0157606767372364 0.000368599073256615 0 0 0 0 0 0
        11.6083532818029 0.742215544511197 -0.0823976531246117 0.00186167650098254 0 0 0 0 0 0 0
        -10.2632535542427 -0.125035689035085 0.0114299144831867 0 0 0 0 0 0 0 0
        4.65297446837297 -0.00192518067137033 0 0 0 0 0 0 0 0 0
        -0.867296219639940 0 0 0 0 0 0 0 0 0 0],
)

########
#=
Optimizations for single component SAFTVRMie
=#

#######

function d(model::SAFTVRMieNN, V, T, z::SingleComp)
    ϵ = model.params.epsilon
    σ = model.params.sigma
    λa = model.params.lambda_a
    λr = model.params.lambda_r
    # return SA[d_vrmie(T, λa[1], λr[1], σ[1], ϵ[1])]
    return [d_vrmie(T, λa[1], λr[1], σ[1], ϵ[1])]
end

#* Association code
#! This shit is gonna be hard I already know
# function a_assoc(model::SAFTVRMieNN, V, T, z, data=nothing)
#     _0 = zero(V + T + first(z)) # This needs to take into account model type?
#     nn = assoc_pair_length(model)
#     iszero(nn) && return _0
#     X_ = @f(X, data)
#     return @f(a_assoc_impl, X_)
# end

# function X(model::EoSModel, V, T, z, data=nothing)
#     nn = assoc_pair_length(model)
#     isone(nn) && return X_exact1(model, V, T, z, data)
#     options = assoc_options(model)
#     K = assoc_site_matrix(model, V, T, z, data)
#     sitesparam = getsites(model)
#     idxs = sitesparam.n_sites.p
#     Xsol = assoc_matrix_solve(K, options)
#     return PackedVofV(idxs, Xsol)
# end

# function a_assoc_impl(model::EoSModel, V, T, z, X_)
#     _0 = zero(first(X_.v))
#     sites = getsites(model)
#     n = sites.n_sites
#     res = _0
#     resᵢₐ = _0
#     for i ∈ @comps
#         ni = n[i]
#         iszero(length(ni)) && continue
#         Xᵢ = X_[i]
#         resᵢₐ = _0
#         for (a, nᵢₐ) ∈ pairs(ni)
#             Xᵢₐ = Xᵢ[a]
#             nᵢₐ = ni[a]
#             resᵢₐ += nᵢₐ * (log(Xᵢₐ) - Xᵢₐ / 2 + 0.5)
#         end
#         res += resᵢₐ * z[i]
#     end
#     return res / sum(z)
# end