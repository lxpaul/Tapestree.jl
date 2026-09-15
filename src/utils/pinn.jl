#=

Dual-headed PINN definitions and utilities

Paul Leroux

2026

=#

module PINNUtils

using Flux
using NPZ
using LinearAlgebra

export AZMethodPINN, forward_shared_net, forward_E_net, forward_E, forward_P_net, forward_P, forward, load_pinn_npz

const SOFTPLUS_C = Float32(log(exp(1.0) - 1.0))

"""
    AZMethodPINN

Julia counterpart to the Python AZMethodPINN class.
Shared net and two sub networks for E and P heads.
"""
struct AZMethodPINN
    num_params::Int
    num_states::Int
    R_max::Float32
    shared_net::Chain
    e_net::Chain
    p_net::Chain
end

"""
    forward_shared_net(model::AZMethodPINN, theta::AbstractVector)
Forward pass through the shared network.
"""
function forward_shared_net(model::AZMethodPINN, theta::AbstractVector)
    return model.shared_net(Float32.(theta))
end

"""
    forward_E_net(model::AZMethodPINN, x::AbstractVector, t::Real)
Extinction probability E at time t
Enforces E(0) = 0 
only the E sub-network is used, and the output is scaled by t to enforce E(0) = 0.
""" 

function forward_E_net(model::AZMethodPINN, x::AbstractVector, t::Real)
    t_f32 = Float32(t)
    sx = vcat(x, [t_f32])
    raw_e = model.e_net(sx)
    return t_f32 * Flux.softplus.(raw_e)
end

"""
    forward_E(model::AZMethodPINN, x::AbstractVector, t::Real)
Extinction probability E at time t
Enforces E(0) = 0
"""

function forward_E(model::AZMethodPINN, x::AbstractVector, t::Real)
    sx = forward_shared_net(model, x)
    return forward_E_net(model, sx, t)
end

"""
    forward_P_net(model::AZMethodPINN, x::AbstractVector, t_start::Real, t_end::Real)
Probability P between times t_start and t_end
Only the P sub-network is used, and the output is scaled by dt to enforce P(t_start, t_start) = 0.
"""

function forward_P_net(model::AZMethodPINN, x::AbstractVector, t_start::Real, t_end::Real)
    K = model.num_states
    t_start_f32 = Float32(t_start)
    t_end_f32 = Float32(t_end)
    dt = t_end_f32 - t_start_f32
    sx = vcat(x, t_start_f32, t_end_f32)
    raw_P_flat = model.p_net(sx)
    raw_P = permutedims(reshape(raw_P_flat, K, K), (2, 1))
    p_off_diag = dt .* Flux.softplus.(raw_P)
    p_diag = Flux.softplus.(dt .* raw_P .+ SOFTPLUS_C)
    P_matrix = Matrix{Float32}(undef, K, K)
    @inbounds for j = 1:K, i = 1:K
        P_matrix[i, j] = (i == j) ? p_diag[i, j] : p_off_diag[i, j]
    end
    return P_matrix
end

"""
    forward_P(model::AZMethodPINN, x::AbstractVector, t_start::Real, t_end::Real)
Probability P between times t_start and t_end
"""
function forward_P(model::AZMethodPINN, x::AbstractVector, t_start::Real, t_end::Real)
    sx = forward_shared_net(model, x)
    return forward_P_net(model, sx, t_start, t_end)
end

"""
    forward(model::AZMethodPINN, x::AbstractVector, t_start::Real, t_end::Real)
Forward pass through the entire model, using the shared network and the E and P sub-networks.
"""
function forward(model::AZMethodPINN, x::AbstractVector, t_start::Real, t_end::Real)
    sx = forward_shared_net(model, x)
    E = forward_E_net(model, sx, t_end)
    P = forward_P_net(model, sx, t_start, t_end)
    return P, E
end

"""
    load_pinn_npz(npz_path::String)
Loads exported PyTorch weights from a .npz file and constructs an AZMethodPINN model.
"""

function load_pinn_npz(npz_path::String)
    data = NPZ.npzread(npz_path)

    function build_chain(prefix::String)
        layers = []
        i = 0
        while haskey(data, "$(prefix)_W$(i)")
            W = Float32.(data["$(prefix)_W$(i)"])
            b = Float32.(vec(data["$(prefix)_b$(i)"]))
            has_next_layer = haskey(data, "$(prefix)_W$(i+1)")
            act = has_next_layer ? Flux.silu : identity
            push!(layers, Dense(W, b, act))
            i += 1
        end
        return Chain(layers...)
    end

    num_params = Int(first(data["num_params"]))
    num_states = Int(first(data["num_states"]))
    R_max = Float32(first(data["R_max"]))
    
    shared_net = build_chain("shared")
    e_net = build_chain("e")
    p_net = build_chain("p")
    
    return AZMethodPINN(
        num_params,
        num_states,
        R_max,
        shared_net,
        e_net,
        p_net
    )
end

end