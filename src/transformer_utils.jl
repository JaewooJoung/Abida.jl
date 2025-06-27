# transformer_utils.jl

using Transformers
using Transformers.Layers 
using Statistics: mean, std
using LinearAlgebra: norm
using NNlib: softmax

#=
    gelu(x)

Gaussian Error Linear Unit activation function.
=#
function gelu(x::Float32)
    return 0.5f0 * x * (1.0f0 + tanh(sqrt(2.0f0 / π) * (x + 0.044715f0 * x^3)))
end

#=
    feed_forward(x, d_ff)

Feed-forward neural network used in transformers.
=#
function feed_forward(x::Matrix{Float32}, d_ff::Int)
    d_model, seq_len = size(x)
    W1 = randn(Float32, d_ff, d_model)
    b1 = zeros(Float32, d_ff, seq_len)
    W2 = randn(Float32, d_model, d_ff)
    b2 = zeros(Float32, d_model, seq_len)

    hidden = gelu.(W1 * x .+ b1)
    return W2 * hidden .+ b2
end

#=
    multi_head_attention(Q, K, V, n_heads)

Multi-head scaled dot-product attention mechanism.
=#
function multi_head_attention(Q::Matrix{Float32}, K::Matrix{Float32}, V::Matrix{Float32}, n_heads::Int)
    d_model, seq_len = size(Q)
    d_k = d_model ÷ n_heads

    Q_ = reshape(Q, d_k, n_heads, seq_len)
    K_ = reshape(K, d_k, n_heads, seq_len)
    V_ = reshape(V, d_k, n_heads, seq_len)

    scores = (Q_' .* K_) / sqrt(d_k)
    attn_weights = softmax(scores, dims=2)
    output = reshape(attn_weights * V_', d_model, seq_len)

    return output
end

#=
    layer_norm(x)

Layer normalization.
=#
function layer_norm(x::Matrix{Float32})
    mean_x = mean(x; dims=1)
    std_x = std(x; dims=1)
    return (x .- mean_x) ./ (std_x .+ 1f-6)
end

#=
    positional_encoding(max_len, d_model)

Generates sinusoidal positional encodings.
=#
function positional_encoding(max_len::Int, d_model::Int)
    pos = collect(0:max_len-1)
    i = collect(0:2:d_model-1)  # Even indices only: 0, 2, 4, ...
    
    # Create angle rates for even positions
    angle_rates = 1f0 ./ (10000f0 .^ (i ./ d_model))
    
    # Calculate angle radians
    angle_rads = pos * angle_rates'  # Results in (max_len, d_model/2)
    
    # Create encoding matrix
    encoding = zeros(Float32, max_len, d_model)
    encoding[:, 1:2:end] = sin.(angle_rads)  # Odd indices (1, 3, 5, ...)
    encoding[:, 2:2:end] = cos.(angle_rads)  # Even indices (2, 4, 6, ...)
    
    return permutedims(encoding, [2, 1])  # (d_model, max_len)
end
