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
    encoding = zeros(Float32, d_model, max_len)
    
    for pos in 1:max_len
        for i in 1:2:d_model
            encoding[i, pos] = sin(pos / (10000^((i-1)/d_model)))
            if i+1 <= d_model
                encoding[i+1, pos] = cos(pos / (10000^((i-1)/d_model)))
            end
        end
    end
    
    return encoding
end
