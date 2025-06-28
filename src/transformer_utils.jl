# transformer_utils.jl+

using Transformers
using Transformers.Layers 
using Statistics: mean, std, var
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
Fixed version that handles matrix dimensions correctly.
=#
function multi_head_attention(Q::Matrix{Float32}, K::Matrix{Float32}, V::Matrix{Float32}, n_heads::Int)
    d_model, seq_len = size(Q)
    d_k = d_model ÷ n_heads
    
    # Ensure d_k is valid
    if d_k == 0
        d_k = 1
        n_heads = d_model
    end
    
    # Initialize output
    output = zeros(Float32, d_model, seq_len)
    
    # Process each head
    for h in 1:n_heads
        # Extract head dimensions
        start_dim = (h-1) * d_k + 1
        end_dim = min(h * d_k, d_model)
        
        if start_dim <= d_model && end_dim <= d_model
            # Extract Q, K, V for this head
            Q_h = Q[start_dim:end_dim, :]  # (d_k, seq_len)
            K_h = K[start_dim:end_dim, :]  # (d_k, seq_len)
            V_h = V[start_dim:end_dim, :]  # (d_k, seq_len)
            
            # Compute attention scores: Q_h^T * K_h
            scores = transpose(Q_h) * K_h  # (seq_len, seq_len)
            scores = scores / sqrt(Float32(d_k))
            
            # Apply softmax
            attn_weights = softmax(scores, dims=2)  # (seq_len, seq_len)
            
            # Apply attention to values: V_h * attn_weights^T
            head_output = V_h * transpose(attn_weights)  # (d_k, seq_len)
            
            # Place back in output
            output[start_dim:end_dim, :] = head_output
        end
    end
    
    return output
end

#=
    layer_norm(x)

Layer normalization.
=#
function layer_norm(x::Matrix{Float32})
    ε = 1f-6
    mean_x = mean(x; dims=1)
    var_x = var(x; dims=1, mean=mean_x)
    return (x .- mean_x) ./ sqrt.(var_x .+ ε)
end

#=
    positional_encoding(max_len, d_model)

Generates sinusoidal positional encodings.
=#
function positional_encoding(max_len::Int, d_model::Int)
    encoding = zeros(Float32, d_model, max_len)
    
    for pos in 1:max_len
        for i in 1:d_model
            if i % 2 == 1  # odd indices (1, 3, 5, ...)
                encoding[i, pos] = sin(pos / (10000.0^((i-1)/d_model)))
            else  # even indices (2, 4, 6, ...)
                encoding[i, pos] = cos(pos / (10000.0^((i-2)/d_model)))
            end
        end
    end
    
    return encoding
end
