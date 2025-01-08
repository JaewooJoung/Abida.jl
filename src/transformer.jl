# transformer.jl

# Softmax implementation
function softmax(x::Matrix{Float32}; dims::Int=1)
    exp_x = exp.(x .- maximum(x, dims=dims))
    return exp_x ./ sum(exp_x, dims=dims)
end

# Layer normalization
function layer_norm(x::Matrix{Float32}; eps::Float32=1f-5)
    mean_x = mean(x, dims=1)
    std_x = std(x, dims=1)
    return (x .- mean_x) ./ (std_x .+ eps)
end

# ReLU activation
relu(x::Float32) = max(0f0, x)
relu(x::Matrix{Float32}) = max.(0f0, x)

# Positional Encoding
function positional_encoding(max_seq_len::Int, d_model::Int)
    pe = zeros(Float32, max_seq_len, d_model)
    for pos in 1:max_seq_len
        for i in 1:2:d_model
            angle = pos / (10000 ^ ((i-1)/d_model))
            pe[pos, i] = sin(angle)
            if i+1 <= d_model
                pe[pos, i+1] = cos(angle)
            end
        end
    end
    return pe
end

# Multi-Head Attention functions
function scaled_dot_product_attention(Q::Matrix{Float32}, K::Matrix{Float32}, V::Matrix{Float32}, mask=nothing)
    d_k = size(K, 1)
    scores = (Q' * K) / sqrt(Float32(d_k))
    
    if mask !== nothing
        scores = scores .* mask
    end
    
    attention_weights = softmax(scores, dims=2)
    return V * attention_weights, attention_weights
end

function multi_head_attention(Q::Matrix{Float32}, K::Matrix{Float32}, V::Matrix{Float32}, 
                            n_head::Int, d_model::Int)
    d_k = d_model ÷ n_head
    d_v = d_k
    
    # Split into heads
    Q_split = reshape(Q, d_k, n_head, :)
    K_split = reshape(K, d_k, n_head, :)
    V_split = reshape(V, d_v, n_head, :)
    
    # Apply attention for each head
    outputs = []
    for h in 1:n_head
        head_output, _ = scaled_dot_product_attention(
            Q_split[:, h, :],
            K_split[:, h, :],
            V_split[:, h, :]
        )
        push!(outputs, head_output)
    end
    
    # Concatenate heads and project
    return vcat(outputs...)
end

function multi_head_attention_with_residual(Q::Matrix{Float32}, K::Matrix{Float32}, V::Matrix{Float32}, 
                                           n_head::Int, d_model::Int)
    d_k = d_model ÷ n_head
    d_v = d_k
    
    # Split into heads
    Q_split = reshape(Q, d_k, n_head, :)
    K_split = reshape(K, d_k, n_head, :)
    V_split = reshape(V, d_v, n_head, :)
    
    # Apply attention for each head
    outputs = []
    for h in 1:n_head
        head_output, _ = scaled_dot_product_attention(
            Q_split[:, h, :],
            K_split[:, h, :],
            V_split[:, h, :]
        )
        push!(outputs, head_output)
    end
    
    # Concatenate heads and project
    output = vcat(outputs...)
    
    # Add residual connection and layer normalization
    output = output + Q
    output = layer_norm(output)
    
    return output
end

# Feed-Forward Network
function feed_forward(x::Matrix{Float32}, d_ff::Int)
    W1 = randn(Float32, d_ff, size(x, 1))
    W2 = randn(Float32, size(x, 1), d_ff)
    b1 = zeros(Float32, d_ff)
    b2 = zeros(Float32, size(x, 1))
    
    return W2 * relu.(W1 * x .+ b1) .+ b2
end

# Dropout
function dropout(x::Matrix{Float32}, rate::Float32)
    mask = rand(size(x)...) .> rate
    return x .* mask ./ (1 - rate)
end
