# types.jl
struct TransformerConfig
    d_model::Int
    n_head::Int
    d_ff::Int
    max_seq_length::Int
end

const DEFAULT_CONFIG = TransformerConfig(128, 4, 512, 64)

export TransformerConfig, DEFAULT_CONFIG

struct Vocabulary
    word_to_idx::Dict{String, Int}
    idx_to_word::Vector{String}
end

Vocabulary() = Vocabulary(Dict(), String[])

# Make WordEmbeddings mutable so we can update the matrix
mutable struct WordEmbeddings
    matrix::Matrix{Float32}
end

struct PositionalEncoding
    matrix::Matrix{Float32}
end

# Make DocumentStore mutable so we can add documents and embeddings
mutable struct DocumentStore
    documents::Vector{String}
    embeddings::Vector{Vector{Float32}}
end

mutable struct AGI
    vocab::Vocabulary
    word_embeddings::WordEmbeddings
    positional_enc::PositionalEncoding
    docs::DocumentStore
    config::TransformerConfig
    conn::DuckDB.Connection
end
