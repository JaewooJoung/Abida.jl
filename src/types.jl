# types.jl

struct TransformerConfig
    d_model::Int       # Dimension of embeddings
    n_head::Int        # Number of attention heads
    d_ff::Int          # Feed-forward layer size
    max_seq_length::Int # Max sequence length
end

const DEFAULT_CONFIG = TransformerConfig(128, 4, 512, 64)

struct Vocabulary
    word_to_idx::Dict{String, Int}
    idx_to_word::Vector{String}
end

Vocabulary() = Vocabulary(Dict(), String[])

struct WordEmbeddings
    matrix::Matrix{Float32}
end

struct PositionalEncoding
    matrix::Matrix{Float32}
end

struct DocumentStore
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
