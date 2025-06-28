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

# Add iteration support for Vocabulary
Base.iterate(v::Vocabulary) = iterate(v.word_to_idx)
Base.iterate(v::Vocabulary, state) = iterate(v.word_to_idx, state)
Base.isempty(v::Vocabulary) = isempty(v.word_to_idx)
Base.length(v::Vocabulary) = length(v.word_to_idx)

# Make WordEmbeddings mutable
mutable struct WordEmbeddings
    matrix::Matrix{Float32}
end

struct PositionalEncoding
    matrix::Matrix{Float32}
end

# Add size method for PositionalEncoding
Base.size(pe::PositionalEncoding) = size(pe.matrix)
Base.size(pe::PositionalEncoding, dim::Int) = size(pe.matrix, dim)

# Make DocumentStore mutable
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

# Add convenience accessors for backward compatibility with tests
Base.getproperty(agi::AGI, name::Symbol) = begin
    if name === :documents
        return agi.docs.documents
    elseif name === :doc_embeddings
        return agi.docs.embeddings
    else
        return getfield(agi, name)
    end
end
