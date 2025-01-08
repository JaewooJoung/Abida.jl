# types.jl

struct TransformerConfig
    d_model::Int        # Embedding dimension
    n_head::Int         # Number of attention heads
    d_ff::Int          # Feed-forward dimension
    dropout::Float32    # Dropout rate
    max_seq_length::Int # Maximum sequence length
end

# Default configuration
const DEFAULT_CONFIG = TransformerConfig(
    64,    # d_model
    4,     # n_head
    256,   # d_ff
    0.1f0, # dropout
    512    # max_seq_length
)

mutable struct AGI
    documents::Vector{String}
    doc_embeddings::Vector{Vector{Float32}}
    vocab::Dict{String,Int}
    word_embeddings::Matrix{Float32}  # Word embedding matrix
    positional_enc::Matrix{Float32}   # Positional encoding matrix
    config::TransformerConfig
    idf::Vector{Float32}
    knowledge_base::Dict{String,Any}
    db_path::String
    conn::DuckDB.DB
end

struct KnowledgeGraph
    entities::Dict{String, Int}
    relations::Dict{String, Int}
    triples::Vector{Tuple{Int, Int, Int}}
end
