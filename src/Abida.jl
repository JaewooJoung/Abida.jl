# src/Abida.jl
module Abida

__precompile__(true)

using LinearAlgebra
using Statistics
using DuckDB
using DBInterface
using Transformers
using TextAnalysis
using Languages
using SparseArrays
using Logging
using JLD2

# Export all public functions
export AGI, learn!, answer, cleanup!, reset_knowledge!
export encode_text, rethink!, reiterate!, lookforword, answer_with_fallback
export answer_with_keyword_fallback, answer_with_tfidf  # Add new functions
export save, load  # Add these exports
export TransformerConfig, DEFAULT_CONFIG
export Vocabulary, WordEmbeddings, PositionalEncoding, DocumentStore

# Include files
include("types.jl")
include("transformer_utils.jl")
include("database.jl")
include("core.jl")

end # module
