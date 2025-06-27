# Abida.jl - Modular AGI System with Transformer & DuckDB Backend
module Abida

using LinearAlgebra
using StatsBase
using TextAnalysis
using Languages
using SparseArrays
using DuckDB
using LogExpFunctions: logsumexp
using Statistics: mean, std
using Printf: @sprintf
using Embeddings
using Transformers
using Logging

# Public API
export AGI, learn!, answer, cleanup!, reset_knowledge!, encode_text, rethink!, reiterate!, lookforword, evaluate, answer_with_fallback

# Internal includes
include("types.jl")
include("transformer_utils.jl")  # renamed from transformer.txt
include("database.jl")
include("core.jl")

end # module Abida
