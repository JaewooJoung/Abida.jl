module Abida

using LinearAlgebra, StatsBase, TextAnalysis, Languages, SparseArrays, DuckDB, Unicode
using LogExpFunctions: logsumexp
using Statistics: mean, std
using Printf: @sprintf
using Embeddings
using Transformers

# Export all public functions
export AGI, learn!, answer, cleanup!, reset_knowledge!, encode_text, rethink!, reiterate!, lookforword, evaluate, answer_with_fallback

# Include all the components
include("types.jl")
include("transformer.jl")
include("database.jl")
include("core.jl")

end # module
