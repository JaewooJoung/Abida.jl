# src/Abida.jl
module Abida

using LinearAlgebra
using Statistics
using DuckDB
using Transformers
using TextAnalysis
using Languages
using SparseArrays
using Embeddings
using Logging

export AGI, learn!, answer, cleanup!, reset_knowledge!

include("types.jl")
include("transformer_utils.jl")  # previously utils.jl
include("database.jl")
include("core.jl")

end # module
