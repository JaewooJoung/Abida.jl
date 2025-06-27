# src/Abida.jl
module Abida

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

export AGI, learn!, answer, cleanup!, reset_knowledge!

include("types.jl")
include("transformer_utils.jl")  # previously utils.jl
include("database.jl")
include("core.jl")

end # module
