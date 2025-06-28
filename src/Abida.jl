# src/Abida.jl
module Abida

# Add precompilation control to avoid method overwriting issues
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

export AGI, learn!, answer, cleanup!, reset_knowledge!

# Include files only once to avoid method overwriting
include("types.jl")
include("transformer_utils.jl")
include("database.jl")
include("core.jl")

end # module
