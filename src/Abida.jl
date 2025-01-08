module Abida

using LinearAlgebra, StatsBase, TextAnalysis, Languages, SparseArrays, DuckDB, Unicode
using LogExpFunctions: logsumexp
using Statistics: mean, std
using Printf: @sprintf
using Embeddings
using Transformers

# Export all public functions
export AGI, learn!, answer, cleanup!, reset_knowledge!, encode_text, contextual_answer, batch_insert_vocabulary, batch_insert_word_embeddings, learning_rate_schedule, dropout, multi_head_attention_with_residual, evaluate_answer, KnowledgeGraph, add_triple!, answer_with_confidence, provide_feedback

# Include all the components
include("types.jl")
include("transformer.jl")
include("database.jl")
include("core.jl")

end # module
