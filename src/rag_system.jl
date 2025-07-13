# rag_system.jl - Enhanced Retrieval-Augmented Generation

using NearestNeighbors
using JSON3
using HTTP
using Dates
using Statistics
using LinearAlgebra
using StatsBase
using TextAnalysis
using Languages
using Unicode

# Enhanced document chunking
struct DocumentChunk
    id::String
    content::String
    source::String
    metadata::Dict{String, Any}
    embedding::Vector{Float32}
    timestamp::DateTime
    chunk_index::Int
    total_chunks::Int
    parent_doc_id::String
end

function DocumentChunk(id::String, content::String, source::String="", 
                      metadata::Dict{String, Any}=Dict{String, Any}(),
                      embedding::Vector{Float32}=Float32[],
                      chunk_index::Int=1, total_chunks::Int=1,
                      parent_doc_id::String="")
    DocumentChunk(id, content, source, metadata, embedding, now(), 
                 chunk_index, total_chunks, parent_doc_id)
end

mutable struct VectorStore
    chunks::Vector{DocumentChunk}
    embeddings_matrix::Matrix{Float32}
    index::Union{Nothing, NearestNeighbors.NNTree}
    dimension::Int
    distance_metric::Metric
    index_type::Symbol  # :brute, :kdtree, :balltree
    chunk_id_to_idx::Dict{String, Int}
    metadata_index::Dict{String, Vector{Int}}  # For metadata-based filtering
end

function VectorStore(dimension::Int=128, distance_metric::Metric=Euclidean(), 
                    index_type::Symbol=:brute)
    VectorStore(DocumentChunk[], zeros(Float32, dimension, 0), nothing, dimension, 
               distance_metric, index_type, Dict{String, Int}(), 
               Dict{String, Vector{Int}}())
end

# Advanced chunking strategies
abstract type ChunkingStrategy end

struct FixedSizeChunking <: ChunkingStrategy
    chunk_size::Int
    overlap::Int
end

struct SentenceChunking <: ChunkingStrategy
    max_sentences::Int
    overlap_sentences::Int
end

struct SemanticChunking <: ChunkingStrategy
    similarity_threshold::Float32
    max_chunk_size::Int
    min_chunk_size::Int
end

struct ParagraphChunking <: ChunkingStrategy
    max_paragraphs::Int
    overlap_paragraphs::Int
end

function chunk_document(text::String, strategy::FixedSizeChunking=FixedSizeChunking(200, 50))
    """
    Split document into overlapping chunks using various strategies
    """
    return chunk_by_words(text, strategy.chunk_size, strategy.overlap)
end

function chunk_by_words(text::String, chunk_size::Int, overlap::Int)
    """
    Chunk by word count with overlap
    """
    words = split(text)
    chunks = String[]
    
    if length(words) <= chunk_size
        return [text]
    end
    
    start_idx = 1
    while start_idx <= length(words)
        end_idx = min(start_idx + chunk_size - 1, length(words))
        chunk = join(words[start_idx:end_idx], " ")
        push!(chunks, chunk)
        
        # Move start index with overlap
        start_idx = end_idx - overlap + 1
        
        if end_idx >= length(words)
            break
        end
    end
    
    return chunks
end

function chunk_by_sentences(text::String, strategy::SentenceChunking)
    """
    Chunk by sentence boundaries
    """
    # Simple sentence splitting (could be improved with NLP libraries)
    sentences = split(text, r"[.!?]+\s+")
    chunks = String[]
    
    if length(sentences) <= strategy.max_sentences
        return [text]
    end
    
    start_idx = 1
    while start_idx <= length(sentences)
        end_idx = min(start_idx + strategy.max_sentences - 1, length(sentences))
        chunk = join(sentences[start_idx:end_idx], ". ") * "."
        push!(chunks, chunk)
        
        start_idx = end_idx - strategy.overlap_sentences + 1
        
        if end_idx >= length(sentences)
            break
        end
    end
    
    return chunks
end

function chunk_by_paragraphs(text::String, strategy::ParagraphChunking)
    """
    Chunk by paragraph boundaries
    """
    paragraphs = split(text, r"\n\s*\n")
    filter!(p -> !isempty(strip(p)), paragraphs)
    chunks = String[]
    
    if length(paragraphs) <= strategy.max_paragraphs
        return [text]
    end
    
    start_idx = 1
    while start_idx <= length(paragraphs)
        end_idx = min(start_idx + strategy.max_paragraphs - 1, length(paragraphs))
        chunk = join(paragraphs[start_idx:end_idx], "\n\n")
        push!(chunks, chunk)
        
        start_idx = end_idx - strategy.overlap_paragraphs + 1
        
        if end_idx >= length(paragraphs)
            break
        end
    end
    
    return chunks
end

function chunk_semantically(ai::AGI, text::String, strategy::SemanticChunking)
    """
    Semantic chunking based on content similarity
    """
    sentences = split(text, r"[.!?]+\s+")
    
    if length(sentences) <= 2
        return [text]
    end
    
    # Compute embeddings for each sentence
    sentence_embeddings = Vector{Vector{Float32}}()
    for sentence in sentences
        if !isempty(strip(sentence))
            emb = encode_text(ai, sentence)
            sentence_emb = normalize(vec(mean(emb, dims=2)))
            push!(sentence_embeddings, sentence_emb)
        end
    end
    
    # Group sentences by semantic similarity
    chunks = String[]
    current_chunk_sentences = [sentences[1]]
    current_chunk_embedding = sentence_embeddings[1]
    
    for i in 2:length(sentences)
        sentence = sentences[i]
        sentence_emb = sentence_embeddings[i]
        
        # Compute similarity with current chunk
        similarity = dot(current_chunk_embedding, sentence_emb)
        
        chunk_text = join(current_chunk_sentences, ". ") * "."
        chunk_length = length(split(chunk_text))
        
        if (similarity >= strategy.similarity_threshold && 
            chunk_length < strategy.max_chunk_size) || 
           chunk_length < strategy.min_chunk_size
            
            # Add to current chunk
            push!(current_chunk_sentences, sentence)
            # Update chunk embedding (running average)
            α = 1.0f0 / length(current_chunk_sentences)
            current_chunk_embedding = (1 - α) * current_chunk_embedding + α * sentence_emb
            current_chunk_embedding = normalize(current_chunk_embedding)
        else
            # Start new chunk
            chunk_text = join(current_chunk_sentences, ". ") * "."
            push!(chunks, chunk_text)
            
            current_chunk_sentences = [sentence]
            current_chunk_embedding = sentence_emb
        end
    end
    
    # Add final chunk
    if !isempty(current_chunk_sentences)
        chunk_text = join(current_chunk_sentences, ". ") * "."
        push!(chunks, chunk_text)
    end
    
    return chunks
end

function chunk_document(ai::AGI, text::String, strategy::SemanticChunking)
    return chunk_semantically(ai, text, strategy)
end

function chunk_document(text::String, strategy::SentenceChunking)
    return chunk_by_sentences(text, strategy)
end

function chunk_document(text::String, strategy::ParagraphChunking)
    return chunk_by_paragraphs(text, strategy)
end

function add_document!(ai::AGI, vector_store::VectorStore, text::String, 
                      source::String="", metadata::Dict{String, Any}=Dict{String, Any}(),
                      strategy::ChunkingStrategy=FixedSizeChunking(200, 50))
    """
    Add document to vector store with advanced chunking and embedding
    """
    # Generate unique document ID
    parent_doc_id = string(hash((source, text, now())))
    
    # Chunk the document
    if isa(strategy, SemanticChunking)
        chunks = chunk_document(ai, text, strategy)
    else
        chunks = chunk_document(text, strategy)
    end
    
    @info "Adding document" source=source chunks=length(chunks) strategy=typeof(strategy)
    
    for (i, chunk_text) in enumerate(chunks)
        # Create unique chunk ID
        chunk_id = string(hash((parent_doc_id, i, chunk_text)))
        
        # Generate embedding
        embeddings = encode_text(ai, chunk_text)
        chunk_embedding = normalize(vec(mean(embeddings, dims=2)))
        
        # Enhanced metadata
        chunk_metadata = merge(metadata, Dict{String, Any}(
            "chunk_index" => i,
            "total_chunks" => length(chunks),
            "parent_doc_id" => parent_doc_id,
            "word_count" => length(split(chunk_text)),
            "char_count" => length(chunk_text),
            "language" => detect_language(chunk_text),
            "contains_code" => contains_code(chunk_text),
            "contains_urls" => contains_urls(chunk_text)
        ))
        
        # Create chunk
        chunk = DocumentChunk(
            chunk_id,
            chunk_text,
            source,
            chunk_metadata,
            chunk_embedding,
            i,
            length(chunks),
            parent_doc_id
        )
        
        push!(vector_store.chunks, chunk)
        vector_store.chunk_id_to_idx[chunk_id] = length(vector_store.chunks)
        
        # Update metadata index
        for (key, value) in chunk_metadata
            key_str = string(key) * ":" * string(value)
            if !haskey(vector_store.metadata_index, key_str)
                vector_store.metadata_index[key_str] = Int[]
            end
            push!(vector_store.metadata_index[key_str], length(vector_store.chunks))
        end
    end
    
    # Rebuild search index
    rebuild_index!(vector_store)
    
    @info "Document added successfully" total_chunks=length(vector_store.chunks)
end

function detect_language(text::String)
    """
    Simple language detection (could be enhanced with proper NLP)
    """
    # Very basic heuristics
    if occursin(r"[а-я]", lowercase(text))
        return "russian"
    elseif occursin(r"[àáâãäåæçèéêëìíîïðñòóôõöøùúûüýþÿ]", lowercase(text))
        return "european"
    elseif occursin(r"[αβγδεζηθικλμνξοπρστυφχψω]", lowercase(text))
        return "greek"
    else
        return "english"
    end
end

function contains_code(text::String)
    """
    Detect if text contains code snippets
    """
    code_indicators = [
        r"def\s+\w+\(",  # Python functions
        r"function\s+\w+\(",  # JavaScript/Julia functions
        r"class\s+\w+",  # Class definitions
        r"import\s+\w+",  # Import statements
        r"#include\s*<",  # C/C++ includes
        r"public\s+static\s+void",  # Java
        r"\{[^}]*\}",  # Code blocks
        r"```",  # Markdown code blocks
        r"^\s*[>]\s*",  # Command line
    ]
    
    for pattern in code_indicators
        if occursin(pattern, text)
            return true
        end
    end
    return false
end

function contains_urls(text::String)
    """
    Detect if text contains URLs
    """
    url_pattern = r"https?://[^\s]+"
    return occursin(url_pattern, text)
end

function rebuild_index!(vector_store::VectorStore)
    """
    Rebuild the search index after adding documents
    """
    if isempty(vector_store.chunks)
        return
    end
    
    # Build embeddings matrix
    n_chunks = length(vector_store.chunks)
    vector_store.embeddings_matrix = zeros(Float32, vector_store.dimension, n_chunks)
    
    for (i, chunk) in enumerate(vector_store.chunks)
        if length(chunk.embedding) == vector_store.dimension
            vector_store.embeddings_matrix[:, i] = chunk.embedding
        else
            @warn "Embedding dimension mismatch" expected=vector_store.dimension actual=length(chunk.embedding)
        end
    end
    
    # Build nearest neighbors index
    if n_chunks > 0
        try
            if vector_store.index_type == :kdtree
                vector_store.index = KDTree(vector_store.embeddings_matrix, vector_store.distance_metric)
            elseif vector_store.index_type == :balltree
                vector_store.index = BallTree(vector_store.embeddings_matrix, vector_store.distance_metric)
            else  # :brute
                vector_store.index = BruteTree(vector_store.embeddings_matrix, vector_store.distance_metric)
            end
        catch e
            @warn "Failed to build index, falling back to brute force" exception=e
            vector_store.index = BruteTree(vector_store.embeddings_matrix, vector_store.distance_metric)
        end
    end
    
    @info "Search index rebuilt" chunks=n_chunks index_type=vector_store.index_type
end

function semantic_search(ai::AGI, vector_store::VectorStore, query::String, 
                        k::Int=5, similarity_threshold::Float32=0.1f0,
                        metadata_filters::Dict{String, Any}=Dict{String, Any}())
    """
    Perform semantic search using vector similarity with optional metadata filtering
    """
    if isempty(vector_store.chunks) || vector_store.index === nothing
        return DocumentChunk[]
    end
    
    # Encode query
    query_embeddings = encode_text(ai, query)
    query_vector = normalize(vec(mean(query_embeddings, dims=2)))
    
    # Apply metadata filters if specified
    valid_indices = if isempty(metadata_filters)
        collect(1:length(vector_store.chunks))
    else
        filter_by_metadata(vector_store, metadata_filters)
    end
    
    if isempty(valid_indices)
        return DocumentChunk[]
    end
    
    # Find nearest neighbors
    try
        all_indices, all_distances = knn(vector_store.index, query_vector, 
                                        min(k * 3, length(vector_store.chunks)), true)
        
        # Filter by valid indices and similarity threshold
        results = DocumentChunk[]
        for (idx, dist) in zip(all_indices, all_distances)
            if idx in valid_indices
                similarity = 1.0f0 / (1.0f0 + dist)  # Convert distance to similarity
                if similarity >= similarity_threshold
                    push!(results, vector_store.chunks[idx])
                    if length(results) >= k
                        break
                    end
                end
            end
        end
        
        return results
    catch e
        @warn "Error in semantic search" exception=e
        return DocumentChunk[]
    end
end

function filter_by_metadata(vector_store::VectorStore, filters::Dict{String, Any})
    """
    Filter chunks by metadata criteria
    """
    valid_indices = Set{Int}()
    
    for (key, value) in filters
        filter_key = string(key) * ":" * string(value)
        if haskey(vector_store.metadata_index, filter_key)
            if isempty(valid_indices)
                union!(valid_indices, vector_store.metadata_index[filter_key])
            else
                intersect!(valid_indices, vector_store.metadata_index[filter_key])
            end
        else
            # If filter not found, no results match
            return Int[]
        end
    end
    
    return collect(valid_indices)
end

function keyword_search(vector_store::VectorStore, query::String, k::Int=10)
    """
    Keyword-based search using TF-IDF-like scoring
    """
    query_words = Set(split(lowercase(query)))
    
    # Score each chunk
    scores = Tuple{DocumentChunk, Float32}[]
    
    for chunk in vector_store.chunks
        chunk_words = split(lowercase(chunk.content))
        chunk_word_set = Set(chunk_words)
        
        # Calculate overlap and normalize by document length
        overlap = length(intersect(query_words, chunk_word_set))
        tf_score = overlap / length(chunk_words)
        
        # Boost score for exact phrase matches
        exact_match_bonus = 0.0f0
        if occursin(lowercase(query), lowercase(chunk.content))
            exact_match_bonus = 0.5f0
        end
        
        # Boost score for title/source matches
        source_bonus = 0.0f0
        if any(word -> occursin(word, lowercase(chunk.source)), query_words)
            source_bonus = 0.2f0
        end
        
        total_score = tf_score + exact_match_bonus + source_bonus
        
        if total_score > 0.0f0
            push!(scores, (chunk, Float32(total_score)))
        end
    end
    
    # Sort by score and return top k
    sort!(scores, by=x->x[2], rev=true)
    return [chunk for (chunk, score) in scores[1:min(k, length(scores))]]
end

function hybrid_search(ai::AGI, vector_store::VectorStore, query::String, 
                      k::Int=5, α::Float32=0.7f0, metadata_filters::Dict{String, Any}=Dict{String, Any}())
    """
    Combine semantic search with keyword matching using weighted scores
    """
    # Get semantic results
    semantic_results = semantic_search(ai, vector_store, query, k * 2, 0.0f0, metadata_filters)
    
    # Get keyword results
    keyword_results = keyword_search(vector_store, query, k * 2)
    
    # Combine and score
    chunk_scores = Dict{String, Float32}()
    
    # Score semantic results
    for (i, chunk) in enumerate(semantic_results)
        # Higher rank = higher score
        semantic_score = 1.0f0 - (i - 1) / length(semantic_results)
        chunk_scores[chunk.id] = get(chunk_scores, chunk.id, 0.0f0) + α * semantic_score
    end
    
    # Score keyword results
    for (i, chunk) in enumerate(keyword_results)
        keyword_score = 1.0f0 - (i - 1) / length(keyword_results)
        chunk_scores[chunk.id] = get(chunk_scores, chunk.id, 0.0f0) + (1 - α) * keyword_score
    end
    
    # Get all unique chunks and sort by combined score
    all_chunks = Dict{String, DocumentChunk}()
    for chunk in vcat(semantic_results, keyword_results)
        all_chunks[chunk.id] = chunk
    end
    
    sorted_chunks = sort(collect(all_chunks), by=x->get(chunk_scores, x[1], 0.0f0), rev=true)
    
    return [chunk for (id, chunk) in sorted_chunks[1:min(k, length(sorted_chunks))]]
end

function rerank_results(ai::AGI, query::String, chunks::Vector{DocumentChunk}, 
                       top_k::Int=5)
    """
    Re-rank search results using cross-attention or other advanced methods
    """
    if length(chunks) <= top_k
        return chunks
    end
    
    # Simple re-ranking based on query-chunk similarity
    query_embeddings = encode_text(ai, query)
    query_vector = normalize(vec(mean(query_embeddings, dims=2)))
    
    scores = Float32[]
    for chunk in chunks
        # More sophisticated scoring
        chunk_emb = chunk.embedding
        
        # Base similarity
        similarity = dot(query_vector, chunk_emb)
        
        # Length penalty (prefer chunks that aren't too short or too long)
        word_count = get(chunk.metadata, "word_count", 100)
        length_penalty = exp(-abs(word_count - 150) / 100)
        
        # Recency bonus (prefer newer content)
        age_hours = Dates.value(now() - chunk.timestamp) / (1000 * 60 * 60)
        recency_bonus = exp(-age_hours / (24 * 30))  # Decay over 30 days
        
        # Source quality bonus (could be learned or manually set)
        source_bonus = contains("wiki", lowercase(chunk.source)) ? 0.1f0 : 0.0f0
        
        final_score = similarity * length_penalty + 0.1f0 * recency_bonus + source_bonus
        push!(scores, final_score)
    end
    
    # Sort and return top k
    sorted_indices = sortperm(scores, rev=true)
    return chunks[sorted_indices[1:min(top_k, length(chunks))]]
end

function rag_answer(ai::AGI, vector_store::VectorStore, question::String, 
                   max_context_length::Int=1000, k::Int=5, 
                   rerank::Bool=true, metadata_filters::Dict{String, Any}=Dict{String, Any}())
    """
    Enhanced RAG-based answering with context retrieval and re-ranking
    """
    # Retrieve relevant documents
    relevant_chunks = hybrid_search(ai, vector_store, question, k * 2, 0.7f0, metadata_filters)
    
    if isempty(relevant_chunks)
        return answer_with_fallback(ai, question, "I don't have enough context to answer this question.")
    end
    
    # Re-rank results for better relevance
    if rerank && length(relevant_chunks) > k
        relevant_chunks = rerank_results(ai, question, relevant_chunks, k)
    else
        relevant_chunks = relevant_chunks[1:min(k, length(relevant_chunks))]
    end
    
    # Build context from retrieved chunks
    context_parts = String[]
    current_length = 0
    sources = String[]
    
    for chunk in relevant_chunks
        chunk_length = length(chunk.content)
        if current_length + chunk_length <= max_context_length
            push!(context_parts, chunk.content)
            push!(sources, chunk.source)
            current_length += chunk_length
        else
            # Truncate if needed
            remaining_space = max_context_length - current_length
            if remaining_space > 50  # Only add if significant space remaining
                truncated = chunk.content[1:min(remaining_space, length(chunk.content))]
                push!(context_parts, truncated)
                push!(sources, chunk.source)
            end
            break
        end
    end
    
    # Combine context with proper formatting
    full_context = join(context_parts, "\n\n---\n\n")
    
    # Generate answer using the combined context
    combined_query = "Context information:\n$full_context\n\nBased on the context above, please answer: $question"
    
    # Use existing answer function with enhanced context
    result = answer_with_context(ai, combined_query, relevant_chunks)
    
    # Add source information
    unique_sources = unique(sources)
    source_info = isempty(unique_sources) ? "" : " [Sources: $(join(unique_sources, ", "))]"
    
    return (result[1] * source_info, result[2], join(unique_sources, ","))
end

function answer_with_context(ai::AGI, query::String, context_chunks::Vector{DocumentChunk})
    """
    Answer using provided context chunks with enhanced scoring
    """
    if isempty(ai.docs.embeddings)
        return ("No knowledge available.", 0.0f0, "")
    end
    
    # Encode the full query (with context)
    q_embedding = normalize(vec(mean(encode_text(ai, query), dims=2)))
    
    # Calculate scores against all documents
    scores = [dot(q_embedding, emb) for emb in ai.docs.embeddings]
    best_idx = argmax(scores)
    best_score = scores[best_idx]
    
    # If score is too low, use context chunks directly
    if best_score < 0.2f0 && !isempty(context_chunks)
        # Try to synthesize an answer from context chunks
        synthesized_answer = synthesize_answer_from_chunks(context_chunks, query)
        if !isempty(synthesized_answer)
            return (synthesized_answer, 0.8f0, context_chunks[1].source)
        else
            return (context_chunks[1].content, 0.6f0, context_chunks[1].source)
        end
    end
    
    response = ai.docs.documents[best_idx]
    
    # Log interaction with context information
    try
        context_sources = [chunk.source for chunk in context_chunks]
        context_info = Dict(
            "context_sources" => context_sources,
            "num_context_chunks" => length(context_chunks),
            "context_metadata" => [chunk.metadata for chunk in context_chunks]
        )
        
        DBInterface.execute(ai.conn, """
            INSERT INTO interactions (id, question, answer) 
            VALUES (nextval('interaction_id_seq'), ?, ?)
        """, (query, response))
    catch e
        @warn "Failed to log RAG interaction" exception=e
    end
    
    return (response, Float32(best_score), response)
end

function synthesize_answer_from_chunks(chunks::Vector{DocumentChunk}, query::String)
    """
    Simple answer synthesis from multiple chunks
    """
    if isempty(chunks)
        return ""
    end
    
    # Extract key sentences that might answer the question
    query_words = Set(split(lowercase(query)))
    
    relevant_sentences = String[]
    for chunk in chunks[1:min(3, length(chunks))]  # Use top 3 chunks
        sentences = split(chunk.content, r"[.!?]+")
        for sentence in sentences
            sentence = strip(sentence)
            if !isempty(sentence)
                sentence_words = Set(split(lowercase(sentence)))
                overlap = length(intersect(query_words, sentence_words))
                if overlap >= 2 || length(sentence_words) <= 10  # Good overlap or short definitive sentence
                    push!(relevant_sentences, sentence * ".")
                end
            end
        end
    end
    
    # Combine and deduplicate
    unique_sentences = unique(relevant_sentences)
    
    if !isempty(unique_sentences)
        return join(unique_sentences[1:min(3, length(unique_sentences))], " ")
    end
    
    return ""
end

# Enhanced document management
function load_documents_from_directory!(ai::AGI, vector_store::VectorStore, 
                                       directory::String, extensions::Vector{String}=[".txt", ".md", ".pdf"],
                                       strategy::ChunkingStrategy=FixedSizeChunking(300, 75))
    """
    Load all documents from a directory into the vector store with progress tracking
    """
    @info "Loading documents from directory" directory=directory extensions=extensions
    
    files_to_process = String[]
    for (root, dirs, files) in walkdir(directory)
        for file in files
            if any(endswith(lowercase(file), ext) for ext in extensions)
                push!(files_to_process, joinpath(root, file))
            end
        end
    end
    
    @info "Found files to process" count=length(files_to_process)
    
    files_loaded = 0
    total_chunks = 0
    
    progress = Progress(length(files_to_process), desc="Loading documents: ")
    
    for filepath in files_to_process
        try
            content = ""
            if endswith(lowercase(filepath), ".pdf")
                # Would need PDF processing library
                @warn "PDF processing not implemented, skipping" file=filepath
                continue
            else
                content = read(filepath, String)
            end
            
            if !isempty(strip(content))
                metadata = Dict{String, Any}(
                    "file_path" => filepath,
                    "file_size" => filesize(filepath),
                    "modified_time" => mtime(filepath),
                    "file_extension" => lowercase(splitext(filepath)[2])
                )
                
                chunks_before = length(vector_store.chunks)
                add_document!(ai, vector_store, content, filepath, metadata, strategy)
                chunks_added = length(vector_store.chunks) - chunks_before
                
                files_loaded += 1
                total_chunks += chunks_added
            end
        catch e
            @warn "Failed to load file" filepath=filepath exception=e
        end
        
        ProgressMeter.next!(progress)
    end
    
    @info "Finished loading documents" files_loaded=files_loaded total_chunks=total_chunks
end

# Real-time knowledge updates
function update_knowledge!(ai::AGI, vector_store::VectorStore, new_info::String, 
                         source::String="user_input", 
                         strategy::ChunkingStrategy=FixedSizeChunking(200, 50))
    """
    Add new information and immediately make it available for retrieval
    """
    # Add to traditional knowledge base
    learn!(ai, new_info)
    
    # Add to vector store for RAG
    metadata = Dict{String, Any}(
        "added_at" => now(), 
        "source_type" => "real_time_update",
        "auto_generated" => false
    )
    add_document!(ai, vector_store, new_info, source, metadata, strategy)
    
    chunk_count = if isa(strategy, SemanticChunking)
        length(chunk_document(ai, new_info, strategy))
    else
        length(chunk_document(new_info, strategy))
    end
    
    @info "Knowledge updated" source=source chunks_added=chunk_count
end

# Web search integration
struct WebSearchConfig
    api_key::String
    search_engine::Symbol  # :google, :bing, :duckduckgo
    max_results::Int
    timeout_seconds::Int
end

function WebSearchConfig(; api_key="", search_engine=:duckduckgo, max_results=5, timeout_seconds=10)
    WebSearchConfig(api_key, search_engine, max_results, timeout_seconds)
end

function web_search_fallback(query::String, config::WebSearchConfig=WebSearchConfig())
    """
    Fallback to web search when local knowledge is insufficient
    """
    try
        if config.search_engine == :duckduckgo
            # DuckDuckGo doesn't require API key
            return duckduckgo_search(query, config.max_results)
        elseif config.search_engine == :google && !isempty(config.api_key)
            return google_search(query, config.api_key, config.max_results)
        elseif config.search_engine == :bing && !isempty(config.api_key)
            return bing_search(query, config.api_key, config.max_results)
        else
            return ("Web search configuration incomplete", 0.0f0, "error")
        end
    catch e
        @warn "Web search failed" exception=e
        return ("Web search unavailable: $(e)", 0.0f0, "error")
    end
end

function duckduckgo_search(query::String, max_results::Int=5)
    """
    Simple DuckDuckGo search (no API key required)
    """
    # This is a simplified implementation
    # In practice, you'd parse HTML or use an unofficial API
    search_url = "https://duckduckgo.com/html/?q=" * HTTP.escapeuri(query)
    
    try
        response = HTTP.get(search_url, timeout=10)
        if response.status == 200
            # Very basic HTML parsing (would need proper parsing in practice)
            content = String(response.body)
            
            # Extract snippets (this is a placeholder)
            snippets = String[]
            # In reality, you'd parse the HTML properly
            push!(snippets, "Web search results for: $query")
            
            return (join(snippets, "\n"), 0.5f0, "duckduckgo")
        end
    catch e
        @warn "DuckDuckGo search failed" exception=e
    end
    
    return ("Web search failed", 0.0f0, "error")
end

function google_search(query::String, api_key::String, max_results::Int=5)
    """
    Google Custom Search API integration
    """
    # Placeholder for Google Custom Search API
    return ("Google search results would appear here for: $query", 0.5f0, "google")
end

function bing_search(query::String, api_key::String, max_results::Int=5)
    """
    Bing Search API integration
    """
    # Placeholder for Bing Search API
    return ("Bing search results would appear here for: $query", 0.5f0, "bing")
end

function comprehensive_answer(ai::AGI, vector_store::VectorStore, question::String, 
                            use_web_fallback::Bool=false, web_config::WebSearchConfig=WebSearchConfig(),
                            metadata_filters::Dict{String, Any}=Dict{String, Any}())
    """
    Comprehensive answering system combining all approaches with intelligent routing
    """
    @info "Processing comprehensive answer" question=question use_web=use_web_fallback
    
    # Try RAG first (most sophisticated)
    rag_response, rag_score, rag_source = rag_answer(ai, vector_store, question, 1000, 5, true, metadata_filters)
    
    @info "RAG result" score=rag_score source=rag_source
    
    # If RAG score is good enough, return it
    if rag_score > 0.4f0
        return (rag_response, rag_score, "rag:$rag_source")
    end
    
    # Try traditional similarity search
    traditional_response, traditional_score, _ = answer_with_keyword_fallback(ai, question)
    
    @info "Traditional result" score=traditional_score
    
    # If traditional search is significantly better, use it
    if traditional_score > rag_score + 0.1f0
        return (traditional_response, traditional_score, "traditional")
    end
    
    # If both scores are low and web fallback is enabled
    if use_web_fallback && max(rag_score, traditional_score) < 0.3f0
        web_response, web_score, web_source = web_search_fallback(question, web_config)
        
        # Add web results to knowledge base for future use
        if web_score > 0.3f0
            try
                update_knowledge!(ai, vector_store, 
                                "Q: $question\nA: $web_response", 
                                "web_search:$web_source")
            catch e
                @warn "Failed to add web result to knowledge base" exception=e
            end
        end
        
        return (web_response, web_score, "web:$web_source")
    end
    
    # Return the better of the two local methods
    if rag_score >= traditional_score
        return (rag_response, rag_score, "rag:$rag_source")
    else
        return (traditional_response, traditional_score, "traditional")
    end
end

# Advanced search and filtering
function search_by_similarity_threshold(vector_store::VectorStore, query_embedding::Vector{Float32}, 
                                       threshold::Float32=0.5f0)
    """
    Find all chunks above a similarity threshold
    """
    results = DocumentChunk[]
    
    for chunk in vector_store.chunks
        similarity = dot(query_embedding, chunk.embedding)
        if similarity >= threshold
            push!(results, chunk)
        end
    end
    
    # Sort by similarity
    sort!(results, by=chunk->dot(query_embedding, chunk.embedding), rev=true)
    return results
end

function search_by_date_range(vector_store::VectorStore, start_date::DateTime, end_date::DateTime)
    """
    Find chunks within a date range
    """
    return filter(chunk -> start_date <= chunk.timestamp <= end_date, vector_store.chunks)
end

function search_by_source_pattern(vector_store::VectorStore, pattern::Regex)
    """
    Find chunks whose source matches a pattern
    """
    return filter(chunk -> occursin(pattern, chunk.source), vector_store.chunks)
end

function get_chunk_statistics(vector_store::VectorStore)
    """
    Get statistics about the vector store
    """
    if isempty(vector_store.chunks)
        return Dict{String, Any}()
    end
    
    word_counts = [get(chunk.metadata, "word_count", 0) for chunk in vector_store.chunks]
    sources = [chunk.source for chunk in vector_store.chunks]
    timestamps = [chunk.timestamp for chunk in vector_store.chunks]
    
    return Dict{String, Any}(
        "total_chunks" => length(vector_store.chunks),
        "unique_sources" => length(unique(sources)),
        "avg_word_count" => mean(word_counts),
        "median_word_count" => median(word_counts),
        "total_words" => sum(word_counts),
        "oldest_chunk" => minimum(timestamps),
        "newest_chunk" => maximum(timestamps),
        "embedding_dimension" => vector_store.dimension,
        "index_type" => vector_store.index_type
    )
end

# Memory and performance optimization
function optimize_vector_store!(vector_store::VectorStore, similarity_threshold::Float32=0.95f0)
    """
    Remove duplicate or very similar chunks to optimize storage and search
    """
    @info "Optimizing vector store" initial_chunks=length(vector_store.chunks)
    
    if length(vector_store.chunks) < 2
        return
    end
    
    chunks_to_remove = Set{Int}()
    
    for i in 1:length(vector_store.chunks)-1
        if i in chunks_to_remove
            continue
        end
        
        for j in i+1:length(vector_store.chunks)
            if j in chunks_to_remove
                continue
            end
            
            # Check content similarity
            similarity = dot(vector_store.chunks[i].embedding, vector_store.chunks[j].embedding)
            
            if similarity >= similarity_threshold
                # Keep the newer chunk or the one with more metadata
                chunk_i = vector_store.chunks[i]
                chunk_j = vector_store.chunks[j]
                
                if chunk_j.timestamp > chunk_i.timestamp || 
                   length(chunk_j.metadata) > length(chunk_i.metadata)
                    push!(chunks_to_remove, i)
                else
                    push!(chunks_to_remove, j)
                end
            end
        end
    end
    
    # Remove duplicate chunks (in reverse order to maintain indices)
    for idx in sort(collect(chunks_to_remove), rev=true)
        deleteat!(vector_store.chunks, idx)
    end
    
    # Clear and rebuild indices
    empty!(vector_store.chunk_id_to_idx)
    empty!(vector_store.metadata_index)
    
    for (i, chunk) in enumerate(vector_store.chunks)
        vector_store.chunk_id_to_idx[chunk.id] = i
        
        for (key, value) in chunk.metadata
            key_str = string(key) * ":" * string(value)
            if !haskey(vector_store.metadata_index, key_str)
                vector_store.metadata_index[key_str] = Int[]
            end
            push!(vector_store.metadata_index[key_str], i)
        end
    end
    
    # Rebuild search index
    rebuild_index!(vector_store)
    
    @info "Vector store optimization completed" 
          removed_chunks=length(chunks_to_remove) 
          final_chunks=length(vector_store.chunks)
end

# Export all RAG components
export DocumentChunk, VectorStore
export FixedSizeChunking, SentenceChunking, SemanticChunking, ParagraphChunking
export add_document!, rebuild_index!, semantic_search, keyword_search, hybrid_search
export rag_answer, answer_with_context, rerank_results
export load_documents_from_directory!, update_knowledge!
export WebSearchConfig, web_search_fallback, comprehensive_answer
export search_by_similarity_threshold, search_by_date_range, search_by_source_pattern
export get_chunk_statistics, optimize_vector_store!
export chunk_document, synthesize_answer_from_chunks