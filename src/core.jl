# core.jl

using DuckDB
using Logging
using JLD2

# Internal includes
include("types.jl")
include("transformer_utils.jl")  # renamed from transformer.txt
include("database.jl")

#=
    AGI(db_path="Abida.duckdb", config=DEFAULT_CONFIG)

Constructs an AGI instance by loading data from disk/database.
=#
function AGI(db_path::String="Abida.duckdb", config::TransformerConfig=DEFAULT_CONFIG)
    try
        # Create database and connection
        db = DuckDB.DB(db_path)
        conn = DuckDB.connect(db)
        
        # Initialize database using the DB object (this should work now)
        init_database(db)

        # Load data using the DB object instead of connection for consistency
        documents, vocab_dict, doc_embeddings, word_embeddings_matrix = load_data(db, config)

        # Build vocabulary structure
        vocab = Vocabulary(vocab_dict, String[])
        # Resize idx_to_word array if needed
        if length(vocab_dict) > 0
            max_idx = maximum(values(vocab_dict))
            resize!(vocab.idx_to_word, max_idx)
            for (word, idx) in vocab_dict
                vocab.idx_to_word[idx] = word
            end
        end

        # Generate positional encoding
        positional_enc = positional_encoding(config.max_seq_length, config.d_model)

        # Create AGI instance
        AGI(
            vocab,
            WordEmbeddings(word_embeddings_matrix),
            PositionalEncoding(positional_enc),
            DocumentStore(documents, doc_embeddings),
            config,
            conn  # Store the connection in the struct
        )
    catch e
        @error "Failed to initialize AGI" exception=e
        rethrow(e)
    end
end

#=
    normalize(v)

Normalizes a vector.
=#
normalize(v::Vector{Float32}) = v / (norm(v) + eps(Float32))

#=
    with_transaction(f, conn)

Runs a function inside a database transaction.
=#
function with_transaction(f, conn)
    DBInterface.execute(conn, "BEGIN TRANSACTION")
    try
        result = f()
        DBInterface.execute(conn, "COMMIT")
        return result
    catch e
        DBInterface.execute(conn, "ROLLBACK")
        rethrow(e)
    end
end

#=
    encode_text(ai, text)

Encodes input text into embeddings.
=#
function encode_text(ai::AGI, text::String)
    words = split(lowercase(text))
    seq_length = min(length(words), ai.config.max_seq_length)
    embeddings = zeros(Float32, ai.config.d_model, seq_length)

    for (i, word) in enumerate(words[1:seq_length])
        if haskey(ai.vocab.word_to_idx, word)
            word_idx = ai.vocab.word_to_idx[word]
            # Check bounds to avoid index errors
            if word_idx <= size(ai.word_embeddings.matrix, 2)
                embeddings[:, i] = ai.word_embeddings.matrix[:, word_idx]
            else
                embeddings[:, i] = randn(Float32, ai.config.d_model)
            end
        else
            embeddings[:, i] = randn(Float32, ai.config.d_model)
        end
        # Add positional encoding (check bounds)
        if i <= size(ai.positional_enc.matrix, 2)
            embeddings[:, i] += ai.positional_enc.matrix[:, i]
        end
    end

    return embeddings
end

#=
    transformer_encode(ai, embeddings)

Applies transformer layers to embeddings.
=#
function transformer_encode(ai::AGI, embeddings::Matrix{Float32})
    # Multi-head attention
    attn_output = multi_head_attention(embeddings, embeddings, embeddings, ai.config.n_head)

    # Add & Norm
    attn_output = attn_output + embeddings
    attn_output = layer_norm(attn_output)

    # Feed-forward network
    ff_output = feed_forward(attn_output, ai.config.d_ff)

    # Add & Norm
    ff_output = ff_output + attn_output
    ff_output = layer_norm(ff_output)

    return vec(mean(ff_output, dims=2))
end

#=
    learn!(ai, text)

Adds new text to the knowledge base.
=#
function learn!(ai::AGI, text::String)
    push!(ai.docs.documents, text)
    words = split(lowercase(text))
    vocab_changed = false

    # Add new words to vocabulary
    for word in words
        if !haskey(ai.vocab.word_to_idx, word)
            vocab_changed = true
            new_idx = length(ai.vocab.idx_to_word) + 1
            ai.vocab.word_to_idx[word] = new_idx
            push!(ai.vocab.idx_to_word, word)
            
            # Insert into database
            DBInterface.execute(ai.conn, """
                INSERT OR IGNORE INTO vocabulary (word, index) VALUES (?, ?)
            """, (word, new_idx))
        end
    end

    # Expand word embeddings matrix if vocabulary grew
    vocab_size = length(ai.vocab.idx_to_word)
    old_size = size(ai.word_embeddings.matrix, 2)
    if vocab_changed && old_size < vocab_size
        new_cols = vocab_size - old_size
        new_embeddings = randn(Float32, ai.config.d_model, new_cols)
        ai.word_embeddings.matrix = hcat(ai.word_embeddings.matrix, new_embeddings)
        
        # Insert new embeddings into database
        for (word, idx) in ai.vocab.word_to_idx
            if idx > old_size
                vector_data = Float64.(ai.word_embeddings.matrix[:, idx])
                DBInterface.execute(ai.conn, """
                    INSERT OR IGNORE INTO word_embeddings (vocab_index, vector) VALUES (?, ?)
                """, (idx, vector_data))
            end
        end
    end

    # Encode text and compute document embedding
    embeddings = encode_text(ai, text)
    doc_embedding = normalize(transformer_encode(ai, embeddings))
    push!(ai.docs.embeddings, doc_embedding)

    # Insert document and embedding into database
    with_transaction(ai.conn) do
        # Get next document ID
        result = DBInterface.execute(ai.conn, "SELECT COALESCE(MAX(id), 0) + 1 as next_id FROM documents")
        next_id = first(result).next_id
        
        # Insert document
        DBInterface.execute(ai.conn, """
            INSERT INTO documents (id, content) VALUES (?, ?)
        """, (next_id, text))

        # Insert embedding
        vector_doubles = Float64.(doc_embedding)
        DBInterface.execute(ai.conn, """
            INSERT INTO embeddings (doc_id, vector) VALUES (?, ?)
        """, (next_id, vector_doubles))
    end
end

#=
    answer(ai, question)

Finds the most relevant document to the given question and returns response, confidence, best_doc
=#
function answer(ai::AGI, question::String)
    isempty(ai.docs.embeddings) && return ("No knowledge yet.", 0.0f0, "")

    q_embedding = normalize(transformer_encode(ai, encode_text(ai, question)))
    scores = [dot(q_embedding, emb) for emb in ai.docs.embeddings]
    best_idx = argmax(scores)
    score = scores[best_idx]
    return (ai.docs.documents[best_idx], Float32(score), ai.docs.documents[best_idx])
end

#=
    reset_knowledge!(ai)

Clears all learned knowledge from the database and resets internal state.
=#
function reset_knowledge!(ai::AGI)
    DBInterface.execute(ai.conn, "DELETE FROM documents")
    DBInterface.execute(ai.conn, "DELETE FROM vocabulary")
    DBInterface.execute(ai.conn, "DELETE FROM embeddings")
    DBInterface.execute(ai.conn, "DELETE FROM word_embeddings")

    ai.docs.documents = String[]
    ai.docs.embeddings = Vector{Float32}[]
    ai.vocab.word_to_idx = Dict{String,Int}()
    ai.vocab.idx_to_word = String[]
    ai.word_embeddings.matrix = zeros(Float32, ai.config.d_model, 1)
end

#=
    rethink!(ai, prompt)

Analyzes relationships between sentences based on similarity.
=#
function rethink!(ai::AGI, prompt::String)
    with_transaction(ai.conn) do
        # Clear existing relationships
        DBInterface.execute(ai.conn, "DELETE FROM sentence_relationships")
        
        # Calculate similarities between all document pairs
        for i in 1:length(ai.docs.documents)-1
            for j in i+1:length(ai.docs.documents)
                if i <= length(ai.docs.embeddings) && j <= length(ai.docs.embeddings)
                    emb_i = ai.docs.embeddings[i]
                    emb_j = ai.docs.embeddings[j]
                    sim = dot(emb_i, emb_j) / (norm(emb_i) * norm(emb_j))
                    
                    DBInterface.execute(ai.conn, """
                        INSERT INTO sentence_relationships (sentence_id_1, sentence_id_2, strength)
                        VALUES (?, ?, ?)
                    """, (i, j, Float64(sim)))
                end
            end
        end
    end
end

#=
    reiterate!(ai)

Re-encodes all stored documents to update their embeddings.
=#
function reiterate!(ai::AGI)
    ai.docs.embeddings = Vector{Float32}[]
    for text in ai.docs.documents
        embeddings = encode_text(ai, text)
        doc_embedding = normalize(transformer_encode(ai, embeddings))
        push!(ai.docs.embeddings, doc_embedding)
    end
end

#=
    lookforword(ai, word)

Searches for documents containing the given word.
=#
function lookforword(ai::AGI, word::String)
    results = String[]
    word_lower = lowercase(word)
    for doc in ai.docs.documents
        if occursin(word_lower, lowercase(doc))
            push!(results, doc)
        end
    end
    return results
end

#=
    answer_with_fallback(ai, question, fallback)

Returns fallback if no confident match is found.
=#
function answer_with_fallback(ai::AGI, question::String, fallback="I don't know.")
    isempty(ai.docs.embeddings) && return fallback
    q_embedding = normalize(transformer_encode(ai, encode_text(ai, question)))
    scores = [dot(q_embedding, emb) for emb in ai.docs.embeddings]
    best_idx = argmax(scores)
    score = scores[best_idx]
    return score > 0.1 ? ai.docs.documents[best_idx] : fallback
end

#=
    cleanup!(ai)

Cleans up resources (e.g., closes database connection).
=#
function cleanup!(ai::AGI)
    try
        close(ai.conn)
    catch e
        @warn "Error closing connection" exception=e
    end
end

#=
    save(ai, path)

Saves AGI state to disk.
=#
function save(ai::AGI, path::String)
    JLD2.jldopen(path, "w") do file
        write(file, "vocab_idx_to_word", ai.vocab.idx_to_word)
        write(file, "vocab_word_to_idx", ai.vocab.word_to_idx)
        write(file, "word_embeddings", ai.word_embeddings.matrix)
        write(file, "positional_enc", ai.positional_enc.matrix)
        write(file, "documents", ai.docs.documents)
        write(file, "doc_embeddings", ai.docs.embeddings)
    end
end

#=
    load(path, config, db_path)

Loads AGI state from disk.
=#
function load(path::String, config::TransformerConfig, db_path::String)
    data = JLD2.jldopen(path, "r")
    vocab_words = read(data, "vocab_idx_to_word", Vector{String})
    vocab_indices = read(data, "vocab_word_to_idx", Dict{String, Int})

    vocab = Vocabulary(vocab_indices, vocab_words)

    # Create DB instance and get connection
    db = DuckDB.DB(db_path)
    conn = DuckDB.connect(db)
    init_database(db)

    AGI(
        vocab,
        WordEmbeddings(read(data, "word_embeddings", Matrix{Float32})),
        PositionalEncoding(read(data, "positional_enc", Matrix{Float32})),
        DocumentStore(read(data, "documents", Vector{String}), read(data, "doc_embeddings", Vector{Vector{Float32}})),
        config,
        conn
    )
end
