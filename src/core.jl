# core.jl - Using DuckDB sequences properly

using DuckDB
using Logging
using JLD2

function AGI(db_path::String="Abida.duckdb", config::TransformerConfig=DEFAULT_CONFIG)
    try
        # Create database and connection
        db = DuckDB.DB(db_path)
        conn = DuckDB.connect(db)
        
        # Initialize database using the DB object
        init_database(db)

        # Load data using the DB object
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
            conn
        )
    catch e
        @error "Failed to initialize AGI" exception=e
        rethrow(e)
    end
end

normalize(v::Vector{Float32}) = v / (norm(v) + eps(Float32))

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
            try
                DBInterface.execute(ai.conn, """
                    INSERT OR IGNORE INTO vocabulary (word, index) VALUES (?, ?)
                """, (word, new_idx))
            catch e
                @warn "Failed to insert vocabulary word" word=word index=new_idx exception=e
            end
        end
    end

    # Expand word embeddings matrix if vocabulary grew
    vocab_size = length(ai.vocab.idx_to_word)
    old_size = size(ai.word_embeddings.matrix, 2)
    if vocab_changed && old_size < vocab_size
        new_cols = vocab_size - old_size
        new_embeddings = randn(Float32, ai.config.d_model, new_cols)
        
        # Create new matrix by concatenating
        ai.word_embeddings.matrix = hcat(ai.word_embeddings.matrix, new_embeddings)
        
        # Insert new embeddings into database
        for (word, idx) in ai.vocab.word_to_idx
            if idx > old_size
                try
                    vector_data = Float64.(ai.word_embeddings.matrix[:, idx])
                    DBInterface.execute(ai.conn, """
                        INSERT OR IGNORE INTO word_embeddings (vocab_index, vector) VALUES (?, ?)
                    """, (idx, vector_data))
                catch e
                    @warn "Failed to insert word embedding" word=word index=idx exception=e
                end
            end
        end
    end

    # Encode text and compute document embedding
    embeddings = encode_text(ai, text)
    doc_embedding = normalize(transformer_encode(ai, embeddings))
    push!(ai.docs.embeddings, doc_embedding)

    # Insert document and embedding into database using sequence
    try
        with_transaction(ai.conn) do
            # Insert document - let the sequence generate the ID automatically
            DBInterface.execute(ai.conn, """
                INSERT INTO documents (content) VALUES (?)
            """, (text,))
            
            # Get the ID that was just generated
            result = DBInterface.execute(ai.conn, "SELECT currval('doc_id_seq') as doc_id")
            doc_id = first(result).doc_id

            # Insert embedding with the same ID
            vector_doubles = Float64.(doc_embedding)
            DBInterface.execute(ai.conn, """
                INSERT INTO embeddings (doc_id, vector) VALUES (?, ?)
            """, (doc_id, vector_doubles))
        end
    catch e
        @warn "Failed to insert document into database" text=text exception=e
    end
end

function answer(ai::AGI, question::String)
    if isempty(ai.docs.embeddings)
        response = "No knowledge yet."
        # Log the interaction - let sequence generate ID automatically
        try
            DBInterface.execute(ai.conn, """
                INSERT INTO interactions (question, answer) VALUES (?, ?)
            """, (question, response))
        catch e
            @warn "Failed to log interaction" exception=e
        end
        return (response, 0.0f0, "")
    end

    q_embedding = normalize(transformer_encode(ai, encode_text(ai, question)))
    scores = [dot(q_embedding, emb) for emb in ai.docs.embeddings]
    best_idx = argmax(scores)
    score = scores[best_idx]
    response = ai.docs.documents[best_idx]
    
    # Log the interaction - let sequence generate ID automatically
    try
        DBInterface.execute(ai.conn, """
            INSERT INTO interactions (question, answer) VALUES (?, ?)
        """, (question, response))
    catch e
        @warn "Failed to log interaction" exception=e
    end
    
    return (response, Float32(score), response)
end

function reset_knowledge!(ai::AGI)
    try
        # Manual cascade delete - delete in correct order to respect foreign key constraints
        with_transaction(ai.conn) do
            DBInterface.execute(ai.conn, "DELETE FROM sentence_relationships")
            DBInterface.execute(ai.conn, "DELETE FROM embeddings")
            DBInterface.execute(ai.conn, "DELETE FROM documents")
            DBInterface.execute(ai.conn, "DELETE FROM word_embeddings")
            DBInterface.execute(ai.conn, "DELETE FROM vocabulary")
            DBInterface.execute(ai.conn, "DELETE FROM feedback")
            DBInterface.execute(ai.conn, "DELETE FROM model_state")
            DBInterface.execute(ai.conn, "DELETE FROM interactions")
            
            # Reset sequences to start from 1 again
            DBInterface.execute(ai.conn, "DROP SEQUENCE IF EXISTS doc_id_seq")
            DBInterface.execute(ai.conn, "DROP SEQUENCE IF EXISTS interaction_id_seq")
            DBInterface.execute(ai.conn, "DROP SEQUENCE IF EXISTS feedback_id_seq")
            DBInterface.execute(ai.conn, "CREATE SEQUENCE doc_id_seq START 1")
            DBInterface.execute(ai.conn, "CREATE SEQUENCE interaction_id_seq START 1")
            DBInterface.execute(ai.conn, "CREATE SEQUENCE feedback_id_seq START 1")
        end

        # Reset in-memory state
        ai.docs.documents = String[]
        ai.docs.embeddings = Vector{Float32}[]
        ai.vocab.word_to_idx = Dict{String,Int}()
        ai.vocab.idx_to_word = String[]
        ai.word_embeddings.matrix = zeros(Float32, ai.config.d_model, 1)
    catch e
        @warn "Failed to reset knowledge" exception=e
    end
end

function rethink!(ai::AGI, prompt::String)
    try
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
    catch e
        @warn "Failed to compute relationships" exception=e
    end
end

function reiterate!(ai::AGI)
    try
        ai.docs.embeddings = Vector{Float32}[]
        for text in ai.docs.documents
            embeddings = encode_text(ai, text)
            doc_embedding = normalize(transformer_encode(ai, embeddings))
            push!(ai.docs.embeddings, doc_embedding)
        end
    catch e
        @warn "Failed to re-encode documents" exception=e
    end
end

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

function answer_with_fallback(ai::AGI, question::String, fallback="I don't know.")
    if isempty(ai.docs.embeddings)
        return fallback
    end
    q_embedding = normalize(transformer_encode(ai, encode_text(ai, question)))
    scores = [dot(q_embedding, emb) for emb in ai.docs.embeddings]
    best_idx = argmax(scores)
    score = scores[best_idx]
    return score > 0.1 ? ai.docs.documents[best_idx] : fallback
end

function cleanup!(ai::AGI)
    try
        close(ai.conn)
    catch e
        @warn "Error closing connection" exception=e
    end
end

function save(ai::AGI, path::String)
    try
        JLD2.jldopen(path, "w") do file
            write(file, "vocab_idx_to_word", ai.vocab.idx_to_word)
            write(file, "vocab_word_to_idx", ai.vocab.word_to_idx)
            write(file, "word_embeddings", ai.word_embeddings.matrix)
            write(file, "positional_enc", ai.positional_enc.matrix)
            write(file, "documents", ai.docs.documents)
            write(file, "doc_embeddings", ai.docs.embeddings)
        end
    catch e
        @error "Failed to save AGI state" path=path exception=e
    end
end

function load(path::String, config::TransformerConfig, db_path::String)
    try
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
    catch e
        @error "Failed to load AGI state" path=path exception=e
        rethrow(e)
    end
end
