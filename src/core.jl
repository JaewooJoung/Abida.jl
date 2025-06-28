# core.jl - Using explicit nextval() calls

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

    # Insert document and embedding into database using explicit nextval()
    try
        with_transaction(ai.conn) do
            # Use explicit nextval() in the INSERT statement
            DBInterface.execute(ai.conn, """
                INSERT INTO documents (id, content) VALUES (nextval('doc_id_seq'), ?)
            """, (text,))
            
            # Get the current value of the sequence (the ID that was just used)
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

# Improved answer() function to replace in core.jl

function answer(ai::AGI, question::String)
    if isempty(ai.docs.embeddings)
        response = "No knowledge yet."
        # Log the interaction using explicit nextval()
        try
            DBInterface.execute(ai.conn, """
                INSERT INTO interactions (id, question, answer) VALUES (nextval('interaction_id_seq'), ?, ?)
            """, (question, response))
        catch e
            @warn "Failed to log interaction" exception=e
        end
        return (response, 0.0f0, "")
    end

    # Get question embedding
    q_embedding = normalize(transformer_encode(ai, encode_text(ai, question)))
    
    # Calculate combined scores (embedding + keyword matching)
    question_words = split(lowercase(question))
    # Filter out very short words
    meaningful_words = [w for w in question_words if length(w) > 2]
    
    best_score = -1.0
    best_idx = 1
    scores = Float32[]
    
    for (i, doc_emb) in enumerate(ai.docs.embeddings)
        # Embedding similarity
        emb_similarity = dot(q_embedding, doc_emb)
        
        # Keyword matching bonus
        doc_lower = lowercase(ai.docs.documents[i])
        keyword_matches = sum([occursin(word, doc_lower) ? 1.0 : 0.0 for word in meaningful_words])
        
        # Combined score with keyword boost
        if keyword_matches > 0
            # Boost embedding score when keywords match
            combined_score = emb_similarity + (keyword_matches * 0.2)
        else
            combined_score = emb_similarity
        end
        
        push!(scores, combined_score)
        
        if combined_score > best_score
            best_score = combined_score
            best_idx = i
        end
    end
    
    response = ai.docs.documents[best_idx]
    final_score = Float32(best_score)
    
    # Log the interaction using explicit nextval()
    try
        DBInterface.execute(ai.conn, """
            INSERT INTO interactions (id, question, answer) VALUES (nextval('interaction_id_seq'), ?, ?)
        """, (question, response))
    catch e
        @warn "Failed to log interaction" exception=e
    end
    
    return (response, final_score, response)
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
            file["vocab_idx_to_word"] = ai.vocab.idx_to_word
            file["vocab_word_to_idx"] = ai.vocab.word_to_idx
            file["word_embeddings"] = ai.word_embeddings.matrix
            file["positional_enc"] = ai.positional_enc.matrix
            file["documents"] = ai.docs.documents
            file["doc_embeddings"] = ai.docs.embeddings
        end
        @info "Successfully saved AGI state" path=path documents=length(ai.docs.documents) vocab_size=length(ai.vocab.word_to_idx)
    catch e
        @error "Failed to save AGI state" path=path exception=e
        rethrow(e)
    end
end

function load(path::String, config::TransformerConfig, db_path::String)
    try
        # Open JLD2 file and read data using correct syntax
        data = JLD2.jldopen(path, "r") do file
            vocab_words = file["vocab_idx_to_word"]
            vocab_indices = file["vocab_word_to_idx"]
            word_embeddings_data = file["word_embeddings"]
            positional_enc_data = file["positional_enc"]
            documents_data = file["documents"]
            doc_embeddings_data = file["doc_embeddings"]
            
            return (
                vocab_words = vocab_words,
                vocab_indices = vocab_indices,
                word_embeddings = word_embeddings_data,
                positional_enc = positional_enc_data,
                documents = documents_data,
                doc_embeddings = doc_embeddings_data
            )
        end

        vocab = Vocabulary(data.vocab_indices, data.vocab_words)

        # Create DB instance and get connection
        db = DuckDB.DB(db_path)
        conn = DuckDB.connect(db)
        init_database(db)

        AGI(
            vocab,
            WordEmbeddings(data.word_embeddings),
            PositionalEncoding(data.positional_enc),
            DocumentStore(data.documents, data.doc_embeddings),
            config,
            conn
        )
    catch e
        @error "Failed to load AGI state" path=path exception=e
        rethrow(e)
    end
end

function answer_with_keyword_fallback(ai::AGI, question::String, similarity_threshold::Float32=0.1f0)
    """
    Enhanced answer function with keyword-based fallback when similarity is low
    """
    if isempty(ai.docs.embeddings)
        return ("No knowledge yet.", 0.0f0, "")
    end

    # First try embedding-based search
    q_embedding = normalize(transformer_encode(ai, encode_text(ai, question)))
    scores = [dot(q_embedding, emb) for emb in ai.docs.embeddings]
    best_idx = argmax(scores)
    best_score = scores[best_idx]
    
    # If similarity score is too low, fall back to keyword matching
    if best_score < similarity_threshold
        @info "Low similarity score ($best_score), trying keyword fallback"
        
        question_words = split(lowercase(question))
        meaningful_words = [w for w in question_words if length(w) > 2]
        
        best_keyword_score = 0.0
        best_keyword_idx = 1
        
        for (i, doc) in enumerate(ai.docs.documents)
            doc_lower = lowercase(doc)
            keyword_score = sum([occursin(word, doc_lower) ? 1.0 : 0.0 for word in meaningful_words])
            
            if keyword_score > best_keyword_score
                best_keyword_score = keyword_score
                best_keyword_idx = i
            end
        end
        
        # Use keyword result if it found any matches
        if best_keyword_score > 0
            return (ai.docs.documents[best_keyword_idx], Float32(best_keyword_score), ai.docs.documents[best_keyword_idx])
        end
    end
    
    # Default to similarity-based result
    return (ai.docs.documents[best_idx], Float32(best_score), ai.docs.documents[best_idx])
end

# Simple TF-IDF based search (alternative approach)
function answer_with_tfidf(ai::AGI, question::String)
    """
    Simple TF-IDF based document retrieval (alternative to embeddings)
    """
    if isempty(ai.docs.documents)
        return ("No knowledge yet.", 0.0f0, "")
    end
    
    question_words = Set(split(lowercase(question)))
    
    best_score = 0.0
    best_idx = 1
    
    for (i, doc) in enumerate(ai.docs.documents)
        doc_words = split(lowercase(doc))
        doc_word_set = Set(doc_words)
        
        # Simple overlap score
        overlap = length(intersect(question_words, doc_word_set))
        
        # Normalize by document length
        normalized_score = overlap / length(doc_words)
        
        if normalized_score > best_score
            best_score = normalized_score
            best_idx = i
        end
    end
    
    return (ai.docs.documents[best_idx], Float32(best_score), ai.docs.documents[best_idx])
end
