# core.jl
using DuckDB
using Logging
using JLD2

include("types.jl")
include("utils.jl")

# Constructor
function AGI(db_path::String="Abida.duckdb", config::TransformerConfig=DEFAULT_CONFIG)
    try
        conn = DBInterface.connect(DuckDB.DB, db_path)
        init_database(conn)

        # Load existing data
        documents, vocab, doc_embeddings, word_embeddings = load_data(conn, config)

        positional_enc = positional_encoding(config.max_seq_length, config.d_model)

        AGI(
            Vocabulary(vocab),
            WordEmbeddings(word_embeddings),
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

# Helper function for vector normalization
normalize(v::Vector{Float32}) = v / (norm(v) + eps(Float32))

# Helper function to wrap database transactions
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

# Tokenize and encode text into embeddings
function encode_text(ai::AGI, text::String)
    words = split(lowercase(text))
    seq_length = min(length(words), ai.config.max_seq_length)

    embeddings = zeros(Float32, ai.config.d_model, seq_length)

    for (i, word) in enumerate(words[1:seq_length])
        if haskey(ai.vocab.word_to_idx, word)
            word_idx = ai.vocab.word_to_idx[word]
            embeddings[:, i] = ai.word_embeddings.matrix[:, word_idx]
        else
            embeddings[:, i] = randn(Float32, ai.config.d_model)
        end
        embeddings[:, i] += ai.positional_enc.matrix[:, i]
    end

    return embeddings
end

# Apply transformer encoding to embeddings
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

# Add new text to knowledge base
function learn!(ai::AGI, text::String)
    push!(ai.docs.documents, text)

    words = split(lowercase(text))
    vocab_changed = false

    for word in words
        if !haskey(ai.vocab.word_to_idx, word)
            vocab_changed = true
            ai.vocab.word_to_idx[word] = length(ai.vocab.idx_to_word) + 1
            push!(ai.vocab.idx_to_word, word)

            DBInterface.execute(ai.conn, """
                INSERT INTO vocabulary (word, index) VALUES (?, ?)
            """, (word, ai.vocab.word_to_idx[word]))
        end
    end

    vocab_size = length(ai.vocab.idx_to_word)

    # Grow word embeddings if needed
    if vocab_changed || size(ai.word_embeddings.matrix, 2) < vocab_size
        old_size = size(ai.word_embeddings.matrix, 2)
        new_cols = vocab_size - old_size
        new_init = randn(Float32, ai.config.d_model, new_cols)
        ai.word_embeddings.matrix = hcat(ai.word_embeddings.matrix, new_init)

        batch_insert_embeddings(ai.conn, new_init, old_size + 1 : vocab_size)
    end

    # Encode document
    embeddings = encode_text(ai, text)
    doc_embedding = transformer_encode(ai, embeddings)
    push!(ai.docs.embeddings, normalize(doc_embedding))

    next_id = first(first(DBInterface.execute(ai.conn, "SELECT COALESCE(MAX(id), 0) + 1 FROM documents")))

    with_transaction(ai.conn) do
        DBInterface.execute(ai.conn, """
            INSERT INTO documents (id, content) VALUES (?, ?)
        """, (next_id, text))

        vector_doubles = convert(Vector{Float64}, doc_embedding)
        vector_list = "ARRAY[" * join(string.(vector_doubles), ",") * "]::DOUBLE[]"

        DBInterface.execute(ai.conn, """
            INSERT INTO embeddings (doc_id, vector) VALUES (?, $vector_list)
        """, (next_id,))
    end
end

function batch_insert_embeddings(conn, embeddings, indices)
    data = [(idx, vec) for (idx, vec) in zip(indices, eachcol(embeddings))]
    words = [ai.vocab.idx_to_word[idx] for idx in indices]
    values = [(w, i, vec) for ((i, vec), w) in zip(data, words)]
    DBInterface.execute(conn, """
        INSERT INTO word_embeddings (word, vocab_index, vector)
        SELECT *
        FROM UNNEST (?, ?, ?::DOUBLE[][])
    """, (words, collect(indices), getindex.(data, 2)))
end

# Find most relevant response to question
function answer(ai::AGI, question::String)
    isempty(ai.docs.embeddings) && return "No knowledge yet."

    q_embedding = normalize(transformer_encode(ai, encode_text(ai, question)))
    scores = [dot(q_embedding, emb) for emb in ai.docs.embeddings]
    best_idx = argmax(scores)
    return ai.docs.documents[best_idx]
end

# Initialize database schema
function init_database(conn)
    DBInterface.execute(conn, """
        CREATE TABLE IF NOT EXISTS vocabulary (
            word VARCHAR PRIMARY KEY,
            index INTEGER
        );
    """)
    DBInterface.execute(conn, """
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY,
            content TEXT
        );
    """)
    DBInterface.execute(conn, """
        CREATE TABLE IF NOT EXISTS embeddings (
            doc_id INTEGER PRIMARY KEY,
            vector DOUBLE[]
        );
    """)
    DBInterface.execute(conn, """
        CREATE TABLE IF NOT EXISTS word_embeddings (
            word VARCHAR,
            vocab_index INTEGER PRIMARY KEY,
            vector DOUBLE[]
        );
    """)
end

# Load data from database
function load_data(conn, config)
    vocab_result = DBInterface.execute(conn, "SELECT word, index FROM vocabulary")
    vocab_dict = Dict{String, Int}()
    vocab_list = String[]
    for row in vocab_result
        vocab_dict[row.word] = row.index
        vocab_list = push!(vocab_list, row.word)
    end

    doc_result = DBInterface.execute(conn, "SELECT content FROM documents")
    documents = [row.content for row in doc_result]

    embed_result = DBInterface.execute(conn, "SELECT vector FROM embeddings")
    doc_embeddings = [vec(Float32.(e.vector)) for e in embed_result]

    we_result = DBInterface.execute(conn, "SELECT vector FROM word_embeddings ORDER BY vocab_index")
    word_embeddings = hcat([Float32.(v.vector) for v in we_result]...)

    return documents, vocab_dict, doc_embeddings, word_embeddings
end

# Save/load model state
function save(ai::AGI, path::String)
    jldsave(path;
        vocab_idx_to_word=ai.vocab.idx_to_word,
        vocab_word_to_idx=ai.vocab.word_to_idx,
        word_embeddings=ai.word_embeddings.matrix,
        positional_enc=ai.positional_enc.matrix,
        documents=ai.docs.documents,
        doc_embeddings=ai.docs.embeddings
    )
end

function load(path::String, config::TransformerConfig, db_path::String)
    data = jldopen(path, "r")
    vocab_words = data["vocab_idx_to_word"]
    vocab_indices = data["vocab_word_to_idx"]
    vocab = Vocabulary()
    for (word, idx) in vocab_indices
        vocab.word_to_idx[word] = idx
        vocab.idx_to_word = vocab_words
    end
    close(data)

    conn = DBInterface.connect(DuckDB.DB, db_path)
    AGI(
        vocab,
        WordEmbeddings(data["word_embeddings"]),
        PositionalEncoding(data["positional_enc"]),
        DocumentStore(data["documents"], data["doc_embeddings"]),
        config,
        conn
    )
end
