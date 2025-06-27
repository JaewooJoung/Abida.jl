# core.jl

using DuckDB
using Logging
using JLD2

include("types.jl")
include("transformer_utils.jl")
include("database.jl")

function AGI(db_path::String="Abida.duckdb", config::TransformerConfig=DEFAULT_CONFIG)
    try
        conn = DBInterface.connect(DuckDB.DB, db_path)
        init_database(conn)

        documents, vocab_dict, doc_embeddings, word_embeddings_matrix = load_data(conn, config)
        vocab = Vocabulary(vocab_dict, ["" for _ in 1:length(vocab_dict)])
        for (word, idx) in vocab_dict
            vocab.idx_to_word[idx] = word
        end

        positional_enc = positional_encoding(config.max_seq_length, config.d_model)

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
            embeddings[:, i] = ai.word_embeddings.matrix[:, word_idx]
        else
            embeddings[:, i] = randn(Float32, ai.config.d_model)
        end
        embeddings[:, i] += ai.positional_enc.matrix[:, i]
    end

    return embeddings
end

function transformer_encode(ai::AGI, embeddings::Matrix{Float32})
    attn_output = multi_head_attention(embeddings, embeddings, embeddings, ai.config.n_head)
    attn_output = attn_output + embeddings
    attn_output = layer_norm(attn_output)

    ff_output = feed_forward(attn_output, ai.config.d_ff)
    ff_output = ff_output + attn_output
    ff_output = layer_norm(ff_output)

    return vec(mean(ff_output, dims=2))
end

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
    old_size = size(ai.word_embeddings.matrix, 2)
    if vocab_changed || old_size < vocab_size
        new_cols = vocab_size - old_size
        new_init = randn(Float32, ai.config.d_model, new_cols)
        ai.word_embeddings.matrix = hcat(ai.word_embeddings.matrix, new_init)
        batch_insert_word_embeddings(ai.conn, new_init, ai.vocab.word_to_idx)
    end

    embeddings = encode_text(ai, text)
    doc_embedding = normalize(transformer_encode(ai, embeddings))
    push!(ai.docs.embeddings, doc_embedding)

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

function answer(ai::AGI, question::String)
    isempty(ai.docs.embeddings) && return "No knowledge yet."

    q_embedding = normalize(transformer_encode(ai, encode_text(ai, question)))
    scores = [dot(q_embedding, emb) for emb in ai.docs.embeddings]
    best_idx = argmax(scores)
    return ai.docs.documents[best_idx]
end

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
    end
    vocab.idx_to_word = vocab_words

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

# Placeholder stubs for exported functions
cleanup!(ai::AGI) = @warn "Not implemented"
reset_knowledge!(ai::AGI) = @warn "Not implemented"
rethink!(ai::AGI, text::String) = @warn "Not implemented"
reiterate!(ai::AGI, text::String) = @warn "Not implemented"
lookforword(ai::AGI, word::String) = @warn "Not implemented"
evaluate(ai::AGI, text::String) = @warn "Not implemented"
answer_with_fallback(ai::AGI, question::String, fallback::String) = answer(ai, question)
