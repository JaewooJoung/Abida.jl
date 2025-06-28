# database.jl

const DuckDBHandle = Union{DuckDB.DB, DuckDB.Connection}

function init_database(db::DuckDBHandle)
    DBInterface.execute(db, """
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY,
            content TEXT NOT NULL
        )
    """)
    DBInterface.execute(db, """
        CREATE TABLE IF NOT EXISTS vocabulary (
            word TEXT PRIMARY KEY,
            index INTEGER NOT NULL
        )
    """)
    DBInterface.execute(db, """
        CREATE TABLE IF NOT EXISTS embeddings (
            doc_id INTEGER PRIMARY KEY,
            vector DOUBLE[] NOT NULL,
            FOREIGN KEY (doc_id) REFERENCES documents(id)
        )
    """)
    DBInterface.execute(db, """
        CREATE TABLE IF NOT EXISTS word_embeddings (
            vocab_index INTEGER PRIMARY KEY,
            vector DOUBLE[] NOT NULL
        )
    """)
    DBInterface.execute(db, """
        CREATE TABLE IF NOT EXISTS model_state (
            key TEXT PRIMARY KEY,
            value DOUBLE[] NOT NULL
        )
    """)
    DBInterface.execute(db, """
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY,
            question TEXT NOT NULL,
            feedback TEXT NOT NULL
        )
    """)
    DBInterface.execute(db, """
        CREATE TABLE IF NOT EXISTS sentence_relationships (
            sentence_id_1 INTEGER,
            sentence_id_2 INTEGER,
            strength FLOAT,
            FOREIGN KEY (sentence_id_1) REFERENCES documents(id),
            FOREIGN KEY (sentence_id_2) REFERENCES documents(id)
        )
    """)
    DBInterface.execute(db, """
        CREATE TABLE IF NOT EXISTS interactions (
            id INTEGER PRIMARY KEY,
            question TEXT,
            answer TEXT,
            timestamp DATETIME
        )
    """)
end

function load_data(db::DuckDBHandle, config::TransformerConfig)
    documents = String[]
    vocab = Dict{String,Int}()
    doc_embeddings = Vector{Float32}[]
    
    result = DBInterface.execute(db, "SELECT content FROM documents ORDER BY id")
    for row in result
        push!(documents, row.content)
    end

    result = DBInterface.execute(db, "SELECT word, index FROM vocabulary")
    for row in result
        vocab[row.word] = row.index
    end

    result = DBInterface.execute(db, "SELECT vector FROM embeddings ORDER BY doc_id")
    for row in result
        push!(doc_embeddings, Float32.(collect(row.vector)))
    end

    word_embeddings = zeros(Float32, config.d_model, max(1, length(vocab)))
    if length(vocab) > 0
        result = DBInterface.execute(db, "SELECT vocab_index, vector FROM word_embeddings ORDER BY vocab_index")
        for row in result
            word_embeddings[:, row.vocab_index] = Float32.(collect(row.vector))
        end
    end

    return documents, vocab, doc_embeddings, word_embeddings
end


function batch_insert_vocabulary(conn::DuckDB.DB, vocab::Dict{String,Int})
    values = [(word, idx) for (word, idx) in vocab]
    DBInterface.execute(conn, "BEGIN TRANSACTION")
    DBInterface.execute(conn, "INSERT INTO vocabulary (word, index) VALUES (?, ?)", values)
    DBInterface.execute(conn, "COMMIT")
end

function batch_insert_word_embeddings(conn::DuckDB.DB, word_embeddings::Matrix{Float32}, vocab::Dict{String,Int})
    values = [(idx, Float64.(word_embeddings[:, idx])) for (word, idx) in vocab]
    DBInterface.execute(conn, "BEGIN TRANSACTION")
    DBInterface.execute(conn, "INSERT INTO word_embeddings (vocab_index, vector) VALUES (?, ?)", values)
    DBInterface.execute(conn, "COMMIT")
end
