using Test
using Abida
using Random

@testset "Abida.jl" begin
    # Create temp directory for test databases
    test_dir = mktempdir()

    @testset "Initialization" begin
        db_path = joinpath(test_dir, "init_test.duckdb")
        agi = AGI(db_path)
        @test agi isa AGI
        @test isempty(agi.documents)
        @test isempty(agi.doc_embeddings)
        cleanup!(agi)
    end

    @testset "Learning and Answering" begin
        db_path = joinpath(test_dir, "learn_test.duckdb")
        agi = AGI(db_path)

        # Test learning
        learn!(agi, "Julia is a fast programming language.")
        @test length(agi.documents) == 1
        @test length(agi.doc_embeddings) == 1
        @test !isempty(agi.vocab)

        # Test simple answer
        response, confidence, best_doc = answer(agi, "Tell me about Julia")
        @test response isa String
        @test !isempty(response)
        @test response != "No knowledge yet."
        @test confidence > 0.0
        @test !isempty(best_doc)

        cleanup!(agi)
    end

    @testset "Database Operations" begin
        db_path = joinpath(test_dir, "persist_test.duckdb")

        # Create first instance
        agi1 = AGI(db_path)
        learn!(agi1, "Test document 1")
        learn!(agi1, "Test document 2")
        cleanup!(agi1)

        # Create second instance with same database
        agi2 = AGI(db_path)
        @test length(agi2.documents) == 2
        @test length(agi2.doc_embeddings) == 2
        cleanup!(agi2)
    end

    @testset "Reset Knowledge" begin
        db_path = joinpath(test_dir, "reset_test.duckdb")
        agi = AGI(db_path)

        # Add some documents
        learn!(agi, "Document 1")
        learn!(agi, "Document 2")
        @test length(agi.documents) == 2

        # Reset knowledge
        reset_knowledge!(agi)

        @test isempty(agi.documents)
        @test isempty(agi.doc_embeddings)
        @test isempty(agi.vocab)

        cleanup!(agi)
    end

    @testset "Transformer Components" begin
        db_path = joinpath(test_dir, "transformer_test.duckdb")
        agi = AGI(db_path)

        # Test positional encoding
        @test size(agi.positional_enc) == (agi.config.max_seq_length, agi.config.d_model)

        # Test text encoding
        text = "test text"
        embeddings = encode_text(agi, text)
        @test size(embeddings, 1) == agi.config.d_model

        cleanup!(agi)
    end

    # Clean up test directory
    rm(test_dir, recursive=true, force=true)
end
