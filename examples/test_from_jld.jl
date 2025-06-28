# test_from_jld2.jl
using Abida

function test_knowledge_from_jld2(jld2_path::String, db_path::String="loaded_knowledge.duckdb")
    println("🔄 Loading AGI from JLD2 file: $jld2_path")
    
    # Load the AGI from JLD2 file
    try
        config = DEFAULT_CONFIG
        agi = Abida.load(jld2_path, config, db_path)
        println("✅ Successfully loaded AGI with $(length(agi.documents)) documents")
        
        return agi
    catch e
        println("❌ Failed to load AGI from JLD2: $e")
        return nothing
    end
end

function interactive_test(agi::AGI)
    println("\n🤖 Interactive Knowledge Testing")
    println("Type your questions (or 'quit' to exit):")
    println("=" ^ 50)
    
    while true
        print("\n❓ Question: ")
        question = readline()
        
        if lowercase(strip(question)) in ["quit", "exit", "q", "stop"]
            println("👋 Goodbye!")
            break
        end
        
        if isempty(strip(question))
            println("Please enter a question.")
            continue
        end
        
        # Get answer
        response, confidence, best_doc = answer(agi, question)
        
        # Display results
        println("💡 Answer: $response")
        println("📊 Confidence: $(round(confidence, digits=4))")
        println("📄 Source document: $best_doc")
        println("-" ^ 50)
    end
end

function run_predefined_tests(agi::AGI)
    println("\n🧪 Running Predefined Knowledge Tests")
    println("=" ^ 50)
    
    # Comprehensive test questions covering different topics
    test_questions = [
        # Geography
        ("What is the capital of France?", "paris"),
        ("What is the tallest mountain in the world?", "everest"),
        ("Which river is the longest in the world?", "nile"),
        
        # Science
        ("At what temperature does water boil?", "100"),
        ("What converts sunlight into chemical energy?", "photosynthesis"),
        ("How many pairs of chromosomes do humans have?", "23"),
        
        # Technology
        ("What is Julia programming language used for?", "computing"),
        ("What type of database is DuckDB?", "sql"),
        ("What does AI stand for?", "artificial intelligence"),
        
        # Math
        ("What is the square root of 16?", "4"),
        ("What does the Pythagorean theorem state?", "triangle"),
        ("What is pi approximately equal to?", "3.14"),
        
        # History
        ("When did World War II end?", "1945"),
        ("When was the Declaration of Independence signed?", "1776"),
        ("When did the Berlin Wall fall?", "1989"),
        
        # Literature
        ("Who wrote Hamlet?", "shakespeare"),
        ("Who wrote 1984?", "orwell"),
        ("Who wrote Pride and Prejudice?", "austen"),
        
        # Animals
        ("How many hearts do octopuses have?", "three"),
        ("Where are kangaroos native to?", "australia"),
        ("What do bees help with?", "pollination"),
        
        # Space
        ("What is Mars known as?", "red planet"),
        ("How many planets are in our solar system?", "eight"),
        ("What orbits Earth and causes tides?", "moon"),
        
        # Random facts
        ("Do bananas count as berries?", "berries"),
        ("Does honey spoil?", "never"),
        ("What percentage of Earth's surface do rainforests cover?", "6")
    ]
    
    correct = 0
    total = length(test_questions)
    results = []
    
    for (i, (question, expected_keyword)) in enumerate(test_questions)
        response, confidence, _ = answer(agi, question)
        
        # Simple keyword matching for correctness
        is_correct = occursin(lowercase(expected_keyword), lowercase(response))
        
        if is_correct
            correct += 1
            status = "✅"
        else
            status = "❌"
        end
        
        push!(results, (question, response, confidence, is_correct))
        
        println("$status [$i/$total] Q: $question")
        println("   A: $response")
        println("   Confidence: $(round(confidence, digits=4))")
        
        if !is_correct
            println("   Expected keyword: '$expected_keyword'")
        end
        println()
    end
    
    # Summary
    accuracy = correct / total * 100
    println("📊 Test Results Summary")
    println("=" ^ 30)
    println("✅ Correct answers: $correct/$total")
    println("📈 Accuracy: $(round(accuracy, digits=1))%")
    println("📊 Average confidence: $(round(mean([r[3] for r in results]), digits=4))")
    
    # Show failed questions
    failed = filter(r -> !r[4], results)
    if !isempty(failed)
        println("\n❌ Failed Questions:")
        for (q, a, conf, _) in failed
            println("   Q: $q")
            println("   A: $a (conf: $(round(conf, digits=4)))")
        end
    end
    
    return accuracy, results
end

function browse_knowledge(agi::AGI)
    println("\n📚 Knowledge Base Browser")
    println("Commands: 'list N' (show N docs), 'search word', 'doc N' (show doc N), 'stats', 'quit'")
    println("=" ^ 70)
    
    while true
        print("\n📖 Command: ")
        input = strip(readline())
        
        if lowercase(input) in ["quit", "exit", "q"]
            break
        elseif input == "stats"
            println("📊 Knowledge Base Statistics:")
            println("   📄 Total documents: $(length(agi.documents))")
            println("   📝 Total vocabulary: $(length(agi.vocab.word_to_idx))")
            println("   🧠 Embedding dimensions: $(size(agi.word_embeddings.matrix))")
            
        elseif startswith(input, "list")
            parts = split(input)
            n = length(parts) > 1 ? parse(Int, parts[2]) : 10
            n = min(n, length(agi.documents))
            
            println("📄 First $n documents:")
            for i in 1:n
                doc = agi.documents[i]
                preview = length(doc) > 80 ? doc[1:80] * "..." : doc
                println("   [$i] $preview")
            end
            
        elseif startswith(input, "search")
            parts = split(input, " ", limit=2)
            if length(parts) > 1
                word = parts[2]
                results = lookforword(agi, word)
                println("🔍 Found $(length(results)) documents containing '$word':")
                for (i, doc) in enumerate(results[1:min(5, length(results))])
                    preview = length(doc) > 80 ? doc[1:80] * "..." : doc
                    println("   [$i] $preview")
                end
                if length(results) > 5
                    println("   ... and $(length(results) - 5) more")
                end
            else
                println("Usage: search <word>")
            end
            
        elseif startswith(input, "doc")
            parts = split(input)
            if length(parts) > 1
                try
                    n = parse(Int, parts[2])
                    if 1 <= n <= length(agi.documents)
                        println("📄 Document $n:")
                        println("   $(agi.documents[n])")
                    else
                        println("Document $n not found. Range: 1-$(length(agi.documents))")
                    end
                catch
                    println("Invalid document number")
                end
            else
                println("Usage: doc <number>")
            end
        else
            println("Unknown command. Type 'quit' to exit.")
        end
    end
end

function main()
    # Check if JLD2 file exists
    jld2_file = "agi_test_state.jld2"
    db_file = "test_knowledge_loaded.duckdb"
    
    if !isfile(jld2_file)
        println("❌ JLD2 file '$jld2_file' not found!")
        println("💡 Please run test_learn.jl first to create the knowledge base.")
        return
    end
    
    # Load AGI from JLD2
    agi = test_knowledge_from_jld2(jld2_file, db_file)
    if agi === nothing
        return
    end
    
    # Main menu
    while true
        println("\n🤖 AGI Knowledge Tester")
        println("=" ^ 30)
        println("1. 🧪 Run predefined tests")
        println("2. 💬 Interactive Q&A")
        println("3. 📚 Browse knowledge base")
        println("4. 🚪 Exit")
        print("\nChoose option (1-4): ")
        
        choice = strip(readline())
        
        if choice == "1"
            accuracy, _ = run_predefined_tests(agi)
        elseif choice == "2"
            interactive_test(agi)
        elseif choice == "3"
            browse_knowledge(agi)
        elseif choice == "4"
            println("👋 Goodbye!")
            break
        else
            println("Invalid choice. Please select 1-4.")
        end
    end
    
    # Cleanup
    cleanup!(agi)
    println("🧹 Cleaned up resources.")
end

# Run if called directly
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
