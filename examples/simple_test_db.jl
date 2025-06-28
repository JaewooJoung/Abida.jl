# simple_test_db.jl - Simple question bot for quick testing
using Abida

function main()
    println("🤖 Simple AGI Test Bot")
    println("=" ^ 30)
    
    # Load AGI (automatically finds database)
    db_files = filter(f -> endswith(f, ".duckdb"), readdir("."))
    
    if isempty(db_files)
        println("❌ No database files found!")
        println("💡 Run mass_learning_bot.jl first to train the AGI")
        return
    end
    
    # Use the most recently modified database
    db_path = db_files[end]  # or sort by modification time
    println("📚 Loading: $db_path")
    
    local agi  # Declare agi in the function scope
    
    try
        agi = AGI(db_path)
        println("✅ Loaded AGI with $(length(agi.documents)) facts")
    catch e
        println("❌ Failed to load AGI: $e")
        return
    end
    
    # Interactive Q&A
    println("\n💬 Ask me anything! (type 'quit' to exit)")
    println("💡 Try: 'What is the Sun?', 'Tell me about water', etc.")
    
    while true
        print("\n❓ Question: ")
        question = readline()
        
        if lowercase(strip(question)) in ["quit", "exit", "q", ""]
            break
        end
        
        try
            response, confidence, _ = answer(agi, question)
            
            println("🤖 Answer: $response")
            println("📊 Confidence: $(round(confidence, digits=3))")
            
            # Simple confidence interpretation
            if confidence > 0.5
                println("✅ High confidence")
            elseif confidence > 0.3
                println("⚠️ Medium confidence")
            else
                println("❓ Low confidence")
            end
            
        catch e
            println("❌ Error: $e")
        end
    end
    
    cleanup!(agi)
    println("👋 Goodbye!")
end

main()
