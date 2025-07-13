# simplified_autonomous_learner.jl - Minimal dependencies version
# Works with basic Julia packages + Abida.jl

using Abida
using SQLite
using HTTP
using Dates
using Logging
using Random

# Set up logging
function setup_logging()
    logger = ConsoleLogger(stdout, Logging.Info)
    global_logger(logger)
    @info "🧠 Simplified Autonomous Learning System Starting"
end

# Simple configuration
struct SimpleConfig
    dict_db_path::String
    abida_db_path::String
    learning_interval_seconds::Int
    words_per_cycle::Int
    max_articles_per_word::Int
    save_interval_cycles::Int
end

function SimpleConfig(;
    dict_db_path="/run/media/crux/ss4/test/newdict/DictSV.sqlite",
    abida_db_path="simple_autonomous_abida.duckdb",
    learning_interval_seconds=30,
    words_per_cycle=2,
    max_articles_per_word=2,
    save_interval_cycles=10
)
    SimpleConfig(dict_db_path, abida_db_path, learning_interval_seconds,
                words_per_cycle, max_articles_per_word, save_interval_cycles)
end

# Simple learner structure
mutable struct SimpleLearner
    config::SimpleConfig
    ai::AGI
    dict_db::SQLite.DB
    learning_stats::Dict{String, Any}
    processed_words::Set{String}
end

function SimpleLearner(config::SimpleConfig)
    @info "🚀 Initializing Simple Autonomous Learner"
    
    # Initialize basic Abida system
    @info "📚 Setting up Abida AI system"
    ai = AGI(config.abida_db_path)
    
    # Connect to Swedish dictionary
    @info "📖 Connecting to Swedish dictionary" path=config.dict_db_path
    if !isfile(config.dict_db_path)
        error("❌ Swedish dictionary not found at: $(config.dict_db_path)")
    end
    
    dict_db = SQLite.DB(config.dict_db_path)
    
    # Test dictionary connection
    try
        result = SQLite.execute(dict_db, "SELECT COUNT(*) FROM swedish_dict") |> collect
        word_count = result[1][1]
        @info "✅ Dictionary connected successfully" total_words=word_count
    catch e
        error("❌ Failed to read dictionary: $e")
    end
    
    # Initialize simple statistics
    learning_stats = Dict{String, Any}(
        "start_time" => now(),
        "cycles_completed" => 0,
        "words_processed" => 0,
        "articles_learned" => 0,
        "errors_encountered" => 0
    )
    
    processed_words = Set{String}()
    
    learner = SimpleLearner(config, ai, dict_db, learning_stats, processed_words)
    
    @info "🎯 Simple Autonomous Learner initialized successfully"
    show_simple_stats(learner)
    
    return learner
end

function get_words_from_dict(learner::SimpleLearner, count::Int=3)
    """Get random Swedish words from dictionary"""
    try
        # Reset processed words if we have too many
        if length(learner.processed_words) > 1000
            empty!(learner.processed_words)
            @info "🔄 Reset processed words for variety"
        end
        
        # Build exclusion clause
        exclusion = if !isempty(learner.processed_words)
            processed_list = collect(learner.processed_words)
            "AND ord NOT IN ($(join(["'$w'" for w in processed_list], ",")))"
        else
            ""
        end
        
        query = """
        SELECT ord, definition 
        FROM swedish_dict 
        WHERE LENGTH(ord) > 3 
        AND LENGTH(definition) > 10
        $exclusion
        ORDER BY RANDOM() 
        LIMIT $count
        """
        
        result = SQLite.execute(learner.dict_db, query) |> collect
        
        words_data = []
        for row in result
            push!(words_data, Dict("word" => row[1], "definition" => row[2]))
        end
        
        @info "📝 Selected words" words=[w["word"] for w in words_data]
        return words_data
        
    catch e
        @error "❌ Error getting words from dictionary" exception=e
        return []
    end
end

function simple_translate(word::String)
    """Simple translation using Google Translate"""
    try
        sleep(0.2)  # Be respectful
        
        encoded_word = HTTP.URIs.escapeuri(word)
        url = "https://translate.googleapis.com/translate_a/single?client=gtx&sl=sv&tl=en&dt=t&q=$encoded_word"
        
        response = HTTP.get(url, readtimeout=10)
        if response.status == 200
            # Simple parsing - just get the first translation
            body_str = String(response.body)
            # Look for pattern like [["translated_word"
            if occursin("[[\"", body_str)
                start_idx = findfirst("[[\"", body_str)
                if start_idx !== nothing
                    search_start = start_idx[end] + 1
                    end_idx = findfirst("\"", body_str[search_start:end])
                    if end_idx !== nothing
                        translated = body_str[search_start:search_start + end_idx[1] - 2]
                        return strip(translated)
                    end
                end
            end
        end
    catch e
        @warn "🌐 Translation failed for: $word" exception=e
    end
    
    return word
end

function fetch_wikipedia_simple(title::String)
    """Simple Wikipedia content fetching"""
    try
        sleep(1.0)  # Be respectful
        
        clean_title = replace(title, " " => "_")
        encoded_title = HTTP.URIs.escapeuri(clean_title)
        url = "https://en.wikipedia.org/wiki/$encoded_title"
        
        headers = ["User-Agent" => "Simple-Abida-Learner/1.0"]
        
        @info "📖 Fetching Wikipedia" title=title
        response = HTTP.get(url, headers=headers, readtimeout=20)
        
        if response.status == 200
            content = String(response.body)
            
            # Simple text extraction - look for content between <p> tags
            paragraphs = String[]
            
            # Find all <p>...</p> content
            p_start = 1
            while true
                p_open = findfirst("<p", content[p_start:end])
                if p_open === nothing
                    break
                end
                p_open = p_start + p_open[1] - 1
                
                p_close = findfirst("</p>", content[p_open:end])
                if p_close === nothing
                    break
                end
                p_close = p_open + p_close[end]
                
                # Extract text between tags
                tag_end = findfirst(">", content[p_open:p_close])
                if tag_end !== nothing
                    text_start = p_open + tag_end[1]
                    paragraph_text = content[text_start:p_close-4]
                    
                    # Clean up basic HTML entities and tags
                    clean_text = replace(paragraph_text, r"<[^>]*>" => "")
                    clean_text = replace(clean_text, "&amp;" => "&")
                    clean_text = replace(clean_text, "&lt;" => "<")
                    clean_text = replace(clean_text, "&gt;" => ">")
                    clean_text = replace(clean_text, "&quot;" => "\"")
                    clean_text = strip(clean_text)
                    
                    if length(clean_text) > 20
                        push!(paragraphs, clean_text)
                    end
                end
                
                p_start = p_close + 1
            end
            
            if !isempty(paragraphs)
                full_text = join(paragraphs[1:min(10, length(paragraphs))], "\n\n")
                @info "✅ Fetched Wikipedia content" title=title length=length(full_text)
                return full_text
            end
        end
        
    catch e
        @warn "❌ Error fetching Wikipedia" title=title exception=e
    end
    
    return nothing
end

function search_wikipedia_simple(word::String, max_results::Int=2)
    """Simple Wikipedia search"""
    try
        sleep(0.5)
        
        encoded_word = HTTP.URIs.escapeuri(word)
        search_url = "https://en.wikipedia.org/api/rest_v1/page/search/$encoded_word"
        headers = ["User-Agent" => "Simple-Abida-Learner/1.0"]
        
        response = HTTP.get(search_url, headers=headers, readtimeout=15)
        
        if response.status == 200
            # Simple JSON parsing for titles
            content = String(response.body)
            titles = String[]
            
            # Look for "title":"..." patterns
            title_pattern = r"\"title\":\"([^\"]+)\""
            matches = eachmatch(title_pattern, content)
            
            for match in matches
                title = match.captures[1]
                if !occursin("disambiguation", lowercase(title)) && 
                   !occursin("list of", lowercase(title)) &&
                   length(titles) < max_results
                    push!(titles, title)
                end
            end
            
            @info "🔍 Found Wikipedia topics" word=word topics=titles
            return titles
        end
        
    catch e
        @warn "❌ Wikipedia search failed" word=word exception=e
    end
    
    return String[]
end

function process_word_simple(learner::SimpleLearner, word_data::Dict)
    """Simple word processing"""
    word = word_data["word"]
    definition = word_data["definition"]
    
    @info "🔍 Processing word" word=word
    
    session_result = Dict{String, Any}(
        "word" => word,
        "articles_learned" => 0,
        "errors" => 0
    )
    
    try
        # Translate to English
        english_word = simple_translate(word)
        @info "🌐 Translation" swedish=word english=english_word
        
        # Search Wikipedia
        topics = search_wikipedia_simple(english_word, learner.config.max_articles_per_word)
        
        if !isempty(topics)
            for topic in topics
                content = fetch_wikipedia_simple(topic)
                
                if content !== nothing
                    # Create learning content
                    learning_text = """
                    LEARNING FROM WIKIPEDIA: $topic
                    SWEDISH WORD: $word
                    ENGLISH TRANSLATION: $english_word
                    DEFINITION: $definition
                    
                    CONTENT:
                    $content
                    """
                    
                    # Add to Abida knowledge base
                    learn!(learner.ai, learning_text)
                    session_result["articles_learned"] += 1
                    
                    @info "✅ Learned from article" topic=topic
                    sleep(1.0)  # Be respectful
                else
                    session_result["errors"] += 1
                end
            end
        else
            @warn "⚠️ No Wikipedia topics found for: $word"
            session_result["errors"] += 1
        end
        
        # Mark as processed
        push!(learner.processed_words, word)
        
    catch e
        @error "❌ Error processing word" word=word exception=e
        session_result["errors"] += 1
    end
    
    # Update stats
    learner.learning_stats["words_processed"] += 1
    learner.learning_stats["articles_learned"] += session_result["articles_learned"]
    learner.learning_stats["errors_encountered"] += session_result["errors"]
    
    @info "📊 Word completed" word=word articles=session_result["articles_learned"]
    
    return session_result
end

function learning_cycle_simple(learner::SimpleLearner)
    """Simple learning cycle"""
    cycle_start = now()
    cycle_num = learner.learning_stats["cycles_completed"] + 1
    
    @info "🔄 Starting learning cycle $cycle_num"
    
    try
        # Get words to learn
        words_to_learn = get_words_from_dict(learner, learner.config.words_per_cycle)
        
        if isempty(words_to_learn)
            @warn "⚠️ No words found, will retry next cycle"
            return
        end
        
        # Process each word
        for word_data in words_to_learn
            process_word_simple(learner, word_data)
            sleep(0.5)  # Brief pause between words
        end
        
        # Update cycle stats
        learner.learning_stats["cycles_completed"] += 1
        cycle_duration = now() - cycle_start
        
        @info "✅ Cycle completed" cycle=cycle_num duration=cycle_duration
        
        # Save periodically
        if learner.learning_stats["cycles_completed"] % learner.config.save_interval_cycles == 0
            save_simple_checkpoint(learner)
        end
        
        # Show stats every 5 cycles
        if learner.learning_stats["cycles_completed"] % 5 == 0
            show_simple_stats(learner)
        end
        
    catch e
        @error "❌ Error in learning cycle" exception=e
        learner.learning_stats["errors_encountered"] += 1
    end
end

function save_simple_checkpoint(learner::SimpleLearner)
    """Save simple checkpoint"""
    try
        timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
        checkpoint_path = "simple_checkpoint_$timestamp.jld2"
        
        @info "💾 Saving checkpoint" path=checkpoint_path
        save(learner.ai, checkpoint_path)
        
        @info "✅ Checkpoint saved" path=checkpoint_path cycles=learner.learning_stats["cycles_completed"]
        
    catch e
        @error "❌ Error saving checkpoint" exception=e
    end
end

function show_simple_stats(learner::SimpleLearner)
    """Show simple statistics"""
    elapsed = now() - learner.learning_stats["start_time"]
    elapsed_minutes = Dates.value(elapsed) / (1000 * 60)
    
    println("\n" * "="^60)
    println("🧠 SIMPLE AUTONOMOUS LEARNING STATUS")
    println("="^60)
    println("⏰ Runtime: $(Dates.format(elapsed, "HH:MM:SS"))")
    println("🔄 Cycles: $(learner.learning_stats["cycles_completed"])")
    println("📝 Words processed: $(learner.learning_stats["words_processed"])")
    println("📚 Articles learned: $(learner.learning_stats["articles_learned"])")
    println("❌ Errors: $(learner.learning_stats["errors_encountered"])")
    println("📄 Total documents: $(length(learner.ai.docs.documents))")
    println("📚 Vocabulary size: $(length(learner.ai.vocab.word_to_idx))")
    
    if elapsed_minutes > 0
        rate = learner.learning_stats["words_processed"] / elapsed_minutes
        println("📈 Learning rate: $(round(rate, digits=2)) words/min")
    end
    
    println("="^60 * "\n")
end

function cleanup_simple!(learner::SimpleLearner)
    """Simple cleanup"""
    try
        @info "🧹 Cleaning up"
        
        save_simple_checkpoint(learner)
        SQLite.close(learner.dict_db)
        cleanup!(learner.ai)
        
        show_simple_stats(learner)
        @info "✅ Cleanup completed"
        
    catch e
        @error "❌ Error during cleanup" exception=e
    end
end

function run_simple_learning(config::SimpleConfig=SimpleConfig())
    """Main function for simple learning"""
    setup_logging()
    
    @info "🚀 Starting Simple Autonomous Learning"
    @info "📖 Using Swedish dictionary for vocabulary learning"
    @info "⚠️ Press Ctrl+C to stop"
    
    learner = nothing
    
    try
        learner = SimpleLearner(config)
        
        @info "🎯 Beginning learning loop"
        
        while true
            learning_cycle_simple(learner)
            
            @info "⏳ Waiting before next cycle" seconds=config.learning_interval_seconds
            sleep(config.learning_interval_seconds)
        end
        
    catch e
        if isa(e, InterruptException)
            @info "🛑 Learning stopped by user"
        else
            @error "❌ Unexpected error" exception=e
        end
    finally
        if learner !== nothing
            cleanup_simple!(learner)
        end
    end
    
    @info "🏁 Simple Learning System stopped"
end

# Different modes
function run_fast_simple()
    config = SimpleConfig(learning_interval_seconds=15, words_per_cycle=1, max_articles_per_word=1)
    run_simple_learning(config)
end

function run_intensive_simple()
    config = SimpleConfig(learning_interval_seconds=45, words_per_cycle=3, max_articles_per_word=2)
    run_simple_learning(config)
end

function run_gentle_simple()
    config = SimpleConfig(learning_interval_seconds=90, words_per_cycle=2, max_articles_per_word=1)
    run_simple_learning(config)
end

# Main execution
if abspath(PROGRAM_FILE) == @__FILE__
    println("🇸🇪 Simple Autonomous Learning from Swedish Dictionary")
    println("🧠 Powered by Abida.jl (Minimal Dependencies)")
    println("="^60)
    
    if length(ARGS) > 0
        mode = ARGS[1]
        if mode == "fast"
            println("🚀 Running FAST mode")
            run_fast_simple()
        elseif mode == "intensive"
            println("💪 Running INTENSIVE mode")
            run_intensive_simple()
        elseif mode == "gentle"
            println("🌱 Running GENTLE mode")
            run_gentle_simple()
        else
            println("🎯 Running DEFAULT mode")
            run_simple_learning()
        end
    else
        run_simple_learning()
    end
end