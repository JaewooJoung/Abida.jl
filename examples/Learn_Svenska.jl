# svenska_wlbot_async.jl - Asynchronous Swedish Wikipedia Learning Bot
using Abida
using HTTP
using JSON3
using Dates
using Gumbo
using Cascadia

# Wikipedia API configuration
const WIKIPEDIA_API_BASE = "https://sv.wikipedia.org/api/rest_v1"  # Swedish Wikipedia
const WIKIPEDIA_SEARCH_API = "https://sv.wikipedia.org/w/api.php"
const SVENSKA_SE_BASE = "https://svenska.se/so/"

# ========== PERFORMANCE TUNING PARAMETERS ==========
# Concurrent request limits
const MAX_CONCURRENT_REQUESTS = 10  # 🔧 INCREASE for faster (try 15-20), DECREASE for stability (5-8)
const BATCH_SIZE = 10              # 🔧 INCREASE for faster (try 20-30), DECREASE for stability (5)

# Learning parameters
const TOTAL_WORDS_TO_LEARN = 1000  # 🔧 CHANGE total words to learn (100, 500, 1000, 5000, etc.)
const WORDS_PER_MEGABATCH = 100    # 🔧 Words processed before saving (50, 100, 200)
const WIKI_ARTICLES_PER_WORD = 5   # 🔧 Number of Wikipedia articles per word (1-10)
const MIN_SENTENCE_LENGTH = 20      # 🔧 Minimum sentence length to learn (10-50)

# Timing parameters
const DELAY_BETWEEN_BATCHES = 2    # 🔧 Seconds to wait between mega-batches (0-10)
const REQUEST_RETRY_DELAY = 0.5    # 🔧 Seconds to wait between retries (0.1-2)
# ====================================================

struct SwedishWikipediaBot
    agi::AGI
    learned_articles::Set{String}
    learned_words::Set{String}
    learning_stats::Dict{String, Any}
    request_semaphore::Base.Semaphore
end

function SwedishWikipediaBot(db_path::String="svenska.duckdb")
    """Create a Swedish Wikipedia learning bot with AGI and concurrency support"""
    
    println("🇸🇪 Initialiserar asynkron svensk Wikipedia-inlärningsbot...")
    
    # Load or create AGI
    agi = AGI(db_path)
    println("✅ AGI initialiserad med $(length(agi.documents)) befintliga dokument")
    
    # Initialize tracking
    learned_articles = Set{String}()
    learned_words = Set{String}()
    learning_stats = Dict(
        "articles_processed" => 0,
        "words_learned" => 0,
        "facts_learned" => 0,
        "translations_made" => 0,
        "start_time" => now(),
        "errors" => 0
    )
    
    # Semaphore for rate limiting
    request_semaphore = Base.Semaphore(MAX_CONCURRENT_REQUESTS)
    
    return SwedishWikipediaBot(agi, learned_articles, learned_words, learning_stats, request_semaphore)
end

# Async HTTP request wrapper
function async_http_get(url::String, semaphore::Base.Semaphore)
    """Perform async HTTP GET with rate limiting"""
    return @async begin
        Base.acquire(semaphore)
        try
            response = HTTP.get(url)
            return response
        finally
            Base.release(semaphore)
        end
    end
end

# Async Swedish word scraping
function async_scrape_swedish_word(url::String, semaphore::Base.Semaphore)
    """Asynchronously scrape Swedish word data"""
    return @async begin
        Base.acquire(semaphore)
        try
            response = HTTP.get(url)
            html_content = String(response.body)
            parsed_html = parsehtml(html_content)
            
            # Define CSS selectors
            selectors = Dict(
                :word => Selector(".orto"),
                :inflections => Selector(".bojning_inline .bojning"),
                :part_of_speech => Selector(".ordklass"),
                :pronunciation => Selector(".uttal"),
                :definition => Selector(".kbetydelse .def"),
                :synonym => Selector(".hv .hvtag"),
                :example => Selector(".syntex"),
                :history => Selector(".etymologi .fb"),
                :published => Selector(".tryck")
            )
            
            # Extract data
            extracted_data = Dict()
            for (key, selector) in selectors
                elements = eachmatch(selector, parsed_html.root)
                if !isempty(elements)
                    extracted_data[key] = replace(strip(nodeText(elements[1])), "\n" => " ")
                else
                    extracted_data[key] = "Ej hittat"
                end
            end
            
            # Format the extracted data
            formatted_string = """
            Ord: $(extracted_data[:word])
            Böjningar: $(extracted_data[:inflections])
            Ordklass: $(extracted_data[:part_of_speech])
            Uttal: $(extracted_data[:pronunciation])
            Definition: $(extracted_data[:definition])
            Synonym: $(extracted_data[:synonym])
            Exempel: $(extracted_data[:example])
            Historia: $(extracted_data[:history])
            Publicerad: $(extracted_data[:published])
            """
            
            return (formatted_string, extracted_data[:word], extracted_data)
        catch e
            return (nothing, nothing, nothing)
        finally
            Base.release(semaphore)
        end
    end
end

# Async translation
function async_gtranslate(text::String, targetlang::String, semaphore::Base.Semaphore, sourcelang::String="auto")
    """Asynchronously translate text"""
    return @async begin
        Base.acquire(semaphore)
        try
            url = "https://translate.googleapis.com/translate_a/single?client=gtx&sl=$sourcelang&tl=$targetlang&dt=t&q=" * HTTP.URIs.escapeuri(text)
            response = HTTP.get(url)
            result = JSON3.read(response.body)
            
            if !isempty(result) && !isempty(result[1])
                translated = join([s[1] for s in result[1]], "")
                return translated
            else
                return nothing
            end
        catch e
            return nothing
        finally
            Base.release(semaphore)
        end
    end
end

# Async Wikipedia search
function async_search_swedish_wikipedia(query::String, limit::Int, semaphore::Base.Semaphore)
    """Asynchronously search Swedish Wikipedia"""
    return @async begin
        Base.acquire(semaphore)
        try
            params = [
                "action" => "query",
                "format" => "json",
                "list" => "search",
                "srsearch" => query,
                "srlimit" => limit,
                "srprop" => "snippet",
                "uselang" => "sv"
            ]
            
            url = WIKIPEDIA_SEARCH_API * "?" * HTTP.URIs.escapeuri(params)
            response = HTTP.get(url)
            
            if response.status == 200
                data = JSON3.read(response.body)
                
                articles = []
                if haskey(data, :query) && haskey(data.query, :search)
                    for result in data.query.search
                        push!(articles, (
                            title = result.title,
                            snippet = get(result, :snippet, ""),
                            pageid = result.pageid
                        ))
                    end
                end
                
                return articles
            else
                return []
            end
        catch e
            return []
        finally
            Base.release(semaphore)
        end
    end
end

# Async Wikipedia summary fetch
function async_get_swedish_wikipedia_summary(title::String, semaphore::Base.Semaphore)
    """Asynchronously get Wikipedia article summary"""
    return @async begin
        Base.acquire(semaphore)
        try
            clean_title = replace(title, " " => "_")
            url = "$WIKIPEDIA_API_BASE/page/summary/$clean_title"
            response = HTTP.get(url)
            
            if response.status == 200
                data = JSON3.read(response.body)
                if haskey(data, :extract) && !isempty(data.extract)
                    return data.extract
                end
            end
            return nothing
        catch e
            return nothing
        finally
            Base.release(semaphore)
        end
    end
end

# Process a batch of words concurrently
function process_word_batch_async(bot::SwedishWikipediaBot, word_ids::Vector{Int})
    """Process multiple words concurrently"""
    
    # Tasks for fetching word data
    word_tasks = []
    for word_id in word_ids
        url = "$(SVENSKA_SE_BASE)?id=$word_id&pz=3#!"
        push!(word_tasks, async_scrape_swedish_word(url, bot.request_semaphore))
    end
    
    # Wait for all word fetches to complete
    word_results = []
    for (i, task) in enumerate(word_tasks)
        try
            result = fetch(task)
            if result[2] !== nothing  # word found
                push!(word_results, (word_id=word_ids[i], data=result))
            end
        catch e
            bot.learning_stats["errors"] += 1
        end
    end
    
    println("📊 Hämtade $(length(word_results)) giltiga ord av $(length(word_ids)) försök")
    
    # Process each valid word
    for word_result in word_results
        formatted, word, word_data = word_result.data
        
        if word in bot.learned_words
            continue
        end
        
        println("\n📚 Bearbetar ord: $word")
        
        # Start all async operations for this word
        tasks = Dict()
        
        # Translation tasks
        if word !== nothing
            tasks[:en_translation] = async_gtranslate(word, "en", bot.request_semaphore, "sv")
            tasks[:ko_translation] = async_gtranslate(word, "ko", bot.request_semaphore, "sv")
            
            # Wikipedia search task
            tasks[:wiki_search] = async_search_swedish_wikipedia(word, WIKI_ARTICLES_PER_WORD, bot.request_semaphore)
        end
        
        # Learn word definition while waiting for async tasks
        facts_learned = 0
        
        if haskey(word_data, :definition) && word_data[:definition] != "Ej hittat"
            fact = "Det svenska ordet '$word' betyder: $(word_data[:definition])"
            learn!(bot.agi, fact)
            facts_learned += 1
        end
        
        if haskey(word_data, :part_of_speech) && word_data[:part_of_speech] != "Ej hittat"
            fact = "Det svenska ordet '$word' är $(word_data[:part_of_speech])"
            learn!(bot.agi, fact)
            facts_learned += 1
        end
        
        if haskey(word_data, :example) && word_data[:example] != "Ej hittat"
            fact = "Exempel på '$word' i en mening: $(word_data[:example])"
            learn!(bot.agi, fact)
            facts_learned += 1
        end
        
        # Fetch translation results
        try
            en_translation = fetch(tasks[:en_translation])
            if en_translation !== nothing
                fact = "The Swedish word '$word' means '$en_translation' in English"
                learn!(bot.agi, fact)
                facts_learned += 1
                bot.learning_stats["translations_made"] += 1
            end
        catch e
            bot.learning_stats["errors"] += 1
        end
        
        try
            ko_translation = fetch(tasks[:ko_translation])
            if ko_translation !== nothing
                fact = "스웨덴어 '$word'는 한국어로 '$ko_translation'입니다"
                learn!(bot.agi, fact)
                facts_learned += 1
                bot.learning_stats["translations_made"] += 1
            end
        catch e
            bot.learning_stats["errors"] += 1
        end
        
        # Process Wikipedia articles
        try
            articles = fetch(tasks[:wiki_search])
            
            if !isempty(articles)
                # Fetch all article summaries concurrently
                summary_tasks = []
                for article in articles
                    if !(article.title in bot.learned_articles)
                        push!(summary_tasks, (
                            title = article.title,
                            task = async_get_swedish_wikipedia_summary(article.title, bot.request_semaphore)
                        ))
                    end
                end
                
                # Process summaries
                for summary_info in summary_tasks
                    try
                        summary = fetch(summary_info.task)
                        if summary !== nothing
                            # Learn all sentences
                            sentences = split(summary, r"[.!?]+")
                            for sentence in sentences
                                clean_sentence = strip(sentence)
                                if length(clean_sentence) > MIN_SENTENCE_LENGTH
                                    learn!(bot.agi, clean_sentence * ".")
                                    facts_learned += 1
                                end
                            end
                            
                            push!(bot.learned_articles, summary_info.title)
                            bot.learning_stats["articles_processed"] += 1
                        end
                    catch e
                        bot.learning_stats["errors"] += 1
                    end
                end
            end
        catch e
            bot.learning_stats["errors"] += 1
        end
        
        # Update stats
        push!(bot.learned_words, word)
        bot.learning_stats["words_learned"] += 1
        bot.learning_stats["facts_learned"] += facts_learned
        
        println("  ✅ Lärt $facts_learned fakta för '$word'")
    end
end

function random_swedish_learning_session_async(bot::SwedishWikipediaBot, num_words::Int=WORDS_PER_MEGABATCH)
    """Learn random Swedish words using async operations"""
    
    println("\n🎲 Slumpmässig svensk inlärningssession")
    println(repeat("=", 50))
    
    # Generate random word IDs
    word_ids = [rand(100001:198119) for _ in 1:num_words]
    
    # Process in batches
    batches = [word_ids[i:min(i+BATCH_SIZE-1, length(word_ids))] for i in 1:BATCH_SIZE:length(word_ids)]
    
    for (batch_num, batch) in enumerate(batches)
        println("\n📦 Bearbetar batch $batch_num av $(length(batches)) ($(length(batch)) ord)")
        process_word_batch_async(bot, batch)
    end
end

function show_swedish_stats(bot::SwedishWikipediaBot)
    """Display Swedish learning statistics"""
    
    elapsed = now() - bot.learning_stats["start_time"]
    elapsed_minutes = Dates.value(elapsed) / 60000
    
    println("\n📊 Svensk inlärningsstatistik")
    println(repeat("=", 30))
    println("📚 Totala dokument: $(length(bot.agi.documents))")
    println("🇸🇪 Svenska ord lärda: $(bot.learning_stats["words_learned"])")
    println("📖 Artiklar bearbetade: $(bot.learning_stats["articles_processed"])")
    println("🧠 Fakta lärda: $(bot.learning_stats["facts_learned"])")
    println("🌐 Översättningar gjorda: $(bot.learning_stats["translations_made"])")
    println("⚠️ Fel: $(bot.learning_stats["errors"])")
    println("⏱️ Sessionstid: $(round(elapsed, Minute))")
    println("📈 Inlärningshastighet: $(round(bot.learning_stats["words_learned"] / max(1, elapsed_minutes), digits=2)) ord/min")
    println("📈 Faktahastighet: $(round(bot.learning_stats["facts_learned"] / max(1, elapsed_minutes), digits=2)) fakta/min")
end

function save_swedish_bot_state(bot::SwedishWikipediaBot, filename::String="svenska_bot_state.jld2")
    """Save the Swedish bot's current state"""
    
    try
        JLD2.jldopen(filename, "w") do file
            file["vocab_idx_to_word"] = bot.agi.vocab.idx_to_word
            file["vocab_word_to_idx"] = bot.agi.vocab.word_to_idx
            file["word_embeddings"] = bot.agi.word_embeddings.matrix
            file["positional_enc"] = bot.agi.positional_enc.matrix
            file["documents"] = bot.agi.docs.documents
            file["doc_embeddings"] = bot.agi.docs.embeddings
            file["learned_articles"] = collect(bot.learned_articles)
            file["learned_words"] = collect(bot.learned_words)
            file["learning_stats"] = bot.learning_stats
        end
        
        println("💾 Bot-tillstånd sparat till: $filename")
        println("📊 Sparade $(length(bot.agi.documents)) dokument")
        return true
    catch e
        println("❌ Misslyckades med att spara bot-tillstånd: $e")
        return false
    end
end

function main()
    println("🇸🇪 Asynkron svensk Wikipedia-inlärningsbot för Abida")
    println(repeat("=", 50))
    println("🚀 Använder upp till $MAX_CONCURRENT_REQUESTS samtidiga anslutningar")
    println("📦 Bearbetar ord i batchar om $BATCH_SIZE")
    
    # Create Swedish bot
    bot = SwedishWikipediaBot("svenska.duckdb")
    
    # Main learning loop - configurable number of words
    total_megabatches = div(TOTAL_WORDS_TO_LEARN, WORDS_PER_MEGABATCH)
    
    println("\n🎯 Startar massiv asynkron inlärning - $TOTAL_WORDS_TO_LEARN svenska ord...")
    println("📦 Uppdelat i $total_megabatches mega-batchar om $WORDS_PER_MEGABATCH ord vardera")
    println("Detta kommer att gå mycket snabbare med parallell bearbetning!")
    println(repeat("=", 50))
    
    start_time = now()
    
    try
        # Learn words in configurable mega-batches
        for batch in 1:total_megabatches
            batch_start = now()
            
            println("\n\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            println("📦 MEGA-BATCH $batch av $total_megabatches (ord $(((batch-1)*WORDS_PER_MEGABATCH)+1) - $(batch*WORDS_PER_MEGABATCH))")
            println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            
            random_swedish_learning_session_async(bot, WORDS_PER_MEGABATCH)
            
            # Save state after each mega-batch
            println("\n💾 Sparar efter mega-batch $batch...")
            save_swedish_bot_state(bot)
            
            # Show stats
            show_swedish_stats(bot)
            
            batch_elapsed = now() - batch_start
            println("⏱️ Batch-tid: $(round(batch_elapsed, Second))")
            
            # Configurable break between mega-batches
            if batch < total_megabatches && DELAY_BETWEEN_BATCHES > 0
                println("\n⏸️ Kort paus... ($DELAY_BETWEEN_BATCHES sekunder)")
                sleep(DELAY_BETWEEN_BATCHES)
            end
        end
        
    catch e
        if isa(e, InterruptException)
            println("\n\n⚠️ Inlärning avbruten av användaren!")
        else
            println("\n\n❌ Ett fel uppstod: $e")
            @show e
        end
    end
    
    # Final stats
    total_elapsed = now() - start_time
    println("\n\n🏁 SLUTLIG SAMMANFATTNING")
    println(repeat("=", 50))
    println("⏱️ Total tid: $(round(total_elapsed, Minute))")
    println("🚀 Genomsnittlig hastighet: $(round(bot.learning_stats["words_learned"] / (Dates.value(total_elapsed) / 60000), digits=2)) ord/minut")
    
    save_swedish_bot_state(bot)
    show_swedish_stats(bot)
    
    # Cleanup
    cleanup!(bot.agi)
    
    println("\n🎉 Asynkron svensk Wikipedia-inlärningssession avslutad!")
    println("📚 Totalt lärda dokument sparade i: svenska.duckdb")
end

# Run the bot
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
