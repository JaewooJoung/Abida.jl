# Autonomous.jl
using Abida
using HTTP
using Gumbo
using Cascadia
using Dates
using URIs
using Logging
using InteractiveUtils

# Set up logging
global_logger(ConsoleLogger(stderr, Logging.Info))

# Function to fetch and parse a Wikipedia page
function fetch_wikipedia_page(title::String)
    encoded_title = replace(title, " " => "_")
    encoded_title = URI.encode(encoded_title)
    url = "https://en.wikipedia.org/wiki/ $encoded_title"
    headers = ["User-Agent" => "Julia-AutonomousStudyBot/1.0"]
    
    @info "Fetching Wikipedia page" url=url
    response = HTTP.get(url; headers=headers, redirect=true)
    
    if response.status != 200
        error("Failed to fetch Wikipedia page: $(response.status)")
    end
    
    return String(response.body)
end

# Function to extract text from Wikipedia HTML
function extract_wikipedia_text(html::String)
    doc = parsehtml(html)
    paras = eachmatch(Selector("div.mw-parser-output > p"), doc.root)
    text = join([text(x) for x in paras], "\n")
    return text
end

# Function to clean and split text into sentences
function clean_and_split_text(text::String)
    cleaned = replace(text, r"\r\n|\r|\n" => " ")
    cleaned = replace(cleaned, r"\s+" => " ")
    cleaned = strip(cleaned)
    
    sentences = split(cleaned, r"(?<=[.!?])\s+")
    valid_sentences = filter(s -> length(strip(s)) >= 20, sentences)
    
    return valid_sentences
end

# Function to learn from a Wikipedia page
function learn_from_wikipedia(ai::AGI, title::String)
    html = fetch_wikipedia_page(title)
    text = extract_wikipedia_text(html)
    sentences = clean_and_split_text(text)

    @info "Learning from Wikipedia page" title=title count=length(sentences)
    DBInterface.execute(ai.conn, "BEGIN TRANSACTION")
    try
        for sentence in sentences
            learn!(ai, sentence)
        end
        DBInterface.execute(ai.conn, "COMMIT")
    catch e
        DBInterface.execute(ai.conn, "ROLLBACK")
        rethrow()
    end
end

# Function to autonomously study from a list of Wikipedia pages
function autonomous_study(ai::AGI, topics::Vector{String}, interval::Second)
    @info "Starting autonomous study..." topics=topics
    counter = 0
    try
        while true
            for topic in topics
                try
                    learn_from_wikipedia(ai, topic)
                    counter += 1
                    
                    # Save periodically
                    if counter % 5 == 0
                        save_path = "agi_state_$(now(:local)).jld2"
                        save(ai, save_path)
                        @info "Saved AGI state" path=save_path
                    end
                    
                    sleep(rand(5:15))  # Avoid rate limiting
                catch e
                    @error "Error processing topic" topic=topic exception=e
                end
            end
            @info "Sleeping before next cycle" seconds=interval.value
            sleep(interval.value)
        end
    catch e
        isa(e, InterruptException) || rethrow()
        @info "Autonomous study interrupted by user"
    end
end

# Main execution
db_path = "autonomous_knowledge.duckdb"
@info "Initializing AGI" db=db_path
ai = AGI(db_path)

# List of topics to study
topics = [
    "Artificial_intelligence",
    "Machine_learning",
    "Natural_language_processing",
    "Computer_vision",
    "Reinforcement_learning"
]

# Start autonomous study (e.g., study each topic every 10 minutes)
@info "Starting autonomous learning loop"
autonomous_study(ai, topics, Second(600))  # 600 seconds = 10 minutes

# Cleanup
@info "Cleaning up"
cleanup!(ai)
