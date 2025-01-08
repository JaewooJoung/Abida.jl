using Abida
using HTTP
using Gumbo
using Cascadia
using Dates

# Function to fetch and parse a Wikipedia page
function fetch_wikipedia_page(title::String)
    url = "https://en.wikipedia.org/wiki/$title"
    response = HTTP.get(url)
    if response.status != 200
        error("Failed to fetch Wikipedia page: $(response.status)")
    end
    html = String(response.body)
    return html
end

# Function to extract text from Wikipedia HTML
function extract_wikipedia_text(html::String)
    doc = parsehtml(html)
    body = doc.root[2]  # The <body> tag is the second child of the root
    text_nodes = eachmatch(Selector("p"), body)  # Select all <p> tags
    text = join([node.text for node in text_nodes], "\n")
    return text
end

# Function to clean and split text into sentences
function clean_and_split_text(text::String)
    # Clean the text
    cleaned = replace(text, r"\r\n|\r|\n" => " ")  # Replace line breaks
    cleaned = replace(cleaned, r"\s+" => " ")      # Normalize whitespace
    cleaned = strip(cleaned)                       # Remove leading/trailing whitespace
    
    # Split into sentences and filter valid ones
    sentences = split(cleaned, r"(?<=[.!?])\s+|(?<=[.!?])$")
    valid_sentences = filter(s -> length(strip(s)) >= 20, sentences)
    
    return valid_sentences
end

# Function to learn from a Wikipedia page
function learn_from_wikipedia(ai::AGI, title::String)
    println("Fetching Wikipedia page: $title")
    html = fetch_wikipedia_page(title)
    text = extract_wikipedia_text(html)
    sentences = clean_and_split_text(text)
    
    println("Learning from Wikipedia page: $title")
    for sentence in sentences
        learn!(ai, sentence)
    end
    println("Finished learning from Wikipedia page: $title")
end

# Function to autonomously study from a list of Wikipedia pages
function autonomous_study(ai::AGI, topics::Vector{String}, interval::Second)
    println("Starting autonomous study...")
    while true
        for topic in topics
            try
                learn_from_wikipedia(ai, topic)
                println("Sleeping for $(interval.value) seconds...")
                sleep(interval.value)
            catch e
                @error "Error learning from topic $topic" exception=e
            end
        end
    end
end

# Main execution
db_path = "autonomous_knowledge.duckdb"
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
autonomous_study(ai, topics, Second(600))  # 600 seconds = 10 minutes
