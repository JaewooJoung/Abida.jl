using Abida
using PDFIO

# --- Functions ---

function read_pdf_text(pdf_path)
    pdf = open(pdf_path, PDFDoc)
    text = ""
    for p in 1:getpagecount(pdf)
        page = getpage(pdf, p)
        text *= pagetotext(page)
    end
    close(pdf)
    return text
end

function clean_and_split_text(text)
    cleaned = replace(text, r"\r\n|\r|\n" => " ")
    cleaned = replace(cleaned, r"\s+" => " ")
    cleaned = strip(cleaned)
    
    sentences = split(cleaned, r"(?<=[.!?])\s+")
    valid_sentences = filter(s -> length(strip(s)) >= 20, sentences)
    
    return valid_sentences
end

function extract_pdf_text(pdf_path)
    println("Extracting text from: $(basename(pdf_path))")
    output_file = string(splitext(pdf_path)[1], ".txt")
    
    try
        text_content = read_pdf_text(pdf_path)
        sentences = clean_and_split_text(text_content)
        
        open(output_file, "w") do io
            for sentence in sentences
                println(io, strip(sentence))
            end
        end
        
        println("Total sentences found: $(length(sentences))")
        println("Saved to: $output_file")
        return output_file
    catch e
        @warn "Failed to process PDF" exception=e
        return nothing
    end
end

function learn_from_text(text_file, oracle)
    lines = filter(!isempty, [strip(l) for l in eachline(text_file)])
    count = length(lines)
    if count == 0 return end
    
    println("Learning from $count sentences...")
    
    DBInterface.execute(oracle.conn, "BEGIN TRANSACTION")
    try
        for line in lines
            learn!(oracle, line)
        end
        DBInterface.execute(oracle.conn, "COMMIT")
    catch e
        DBInterface.execute(oracle.conn, "ROLLBACK")
        rethrow(e)
    end
    
    println("Learned $count sentences")
end

# --- Main Logic ---

folder_path = "/media/crux/ss4/kdb"
db_file = joinpath(folder_path, "uknow.duckdb")

# Initialize AGI
try
    global oracle = AGI(db_file)
catch e
    @error "Failed to initialize AGI" exception=e
    exit()
end

# Get PDFs
pdf_files = filter(f -> endswith(lowercase(f), ".pdf"), readdir(folder_path))
println("Found $(length(pdf_files)) PDF files")

# Process each PDF
for (i, pdf_file) in enumerate(pdf_files)
    pdf_path = joinpath(folder_path, pdf_file)
    println("\nProcessing file $i/$(length(pdf_files)): $pdf_file")
    
    text_file = extract_pdf_text(pdf_path)
    if text_file === nothing continue end
    
    learn_from_text(text_file, oracle)
end

# Rethink once
println("Rethinking...")
rethink!(oracle, "Processed all books")

# Reiterate
println("Reiterating...")
reiterate!(oracle)

# Save state
println("Saving knowledge base...")
save(oracle, joinpath(folder_path, "oracle_state.jld2"))

# Test questions
questions = [
    "What is the main theme of these books?",
    "Who are the main characters?",
    "What happens in the stories?",
    "What are the common themes?"
]

println("\nTesting knowledge:")
for question in questions
    println("\nQ: $question")
    response, confidence, best_doc = answer(oracle, question)
    if confidence < 0.1f0
        response = "I'm unsure about that."
    end
    println("A: $response")
    println("Confidence: $confidence")
end

# Interactive loop
println("\nEnter your questions (type 'exit' to quit):")
while true
    print("\nYour question: ")
    question = readline()
    
    if lowercase(question) == "exit"
        break
    end
    
    if isempty(question)
        continue
    end

    # Show related sentences
    results = lookforword(oracle, question)
    if !isempty(results)
        println("\nRelevant info:")
        for res in results[1:min(3, length(results))]
            println("- $res")
        end
    else
        println("No relevant content found.")
    end

    # Get answer
    response, confidence, best_doc = answer(oracle, question)
    if confidence < 0.1f0
        response = "I'm unsure about that."
    end
    println("Answer: $response")
    println("Confidence: $confidence")
    
    # Log interaction
    DBInterface.execute(oracle.conn, """
        INSERT INTO interactions (question, answer, timestamp)
        VALUES (?, ?, datetime('now'))
    """, (question, response))
end

# Cleanup
cleanup!(oracle)
