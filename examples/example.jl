using Abida
using PDFIO

# Extract text from PDF using pdftotext
function read_pdf_text(pdf_path)
    read(`pdftotext -layout -nopgbrk $pdf_path -`, String)
end

# Clean and split text into sentences
function clean_and_split_text(text)
    # Clean the text
    cleaned = replace(text, r"\r\n|\r|\n" => " ")  # Replace line breaks
    cleaned = replace(cleaned, r"\s+" => " ")      # Normalize whitespace
    cleaned = strip(cleaned)                       # Remove leading/trailing whitespace
    
    # Split into sentences and filter valid ones
    sentences = split(cleaned, r"(?<=[.!?])\s+|(?<=[.!?])$")
    valid_sentences = filter(s -> length(strip(s)) >= 20, sentences)
    
    return valid_sentences
end

# Extract PDF text and save to file
function extract_pdf_text(pdf_path)
    println("Extracting text from: $(basename(pdf_path))")
    
    # Create output path in same folder as PDF
    output_file = string(splitext(pdf_path)[1], ".txt")
    
    # Extract and process text
    text_content = read_pdf_text(pdf_path)
    sentences = clean_and_split_text(text_content)
    
    # Save sentences to file
    open(output_file, "w") do io
        for sentence in sentences
            println(io, strip(sentence))
        end
    end
    
    println("Total sentences found: $(length(sentences))")
    println("Saved to: $output_file")
    
    return output_file
end

# Learn from extracted text
function learn_from_text(text_file, oracle)
    println("Learning from: $(basename(text_file))")
    
    # Read and learn from each line
    count = 0
    for line in eachline(text_file)
        line = String(strip(line))  # Convert SubString to String
        if !isempty(line)
            learn!(oracle, line)
            count += 1
            if count % 100 == 0
                print(".")  # Progress indicator
            end
        end
    end
    
    println("\nLearned $count sentences from $(basename(text_file))")
end

# Main execution
folder_path = "/home/crux/문서/book2/"
db_file = joinpath(folder_path, "book_knowledge.duckdb")

# Initialize AGI once
oracle = AGI(db_file)

# Get all PDF files
pdf_files = filter(f -> endswith(lowercase(f), ".pdf"), readdir(folder_path))
println("Found $(length(pdf_files)) PDF files")

# Process each PDF file
for (i, pdf_file) in enumerate(pdf_files)
    pdf_path = joinpath(folder_path, pdf_file)
    
    println("\nProcessing file $i/$(length(pdf_files)): $pdf_file")
    
    # Extract text and get the output file path
    text_file = extract_pdf_text(pdf_path)
    
    # Learn from this file
    learn_from_text(text_file, oracle)
end

# Example questions to test knowledge
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
    println("A: $response")
    println("Confidence: $confidence")
    println("Best Document: $best_doc")
end

# Interactive mode
println("\nEnter your questions (type 'exit' to quit):")
while true
    print("\nYour question: ")
    question = readline()
    
    if lowercase(question) == "exit"
        break
    end
    
    if !isempty(question)
        response, confidence, best_doc = answer(oracle, question)
        println("Answer: $response")
        println("Confidence: $confidence")
        println("Best Document: $best_doc")
    end
end

# Cleanup
cleanup!(oracle)
