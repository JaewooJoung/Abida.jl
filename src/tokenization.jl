# tokenization.jl - Enhanced tokenization with BPE support

using JSON3
using OrderedCollections
using Unicode

struct BPETokenizer
    vocab::Dict{String, Int}
    merges::Vector{Tuple{String, String}}
    special_tokens::Dict{String, Int}
    max_token_length::Int
    unk_token::String
    pad_token::String
    eos_token::String
end

function BPETokenizer(vocab_file::String, merges_file::String)
    # Load vocabulary
    vocab_data = JSON3.read(read(vocab_file, String))
    vocab = Dict{String, Int}(k => v for (k, v) in vocab_data)

        # Load merge rules
        merges = Tuple{String, String}[]
        for line in readlines(merges_file)
            if !startswith(line, "#") && !isempty(strip(line))
                parts = split(strip(line))
                if length(parts) >= 2
                    push!(merges, (parts[1], parts[2]))
                end
            end
        end

        # Define special tokens
        special_tokens = Dict(
            "<|endoftext|>" => get(vocab, "<|endoftext|>", length(vocab) + 1),
            "<|unk|>" => get(vocab, "<|unk|>", length(vocab) + 2),
            "<|pad|>" => get(vocab, "<|pad|>", length(vocab) + 3),
            "<|bos|>" => get(vocab, "<|bos|>", length(vocab) + 4),
            "<|eos|>" => get(vocab, "<|eos|>", length(vocab) + 5)
            )

        # Add special tokens to vocab if not present
        for (token, id) in special_tokens
            if !haskey(vocab, token)
                vocab[token] = id
            end
        end

        BPETokenizer(vocab, merges, special_tokens, 50, "<|unk|>", "<|pad|>", "<|endoftext|>")
    end

    # Simple tokenizer constructor for when you don't have pre-trained BPE
    function BPETokenizer()
        vocab = Dict{String, Int}()
        merges = Tuple{String, String}[]

        special_tokens = Dict(
            "<|endoftext|>" => 1,
            "<|unk|>" => 2,
            "<|pad|>" => 3,
            "<|bos|>" => 4,
            "<|eos|>" => 5
            )

        # Add special tokens to vocab
        for (token, id) in special_tokens
            vocab[token] = id
        end

        BPETokenizer(vocab, merges, special_tokens, 50, "<|unk|>", "<|pad|>", "<|endoftext|>")
    end

    function get_pairs(word::Vector{String})
        pairs = Set{Tuple{String, String}}()
        if length(word) < 2
            return pairs
        end

        prev_char = word[1]
        for char in word[2:end]
            push!(pairs, (prev_char, char))
            prev_char = char
        end
        return pairs
    end

    function bpe_encode(tokenizer::BPETokenizer, word::String)
        if length(word) == 1
            return [word]
        end

        # Convert word to character array with special end-of-word marker
        word_tokens = [string(c) for c in word]
            if length(word_tokens) > 0
                word_tokens[end] = word_tokens[end] * "</w>"
            end

            while length(word_tokens) > 1
                pairs = get_pairs(word_tokens)
                if isempty(pairs)
                    break
                end

                # Find the highest priority merge
                bigram = nothing
                min_rank = typemax(Int)

                for pair in pairs
                    rank = findfirst(x -> x == pair, tokenizer.merges)
                    if rank !== nothing && rank < min_rank
                        min_rank = rank
                        bigram = pair
                    end
                end

                if bigram === nothing
                    break
                end

                # Apply the merge
                first, second = bigram
                new_word = String[]
                i = 1
                while i <= length(word_tokens)
                    if i < length(word_tokens) && word_tokens[i] == first && word_tokens[i+1] == second
                        push!(new_word, first * second)
                        i += 2
                    else
                        push!(new_word, word_tokens[i])
                        i += 1
                    end
                end
                word_tokens = new_word
            end

            return word_tokens
        end

        function preprocess_text(text::String)
            """
            Preprocess text before tokenization
            """
            # Normalize unicode
            text = Unicode.normalize(text, :NFC)

            # Handle contractions and common patterns
            text = replace(text, r"'re" => " are")
            text = replace(text, r"'ve" => " have")
            text = replace(text, r"'ll" => " will")
            text = replace(text, r"'d" => " would")
            text = replace(text, r"'m" => " am")
            text = replace(text, r"n't" => " not")
            text = replace(text, r"'s" => " 's")  # Keep possessive 's separate

            # Normalize whitespace
            text = replace(text, r"\s+" => " ")
            text = strip(text)

            return text
        end

        function word_split(text::String)
            """
            Split text into words with better handling of punctuation
            """
            # Add spaces around punctuation
            text = replace(text, r"([.!?,:;\"'()\[\]{}])" => s" \1 ")

            # Handle numbers and special cases
            text = replace(text, r"(\d+)([a-zA-Z]+)" => s"\1 \2")  # Split "123abc" -> "123 abc"
            text = replace(text, r"([a-zA-Z]+)(\d+)" => s"\1 \2")  # Split "abc123" -> "abc 123"

            # Normalize whitespace again
            text = replace(text, r"\s+" => " ")
            text = strip(text)

            return split(text)
        end

        function tokenize(tokenizer::BPETokenizer, text::String; add_special_tokens::Bool=false)
            """
            Tokenize text using BPE
            """
            # Preprocess text
            text = preprocess_text(text)

            # Split into words
            words = word_split(text)

            tokens = String[]

            # Add beginning of sequence token if requested
            if add_special_tokens && haskey(tokenizer.special_tokens, "<|bos|>")
                push!(tokens, "<|bos|>")
            end

            for word in words
                if haskey(tokenizer.special_tokens, word)
                    # It's a special token
                    push!(tokens, word)
                    elseif haskey(tokenizer.vocab, word)
                    # Word is directly in vocabulary
                    push!(tokens, word)
                else
                    # Apply BPE encoding
                    bpe_tokens = bpe_encode(tokenizer, lowercase(word))
                    append!(tokens, bpe_tokens)
                end
            end

            # Add end of sequence token if requested
            if add_special_tokens && haskey(tokenizer.special_tokens, tokenizer.eos_token)
                push!(tokens, tokenizer.eos_token)
            end

            return tokens
        end

        function tokens_to_ids(tokenizer::BPETokenizer, tokens::Vector{String})
            """
            Convert tokens to integer IDs
            """
            ids = Int[]
            for token in tokens
                if haskey(tokenizer.vocab, token)
                    push!(ids, tokenizer.vocab[token])
                    elseif haskey(tokenizer.special_tokens, tokenizer.unk_token)
                    push!(ids, tokenizer.special_tokens[tokenizer.unk_token])
                else
                    push!(ids, 1)  # fallback to ID 1
                end
            end
            return ids
        end

        function ids_to_tokens(tokenizer::BPETokenizer, ids::Vector{Int})
            """
            Convert integer IDs back to tokens
            """
            # Create reverse mapping
            id_to_token = Dict{Int, String}()
            for (token, id) in tokenizer.vocab
                id_to_token[id] = token
            end

            tokens = String[]
            for id in ids
                if haskey(id_to_token, id)
                    push!(tokens, id_to_token[id])
                else
                    push!(tokens, tokenizer.unk_token)
                end
            end
            return tokens
        end

        function decode(tokenizer::BPETokenizer, tokens::Vector{String})
            """
            Decode tokens back to text
            """
            # Remove special tokens
            filtered_tokens = String[]
            for token in tokens
                if !haskey(tokenizer.special_tokens, token)
                    push!(filtered_tokens, token)
                end
            end

            # Join tokens and handle end-of-word markers
            text = join(filtered_tokens, "")
            text = replace(text, "</w>" => " ")

            # Clean up extra spaces
            text = replace(text, r"\s+" => " ")
            text = strip(text)

            return text
        end

        function encode_text_bpe(ai::AGI, text::String, tokenizer::BPETokenizer; add_special_tokens::Bool=false)
            """
            Encode text using BPE tokenizer and convert to embeddings
            """
            tokens = tokenize(tokenizer, text, add_special_tokens=add_special_tokens)
            token_ids = tokens_to_ids(tokenizer, tokens)

            seq_length = min(length(token_ids), ai.config.max_seq_length)
            embeddings = zeros(Float32, ai.config.d_model, seq_length)

            for (i, token_id) in enumerate(token_ids[1:seq_length])
                # Use token_id to get embedding
                if token_id <= size(ai.word_embeddings.matrix, 2)
                    embeddings[:, i] = ai.word_embeddings.matrix[:, token_id]
                else
                    # Use random embedding for unknown tokens
                    embeddings[:, i] = randn(Float32, ai.config.d_model) * 0.1f0
                end

                # Add positional encoding
                if i <= size(ai.positional_enc.matrix, 2)
                    embeddings[:, i] += ai.positional_enc.matrix[:, i]
                end
            end

            return embeddings
        end

        # Enhanced vocabulary building for subword tokens
        function build_bpe_vocabulary(texts::Vector{String}, vocab_size::Int=8000, min_frequency::Int=2)
            """
            Build BPE vocabulary from a corpus of texts
            """
            @info "Building BPE vocabulary" vocab_size=vocab_size min_frequency=min_frequency

            # Preprocess all texts
            processed_texts = [preprocess_text(text) for text in texts]

                # Character frequency counting
                char_freq = Dict{String, Int}()
                word_freq = Dict{String, Int}()

                for text in processed_texts
                    words = word_split(text)
                    for word in words
                        word = lowercase(word)
                        # Count word frequency
                        word_freq[word] = get(word_freq, word, 0) + 1

                        # Count character frequency
                        for char in word
                            char_str = string(char)
                            char_freq[char_str] = get(char_freq, char_str, 0) + 1
                        end
                    end
                end

                # Filter words by minimum frequency
                filtered_word_freq = Dict{String, Int}()
                for (word, freq) in word_freq
                    if freq >= min_frequency
                        filtered_word_freq[word] = freq
                    end
                end

                @info "Word statistics" total_words=length(word_freq) filtered_words=length(filtered_word_freq)

                # Initialize vocabulary with characters and special tokens
                vocab = Dict{String, Int}()

                # Add special tokens first
                special_tokens = ["<|endoftext|>", "<|unk|>", "<|pad|>", "<|bos|>", "<|eos|>"]
                for (i, token) in enumerate(special_tokens)
                    vocab[token] = i
                end

                # Add frequent characters
                sorted_chars = sort(collect(char_freq), by=x->x[2], rev=true)
                for (char, freq) in sorted_chars
                    if freq >= min_frequency && !haskey(vocab, char)
                        vocab[char] = length(vocab) + 1
                    end
                    if length(vocab) >= vocab_size ÷ 2  # Reserve half for merges
                        break
                    end
                end

                # Convert words to character sequences with end-of-word markers
                word_tokens = Dict{Vector{String}, Int}()
                for (word, freq) in filtered_word_freq
                    chars = [string(c) for c in word]
                        if !isempty(chars)
                            chars[end] = chars[end] * "</w>"
                            word_tokens[chars] = freq
                        end
                    end

                    merges = Tuple{String, String}[]

                    @info "Starting BPE merge process" initial_vocab_size=length(vocab)

                    # Iteratively find best merges
                    progress_interval = max(1, (vocab_size - length(vocab)) ÷ 20)

                    while length(vocab) < vocab_size
                        # Count pairs
                        pair_freq = Dict{Tuple{String, String}, Int}()

                        for (word, freq) in word_tokens
                            if length(word) < 2
                                continue
                            end

                            pairs = get_pairs(word)
                            for pair in pairs
                                pair_freq[pair] = get(pair_freq, pair, 0) + freq
                            end
                        end

                        if isempty(pair_freq)
                            @info "No more pairs to merge"
                            break
                        end

                        # Find most frequent pair
                        best_pair = argmax(pair_freq)
                        best_freq = pair_freq[best_pair]

                        # Skip pairs with very low frequency
                        if best_freq < min_frequency
                            @info "Best pair frequency too low" freq=best_freq min_freq=min_frequency
                            break
                        end

                        first, second = best_pair
                        merged = first * second

                        # Add to vocabulary
                        vocab[merged] = length(vocab) + 1
                        push!(merges, best_pair)

                        # Progress reporting
                        if length(merges) % progress_interval == 0
                            @info "BPE progress" vocab_size=length(vocab) merges=length(merges) last_pair=best_pair freq=best_freq
                        end

                        # Update word frequencies with the merge
                        new_word_tokens = Dict{Vector{String}, Int}()
                        for (word, freq) in word_tokens
                            new_word = String[]
                            i = 1
                            while i <= length(word)
                                if i < length(word) && word[i] == first && word[i+1] == second
                                    push!(new_word, merged)
                                    i += 2
                                else
                                    push!(new_word, word[i])
                                    i += 1
                                end
                            end
                            new_word_tokens[new_word] = freq
                        end
                        word_tokens = new_word_tokens
                    end

                    @info "BPE vocabulary construction completed" final_vocab_size=length(vocab) total_merges=length(merges)

                    return vocab, merges
                end

                function save_bpe_tokenizer(tokenizer::BPETokenizer, vocab_file::String, merges_file::String)
                    """
                    Save BPE tokenizer to files
                    """
                    # Save vocabulary
                    JSON3.write(vocab_file, tokenizer.vocab)

                    # Save merges
                    open(merges_file, "w") do f
                        println(f, "#version: 0.2")
                        for (first, second) in tokenizer.merges
                            println(f, "$first $second")
                        end
                    end

                    @info "BPE tokenizer saved" vocab_file=vocab_file merges_file=merges_file
                end

                function create_bpe_tokenizer_from_texts(texts::Vector{String}, vocab_size::Int=8000,
                                                         min_frequency::Int=2)
                    """
                    Create a BPE tokenizer from a collection of texts
                    """
                    vocab, merges = build_bpe_vocabulary(texts, vocab_size, min_frequency)

                    special_tokens = Dict(
                        "<|endoftext|>" => vocab["<|endoftext|>"],
                        "<|unk|>" => vocab["<|unk|>"],
                        "<|pad|>" => vocab["<|pad|>"],
                        "<|bos|>" => vocab["<|bos|>"],
                        "<|eos|>" => vocab["<|eos|>"]
                        )

                    return BPETokenizer(vocab, merges, special_tokens, 50, "<|unk|>", "<|pad|>", "<|endoftext|>")
                end

                # Integration with existing AGI system
                function update_agi_vocabulary!(ai::AGI, tokenizer::BPETokenizer)
                    """
                    Update AGI vocabulary to match BPE tokenizer
                    """
                    @info "Updating AGI vocabulary with BPE tokenizer"

                    # Clear existing vocabulary
                    ai.vocab.word_to_idx = Dict{String, Int}()
                    ai.vocab.idx_to_word = String[]

                    # Create reverse mapping for tokenizer vocab
                    id_to_token = Dict{Int, String}()
                    for (token, id) in tokenizer.vocab
                        id_to_token[id] = token
                    end

                    # Sort by ID to maintain order
                    sorted_ids = sort(collect(keys(id_to_token)))

                    # Update AGI vocabulary
                    for (idx, id) in enumerate(sorted_ids)
                        token = id_to_token[id]
                        ai.vocab.word_to_idx[token] = idx
                        push!(ai.vocab.idx_to_word, token)
                    end

                    # Resize word embeddings matrix if needed
                    vocab_size = length(ai.vocab.idx_to_word)
                    current_size = size(ai.word_embeddings.matrix, 2)

                    if vocab_size != current_size
                        if vocab_size > current_size
                            # Expand embeddings matrix
                            new_embeddings = randn(Float32, ai.config.d_model, vocab_size - current_size) * 0.1f0
                            ai.word_embeddings.matrix = hcat(ai.word_embeddings.matrix, new_embeddings)
                        else
                            # Shrink embeddings matrix
                            ai.word_embeddings.matrix = ai.word_embeddings.matrix[:, 1:vocab_size]
                        end
                    end

                    @info "AGI vocabulary updated" vocab_size=vocab_size embedding_size=size(ai.word_embeddings.matrix, 2)
                end

                # Helper function for testing tokenization
                function test_tokenization(tokenizer::BPETokenizer, test_text::String)
                    """
                    Test tokenization and decoding roundtrip
                    """
                    println("Original text: '$test_text'")

                    # Tokenize
                    tokens = tokenize(tokenizer, test_text)
                    println("Tokens: $tokens")

                    # Convert to IDs
                    ids = tokens_to_ids(tokenizer, tokens)
                    println("Token IDs: $ids")

                    # Convert back to tokens
                    recovered_tokens = ids_to_tokens(tokenizer, ids)
                    println("Recovered tokens: $recovered_tokens")

                    # Decode back to text
                    decoded_text = decode(tokenizer, recovered_tokens)
                    println("Decoded text: '$decoded_text'")

                    # Check if roundtrip is successful
                    original_normalized = preprocess_text(test_text)
                    success = original_normalized == decoded_text
                    println("Roundtrip successful: $success")

                    return success
                end

                # Export functions
                export BPETokenizer, tokenize, tokens_to_ids, ids_to_tokens, decode
                export encode_text_bpe, build_bpe_vocabulary, save_bpe_tokenizer
                export create_bpe_tokenizer_from_texts, update_agi_vocabulary!
                export preprocess_text, test_tokenization
